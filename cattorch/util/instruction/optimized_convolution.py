"""Convolution and pooling production kernels."""

import math

import torch

from cattorch.kernel_api import Kernel, KernelContext
from cattorch.util.scratch.dsl import (
    add,
    append,
    change_var,
    clear,
    div,
    eq,
    gt,
    if_,
    if_else,
    item,
    length,
    mathop,
    mul,
    repeat,
    set_var,
    sub,
    var,
)


from cattorch.util.instruction.optimized_common import _static_unrolled_repeat, argument_at

def pool_windows(length, size, stride, padding, dilation, ceil_mode):
    """PyTorch's pooling windows along one dimension.

    Each window is ``(positions, extent)``: the input positions it reads,
    some of which may be padding, and how many padded positions average
    pooling counts.
    """
    reach = dilation * (size - 1) + 1
    span = length + 2 * padding - reach
    outputs = (-(-span // stride) if ceil_mode else span // stride) + 1
    # PyTorch drops a last window that would start in the padding.
    if ceil_mode and (outputs - 1) * stride >= length + padding:
        outputs -= 1
    windows = []
    for index in range(outputs):
        start = index * stride - padding
        # Average pooling counts padding, but not beyond it.
        extent = min(start + size, length + padding) - start
        windows.append((range(start, start + reach, dilation), extent))
    return windows


def adaptive_windows(length, outputs):
    """Adaptive pooling's windows along one dimension, as in ``pool_windows``."""
    windows = []
    for index in range(outputs):
        start = math.floor(index * length / outputs)
        end = math.ceil((index + 1) * length / outputs)
        windows.append((range(start, end), end - start))
    return windows


class _ConvolutionLayout:
    """A convolution's sizes, and its export-time spatial and kernel maps.

    The maps are shared by every batch, input channel, and output channel.
    This avoids storing a full per-MAC schedule while removing coordinate
    arithmetic and padding conditionals from the runtime inner loop.
    """

    def __init__(self, ctx: KernelContext):
        source, weight = ctx.inputs["input"], ctx.inputs["weight"]
        input_shape, weight_shape = source.shape, weight.shape
        stride, padding = ctx.scalars["stride"], ctx.scalars["padding"]
        dimensions = len(weight_shape) - 2
        dilation = ctx.scalars.get("dilation") or [1] * dimensions
        groups = ctx.scalars.get("groups")
        self.groups = 1 if groups is None else groups
        if len(input_shape) == dimensions + 1:
            # Unbatched: the same layout as a batch of one.
            input_shape = (1, *input_shape)
        if len(input_shape) == 3:
            self.batches, self.input_channels, self.input_width = input_shape
            self.input_height = 1
            kernel_height, kernel_width = 1, weight_shape[2]
            stride_height, stride_width = 1, stride[0]
            padding_height, padding_width = 0, padding[0]
            dilation_height, dilation_width = 1, dilation[0]
        else:
            self.batches, self.input_channels, self.input_height, self.input_width = input_shape
            kernel_height, kernel_width = weight_shape[2:]
            # One value applies to both dimensions, as in PyTorch.
            stride_height, stride_width = stride * 2 if len(stride) == 1 else stride
            padding_height, padding_width = padding * 2 if len(padding) == 1 else padding
            dilation_height, dilation_width = (
                (dilation[0], dilation[0]) if len(dilation) == 1 else dilation
            )
        self.output_channels = weight_shape[0]
        # Each output channel reads only its group's input channels.
        self.group_channels = weight_shape[1]
        self.outputs_per_group = self.output_channels // self.groups
        if source.dynamic_axis is not None:
            self.batches = div(
                length(source.list), self.input_channels * self.input_height * self.input_width,
            )
        output_height = (
            self.input_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1
        ) // stride_height + 1
        output_width = (
            self.input_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1
        ) // stride_width + 1
        self.input_plane = self.input_height * self.input_width
        self.kernel_plane = kernel_height * kernel_width
        self.output_spatial = output_height * output_width

        self.map_starts, self.map_counts = [], []
        self.input_offsets, self.weight_offsets = [], []
        for output_row in range(output_height):
            for output_column in range(output_width):
                self.map_starts.append(len(self.input_offsets) + 1)
                count = 0
                for kernel_row in range(kernel_height):
                    input_row = (
                        output_row * stride_height - padding_height + kernel_row * dilation_height
                    )
                    for kernel_column in range(kernel_width):
                        input_column = (
                            output_column * stride_width - padding_width
                            + kernel_column * dilation_width
                        )
                        if 0 <= input_row < self.input_height and 0 <= input_column < self.input_width:
                            self.input_offsets.append(input_row * self.input_width + input_column)
                            self.weight_offsets.append(kernel_row * kernel_width + kernel_column)
                            count += 1
                self.map_counts.append(count)

        # A stored weight with many all-zero input channels skips them.
        self.sparse = None
        value = weight.value
        if isinstance(value, torch.Tensor) and value.ndim in {3, 4}:
            starts, counts, channels = [], [], []
            for output in range(self.output_channels):
                starts.append(len(channels) + 1)
                before = len(channels)
                for channel in range(self.group_channels):
                    if bool(torch.any(value[output, channel] != 0)):
                        channels.append(channel)
                counts.append(len(channels) - before)
            total = self.output_channels * self.group_channels
            if total and len(channels) <= total * 0.75:
                self.sparse = (starts, counts, channels)


_CONVOLUTION_VARIABLES = (
    "spatial", "bias_index", "batch_input_start", "output_weight_start",
    "sum", "map_start", "map_index", "valid_count",
    "input_base", "weight_base", "weight_index",
    "active_start", "active_count", "active_map_index", "active_channel",
)


def _emit_convolution(ctx: KernelContext):
    layout = _ConvolutionLayout(ctx)
    for name in _CONVOLUTION_VARIABLES:
        ctx.temp_var(name)
    if layout.groups != 1:
        ctx.temp_var("group_start")
    source, weight, bias = ctx.inputs["input"].list, ctx.inputs["weight"].list, ctx.inputs.get("bias")
    out = ctx.output.list
    map_starts = ctx.static_list("_conv_map_starts", layout.map_starts)
    map_counts = ctx.static_list("_conv_map_counts", layout.map_counts)
    input_offsets = ctx.static_list("_conv_input_offsets", layout.input_offsets)
    weight_offsets = ctx.static_list("_conv_weight_offsets", layout.weight_offsets)

    accumulator = item(bias.list, var("bias_index")) if bias else 0
    mapped_input = item(source, add(var("input_base"), item(input_offsets, var("map_index"))))
    mapped_product = mul(
        mapped_input, item(weight, add(var("weight_base"), item(weight_offsets, var("map_index")))),
    )
    sequential_step = (
        change_var("sum", mul(mapped_input, item(weight, var("weight_index")))),
        change_var("map_index", 1),
        change_var("weight_index", 1),
    )
    chunks, remainder = divmod(layout.kernel_plane, 4)
    sequential_taps = ()
    if chunks:
        sequential_taps += (repeat(chunks, sequential_step * 4),)
    sequential_taps += sequential_step * remainder
    clipped_taps = repeat(var("valid_count"), (
        change_var("sum", mapped_product),
        change_var("map_index", 1),
    ))

    # The first input of the output channel's group.
    group_input_start = var("batch_input_start") if layout.groups == 1 else var("group_start")
    if layout.sparse is None:
        full_kernel = (
            set_var("weight_index", var("output_weight_start")),
            repeat(layout.group_channels, (
                set_var("map_index", var("map_start")),
                *sequential_taps,
                change_var("input_base", layout.input_plane),
            )),
        )
        clipped_kernel = (
            set_var("weight_base", var("output_weight_start")),
            repeat(layout.group_channels, (
                set_var("map_index", var("map_start")),
                clipped_taps,
                change_var("input_base", layout.input_plane),
                change_var("weight_base", layout.kernel_plane),
            )),
        )
        output_setup = []
    else:
        starts, counts, channels = layout.sparse
        active_starts = ctx.static_list("_conv_active_starts", starts)
        active_counts = ctx.static_list("_conv_active_counts", counts)
        active_channels = ctx.static_list("_conv_active_channels", channels)

        def active_kernel(weight_variable, taps):
            return (
                set_var("active_map_index", var("active_start")),
                repeat(var("active_count"), (
                    set_var("active_channel", item(active_channels, var("active_map_index"))),
                    set_var("input_base", add(
                        group_input_start, mul(var("active_channel"), layout.input_plane),
                    )),
                    set_var(weight_variable, add(
                        var("output_weight_start"), mul(var("active_channel"), layout.kernel_plane),
                    )),
                    set_var("map_index", var("map_start")),
                    *taps,
                    change_var("active_map_index", 1),
                )),
            )
        full_kernel = active_kernel("weight_index", sequential_taps)
        clipped_kernel = active_kernel("weight_base", (clipped_taps,))
        output_setup = [
            set_var("active_start", item(active_starts, var("bias_index"))),
            set_var("active_count", item(active_counts, var("bias_index"))),
        ]
    if layout.groups != 1:
        group = mathop("floor", div(sub(var("bias_index"), 1), layout.outputs_per_group))
        output_setup.insert(0, set_var("group_start", add(
            var("batch_input_start"), mul(group, layout.group_channels * layout.input_plane),
        )))
    return (
        clear(out),
        set_var("batch_input_start", 1),
        repeat(layout.batches, (
            set_var("bias_index", 1),
            set_var("output_weight_start", 1),
            repeat(layout.output_channels, (
                *output_setup,
                set_var("spatial", 1),
                repeat(layout.output_spatial, (
                    set_var("sum", accumulator),
                    set_var("map_start", item(map_starts, var("spatial"))),
                    set_var("valid_count", item(map_counts, var("spatial"))),
                    set_var("input_base", group_input_start),
                    if_else(eq(var("valid_count"), layout.kernel_plane), full_kernel, clipped_kernel),
                    append(out, var("sum")),
                    change_var("spatial", 1),
                )),
                change_var("bias_index", 1),
                change_var("output_weight_start", layout.group_channels * layout.kernel_plane),
            )),
            change_var("batch_input_start", layout.input_channels * layout.input_plane),
        )),
    )


CONVOLUTION_KERNELS = tuple(
    Kernel(target, _emit_convolution, name="convolution_optimized")
    for target in ("aten.conv1d.default", "aten.conv2d.default")
)


class _PoolingLayout:
    """A pooling call's sizes and export-time receptive-field maps."""

    def __init__(self, ctx: KernelContext):
        source = ctx.inputs["self"]
        shape = source.shape
        op = ctx.target
        if len(shape) == (2 if "pool1d" in op else 3):
            # Unbatched: the same layout as a batch of one.
            shape = (1, *shape)
        self.batches = (
            div(length(source.list), math.prod(shape[1:])) if source.dynamic_axis is not None
            else shape[0]
        )
        self.channels = shape[1]
        if len(shape) == 3:
            self.input_height, self.input_width = 1, shape[2]
        else:
            self.input_height, self.input_width = shape[2:]
        self.input_plane = self.input_height * self.input_width
        self.is_max = "max_pool" in op
        self.is_adaptive = "adaptive_avg_pool" in op
        rows, columns = self._windows(ctx, len(shape) == 3)
        self.output_spatial = len(rows) * len(columns)

        self.map_starts, self.map_counts, self.input_offsets = [], [], []
        divisors = []
        for (row_positions, row_extent), (column_positions, column_extent) in (
            (row, column) for row in rows for column in columns
        ):
            self.map_starts.append(len(self.input_offsets) + 1)
            count = 0
            for row in row_positions:
                for column in column_positions:
                    if 0 <= row < self.input_height and 0 <= column < self.input_width:
                        self.input_offsets.append(row * self.input_width + column)
                        count += 1
            self.map_counts.append(count)
            if self.divisor_override is not None:
                divisors.append(self.divisor_override)
            elif self.include_padding:
                divisors.append(row_extent * column_extent)
            else:
                divisors.append(count)
        # One divisor for every window is a constant; otherwise a list.
        self.divisors = divisors
        self.uniform_divisor = divisors[0] if divisors and len(set(divisors)) == 1 else None
        # Per-window divisors need the window's index, which only the
        # general traversal keeps.
        self.uniform_count = (
            self.map_counts[0]
            if self.map_counts and len(set(self.map_counts)) == 1
            and (self.is_max or self.uniform_divisor is not None)
            else None
        )

    def _windows(self, ctx, one_dimensional):
        """Per output row and column: input positions and padded extent."""
        self.divisor_override = None
        if self.is_adaptive:
            self.include_padding = False
            sizes = argument_at(ctx, 1)
            sizes = (1, sizes[0]) if one_dimensional else sizes
            return (
                adaptive_windows(self.input_height, sizes[0]),
                adaptive_windows(self.input_width, sizes[1]),
            )

        def option(position, default):
            value = argument_at(ctx, position)
            return default if value is None or value == [] else value

        def spatial(value, trivial):
            """``(height, width)``; 1D pooling has a trivial height."""
            value = list(value) if isinstance(value, (list, tuple)) else [value]
            if one_dimensional:
                return trivial, value[0]
            return (value[0], value[0]) if len(value) == 1 else tuple(value)

        kernel_size = argument_at(ctx, 1)
        kernel = spatial(kernel_size, 1)
        # An omitted or empty stride means the kernel size, as in PyTorch.
        stride = spatial(option(2, kernel_size), 1)
        padding = spatial(option(3, [0]), 0)
        if self.is_max:
            dilation = spatial(option(4, [1]), 1)
            ceil_mode = bool(option(5, False))
            self.include_padding = False
        else:
            dilation = (1, 1)
            ceil_mode = bool(option(4, False))
            self.include_padding = bool(option(5, True))
            self.divisor_override = option(6, None)
        return (
            pool_windows(self.input_height, kernel[0], stride[0], padding[0], dilation[0], ceil_mode),
            pool_windows(self.input_width, kernel[1], stride[1], padding[1], dilation[1], ceil_mode),
        )


def _emit_pooling(ctx: KernelContext):
    layout = _PoolingLayout(ctx)
    for name in ("spatial", "input_base", "map_index", "remaining", "result", "value"):
        ctx.temp_var(name)
    out = ctx.output.list
    if layout.uniform_count is None:
        map_starts = ctx.static_list("_pool_map_starts", layout.map_starts)
        map_counts = ctx.static_list("_pool_map_counts", layout.map_counts)
    input_offsets = ctx.static_list("_pool_input_offsets", layout.input_offsets)
    if layout.is_max or layout.uniform_divisor is not None:
        divisor = layout.uniform_divisor
    elif layout.divisors == layout.map_counts:
        divisor = var("remaining")
    else:
        divisor = item(ctx.static_list("_pool_divisors", layout.divisors), var("spatial"))

    current = item(
        ctx.inputs["self"].list, add(var("input_base"), item(input_offsets, var("map_index"))),
    )
    if layout.is_max:
        reduction = (
            set_var("value", current),
            if_(gt(var("value"), var("result")), (set_var("result", var("value")),)),
        )
        finish = append(out, var("result"))
    else:
        reduction = (change_var("result", current),)
        finish = append(out, div(var("result"), divisor))
    reduction_step = (*reduction, change_var("map_index", 1))

    if layout.uniform_count is not None:
        if layout.is_max:
            initialize = (set_var("result", current), change_var("map_index", 1))
            # A one-element window is its own maximum.
            reduction_body = (
                _static_unrolled_repeat(layout.uniform_count - 1, reduction_step)
                if layout.uniform_count > 1 else ()
            )
        else:
            initialize = (set_var("result", 0),)
            reduction_body = _static_unrolled_repeat(layout.uniform_count, reduction_step)
        channel_body = (
            set_var("map_index", 1),
            repeat(layout.output_spatial, (*initialize, *reduction_body, finish)),
        )
    else:
        if layout.is_max:
            initialize = (
                set_var("result", current),
                change_var("map_index", 1),
                change_var("remaining", -1),
            )
        else:
            initialize = (set_var("result", 0),)
        channel_body = (
            set_var("spatial", 1),
            repeat(layout.output_spatial, (
                set_var("map_index", item(map_starts, var("spatial"))),
                set_var("remaining", item(map_counts, var("spatial"))),
                *initialize,
                repeat(var("remaining"), reduction_step),
                finish,
                change_var("spatial", 1),
            )),
        )
    return (
        clear(out),
        set_var("input_base", 1),
        repeat(layout.batches, (
            repeat(layout.channels, (
                *channel_body,
                change_var("input_base", layout.input_plane),
            )),
        )),
    )


POOLING_KERNELS = tuple(
    Kernel(target, _emit_pooling, name="pooling_optimized")
    for target in (
        "aten.max_pool1d.default", "aten.max_pool2d.default", "aten.avg_pool1d.default",
        "aten.avg_pool2d.default", "aten.adaptive_avg_pool2d.default",
    )
)


__all__ = ["CONVOLUTION_KERNELS", "POOLING_KERNELS", "adaptive_windows", "pool_windows"]
