"""Convolution and pooling production kernels."""

import math

import torch

from cattorch.util.instruction.instruction import Instruction
from cattorch.util.scratch.dsl import (
    Program,
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
    lt,
    mathop,
    mod,
    mul,
    repeat,
    replace,
    set_var,
    sub,
    var,
)


from cattorch.util.instruction.optimized_common import (
    _broadcast_index,
    _static_unrolled_repeat,
    _unrolled_copy_statements,
    _unrolled_dot_statements,
    _unrolled_offset_dot_statements,
)

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


class OptimizedConvolutionInstruction(Instruction):
    """Convolution using compact, export-time spatial/kernel index maps.

    The maps are shared by every batch, input channel, and output channel. This
    avoids storing a full per-MAC schedule while removing coordinate arithmetic
    and padding conditionals from the runtime inner loop.
    """

    def prepare(self):
        input_shape = self.args[0].shape
        weight_shape = self.args[1].shape
        stride = self.args[3].value
        padding = self.args[4].value
        dimensions = len(weight_shape) - 2
        dilation = (
            self.args[5].value if len(self.args) > 5 and self.args[5].value is not None
            else [1] * dimensions
        )
        self.groups = (
            self.args[6].value if len(self.args) > 6 and self.args[6].value is not None
            else 1
        )
        if len(input_shape) == dimensions + 1:
            # Unbatched: the same layout as a batch of one.
            input_shape = (1, *input_shape)
        if len(input_shape) == 3:
            self.batches, self.input_channels, self.input_width = input_shape
            self.input_height = 1
            self.output_channels = weight_shape[0]
            self.kernel_height = 1
            self.kernel_width = weight_shape[2]
            self.stride_height, self.stride_width = 1, stride[0]
            self.padding_height, self.padding_width = 0, padding[0]
            self.dilation_height, self.dilation_width = 1, dilation[0]
        else:
            (
                self.batches, self.input_channels,
                self.input_height, self.input_width,
            ) = input_shape
            self.output_channels = weight_shape[0]
            self.kernel_height, self.kernel_width = weight_shape[2:]
            self.stride_height, self.stride_width = stride
            self.padding_height, self.padding_width = padding
            self.dilation_height, self.dilation_width = (
                (dilation[0], dilation[0]) if len(dilation) == 1 else dilation
            )
        # Each output channel reads only its group's input channels.
        self.group_channels = weight_shape[1]
        self.outputs_per_group = self.output_channels // self.groups

        if self.args[0].dynamic:
            self.batches = div(
                length("T1"),
                self.input_channels * self.input_height * self.input_width,
            )

        self.output_height = (
            self.input_height + 2 * self.padding_height
            - self.dilation_height * (self.kernel_height - 1) - 1
        ) // self.stride_height + 1
        self.output_width = (
            self.input_width + 2 * self.padding_width
            - self.dilation_width * (self.kernel_width - 1) - 1
        ) // self.stride_width + 1
        self.input_plane = self.input_height * self.input_width
        self.kernel_plane = self.kernel_height * self.kernel_width
        self.output_spatial = self.output_height * self.output_width
        self.has_bias = self.args[2].value is not None or bool(self.args[2].shape)
        self.sparse_channel_starts = None

        self.map_starts = []
        self.map_counts = []
        self.input_offsets = []
        self.weight_offsets = []
        for output_row in range(self.output_height):
            for output_column in range(self.output_width):
                self.map_starts.append(len(self.input_offsets) + 1)
                count = 0
                for kernel_row in range(self.kernel_height):
                    input_row = (
                        output_row * self.stride_height
                        - self.padding_height + kernel_row * self.dilation_height
                    )
                    for kernel_column in range(self.kernel_width):
                        input_column = (
                            output_column * self.stride_width
                            - self.padding_width + kernel_column * self.dilation_width
                        )
                        if (
                            0 <= input_row < self.input_height
                            and 0 <= input_column < self.input_width
                        ):
                            self.input_offsets.append(
                                input_row * self.input_width + input_column,
                            )
                            self.weight_offsets.append(
                                kernel_row * self.kernel_width + kernel_column,
                            )
                            count += 1
                self.map_counts.append(count)

    def transform_weights(self, static_lists):
        raw_key = self.args[1].name.removeprefix("W_")
        weight = static_lists.get(raw_key)
        if not isinstance(weight, torch.Tensor) or weight.ndim not in {3, 4}:
            return
        starts = []
        counts = []
        channels = []
        total = self.output_channels * self.group_channels
        for output in range(self.output_channels):
            starts.append(len(channels) + 1)
            before = len(channels)
            for channel in range(self.group_channels):
                if bool(torch.any(weight[output, channel] != 0)):
                    channels.append(channel)
            counts.append(len(channels) - before)
        if total and len(channels) <= total * 0.75:
            self.sparse_channel_starts = starts
            self.sparse_channel_counts = counts
            self.sparse_channels = channels

    def finalize(self):
        accumulator = item("T3", var("bias_index")) if self.has_bias else 0
        mapped_product = mul(
            item(
                "T1",
                add(
                    var("input_base"),
                    item("_conv_input_offsets", var("map_index")),
                ),
            ),
            item(
                "T2",
                add(
                    var("weight_base"),
                    item("_conv_weight_offsets", var("map_index")),
                ),
            ),
        )
        sequential_product = mul(
            item(
                "T1",
                add(
                    var("input_base"),
                    item("_conv_input_offsets", var("map_index")),
                ),
            ),
            item("T2", var("weight_index")),
        )
        sequential_step = (
            change_var("sum", sequential_product),
            change_var("map_index", 1),
            change_var("weight_index", 1),
        )
        chunks, remainder = divmod(self.kernel_plane, 4)
        sequential_taps = ()
        if chunks:
            sequential_taps += (repeat(chunks, sequential_step * 4),)
        sequential_taps += sequential_step * remainder

        # The first input of the output channel's group.
        group_input_start = var("batch_input_start") if self.groups == 1 else var("group_start")
        if self.sparse_channel_starts is None:
            full_kernel = (
                set_var("weight_index", var("output_weight_start")),
                repeat(self.group_channels, (
                    set_var("map_index", var("map_start")),
                    *sequential_taps,
                    change_var("input_base", self.input_plane),
                )),
            )
            clipped_kernel = (
                set_var("weight_base", var("output_weight_start")),
                repeat(self.group_channels, (
                    set_var("map_index", var("map_start")),
                    repeat(var("valid_count"), (
                        change_var("sum", mapped_product),
                        change_var("map_index", 1),
                    )),
                    change_var("input_base", self.input_plane),
                    change_var("weight_base", self.kernel_plane),
                )),
            )
            output_setup = []
            sparse_lists = ()
            sparse_values = {}
        else:
            channel_input_base = add(
                group_input_start,
                mul(var("active_channel"), self.input_plane),
            )
            channel_weight_base = add(
                var("output_weight_start"),
                mul(var("active_channel"), self.kernel_plane),
            )
            active_finish = (change_var("active_map_index", 1),)
            full_kernel = (
                set_var("active_map_index", var("active_start")),
                repeat(var("active_count"), (
                    set_var(
                        "active_channel",
                        item("_conv_active_channels", var("active_map_index")),
                    ),
                    set_var("input_base", channel_input_base),
                    set_var("weight_index", channel_weight_base),
                    set_var("map_index", var("map_start")),
                    *sequential_taps,
                    *active_finish,
                )),
            )
            clipped_kernel = (
                set_var("active_map_index", var("active_start")),
                repeat(var("active_count"), (
                    set_var(
                        "active_channel",
                        item("_conv_active_channels", var("active_map_index")),
                    ),
                    set_var("input_base", channel_input_base),
                    set_var("weight_base", channel_weight_base),
                    set_var("map_index", var("map_start")),
                    repeat(var("valid_count"), (
                        change_var("sum", mapped_product),
                        change_var("map_index", 1),
                    )),
                    *active_finish,
                )),
            )
            output_setup = [
                set_var(
                    "active_start",
                    item("_conv_active_starts", var("bias_index")),
                ),
                set_var(
                    "active_count",
                    item("_conv_active_counts", var("bias_index")),
                ),
            ]
            sparse_lists = (
                "_conv_active_starts", "_conv_active_counts", "_conv_active_channels",
            )
            sparse_values = {
                "_conv_active_starts": self.sparse_channel_starts,
                "_conv_active_counts": self.sparse_channel_counts,
                "_conv_active_channels": self.sparse_channels,
            }
        if self.groups != 1:
            group = mathop("floor", div(sub(var("bias_index"), 1), self.outputs_per_group))
            output_setup.insert(0, set_var("group_start", add(
                var("batch_input_start"),
                mul(group, self.group_channels * self.input_plane),
            )))
        return Program(
            "convolution_optimized",
            variables=(
                "spatial", "bias_index", "batch_input_start",
                "output_weight_start",
                "sum", "map_start", "map_index", "valid_count",
                "input_base", "weight_base", "weight_index",
                "active_start", "active_count", "active_map_index", "active_channel",
                *(("group_start",) if self.groups != 1 else ()),
            ),
            lists=(
                "T1", "T2", "T3", "T4", "_conv_map_starts",
                "_conv_map_counts", "_conv_input_offsets",
                "_conv_weight_offsets",
                *sparse_lists,
            ),
            list_values={
                "_conv_map_starts": self.map_starts,
                "_conv_map_counts": self.map_counts,
                "_conv_input_offsets": self.input_offsets,
                "_conv_weight_offsets": self.weight_offsets,
                **sparse_values,
            },
            body=(
                clear("T4"),
                set_var("batch_input_start", 1),
                repeat(self.batches, (
                    set_var("bias_index", 1),
                    set_var("output_weight_start", 1),
                    repeat(self.output_channels, (
                        *output_setup,
                        set_var("spatial", 1),
                        repeat(self.output_spatial, (
                            set_var("sum", accumulator),
                            set_var(
                                "map_start",
                                item("_conv_map_starts", var("spatial")),
                            ),
                            set_var(
                                "valid_count",
                                item("_conv_map_counts", var("spatial")),
                            ),
                            set_var("input_base", group_input_start),
                            if_else(
                                eq(var("valid_count"), self.kernel_plane),
                                full_kernel,
                                clipped_kernel,
                            ),
                            append("T4", var("sum")),
                            change_var("spatial", 1),
                        )),
                        change_var("bias_index", 1),
                        change_var(
                            "output_weight_start",
                            self.group_channels * self.kernel_plane,
                        ),
                    )),
                    change_var(
                        "batch_input_start",
                        self.input_channels * self.input_plane,
                    ),
                )),
            ),
        ).compile()


class OptimizedPoolingInstruction(Instruction):
    """Pooling with export-time receptive-field maps and sequential output."""

    def prepare(self):
        shape = self.args[0].shape
        op = str(self.torch_name)
        if len(shape) == (2 if "pool1d" in op else 3):
            # Unbatched: the same layout as a batch of one.
            shape = (1, *shape)
        self.batches = (
            div(length("T1"), math.prod(shape[1:]))
            if self.args[0].dynamic else shape[0]
        )
        self.channels = shape[1]
        if len(shape) == 3:
            self.input_height, self.input_width = 1, shape[2]
        else:
            self.input_height, self.input_width = shape[2:]
        self.input_plane = self.input_height * self.input_width
        self.is_max = "max_pool" in op
        self.is_adaptive = "adaptive_avg_pool" in op
        rows, columns = self._windows(len(shape) == 3)
        self.output_height, self.output_width = len(rows), len(columns)

        self.map_starts = []
        self.map_counts = []
        self.input_offsets = []
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
        self.output_spatial = self.output_height * self.output_width
        # Per-window divisors need the window's index, which only the
        # general traversal keeps.
        self.uniform_count = (
            self.map_counts[0]
            if self.map_counts and len(set(self.map_counts)) == 1
            and (self.is_max or self.uniform_divisor is not None)
            else None
        )

    def _windows(self, one_dimensional):
        """Per output row and column: input positions and padded extent."""
        self.divisor_override = None
        if self.is_adaptive:
            self.include_padding = False
            sizes = self.args[1].value
            sizes = (1, sizes[0]) if one_dimensional else sizes

            return (
                adaptive_windows(self.input_height, sizes[0]),
                adaptive_windows(self.input_width, sizes[1]),
            )

        def option(position, default):
            value = self.args[position].value if len(self.args) > position else None
            return default if value is None or value == [] else value

        def spatial(value, trivial):
            """``(height, width)``; 1D pooling has a trivial height."""
            value = list(value) if isinstance(value, (list, tuple)) else [value]
            if one_dimensional:
                return trivial, value[0]
            return (value[0], value[0]) if len(value) == 1 else tuple(value)

        kernel = spatial(self.args[1].value, 1)
        # An omitted or empty stride means the kernel size, as in PyTorch.
        stride = spatial(option(2, self.args[1].value), 1)
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
            pool_windows(
                self.input_height, kernel[0], stride[0], padding[0], dilation[0], ceil_mode,
            ),
            pool_windows(
                self.input_width, kernel[1], stride[1], padding[1], dilation[1], ceil_mode,
            ),
        )

    def _divisor(self):
        if self.uniform_divisor is not None:
            return self.uniform_divisor
        if self.divisors == self.map_counts:
            return var("remaining")
        return item("_pool_divisors", var("spatial"))

    def finalize(self):
        current = item(
            "T1",
            add(
                var("input_base"),
                item("_pool_input_offsets", var("map_index")),
            ),
        )
        if self.is_max:
            reduction = (
                set_var("value", current),
                if_(gt(var("value"), var("result")), (
                    set_var("result", var("value")),
                )),
            )
            finish = append("T2", var("result"))
        else:
            reduction = (change_var("result", current),)

        reduction_step = (*reduction, change_var("map_index", 1))
        if self.uniform_count is not None:
            if self.is_max:
                initialize = (
                    set_var("result", current),
                    change_var("map_index", 1),
                )
                reduction_body = _static_unrolled_repeat(
                    self.uniform_count - 1, reduction_step,
                )
                finish = append("T2", var("result"))
            else:
                initialize = (set_var("result", 0),)
                reduction_body = _static_unrolled_repeat(
                    self.uniform_count, reduction_step,
                )
                finish = append("T2", div(var("result"), self._divisor()))
            channel_body = (
                set_var("map_index", 1),
                repeat(self.output_spatial, (
                    *initialize,
                    *reduction_body,
                    finish,
                )),
            )
            map_lists = ("_pool_input_offsets",)
            map_values = {"_pool_input_offsets": self.input_offsets}
        else:
            if self.is_max:
                initialize = (
                    set_var("result", current),
                    change_var("map_index", 1),
                    change_var("remaining", -1),
                )
                finish = append("T2", var("result"))
            else:
                initialize = (set_var("result", 0),)
                finish = append("T2", div(var("result"), self._divisor()))
            channel_body = (
                set_var("spatial", 1),
                repeat(self.output_spatial, (
                    set_var(
                        "map_index",
                        item("_pool_map_starts", var("spatial")),
                    ),
                    set_var(
                        "remaining",
                        item("_pool_map_counts", var("spatial")),
                    ),
                    *initialize,
                    repeat(var("remaining"), reduction_step),
                    finish,
                    change_var("spatial", 1),
                )),
            )
            map_lists = (
                "_pool_map_starts", "_pool_map_counts", "_pool_input_offsets",
            )
            map_values = {
                "_pool_map_starts": self.map_starts,
                "_pool_map_counts": self.map_counts,
                "_pool_input_offsets": self.input_offsets,
            }

        if not self.is_max and self.uniform_divisor is None and self.divisors != self.map_counts:
            map_lists = (*map_lists, "_pool_divisors")
            map_values["_pool_divisors"] = self.divisors
        return Program(
            "pooling_optimized",
            variables=(
                "spatial", "input_base", "map_index",
                "remaining", "result", "value",
            ),
            lists=("T1", "T2", *map_lists),
            list_values=map_values,
            body=(
                clear("T2"),
                set_var("input_base", 1),
                repeat(self.batches, (
                    repeat(self.channels, (
                        *channel_body,
                        change_var("input_base", self.input_plane),
                    )),
                )),
            ),
        ).compile()
