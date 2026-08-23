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
        if len(input_shape) == 3:
            self.batches, self.input_channels, self.input_width = input_shape
            self.input_height = 1
            self.output_channels = weight_shape[0]
            self.kernel_height = 1
            self.kernel_width = weight_shape[2]
            self.stride_height, self.stride_width = 1, stride[0]
            self.padding_height, self.padding_width = 0, padding[0]
        else:
            (
                self.batches, self.input_channels,
                self.input_height, self.input_width,
            ) = input_shape
            self.output_channels = weight_shape[0]
            self.kernel_height, self.kernel_width = weight_shape[2:]
            self.stride_height, self.stride_width = stride
            self.padding_height, self.padding_width = padding

        self.output_height = (
            self.input_height + 2 * self.padding_height - self.kernel_height
        ) // self.stride_height + 1
        self.output_width = (
            self.input_width + 2 * self.padding_width - self.kernel_width
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
                        - self.padding_height + kernel_row
                    )
                    for kernel_column in range(self.kernel_width):
                        input_column = (
                            output_column * self.stride_width
                            - self.padding_width + kernel_column
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
        total = self.output_channels * self.input_channels
        for output in range(self.output_channels):
            starts.append(len(channels) + 1)
            before = len(channels)
            for channel in range(self.input_channels):
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

        if self.sparse_channel_starts is None:
            full_kernel = (
                set_var("weight_index", var("output_weight_start")),
                repeat(self.input_channels, (
                    set_var("map_index", var("map_start")),
                    *sequential_taps,
                    change_var("input_base", self.input_plane),
                )),
            )
            clipped_kernel = (
                set_var("weight_base", var("output_weight_start")),
                repeat(self.input_channels, (
                    set_var("map_index", var("map_start")),
                    repeat(var("valid_count"), (
                        change_var("sum", mapped_product),
                        change_var("map_index", 1),
                    )),
                    change_var("input_base", self.input_plane),
                    change_var("weight_base", self.kernel_plane),
                )),
            )
            output_setup = ()
            sparse_lists = ()
            sparse_values = {}
        else:
            channel_input_base = add(
                var("batch_input_start"),
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
            output_setup = (
                set_var(
                    "active_start",
                    item("_conv_active_starts", var("bias_index")),
                ),
                set_var(
                    "active_count",
                    item("_conv_active_counts", var("bias_index")),
                ),
            )
            sparse_lists = (
                "_conv_active_starts", "_conv_active_counts", "_conv_active_channels",
            )
            sparse_values = {
                "_conv_active_starts": self.sparse_channel_starts,
                "_conv_active_counts": self.sparse_channel_counts,
                "_conv_active_channels": self.sparse_channels,
            }
        return Program(
            "convolution_optimized",
            variables=(
                "spatial", "bias_index", "batch_input_start",
                "output_weight_start",
                "sum", "map_start", "map_index", "valid_count",
                "input_base", "weight_base", "weight_index",
                "active_start", "active_count", "active_map_index", "active_channel",
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
                            set_var("input_base", var("batch_input_start")),
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
                            self.input_channels * self.kernel_plane,
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
        self.batches = shape[0]
        self.channels = shape[1]
        if len(shape) == 3:
            self.input_height, self.input_width = 1, shape[2]
        else:
            self.input_height, self.input_width = shape[2:]
        self.input_plane = self.input_height * self.input_width
        op = str(self.torch_name)
        self.is_max = "max_pool" in op
        self.is_adaptive = "adaptive_avg_pool" in op

        if self.is_adaptive:
            self.output_height, self.output_width = self.args[1].value
            fields = []
            for output_row in range(self.output_height):
                row_start = math.floor(output_row * self.input_height / self.output_height)
                row_end = math.ceil((output_row + 1) * self.input_height / self.output_height)
                for output_column in range(self.output_width):
                    column_start = math.floor(
                        output_column * self.input_width / self.output_width,
                    )
                    column_end = math.ceil(
                        (output_column + 1) * self.input_width / self.output_width,
                    )
                    fields.append((row_start, row_end, column_start, column_end))
            self.include_padding = False
            self.kernel_area = None
        else:
            kernel = self.args[1].value
            stride = self.args[2].value
            padding = self.args[3].value if len(self.args) > 3 else [0]
            if len(shape) == 3:
                kernel_height, kernel_width = 1, kernel[0]
                stride_height, stride_width = 1, stride[0]
                padding_height, padding_width = 0, padding[0]
            else:
                kernel_height, kernel_width = (
                    (kernel[0], kernel[0]) if len(kernel) == 1 else kernel
                )
                stride_height, stride_width = (
                    (stride[0], stride[0]) if len(stride) == 1 else stride
                )
                padding_height, padding_width = (
                    (padding[0], padding[0]) if len(padding) == 1 else padding
                )
            self.output_height = (
                self.input_height + 2 * padding_height - kernel_height
            ) // stride_height + 1
            self.output_width = (
                self.input_width + 2 * padding_width - kernel_width
            ) // stride_width + 1
            fields = []
            for output_row in range(self.output_height):
                row_start = output_row * stride_height - padding_height
                for output_column in range(self.output_width):
                    column_start = output_column * stride_width - padding_width
                    fields.append((
                        row_start, row_start + kernel_height,
                        column_start, column_start + kernel_width,
                    ))
            self.include_padding = not self.is_max
            self.kernel_area = kernel_height * kernel_width

        self.map_starts = []
        self.map_counts = []
        self.input_offsets = []
        for row_start, row_end, column_start, column_end in fields:
            self.map_starts.append(len(self.input_offsets) + 1)
            count = 0
            for row in range(row_start, row_end):
                for column in range(column_start, column_end):
                    if 0 <= row < self.input_height and 0 <= column < self.input_width:
                        self.input_offsets.append(row * self.input_width + column)
                        count += 1
            self.map_counts.append(count)
        self.output_spatial = self.output_height * self.output_width
        self.uniform_count = (
            self.map_counts[0]
            if self.map_counts and len(set(self.map_counts)) == 1
            else None
        )

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
                denominator = (
                    self.kernel_area if self.include_padding else self.uniform_count
                )
                finish = append("T2", div(var("result"), denominator))
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
                denominator = self.kernel_area if self.include_padding else var("remaining")
                finish = append("T2", div(var("result"), denominator))
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
