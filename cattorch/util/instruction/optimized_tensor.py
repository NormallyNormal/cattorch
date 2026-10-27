"""Embedding, layout, masking, concatenation, and slicing kernels."""

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
    for_each,
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

class OptimizedEmbeddingInstruction(Instruction):
    """Embedding lookup that computes the weight-row offset once per token."""

    def prepare(self):
        # A runtime-length token list sets the count when the code runs.
        self.tokens = length("T2") if self.args[1].dynamic else math.prod(self.args[1].shape)
        self.width = self.args[0].shape[1]

    def finalize(self):
        interleaved = getattr(self, "interleaved_weight", False)
        token_id = item("T2", var("token"))
        weight_start = (
            add(
                add(
                    mul(mathop("floor", div(token_id, 4)), 4 * self.width),
                    mod(token_id, 4),
                ),
                1,
            )
            if interleaved
            else add(mul(token_id, self.width), 1)
        )
        return Program(
            "embedding_optimized",
            variables=("token", "weight_index"),
            lists=("T1", "T2", "T3"),
            body=(
                clear("T3"),
                set_var("token", 1),
                repeat(self.tokens, (
                    set_var(
                        "weight_index",
                        weight_start,
                    ),
                    *_static_unrolled_repeat(self.width, (
                        append("T3", item("T1", var("weight_index"))),
                        change_var("weight_index", 4 if interleaved else 1),
                    )),
                    change_var("token", 1),
                )),
            ),
        ).compile()


class EmbeddingAddInstruction(Instruction):
    """Fuse two equal-width embedding lookups and their elementwise add."""

    def prepare(self):
        # A runtime-length token list sets the count when the code runs.
        self.tokens = length("T2") if self.args[1].dynamic else math.prod(self.args[1].shape)
        self.width = self.args[0].shape[1]

    def finalize(self):
        interleaved = getattr(self, "interleaved_sides", (False, False))
        first_start = self._weight_start(item("T2", var("token")), interleaved[0])
        second_start = self._weight_start(item("T4", var("token")), interleaved[1])
        return Program(
            "embedding_add_optimized",
            variables=("token", "first_weight", "second_weight"),
            lists=("T1", "T2", "T3", "T4", "T5"),
            body=(
                clear("T5"),
                set_var("token", 1),
                repeat(self.tokens, (
                    set_var(
                        "first_weight",
                        first_start,
                    ),
                    set_var(
                        "second_weight",
                        second_start,
                    ),
                    *_static_unrolled_repeat(self.width, (
                        append(
                            "T5",
                            add(
                                item("T1", var("first_weight")),
                                item("T3", var("second_weight")),
                            ),
                        ),
                        change_var("first_weight", 4 if interleaved[0] else 1),
                        change_var("second_weight", 4 if interleaved[1] else 1),
                    )),
                    change_var("token", 1),
                )),
            ),
        ).compile()

    def _weight_start(self, token, interleaved):
        if not interleaved:
            return add(mul(token, self.width), 1)
        return add(
            add(
                mul(mathop("floor", div(token, 4)), 4 * self.width),
                mod(token, 4),
            ),
            1,
        )


class OptimizedTransposeInstruction(Instruction):
    """Transpose by appending output order instead of preallocating/replacing."""

    def prepare(self):
        input_shape = self.args[0].shape
        ndim = len(input_shape)
        op = str(self.torch_name)
        if op == "aten.numpy_T.default":
            permutation = list(reversed(range(ndim)))
        elif op == "aten.transpose.int":
            dim0 = self.args[1].value % ndim
            dim1 = self.args[2].value % ndim
            permutation = list(range(ndim))
            permutation[dim0], permutation[dim1] = (
                permutation[dim1], permutation[dim0],
            )
        elif op == "aten.permute.default":
            permutation = [dim % ndim for dim in self.args[1].value]
        else:
            raise NotImplementedError(op)

        self.permutation = permutation
        if self.args[0].dynamic:
            static_width = math.prod(
                extent for index, extent in enumerate(input_shape)
                if index != self.args[0].dynamic_axis
            )
            self.runtime_extent = div(length("T1"), static_width)
            self.total = length("T1")
            self.input_order = None
        else:
            self.total = math.prod(input_shape)
            indices = torch.arange(self.total).reshape(input_shape)
            self.input_order = (indices.permute(permutation).flatten() + 1).tolist()

    def _dynamic_input_index(self):
        input_shape = list(self.args[0].shape)
        input_shape[self.args[0].dynamic_axis] = self.runtime_extent
        output_shape = [input_shape[index] for index in self.permutation]

        def product(values):
            result = 1
            for value in values:
                result = mul(result, value)
            return result

        flat = sub(var("index"), 1)
        input_index = 0
        for output_axis, input_axis in enumerate(self.permutation):
            output_stride = product(output_shape[output_axis + 1:])
            coordinate = mod(
                mathop("floor", div(flat, output_stride)),
                output_shape[output_axis],
            )
            input_stride = product(input_shape[input_axis + 1:])
            input_index = add(input_index, mul(coordinate, input_stride))
        return add(input_index, 1)

    def finalize(self):
        if self.input_order is None:
            body = (
                clear("T2"),
                set_var("index", 1),
                repeat(self.total, (
                    append("T2", item("T1", self._dynamic_input_index())),
                    change_var("index", 1),
                )),
            )
            lists = ("T1", "T2")
            values = {}
        else:
            body = (
                clear("T2"),
                set_var("index", 1),
                *_static_unrolled_repeat(self.total, (
                    append(
                        "T2",
                        item(
                            "T1",
                            item("_transpose_input_order", var("index")),
                        ),
                    ),
                    change_var("index", 1),
                )),
            )
            lists = ("T1", "T2", "_transpose_input_order")
            values = {"_transpose_input_order": self.input_order}
        return Program(
            "transpose_optimized",
            variables=("index",),
            lists=lists,
            list_values=values,
            body=body,
        ).compile()


class OptimizedMaskedFillInstruction(Instruction):
    """Masked fill specialized to avoid broadcasting math when shapes match."""

    def prepare(self):
        self.input_size = math.prod(self.args[0].shape)
        self.fill = self.args[2].value
        self.mask_index, mask_map = _broadcast_index(
            self.args[1].shape, self.args[0].shape, "_mask_broadcast_map",
        )
        self.broadcast_lists = (
            {"_mask_broadcast_map": mask_map} if mask_map is not None else {}
        )

    def finalize(self):
        return Program(
            "masked_fill_optimized",
            variables=("index",),
            lists=("T1", "T2", "T3", *self.broadcast_lists),
            list_values=self.broadcast_lists,
            body=(
                clear("T3"),
                set_var("index", 1),
                *_static_unrolled_repeat(self.input_size, (
                    if_else(
                        eq(item("T2", self.mask_index), 1),
                        (append("T3", self.fill),),
                        (append("T3", item("T1", var("index"))),),
                    ),
                    change_var("index", 1),
                )),
            ),
        ).compile()


class OptimizedCatInstruction(Instruction):
    """Pairwise concatenation with partially unrolled contiguous copies."""

    def prepare(self):
        self.outer = self.args[2].value
        self.first_chunk = self.args[3].value
        self.second_chunk = self.args[4].value

    def finalize(self):
        return Program(
            "cat_optimized",
            variables=("first_index", "second_index"),
            lists=("T1", "T2", "T3"),
            body=(
                clear("T3"),
                set_var("first_index", 1),
                set_var("second_index", 1),
                repeat(self.outer, (
                    *_unrolled_copy_statements(
                        "T1", "first_index", self.first_chunk,
                    ),
                    *_unrolled_copy_statements(
                        "T2", "second_index", self.second_chunk,
                    ),
                )),
            ),
        ).compile()


class OptimizedGetItemInstruction(Instruction):
    """Static slice/split copy with a partially unrolled inner span."""

    def prepare(self):
        self.chunk = self.args[1].value
        self.rows = self.args[2].value
        self.skip = self.args[3].value
        self.offset = self.args[4].value
        self.dynamic_copy = (
            self.args[0].dynamic and self.rows == 1 and not self.skip and not self.offset
        )

    def finalize(self):
        if self.dynamic_copy:
            return Program(
                "getitem_dynamic_copy",
                variables=("source_index",),
                lists=("T1", "T2"),
                body=(
                    clear("T2"),
                    for_each(
                        "source_index", length("T1"),
                        (append("T2", item("T1", var("source_index"))),),
                    ),
                ),
            ).compile()
        copy = ()
        if self.chunk:
            copy = _unrolled_copy_statements(
                "T1", "source_index", self.chunk, destination="T2",
            )
            if self.skip:
                copy += (change_var("source_index", self.skip),)
        traversal = (repeat(self.rows, copy),) if copy and self.rows else ()
        return Program(
            "getitem_optimized",
            variables=("source_index",),
            lists=("T1", "T2"),
            body=(
                clear("T2"),
                set_var("source_index", self.offset + 1),
                *traversal,
            ),
        ).compile()


class StridedSliceInstruction(Instruction):
    """Copy a fixed positive-step dimension slice without index maps."""

    def prepare(self):
        self.rows = self.args[1].value
        self.selected = self.args[2].value
        self.trailing = self.args[3].value
        self.row_stride = self.args[4].value
        self.offset = self.args[5].value
        self.step_skip = self.args[6].value

    def finalize(self):
        return Program(
            "slice_strided",
            variables=("outer start", "source index"),
            lists=("T1", "T2"),
            body=(
                clear("T2"),
                set_var("outer start", self.offset + 1),
                repeat(self.rows, (
                    set_var("source index", var("outer start")),
                    repeat(self.selected, (
                        *_unrolled_copy_statements(
                            "T1", "source index", self.trailing, destination="T2",
                        ),
                        change_var("source index", self.step_skip),
                    )),
                    change_var("outer start", self.row_stride),
                )),
            ),
        ).compile()
