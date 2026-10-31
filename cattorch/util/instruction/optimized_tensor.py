"""Embedding, layout, masking, concatenation, and slicing kernels."""

import math

import torch

from cattorch.kernel_api import Kernel, KernelContext
from cattorch.util.scratch.sharding import SCRATCH_LIST_LIMIT
from cattorch.util.scratch.dsl import (
    add,
    append,
    change_var,
    clear,
    div,
    eq,
    for_each,
    if_else,
    item,
    length,
    mathop,
    mod,
    mul,
    repeat,
    set_var,
    sub,
    var,
)
from cattorch.util.instruction.optimized_common import (
    _broadcast_index,
    _static_unrolled_repeat,
    _unrolled_copy_statements,
)


def _token_count(tokens):
    # A runtime-length token list sets the count when the code runs.
    return length(tokens.list) if tokens.dynamic_axis is not None else tokens.numel


def _weight_start(token, width: int, interleaved: bool):
    """The first list position of ``token``'s embedding row.

    An interleaved table, shared with a tied output head, stores four rows
    side by side.
    """
    if not interleaved:
        return add(mul(token, width), 1)
    return add(add(mul(mathop("floor", div(token, 4)), 4 * width), mod(token, 4)), 1)


def _emit_embedding(ctx: KernelContext):
    """Embedding lookup that computes the weight-row offset once per token.

    ``ctx.options["interleaved_weight"]`` says the table is interleaved.
    """
    weight, tokens = ctx.inputs["weight"], ctx.inputs["indices"]
    interleaved = bool(ctx.options.get("interleaved_weight"))
    width = weight.shape[1]
    token, weight_index = ctx.temp_var("token"), ctx.temp_var("weight_index")
    out = ctx.output.list
    return (
        clear(out),
        set_var(token, 1),
        repeat(_token_count(tokens), (
            set_var(weight_index, _weight_start(item(tokens.list, var(token)), width, interleaved)),
            *_static_unrolled_repeat(width, (
                append(out, item(weight.list, var(weight_index))),
                change_var(weight_index, 4 if interleaved else 1),
            )),
            change_var(token, 1),
        )),
    )


EMBEDDING_KERNEL = Kernel("aten.embedding.default", _emit_embedding, name="embedding_optimized")


def _emit_embedding_add(ctx: KernelContext):
    """Fuse two equal-width embedding lookups and their elementwise add.

    ``ctx.options["interleaved_sides"]`` says which tables are interleaved.
    """
    first_table, first_tokens = ctx.inputs["first_weight"], ctx.inputs["first_indices"]
    second_table, second_tokens = ctx.inputs["second_weight"], ctx.inputs["second_indices"]
    interleaved = ctx.options.get("interleaved_sides", (False, False))
    width = first_table.shape[1]
    token = ctx.temp_var("token")
    first, second = ctx.temp_var("first_weight"), ctx.temp_var("second_weight")
    out = ctx.output.list
    return (
        clear(out),
        set_var(token, 1),
        repeat(_token_count(first_tokens), (
            set_var(first, _weight_start(item(first_tokens.list, var(token)), width, interleaved[0])),
            set_var(second, _weight_start(
                item(second_tokens.list, var(token)), width, interleaved[1],
            )),
            *_static_unrolled_repeat(width, (
                append(out, add(
                    item(first_table.list, var(first)), item(second_table.list, var(second)),
                )),
                change_var(first, 4 if interleaved[0] else 1),
                change_var(second, 4 if interleaved[1] else 1),
            )),
            change_var(token, 1),
        )),
    )


EMBEDDING_ADD_KERNEL = Kernel(
    "cattorch.embedding_add", _emit_embedding_add, name="embedding_add_optimized",
    arguments=("first_weight", "first_indices", "second_weight", "second_indices"),
)


def _permutation(ctx: KernelContext, rank: int) -> list[int]:
    if ctx.target == "aten.numpy_T.default":
        return list(reversed(range(rank)))
    if ctx.target == "aten.transpose.int":
        dim0, dim1 = ctx.scalars["dim0"] % rank, ctx.scalars["dim1"] % rank
        permutation = list(range(rank))
        permutation[dim0], permutation[dim1] = permutation[dim1], permutation[dim0]
        return permutation
    if ctx.target == "aten.permute.default":
        return [dim % rank for dim in ctx.scalars["dims"]]
    raise NotImplementedError(ctx.target)


def _computed_input_index(source, permutation, index):
    """The input position of output position ``index``, from strides.

    For a runtime-length input, the extent of its runtime axis comes from
    the list's length.
    """
    input_shape = list(source.shape)
    if source.dynamic_axis is not None:
        static_width = math.prod(
            extent for axis, extent in enumerate(input_shape) if axis != source.dynamic_axis
        )
        input_shape[source.dynamic_axis] = div(length(source.list), static_width)
    output_shape = [input_shape[axis] for axis in permutation]

    def product(values):
        result = 1
        for value in values:
            result = mul(result, value)
        return result

    flat = sub(var(index), 1)
    input_index = 0
    for output_axis, input_axis in enumerate(permutation):
        coordinate = mod(
            mathop("floor", div(flat, product(output_shape[output_axis + 1:]))),
            output_shape[output_axis],
        )
        input_index = add(input_index, mul(coordinate, product(input_shape[input_axis + 1:])))
    return add(input_index, 1)


def _emit_transpose(ctx: KernelContext):
    """Transpose by appending in output order.

    A static input of at most one Scratch list reads through a table of
    input positions; otherwise each position is computed from strides.
    """
    source = ctx.inputs["self"]
    permutation = _permutation(ctx, len(source.shape))
    index = ctx.temp_var("index")
    out = ctx.output.list
    if source.dynamic_axis is None and source.numel <= SCRATCH_LIST_LIMIT:
        order = torch.arange(source.numel).reshape(source.shape).permute(permutation).flatten() + 1
        table = ctx.static_list("_transpose_input_order", order.tolist())
        return (
            clear(out),
            set_var(index, 1),
            *_static_unrolled_repeat(source.numel, (
                append(out, item(source.list, item(table, var(index)))),
                change_var(index, 1),
            )),
        )
    total = length(source.list) if source.dynamic_axis is not None else source.numel
    return (
        clear(out),
        set_var(index, 1),
        repeat(total, (
            append(out, item(source.list, _computed_input_index(source, permutation, index))),
            change_var(index, 1),
        )),
    )


TRANSPOSE_KERNELS = tuple(
    Kernel(target, _emit_transpose, name="transpose_optimized")
    for target in ("aten.numpy_T.default", "aten.transpose.int", "aten.permute.default")
)


def _emit_masked_fill(ctx: KernelContext):
    """Masked fill specialized to avoid broadcasting math when shapes match."""
    source, mask = ctx.inputs["self"], ctx.inputs["mask"]
    index = ctx.temp_var("index")
    mask_index, mask_map = _broadcast_index(
        mask.shape, source.shape, "_mask_broadcast_map", var(index),
    )
    out = ctx.output.list
    if mask_map is not None:
        ctx.static_list("_mask_broadcast_map", mask_map)
    return (
        clear(out),
        set_var(index, 1),
        *_static_unrolled_repeat(source.numel, (
            if_else(
                eq(item(mask.list, mask_index), 1),
                (append(out, ctx.scalars["value"]),),
                (append(out, item(source.list, var(index))),),
            ),
            change_var(index, 1),
        )),
    )


MASKED_FILL_KERNEL = Kernel(
    "aten.masked_fill.Scalar", _emit_masked_fill, name="masked_fill_optimized",
)


def _emit_cat(ctx: KernelContext):
    """Pairwise concatenation with partially unrolled contiguous copies.

    Each of ``outer`` rows copies ``first_chunk`` values from the first input,
    then ``second_chunk`` from the second.
    """
    first, second = ctx.inputs["first"], ctx.inputs["second"]
    first_index, second_index = ctx.temp_var("first_index"), ctx.temp_var("second_index")
    out = ctx.output.list
    return (
        clear(out),
        set_var(first_index, 1),
        set_var(second_index, 1),
        repeat(ctx.scalars["outer"], (
            *_unrolled_copy_statements(first.list, first_index, ctx.scalars["first_chunk"], destination=out),
            *_unrolled_copy_statements(second.list, second_index, ctx.scalars["second_chunk"], destination=out),
        )),
    )


CAT_KERNEL = Kernel(
    "cattorch.cat_pair", _emit_cat, name="cat_optimized",
    arguments=("first", "second", "outer", "first_chunk", "second_chunk"),
)


def _emit_getitem(ctx: KernelContext):
    """Static slice or split: ``rows`` runs of ``chunk`` values, ``skip`` apart.

    The copy starts ``offset`` values in. A runtime-length input copied whole
    copies its whole list.
    """
    source = ctx.inputs["source"]
    chunk, rows = ctx.scalars["chunk"], ctx.scalars["rows"]
    skip, offset = ctx.scalars["skip"], ctx.scalars["offset"]
    source_index = ctx.temp_var("source_index")
    out = ctx.output.list
    if source.dynamic_axis is not None and rows == 1 and not skip and not offset:
        return (
            clear(out),
            for_each(source_index, length(source.list), (
                append(out, item(source.list, var(source_index))),
            )),
        )
    copy = ()
    if chunk:
        copy = _unrolled_copy_statements(source.list, source_index, chunk, destination=out)
        if skip:
            copy += (change_var(source_index, skip),)
    traversal = (repeat(rows, copy),) if copy and rows else ()
    return (clear(out), set_var(source_index, offset + 1), *traversal)


GETITEM_KERNEL = Kernel(
    "cattorch.getitem", _emit_getitem, name="getitem_optimized",
    arguments=("source", "chunk", "rows", "skip", "offset"),
)


def _emit_strided_slice(ctx: KernelContext):
    """Copy a fixed positive-step slice of one dimension without index maps.

    Each of ``rows`` outer rows, ``row_stride`` apart, copies ``selected``
    runs of ``trailing`` values, ``step_skip`` apart, starting ``offset`` in.
    """
    source = ctx.inputs["source"]
    outer, index = ctx.temp_var("outer start"), ctx.temp_var("source index")
    out = ctx.output.list
    return (
        clear(out),
        set_var(outer, ctx.scalars["offset"] + 1),
        repeat(ctx.scalars["rows"], (
            set_var(index, var(outer)),
            repeat(ctx.scalars["selected"], (
                *_unrolled_copy_statements(source.list, index, ctx.scalars["trailing"], destination=out),
                change_var(index, ctx.scalars["step_skip"]),
            )),
            change_var(outer, ctx.scalars["row_stride"]),
        )),
    )


STRIDED_SLICE_KERNEL = Kernel(
    "cattorch.strided_slice", _emit_strided_slice, name="slice_strided",
    arguments=("source", "rows", "selected", "trailing", "row_stride", "offset", "step_skip"),
)


__all__ = [
    "CAT_KERNEL", "EMBEDDING_ADD_KERNEL", "EMBEDDING_KERNEL", "GETITEM_KERNEL",
    "MASKED_FILL_KERNEL", "STRIDED_SLICE_KERNEL", "TRANSPOSE_KERNELS",
]
