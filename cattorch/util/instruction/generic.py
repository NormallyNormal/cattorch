"""Generic kernels for operations without a specialized lowering.

These are Tier-2 kernels in the general-lowering design: they accept any shape
and any set of dimensions, at the cost of an index table where a specialized
kernel would compute indices arithmetically.
"""

from __future__ import annotations

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
    lt,
    mul,
    not_,
    repeat,
    set_var,
    sub,
    var,
)


def _normalize_dims(dims, rank: int) -> tuple[int, ...]:
    if dims is None:
        return tuple(range(rank))
    if isinstance(dims, int):
        dims = (dims,)
    dims = tuple(dims)
    if not dims:
        return tuple(range(rank))
    return tuple(sorted({dim % rank for dim in dims})) if rank else ()


# target -> (reduction, dims argument position or None for all, keepdim position)
REDUCTIONS = {
    "aten.sum.dim_IntList": ("sum", 1, 2),
    "aten.sum.default": ("sum", None, None),
    "aten.mean.dim": ("mean", 1, 2),
    "aten.mean.default": ("mean", None, None),
    "aten.prod.dim_int": ("prod", 1, 2),
    "aten.prod.default": ("prod", None, None),
    "aten.amax.default": ("max", 1, 2),
    "aten.amin.default": ("min", 1, 2),
    "aten.max.default": ("max", None, None),
    "aten.min.default": ("min", None, None),
    "aten.argmax.default": ("argmax", 1, 2),
    "aten.argmin.default": ("argmin", 1, 2),
    "aten.any.dim": ("any", 1, 2),
    "aten.any.dims": ("any", 1, 2),
    "aten.any.default": ("any", None, None),
    "aten.all.dim": ("all", 1, 2),
    "aten.all.dims": ("all", 1, 2),
    "aten.all.default": ("all", None, None),
}


def _reduction_layout(shape, dims):
    """Return (outputs, reduced, base offsets, reduced offsets), all one-based."""
    shape = tuple(shape)
    rank = len(shape)
    kept = [dim for dim in range(rank) if dim not in dims]
    strides = [math.prod(shape[dim + 1:]) for dim in range(rank)]
    kept_shape = [shape[dim] for dim in kept]
    reduced_shape = [shape[dim] for dim in dims]
    bases = [0]
    for dim, extent in zip(kept, kept_shape):
        bases = [base + index * strides[dim] for base in bases for index in range(extent)]
    offsets = [0]
    for dim, extent in zip(dims, reduced_shape):
        offsets = [
            offset + index * strides[dim] for offset in offsets for index in range(extent)
        ]
    return math.prod(kept_shape), math.prod(reduced_shape), bases, offsets


def _emit_reduction(ctx: KernelContext):
    """Reduce any set of dimensions with sum, mean, prod, max, min, arg*, any, all.

    When the reduced dimensions are the trailing ones, each output reads one
    contiguous run and the kernel advances a single index. Otherwise it reads
    through a table of output base offsets and a table of reduced offsets.
    """
    kind, dims_position, _ = REDUCTIONS[ctx.target]
    source = ctx.inputs["self"]
    shape = source.shape
    dims = None if dims_position is None else ctx.scalars.get("dim")
    if kind in {"argmax", "argmin"} and dims is None:
        dims = tuple(range(len(shape)))
    dims = _normalize_dims(dims, len(shape))
    outputs, reduced, bases, offsets = _reduction_layout(shape, dims)
    trailing = dims == tuple(range(len(shape) - len(dims), len(shape)))
    # Not "output": a variable of that name would sit beside the public list.
    index, output_index, base, position, acc, best = (
        ctx.temp_var(name)
        for name in ("index", "output_index", "base", "position", "acc", "best")
    )
    if trailing:
        value = item(source.list, var(index))
        advance = (change_var(index, 1),)
    else:
        bases_list = ctx.static_list("_reduce_bases", [start + 1 for start in bases])
        offsets_list = ctx.static_list("_reduce_offsets", offsets)
        value = item(source.list, add(var(base), item(offsets_list, var(position))))
        advance = ()
    out = ctx.output.list
    step = (change_var(position, 1),)
    if kind in {"sum", "mean"}:
        reduce = (
            set_var(acc, 0), set_var(position, 1),
            repeat(reduced, (change_var(acc, value), *advance, *step)),
        )
    elif kind == "prod":
        reduce = (
            set_var(acc, 1), set_var(position, 1),
            repeat(reduced, (set_var(acc, mul(var(acc), value)), *advance, *step)),
        )
    elif kind in {"any", "all"}:
        hit, miss = (1, 0) if kind == "any" else (0, 1)
        condition = not_(eq(value, 0)) if kind == "any" else eq(value, 0)
        reduce = (
            set_var(acc, miss), set_var(position, 1),
            repeat(reduced, (if_(condition, (set_var(acc, hit),)), *advance, *step)),
        )
    else:
        compare = gt if kind in {"max", "argmax"} else lt
        # Ties keep the first position, like PyTorch's argmax and argmin.
        update = (set_var(acc, value), set_var(best, sub(var(position), 1)))
        reduce = (
            set_var(position, 1), set_var(acc, value), set_var(best, 0),
            *advance, change_var(position, 1),
            repeat(reduced - 1, (if_(compare(value, var(acc)), update), *advance, *step)),
        )
    result = (
        div(var(acc), reduced) if kind == "mean"
        else var(best) if kind in {"argmax", "argmin"} else var(acc)
    )
    per_output = (*reduce, append(out, result))
    if trailing:
        loop = (set_var(index, 1), repeat(outputs, per_output))
    else:
        loop = (
            set_var(output_index, 1),
            repeat(outputs, (
                set_var(base, item(bases_list, var(output_index))),
                *per_output,
                change_var(output_index, 1),
            )),
        )
    return (clear(out), *loop)


REDUCTION_KERNELS = tuple(
    Kernel(target, _emit_reduction, name="reduction_generic") for target in REDUCTIONS
)


def index_map(target, input_shape, args, kwargs):
    """Describe a data-movement operation as a one-based gather with fill.

    ``args[0]`` is replaced by ``1..N`` so the result names the source element
    of every output position. Positions that don't change when the input is
    zeroed instead are filled with a constant. Returns ``(indices, fill)``,
    where filled positions have index 0, or raises ``ValueError`` when the
    operation is not pure data movement with one fill value.
    """
    size = math.prod(input_shape)
    probe = torch.arange(1, size + 1, dtype=torch.float64).reshape(input_shape)
    zeros = torch.zeros(input_shape, dtype=torch.float64)
    with torch.no_grad():
        indices = target(probe, *args, **kwargs)
        filled = target(zeros, *args, **kwargs)
    if not isinstance(indices, torch.Tensor):
        raise ValueError("the operation does not return one tensor")
    indices = indices.detach().reshape(-1)
    filled = filled.detach().reshape(-1)
    fill_positions = indices == filled
    moved = indices[~fill_positions]
    if (
        not torch.equal(moved, moved.round())
        or bool((moved < 1).any())
        or bool((moved > size).any())
    ):
        raise ValueError("the operation combines values instead of moving them")
    fill_values = filled[fill_positions].unique()
    if fill_values.numel() > 1:
        raise ValueError("the operation fills with more than one value")
    fill = float(fill_values[0]) if fill_values.numel() else None
    if fill is not None and fill.is_integer():
        fill = int(fill)
    indices = torch.where(fill_positions, torch.zeros_like(indices), indices)
    return [int(value) for value in indices.tolist()], fill


def _emit_index_map(ctx: KernelContext):
    """Copy input elements through a static one-based index table.

    Covers expand, repeat, flip, roll, constant padding, triangular masks,
    nearest upsampling, and gathers with static indices. ``ctx.options``
    holds the table, ``indices``, and the ``fill`` value a zero entry writes.
    """
    indices, fill = ctx.options["indices"], ctx.options["fill"]
    source = next(iter(ctx.inputs.values()))
    index, mapped = ctx.temp_var("index"), ctx.temp_var("mapped")
    table = ctx.static_list("_index_map", indices)
    out = ctx.output.list
    if fill is None:
        write = (append(out, item(source.list, item(table, var(index)))),)
    else:
        write = (
            set_var(mapped, item(table, var(index))),
            if_else(
                eq(var(mapped), 0),
                (append(out, fill),),
                (append(out, item(source.list, var(mapped))),),
            ),
        )
    return (
        clear(out),
        set_var(index, 1),
        repeat(len(indices), (*write, change_var(index, 1))),
    )


# Built for any data-movement operation; the transpiler supplies the table.
INDEX_MAP_KERNEL = Kernel("cattorch.index_map", _emit_index_map, name="index_map_generic")


__all__ = ["INDEX_MAP_KERNEL", "REDUCTIONS", "REDUCTION_KERNELS", "index_map"]
