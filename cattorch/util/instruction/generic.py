"""Generic kernels for operations without a specialized lowering.

These are Tier-2 kernels in the general-lowering design: they accept any shape
and any set of dimensions, at the cost of an index table where a specialized
kernel would compute indices arithmetically.
"""

from __future__ import annotations

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
    lt,
    mul,
    not_,
    repeat,
    set_var,
    sub,
    var,
)


def _tensor_slots(args) -> tuple[dict[int, str], str]:
    """Map argument positions to ``T`` list slots, and return the output slot.

    The transpiler binds every non-constant argument, including explicit
    ``_none`` placeholders, to consecutive ``T`` lists before the output.
    """
    slots = {}
    for position, argument in enumerate(args):
        if not str(argument.name).startswith("C_"):
            slots[position] = f"T{len(slots) + 1}"
    return slots, f"T{len(slots) + 1}"


def _value(args, position, default=None):
    if position >= len(args):
        return default
    value = args[position].value
    return default if value is None else value


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


class ReductionInstruction(Instruction):
    """Reduce any set of dimensions with sum, mean, prod, max, min, arg*, any, all.

    When the reduced dimensions are the trailing ones, each output reads one
    contiguous run and the kernel advances a single index. Otherwise it reads
    through a table of output base offsets and a table of reduced offsets.
    """

    def prepare(self):
        self.op = str(self.torch_name)
        self.kind, dims_position, keepdim_position = REDUCTIONS[self.op]
        shape = tuple(self.args[0].shape)
        dims = None if dims_position is None else _value(self.args, dims_position)
        if self.kind in {"argmax", "argmin"} and dims is None:
            dims = tuple(range(len(shape)))
        self.dims = _normalize_dims(dims, len(shape))
        self.slots, self.output_list = _tensor_slots(self.args)
        self.outputs, self.reduced, bases, offsets = _reduction_layout(shape, self.dims)
        self.trailing = self.dims == tuple(range(len(shape) - len(self.dims), len(shape)))
        self.tables = {} if self.trailing else {
            "_reduce_bases": [base + 1 for base in bases],
            "_reduce_offsets": offsets,
        }

    def _read(self):
        if self.trailing:
            return item("T1", var("index"))
        return item("T1", add(var("base"), item("_reduce_offsets", var("position"))))

    def _advance(self):
        return (change_var("index", 1),) if self.trailing else ()

    def _reduce_body(self):
        value = self._read()
        advance = self._advance()
        step = (change_var("position", 1),)
        if self.kind in {"sum", "mean"}:
            return (
                set_var("acc", 0),
                set_var("position", 1),
                repeat(self.reduced, (
                    change_var("acc", value), *advance, *step,
                )),
            )
        if self.kind == "prod":
            return (
                set_var("acc", 1),
                set_var("position", 1),
                repeat(self.reduced, (
                    set_var("acc", mul(var("acc"), value)), *advance, *step,
                )),
            )
        if self.kind in {"any", "all"}:
            hit, miss = (1, 0) if self.kind == "any" else (0, 1)
            condition = not_(eq(value, 0)) if self.kind == "any" else eq(value, 0)
            return (
                set_var("acc", miss),
                set_var("position", 1),
                repeat(self.reduced, (
                    if_(condition, (set_var("acc", hit),)), *advance, *step,
                )),
            )
        compare = gt if self.kind in {"max", "argmax"} else lt
        first = (
            set_var("position", 1),
            set_var("acc", value),
            set_var("best", 0),
            *advance,
            change_var("position", 1),
        )
        # Ties keep the first position, like PyTorch's argmax and argmin.
        update = (set_var("acc", value), set_var("best", sub(var("position"), 1)))
        return first + (
            repeat(self.reduced - 1, (
                if_(compare(value, var("acc")), update), *advance, *step,
            )),
        )

    def _result(self):
        if self.kind == "mean":
            return div(var("acc"), self.reduced)
        if self.kind in {"argmax", "argmin"}:
            return var("best")
        return var("acc")

    def finalize(self):
        per_output = self._reduce_body() + (append(self.output_list, self._result()),)
        if self.trailing:
            setup = (set_var("index", 1),)
            body = per_output
        else:
            setup = (set_var("output", 1),)
            body = (
                set_var("base", item("_reduce_bases", var("output"))),
                *per_output,
                change_var("output", 1),
            )
        lists = (*self.slots.values(), *self.tables, self.output_list)
        return Program(
            "reduction_generic",
            variables=("index", "output", "base", "position", "acc", "best"),
            lists=lists,
            list_values=self.tables,
            body=(clear(self.output_list), *setup, repeat(self.outputs, body)),
        ).compile()


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


class IndexMapInstruction(Instruction):
    """Copy input elements through a static one-based index table.

    Covers expand, repeat, flip, roll, constant padding, triangular masks,
    nearest upsampling, and gathers with static indices. A zero entry writes
    the operation's fill value.
    """

    def __init__(self, torch_name, output, *args, indices, fill):
        self.indices = indices
        self.fill = fill
        super().__init__(torch_name, output, *args)

    def prepare(self):
        pass

    def finalize(self):
        source = item("T1", var("mapped"))
        if self.fill is None:
            write = (append("T2", item("T1", item("_index_map", var("index")))),)
        else:
            write = (
                set_var("mapped", item("_index_map", var("index"))),
                if_else(
                    eq(var("mapped"), 0),
                    (append("T2", self.fill),),
                    (append("T2", source),),
                ),
            )
        return Program(
            "index_map_generic",
            variables=("index", "mapped"),
            lists=("T1", "_index_map", "T2"),
            list_values={"_index_map": self.indices},
            body=(
                clear("T2"),
                set_var("index", 1),
                repeat(len(self.indices), (*write, change_var("index", 1))),
            ),
        ).compile()


__all__ = [
    "IndexMapInstruction", "REDUCTIONS", "ReductionInstruction",
    "index_map",
]
