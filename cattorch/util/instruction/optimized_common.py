"""Optimized kernels authored with the internal Scratch DSL."""

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
    static_unrolled_repeat,
    replace,
    set_var,
    sub,
    var,
)


def _unrolled_dot_statements(
    inner: int,
    *,
    left_list: str = "T1",
    right_list: str,
    right_index: str,
    right_step: int,
    factor: int = 4,
):
    """Build a partially unrolled dot product with no dead final increments."""
    product = change_var(
        "sum",
        mul(
            item(left_list, var("left_index")),
            item(right_list, var(right_index)),
        ),
    )
    advancing_step = (
        product,
        change_var("left_index", 1),
        change_var(right_index, right_step),
    )
    final_step = (product,)

    return (static_unrolled_repeat(
        inner,
        advancing_step,
        max_factor=factor,
        final_step=final_step,
        priority=3,
    ),)


def _unrolled_offset_dot_statements(
    inner: int,
    *,
    left_list: str = "T1",
    right_list: str,
    right_offset: str,
    factor: int = 4,
):
    """Build a contiguous dot using ``right_offset + left_index`` addressing.

    Scratch executes an addition reporter more cheaply than a second variable
    mutation in this hot loop.  The offset is established once per row/output,
    while ``left_index`` is the only index changed for each multiply.
    """
    product = change_var(
        "sum",
        mul(
            item(left_list, var("left_index")),
            item(right_list, add(var(right_offset), var("left_index"))),
        ),
    )
    advancing_step = (product, change_var("left_index", 1))
    final_step = (product,)

    return (static_unrolled_repeat(
        inner,
        advancing_step,
        max_factor=factor,
        final_step=final_step,
        priority=3,
    ),)


def _unrolled_copy_statements(
    source: str,
    index: str,
    count: int,
    factor: int = 8,
    destination: str = "T3",
):
    """Append a contiguous span with the browser-proven lightweight unroll."""
    step = (
        append(destination, item(source, var(index))),
        change_var(index, 1),
    )
    return (static_unrolled_repeat(
        count, step, max_factor=factor, priority=1,
    ),)


def _static_unrolled_repeat(count: int, step, factor: int = 4):
    """Lower a static repeat count with fewer Scratch loop dispatches."""
    return (static_unrolled_repeat(
        count, tuple(step), max_factor=factor, priority=2,
    ),)


def _broadcast_index(shape, output_shape, map_name: str):
    """Return a cheap exact flat index and any required export-time map."""
    shape = tuple(shape)
    output_shape = tuple(output_shape)
    size = math.prod(shape)
    if size == 1:
        return 1, None
    if shape == output_shape:
        return var("index"), None

    # A suffix tensor repeats as one contiguous flat span, covering common
    # bias and causal-mask broadcasting without an auxiliary list lookup.
    trimmed = shape
    while trimmed and trimmed[0] == 1:
        trimmed = trimmed[1:]
    if trimmed and len(trimmed) <= len(output_shape) and trimmed == output_shape[-len(trimmed):]:
        return add(mod(sub(var("index"), 1), math.prod(trimmed)), 1), None

    padded = (1,) * (len(output_shape) - len(shape)) + shape
    indices = torch.arange(size).reshape(padded).expand(output_shape).reshape(-1)
    values = (indices + 1).tolist()
    return item(map_name, var("index")), values
