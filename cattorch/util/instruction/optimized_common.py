"""Building blocks shared by the library's kernels."""

import math

import torch

from cattorch.util.scratch.dsl import (
    add,
    append,
    change_var,
    div,
    item,
    mathop,
    mod,
    mul,
    static_unrolled_repeat,
    sub,
    var,
)


# Activation formulas shared by the elementwise, linear-epilogue and expert
# kernels, so their results agree exactly.

def sigmoid_of(value):
    """``1 / (1 + e^-x)``."""
    return div(1, add(1, mathop("e ^", mul(-1, value))))


def tanh_of(value):
    """``2 / (1 + e^-2x) - 1``, which doesn't overflow for large ``x``."""
    return sub(div(2, add(1, mathop("e ^", mul(-2, value)))), 1)


def quick_gelu_of(value):
    """Fast mode's GELU: ``x * sigmoid(1.702 x)``."""
    return mul(value, div(1, add(1, mathop("e ^", mul(-1.702, value)))))


def tanh_gelu_inner(value):
    """``sqrt(2 / pi) * (x + 0.044715 x^3)``; tanh GELU is
    ``0.5 x (1 + tanh(inner))``."""
    return mul(0.7978845608028654, add(value, mul(0.044715, mul(mul(value, value), value))))


def _unrolled_dot_statements(
    inner: int,
    *,
    left_list: str,
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
    left_list: str,
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
    *,
    destination: str,
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


def argument_at(ctx, position: int, default=None):
    """The tensor or value a kernel's call passed at ``position``.

    ``default`` if the call passed fewer arguments.
    """
    if position >= len(ctx.argument_names):
        return default
    name = ctx.argument_names[position]
    return ctx.inputs[name] if name in ctx.inputs else ctx.scalars.get(name)


def _broadcast_index(shape, output_shape, map_name: str, index=None):
    """Return a cheap exact flat index and any required export-time map."""
    index = var("index") if index is None else index
    shape = tuple(shape)
    output_shape = tuple(output_shape)
    size = math.prod(shape)
    if size == 1:
        return 1, None
    if shape == output_shape:
        return index, None

    # A suffix tensor repeats as one contiguous flat span, covering common
    # bias and causal-mask broadcasting without an auxiliary list lookup.
    trimmed = shape
    while trimmed and trimmed[0] == 1:
        trimmed = trimmed[1:]
    if trimmed and len(trimmed) <= len(output_shape) and trimmed == output_shape[-len(trimmed):]:
        return add(mod(sub(index, 1), math.prod(trimmed)), 1), None

    padded = (1,) * (len(output_shape) - len(shape)) + shape
    indices = torch.arange(size).reshape(padded).expand(output_shape).reshape(-1)
    values = (indices + 1).tolist()
    return item(map_name, index), values
