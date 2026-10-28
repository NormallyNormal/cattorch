"""Scratch statements and expressions for writing kernels (experimental).

A kernel's ``emit`` function returns a tuple of the statements built here;
``cattorch.experimental.Kernel`` describes how they join the sprite. This is
the subset of cattorch's internal kernel language that is public. It may
change before 1.0, together with the kernel interface.

Expressions: :func:`var`, :func:`item`, :func:`length`, arithmetic
(:func:`add`, :func:`sub`, :func:`mul`, :func:`div`, :func:`mod`), comparisons
(:func:`eq`, :func:`lt`, :func:`gt`, :func:`not_`), and Scratch's math blocks
through :func:`mathop` and :func:`round_`. Plain numbers are literals.

Statements: variables (:func:`set_var`, :func:`change_var`), lists
(:func:`clear`, :func:`append`, :func:`replace`, :func:`delete`), loops
(:func:`repeat`, :func:`repeat_until`, :func:`for_each`, and
:func:`static_unrolled_repeat` for a loop whose count is known at export),
and conditions (:func:`if_`, :func:`if_else`).

Scratch's ``mathop`` names are ``abs``, ``floor``, ``ceiling``, ``sqrt``,
``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``, ``ln``, ``log``,
``e ^``, and ``10 ^``. Its trigonometry works in degrees, and its lists are
indexed from 1.
"""

from __future__ import annotations

from typing import Iterable

from cattorch.util.scratch.dsl import (
    Expr,
    Statement,
    _render_statements,
    add,
    append,
    change_var,
    clear,
    delete,
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
    not_,
    repeat,
    repeat_until,
    replace,
    round_,
    set_var,
    static_unrolled_repeat,
    sub,
    var,
)


def render(body: Iterable[Statement]) -> str:
    """Readable pseudocode for a kernel body, for review and tests."""
    lines: list[str] = []
    _render_statements(tuple(body), lines, 0)
    return "\n".join(lines)


__all__ = [
    "Expr", "Statement", "add", "append", "change_var", "clear", "delete", "div",
    "eq", "for_each", "gt", "if_", "if_else", "item", "length", "lt", "mathop",
    "mod", "mul", "not_", "render", "repeat", "repeat_until", "replace", "round_",
    "set_var", "static_unrolled_repeat", "sub", "var",
]
