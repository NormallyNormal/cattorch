"""Run a one-row kernel once per row of a runtime-length input.

The model is traced with one row of its dynamic input, so every kernel is
compiled for exactly one row. For an operation that treats rows
independently, the generated program is wrapped in a loop: each row of the
dynamic inputs is copied to a private list, the original body runs on it,
and its output row is appended to the real output.
"""

from __future__ import annotations

import contextlib
import dataclasses

from cattorch.util.scratch import dsl
from cattorch.util.scratch.dsl import (
    Program, add, append, clear, div, for_each, item, length, mul, repeat, set_var,
    change_var, var,
)

# Statements and expressions whose ``name`` is a list, not a variable.
_LIST_NODES = (
    dsl.ListItem, dsl.ListLength, dsl.ListIndex, dsl.ClearList, dsl.AppendList,
    dsl.ReplaceListItem, dsl.DeleteListItem,
)


@contextlib.contextmanager
def captured_programs():
    """Record every ``Program`` compiled inside the block."""
    programs: list[Program] = []
    original = Program.compile

    def compile(self, *args, **kwargs):
        programs.append(self)
        return original(self, *args, **kwargs)

    Program.compile = compile  # type: ignore[method-assign]
    try:
        yield programs
    finally:
        Program.compile = original  # type: ignore[method-assign]


def _renamed(value, names: dict[str, str]):
    if isinstance(value, tuple):
        return tuple(_renamed(entry, names) for entry in value)
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        return value
    changes = {}
    for field in dataclasses.fields(value):
        current = getattr(value, field.name)
        if field.name == "name" and isinstance(value, _LIST_NODES) and current in names:
            changes[field.name] = names[current]
        else:
            renamed = _renamed(current, names)
            if renamed is not current:
                changes[field.name] = renamed
    return dataclasses.replace(value, **changes) if changes else value


def row_loop(program: Program, rows: dict[str, int], output: str, output_row: int) -> Program:
    """``program`` repeated over rows.

    ``rows`` maps each dynamic input slot (``T1``…) to its items per row, and
    ``output`` is the output slot with ``output_row`` items per row.
    """
    private = {slot: f"_row {slot}" for slot in (*rows, output)}
    body = _renamed(tuple(program.body), private)
    first, first_size = next(iter(rows.items()))
    row, position = "_cattorch row", "_cattorch row item"
    copies = []
    for slot, size in rows.items():
        copies += [
            clear(private[slot]),
            for_each(position, size, (
                append(private[slot], item(slot, add(mul(var(row), size), var(position)))),
            )),
        ]
    loop = (
        clear(output),
        set_var(row, 0),
        repeat(div(length(first), first_size), (
            *copies,
            *body,
            for_each(position, output_row, (
                append(output, item(private[output], var(position))),
            )),
            change_var(row, 1),
        )),
    )
    return Program(
        f"{program.name} by row",
        variables=(*program.variables, row, position),
        lists=(*program.lists, *private.values()),
        list_values=program.list_values,
        variable_values=program.variable_values,
        body=loop,
        optimize_loops=program.optimize_loops,
    )
