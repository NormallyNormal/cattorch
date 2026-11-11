"""The Scratch-side interface of exported sprites, versioned for 1.x.

``docs/scratch-interface.md`` is the contract; this module is where the code
takes every public name and status value from, so that each is written once.

A name in a finished sprite is public when it starts with ``cattorch `` or
is one of a model's ``input``/``output`` lists; every other name starts with
``_`` and is internal. ``is_public_data`` decides it for lists and
variables, and finalization prefixes the rest (``internal_name``). Within an
interface version no public name is renamed or removed, and its meaning,
and each status value's, doesn't change; new ones may be added.
"""

from __future__ import annotations

import re

# 1.0 is the first release whose sprites carry the version: a sprite without
# the variable is from before 1.0. It changes only when a public name or
# status value is renamed or removed, or changes meaning, which 1.x doesn't
# do; adding names doesn't change it.
INTERFACE_VERSION = 1
VERSION_VARIABLE = "cattorch interface version"
PUBLIC_PREFIX = "cattorch "
INTERNAL_PREFIX = "_"

# Every sprite: why the last block that ended ended as it did.
STATUS_VARIABLE = "cattorch status"
# Every sprite with a ``cattorch init`` block: whether the weights are decoded.
INITIALIZED_VARIABLE = "cattorch initialized"

# Generation and tokenizer sprites.
TOKEN_IDS = "cattorch token ids"
LOGITS = "cattorch logits"
TOP_K_VALUES = "cattorch top k values"
TOP_K_IDS = "cattorch top k ids"
CACHE_LENGTH = "cattorch cache length"
MAX_CONTEXT = "cattorch max context"
TEXT = "cattorch text"

# The blocks, by kind of sprite.
INIT, FORWARD, PREPARE_FOR_SAVE = "cattorch init", "cattorch forward", "cattorch prepare for save"
RESET, PREFILL, DECODE = "cattorch reset", "cattorch prefill", "cattorch decode"
TOKENIZE, DETOKENIZE = "cattorch tokenize", "cattorch detokenize"


def completion(block: str) -> str:
    """The broadcast a block sends, with ``broadcast and wait``, whenever it ends."""
    return f"{block} complete"


# ``cattorch status`` values, one per situation (the table in
# docs/scratch-interface.md). ``LIST`` in a value is the public list at
# fault. Every block sets the status when it ends; a sprite starts with the
# initial value, before any block has ended.
INITIAL_STATUS = ""
OK = "ok"
CONTEXT_FULL = "context full"
OVERTAKEN = "another call ran during this one"
INTERRUPTED = "an earlier call was stopped or is still running; reset first"


def wrong_number_of_items(list_name: str) -> str:
    """A fixed-size list doesn't hold its number of items, or a variable-length
    one doesn't hold a whole number of rows."""
    return f"wrong number of items: {list_name}"


def too_few_items(list_name: str) -> str:
    return f"too few items: {list_name}"


def too_many_items(list_name: str) -> str:
    return f"too many items: {list_name}"


def invalid_item(list_name: str) -> str:
    """An item isn't a number, or, for whole-number and index lists, isn't a
    whole number in range."""
    return f"invalid item: {list_name}"


def state_full(list_name: str) -> str:
    """An append would take a state list past its capacity."""
    return f"state full: {list_name}"


def input_name(index: int) -> str:
    """The list for a model's ``index``-th input: ``input``, ``input_1``, ..."""
    return "input" if index == 0 else f"input_{index}"


def output_name(index: int) -> str:
    """The list for a model's ``index``-th output: ``output``, ``output_1``, ..."""
    return "output" if index == 0 else f"output_{index}"


def state_list(state: str) -> str:
    return f"cattorch state {state}"


def entry_list(entrypoint: str, binding: str) -> str:
    """A program entry point's input or output list."""
    return f"cattorch {entrypoint} {binding}"


def entry_procedure(entrypoint: str) -> str:
    return f"cattorch {entrypoint}"


# How the weights are stored, for display, and every value each can hold.
STORAGE_VARIABLES = {
    "cattorch storage compression": ("none", "base92", "base92-huffman-int4"),
    "cattorch storage precision": ("float32", "float16", "int8", "int6", "int4"),
    "cattorch storage scale precision": ("float16", "float32"),
    "cattorch quantization method": ("none", "symmetric", "gptq"),
    "cattorch quantization bits": (0, 4, 6, 8),
}


# Names with no part chosen by the export: a program's entry point, state
# and list names, model inputs and outputs, and project names are added per
# export.
PUBLIC_PROCEDURES = frozenset({
    INIT, FORWARD, PREPARE_FOR_SAVE, RESET, PREFILL, DECODE, TOKENIZE, DETOKENIZE,
})
PUBLIC_DATA = frozenset({
    VERSION_VARIABLE, STATUS_VARIABLE, INITIALIZED_VARIABLE,
    TOKEN_IDS, LOGITS, CACHE_LENGTH, MAX_CONTEXT, TOP_K_VALUES, TOP_K_IDS, TEXT,
    *STORAGE_VARIABLES,
})
PUBLIC_BROADCASTS = frozenset(completion(block) for block in PUBLIC_PROCEDURES)


def shard_variables(list_name: str) -> tuple[str, str]:
    """The variables that describe a split list: its shard count and logical length.

    They are ``cattorch BASE shard count`` and ``cattorch BASE logical length``,
    where BASE is the list's name without the ``cattorch `` it may start with.
    """
    base = list_name.removeprefix(PUBLIC_PREFIX)
    return f"{PUBLIC_PREFIX}{base} shard count", f"{PUBLIC_PREFIX}{base} logical length"


def shard_name(list_name: str, position: int) -> str:
    """The ``position``-th (from 0) physical list of a split list: ``NAME``, ``NAME shard 2``, ..."""
    return list_name if position == 0 else f"{list_name} shard {position + 1}"


_SHARD = re.compile(r"(.+) shard (?:[2-9]|[1-9][0-9]+)")
_SHARD_VARIABLE = re.compile(r"cattorch (.+) (?:shard count|logical length)")
_MODEL_LIST = re.compile(r"(?:input|output)(?:_[1-9][0-9]*)?")


def is_public_data(name: str, extra: frozenset[str] = frozenset(), *, is_list: bool = True) -> bool:
    """Whether a list (or, with ``is_list=False``, a variable) is part of the interface.

    The one rule: the fixed names above, the ``extra`` names an export
    declares (a program's entry point and state lists), and a model's
    ``input``/``output`` lists. A public list's shards (``NAME shard 2``) are
    public, as are the variables describing it (``shard_variables``).
    """
    if not is_list:
        match = _SHARD_VARIABLE.fullmatch(name)
        if match is not None and (
            is_public_data(match.group(1), extra)
            or is_public_data(PUBLIC_PREFIX + match.group(1), extra)
        ):
            return True
        return name in PUBLIC_DATA or name in extra
    match = _SHARD.fullmatch(name)
    base = name if match is None else match.group(1)
    return base in PUBLIC_DATA or base in extra or _MODEL_LIST.fullmatch(base) is not None


def is_public_procedure(name: str, extra: frozenset[str] = frozenset()) -> bool:
    return name in PUBLIC_PROCEDURES or name in extra


def internal_name(name: str) -> str:
    """The name an internal list, variable or block gets: ``_`` and its name."""
    return name if name.startswith(INTERNAL_PREFIX) else INTERNAL_PREFIX + name


def final_data_name(
    name: str, extra: frozenset[str] = frozenset(), *, is_list: bool = True,
) -> str:
    """The name a list or variable has in the finished sprite."""
    return name if is_public_data(name, extra, is_list=is_list) else internal_name(name)


# Projects from ``build_project``. Their global lists, variables and
# messages are named after the sprite they are for.
def project_name(sprite: str, name: str) -> str:
    """A project's global list, variable or message for a sprite's ``name``.

    It is the sprite's name, a space, and ``name`` without the ``cattorch ``
    it may start with: ``points input``, ``story token ids``,
    ``story status``, ``story prefill``.
    """
    return f"{sprite} {name.removeprefix(PUBLIC_PREFIX)}"


def bridge_procedure(message: str) -> str:
    """The block, in the model's or tokenizer's sprite, that a bridge message runs."""
    return f"cattorch bridge {message}"


# What a block checks an input list's items for, before it runs.
NUMBERS, WHOLE_NUMBERS = "numbers", "whole numbers"


def item_checks(
    list_name: str, kind: str | None, rows: int | None = None, *, lists=None, status=None,
):
    """Statements that set the status to ``invalid_item(list_name)`` unless every item is valid.

    ``kind`` is NUMBERS (any number, ``Infinity`` included), WHOLE_NUMBERS, or
    None for no check (bool lists). With ``rows``, a whole number must also be
    from 0 to ``rows - 1``: an index into a table of that many rows.
    ``lists`` are the lists read, by default ``list_name`` (the shards of a
    split list, or a copy). ``status`` names the variable to set,
    ``cattorch status`` by default.
    """
    from cattorch.util.scratch.dsl import (
        add, eq, for_each, if_, item, length, lt, mul, not_, round_, set_var, sub, var,
    )

    if kind is None:
        return ()
    status = status or STATUS_VARIABLE
    index = "_cattorch check index"
    invalid = (set_var(status, invalid_item(list_name)),)
    statements = []
    for read in lists or (list_name,):
        value = item(read, var(index))
        checks: tuple = ()
        if kind == NUMBERS:
            # Text, an empty item and NaN aren't equal to themselves plus 0.
            checks = (if_(not_(eq(value, add(value, 0))), invalid),)
        else:
            checks = (if_(not_(eq(value, round_(value))), invalid),)
            if rows is not None:
                # Negative exactly when the index is outside 0 to rows - 1.
                checks += (if_(lt(mul(value, sub(rows - 1, value)), 0), invalid),)
        statements.append(for_each(index, length(read), checks))
    return tuple(statements)


def length_checks(
    list_name: str, *, numel: int | None = None, width: int = 1,
    min_items: int = 1, max_items: int | None = None, lists=None, status=None,
):
    """Statements that set the status when a list holds the wrong number of items.

    A fixed list holds ``numel`` items. A variable-length one holds a whole
    number of ``width``-item rows, from ``min_items`` to ``max_items`` items
    (no bound for 0 and None); the first applicable of too few, too many and
    not whole rows is reported.
    ``lists`` are the lists counted, by default ``list_name``.
    """
    from cattorch.util.scratch.dsl import add, eq, gt, if_, if_else, length, lt, mod, not_, set_var

    status = status or STATUS_VARIABLE
    reads = list(lists or (list_name,))
    count = length(reads[0])
    for read in reads[1:]:
        count = add(count, length(read))
    if numel is not None:
        return (if_(not_(eq(count, numel)), (
            set_var(status, wrong_number_of_items(list_name)),
        )),)
    checks: tuple = ()
    if width > 1:
        checks = (if_(gt(mod(count, width), 0), (
            set_var(status, wrong_number_of_items(list_name)),
        )),)
    if max_items is not None:
        too_many = (set_var(status, too_many_items(list_name)),)
        checks = (
            (if_else(gt(count, max_items), too_many, checks),) if checks
            else (if_(gt(count, max_items), too_many),)
        )
    if min_items <= 0:
        return checks
    too_few = (set_var(status, too_few_items(list_name)),)
    if not checks:
        return (if_(lt(count, min_items), too_few),)
    return (if_else(lt(count, min_items), too_few, checks),)
