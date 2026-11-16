"""The Scratch-side interface of exported sprites, versioned for 1.x.

``docs/scratch-interface.md`` is the contract; this module is where the code
takes every public name and status value from, so that each is written once.

Each export has a NAME, its sprite's name. A project uses an exported sprite
without opening it: it broadcasts ``NAME OP`` (``message``) and reads and
writes global lists and variables named ``NAME ...`` (``global_name``). The
sprite is built with the names below (``cattorch forward``, ``input``,
``cattorch status``, ...), and finalization (``globalize`` in
``cattorch.util.scratch.globals``) turns them into that interface:

- The public blocks become internal procedures (``_cattorch forward``),
  each run by a ``when I receive [NAME forward]`` script, and every
  completion broadcast becomes ``NAME OP complete``.
- The data that changes while a project runs (``is_global_data``: every
  public list, and the variables in ``RUNTIME_VARIABLES``) becomes global:
  the sprite refers to ``NAME input``, ``NAME status``, ... without defining
  them, and Scratch creates them on the stage when the sprite is uploaded.
- The other public variables (``cattorch interface version``, the storage
  descriptors, ``cattorch max context``, shard counts, and
  ``cattorch initialized``) stay sprite-local, under their own names, for
  other sprites to read with ``[VAR] of [SPRITE]``. Weights and working
  lists are internal and local.

Every other name starts with ``_`` and is internal; ``is_public_data``
tells the public names from the rest as built, and finalization prefixes
the rest (``internal_name``). Within an interface version no public name is
renamed or removed, and its meaning, and each status value's, doesn't
change; new ones may be added.
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


# Internal names the generation exports share. They are not part of the
# interface; they are here so that the modules that build a generation
# sprite (KV-cached and recurrent) and those that read one agree on them.
# The one-token step that prefill and decode run, and the reset of the
# states that ``cattorch reset``, prefill and prepare for save run.
TOKEN_STEP = "_cattorch token step"
STATE_RESET = "_cattorch reset state"
# The one-token list the step reads, and the logits it writes; the public
# logits are a copy.
STEP_TOKEN = "_cattorch step token"
STEP_LOGITS = "_cattorch step logits"
# The copy of the prompt that prefill steps through, as built; the finished
# sprite has it under its internal name, ``internal_name(PREFILL_BUFFER)``.
PREFILL_BUFFER = "cattorch prefill buffer"
# 1 while a step computes its outputs; a hidden prefill token sets it to 0
# and skips the operations only the outputs need.
PROJECT_OUTPUT = "cattorch project output"


def completion(block: str) -> str:
    """The broadcast a block sends, with ``broadcast and wait``, whenever it ends."""
    return f"{block} complete"


# ``cattorch status`` values, one per situation (the table in
# docs/scratch-interface.md). ``LIST`` in a value is the public list at
# fault. Every block sets the status when it ends; a sprite starts with the
# initial value, before any block has ended.
INITIAL_STATUS = ""
# Set as a call's ``when I receive`` script starts, and replaced by how the
# call ended when it ends. A call another broadcast of the same message
# restarts (Scratch then stops the running script and starts it again) never
# ends, so the script whose ``broadcast and wait`` it ended reads this
# rather than an earlier call's status; so does anything reading the status
# while a call runs, or after the stop sign stopped one.
NOT_FINISHED = "not finished"
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
# A program sprite's blocks besides its entry points'.
PROGRAM_LIFECYCLE = (INIT, RESET, PREPARE_FOR_SAVE)


def broadcast_key(message: str) -> str:
    """What Scratch tells broadcasts apart by: it matches them ignoring case.

    Variable, list and block names, by contrast, are matched exactly.
    """
    return message.lower()


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


# Only what a project reads or writes to use a sprite is global: every
# public list (``is_global_data``) and, of the variables, these. The other
# public variables stay in the sprite, where other sprites read them with
# ``[VAR] of [SPRITE]``: the constants fixed at export, and
# ``cattorch initialized``, which only says whether the next call decodes
# the weights first.
RUNTIME_VARIABLES = frozenset({STATUS_VARIABLE, CACHE_LENGTH, TEXT})
# The one broadcast every model, generation and program sprite answers, under
# this name whatever the sprite's: it runs the sprite's prepare for save, so
# that one ``broadcast and wait`` prepares a whole project for saving. It
# sends no completion broadcast of its own (each sprite still sends its
# ``NAME prepare for save complete``); ``broadcast and wait`` returns once
# every sprite is done.
SHARED_PREPARE_FOR_SAVE = "cattorch prepare for save"
SHARED_MESSAGES = (SHARED_PREPARE_FOR_SAVE,)


def global_name(export: str, name: str) -> str:
    """The global list or variable, or the message, for an export's ``name``.

    It is the export's NAME, a space, and ``name`` without the ``cattorch ``
    it may start with: ``points input``, ``points output shard 2``,
    ``story token ids``, ``story status``, ``story prefill``,
    ``story prefill complete``.
    """
    return f"{export} {name.removeprefix(PUBLIC_PREFIX)}"


def message(export: str, block: str) -> str:
    """The broadcast that runs an export's ``block``: ``NAME forward`` for ``cattorch forward``."""
    return global_name(export, block)


def is_global_data(name: str, extra: frozenset[str] = frozenset(), *, is_list: bool = True) -> bool:
    """Whether a public list or variable, as built, becomes global in the finished sprite.

    Every public list does, and the variables in ``RUNTIME_VARIABLES``; the
    other public variables are constants that stay in the sprite.
    """
    if is_list:
        return is_public_data(name, extra)
    return name in RUNTIME_VARIABLES


# Names Scratch keeps for its own menus (``go to [_mouse_]``,
# ``([x position] of [_stage_])``): a sprite can't be named one of them.
RESERVED_SPRITE_NAMES = frozenset({"_mouse_", "_stage_", "_edge_", "_myself_", "_random_"})


def check_export_name(name: object) -> str:
    """``name`` if it can name an export (its sprite, globals and messages), else ``ValueError``.

    It must be text that isn't empty or only spaces, nor a name Scratch
    keeps for its menus (``RESERVED_SPRITE_NAMES``). A NAME whose messages
    Scratch would match with a shared message (``SHARED_MESSAGES``) is
    refused where the messages are known (``check_messages``).
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    if name in RESERVED_SPRITE_NAMES:
        raise ValueError(f"name {name!r} is reserved by Scratch for its menus; choose another")
    return name


def check_export(name: object, blocks) -> str:
    """Check an export's NAME before exporting it, for its public ``blocks`` as built.

    As ``check_export_name``, and its messages (``NAME OP`` and
    ``NAME OP complete``) must not be the same to Scratch as each other or
    as a shared message. Returns the name.
    """
    name = check_export_name(name)
    owner = f"sprite {name!r}"
    check_messages(
        (owner, text) for block in blocks
        for text in (message(name, block), completion(message(name, block)))
    )
    return name


def check_messages(messages, *, shared=SHARED_MESSAGES) -> None:
    """Refuse broadcasts that Scratch, which matches them ignoring case, can't tell apart.

    ``messages`` are ``(owner, message)`` pairs, the owner saying where the
    message comes from for the error (``"sprite 'points'"``). No two may be
    equal ignoring case unless they are the same message of the same owner,
    and none may equal one of ``shared``, which every sprite answers.
    """
    seen: dict[str, tuple[str, str]] = {
        broadcast_key(text): ("cattorch", text) for text in shared
    }
    for owner, text in messages:
        key = broadcast_key(text)
        if key not in seen:
            seen[key] = (owner, text)
            continue
        other_owner, other = seen[key]
        if (other_owner, other) == (owner, text):
            continue
        if other_owner == "cattorch":
            raise ValueError(
                f"{owner}'s broadcast {text!r} is the message {other!r} that every exported "
                "sprite answers, which Scratch matches ignoring case; choose another name"
            )
        raise ValueError(
            f"{owner}'s broadcast {text!r} and {other_owner}'s {other!r} are the same message to "
            "Scratch, which matches broadcasts ignoring case; rename one"
        )


def check_global_names(names) -> None:
    """Refuse two global lists or variables with the same name.

    ``names`` are ``(owner, name)`` pairs. Scratch keeps lists and variables
    apart, but two globals of one name, even a list and a variable, would
    be the same name in a project's palette and monitors, and two of a kind
    the same global.
    """
    seen: dict[str, str] = {}
    for owner, name in names:
        if name in seen and seen[name] != owner:
            raise ValueError(
                f"{owner} and {seen[name]} would both use the global {name!r}; rename one"
            )
        seen.setdefault(name, owner)


# What a block checks an input list's items for, before it runs.
NUMBERS, WHOLE_NUMBERS = "numbers", "whole numbers"
# The largest whole number an integer input takes, and the negative of the
# smallest: 2^53, past which a Scratch number (a double) skips integers.
WHOLE_LIMIT = 2**53
# A bool list: each item is Scratch's true or false (as a boolean block gives
# them), or the number 1 or 0 (also as text, such as "1"). Kernels read an
# item as false when it equals 0, and arithmetic reads true as 1; text such
# as "true" or "false" is 0 in arithmetic but isn't equal to 0, so it is
# refused rather than read two ways.
BOOLEANS = "booleans"


def item_checks(
    list_name: str, kind: str | None, rows: int | tuple[int, int] | None = None, *,
    lists=None, status=None,
):
    """Statements that set the status to ``invalid_item(list_name)`` unless every item is valid.

    ``kind`` is NUMBERS (any number, ``Infinity`` included), WHOLE_NUMBERS
    (from ``-WHOLE_LIMIT`` to ``WHOLE_LIMIT``), BOOLEANS (true or false, see
    ``BOOLEANS``), or None for no check. With ``rows``, a whole number must
    also be from 0 to ``rows - 1``: an index into a table of that many rows;
    or, as ``(low, high)``, from ``low`` to ``high - 1``, such as ``-size`` to
    ``size - 1`` for an index PyTorch counts from the end when negative.
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
        elif kind == BOOLEANS:
            # True and false are equal to themselves plus 0 (1 and 0), as
            # numbers are; text isn't. Of those, only 1 and 0 are booleans.
            checks = (
                if_(not_(eq(value, add(value, 0))), invalid),
                if_(not_(eq(value, 1)), (if_(not_(eq(value, 0)), invalid),)),
            )
        else:
            checks = (if_(not_(eq(value, round_(value))), invalid),)
            if rows is None:
                # Negative exactly when the number is outside -2^53 to 2^53,
                # the integers a Scratch number holds exactly; Infinity is too.
                checks += (if_(lt(mul(add(value, WHOLE_LIMIT), sub(WHOLE_LIMIT, value)), 0), invalid),)
            else:
                low, high = (0, rows) if isinstance(rows, int) else rows
                # Negative exactly when the index is outside low to high - 1.
                below = value if low == 0 else sub(value, low)
                checks += (if_(lt(mul(below, sub(high - 1, value)), 0), invalid),)
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


def shard_checks(list_name: str, shards, size: int, *, status=None):
    """Statements that set the status when a split list's items aren't laid out in order.

    A list split into ``shards`` (``shard_name``) holds its items in order:
    each shard holds at most ``size``, and every shard before one that holds
    any holds exactly ``size``. With ``length_checks`` on the total, that is
    each shard's own length: for a fixed list, every shard full but the last,
    which holds the rest. A list laid out otherwise reports
    ``wrong_number_of_items(list_name)``.
    """
    from cattorch.util.scratch.dsl import eq, gt, if_, length, not_, set_var

    status = status or STATUS_VARIABLE
    wrong = (set_var(status, wrong_number_of_items(list_name)),)
    checks: list = []
    for position, shard in enumerate(shards):
        checks.append(if_(gt(length(shard), size), wrong))
        if position + 1 < len(shards):
            checks.append(if_(gt(length(shards[position + 1]), 0), (
                if_(not_(eq(length(shard), size)), wrong),
            )))
    return tuple(checks)
