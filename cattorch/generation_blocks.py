"""The public generation blocks of a sprite, around a one-token step.

``_add_generation_blocks`` adds ``cattorch reset``, ``cattorch prefill`` and
``cattorch decode``, the token checks, the top-k selection and the
cache-length bookkeeping to a sprite that has a one-token step procedure.
KV-cached generation (``_wrap_generation_lifecycle``) and recurrent
generation (``cattorch.recurrent_generation``) both use it, so every
generation sprite has the same interface. Their guards against stopped and
overlapping calls follow ``cattorch.guards``.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence

from cattorch import interface
from cattorch.guards import _BUSY, _EPOCH, _add_epoch_procedure, _epoch_current
from cattorch.interface import STATUS_VARIABLE
from cattorch.procedures import (
    _add_to_procedure, _add_warp_procedure, _find_procedure_definition, _merge_lists_by_name,
    _merge_variables_by_name, _used_lists,
)
from cattorch.util.scratch.dsl import (
    Program, append, broadcast_and_wait, call, change_var, clear, eq, for_each, gt, if_,
    if_else, item, length, not_, repeat, repeat_until, replace, set_var, sub, var,
)


_TOP_K_VALUES = interface.TOP_K_VALUES
_TOP_K_IDS = interface.TOP_K_IDS
_TOP_K_PROCEDURE = "_cattorch select top k"


# An empty slot of the top-k lists. Scratch compares a number with an empty
# string as text, and every number's text, -Infinity and NaN included, sorts
# after "", so the first ``top_k`` logits always fill the empty slots, with
# the same strict comparison that keeps the lower token ID first among
# equal logits. A -Infinity placeholder would never be replaced by a
# -Infinity logit, leaving an ID of -1. NaN's text also sorts after every
# number's, so a NaN logit ranks first, as in ``torch.topk``.
_TOP_K_EMPTY = ""


def _top_k_placeholders(top_k: int) -> list:
    initialize = [clear(_TOP_K_VALUES), clear(_TOP_K_IDS)]
    for _ in range(top_k):
        initialize.extend((append(_TOP_K_VALUES, _TOP_K_EMPTY), append(_TOP_K_IDS, -1)))
    return initialize


def _top_k_insertion(rank: int, top_k: int):
    """Shift a fixed sorted prefix and insert the current logit at ``rank``."""
    shifts: list = []
    for position in range(top_k, rank, -1):
        shifts.extend((
            replace(_TOP_K_VALUES, position, item(_TOP_K_VALUES, position - 1)),
            replace(_TOP_K_IDS, position, item(_TOP_K_IDS, position - 1)),
        ))
    shifts.extend((
        replace(_TOP_K_VALUES, rank, var("top k current")),
        # Scratch lists are one-based; model token IDs are zero-based.
        replace(_TOP_K_IDS, rank, sub(var("top k index"), 1)),
    ))
    return tuple(shifts)


def _top_k_decision(rank: int, top_k: int):
    """Build the short-circuit insertion ladder used by fixed-small top-k."""
    if rank == top_k:
        return _top_k_insertion(rank, top_k)
    return (
        if_else(
            gt(var("top k current"), item(_TOP_K_VALUES, rank)),
            _top_k_insertion(rank, top_k),
            _top_k_decision(rank + 1, top_k),
        ),
    )


def _build_top_k_program(top_k: int, *, logits: str = "output"):
    """Select sorted top-k logits in one pass without mutating ``logits``."""
    initialize = _top_k_placeholders(top_k)
    return Program(
        "generation_top_k",
        variables=("top k index", "top k current"),
        lists=(logits, _TOP_K_VALUES, _TOP_K_IDS),
        body=(
            *initialize,
            for_each("top k index", length(logits), (
                set_var("top k current", item(logits, var("top k index"))),
                if_(
                    gt(var("top k current"), item(_TOP_K_VALUES, top_k)),
                    _top_k_decision(1, top_k),
                ),
            )),
        ),
    )


def _build_compact_top_k_program(top_k: int, *, logits: str = "output"):
    """Select top-k with constant-size dynamic insertion loops."""
    initialize = _top_k_placeholders(top_k)
    return Program(
        "generation_top_k_compact",
        variables=(
            "top k index", "top k current", "top k rank",
            "top k shift",
        ),
        lists=(logits, _TOP_K_VALUES, _TOP_K_IDS),
        body=(
            *initialize,
            for_each("top k index", length(logits), (
                set_var("top k current", item(logits, var("top k index"))),
                if_(
                    gt(var("top k current"), item(_TOP_K_VALUES, top_k)),
                    (
                        set_var("top k rank", 1),
                        repeat_until(
                            gt(
                                var("top k current"),
                                item(_TOP_K_VALUES, var("top k rank")),
                            ),
                            (change_var("top k rank", 1),),
                        ),
                        set_var("top k shift", top_k),
                        repeat(sub(top_k, var("top k rank")), (
                            replace(
                                _TOP_K_VALUES,
                                var("top k shift"),
                                item(_TOP_K_VALUES, sub(var("top k shift"), 1)),
                            ),
                            replace(
                                _TOP_K_IDS,
                                var("top k shift"),
                                item(_TOP_K_IDS, sub(var("top k shift"), 1)),
                            ),
                            change_var("top k shift", -1),
                        )),
                        replace(
                            _TOP_K_VALUES,
                            var("top k rank"),
                            var("top k current"),
                        ),
                        # Scratch lists are one-based; token IDs are zero-based.
                        replace(
                            _TOP_K_IDS,
                            var("top k rank"),
                            sub(var("top k index"), 1),
                        ),
                    ),
                ),
            )),
        ),
    )


# Writes a reporter of the consumed-token count as block ``identifier`` under
# ``parent``.
_TokenCount = Callable[[dict, str, str], None]


def _guard_cache_capacity(first_cache: tuple[str, int]) -> _TokenCount:
    """Measure the generation capacity guard in rows of a K/V cache.

    The count is the cache's length over its per-token width, so a step whose
    rows are rolled back also gives its token back.
    """
    cache_name, cache_width = first_cache

    def report(sprite: dict, identifier: str, parent: str) -> None:
        cache_id = next(
            sid for sid, entry in sprite["lists"].items() if entry[0] == cache_name
        )
        list_length = f"{identifier}_list_length"
        sprite["blocks"][identifier] = {
            "opcode": "operator_divide", "next": None, "parent": parent,
            "inputs": {
                "NUM1": [3, list_length, [4, 0]],
                "NUM2": [1, [4, cache_width]],
            },
            "fields": {}, "shadow": False, "topLevel": False,
        }
        sprite["blocks"][list_length] = {
            "opcode": "data_lengthoflist", "next": None, "parent": identifier,
            "inputs": {}, "fields": {"LIST": [cache_name, cache_id]},
            "shadow": False, "topLevel": False,
        }

    return report


def _add_generation_blocks(
    sprite: dict,
    *,
    step: str = interface.FORWARD,
    reset: str = "cattorch reset cache",
    restore: str | None = None,
    tokens: str = interface.TOKEN_IDS,
    step_tokens: str = "input",
    logits: str = interface.LOGITS,
    step_logits: str = "output",
    state_lists: Sequence[str] = (),
    capacity: int | None,
    hidden_prefill: bool,
    top_k: int | None,
    compact_top_k: bool,
    vocabulary: int | None = None,
    token_count: _TokenCount | None = None,
) -> dict:
    """Add the public reset/prefill/decode blocks around a one-token step.

    ``step`` must already be a procedure whose first block is the automatic
    init check; it reads the one token in ``step_tokens`` and writes
    ``step_logits``. Prefill and decode check the public ``tokens`` first,
    and copy the step's logits to the public ``logits``, and the top-k lists,
    only at their end, when they succeeded and no other call started
    meanwhile, so a call that fails writes no output. A ``reset`` procedure
    is created that calls the ``restore`` procedure, if given, and clears
    ``state_lists``; the public ``cattorch reset`` runs it. Without a
    ``token_count``, ``cattorch cache length`` counts steps that finish with
    an ok status. A ``capacity`` of None sets no token limit.
    """
    buffer = interface.PREFILL_BUFFER
    top_k_lists = (_TOP_K_VALUES, _TOP_K_IDS) if top_k is not None else ()
    if top_k is not None:
        sprite = _add_warp_procedure(
            sprite,
            _TOP_K_PROCEDURE,
            (
                _build_compact_top_k_program(top_k, logits=logits)
                if compact_top_k
                else _build_top_k_program(top_k, logits=logits)
            ),
            x=640,
            y=480,
        )

    # Prefill runs the internal reset too, and writes the public outputs only
    # if it succeeds; the public reset clears them.
    reset_lists = (*state_lists, step_logits)
    reset_program = Program(
        "generation_reset",
        lists=reset_lists,
        body=(
            *((call(restore),) if restore is not None else ()),
            *(clear(name) for name in reset_lists),
        ),
    )
    sprite = _add_warp_procedure(sprite, reset, reset_program, x=640, y=0)

    current = _epoch_current()
    ok = eq(var(STATUS_VARIABLE), interface.OK)
    overtaken = (
        set_var(_BUSY, 1),
        change_var(_EPOCH, 1),
        # Another call may have finished decoding the weights while this one
        # was, and this one then left them half overwritten.
        set_var(interface.INITIALIZED_VARIABLE, 0),
        set_var(STATUS_VARIABLE, interface.OVERTAKEN),
    )
    copy = "_cattorch commit index"
    commit = (
        clear(logits),
        for_each(copy, length(step_logits), (append(logits, item(step_logits, var(copy))),)),
        *((call(_TOP_K_PROCEDURE),) if top_k is not None else ()),
    )
    # A call overtaken while it copies leaves the outputs mixed.
    mixed = (clear(logits), *(clear(name) for name in top_k_lists))
    # The states stay busy until the outputs are copied, so a decode that
    # starts meanwhile is refused rather than build on a call that may yet
    # fail.
    finish = (
        if_(current, (if_(ok, (*commit, if_(not_(current), mixed))),)),
        if_else(current, (set_var(_BUSY, 0),), overtaken),
    )
    lifecycle_variables = (_BUSY, _EPOCH, interface.INITIALIZED_VARIABLE, STATUS_VARIABLE, copy)
    lifecycle_lists = (logits, step_logits, *top_k_lists)

    def add_public(name: str, body: Program, *, y: int, refusal=None) -> None:
        """Add ``name``, which takes the states over and runs ``body`` with its epoch."""
        nonlocal sprite
        sprite = _add_epoch_procedure(
            sprite, name, body, x=640, y=y, take_over=(set_var(_BUSY, 1),), refusal=refusal,
            variables=lifecycle_variables, lists=lifecycle_lists,
            completion=interface.completion(name),
        )

    add_public(interface.RESET, Program(
        "generation_reset_in_epoch", variables=lifecycle_variables, lists=lifecycle_lists,
        body=(
            call(reset), clear(logits), *(clear(name) for name in top_k_lists),
            if_else(current, (set_var(_BUSY, 0),), overtaken),
        ),
    ), y=640)

    # A token that isn't a whole number in the vocabulary would read past the
    # embedding table and still report ok.
    def check_tokens(source: str) -> tuple:
        return interface.item_checks(
            tokens, interface.WHOLE_NUMBERS, vocabulary, lists=(source,),
        )

    def one_step(source: str, position) -> tuple:
        return (clear(step_tokens), append(step_tokens, item(source, position)), call(step))

    decode = Program(
        "generation_decode",
        variables=(
            *lifecycle_variables, "_cattorch check index",
            *((interface.PROJECT_OUTPUT,) if hidden_prefill else ()),
        ),
        lists=(tokens, step_tokens, *lifecycle_lists),
        body=(
            # A stopped hidden prefill can leave it 0.
            *((set_var(interface.PROJECT_OUTPUT, 1),) if hidden_prefill else ()),
            set_var(STATUS_VARIABLE, interface.OK),
            *interface.length_checks(tokens, numel=1),
            if_(ok, check_tokens(tokens)),
            if_(ok, one_step(tokens, 1)),
            *finish,
        ),
    )
    # A refusal sets the status, so a call still running couldn't report its
    # own; it is stopped too.
    add_public(interface.DECODE, decode, y=160, refusal=(
        change_var(_EPOCH, 1), set_var(STATUS_VARIABLE, interface.INTERRUPTED),
    ))

    # The prompt is checked, from a copy, before the prefill resets anything,
    # so a prefill that fails changes nothing. One that another call took
    # over skips the rest of its work.
    if hidden_prefill:
        steps: tuple = (
            if_(current, (set_var(interface.PROJECT_OUTPUT, 0),)),
            for_each("prefill index", sub(length(buffer), 1), (
                if_(current, one_step(buffer, var("prefill index"))),
            )),
            if_(current, (
                set_var(interface.PROJECT_OUTPUT, 1), *one_step(buffer, length(buffer)),
            )),
        )
    else:
        steps = (
            for_each("prefill index", length(buffer), (
                if_(current, one_step(buffer, var("prefill index"))),
            )),
        )
    prefill = Program(
        "generation_prefill",
        variables=(
            "prefill index", "_cattorch check index", *lifecycle_variables,
            *((interface.PROJECT_OUTPUT,) if hidden_prefill else ()),
        ),
        lists=(tokens, buffer, step_tokens, *lifecycle_lists),
        body=(
            set_var(STATUS_VARIABLE, interface.OK),
            clear(buffer),
            for_each("prefill index", length(tokens), (
                append(buffer, item(tokens, var("prefill index"))),
            )),
            *interface.length_checks(tokens, max_items=capacity, lists=(buffer,)),
            if_(ok, check_tokens(buffer)),
            if_(ok, (if_(current, (call(reset),)), *steps)),
            *finish,
        ),
    )
    add_public(interface.PREFILL, prefill, y=320)
    if hidden_prefill:
        # Only prefill clears it, so a decode before any prefill must find 1.
        for entry in sprite["variables"].values():
            if entry[0] == interface.PROJECT_OUTPUT:
                entry[1] = 1
    if top_k_lists:
        _merge_lists_by_name(sprite, set(top_k_lists))

    # Splice the guard and bookkeeping in after the step's automatic init check.
    blocks = sprite["blocks"]
    step_def = _find_procedure_definition(sprite, step)
    init_check = blocks[step_def]["next"]
    old_root = blocks[init_check]["next"]
    suffix = uuid.uuid4().hex[:8]
    prefix = f"cattorch_generation_guard_{suffix}"
    status_id = f"{prefix}_status"
    cache_length_id = f"{prefix}_cache_length"
    sprite.setdefault("variables", {})[status_id] = [STATUS_VARIABLE, interface.INITIAL_STATUS]
    sprite["variables"][cache_length_id] = [interface.CACHE_LENGTH, 0]
    if capacity is not None:
        sprite["variables"][f"{prefix}_max_context"] = [interface.MAX_CONTEXT, capacity]

    def count_input(identifier: str, parent: str, shadow: list) -> list:
        if token_count is None:
            return [3, [12, interface.CACHE_LENGTH, cache_length_id], shadow]
        token_count(sprite, identifier, parent)
        return [3, identifier, shadow]

    success = f"{prefix}_success"
    if capacity is None:
        blocks[init_check]["next"] = success
        success_parent = init_check
    else:
        guard = f"{prefix}_guard"
        condition = f"{prefix}_condition"
        failure = f"{prefix}_failure"
        blocks[init_check]["next"] = guard
        blocks[guard] = {
            "opcode": "control_if_else", "next": None, "parent": init_check,
            "inputs": {
                "CONDITION": [2, condition],
                "SUBSTACK": [2, success],
                "SUBSTACK2": [2, failure],
            },
            "fields": {}, "shadow": False, "topLevel": False,
        }
        blocks[condition] = {
            "opcode": "operator_lt", "next": None, "parent": guard,
            "inputs": {"OPERAND1": None, "OPERAND2": [1, [4, capacity]]},
            "fields": {}, "shadow": False, "topLevel": False,
        }
        blocks[condition]["inputs"]["OPERAND1"] = count_input(
            f"{prefix}_normalized_length", condition, [10, ""],
        )
        success_parent = guard
    blocks[success] = {
        "opcode": "data_setvariableto", "next": old_root, "parent": success_parent,
        "inputs": {"VALUE": [1, [10, interface.OK]]},
        "fields": {"VARIABLE": [STATUS_VARIABLE, status_id]},
        "shadow": False, "topLevel": False,
    }
    blocks[old_root]["parent"] = success
    if capacity is not None:
        # A full context: the step writes nothing.
        blocks[failure] = {
            "opcode": "data_setvariableto", "next": None, "parent": guard,
            "inputs": {"VALUE": [1, [10, interface.CONTEXT_FULL]]},
            "fields": {"VARIABLE": [STATUS_VARIABLE, status_id]},
            "shadow": False, "topLevel": False,
        }

    # Update the public cache-length variable after a successful step.
    terminal = old_root
    while blocks[terminal].get("next") is not None:
        terminal = blocks[terminal]["next"]
    if token_count is None:
        tail = f"{prefix}_count_step"
        step_ok = f"{prefix}_count_step_condition"
        increment = f"{prefix}_count_step_increment"
        blocks[tail] = {
            "opcode": "control_if", "next": None, "parent": terminal,
            "inputs": {"CONDITION": [2, step_ok], "SUBSTACK": [2, increment]},
            "fields": {}, "shadow": False, "topLevel": False,
        }
        blocks[step_ok] = {
            "opcode": "operator_equals", "next": None, "parent": tail,
            "inputs": {
                "OPERAND1": [3, [12, STATUS_VARIABLE, status_id], [10, ""]],
                "OPERAND2": [1, [10, interface.OK]],
            },
            "fields": {}, "shadow": False, "topLevel": False,
        }
        blocks[increment] = {
            "opcode": "data_changevariableby", "next": None, "parent": tail,
            "inputs": {"VALUE": [1, [4, 1]]},
            "fields": {"VARIABLE": [interface.CACHE_LENGTH, cache_length_id]},
            "shadow": False, "topLevel": False,
        }
    else:
        tail = f"{prefix}_set_cache_length"
        blocks[tail] = {
            "opcode": "data_setvariableto", "next": None, "parent": terminal,
            "inputs": {"VALUE": None},
            "fields": {"VARIABLE": [interface.CACHE_LENGTH, cache_length_id]},
            "shadow": False, "topLevel": False,
        }
        blocks[tail]["inputs"]["VALUE"] = count_input(
            f"{prefix}_final_normalized_length", tail, [4, 0],
        )
    blocks[terminal]["next"] = tail
    # Reset also resets public state variables.
    reset_def = _find_procedure_definition(sprite, reset)
    reset_root = blocks[reset_def]["next"]
    reset_status = f"{prefix}_reset_status"
    reset_length = f"{prefix}_reset_length"
    blocks[reset_def]["next"] = reset_status
    blocks[reset_status] = {
        "opcode": "data_setvariableto", "next": reset_length, "parent": reset_def,
        "inputs": {"VALUE": [1, [10, interface.OK]]},
        "fields": {"VARIABLE": [STATUS_VARIABLE, status_id]},
        "shadow": False, "topLevel": False,
    }
    blocks[reset_length] = {
        "opcode": "data_setvariableto", "next": reset_root, "parent": reset_status,
        "inputs": {"VALUE": [1, [4, 0]]},
        "fields": {"VARIABLE": [interface.CACHE_LENGTH, cache_length_id]},
        "shadow": False, "topLevel": False,
    }
    blocks[reset_root]["parent"] = reset_length
    _merge_variables_by_name(sprite, {STATUS_VARIABLE})
    return sprite


def _wrap_generation_lifecycle(
    sprite: dict,
    *,
    cache_names: list[str],
    first_cache: tuple[str, int],
    max_context: int,
    hidden_prefill: bool,
    top_k: int | None,
    compact_top_k: bool,
    vocabulary: int | None = None,
) -> dict:
    """Add the generation blocks for K/V-cached generation."""
    return _add_generation_blocks(
        sprite,
        state_lists=cache_names,
        capacity=max_context,
        hidden_prefill=hidden_prefill,
        top_k=top_k,
        compact_top_k=compact_top_k,
        vocabulary=vocabulary,
        token_count=_guard_cache_capacity(first_cache),
        reset=interface.STATE_RESET,
    )


def _finish_generation_prepare(sprite: dict, reset: str) -> None:
    """Make a generation sprite's ``cattorch prepare for save`` reset it too.

    It clears the decoded weights, and with them its cache, an internal list
    too large to save, so it resets as ``cattorch reset`` does (by the
    internal ``reset`` procedure) and takes over from any running call.
    """
    prepare = interface.PREPARE_FOR_SAVE
    outputs = [
        name for name in (interface.LOGITS, _TOP_K_VALUES, _TOP_K_IDS)
        if any(entry[0] == name for entry in sprite.get("lists", {}).values())
    ]
    # The step's own lists and the prompt's copy hold only what the last call
    # left; the reset that follows refills the states.
    used = _used_lists(sprite)
    # What prepare for save, and the procedures it calls, clear already.
    blocks = sprite["blocks"]
    cleared: set[str] = set()
    pending, seen = [prepare], set()
    while pending:
        procedure = pending.pop()
        if procedure in seen:
            continue
        seen.add(procedure)
        try:
            current = blocks[_find_procedure_definition(sprite, procedure)].get("next")
        except ValueError:
            continue
        while current is not None:
            block = blocks[current]
            if block.get("opcode") == "data_deletealloflist":
                cleared.add(block["fields"]["LIST"][0])
            elif block.get("opcode") == "procedures_call":
                pending.append(block["mutation"]["proccode"])
            current = block.get("next")
    working = [
        entry[0] for entry in sprite.get("lists", {}).values()
        if not entry[1] and entry[0] in used and entry[0] not in cleared
        and entry[0] not in (interface.TOKEN_IDS, *outputs)
    ]
    _add_to_procedure(sprite, prepare, Program(
        "generation_prepare_start", variables=(_EPOCH,), body=(change_var(_EPOCH, 1),),
    ), first=True)
    _add_to_procedure(sprite, prepare, Program(
        "generation_prepare_end", variables=(_BUSY, STATUS_VARIABLE),
        lists=(*outputs, *working), body=(
            *(clear(name) for name in dict.fromkeys(working)),
            call(reset), *(clear(name) for name in outputs), set_var(_BUSY, 0),
            set_var(STATUS_VARIABLE, interface.OK),
            broadcast_and_wait(interface.completion(prepare)),
        ),
    ))
