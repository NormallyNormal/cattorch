"""Guards against stopped and overlapping calls of an exported sprite's blocks.

Scratch can pause a warp block where it calls a procedure or repeats a loop,
and run other scripts meanwhile, and a script can be stopped part way. The
blocks of one sprite share their working lists, the decoded weights and any
states, so every kind of export uses one model to keep a call that another
one ran during, or a stopped one, from reporting ``ok`` wrongly:

- ``_EPOCH`` is changed by every guarded call as it starts. Its public block
  runs the work in an epoch procedure (``_add_epoch_procedure``), which takes
  the epoch it started in as an ``epoch`` argument; ``_epoch_current()`` tells
  whether another call started since. A call that finds it overtaken writes
  no output, changes the epoch again (``_overtaken``), so that a call it
  resumed during, which may have read what it wrote, fails too, and reports
  ``interface.OVERTAKEN``.
- ``_BUSY`` is 1 while a call changes states, and stays 1 when that call is
  stopped or overtaken part way: the states are then in an unknown
  condition, and only a reset clears it. Calls that would build on the states
  are refused meanwhile with ``interface.INTERRUPTED``.
- ``cattorch init`` runs in an epoch procedure too. It marks the weights
  undecoded as it starts and, if overtaken, again at its end
  (``_init_outcome``), since a call running meanwhile may have decoded them
  and this init left them half overwritten. The weight decoders take the
  epoch of the call that started them (``_stop_overtaken_decoding``).
- Every public block sets ``cattorch status`` and sends its completion
  broadcast whenever it ends.

What differs between the kinds of export is what is busy, and what is
refused:

=============  ===================  ===================  ===================  ==============
               plain export         program              generation           tokenizer
=============  ===================  ===================  ===================  ==============
epoch          forward, init        each entry point,    reset, prefill,      tokenize,
procedures                          reset, init          decode, init         detokenize
busy flag      none                 reset, and the       reset, prefill and   none
                                    commit of an entry   decode, for the
                                    point that updates   whole call
                                    states (and the
                                    first-call state
                                    fill)
refused while  nothing              every entry point,   decode; the refusal  nothing
busy                                if the program has   also changes the
                                    states; a running    epoch, stopping a
                                    call later reports   running call
                                    its own status
overtaken      outputs staged in    skips its commit and skips its commit,    skips its copy
call           private lists are    marks its entry      marks the weights    to the public
               not copied           point's weights      undecoded; busy      list or
                                    undecoded; one       stays 1              variable
                                    overtaken while it
                                    commits leaves busy
                                    at 1
overtaken      outputs cleared      outputs cleared      logits and top-k     (not checked)
while copying                                            cleared
prepare for    changes the epoch    changes the epoch    changes the epoch,   (no weights)
save                                                     resets the states,
                                                         clears busy
=============  ===================  ===================  ===================  ==============

A plain export and a tokenizer have no states, so their calls are never
refused: an overlapping call only makes the earlier one report that it was
overtaken. A program's entry points are refused only while busy, and only
in a program with states; an entry point that updates no
state never sets busy, so stateless calls may overlap each other. A
generation sprite's states are the cache or recurrent states that every
call reads and writes, so each call holds busy throughout.

Where each policy is written: ``_guard_plain_calls`` (plain);
``program_export._build_entry_wrapper`` and the lifecycle blocks in
``transpile_program`` (programs; a ``RecurrentGenerationProgram`` turns them
off with ``_guard_overlaps=False`` and gets the generation guards instead);
``sprite._add_generation_blocks``, ``sprite._finish_generation_prepare`` and
``_guard_generation_init`` (generation); ``tokenizer._add_tokenizer_interface``.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from cattorch import interface
from cattorch.interface import STATUS_VARIABLE
from cattorch.procedures import (
    _add_argument_reporters, _add_to_procedure, _add_warp_procedure, _find_procedure_definition,
    _owning_definitions, _procedure_arguments, _procedure_mutation,
)
from cattorch.util.scratch.dsl import (
    Program, Statement, append, argument, argument_id, broadcast_and_wait, call, change_var,
    clear, eq, for_each, if_, if_else, item, length, not_, set_var, var,
)
from cattorch.util.scratch.interface import refresh_data_display_names


# 1 while a call is changing the states: while reset, prefill or decode runs in
# a generation sprite, or while a program's reset or entry point writes them.
# Still 1 when another call starts if that one was stopped part way or is
# paused by Scratch, and the states are in an unknown condition.
_BUSY = "_cattorch busy"
# Changed by every call that starts, and by every call that finds, when it
# ends, that it changed since that call started: another call then took over
# while it was paused, and wrote to what it uses, or will write to it, when
# it resumes.
_EPOCH = "_cattorch epoch"


def _epoch_current():
    """Whether no call has started since the one running the epoch procedure body."""
    return eq(argument("epoch"), var(_EPOCH))


def _overtaken() -> tuple[Statement, ...]:
    """What a call does on finding that another one started during it.

    It changes the epoch too: a call it resumed during, still running, may
    have read what it wrote meanwhile, and fails as well.
    """
    return (change_var(_EPOCH, 1), set_var(STATUS_VARIABLE, interface.OVERTAKEN))


def _init_outcome(flags: Sequence[str]) -> Statement:
    """How ``cattorch init`` ends: ok, or overtaken with the weights marked undecoded.

    A call that ran meanwhile may have finished decoding the weights while
    this init was, and this one then left them half overwritten. ``flags``
    are the variables that say the weights are decoded.
    """
    return if_else(_epoch_current(), (set_var(STATUS_VARIABLE, interface.OK),), (
        *(set_var(flag, 0) for flag in flags), *_overtaken(),
    ))


def _add_epoch_procedure(
    sprite: dict, name: str, body, *, x: int, y: int, take_over=(), refusal=None,
    variables: Sequence[str] = (), lists: Sequence[str] = (), completion: str | None = None,
) -> dict:
    """Add the public ``name``, which changes the epoch and runs ``body`` with it.

    Scratch can pause a warp block and run other scripts meanwhile, so the
    body, an internal procedure with an ``epoch`` argument, tells by
    ``_epoch_current()`` whether another call started since. ``take_over``
    runs first; with a ``refusal``, that runs instead while ``_BUSY`` is 1.
    ``variables`` and ``lists`` are what those two use. The block ends,
    whatever happened, with a ``broadcast and wait`` of ``completion``.
    """
    sprite = _add_warp_procedure(
        sprite, _epoch_procedure(name), body, x=x + 320, y=y, arguments=("epoch",),
    )
    return _add_epoch_entry(
        sprite, name, f"{body.name}_public", x=x, y=y, take_over=take_over, refusal=refusal,
        variables=(_BUSY, _EPOCH, *variables), lists=lists, completion=completion,
    )


def _epoch_procedure(name: str) -> str:
    """The internal procedure, with an ``epoch`` argument, that the public ``name`` runs."""
    return f"_cattorch {name.removeprefix('cattorch ')} in epoch %s"


def _add_epoch_entry(
    sprite: dict, name: str, program_name: str, *, x: int, y: int, take_over=(), refusal=None,
    variables: Sequence[str] = (), lists: Sequence[str] = (), completion: str | None = None,
) -> dict:
    """Add the public ``name``, which changes the epoch and runs its epoch procedure.

    As ``_add_epoch_procedure``, for an epoch procedure the sprite already has;
    ``variables`` are all the ones the public block uses.
    """
    start = (*take_over, change_var(_EPOCH, 1), call(_epoch_procedure(name), epoch=var(_EPOCH)))
    body = start if refusal is None else (if_else(eq(var(_BUSY), 1), refusal, start),)
    if completion is not None:
        body = (*body, broadcast_and_wait(completion))
    public = Program(
        program_name, variables=dict.fromkeys(variables), lists=lists, body=body,
    )
    return _add_warp_procedure(sprite, name, public, x=x, y=y)


def _take_epoch_argument(blocks: dict, definition: str, new: str) -> str:
    """Rename the procedure ``definition`` defines to ``new``, with one ``epoch`` argument.

    ``new`` has one ``%s``. It stays a warp block if it was one. Only the
    definition is changed, not the calls to it. Returns ``new``.
    """
    prototype = blocks[definition]["inputs"]["custom_block"][1]
    mutation = _procedure_mutation(new, warp=True, arguments=("epoch",))
    mutation["warp"] = blocks[prototype]["mutation"].get("warp", "false")
    blocks[prototype]["mutation"] = mutation
    blocks[prototype]["inputs"] = {}
    _add_argument_reporters(blocks, prototype, ("epoch",))
    return new


def _make_epoch_procedure(sprite: dict, name: str) -> str:
    """Make procedure ``name`` its own epoch procedure; returns that one's name."""
    return _take_epoch_argument(
        sprite["blocks"], _find_procedure_definition(sprite, name), _epoch_procedure(name),
    )


def _mark_init_decoding(sprite: dict, prefix: str) -> None:
    """Keep the weights marked undecoded while ``cattorch init`` decodes them.

    Init marks them so as it starts, and again at its end if another call
    started meanwhile; it then also changes the epoch, so that a call it
    resumed during fails, and reports that it was overtaken. Either way it
    ends by setting the status and broadcasting ``cattorch init complete``,
    also when another block runs it to decode the weights first: an init a
    block starts itself is like one called. ``cattorch init`` must later run
    in an epoch procedure, with ``_add_init_epoch_entry``.

    This patches the init a plain or generation export already built. A
    program builds its init from its entry points' inits, so
    ``transpile_program`` writes the same guard into that body as it builds
    it: each entry point's flag is cleared before its init runs, and
    ``_init_outcome`` ends the body.
    """
    init, initialized = interface.INIT, interface.INITIALIZED_VARIABLE
    # A call that starts meanwhile decodes the weights itself.
    _add_to_procedure(sprite, init, Program(
        f"{prefix}_init_start", variables=(initialized,), body=(set_var(initialized, 0),),
    ), first=True)
    _add_to_procedure(sprite, init, Program(
        f"{prefix}_init_end", variables=(initialized, _EPOCH, STATUS_VARIABLE),
        body=(_init_outcome((initialized,)), broadcast_and_wait(interface.completion(init))),
    ))


def _add_init_epoch_entry(sprite: dict, inner: str, prefix: str, *, x: int, y: int) -> dict:
    """Add the public ``cattorch init``, which changes the epoch and runs ``inner``.

    ``inner`` is init's epoch procedure, from ``_make_epoch_procedure``. Calls
    to init inside the sprite, such as a step's check that the weights are
    decoded, run ``inner`` with the epoch as it is then, so that init checks
    itself as when it's called directly; ``_stop_overtaken_decoding`` then
    passes them the epoch of the call making them instead, where weights are
    decoded. Loose blocks on the canvas keep running the public block.
    """
    init = interface.INIT
    sprite = _add_epoch_entry(sprite, init, f"{prefix}_init_public", x=x, y=y, variables=(_EPOCH,))
    epoch_id = next(
        identifier for identifier, entry in sprite["variables"].items() if entry[0] == _EPOCH
    )
    for block in sprite["blocks"].values():
        if (
            block.get("opcode") == "procedures_call" and not block.get("topLevel")
            and block["mutation"].get("proccode") == init
        ):
            block["mutation"] = _procedure_mutation(inner, arguments=("epoch",))
            block["inputs"] = {argument_id("epoch"): [3, [12, _EPOCH, epoch_id], [10, ""]]}
    return sprite


def _guard_generation_init(sprite: dict) -> dict:
    """Guard ``cattorch init`` of a generation sprite as a plain export's is.

    The generation blocks already run in epoch procedures. Init changes the
    epoch too, so a reset, prefill or decode that it starts during fails, as
    one that another of them starts during does, and its own decoding is
    checked with ``_mark_init_decoding``. Call it once the sprite's init is
    complete.
    """
    _mark_init_decoding(sprite, "generation")
    inner = _make_epoch_procedure(sprite, interface.INIT)
    sprite = _add_init_epoch_entry(sprite, inner, "generation", x=0, y=800)
    _stop_overtaken_decoding(sprite)
    return sprite


_LOOPS = ("control_repeat", "control_for_each", "control_repeat_until")


# Whether the weight decoders check the epoch at every place Scratch can
# pause them (``_stop_overtaken_decoding``), so that a ``cattorch init``, or
# a block decoding the weights in a freshly loaded project, stops once
# another block has taken over instead of decoding on when it resumes, where
# a block running meanwhile can report ok with weights it half overwrote.
# Off for now: it costs a compressed sprite 110 to 380 blocks in small and
# medium exports (nearly 1,000 in a four-layer int8 generator) and makes
# init run 6 to 23% more blocks, about 30% longer in scratch-vm for a
# medium int8 MLP. The docs name the overlap it would close as a known
# limit. Internal: tests switch it on to keep it working.
_GUARD_DECODERS = False


def _epoch_input(sprite: dict, call_id: str, owner: str | None, epoch_id: str) -> list:
    """The ``epoch`` input of ``call_id``: the epoch its procedure runs in, or the current one.

    A call made in a procedure with an ``epoch`` argument passes that on;
    any other passes the epoch as it is when the call runs.
    """
    blocks = sprite["blocks"]
    if owner is None or "epoch" not in _procedure_arguments(blocks, owner):
        return [3, [12, _EPOCH, epoch_id], [10, ""]]
    reporter = f"{call_id}_epoch"
    blocks[reporter] = {
        "opcode": "argument_reporter_string_number", "next": None, "parent": call_id,
        "inputs": {}, "fields": {"VALUE": ["epoch", None]}, "shadow": False, "topLevel": False,
    }
    return [3, reporter, [10, ""]]


def _stop_overtaken_decoding(sprite: dict) -> None:
    """Keep a call from decoding the weights once another call has started.

    The procedures init's epoch procedure runs directly, such as each entry
    point's init, a recurrent sprite's state fill or a plain sprite's
    unpacking, take the epoch of the call that started them and check it as
    they start. A procedure that starts a decode without an epoch of its
    own, such as a step's check that the weights are decoded, is given the
    epoch of the call running it where every call to it can pass one on. A
    call taken over before it began decoding then decodes nothing, rather
    than decode in the epoch of the call that took over, as if current.

    With ``_GUARD_DECODERS``, a decode taken over part way stops too.
    Scratch can pause a warp block where it calls a procedure or repeats a
    loop, and run other scripts meanwhile. A decode overtaken by a call that
    decodes the weights too, or uses them, must not write to them, nor to
    the decoder's shared lists and variables, once it resumes. Then each
    procedure that init's epoch procedure runs, the decoders, takes the
    epoch and checks it before it writes anything after a place Scratch can
    pause it: at its start, at the start of every repeat of a loop, and
    after a loop or a call, whose callee may have stopped. An overtaken one
    stops; init then finds the epoch changed and leaves the weights for the
    next call to decode again.

    Sprites whose weights need no decoding are left as they are.
    """
    blocks = sprite["blocks"]
    definitions = {}
    for identifier, block in blocks.items():
        if block.get("opcode") == "procedures_definition":
            prototype = blocks[block["inputs"]["custom_block"][1]]
            definitions[prototype["mutation"]["proccode"]] = identifier
    root = _epoch_procedure(interface.INIT)
    if root not in definitions:
        return
    owners = _owning_definitions(blocks)
    calls: dict[str, list[str]] = {}
    for identifier, block in blocks.items():
        if block.get("opcode") == "procedures_call":
            calls.setdefault(owners[identifier] or "", []).append(identifier)

    guarded: list[str] = []
    pending = [definitions[root]]
    while pending:
        for call_id in calls.get(pending.pop(), []):
            name = blocks[call_id]["mutation"]["proccode"]
            definition = definitions.get(name)
            if (
                definition is None or name == root or name in guarded
                or _procedure_arguments(blocks, definition)
            ):
                continue
            guarded.append(name)
            pending.append(definition)
    # Weights stored uncompressed are decoded by no loop.
    decoding = {definitions[name] for name in guarded}
    if not any(
        block.get("opcode") in _LOOPS and owners[identifier] in decoding
        for identifier, block in blocks.items()
    ):
        return
    if not _GUARD_DECODERS:
        direct = {blocks[call_id]["mutation"]["proccode"] for call_id in calls[definitions[root]]}
        guarded = [name for name in guarded if name in direct]
    epoch_id = next(
        identifier for identifier, entry in sprite["variables"].items() if entry[0] == _EPOCH
    )


    callers: dict[str, list[str]] = {}
    for identifier, block in blocks.items():
        if block.get("opcode") == "procedures_call":
            callers.setdefault(block["mutation"]["proccode"], []).append(identifier)

    def take_epoch(name: str) -> str:
        return _take_epoch_argument(blocks, definitions[name], f"{name} for epoch %s")

    renamed = {name: take_epoch(name) for name in guarded}

    def with_epoch(definition: str | None) -> bool:
        return definition is not None and "epoch" in _procedure_arguments(blocks, definition)

    def passing(definition: str | None, found: set[str]) -> bool:
        """Whether ``definition`` can take the epoch from every call to it, adding to ``found``.

        Only an internal procedure without arguments, run only from other
        procedures, each with an epoch or able to take one.
        """
        if definition is None:
            return False
        if with_epoch(definition) or definition in found:
            return True
        name = blocks[blocks[definition]["inputs"]["custom_block"][1]]["mutation"]["proccode"]
        if not name.startswith("_") or _procedure_arguments(blocks, definition):
            return False
        found.add(definition)
        return all(
            not blocks[call_id].get("topLevel") and passing(owners[call_id], found)
            for call_id in callers.get(name, [])
        )

    # A procedure that starts a decode, such as a forward's or step's check
    # that the weights are decoded, starts it in the epoch the call running
    # it started in, passed down to it; a call overtaken before it starts
    # the decode then decodes nothing.
    threaded: set[str] = set()
    for name in (root, *guarded):
        for call_id in callers.get(name, []):
            found: set[str] = set()
            if passing(owners[call_id], found):
                threaded |= found
    procedure_names = {
        definition: blocks[blocks[definition]["inputs"]["custom_block"][1]]["mutation"][
            "proccode"
        ]
        for definition in threaded
    }
    renamed.update({name: take_epoch(name) for name in procedure_names.values()})
    for identifier, block in list(blocks.items()):
        if block.get("opcode") != "procedures_call":
            continue
        name = block["mutation"]["proccode"]
        new = renamed.get(name)
        if new is None and name != root:
            continue
        block["mutation"] = _procedure_mutation(new or name, arguments=("epoch",))
        block["inputs"] = {
            argument_id("epoch"): _epoch_input(sprite, identifier, owners[identifier], epoch_id),
        }

    def check(parent: str, following: str | None) -> str:
        token = uuid.uuid4().hex[:8]
        guard = f"cattorch_overtaken_{token}"
        condition, equal = f"{guard}_not", f"{guard}_current"
        reporter, stop = f"{guard}_epoch", f"{guard}_stop"
        blocks[guard] = {
            "opcode": "control_if", "next": following, "parent": parent,
            "inputs": {"CONDITION": [2, condition], "SUBSTACK": [2, stop]},
            "fields": {}, "shadow": False, "topLevel": False,
        }
        blocks[condition] = {
            "opcode": "operator_not", "next": None, "parent": guard,
            "inputs": {"OPERAND": [2, equal]}, "fields": {}, "shadow": False, "topLevel": False,
        }
        blocks[equal] = {
            "opcode": "operator_equals", "next": None, "parent": condition,
            "inputs": {
                "OPERAND1": [3, reporter, [10, ""]],
                "OPERAND2": [3, [12, _EPOCH, epoch_id], [10, ""]],
            },
            "fields": {}, "shadow": False, "topLevel": False,
        }
        blocks[reporter] = {
            "opcode": "argument_reporter_string_number", "next": None, "parent": equal,
            "inputs": {}, "fields": {"VALUE": ["epoch", None]}, "shadow": False,
            "topLevel": False,
        }
        blocks[stop] = {
            "opcode": "control_stop", "next": None, "parent": guard, "inputs": {},
            "fields": {"STOP_OPTION": ["this script", None]}, "shadow": False,
            "topLevel": False,
            "mutation": {"tagName": "mutation", "children": [], "hasnext": "false"},
        }
        if following is not None:
            blocks[following]["parent"] = guard
        return guard

    def pauses(identifier: str) -> bool:
        # Where Scratch can pause: a loop, or a call, whose callee checks
        # first but may have stopped part way.
        return blocks[identifier]["opcode"] in (*_LOOPS, "procedures_call")

    def guard(first: str, *, start: bool, after_last: bool) -> str:
        """Check the chain from ``first``; returns its new first block."""
        chain = [first]
        while blocks[chain[-1]].get("next") is not None:
            chain.append(blocks[chain[-1]]["next"])
        for position, identifier in enumerate(chain):
            block = blocks[identifier]
            opcode = block["opcode"]
            for key in ("SUBSTACK", "SUBSTACK2"):
                substack = block.get("inputs", {}).get(key)
                if not substack or substack[1] is None:
                    continue
                loop = opcode in _LOOPS
                substack[1] = guard(substack[1], start=loop, after_last=not loop)
            if pauses(identifier) and (position < len(chain) - 1 or after_last):
                block["next"] = check(identifier, block.get("next"))
        if start and not pauses(first):
            parent = blocks[first]["parent"]
            head = check(parent, first)
            if blocks[parent].get("next") == first:
                blocks[parent]["next"] = head
            return head
        return first

    for name in guarded:
        definition = definitions[name]
        first = blocks[definition].get("next")
        if first is not None and not _GUARD_DECODERS:
            blocks[definition]["next"] = check(definition, first)
        elif first is not None:
            blocks[definition]["next"] = guard(first, start=True, after_last=False)


def _guard_plain_calls(sprite: dict, outputs: Sequence[str]) -> dict:
    """Keep stopped and overlapping calls of a plain export from reporting ok wrongly.

    Scratch can pause a warp block and run other scripts meanwhile, and calls
    share temporary lists and the decoded weights. As program entry points
    do, ``cattorch forward`` and ``cattorch init`` change the epoch and run in
    an epoch procedure, and ``cattorch prepare for save`` changes the epoch.

    The forward computes into private copies of ``outputs``, every output
    list and shard, and copies them to the public lists at its end only if
    no other call started meanwhile and ``cattorch status`` is ok, so that a
    call another one ran during never writes the outputs: it reports so,
    and changes the epoch, so that a call it resumed during fails too. Only
    a call overtaken while it copies, which leaves them mixed, clears them.
    An init that another call started during, the forward's own included,
    leaves the weights for the next call to decode again. Every block sets
    the status and sends its completion broadcast as it ends.
    """
    forward = interface.FORWARD
    current = _epoch_current()
    staged = {name: f"_cattorch staged {name}" for name in outputs}
    renamed = {}
    for identifier, entry in sprite.get("lists", {}).items():
        if entry[0] in staged:
            entry[0] = renamed[identifier] = staged[entry[0]]
    refresh_data_display_names(sprite["blocks"], renamed)
    copy = "_cattorch commit index"
    commit = tuple(
        statement
        for public, private in staged.items()
        for statement in (
            clear(public),
            for_each(copy, length(private), (append(public, item(private, var(copy))),)),
        )
    )
    mixed = tuple(clear(public) for public in staged)
    _add_to_procedure(sprite, forward, Program(
        "plain_forward_end", variables=(_EPOCH, STATUS_VARIABLE, copy),
        lists=(*staged, *staged.values()),
        body=(
            if_(current, (if_(eq(var(STATUS_VARIABLE), interface.OK), (
                *commit, if_(not_(current), mixed),
            )),)),
            if_(not_(current), _overtaken()),
        ),
    ))
    _mark_init_decoding(sprite, "plain")
    _make_epoch_procedure(sprite, forward)
    inner_init = _make_epoch_procedure(sprite, interface.INIT)
    sprite = _add_epoch_entry(
        sprite, forward, "plain_forward_public", x=0, y=320, variables=(_EPOCH,),
        completion=interface.completion(forward),
    )
    sprite = _add_init_epoch_entry(sprite, inner_init, "plain", x=0, y=480)
    _stop_overtaken_decoding(sprite)
    try:
        _find_procedure_definition(sprite, interface.PREPARE_FOR_SAVE)
    except ValueError:
        return sprite
    # Clearing the weights takes them from a running call.
    _add_to_procedure(sprite, interface.PREPARE_FOR_SAVE, Program(
        "plain_prepare_start", variables=(_EPOCH,), body=(change_var(_EPOCH, 1),),
    ), first=True)
    _add_to_procedure(sprite, interface.PREPARE_FOR_SAVE, Program(
        "plain_prepare_end", variables=(STATUS_VARIABLE,), body=(
            set_var(STATUS_VARIABLE, interface.OK),
            broadcast_and_wait(interface.completion(interface.PREPARE_FOR_SAVE)),
        ),
    ))
    return sprite
