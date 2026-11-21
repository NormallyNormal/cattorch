"""Turn a finished sprite's public names into its global interface.

A sprite is built with public blocks (``cattorch forward``), completion
broadcasts (``cattorch forward complete``) and public lists and variables
(``input``, ``cattorch status``) of its own. Users must never need to open
a model sprite (a large one makes the editor slow while it is selected), and
other sprites can't call a sprite's blocks or see its local lists, so
``globalize`` gives it the interface other sprites use instead, named after
the export (NAME, the sprite's name):

- each public block becomes an internal procedure (``_cattorch forward``)
  that a ``when I receive [NAME forward]`` script runs, after setting
  ``NAME status`` to ``interface.NOT_FINISHED``; a model, generation or
  program sprite also runs its prepare for save on the shared
  ``cattorch prepare for save``;
- each completion broadcast becomes ``NAME forward complete``;
- the data a project reads or writes (``interface.is_global_data``) becomes
  global: the sprite refers to ``NAME input``, ``NAME status``, ... by name
  and a fresh ID without defining them, and Scratch creates them on the
  stage when the sprite is uploaded (or binds them to existing globals of
  the same name); statuses naming a list name its global;
- everything else stays in the sprite.

Applied to a sprite that is already global, it changes nothing.
"""

from __future__ import annotations

import re
import uuid

from cattorch import interface
from cattorch.util.scratch.interface import refresh_data_display_names
from cattorch.util.scratch.remap import remap_ids

_STATUS_OF_LIST = (
    interface.wrong_number_of_items, interface.too_few_items, interface.too_many_items,
    interface.invalid_item, interface.state_full,
)
# The status the blocks keep while they run; ``NAME status`` is set from it
# when a call ends.
_LOCAL_STATUS = interface.internal_name(interface.STATUS_VARIABLE)
_SHARD_SUFFIX = re.compile(r" shard (?:[2-9]|[1-9][0-9]+)$")


def _fresh_id(kind: str, taken: set[str]) -> str:
    while True:
        identifier = f"cattorch_{kind}_{uuid.uuid4().hex[:20]}"
        if identifier not in taken:
            taken.add(identifier)
            return identifier


def _all_ids(sprite: dict) -> set[str]:
    taken: set[str] = set()
    for section in ("blocks", "variables", "lists", "broadcasts", "comments"):
        taken.update(sprite.get(section, {}))
    return taken


def _broadcast_inputs(blocks: dict):
    """Every ``[11, message, id]`` a block's inputs hold, and every hat's message field."""
    for block in blocks.values():
        if not isinstance(block, dict):
            continue
        option = block.get("fields", {}).get("BROADCAST_OPTION")
        if option is not None:
            yield option
        for value in block.get("inputs", {}).values():
            if not isinstance(value, list):
                continue
            for part in value[1:]:
                if isinstance(part, list) and len(part) >= 3 and part[0] == 11:
                    yield part


def _public_procedures(sprite: dict, public_procedures: frozenset[str]) -> list[str]:
    codes = []
    for block in sprite.get("blocks", {}).values():
        if isinstance(block, dict) and block.get("opcode") == "procedures_prototype":
            code = block["mutation"]["proccode"]
            if (
                not code.startswith(interface.INTERNAL_PREFIX)
                and interface.is_public_procedure(code, public_procedures)
            ):
                codes.append(code)
    return sorted(codes)


def export_messages(export: str, procedures, *, shared: bool) -> list[str]:
    """The broadcasts an export's sprite answers and sends, for its public ``procedures``."""
    messages = []
    for procedure in procedures:
        call = interface.message(export, procedure)
        messages += [call, interface.completion(call)]
    if shared:
        messages.append(interface.SHARED_PREPARE_FOR_SAVE)
    return messages


def globalize(
    sprite: dict, export: str, *,
    public_data_names: frozenset[str] = frozenset(),
    public_procedures: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Give ``sprite`` the global interface of the export ``export``, in place.

    ``public_data_names`` and ``public_procedures`` are the export's own
    public names besides the fixed ones, as built (a program's lists and
    entry points). Returns each data name made global, as built, mapped to
    its global name. Raises ``ValueError`` when two of the export's globals,
    or two of its broadcasts, would be the same to Scratch.
    """
    interface.check_export_name(export)
    blocks = sprite.setdefault("blocks", {})
    taken = _all_ids(sprite)
    procedures = _public_procedures(sprite, public_procedures)
    existing_codes = {
        block["mutation"]["proccode"] for block in blocks.values()
        if isinstance(block, dict) and isinstance(block.get("mutation"), dict)
        and "proccode" in block["mutation"]
    }
    internal = {procedure: interface.internal_name(procedure) for procedure in procedures}
    clashes = sorted(set(internal.values()) & existing_codes)
    if clashes:
        raise RuntimeError(f"cattorch internal error: procedures {clashes} already exist")
    shared = interface.PREPARE_FOR_SAVE in procedures
    owner = f"sprite {export!r}"
    interface.check_messages(
        (owner, text) for text in export_messages(export, procedures, shared=shared)
        if text not in interface.SHARED_MESSAGES
    )

    # A loose block on the canvas, to click, runs a call as a project does.
    for block in blocks.values():
        mutation = block.get("mutation") if isinstance(block, dict) else None
        if (
            block.get("topLevel") and block.get("opcode") == "procedures_call"
            and isinstance(mutation, dict) and mutation.get("proccode") in internal
        ):
            text = interface.message(export, mutation["proccode"])
            del block["mutation"]
            block.update(
                opcode="event_broadcastandwait", fields={},
                inputs={"BROADCAST_INPUT": [1, [11, text, ""]]},
            )

    # Blocks become internal procedures, run by broadcasts.
    for block in blocks.values():
        mutation = block.get("mutation") if isinstance(block, dict) else None
        if isinstance(mutation, dict) and mutation.get("proccode") in internal:
            mutation["proccode"] = internal[mutation["proccode"]]

    # Data a project uses becomes global. The status the blocks keep as they
    # run, and use as a working value, stays in the sprite, as
    # ``_cattorch status``: ``NAME status`` gets it only once a call ends.
    renamed: dict[str, str] = {}
    id_map: dict[str, str] = {}
    names: dict[str, str] = {}
    global_names = []
    local_status = None
    for section in ("lists", "variables"):
        slots = sprite.get(section, {})
        for identifier in list(slots):
            name = slots[identifier][0]
            if section == "variables" and name == interface.STATUS_VARIABLE:
                slots[identifier][0] = names[identifier] = _LOCAL_STATUS
                slots[identifier][1] = interface.INITIAL_STATUS
                local_status = identifier
                continue
            if not interface.is_global_data(name, public_data_names, is_list=section == "lists"):
                continue
            new = interface.global_name(export, name)
            new_id = _fresh_id("global", taken)
            renamed[name] = new
            id_map[identifier] = new_id
            names[new_id] = new
            global_names.append((owner, new))
            del slots[identifier]
    if procedures:
        global_names.append((owner, interface.global_name(export, interface.STATUS_VARIABLE)))
    interface.check_global_names(global_names)
    if id_map:
        blocks = sprite["blocks"] = remap_ids(blocks, id_map)
    refresh_data_display_names(blocks, names)

    # A status names a list by its global name.
    statuses = {
        status(old): status(new)
        for old, new in renamed.items() if not _SHARD_SUFFIX.search(old)
        for status in _STATUS_OF_LIST
    }
    if statuses:
        stack: list = [blocks]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, list):
                if (
                    len(value) == 2 and value[0] == 10 and isinstance(value[1], str)
                    and value[1] in statuses
                ):
                    value[1] = statuses[value[1]]
                else:
                    stack.extend(value)

    # A call's completion broadcast is sent by the script that runs it, once
    # it has set ``NAME status``, rather than by the blocks: one per call.
    completions = {interface.completion(procedure) for procedure in procedures}
    for identifier in [
        identifier for identifier, block in blocks.items()
        if isinstance(block, dict) and block.get("opcode") == "event_broadcastandwait"
        and block["inputs"]["BROADCAST_INPUT"][1][1] in completions
    ]:
        _unlink(blocks, identifier)
    known = {
        *export_messages(export, [*procedures, *_answered(blocks, export)], shared=False),
        *interface.SHARED_MESSAGES,
    }
    for reference in _broadcast_inputs(blocks):
        if reference[_text_index(reference)] not in known:
            raise RuntimeError(
                f"cattorch internal error: the sprite broadcasts "
                f"{reference[_text_index(reference)]!r}, which isn't one of its interface's messages"
            )

    status_ids = ("", "")
    if procedures:
        if local_status is None:
            local_status = _fresh_id("status", taken)
            sprite.setdefault("variables", {})[local_status] = [
                _LOCAL_STATUS, interface.INITIAL_STATUS,
            ]
        status_ids = (_fresh_id("global", taken), local_status)
    for position, procedure in enumerate(procedures):
        messages = [interface.message(export, procedure)]
        if procedure == interface.PREPARE_FOR_SAVE:
            messages.append(interface.SHARED_PREPARE_FOR_SAVE)
        for offset, text in enumerate(messages):
            _add_receiver(
                blocks, taken, text, internal[procedure], export, status_ids,
                completion=interface.completion(interface.message(export, procedure)),
                y=240 * (position + offset * len(procedures)),
            )

    # One fresh ID per message, so that Scratch, which binds a message to an existing one
    # with its ID, never binds it to another export's. Messages are global
    # in Scratch, so the sprite declares none: Scratch creates each on the
    # stage, or binds it to the stage's message of the same name. (One
    # declared in the sprite would be a local message, which Scratch renames
    # to ``SPRITE: message`` when the stage has one of that name.)
    ids: dict[str, str] = {}
    for reference in _broadcast_inputs(blocks):
        text = reference[_text_index(reference)]
        if text not in ids:
            ids[text] = _fresh_id("message", taken)
        reference[-1] = ids[text]
    sprite["broadcasts"] = {}
    return renamed


def global_references(sprite: dict) -> dict[str, dict[str, str]]:
    """What ``sprite``'s blocks use without defining: ``{section: {id: name}}``.

    The sections are ``variables``, ``lists`` and ``broadcasts``: the globals
    and messages Scratch puts on the stage when the sprite is uploaded.
    """
    found: dict[str, dict[str, str]] = {"variables": {}, "lists": {}, "broadcasts": {}}
    kinds = {12: "variables", 13: "lists", 11: "broadcasts"}
    for block in sprite.get("blocks", {}).values():
        if not isinstance(block, dict):
            if isinstance(block, list) and len(block) >= 3 and block[0] in kinds:
                found[kinds[block[0]]][block[2]] = block[1]
            continue
        fields = block.get("fields", {})
        for key, section in (
            ("VARIABLE", "variables"), ("LIST", "lists"), ("BROADCAST_OPTION", "broadcasts"),
        ):
            field = fields.get(key)
            if field is not None and len(field) >= 2 and field[1]:
                found[section][field[1]] = field[0]
        for value in block.get("inputs", {}).values():
            if isinstance(value, list):
                for part in value[1:]:
                    if isinstance(part, list) and len(part) >= 3 and part[0] in kinds:
                        found[kinds[part[0]]][part[2]] = part[1]
    for section in ("variables", "lists"):
        for identifier in sprite.get(section, {}):
            found[section].pop(identifier, None)
    return found


def install(stage: dict, sprite: dict, values=None) -> dict[str, list[str]]:
    """Put ``sprite`` into a project: declare on ``stage`` the globals and messages it uses.

    This is what uploading the sprite does in Scratch, written into the
    project: each global and message is bound to the stage's one of the same
    name (and kind), or declared with ``sprite``'s ID. ``values`` gives the
    value a newly declared global starts with, by name: by default a
    variable holds 0 and a list is empty, as in Scratch. Returns the names of
    the globals declared or bound, ``{"variables": [...], "lists": [...]}``.
    """
    values = values or {}
    found = global_references(sprite)
    remap: dict[str, str] = {}
    used: dict[str, list[str]] = {"variables": [], "lists": []}
    stage.setdefault("broadcasts", {})
    for section, references in found.items():
        slots = stage.setdefault(section, {})
        if section == "broadcasts":
            by_name = {message.lower(): identifier for identifier, message in slots.items()}
        else:
            by_name = {entry[0]: identifier for identifier, entry in slots.items()}
        for identifier, name in references.items():
            key = name.lower() if section == "broadcasts" else name
            if key in by_name:
                if by_name[key] != identifier:
                    remap[identifier] = by_name[key]
            elif section == "broadcasts":
                slots[identifier] = name
                by_name[key] = identifier
            else:
                default: object = 0 if section == "variables" else []
                slots[identifier] = [name, values.get(name, default)]
                by_name[key] = identifier
            if section != "broadcasts" and name not in used[section]:
                used[section].append(name)
    if remap:
        sprite["blocks"] = remap_ids(sprite["blocks"], remap)
    return used


def _text_index(reference: list) -> int:
    """Where a message's text is: ``[text, id]`` in a field, ``[11, text, id]`` in an input."""
    return 1 if len(reference) == 3 else 0


def _answered(blocks: dict, export: str) -> list[str]:
    """The blocks whose messages the sprite already answers, as built names."""
    found = []
    for block in blocks.values():
        if isinstance(block, dict) and block.get("opcode") == "event_whenbroadcastreceived":
            text = block["fields"]["BROADCAST_OPTION"][0]
            if text.startswith(f"{export} "):
                found.append(interface.PUBLIC_PREFIX + text.removeprefix(f"{export} "))
    return found


def _unlink(blocks: dict, identifier: str) -> None:
    """Take the statement ``identifier`` out of its script, joining the blocks around it."""
    block = blocks.pop(identifier)
    following, parent = block.get("next"), block.get("parent")
    if following is not None:
        blocks[following]["parent"] = parent
    if parent is None:
        if following is not None:
            blocks[following].update(topLevel=True, x=block.get("x", 0), y=block.get("y", 0))
        return
    owner = blocks[parent]
    if owner.get("next") == identifier:
        owner["next"] = following
        return
    for key, value in list(owner.get("inputs", {}).items()):
        if isinstance(value, list) and len(value) >= 2 and value[1] == identifier:
            if following is None:
                del owner["inputs"][key]
            else:
                value[1] = following


def _add_receiver(
    blocks: dict, taken: set[str], text: str, procedure: str, export: str,
    status_ids: tuple[str, str], *, completion: str, y: int,
) -> None:
    """``when I receive [text]``: run ``procedure``, publish its status, broadcast ``completion``.

    ``NAME status`` reads ``interface.NOT_FINISHED`` until the call ends: a
    script whose call another broadcast of ``text`` restarted (and so never
    ends) reads that, not an earlier call's status.
    """
    from cattorch.procedures import _procedure_mutation

    hat, mark, run, publish, done = (_fresh_id("receiver", taken) for _ in range(5))
    status = [interface.global_name(export, interface.STATUS_VARIABLE), status_ids[0]]
    blocks[hat] = {
        "opcode": "event_whenbroadcastreceived", "next": mark, "parent": None, "inputs": {},
        "fields": {"BROADCAST_OPTION": [text, ""]}, "shadow": False, "topLevel": True,
        "x": -640, "y": y,
    }
    blocks[mark] = {
        "opcode": "data_setvariableto", "next": run, "parent": hat,
        "inputs": {"VALUE": [1, [10, interface.NOT_FINISHED]]},
        "fields": {"VARIABLE": list(status)}, "shadow": False, "topLevel": False,
    }
    blocks[run] = {
        "opcode": "procedures_call", "next": publish, "parent": mark, "inputs": {}, "fields": {},
        "shadow": False, "topLevel": False, "mutation": _procedure_mutation(procedure),
    }
    blocks[publish] = {
        "opcode": "data_setvariableto", "next": done, "parent": run,
        "inputs": {"VALUE": [3, [12, _LOCAL_STATUS, status_ids[1]], [10, ""]]},
        "fields": {"VARIABLE": list(status)}, "shadow": False, "topLevel": False,
    }
    blocks[done] = {
        "opcode": "event_broadcastandwait", "next": None, "parent": publish,
        "inputs": {"BROADCAST_INPUT": [1, [11, completion, ""]]}, "fields": {},
        "shadow": False, "topLevel": False,
    }
