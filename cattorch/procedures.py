"""Custom blocks (procedures) in a sprite's block dictionary.

Helpers that every kind of export uses to add a DSL program to a sprite as a
warp block, find and edit a procedure, and share the data that several
added programs declare by display name.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence

from cattorch import interface
from cattorch.interface import STATUS_VARIABLE
from cattorch.util.scratch.dsl import argument_id
from cattorch.util.scratch.remap import remap_ids


def _merge_slots_by_name(
    sprite: dict, section: str, names: set[str],
    added: tuple[frozenset[str], frozenset[str]] | None,
) -> None:
    """Keep the first of the slots in ``section`` that share a name in ``names``.

    ``added`` is the IDs of the slots and blocks just added to the sprite.
    When every merged-away slot is among those slots, only those blocks can
    refer to one, so only they are remapped.
    """
    slots = sprite.get(section, {})
    ids_by_name: dict[str, list[str]] = {}
    for identifier, entry in slots.items():
        if entry[0] in names:
            ids_by_name.setdefault(entry[0], []).append(identifier)
    remap = {}
    for ids in ids_by_name.values():
        winner = ids[0]
        for sid in ids[1:]:
            remap[sid] = winner
            del slots[sid]
    if not remap:
        return
    blocks = sprite["blocks"]
    if added is not None and remap.keys() <= added[0]:
        blocks.update(remap_ids({identifier: blocks[identifier] for identifier in added[1]}, remap))
    else:
        sprite["blocks"] = remap_ids(blocks, remap, share_unchanged=True)


def _merge_lists_by_name(
    sprite, names: set[str], *, added: tuple[frozenset[str], frozenset[str]] | None = None,
) -> None:
    """Unify repeated DSL declarations of persistent generation lists."""
    _merge_slots_by_name(sprite, "lists", names, added)


def _merge_variables_by_name(
    sprite, names: set[str], *, added: tuple[frozenset[str], frozenset[str]] | None = None,
) -> None:
    """Unify repeated DSL declarations of persistent generation variables."""
    _merge_slots_by_name(sprite, "variables", names, added)


def _procedure_mutation(
    name: str, *, warp: bool = False, arguments: Sequence[str] = (),
) -> dict:
    def encode(values) -> str:
        return json.dumps(list(values), separators=(",", ":"))

    mutation = {
        "tagName": "mutation",
        "children": [],
        "proccode": name,
        "argumentids": encode(argument_id(argument) for argument in arguments),
    }
    if warp:
        mutation.update({
            "argumentnames": encode(arguments),
            "argumentdefaults": encode("" for _ in arguments),
            "warp": "true",
        })
    return mutation


def _add_warp_procedure(
    sprite: dict, name: str, program, *, x: int, y: int, arguments: Sequence[str] = (),
) -> dict:
    """Combine a DSL command stack and expose it as a warp block.

    ``name`` is the proccode, with a ``%s`` for each of ``arguments``.
    """
    data = program.compile()
    roots = [
        bid for bid, block in data["blocks"].items()
        if block.get("topLevel") and block.get("parent") is None
    ]
    if len(roots) != 1:
        raise ValueError(f"expected one root for procedure {name!r}")
    root = roots[0]
    suffix = uuid.uuid4().hex[:8]
    definition = f"cattorch_generation_{suffix}_definition"
    prototype = f"cattorch_generation_{suffix}_prototype"
    data["blocks"][root]["topLevel"] = False
    data["blocks"][root].pop("x", None)
    data["blocks"][root].pop("y", None)
    data["blocks"][root]["parent"] = definition
    data["blocks"][definition] = {
        "opcode": "procedures_definition", "next": root, "parent": None,
        "inputs": {"custom_block": [1, prototype]}, "fields": {},
        "shadow": False, "topLevel": True, "x": x, "y": y,
    }
    data["blocks"][prototype] = {
        "opcode": "procedures_prototype", "next": None, "parent": definition,
        "inputs": {}, "fields": {}, "shadow": True, "topLevel": False,
        "mutation": _procedure_mutation(name, warp=True, arguments=arguments),
    }
    for argument in arguments:
        reporter = f"{prototype}_{argument_id(argument)}"
        data["blocks"][prototype]["inputs"][argument_id(argument)] = [1, reporter]
        data["blocks"][reporter] = {
            "opcode": "argument_reporter_string_number", "next": None,
            "parent": prototype, "inputs": {}, "fields": {"VALUE": [argument, None]},
            "shadow": True, "topLevel": False,
        }
    # Lifecycle wrapping gives the sprite several legitimate top-level roots,
    # whereas the ordinary block combiner expects one computation chain.  A
    # procedure is independent, so merge it directly after uniquifying every
    # generated ID, then share data the sprite already has by display name.
    declared = {
        section: {entry[0] for entry in data.get(section, {}).values()}
        for section in ("variables", "lists")
    }
    mapping = {
        identifier: f"{identifier}_{suffix}"
        for section in ("blocks", "variables", "lists", "broadcasts")
        for identifier in data.get(section, {})
    }
    data = remap_ids(data, mapping)
    sprite["blocks"].update(data.get("blocks", {}))
    sprite.setdefault("variables", {}).update(data.get("variables", {}))
    sprite.setdefault("lists", {}).update(data.get("lists", {}))
    existing_broadcasts = {
        message: identifier
        for identifier, message in sprite.setdefault("broadcasts", {}).items()
    }
    broadcast_remap = {}
    for identifier, message in data.get("broadcasts", {}).items():
        winner = existing_broadcasts.get(message)
        if winner is None:
            sprite["broadcasts"][identifier] = message
            existing_broadcasts[message] = identifier
        else:
            broadcast_remap[identifier] = winner
    if broadcast_remap:
        sprite["blocks"] = remap_ids(sprite["blocks"], broadcast_remap, share_unchanged=True)
    # Every ID in data carries this call's fresh suffix, so the sprite's
    # other blocks cannot refer to the slots it added.
    added = (
        frozenset((*data.get("variables", {}), *data.get("lists", {}))),
        frozenset(data.get("blocks", {})),
    )
    _merge_variables_by_name(sprite, declared["variables"], added=added)
    _merge_lists_by_name(sprite, declared["lists"], added=added)
    return sprite


def _find_procedure_definition(sprite: dict, name: str) -> str:
    for bid, block in sprite["blocks"].items():
        if block.get("opcode") != "procedures_definition":
            continue
        custom = block.get("inputs", {}).get("custom_block")
        if not custom:
            continue
        prototype = sprite["blocks"].get(custom[1], {})
        if prototype.get("mutation", {}).get("proccode") == name:
            return bid
    raise ValueError(f"procedure not found: {name}")


def _remove_canvas_calls(sprite: dict, name: str) -> None:
    """Remove the loose call blocks for ``name`` that lifecycle wrapping left on the canvas."""
    blocks = sprite["blocks"]
    for identifier in [
        identifier for identifier, block in blocks.items()
        if block.get("opcode") == "procedures_call" and block.get("topLevel")
        and block.get("mutation", {}).get("proccode") == name
    ]:
        del blocks[identifier]


def _remove_completion_broadcast(sprite: dict, message: str) -> None:
    """Remove every ``broadcast and wait`` of ``message`` that ends a procedure.

    A ``cattorch status`` set just before the broadcast goes with it; the
    two must be the last blocks of their procedure.
    """
    blocks = sprite["blocks"]
    for identifier, block in list(blocks.items()):
        if block.get("opcode") != "event_broadcastandwait":
            continue
        menu = block.get("inputs", {}).get("BROADCAST_INPUT", [None, None])[1]
        if not (isinstance(menu, list) and len(menu) > 1 and menu[1] == message):
            continue
        last = identifier
        parent = block.get("parent")
        if parent is not None and blocks[parent].get("fields", {}).get(
            "VARIABLE", [None],
        )[0] == STATUS_VARIABLE:
            last, parent = parent, blocks[parent].get("parent")
        if parent is not None:
            blocks[parent]["next"] = None
        del blocks[identifier]
        if last != identifier:
            del blocks[last]


def _insert_init_call(sprite: dict, procedure: str) -> None:
    """Make ``cattorch init`` call ``procedure`` before everything it already runs."""
    blocks = sprite["blocks"]
    init_definition = _find_procedure_definition(sprite, interface.INIT)
    old_root = blocks[init_definition]["next"]
    identifier = f"cattorch_storage_init_{uuid.uuid4().hex[:8]}"
    blocks[identifier] = {
        "opcode": "procedures_call", "next": old_root, "parent": init_definition,
        "inputs": {}, "fields": {}, "shadow": False, "topLevel": False,
        "mutation": _procedure_mutation(procedure),
    }
    blocks[init_definition]["next"] = identifier
    if old_root in blocks:
        blocks[old_root]["parent"] = identifier


def _add_to_procedure(sprite: dict, name: str, program, *, first: bool = False) -> None:
    """Run ``program``'s statements at the end of procedure ``name``, or ``first``."""
    temporary = f"_cattorch adding to {name}"
    _add_warp_procedure(sprite, temporary, program, x=0, y=0)
    blocks = sprite["blocks"]
    definition = _find_procedure_definition(sprite, temporary)
    head = blocks[definition]["next"]
    del blocks[blocks[definition]["inputs"]["custom_block"][1]]
    del blocks[definition]
    tail = head
    while blocks[tail].get("next") is not None:
        tail = blocks[tail]["next"]
    target = _find_procedure_definition(sprite, name)
    if first:
        rest = blocks[target]["next"]
        blocks[target]["next"], blocks[head]["parent"] = head, target
        blocks[tail]["next"] = rest
        if rest is not None:
            blocks[rest]["parent"] = tail
        return
    last = target
    while blocks[last].get("next") is not None:
        last = blocks[last]["next"]
    blocks[last]["next"], blocks[head]["parent"] = head, last


def _procedure_arguments(blocks: dict, definition: str) -> list[str]:
    prototype = blocks[blocks[definition]["inputs"]["custom_block"][1]]
    return json.loads(prototype["mutation"].get("argumentnames") or "[]")


def _owning_definitions(blocks: dict) -> dict[str, str | None]:
    """The procedure definition each block is part of, or None outside one."""
    owners: dict[str, str | None] = {}
    for identifier in blocks:
        path = []
        current: str | None = identifier
        while current is not None and current not in owners:
            path.append(current)
            parent = blocks[current].get("parent")
            if parent is None:
                owners[current] = (
                    current if blocks[current].get("opcode") == "procedures_definition" else None
                )
                break
            current = parent
        owner = owners[current] if current is not None else None
        for visited in path:
            owners[visited] = owner
    return owners
