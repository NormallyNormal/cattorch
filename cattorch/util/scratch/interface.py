"""Logical interface lists and ID-preserving Scratch symbol renaming."""

from __future__ import annotations

import re
from collections.abc import Mapping

from cattorch.errors import UnsupportedModelError
from cattorch.interface import shard_name, shard_variables


def logical_list_names(lists, name: str) -> tuple[str, ...]:
    names: list[str] = []
    while shard_name(name, len(names)) in lists:
        names.append(shard_name(name, len(names)))
    return tuple(names)


def logical_list_chunks(lists, name: str, values: list):
    """Partition public interface values, including empty trailing shards."""
    from cattorch.util.scratch.sharding import SCRATCH_LIST_LIMIT

    names = logical_list_names(lists, name)
    required = max(1, (len(values) + SCRATCH_LIST_LIMIT - 1) // SCRATCH_LIST_LIMIT)
    if len(names) < required:
        missing = shard_name(name, len(names))
        raise UnsupportedModelError(f"generated sprite is missing expected input list {missing!r}")
    return tuple(
        (physical, values[index * SCRATCH_LIST_LIMIT : (index + 1) * SCRATCH_LIST_LIMIT])
        for index, physical in enumerate(names)
    )


def write_logical_list(lists, name: str, values: list) -> None:
    for physical, chunk in logical_list_chunks(lists, name, values):
        lists[physical] = chunk


def read_logical_list(lists, name: str) -> list:
    return [value for physical in logical_list_names(lists, name) for value in lists[physical]]


def write_sprite_list(sprite: dict, name: str, values: list) -> None:
    entries = {entry[0]: entry for entry in sprite.get("lists", {}).values()}
    for physical, chunk in logical_list_chunks(entries, name, values):
        entries[physical][1] = chunk


def refresh_data_display_names(blocks: dict, names: Mapping[str, str]) -> None:
    """Write each ID's name next to it wherever ``blocks`` refers to it.

    ``names`` maps variable, list and broadcast IDs to their names; the
    blocks are walked once however many there are.
    """
    if not names:
        return
    stack = [blocks]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            for index in range(1, len(value)):
                current = value[index]
                if isinstance(current, str) and current in names and isinstance(value[index - 1], str):
                    value[index - 1] = names[current]
            stack.extend(value)


def _rename_variable_entry(sprite: dict, old: str, new: str, names: dict[str, str]) -> None:
    for identifier, entry in sprite.get("variables", {}).items():
        if entry[0] == old:
            entry[0] = names[identifier] = new


def _rename_list_entries(sprite: dict, old: str, new: str, names: dict[str, str]) -> None:
    pattern = re.compile(re.escape(old) + r"( shard (?:[2-9][0-9]*|1[0-9]+))?$")
    for identifier, entry in sprite.get("lists", {}).items():
        match = pattern.fullmatch(entry[0])
        if match:
            entry[0] = names[identifier] = new + (match.group(1) or "")
    for old_variable, new_variable in zip(shard_variables(old), shard_variables(new)):
        _rename_variable_entry(sprite, old_variable, new_variable, names)


def rename_variables(sprite: dict, renames: Mapping[str, str]) -> None:
    """Rename variables in order, then update the blocks in one walk."""
    names: dict[str, str] = {}
    for old, new in renames.items():
        _rename_variable_entry(sprite, old, new, names)
    refresh_data_display_names(sprite.get("blocks", {}), names)


def rename_lists(sprite: dict, renames: Mapping[str, str]) -> None:
    """Rename logical lists, all their physical shards, and size metadata.

    The renames apply in order, as separate calls would, and the blocks are
    updated in one walk.
    """
    names: dict[str, str] = {}
    for old, new in renames.items():
        _rename_list_entries(sprite, old, new, names)
    refresh_data_display_names(sprite.get("blocks", {}), names)


def rename_variable(sprite: dict, old: str, new: str) -> None:
    rename_variables(sprite, {old: new})


def rename_list(sprite: dict, old: str, new: str) -> None:
    rename_lists(sprite, {old: new})


def rename_procedure(sprite: dict, old: str, new: str) -> None:
    for block in sprite.get("blocks", {}).values():
        mutation = block.get("mutation")
        if isinstance(mutation, dict) and mutation.get("proccode") == old:
            mutation["proccode"] = new
