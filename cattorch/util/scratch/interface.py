"""Logical interface lists and ID-preserving Scratch symbol renaming."""

from __future__ import annotations

import re

from cattorch.errors import UnsupportedModelError


def shard_name(name: str, position: int) -> str:
    return name if position == 0 else f"{name} shard {position + 1}"


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


def refresh_data_display_name(blocks: dict, identifier: str, name: str) -> None:
    stack = [blocks]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            for index in range(1, len(value)):
                if value[index] == identifier and isinstance(value[index - 1], str):
                    value[index - 1] = name
            stack.extend(value)


def rename_variable(sprite: dict, old: str, new: str) -> None:
    for identifier, entry in sprite.get("variables", {}).items():
        if entry[0] == old:
            entry[0] = new
            refresh_data_display_name(sprite.get("blocks", {}), identifier, new)


def rename_list(sprite: dict, old: str, new: str) -> None:
    """Rename a logical list, all its physical shards, and size metadata."""
    pattern = re.compile(re.escape(old) + r"( shard (?:[2-9][0-9]*|1[0-9]+))?$")
    for identifier, entry in sprite.get("lists", {}).items():
        match = pattern.fullmatch(entry[0])
        if match:
            entry[0] = new + (match.group(1) or "")
            refresh_data_display_name(sprite.get("blocks", {}), identifier, entry[0])
    for suffix in ("shard count", "logical length"):
        rename_variable(
            sprite,
            f"cattorch {old.removeprefix('cattorch ')} {suffix}",
            f"cattorch {new.removeprefix('cattorch ')} {suffix}",
        )


def rename_procedure(sprite: dict, old: str, new: str) -> None:
    for block in sprite.get("blocks", {}).values():
        mutation = block.get("mutation")
        if isinstance(mutation, dict) and mutation.get("proccode") == old:
            mutation["proccode"] = new
