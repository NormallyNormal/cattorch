"""Schema-aware remapping of Scratch block and data identifiers."""

from __future__ import annotations

import copy
import json


def _remap_embedded_reporter(value, mapping: dict[str, str]):
    """Remap the ID in an embedded variable/list/broadcast reporter tuple."""
    if not isinstance(value, list):
        return value
    result = copy.deepcopy(value)
    # Scratch primitive tuples 11, 12 and 13 store their typed data ID last:
    # [11, broadcast name, id], [12, variable name, id], [13, list name, id].
    if len(result) >= 3 and result[0] in {11, 12, 13}:
        if isinstance(result[2], str):
            result[2] = mapping.get(result[2], result[2])
    return result


def _remap_input(value, mapping: dict[str, str]):
    """Remap block/data references in one Scratch input specification."""
    result = copy.deepcopy(value)
    if not isinstance(result, list):
        return result
    # Input slots are [kind, primary] or [kind, primary, shadow]. A string in
    # either position is a block ID. Embedded primitive tuples may contain a
    # typed variable/list/broadcast ID, while their literal text must remain
    # untouched.
    for index in range(1, min(len(result), 3)):
        item = result[index]
        if isinstance(item, str):
            result[index] = mapping.get(item, item)
        elif isinstance(item, list):
            result[index] = _remap_embedded_reporter(item, mapping)
    return result


def _remap_blocks(blocks: dict, mapping: dict[str, str]) -> dict:
    remapped = {}
    for identifier, original in blocks.items():
        block = copy.deepcopy(original)
        for key in ("next", "parent", "comment"):
            reference = block.get(key)
            if isinstance(reference, str):
                block[key] = mapping.get(reference, reference)
        block["inputs"] = {
            name: _remap_input(value, mapping)
            for name, value in block.get("inputs", {}).items()
        }
        for field in block.get("fields", {}).values():
            if (
                isinstance(field, list)
                and len(field) >= 2
                and isinstance(field[1], str)
            ):
                field[1] = mapping.get(field[1], field[1])

        # Procedure argument IDs are JSON-encoded in mutations. They are not
        # normally sprite section keys, but remap any explicitly requested IDs
        # without touching procedure names or other literal mutation strings.
        mutation = block.get("mutation")
        if isinstance(mutation, dict) and isinstance(mutation.get("argumentids"), str):
            try:
                argument_ids = json.loads(mutation["argumentids"])
            except (TypeError, json.JSONDecodeError):
                argument_ids = None
            if isinstance(argument_ids, list):
                mutation["argumentids"] = json.dumps(
                    [mapping.get(item, item) for item in argument_ids],
                    separators=(",", ":"),
                )
        remapped[mapping.get(identifier, identifier)] = block
    return remapped


def remap_ids(data: dict, mapping: dict[str, str]) -> dict:
    """Return ``data`` with Scratch IDs remapped without changing literals.

    ``data`` may be a complete sprite dictionary or a block dictionary. Only
    schema positions which actually contain IDs are rewritten. In particular,
    variable values and list contents are deliberately preserved; treating all
    equal JSON strings as references corrupts token vocabularies when a block
    happens to have an ID such as ``"a"``.
    """
    mapping = {key: value for key, value in mapping.items() if key != value}
    if not mapping:
        return data

    if "blocks" not in data:
        return _remap_blocks(data, mapping)

    result = copy.deepcopy(data)
    result["blocks"] = _remap_blocks(result.get("blocks", {}), mapping)
    for section in ("variables", "lists", "broadcasts"):
        result[section] = {
            mapping.get(identifier, identifier): value
            for identifier, value in result.get(section, {}).items()
        }
    comments = {}
    for identifier, original in result.get("comments", {}).items():
        comment = copy.deepcopy(original)
        block_id = comment.get("blockId")
        if isinstance(block_id, str):
            comment["blockId"] = mapping.get(block_id, block_id)
        comments[mapping.get(identifier, identifier)] = comment
    result["comments"] = comments
    return result


__all__ = ["remap_ids"]
