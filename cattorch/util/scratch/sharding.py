"""Lower logical Scratch tensor lists to capped physical list shards."""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass


SCRATCH_LIST_LIMIT = 200_000


@dataclass(frozen=True)
class ListShard:
    name: str
    identifier: str
    capacity: int


@dataclass(frozen=True)
class ListLayout:
    name: str
    logical_size: int
    shards: tuple[ListShard, ...]


class _BlockSharder:
    def __init__(self, sprite: dict, layouts: dict[str, ListLayout]):
        self.sprite = sprite
        self.blocks = sprite["blocks"]
        self.by_id = {
            layout.shards[0].identifier: layout
            for layout in layouts.values() if len(layout.shards) > 1
        }
        self.counter = itertools.count(1)

    @staticmethod
    def _offsets(layout: ListLayout):
        offset = 0
        for shard in layout.shards:
            yield offset
            offset += shard.capacity

    def new_id(self, label: str) -> str:
        return f"cattorch_shard_{label}_{next(self.counter)}"

    @staticmethod
    def _literal(value):
        return [1, [4, value]]

    @staticmethod
    def _reporter(identifier):
        return [3, identifier, [4, 0]]

    def _set_parent(self, spec, parent):
        if (
            isinstance(spec, list) and len(spec) > 1 and spec[0] == 3
            and isinstance(spec[1], str) and spec[1] in self.blocks
        ):
            self.blocks[spec[1]]["parent"] = parent

    def _clone_input(self, spec, parent):
        result = copy.deepcopy(spec)
        if (
            isinstance(spec, list) and len(spec) > 1 and spec[0] == 3
            and isinstance(spec[1], str) and spec[1] in self.blocks
        ):
            result[1] = self._clone_reporter(spec[1], parent)
        return result

    def _clone_reporter(self, identifier, parent):
        source = self.blocks[identifier]
        clone_id = self.new_id("reporter")
        clone = copy.deepcopy(source)
        clone["parent"] = parent
        clone["next"] = None
        self.blocks[clone_id] = clone
        for name, spec in list(clone.get("inputs", {}).items()):
            clone["inputs"][name] = self._clone_input(spec, clone_id)
        return clone_id

    def _shift_index(self, spec, offset, parent, *, clone):
        value = self._clone_input(spec, parent) if clone else spec
        if offset == 0:
            self._set_parent(value, parent)
            return value
        subtract_id = self.new_id("index")
        self.blocks[subtract_id] = {
            "opcode": "operator_subtract", "next": None, "parent": parent,
            "inputs": {"NUM1": value, "NUM2": self._literal(offset)},
            "fields": {}, "shadow": False, "topLevel": False,
        }
        self._set_parent(value, subtract_id)
        return self._reporter(subtract_id)

    def _read_reporter(self, shard, index, parent, offset, *, clone):
        identifier = self.new_id("read")
        self.blocks[identifier] = {
            "opcode": "data_itemoflist", "next": None, "parent": parent,
            "inputs": {
                "INDEX": self._shift_index(index, offset, identifier, clone=clone),
            },
            "fields": {"LIST": [shard.name, shard.identifier]},
            "shadow": False, "topLevel": False,
        }
        return identifier

    def _length_reporter(self, shard, parent):
        identifier = self.new_id("length")
        self.blocks[identifier] = {
            "opcode": "data_lengthoflist", "next": None, "parent": parent,
            "inputs": {}, "fields": {"LIST": [shard.name, shard.identifier]},
            "shadow": False, "topLevel": False,
        }
        return identifier

    def _sum_reporters(self, root_id, reporter_ids):
        def build(identifier, values):
            left = values[0]
            if len(values) == 2:
                right = values[1]
            else:
                right = self.new_id("sum")
                build(right, values[1:])
            self.blocks[identifier] = {
                "opcode": "operator_add", "next": None,
                "parent": self.blocks[left]["parent"],
                "inputs": {
                    "NUM1": self._reporter(left),
                    "NUM2": self._reporter(right),
                },
                "fields": {}, "shadow": False, "topLevel": False,
            }
            self.blocks[left]["parent"] = identifier
            self.blocks[right]["parent"] = identifier

        original_parent = self.blocks[root_id].get("parent")
        build(root_id, reporter_ids)
        self.blocks[root_id]["parent"] = original_parent

    def _rewrite_read(self, identifier, block, layout):
        index = block["inputs"]["INDEX"]
        reads = [
            self._read_reporter(
                shard, index, identifier, offset,
                clone=position > 0,
            )
            for position, (shard, offset) in enumerate(
                zip(layout.shards, self._offsets(layout))
            )
        ]
        self._sum_reporters(identifier, reads)

    def _rewrite_length(self, identifier, block, layout):
        lengths = [self._length_reporter(shard, identifier) for shard in layout.shards]
        self._sum_reporters(identifier, lengths)

    def _chain_commands(self, identifier, blocks):
        original = self.blocks[identifier]
        old_next = original.get("next")
        old_parent = original.get("parent")
        previous_id = None
        for index, replacement in enumerate(blocks):
            current_id = identifier if index == 0 else self.new_id("command")
            replacement["parent"] = old_parent if index == 0 else previous_id
            replacement["next"] = None
            self.blocks[current_id] = replacement
            if index:
                self.blocks[previous_id]["next"] = current_id
            previous_id = current_id
        assert previous_id is not None
        self.blocks[previous_id]["next"] = old_next
        if old_next in self.blocks:
            self.blocks[old_next]["parent"] = previous_id

    def _rewrite_clear(self, identifier, block, layout):
        self._chain_commands(identifier, [
            {
                "opcode": "data_deletealloflist", "inputs": {},
                "fields": {"LIST": [shard.name, shard.identifier]},
                "shadow": False, "topLevel": False,
            }
            for shard in layout.shards
        ])

    def _rewrite_replace(self, identifier, block, layout):
        index = block["inputs"]["INDEX"]
        item = block["inputs"]["ITEM"]
        replacements = []
        for position, (shard, offset) in enumerate(
            zip(layout.shards, self._offsets(layout))
        ):
            command_id = identifier if position == 0 else "unused"
            replacements.append({
                "opcode": "data_replaceitemoflist",
                "inputs": {
                    "INDEX": self._shift_index(
                        index, offset, command_id,
                        clone=position > 0,
                    ),
                    "ITEM": item if position == 0 else self._clone_input(item, command_id),
                },
                "fields": {"LIST": [shard.name, shard.identifier]},
                "shadow": False, "topLevel": False,
            })
        self._chain_commands(identifier, replacements)
        # Repair reporter parents after command IDs have been assigned.
        current = identifier
        for _ in replacements:
            for spec in self.blocks[current]["inputs"].values():
                self._set_parent(spec, current)
            current = self.blocks[current].get("next")

    def _append_block(self, shard, item, parent, *, identifier=None):
        identifier = identifier or self.new_id("append")
        self.blocks[identifier] = {
            "opcode": "data_addtolist", "next": None, "parent": parent,
            "inputs": {"ITEM": item},
            "fields": {"LIST": [shard.name, shard.identifier]},
            "shadow": False, "topLevel": False,
        }
        self._set_parent(item, identifier)
        return identifier

    def _rewrite_append(self, identifier, block, layout):
        item = block["inputs"]["ITEM"]
        old_parent = block.get("parent")
        old_next = block.get("next")

        def build(condition_id, position, parent):
            shard = layout.shards[position]
            if position == len(layout.shards) - 1:
                return self._append_block(
                    shard, self._clone_input(item, condition_id), parent,
                    identifier=condition_id,
                )
            length_id = self._length_reporter(shard, condition_id)
            comparison_id = self.new_id("capacity")
            self.blocks[comparison_id] = {
                "opcode": "operator_lt", "next": None, "parent": condition_id,
                "inputs": {
                    "OPERAND1": self._reporter(length_id),
                    "OPERAND2": self._literal(shard.capacity),
                },
                "fields": {}, "shadow": False, "topLevel": False,
            }
            self.blocks[length_id]["parent"] = comparison_id
            then_id = self._append_block(
                shard,
                item if position == 0 else self._clone_input(item, condition_id),
                condition_id,
            )
            else_id = self.new_id("append_route")
            self.blocks[condition_id] = {
                "opcode": "control_if_else", "next": None, "parent": parent,
                "inputs": {
                    "CONDITION": self._reporter(comparison_id),
                    "SUBSTACK": [2, then_id], "SUBSTACK2": [2, else_id],
                },
                "fields": {}, "shadow": False, "topLevel": False,
            }
            self.blocks[comparison_id]["parent"] = condition_id
            build(else_id, position + 1, condition_id)
            return condition_id

        build(identifier, 0, old_parent)
        self.blocks[identifier]["next"] = old_next
        if old_next in self.blocks:
            self.blocks[old_next]["parent"] = identifier

    def apply(self):
        for identifier in list(self.blocks):
            block = self.blocks.get(identifier)
            if block is None:
                continue
            field = block.get("fields", {}).get("LIST")
            if not field or field[1] not in self.by_id:
                continue
            layout = self.by_id[field[1]]
            opcode = block["opcode"]
            if opcode == "data_itemoflist":
                self._rewrite_read(identifier, block, layout)
            elif opcode == "data_lengthoflist":
                self._rewrite_length(identifier, block, layout)
            elif opcode == "data_deletealloflist":
                self._rewrite_clear(identifier, block, layout)
            elif opcode == "data_replaceitemoflist":
                self._rewrite_replace(identifier, block, layout)
            elif opcode == "data_addtolist":
                self._rewrite_append(identifier, block, layout)
            else:
                raise NotImplementedError(
                    f"Sharded tensor list does not support Scratch opcode {opcode}"
                )


def shard_sprite_lists(
    sprite: dict,
    logical_sizes: dict[str, int],
    *,
    limit: int = SCRATCH_LIST_LIMIT,
    alignments: dict[str, int] | None = None,
    limits: dict[str, int] | None = None,
) -> dict[str, ListLayout]:
    """Split oversized logical lists and rewrite every supported list access."""
    if limit != SCRATCH_LIST_LIMIT:
        raise ValueError("Scratch physical shard size is fixed at 200,000")
    layouts = {}
    alignments = alignments or {}
    limits = limits or {}
    lists = sprite.setdefault("lists", {})
    for identifier, entry in list(lists.items()):
        name, contents = entry
        logical_size = max(logical_sizes.get(name, 0), len(contents))
        alignment = alignments.get(name, 1)
        list_limit = limits.get(name, limit)
        if list_limit < 1 or list_limit > limit:
            raise ValueError(f"Invalid physical shard limit {list_limit} for {name}")
        if alignment < 1 or alignment > list_limit:
            raise ValueError(f"Invalid shard alignment {alignment} for {name}")
        shard_limit = list_limit - (list_limit % alignment)
        count = max(1, (logical_size + shard_limit - 1) // shard_limit)
        shards = []
        for position in range(count):
            shard_name = name if position == 0 else f"{name} shard {position + 1}"
            shard_id = identifier if position == 0 else f"{identifier}__shard_{position + 1}"
            start = position * shard_limit
            capacity = min(shard_limit, max(0, logical_size - start))
            if position:
                lists[shard_id] = [
                    shard_name,
                    list(contents[start:start + shard_limit]),
                ]
            else:
                entry[1] = list(contents[:shard_limit])
            shards.append(ListShard(shard_name, shard_id, capacity))
        layouts[name] = ListLayout(name, logical_size, tuple(shards))

    _BlockSharder(sprite, layouts).apply()

    variables = sprite.setdefault("variables", {})
    for name, layout in layouts.items():
        if len(layout.shards) < 2 or name.startswith("_"):
            continue
        safe = "".join(char if char.isalnum() else "_" for char in name)
        label = name.removeprefix("cattorch ")
        variables[f"cattorch_{safe}_shard_count"] = [
            f"cattorch {label} shard count", len(layout.shards),
        ]
        variables[f"cattorch_{safe}_logical_length"] = [
            f"cattorch {label} logical length", layout.logical_size,
        ]
    return layouts
