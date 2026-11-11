"""Scratch sprite assembly, lifecycle, storage, and generation wiring."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import uuid
import weakref
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, AbstractSet, Literal, cast

import numpy as np
import torch

from cattorch.graph import _GenerationSpec, _GraphInfo, _get_shape
from cattorch.quantization import effective_group_size
from cattorch.storage import (
    EncodedList,
    StorageConfig,
    build_decoder_programs,
    build_shared_unpack_program,
    group_scales,
    rounded_values,
    _float_bytes,
    _quantize_arrays,
    _quantized_bytes,
)
from cattorch.util.scratch.remap import remap_ids
from cattorch.util.scratch.sharding import SCRATCH_LIST_LIMIT, ListLayout
from cattorch.util.scratch.tensor_adder import StaticValues, static_list_contents
from cattorch.interface import STATUS_VARIABLE

if TYPE_CHECKING:
    from cattorch.transpiler import _Compiler
    from cattorch.util.scratch.dsl import Program

log = logging.getLogger(__name__)


# ── Sprite post-processing ───────────────────────────────────────────────────


def _merge_duplicate_lists(sprite):
    """Merge local lists (names starting with '_') that have identical contents."""
    lists = sprite.get("lists", {})

    def local_base(name: str) -> str:
        for index in range(len(name) - 1, 0, -1):
            if name[index] == '_' and name[index + 1:].isdigit():
                return name[:index]
        return name

    # Only combine repeated instances of the same logical local.  Two
    # differently named empty work lists can be live simultaneously inside a
    # single instruction (notably the MoE router's logits, IDs, and weights),
    # so content alone is not a safe aliasing proof.
    by_content: dict[tuple[str, tuple], list[tuple[str, str]]] = {}
    for sid, entry in lists.items():
        display_name = entry[0]
        if not display_name.startswith("_"):
            continue
        content_key = (local_base(display_name), tuple(entry[1]))
        by_content.setdefault(content_key, []).append((sid, display_name))

    remap = {}
    for group in by_content.values():
        if len(group) < 2:
            continue
        winner_sid = group[0][0]
        for sid, display_name in group[1:]:
            log.info("Merging duplicate list %s (%s) -> %s (%s)",
                     display_name, sid, group[0][1], winner_sid)
            remap[sid] = winner_sid
            del lists[sid]

    if not remap:
        return

    sprite["blocks"] = remap_ids(sprite["blocks"], remap, share_unchanged=True)

    base_counts: dict[str, int] = {}
    for entry in lists.values():
        name = entry[0]
        if not name.startswith("_"):
            continue
        base = name
        for i in range(len(name) - 1, 0, -1):
            if name[i] == '_' and name[i+1:].isdigit():
                base = name[:i]
                break
        count = base_counts.get(base, 0) + 1
        base_counts[base] = count
        entry[0] = base if count == 1 else f"{base}_{count}"


def _deduplicate_static_tensors(
    sprite: dict,
    static_tensors: dict[str, torch.Tensor | None],
    *,
    effective_ndims: dict[str, int] | None = None,
) -> dict[str, torch.Tensor | None]:
    """Share bit-identical flattened tensors before storage and sharding.

    Tensor shape is intentionally not part of the identity: Scratch kernels
    access the physical lists as flat arrays and have already compiled their
    shape-specific index arithmetic. Hash matches are confirmed byte-for-byte
    so this transform remains exact even for NaNs and signed zeroes.
    """
    winners: dict[tuple[str, int, int, bytes], tuple[str, torch.Tensor]] = {}
    aliases: dict[str, str] = {}
    kept: dict[str, torch.Tensor | None] = {}
    # A tensor that no block reads, such as one a fast-mode kernel replaced,
    # can't stand in for the others: its list isn't in the sprite.
    in_sprite = {entry[0] for entry in sprite.get("lists", {}).values()}
    for key, tensor in static_tensors.items():
        if not isinstance(tensor, torch.Tensor) or f"W_{key}" not in in_sprite:
            kept[key] = tensor
            continue
        contiguous = tensor.detach().cpu().contiguous()
        # The bytes in place, not a copy of every weight held until the end.
        raw = contiguous.reshape(-1).view(torch.uint8)
        identity = (
            str(contiguous.dtype), contiguous.numel(),
            (effective_ndims or {}).get(key, tensor.ndim),
            hashlib.sha256(raw.numpy()).digest(),
        )
        previous = winners.get(identity)
        if previous is not None and torch.equal(previous[1], raw):
            aliases[f"W_{key}"] = f"W_{previous[0]}"
            log.info("Sharing identical static tensor W_%s -> W_%s", key, previous[0])
        else:
            winners[identity] = (key, raw)
            kept[key] = tensor
    if not aliases:
        return kept

    lists = sprite.get("lists", {})
    ids_by_name: dict[str, list[str]] = {}
    for identifier, entry in lists.items():
        ids_by_name.setdefault(entry[0], []).append(identifier)
    id_remap: dict[str, str] = {}
    name_by_winner_id: dict[str, str] = {}
    for duplicate_name, winner_name in aliases.items():
        winner_ids = ids_by_name.get(winner_name, [])
        if not winner_ids:
            continue
        winner_id = winner_ids[0]
        name_by_winner_id[winner_id] = winner_name
        for duplicate_id in ids_by_name.get(duplicate_name, []):
            id_remap[duplicate_id] = winner_id
            lists.pop(duplicate_id, None)
    if id_remap:
        sprite["blocks"] = remap_ids(sprite["blocks"], id_remap, share_unchanged=True)

        # Scratch fields redundantly store a display name next to the ID.
        stack = [sprite["blocks"]]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, list):
                if (
                    len(value) == 2
                    and isinstance(value[1], str)
                    and value[1] in name_by_winner_id
                ):
                    value[0] = name_by_winner_id[value[1]]
                stack.extend(value)
    return kept


def _apply_tied_static_aliases(
    sprite: dict,
    layouts: dict[str, ListLayout],
    static_tensors: dict[str, torch.Tensor | None],
    aliases: dict[str, tuple[str, ...]],
) -> dict[str, str]:
    """Point aligned embedding shards at the output head's row shards.

    Returns the tables that keep their own copy, with the reason.
    """
    lists = sprite.get("lists", {})
    remap: dict[str, str] = {}
    display_names: dict[str, str] = {}
    copies: dict[str, str] = {}
    for source_name, destination_names in aliases.items():
        source = layouts.get(source_name)
        if source is None:
            continue  # only the head reads the table
        destinations = [layouts.get(name) for name in destination_names]
        if (
            any(layout is None or len(layout.shards) != 1 for layout in destinations)
            or [shard.capacity for shard in source.shards]
            != [layout.shards[0].capacity for layout in destinations if layout is not None]
        ):
            # The compiler cuts the head's shards to the table's physical
            # lists, so this is not expected. Both copies remain correct.
            copies[source_name] = "its shards do not line up with the output head's"
            log.warning("Tied embedding/head sharing skipped: %s", copies[source_name])
            continue
        for source_shard, destination_layout in zip(source.shards, destinations):
            assert destination_layout is not None
            destination_shard = destination_layout.shards[0]
            if lists[source_shard.identifier][1] != lists[destination_shard.identifier][1]:
                raise ValueError("tied embedding/head shard contents do not match")
            remap[source_shard.identifier] = destination_shard.identifier
            display_names[destination_shard.identifier] = destination_shard.name
            del lists[source_shard.identifier]
        del layouts[source_name]
        static_tensors.pop(source_name.removeprefix("W_"), None)

    if not remap:
        return copies
    sprite["blocks"] = remap_ids(sprite["blocks"], remap, share_unchanged=True)
    stack = [sprite["blocks"]]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            if (
                len(value) == 2
                and isinstance(value[1], str)
                and value[1] in display_names
            ):
                value[0] = display_names[value[1]]
            stack.extend(value)
    return copies


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


_LAYER_PATH = re.compile(r"(?:^|\.)blocks\.(\d+)(?:\.|$)")


def _graph_layer(node) -> int | None:
    for entry in reversed(tuple(node.meta.get("nn_module_stack", {}).values())):
        if not entry or not entry[0]:
            continue
        match = _LAYER_PATH.search(entry[0])
        if match:
            return int(match.group(1))
    return None


def _chain_until(blocks: dict, start: str, stop: str | None) -> list[str]:
    chain = []
    current = start
    while current is not None and current != stop:
        if current not in blocks or current in chain:
            raise ValueError("malformed generated command chain")
        chain.append(current)
        current = blocks[current].get("next")
    if stop is not None and current != stop:
        raise ValueError("repeated layer roots are not consecutive")
    return chain


def _descendants(
    blocks: dict, commands: set[str], all_commands: set[str],
) -> set[str]:
    """Return nested blocks whose nearest command-chain ancestor is owned."""
    owned = set(commands)
    for identifier, block in blocks.items():
        if identifier in all_commands:
            continue
        parent = block.get("parent")
        visited = set()
        while parent is not None and parent not in all_commands and parent not in visited:
            visited.add(parent)
            parent = blocks.get(parent, {}).get("parent")
        if parent in commands:
            owned.add(identifier)
    return owned


def _program_stack(sprite: dict, program) -> tuple[str, str]:
    """Merge a DSL stack by display name and return its root and tail."""
    data = program.compile()
    suffix = uuid.uuid4().hex[:8]
    initial = {
        identifier: f"{identifier}_{suffix}"
        for section in ("blocks", "variables", "lists")
        for identifier in data.get(section, {})
    }
    data = remap_ids(data, initial)
    blocks = data["blocks"]
    root = next(
        identifier for identifier, block in blocks.items()
        if block.get("topLevel") and block.get("parent") is None
    )
    tail = root
    while blocks[tail].get("next") is not None:
        tail = blocks[tail]["next"]

    mapping = {}
    for section in ("variables", "lists"):
        existing = {entry[0]: identifier for identifier, entry in sprite[section].items()}
        for identifier, entry in data[section].items():
            if entry[0] in existing:
                mapping[identifier] = existing[entry[0]]
            else:
                sprite[section][identifier] = entry
    if mapping:
        blocks = remap_ids(blocks, mapping)
    sprite["blocks"].update(blocks)
    return root, tail


def _share_repeated_transformer_layers(
    sprite: dict, compiler, storage: StorageConfig, *,
    layer_tail: tuple[str, str] | None = None,
) -> bool:
    """Factor structurally identical ``blocks.N`` slabs into one warp block.

    Scratch cannot pass lists to custom blocks. The shared slab therefore uses
    the first layer's activation lists, copies only its final hidden state
    between calls, and selects concatenated weight banks through scalar bases.
    ``layer_tail`` is ``(gate, final)``: the gate variable is set to 1 before
    each call, and to the ``final`` variable's value before the last one.
    """
    def reject(reason: str) -> bool:
        log.info("Shared transformer layer skipped: %s", reason)
        return False

    if (
        compiler.graph.generation.first_cache is not None
        and not compiler.graph.generation.banked_caches
    ):
        # Cached K/V lists are statically named in Scratch. Until attention is
        # split into thin per-layer cache wrappers, sharing the whole slab
        # would incorrectly direct every call to layer zero's cache.
        return reject("cached generation requires banked caches")

    nodes = {node.name: node for node in compiler.graph.nodes}
    emitted = [
        (name, root, _graph_layer(nodes[name]))
        for name, root in compiler.emitted_roots
        if name in nodes
    ]
    layers = sorted({layer for _, _, layer in emitted if layer is not None})
    if len(layers) < 2 or layers != list(range(len(layers))):
        return reject("layers are not a consecutive repeated stack")
    grouped = [
        [(name, root) for name, root, found in emitted if found == layer]
        for layer in layers
    ]
    def literals(value):
        """An argument with tensors blanked out: what the code bakes in."""
        if isinstance(value, torch.fx.Node):
            return None
        if isinstance(value, (list, tuple)):
            return tuple(literals(item) for item in value)
        if isinstance(value, dict):
            return tuple(sorted((key, literals(item)) for key, item in value.items()))
        if isinstance(value, torch.Tensor):
            return tuple(value.shape), str(value.dtype)
        return repr(value)

    # Constants such as an attention window or a scale are compiled into each
    # layer's code, so layers that differ in them can't share it.
    # Fused nodes have no root of their own, so compare every node.
    signatures = [
        [
            (
                str(node.target), literals(node.meta.get("val")),
                literals(node.args), literals(dict(node.kwargs)),
            )
            for node in compiler.graph.nodes
            if node.op == "call_function" and _graph_layer(node) == layer
        ]
        for layer in layers
    ]
    if any(signature != signatures[0] for signature in signatures[1:]):
        return reject("emitted operation signatures differ")

    blocks = sprite["blocks"]
    starts = [group[0][1] for group in grouped]
    emitted_roots = [root for _, root in compiler.emitted_roots]
    last_position = max(emitted_roots.index(root) for _, root in grouped[-1])
    successor = emitted_roots[last_position + 1] if last_position + 1 < len(emitted_roots) else None
    command_groups = [
        _chain_until(blocks, start, starts[index + 1] if index + 1 < len(starts) else successor)
        for index, start in enumerate(starts)
    ]
    main_root = next(
        identifier for identifier, block in blocks.items()
        if block.get("topLevel") and block.get("parent") is None
    )
    all_commands = set(_chain_until(blocks, main_root, None))
    owned_groups = [
        _descendants(blocks, set(commands), all_commands)
        for commands in command_groups
    ]

    static_names = {f"W_{name}" for name in compiler.static_lists}
    list_ids = {entry[0]: identifier for identifier, entry in sprite["lists"].items()}
    role_names = []
    for owned in owned_groups:
        found = []
        for identifier, block in blocks.items():
            if identifier not in owned:
                continue
            field = block.get("fields", {}).get("LIST")
            if field and field[0] in static_names and field[0] not in found:
                found.append(field[0])
        role_names.append(found)
    if not role_names[0] or any(len(names) != len(role_names[0]) for names in role_names):
        log.info("Repeated layer static roles: %s", role_names)
        return reject("static role counts differ")

    bank_roles = []
    for role, names in enumerate(zip(*role_names)):
        if len(set(names)) == 1:
            continue
        entries = [sprite["lists"][list_ids[name]] for name in names]
        sizes = [len(entry[1]) for entry in entries]
        if len(set(sizes)) != 1:
            return reject("a candidate weight bank is misaligned")
        # Logical layer banks may exceed one physical Scratch list. The
        # ordinary precision-aware sharder runs after this transform and
        # rewrites indexed reads across row-aligned physical shards.
        bank_roles.append((role, names, sizes[0]))
    if not bank_roles:
        return reject("no layer-varying static weights were found")
    banked = [name for _, names, _ in bank_roles for name in names]
    if len(set(banked)) != len(banked):
        # A deduplicated tensor read by only some layers, or in a different
        # role per layer, can't become one row of a bank.
        return reject("a static tensor would fill more than one bank slot")
    log.info("Repeated layer bank roles: %s", bank_roles)

    layer_zero_owned = owned_groups[0]

    # Finish every fallible structural check before changing the sprite or the
    # compiler's static tensors.  A rejected candidate must be a true no-op:
    # leaving a half-built bank behind makes later layers silently read layer
    # zero's weights even though no shared procedure was installed.
    for _, names, _ in bank_roles:
        winner_id = list_ids[names[0]]
        for identifier in layer_zero_owned:
            block = blocks.get(identifier)
            field = block.get("fields", {}).get("LIST") if block else None
            if field and field[1] == winner_id and block["opcode"] != "data_itemoflist":
                return reject("a banked tensor uses a non-read list operation")

    layer_names = [
        {node.name for node in compiler.graph.nodes if _graph_layer(node) == layer}
        for layer in layers
    ]
    output_nodes = []
    for index, members in enumerate(layer_names):
        consumers = layer_names[index + 1] if index + 1 < len(layer_names) else {
            node.name for node in compiler.graph.nodes if _graph_layer(node) is None
        }
        used = {
            arg.name
            for consumer in compiler.graph.nodes if consumer.name in consumers
            for arg in consumer.all_input_nodes
            if arg.name in members and arg.name in compiler.scope.assignments
        }
        if not used:
            return reject("could not identify a carried layer output")
        output_nodes.append(max(used, key=lambda name: next(
            i for i, node in enumerate(compiler.graph.nodes) if node.name == name
        )))
    output_lists = [f"T{compiler.scope.assignments[name]}" for name in output_nodes]
    first_inputs = [
        arg.name
        for node in compiler.graph.nodes if node.name in layer_names[0]
        for arg in node.all_input_nodes
        if arg.name not in layer_names[0]
        and (
            arg.name in compiler.scope.assignments
            or arg.name in compiler.graph.input_names
        )
    ]
    if not first_inputs:
        return reject("could not identify the first layer input")
    first_input = first_inputs[-1]
    input_list = (
        f"T{compiler.scope.assignments[first_input]}"
        if first_input in compiler.scope.assignments
        else compiler.graph.input_names[first_input]
    )
    output_list = output_lists[0]
    predecessor = next(
        (identifier for identifier, block in blocks.items() if block.get("next") == starts[0]),
        None,
    )
    if predecessor is None:
        return reject("the repeated layer has no chain predecessor")

    base_variables = []
    for role, names, size in bank_roles:
        winner_id = list_ids[names[0]]
        winner = sprite["lists"][winner_id]
        parts = [sprite["lists"][list_ids[name]][1] for name in names]
        if all(isinstance(part, StaticValues) for part in parts):
            winner[1] = StaticValues(np.concatenate([part.array for part in parts]))
        else:
            winner[1] = [value for part in parts for value in part]
        for name in names[1:]:
            sprite["lists"].pop(list_ids[name], None)

        tensors = [compiler.static_lists.pop(name.removeprefix("W_")) for name in names]
        combined_layouts = []
        layout_offset = 0
        shard_alignments = []
        for name, tensor in zip(names, tensors):
            key = name.removeprefix("W_")
            shard_alignments.append(compiler.static_shard_alignments.pop(name, 1))
            for start, rows, inner in compiler.interleaved_static_layouts.pop(key, ()):
                combined_layouts.append((layout_offset + start, rows, inner))
            layout_offset += tensor.numel()
        all_matrix = all(isinstance(tensor, torch.Tensor) and tensor.ndim >= 2 for tensor in tensors)
        bank_tensor = (
            torch.tensor(np.asarray(winner[1], dtype=np.float64), dtype=torch.float32)
            if isinstance(winner[1], StaticValues)
            else torch.tensor(winner[1], dtype=torch.float32)
        )
        if all_matrix and size >= storage.min_quantized_values:
            matrix_shapes = {tuple(tensor.shape) for tensor in tensors}
            if len(matrix_shapes) == 1:
                bank_tensor = bank_tensor.reshape(len(names), *next(iter(matrix_shapes)))
            else:
                bank_tensor = bank_tensor.reshape(len(names), -1)
        compiler.static_lists[names[0].removeprefix("W_")] = bank_tensor
        sources = [compiler.static_source_numels.pop(name.removeprefix("W_"), 0) for name in names]
        if any(sources):
            # Row shards of weights: kept like those weights.
            compiler.static_source_numels[names[0].removeprefix("W_")] = max(
                bank_tensor.numel(), *sources,
            )
        bank_alignment = math.lcm(
            *shard_alignments,
            *(4 * inner for _, _, inner in combined_layouts),
        )
        if bank_alignment > 1:
            compiler.static_shard_alignments[names[0]] = bank_alignment
        if combined_layouts:
            compiler.interleaved_static_layouts[
                names[0].removeprefix("W_")
            ] = tuple(combined_layouts)

        base_name = f"cattorch layer weight base {role}"
        base_id = f"cattorch_layer_weight_base_{role}_{uuid.uuid4().hex[:8]}"
        sprite["variables"][base_id] = [base_name, 0]
        base_variables.append((base_name, size))
        for identifier in list(layer_zero_owned):
            block = blocks.get(identifier)
            field = block.get("fields", {}).get("LIST") if block else None
            if not field or field[1] != winner_id:
                continue
            field[0] = winner[0]
            add_id = f"cattorch_layer_bank_index_{uuid.uuid4().hex}"
            old_index = block["inputs"]["INDEX"]
            blocks[add_id] = {
                "opcode": "operator_add", "next": None, "parent": identifier,
                "inputs": {
                    "NUM1": old_index,
                    "NUM2": [3, [12, base_name, base_id], [10, ""]],
                },
                "fields": {}, "shadow": False, "topLevel": False,
            }
            if (
                isinstance(old_index, list) and len(old_index) > 1
                and old_index[0] == 3 and isinstance(old_index[1], str)
                and old_index[1] in blocks
            ):
                blocks[old_index[1]]["parent"] = add_id
            block["inputs"]["INDEX"] = [3, add_id, [4, 0]]

    log.info(
        "Repeated layer activation path: input=%s outputs=%s",
        input_list, output_lists,
    )

    first_end = command_groups[0][-1]
    blocks[first_end]["next"] = None
    procedure_name = "cattorch shared transformer layer"
    definition = f"cattorch_shared_layer_definition_{uuid.uuid4().hex[:8]}"
    prototype = f"cattorch_shared_layer_prototype_{uuid.uuid4().hex[:8]}"
    blocks[definition] = {
        "opcode": "procedures_definition", "next": starts[0], "parent": None,
        "inputs": {"custom_block": [1, prototype]}, "fields": {},
        "shadow": False, "topLevel": True, "x": 640, "y": 0,
    }
    blocks[prototype] = {
        "opcode": "procedures_prototype", "next": None, "parent": definition,
        "inputs": {}, "fields": {}, "shadow": True, "topLevel": False,
        "mutation": _procedure_mutation(procedure_name, warp=True),
    }
    blocks[starts[0]]["parent"] = definition
    blocks[starts[0]]["topLevel"] = False
    blocks[starts[0]].pop("x", None)
    blocks[starts[0]].pop("y", None)

    for owned in owned_groups[1:]:
        for identifier in owned:
            blocks.pop(identifier, None)

    from cattorch.util.scratch.dsl import Program, append, call, clear, for_each, item, length, set_var, var
    body: list = []
    copy_index = "cattorch shared layer copy index"
    cache_base = (
        "cattorch layer cache base"
        if compiler.graph.generation.banked_caches else None
    )
    for layer in layers:
        body.extend(set_var(name, layer * size) for name, size in base_variables)
        if layer_tail is not None:
            gate, final = layer_tail
            body.append(set_var(gate, var(final) if layer == layers[-1] else 1))
        if cache_base is not None:
            body.append(set_var(
                cache_base,
                layer * compiler.graph.generation.cache_layer_width,
            ))
        body.append(call(procedure_name))
        if layer != layers[-1] and input_list != output_list:
            body.extend((
                clear(input_list),
                for_each(copy_index, length(output_list), (
                    append(input_list, item(output_list, var(copy_index))),
                )),
            ))
    final_output_list = output_lists[-1]
    if final_output_list != output_list:
        body.extend((
            clear(final_output_list),
            for_each(copy_index, length(output_list), (
                append(final_output_list, item(output_list, var(copy_index))),
            )),
        ))
    wrapper = Program(
        "shared_layer_calls",
        variables=(
            copy_index,
            *(name for name, _ in base_variables),
            *((cache_base,) if cache_base is not None else ()),
            *(layer_tail or ()),
        ),
        lists=tuple(dict.fromkeys((input_list, output_list, final_output_list))),
        body=tuple(body),
    )
    wrapper_root, wrapper_tail = _program_stack(sprite, wrapper)
    # `_program_stack` may replace its temporary block dictionary while
    # remapping data IDs; refresh the shared reference before splicing.
    blocks = sprite["blocks"]
    blocks[predecessor]["next"] = wrapper_root
    blocks[wrapper_root]["parent"] = predecessor
    blocks[wrapper_root]["topLevel"] = False
    blocks[wrapper_root].pop("x", None)
    blocks[wrapper_root].pop("y", None)
    blocks[wrapper_tail]["next"] = successor
    if successor in blocks:
        blocks[successor]["parent"] = wrapper_tail

    log.info(
        "Shared wrapper splice root=%s parent=%s top=%s predecessor_next=%s",
        wrapper_root,
        sprite["blocks"][wrapper_root].get("parent"),
        sprite["blocks"][wrapper_root].get("topLevel"),
        sprite["blocks"][predecessor].get("next"),
    )
    if cache_base is not None:
        _merge_variables_by_name(sprite, {cache_base})
    if layer_tail is not None:
        _merge_variables_by_name(sprite, set(layer_tail))
    return True


def _merge_variables_by_name(
    sprite, names: set[str], *, added: tuple[frozenset[str], frozenset[str]] | None = None,
) -> None:
    """Unify repeated DSL declarations of persistent generation variables."""
    _merge_slots_by_name(sprite, "variables", names, added)


def _remove_unused(sprite):
    """Remove unreachable blocks and unreferenced data slots."""
    blocks = sprite.get("blocks", {})
    reachable: set[str] = set()
    pending = [
        identifier
        for identifier, block in blocks.items()
        if block.get("topLevel") and block.get("parent") is None
    ]
    while pending:
        identifier = pending.pop()
        if identifier in reachable or identifier not in blocks:
            continue
        reachable.add(identifier)
        stack = [blocks[identifier]]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, (list, tuple)):
                stack.extend(value)
            elif isinstance(value, str) and value in blocks and value not in reachable:
                pending.append(value)
    for identifier in set(blocks) - reachable:
        log.info("Removing unreachable block: %s", identifier)
        del blocks[identifier]

    referenced: set[str] = set()
    stack = [blocks]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        elif isinstance(value, str):
            referenced.add(value)

    from cattorch.interface import is_public_data

    for section in ("lists", "variables"):
        slots = sprite.get(section, {})
        # Public names stay even when no block reads them, such as the input
        # of a model that only uses its shape.
        unused = [
            sid for sid, entry in slots.items()
            if sid not in referenced
            and not entry[0].startswith("cattorch ") and not is_public_data(entry[0])
        ]
        for sid in unused:
            log.info("Removing unused %s: %s", section.rstrip("s"), slots[sid][0])
            del slots[sid]


def _procedure_mutation(
    name: str, *, warp: bool = False, arguments: Sequence[str] = (),
) -> dict:
    from cattorch.util.scratch.dsl import argument_id

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


def _wrap_optimized_lifecycle(sprite: dict) -> None:
    """Put the compiled stack behind idempotent init and warp forward blocks."""
    blocks = sprite["blocks"]
    roots = [
        bid for bid, block in blocks.items()
        if block.get("topLevel") and block.get("parent") is None
        and block.get("opcode") != "procedures_definition"
    ]
    if len(roots) != 1:
        raise ValueError(f"expected one compiled root before lifecycle wrapping, got {roots}")

    suffix = uuid.uuid4().hex[:8]
    prefix = f"cattorch_lifecycle_{suffix}"
    initialized_id = f"{prefix}_initialized"
    sprite.setdefault("variables", {})[initialized_id] = ["cattorch initialized", 0]

    old_root = roots[0]
    blocks[old_root]["topLevel"] = False
    blocks[old_root].pop("x", None)
    blocks[old_root].pop("y", None)

    forward_name = "cattorch forward"
    init_name = "cattorch init"
    forward_call = f"{prefix}_forward_call"
    init_call = f"{prefix}_init_call"
    forward_def = f"{prefix}_forward_definition"
    forward_proto = f"{prefix}_forward_prototype"
    init_def = f"{prefix}_init_definition"
    init_proto = f"{prefix}_init_prototype"
    init_check = f"{prefix}_init_check"
    init_check_expr = f"{prefix}_init_check_expr"
    nested_init_call = f"{prefix}_nested_init_call"
    set_initialized = f"{prefix}_set_initialized"

    for call_id, procedure, y in ((forward_call, forward_name, 0), (init_call, init_name, 80)):
        blocks[call_id] = {
            "next": None, "parent": None, "inputs": {}, "fields": {},
            "shadow": False, "topLevel": True,
            "opcode": "procedures_call",
            "mutation": _procedure_mutation(procedure),
            "x": 0, "y": y,
        }

    blocks[forward_def] = {
        "opcode": "procedures_definition", "next": init_check, "parent": None,
        "inputs": {"custom_block": [1, forward_proto]}, "fields": {},
        "shadow": False, "topLevel": True, "x": 320, "y": 0,
    }
    blocks[forward_proto] = {
        "opcode": "procedures_prototype", "next": None, "parent": forward_def,
        "inputs": {}, "fields": {}, "shadow": True, "topLevel": False,
        "mutation": _procedure_mutation(forward_name, warp=True),
    }
    blocks[init_def] = {
        "opcode": "procedures_definition", "next": set_initialized, "parent": None,
        "inputs": {"custom_block": [1, init_proto]}, "fields": {},
        "shadow": False, "topLevel": True, "x": 320, "y": 160,
    }
    blocks[init_proto] = {
        "opcode": "procedures_prototype", "next": None, "parent": init_def,
        "inputs": {}, "fields": {}, "shadow": True, "topLevel": False,
        "mutation": _procedure_mutation(init_name, warp=True),
    }

    blocks[init_check] = {
        "opcode": "control_if", "next": old_root, "parent": forward_def,
        "inputs": {
            "CONDITION": [2, init_check_expr],
            "SUBSTACK": [2, nested_init_call],
        },
        "fields": {}, "shadow": False, "topLevel": False,
    }
    blocks[init_check_expr] = {
        "opcode": "operator_equals", "next": None, "parent": init_check,
        "inputs": {
            "OPERAND1": [3, [12, "cattorch initialized", initialized_id], [10, ""]],
            "OPERAND2": [1, [4, 0]],
        },
        "fields": {}, "shadow": False, "topLevel": False,
    }
    blocks[nested_init_call] = {
        "opcode": "procedures_call", "next": None, "parent": init_check,
        "inputs": {}, "fields": {}, "shadow": False, "topLevel": False,
        "mutation": _procedure_mutation(init_name),
    }
    blocks[set_initialized] = {
        "opcode": "data_setvariableto", "next": None, "parent": init_def,
        "inputs": {"VALUE": [1, [4, 1]]},
        "fields": {"VARIABLE": ["cattorch initialized", initialized_id]},
        "shadow": False, "topLevel": False,
    }
    blocks[old_root]["parent"] = init_check


def _add_warp_procedure(
    sprite: dict, name: str, program, *, x: int, y: int, arguments: Sequence[str] = (),
) -> dict:
    """Combine a DSL command stack and expose it as a warp block.

    ``name`` is the proccode, with a ``%s`` for each of ``arguments``.
    """
    from cattorch.util.scratch.dsl import argument_id

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
    raise ValueError(f"procedure not found after generation wrapping: {name}")


def _remove_canvas_calls(sprite: dict, name: str) -> None:
    """Remove the loose call blocks for ``name`` that lifecycle wrapping left on the canvas."""
    blocks = sprite["blocks"]
    for identifier in [
        identifier for identifier, block in blocks.items()
        if block.get("opcode") == "procedures_call" and block.get("topLevel")
        and block.get("mutation", {}).get("proccode") == name
    ]:
        del blocks[identifier]


def _add_procedure_completion_broadcast(
    sprite: dict,
    procedure: str,
    message: str,
    *,
    status: str | None = None,
) -> None:
    """Broadcast a public lifecycle notification after a procedure finishes.

    ``broadcast and wait`` is intentional: lifecycle procedures are warp
    blocks, so an ordinary broadcast would not let receiver scripts run at the
    boundary before the caller continued into the next expensive phase.
    """
    broadcasts = sprite.setdefault("broadcasts", {})
    broadcast_id = next(
        (
            identifier
            for identifier, existing_message in broadcasts.items()
            if existing_message == message
        ),
        None,
    )
    if broadcast_id is None:
        broadcast_id = f"cattorch_broadcast_{uuid.uuid4().hex[:8]}"
        broadcasts[broadcast_id] = message

    blocks = sprite["blocks"]
    definition = _find_procedure_definition(sprite, procedure)
    terminal = blocks[definition].get("next")
    if terminal is None:
        raise ValueError(f"procedure {procedure!r} has no executable body")
    while blocks[terminal].get("next") is not None:
        terminal = blocks[terminal]["next"]

    notification = f"cattorch_completion_{uuid.uuid4().hex[:8]}"
    if status is not None:
        status_id = next(
            identifier
            for identifier, entry in sprite.get("variables", {}).items()
            if entry[0] == STATUS_VARIABLE
        )
        status_block = f"cattorch_completion_status_{uuid.uuid4().hex[:8]}"
        blocks[terminal]["next"] = status_block
        blocks[status_block] = {
            "opcode": "data_setvariableto", "next": notification,
            "parent": terminal,
            "inputs": {"VALUE": [1, [10, status]]},
            "fields": {"VARIABLE": [STATUS_VARIABLE, status_id]},
            "shadow": False, "topLevel": False,
        }
        notification_parent = status_block
    else:
        blocks[terminal]["next"] = notification
        notification_parent = terminal
    blocks[notification] = {
        "opcode": "event_broadcastandwait",
        "next": None,
        "parent": notification_parent,
        "inputs": {
            "BROADCAST_INPUT": [1, [11, message, broadcast_id]],
        },
        "fields": {},
        "shadow": False,
        "topLevel": False,
    }


def _remove_completion_broadcast(sprite: dict, message: str) -> None:
    """Undo ``_add_procedure_completion_broadcast`` for ``message``.

    The notification, and the status it sets, are the last blocks of their
    procedure.
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


def _logical_list_sizes(
    compiler: _Compiler,
    graph: _GraphInfo,
    generation: _GenerationSpec | None,
    output_name: str | None,
) -> dict[str, int]:
    """Infer peak logical capacities for every tensor list before sharding."""
    sizes: dict[str, int] = {}

    def record(name: str, shape) -> None:
        try:
            size = math.prod(shape)
        except TypeError:
            return
        sizes[name] = max(sizes.get(name, 0), int(size))

    for node in graph.nodes:
        if node.name in compiler.scope.assignments:
            record(f"T{compiler.scope.assignments[node.name]}", _get_shape(node))
        if node.name in graph.input_names:
            record(graph.input_names[node.name], _get_shape(node))

    for key, tensor in compiler.static_lists.items():
        if isinstance(tensor, torch.Tensor):
            sizes[f"W_{key}"] = tensor.numel()

    if output_name and output_name in sizes:
        sizes["output"] = sizes.pop(output_name)

    if generation is not None:
        for name, cache_width in graph.generation.cache_widths.items():
            sizes[name] = generation.max_context * cache_width
        sizes["cattorch prefill buffer"] = generation.max_context
        if generation.top_k is not None:
            sizes[_TOP_K_VALUES] = generation.top_k
            sizes[_TOP_K_IDS] = generation.top_k
    for name, size in compiler.logical_sizes.items():
        sizes[name] = max(sizes.get(name, 0), size)
    return sizes


def _insert_init_call(sprite: dict, procedure: str) -> None:
    blocks = sprite["blocks"]
    init_definition = _find_procedure_definition(sprite, "cattorch init")
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


_StoragePrecision = Literal["float32", "float16", "int8", "int6", "int4"]


def _static_storage_precision(
    tensor: torch.Tensor | None,
    storage: StorageConfig,
    effective_ndim: int | None = None,
    interleave_segments: tuple[tuple[int, int, int], ...] = (),
    source_numel: int | None = None,
) -> _StoragePrecision:
    """How storage keeps ``tensor``: decided once for the whole tensor.

    A row shard the compiler cut from a weight gives that weight's size as
    ``source_numel``, so it is kept like the rest of the weight.
    """
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        return "float32"
    ndim = tensor.ndim if effective_ndim is None else effective_ndim
    if storage.precision in {"int8", "int6", "int4"} and ndim < 2:
        return "float16"
    if storage.precision in {"int8", "int6", "int4"} and (
        # Quantization leaves matrices below the threshold unquantized.
        (source_numel or tensor.numel()) < storage.min_quantized_values
        or not _on_integer_codec_grid(tensor, storage, interleave_segments)
    ):
        # QuantizationConfig decides which weights are lossy. Storage must not
        # quantize anything it did not reconstruct, such as buffers (RoPE
        # tables, masks) or values derived from weights by other arithmetic.
        finite = tensor.detach()[torch.isfinite(tensor.detach())]
        if finite.numel() and float(finite.abs().max()) > 65504:
            return "float32"
        return "float16"
    return storage.precision


def _on_integer_codec_grid(
    tensor: torch.Tensor,
    storage: StorageConfig,
    interleave_segments: tuple[tuple[int, int, int], ...] = (),
) -> bool:
    """Return whether integer storage reproduces ``tensor`` exactly.

    Grouped kernels keep static tensors in a four-row physical order. Storage
    quantizes the logical order, so undo that permutation first.
    """
    group_size = _static_storage_group_size(tensor, storage)
    # Compilation, sharding and storage each ask about the same tensors.
    key = (
        storage.precision, storage.scale_precision, group_size,
        tuple(interleave_segments), tensor._version,
    )
    cached = _CODEC_GRID_CACHE.get(id(tensor))
    if cached is not None and cached[0]() is tensor and cached[1] == key:
        return cached[2]
    result = _check_integer_codec_grid(tensor, storage, group_size, interleave_segments)
    if cached is None or cached[0]() is not tensor:
        weakref.finalize(tensor, _CODEC_GRID_CACHE.pop, id(tensor), None)
    _CODEC_GRID_CACHE[id(tensor)] = (weakref.ref(tensor), key, result)
    return result


# id(tensor) -> (weak reference, the question's key, answer).
_CODEC_GRID_CACHE: dict[int, tuple[weakref.ref, tuple, bool]] = {}
# Groups checked per step, to bound the float64 temporaries for large tables.
_CODEC_GRID_CHUNK_GROUPS = 1 << 13


def _check_integer_codec_grid(
    tensor: torch.Tensor,
    storage: StorageConfig,
    group_size: int,
    interleave_segments: tuple[tuple[int, int, int], ...],
) -> bool:
    # NumPy rather than torch: these are many small passes, and torch's
    # thread pool makes each one much slower on a busy machine.
    flat = tensor.detach().cpu().reshape(-1)
    native = flat.dtype in (torch.float16, torch.float32, torch.float64)
    # A format NumPy lacks, such as bfloat16, widens exactly to float64 and
    # rounds back through torch.
    values = flat.numpy(force=True) if native else flat.to(torch.float64).numpy()
    for start, rows, inner in interleave_segments:
        if rows % 4 or start < 0 or start + rows * inner > values.size:
            return False
    if values.size == 0 or values.size % group_size:
        return False
    maximum_code = {"int8": 127, "int6": 31, "int4": 7}[storage.precision]
    # Every group is checked on its own, so whole groups at a time, in
    # logical order, give the same answer as all of them at once.
    for groups in _logical_group_chunks(values, interleave_segments, group_size):
        work = groups.astype(np.float64)
        if not np.isfinite(work).all():
            return False
        # The encoder's own scale rule, so a tensor passes exactly when the
        # encoder reproduces it.
        scale = group_scales(
            np.abs(work).max(axis=1), maximum_code, storage.scale_precision,
        )[:, None]
        if not np.isfinite(scale).all():
            return False
        reconstructed = np.clip(np.round(work / scale), -maximum_code, maximum_code) * scale
        if native:
            rounded = reconstructed.astype(values.dtype)
        else:
            rounded = torch.from_numpy(reconstructed).to(flat.dtype).double().numpy()
        if not np.array_equal(rounded, groups):
            return False
    return True


def _logical_group_chunks(
    values: np.ndarray,
    segments: tuple[tuple[int, int, int], ...],
    group_size: int,
):
    """Yield ``values`` in logical order as ``(groups, group_size)`` blocks.

    Grouped kernels store four-row blocks column-interleaved; each segment
    is undone on its own as it is reached, instead of through one gather of
    the whole tensor.
    """
    chunk_values = _CODEC_GRID_CHUNK_GROUPS * group_size
    carry = values[:0]
    for piece in _logical_pieces(values, segments, chunk_values):
        if carry.size:
            piece = np.concatenate((carry, piece))
        usable = piece.size - piece.size % group_size
        if usable:
            yield piece[:usable].reshape(-1, group_size)
        carry = piece[usable:]


def _logical_pieces(
    values: np.ndarray,
    segments: tuple[tuple[int, int, int], ...],
    chunk_values: int,
):
    """Yield ``values`` in logical order, about ``chunk_values`` at a time."""
    ordered = sorted(segments)
    if any(
        ordered[index][0] + ordered[index][1] * ordered[index][2] > ordered[index + 1][0]
        for index in range(len(ordered) - 1)
    ):
        # Overlapping segments compose; gather through every permutation.
        order = np.arange(values.size)
        for start, rows, inner in segments:
            end = start + rows * inner
            order[start:end] = order[start:end].reshape(rows // 4, inner, 4).transpose(0, 2, 1).reshape(-1)
        values = values[order]
        ordered = []
    position = 0
    for start, rows, inner in [*ordered, (values.size, 0, 1)]:
        for offset in range(position, start, chunk_values):
            yield values[offset:min(start, offset + chunk_values)]
        blocks = values[start:start + rows * inner].reshape(rows // 4, inner, 4)
        step = max(1, chunk_values // (4 * inner))
        for block in range(0, rows // 4, step):
            yield blocks[block:block + step].transpose(0, 2, 1).reshape(-1)
        position = start + rows * inner


def _static_storage_group_size(
    tensor: torch.Tensor | None,
    storage: StorageConfig,
) -> int:
    """The quantization group width of one logical matrix, as quantization
    chose it: groups never cross a row."""
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2 or tensor.shape[-1] < 1:
        return storage.group_size
    return effective_group_size(int(tensor.shape[-1]), storage.group_size)


def _static_storage_shard_limits(
    static_tensors: dict[str, torch.Tensor | None],
    storage: StorageConfig,
    *,
    effective_ndims: dict[str, int] | None = None,
    interleaved_layouts: dict[str, tuple[tuple[int, int, int], ...]] | None = None,
    source_numels: dict[str, int] | None = None,
) -> dict[str, int]:
    """Cap static shards so every temporary decode stream fits in Scratch."""
    if not storage.compression:
        return {}
    limits: dict[str, int] = {}
    for key, tensor in static_tensors.items():
        precision = _static_storage_precision(
            tensor, storage, (effective_ndims or {}).get(key),
            (interleaved_layouts or {}).get(key, ()),
            (source_numels or {}).get(key),
        )
        group_size = _static_storage_group_size(tensor, storage)
        if precision == "float32":
            limit = SCRATCH_LIST_LIMIT // 4
        elif precision == "float16":
            limit = SCRATCH_LIST_LIMIT // 2
        else:
            scale_bytes = 4 if storage.scale_precision == "float32" else 2
            scale_groups = SCRATCH_LIST_LIMIT // scale_bytes
            limit = min(SCRATCH_LIST_LIMIT, scale_groups * group_size)
            # Each physical payload is quantized independently.  Ending a
            # shard mid-group would restart scale selection in the next shard
            # and change weights compared with the logical matrix (notably
            # shared MoE banks whose effective row group is 48 values).
            limit -= limit % group_size
            if limit < group_size:
                raise ValueError("quantized static shard cannot fit one group")
        limits[f"W_{key}"] = limit
    return limits


def _validate_nonfloating_static_tensors(
    static_tensors: dict[str, torch.Tensor | None],
    *,
    compressed: bool,
) -> None:
    """Reject integer constants which Scratch's selected number path changes."""
    scratch_dtype = torch.float32 if compressed else torch.float64
    label = "float32 compressed storage" if compressed else "Scratch numbers"
    for key, tensor in static_tensors.items():
        if not isinstance(tensor, torch.Tensor) or tensor.is_floating_point():
            continue
        if tensor.is_complex():
            raise ValueError(f"static tensor {key!r} has unsupported complex values")
        original = tensor.detach().cpu()
        try:
            round_trip = original.to(scratch_dtype).to(original.dtype)
        except (RuntimeError, TypeError) as error:
            raise ValueError(
                f"static tensor {key!r} cannot be represented by {label}"
            ) from error
        if not torch.equal(original, round_trip):
            suggestion = (
                "; use compression=False for integers exactly representable "
                "by Scratch doubles"
                if compressed else ""
            )
            raise ValueError(
                f"static tensor {key!r} contains values not exactly "
                f"representable by {label}{suggestion}"
            )


def _deinterleave_static_values(
    values: list,
    segments: tuple[tuple[int, int, int], ...],
) -> list:
    """Convert four-row physical groups back to logical output-major rows."""
    if isinstance(values, StaticValues):
        array = values.array.copy()
        for start, rows, inner in segments:
            if rows % 4 or start < 0 or start + rows * inner > len(values):
                raise ValueError("malformed grouped static-weight layout")
            end = start + rows * inner
            array[start:end] = (
                values.array[start:end].reshape(rows // 4, inner, 4).transpose(0, 2, 1).reshape(-1)
            )
        return StaticValues(array)
    result = list(values)
    for start, rows, inner in segments:
        if rows % 4 or start < 0 or start + rows * inner > len(values):
            raise ValueError("malformed grouped static-weight layout")
        source = values[start:start + rows * inner]
        logical: list = []
        for group_start in range(0, len(source), 4 * inner):
            group = source[group_start:group_start + 4 * inner]
            for lane in range(4):
                logical.extend(group[column * 4 + lane] for column in range(inner))
        result[start:start + rows * inner] = logical
    return result


def _interleave_static_values(
    values: list,
    segments: tuple[tuple[int, int, int], ...],
) -> list:
    """Apply the grouped kernel's four-row physical storage permutation."""
    result = list(values)
    for start, rows, inner in segments:
        source = values[start:start + rows * inner]
        physical: list = []
        for group_start in range(0, len(source), 4 * inner):
            group = source[group_start:group_start + 4 * inner]
            for column in range(inner):
                physical.extend(
                    group[lane * inner + column] for lane in range(4)
                )
        result[start:start + rows * inner] = physical
    return result


def _physical_interleave_segments(
    layout: ListLayout,
    segments: tuple[tuple[int, int, int], ...],
) -> dict[str, tuple[tuple[int, int, int], ...]]:
    """Map logical-list segment offsets onto its physical Scratch shards."""
    result = {}
    shard_start = 0
    for shard in layout.shards:
        shard_end = shard_start + shard.capacity
        local = []
        for start, rows, inner in segments:
            end = start + rows * inner
            if end <= shard_start or start >= shard_end:
                continue
            overlap_start = max(start, shard_start)
            overlap_end = min(end, shard_end)
            group_values = 4 * inner
            if (
                (overlap_start - start) % group_values
                or (overlap_end - overlap_start) % group_values
            ):
                raise ValueError(
                    "grouped weight sharding must preserve complete four-row groups"
                )
            local.append((
                overlap_start - shard_start,
                (overlap_end - overlap_start) // inner,
                inner,
            ))
        if local:
            result[shard.name] = tuple(local)
        shard_start = shard_end
    return result


def _build_interleave_program(
    specs: list[tuple[str, tuple[tuple[int, int, int], ...]]],
):
    """Build one init-only row-major to four-row storage permutation."""
    from cattorch.util.scratch.dsl import (
        Program, add, append, change_var, clear, for_each, item, length,
        repeat, set_var, var,
    )

    temporary = "cattorch interleave buffer"
    body = []
    for name, segments in specs:
        body.append(clear(temporary))
        for start, rows, inner in segments:
            body.extend((
                set_var("interleave group start", start + 1),
                repeat(rows // 4, (
                    set_var("interleave column", 0),
                    repeat(inner, (
                        *tuple(
                            append(
                                temporary,
                                item(
                                    name,
                                    add(
                                        var("interleave group start"),
                                        add(lane * inner, var("interleave column")),
                                    ),
                                ),
                            )
                            for lane in range(4)
                        ),
                        change_var("interleave column", 1),
                    )),
                    change_var("interleave group start", 4 * inner),
                )),
            ))
        body.extend((
            clear(name),
            for_each(
                "interleave copy index",
                length(temporary),
                (append(
                    name,
                    item(temporary, var("interleave copy index")),
                ),),
            ),
        ))
    body.append(clear(temporary))
    return Program(
        "interleave_grouped_weights",
        variables=(
            "interleave group start", "interleave column",
            "interleave copy index",
        ),
        lists=(temporary, *(name for name, _ in specs)),
        body=tuple(body),
    )


def _apply_static_storage(
    sprite: dict,
    layouts: dict[str, ListLayout],
    static_tensors: dict[str, torch.Tensor | None],
    storage: StorageConfig,
    *,
    interleaved_layouts: dict[
        str, tuple[tuple[int, int, int], ...]
    ] | None = None,
    effective_ndims: dict[str, int] | None = None,
    source_numels: dict[str, int] | None = None,
) -> set[str]:
    """Encode static physical shards and install the one-time unpacker."""
    source_numels = source_numels or {}
    static_names = {f"W_{key}" for key in static_tensors}
    interleaved_layouts = interleaved_layouts or {}
    physical_interleaves = {}
    for key, segments in interleaved_layouts.items():
        logical_name = f"W_{key}"
        layout = layouts.get(logical_name)
        if layout is not None:
            physical_interleaves.update(
                _physical_interleave_segments(layout, segments)
            )
    _validate_nonfloating_static_tensors(
        static_tensors, compressed=storage.compression,
    )
    if not storage.compression:
        if storage.precision in {"float16", "int8", "int6", "int4"}:
            for logical_name in static_names:
                tensor = static_tensors.get(logical_name.removeprefix("W_"))
                if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                    continue
                precision = _static_storage_precision(
                    tensor, storage, (effective_ndims or {}).get(
                        logical_name.removeprefix("W_"),
                    ),
                    interleaved_layouts.get(logical_name.removeprefix("W_"), ()),
                    source_numels.get(logical_name.removeprefix("W_")),
                )
                group_size = _static_storage_group_size(tensor, storage)
                for shard in layouts.get(logical_name, ListLayout(logical_name, 0, ())).shards:
                    contents = sprite["lists"][shard.identifier][1]
                    segments = physical_interleaves.get(shard.name, ())
                    logical_contents = (
                        _deinterleave_static_values(contents, segments)
                        if segments else contents
                    )
                    rounded = rounded_values(
                        logical_contents,
                        precision,
                        group_size,
                        storage.scale_precision,
                    )
                    sprite["lists"][shard.identifier][1] = (
                        _interleave_static_values(rounded, segments)
                        if segments else rounded
                    )
        _materialize_static_values(sprite)
        return static_names

    specs: list[EncodedList] = []
    runtime_interleaves = []
    for logical_name in sorted(static_names):
        layout = layouts.get(logical_name)
        if layout is None:
            continue
        tensor = static_tensors.get(logical_name.removeprefix("W_"))
        precision = _static_storage_precision(
            tensor, storage, (effective_ndims or {}).get(
                logical_name.removeprefix("W_"),
            ),
            interleaved_layouts.get(logical_name.removeprefix("W_"), ()),
            source_numels.get(logical_name.removeprefix("W_")),
        )
        group_size = _static_storage_group_size(tensor, storage)
        for position, shard in enumerate(layout.shards, 1):
            contents = sprite["lists"][shard.identifier][1]
            if not contents:
                continue
            payload_name = f"cattorch payload {len(specs) + 1}"
            segments = physical_interleaves.get(shard.name, ())
            encoded_contents = (
                _deinterleave_static_values(contents, segments)
                if segments else contents
            )
            if precision in ("int8", "int6", "int4"):
                codes, scales = _quantize_arrays(
                    encoded_contents,
                    precision,
                    group_size,
                    storage.scale_precision,
                )
                raw_payload = _quantized_bytes(codes, precision)
                scale_number = len(specs) + 1
                # With payload_bytes on every spec the unpacker encodes whole
                # banks itself, so the per-list Base92 payloads stay empty.
                specs.append(EncodedList(
                    shard.name,
                    payload_name,
                    "",
                    precision,
                    value_count=len(contents),
                    group_size=group_size,
                    scale_list_name=f"cattorch quant scales {scale_number}",
                    scale_payload_name=f"cattorch scale payload {scale_number}",
                    scale_precision=storage.scale_precision,
                    payload_bytes=raw_payload,
                    scale_values=tuple(scales.tolist()),
                ))
            else:
                raw_payload = _float_bytes(
                    encoded_contents, cast(Literal["float32", "float16"], precision),
                )
                specs.append(EncodedList(
                    shard.name,
                    payload_name,
                    "",
                    precision,
                    payload_bytes=raw_payload,
                ))
            if segments:
                runtime_interleaves.append((shard.name, segments))
            sprite["lists"][shard.identifier][1] = []
        # Its shards hold the payload now; let a derived tensor go.
        static_tensors[logical_name.removeprefix("W_")] = None

    if specs:
        for index, (decoder_name, program) in enumerate(
            build_decoder_programs(specs).items()
        ):
            _add_warp_procedure(
                sprite,
                decoder_name,
                program,
                x=960,
                y=480 + index * 120,
            )
        sprite = _add_warp_procedure(
            sprite,
            "cattorch unpack weights",
            build_shared_unpack_program(specs),
            x=640,
            y=480,
        )
        _merge_lists_by_name(
            sprite,
            {spec.name for spec in specs}
            | {spec.scale_list_name for spec in specs if spec.scale_list_name}
            | {
                "cattorch f32 powers", "cattorch f16 powers", "cattorch decoded bytes",
                "cattorch decoded values", "cattorch decoded scales",
                "cattorch huffman table", "cattorch huffman powers",
            },
        )
        _merge_variables_by_name(
            sprite,
            {
                entry[0]
                for entry in sprite.get("variables", {}).values()
                if entry[0].startswith("decode ")
                or entry[0] == "cattorch active payload"
            },
        )
        if runtime_interleaves:
            _add_warp_procedure(
                sprite,
                "cattorch interleave grouped weights",
                _build_interleave_program(runtime_interleaves),
                x=640,
                y=600,
            )
            _merge_lists_by_name(
                sprite,
                {name for name, _ in runtime_interleaves}
                | {"cattorch interleave buffer"},
            )
            _merge_variables_by_name(
                sprite,
                {
                    "interleave group start", "interleave column",
                    "interleave copy index",
                },
            )
            # Calls are prepended to init, so add the permutation first and
            # unpack second to execute them in unpack -> interleave order.
            _insert_init_call(sprite, "cattorch interleave grouped weights")
        _insert_init_call(sprite, "cattorch unpack weights")
    _materialize_static_values(sprite)
    return static_names


def _materialize_static_values(sprite: dict) -> None:
    """Turn the static values storage left in place into the lists JSON needs."""
    for entry in sprite.get("lists", {}).values():
        entry[1] = static_list_contents(entry[1])


def _add_prepare_for_save(
    sprite: dict,
    layouts: dict[str, ListLayout],
    *,
    static_names: set[str],
    compressed: bool,
    inputs: AbstractSet[str] = frozenset(),
) -> None:
    """Add a public procedure that drops rebuildable runtime list contents.

    Input lists are kept: they hold the caller's values, and a program's
    persistent state is passed to its entry points as inputs.
    """
    from cattorch.util.scratch.dsl import Program, clear

    names = [
        shard.name
        for layout in layouts.values()
        if not layout.name.startswith("_") and layout.name not in inputs
        and (compressed or layout.name not in static_names)
        for shard in layout.shards
    ]
    if not names:
        return
    program = Program(
        "prepare_for_save",
        lists=names,
        body=tuple(clear(name) for name in names),
    )
    _add_warp_procedure(
        sprite, "cattorch prepare for save", program, x=640, y=640,
    )

    blocks = sprite["blocks"]
    definition = _find_procedure_definition(sprite, "cattorch prepare for save")
    old_root = blocks[definition]["next"]
    initialized_id = next(
        sid for sid, entry in sprite["variables"].items()
        if entry[0] == "cattorch initialized"
    )
    reset = f"cattorch_prepare_save_{uuid.uuid4().hex[:8]}"
    blocks[reset] = {
        "opcode": "data_setvariableto", "next": old_root, "parent": definition,
        "inputs": {"VALUE": [1, [4, 0]]},
        "fields": {"VARIABLE": ["cattorch initialized", initialized_id]},
        "shadow": False, "topLevel": False,
    }
    blocks[definition]["next"] = reset
    blocks[old_root]["parent"] = reset


def _gate_chain(
    sprite: dict, start: str, gate: str = "cattorch project output", stop: str | None = None,
) -> None:
    """Run the command chain from ``start`` only while the ``gate`` variable is 1.

    The chain runs to its end, or up to ``stop``, which then follows the new
    ``if`` block. Hidden prefill uses it to skip what a prompt token's logits
    alone need, such as the final layer's attention output and the output
    projection. The gate variable is created, set to 1, if the sprite has none.
    """
    blocks = sprite["blocks"]
    predecessors = [
        identifier for identifier, block in blocks.items() if block.get("next") == start
    ]
    if len(predecessors) != 1:
        raise ValueError(f"expected one block before the gated chain, got {predecessors}")
    predecessor = predecessors[0]
    last = start
    while blocks[last].get("next") not in (stop, None):
        last = blocks[last]["next"]
    if blocks[last].get("next") != stop:
        raise ValueError("the gated chain does not end at the given block")
    blocks[last]["next"] = None

    variables = sprite.setdefault("variables", {})
    variable_id = next(
        (identifier for identifier, entry in variables.items() if entry[0] == gate), None,
    )
    if variable_id is None:
        variable_id = f"cattorch_gate_{uuid.uuid4().hex[:8]}"
        variables[variable_id] = [gate, 1]
    token = uuid.uuid4().hex[:8]
    guard = f"cattorch_gate_{token}_if"
    condition = f"cattorch_gate_{token}_condition"
    blocks[predecessor]["next"] = guard
    blocks[guard] = {
        "opcode": "control_if", "next": stop, "parent": predecessor,
        "inputs": {"CONDITION": [2, condition], "SUBSTACK": [2, start]},
        "fields": {}, "shadow": False, "topLevel": False,
    }
    blocks[condition] = {
        "opcode": "operator_equals", "next": None, "parent": guard,
        "inputs": {
            "OPERAND1": [3, [12, gate, variable_id], [10, ""]],
            "OPERAND2": [1, [4, 1]],
        },
        "fields": {}, "shadow": False, "topLevel": False,
    }
    blocks[start]["parent"] = guard
    if stop is not None:
        blocks[stop]["parent"] = guard


_TOP_K_VALUES = "cattorch top k values"
_TOP_K_IDS = "cattorch top k ids"
_TOP_K_PROCEDURE = "cattorch select top k"
# 1 while a call is changing the states: while reset, prefill or decode runs in
# a generation sprite, or while a program's reset or entry point writes them.
# Still 1 when another call starts if that one was stopped part way or is
# paused by Scratch, and the states are in an unknown condition.
_BUSY = "_cattorch busy"
# Changed by every call that starts, and by every generation call that finds,
# when it ends, that it changed since that call started: another call then
# took over the states while it was paused, and wrote to them, or will write
# to them, when it resumes.
_EPOCH = "_cattorch epoch"
_INTERRUPTED = (
    "an earlier reset, prefill or decode was stopped or is still running; "
    "reset or prefill first"
)
_OVERTAKEN = (
    "another reset, prefill, decode or init ran during this one; reset or prefill first"
)


def _epoch_current():
    """Whether no call has started since the one running the epoch procedure body."""
    from cattorch.util.scratch.dsl import argument, eq, var

    return eq(argument("epoch"), var(_EPOCH))


def _add_epoch_procedure(
    sprite: dict, name: str, body, *, x: int, y: int, take_over=(), refusal=None,
    variables: Sequence[str] = (), lists: Sequence[str] = (),
) -> dict:
    """Add the public ``name``, which changes the epoch and runs ``body`` with it.

    Scratch can pause a warp block and run other scripts meanwhile, so the
    body, an internal procedure with an ``epoch`` argument, tells by
    ``_epoch_current()`` whether another call started since. ``take_over``
    runs first; with a ``refusal``, that runs instead while ``_BUSY`` is 1.
    ``variables`` and ``lists`` are what those two use.
    """
    sprite = _add_warp_procedure(
        sprite, _epoch_procedure(name), body, x=x + 320, y=y, arguments=("epoch",),
    )
    return _add_epoch_entry(
        sprite, name, f"{body.name}_public", x=x, y=y, take_over=take_over, refusal=refusal,
        variables=(_BUSY, _EPOCH, *variables), lists=lists,
    )


def _epoch_procedure(name: str) -> str:
    """The internal procedure, with an ``epoch`` argument, that the public ``name`` runs."""
    return f"_cattorch {name.removeprefix('cattorch ')} in epoch %s"


def _add_epoch_entry(
    sprite: dict, name: str, program_name: str, *, x: int, y: int, take_over=(), refusal=None,
    variables: Sequence[str] = (), lists: Sequence[str] = (),
) -> dict:
    """Add the public ``name``, which changes the epoch and runs its epoch procedure.

    As ``_add_epoch_procedure``, for an epoch procedure the sprite already has;
    ``variables`` are all the ones the public block uses.
    """
    from cattorch.util.scratch.dsl import Program, call, change_var, eq, if_else, var

    start = (*take_over, change_var(_EPOCH, 1), call(_epoch_procedure(name), epoch=var(_EPOCH)))
    public = Program(
        program_name, variables=dict.fromkeys(variables), lists=lists,
        body=start if refusal is None else (if_else(eq(var(_BUSY), 1), refusal, start),),
    )
    return _add_warp_procedure(sprite, name, public, x=x, y=y)


# What a program entry point or a plain forward reports when another call
# started while it ran.
_CALL_OVERTAKEN = "another call ran during this one"


def _unlink_broadcasts(sprite: dict, message: str) -> None:
    """Remove every ``broadcast and wait`` of ``message``; each must end its chain."""
    blocks = sprite["blocks"]
    for identifier, block in list(blocks.items()):
        if block.get("opcode") != "event_broadcastandwait":
            continue
        menu = block.get("inputs", {}).get("BROADCAST_INPUT", [None, None])[1]
        if not (isinstance(menu, list) and len(menu) > 1 and menu[1] == message):
            continue
        if block.get("next") is not None:
            raise ValueError(f"the {message!r} broadcast doesn't end its chain")
        parent = blocks[block["parent"]]
        if parent.get("next") == identifier:
            parent["next"] = None
        else:
            # The only block of an ``if`` branch.
            inputs = parent.get("inputs", {})
            for key in ("SUBSTACK", "SUBSTACK2"):
                if inputs.get(key, [None, None])[1] == identifier:
                    del inputs[key]
        del blocks[identifier]


def _take_epoch_argument(sprite: dict, name: str) -> str:
    """Rename procedure ``name`` to its epoch procedure, which has an ``epoch`` argument.

    Only its definition is renamed, not the calls to it. Returns the new name.
    """
    from cattorch.util.scratch.dsl import argument_id

    blocks = sprite["blocks"]
    inner = _epoch_procedure(name)
    prototype = blocks[_find_procedure_definition(sprite, name)]["inputs"]["custom_block"][1]
    blocks[prototype]["mutation"] = _procedure_mutation(inner, warp=True, arguments=("epoch",))
    reporter = f"{prototype}_{argument_id('epoch')}"
    blocks[prototype]["inputs"] = {argument_id("epoch"): [1, reporter]}
    blocks[reporter] = {
        "opcode": "argument_reporter_string_number", "next": None,
        "parent": prototype, "inputs": {}, "fields": {"VALUE": ["epoch", None]},
        "shadow": True, "topLevel": False,
    }
    return inner


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


def _mark_init_decoding(sprite: dict, prefix: str) -> None:
    """Keep the weights marked undecoded while ``cattorch init`` decodes them.

    Init marks them so as it starts, and again at its end if another call
    started meanwhile; it then also changes the epoch, so that a call it
    resumed during fails. ``cattorch init`` must later run in an epoch
    procedure, with ``_add_init_epoch_entry``.
    """
    from cattorch.util.scratch.dsl import Program, change_var, if_, not_, set_var

    init, initialized = "cattorch init", "cattorch initialized"
    # A call that starts meanwhile decodes the weights itself.
    _add_to_procedure(sprite, init, Program(
        f"{prefix}_init_start", variables=(initialized,), body=(set_var(initialized, 0),),
    ), first=True)
    # That call may have finished decoding them while this init was, and
    # this one then left them half overwritten.
    _add_to_procedure(sprite, init, Program(
        f"{prefix}_init_end", variables=(initialized, _EPOCH),
        body=(if_(not_(_epoch_current()), (set_var(initialized, 0), change_var(_EPOCH, 1))),),
    ))


def _add_init_epoch_entry(sprite: dict, inner: str, prefix: str, *, x: int, y: int) -> dict:
    """Add the public ``cattorch init``, which changes the epoch and runs ``inner``.

    ``inner`` is init's epoch procedure, from ``_take_epoch_argument``. Calls
    to init inside the sprite, such as a step's check that the weights are
    decoded, run ``inner`` with the epoch as it is then, so that init checks
    itself as when it's called directly; ``_stop_overtaken_decoding`` then
    passes them the epoch of the call making them instead, where weights are
    decoded. Loose blocks on the canvas keep running the public block.
    """
    from cattorch.util.scratch.dsl import argument_id

    init = "cattorch init"
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
    inner = _take_epoch_argument(sprite, "cattorch init")
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


def _epoch_input(sprite: dict, call: str, owner: str | None, epoch_id: str) -> list:
    """The ``epoch`` input of ``call``: the epoch its procedure runs in, or the current one.

    A call made in a procedure with an ``epoch`` argument passes that on;
    any other passes the epoch as it is when the call runs.
    """
    blocks = sprite["blocks"]
    if owner is None or "epoch" not in _procedure_arguments(blocks, owner):
        return [3, [12, _EPOCH, epoch_id], [10, ""]]
    reporter = f"{call}_epoch"
    blocks[reporter] = {
        "opcode": "argument_reporter_string_number", "next": None, "parent": call,
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
    root = _epoch_procedure("cattorch init")
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
        for call in calls.get(pending.pop(), []):
            name = blocks[call]["mutation"]["proccode"]
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
        direct = {blocks[call]["mutation"]["proccode"] for call in calls[definitions[root]]}
        guarded = [name for name in guarded if name in direct]
    epoch_id = next(
        identifier for identifier, entry in sprite["variables"].items() if entry[0] == _EPOCH
    )

    from cattorch.util.scratch.dsl import argument_id

    callers: dict[str, list[str]] = {}
    for identifier, block in blocks.items():
        if block.get("opcode") == "procedures_call":
            callers.setdefault(block["mutation"]["proccode"], []).append(identifier)

    def take_epoch(name: str) -> str:
        prototype_id = blocks[definitions[name]]["inputs"]["custom_block"][1]
        prototype = blocks[prototype_id]
        new = f"{name} for epoch %s"
        mutation = _procedure_mutation(new, warp=True, arguments=("epoch",))
        mutation["warp"] = prototype["mutation"].get("warp", "false")
        prototype["mutation"] = mutation
        reporter = f"{prototype_id}_{argument_id('epoch')}"
        prototype["inputs"] = {argument_id("epoch"): [1, reporter]}
        blocks[reporter] = {
            "opcode": "argument_reporter_string_number", "next": None,
            "parent": prototype_id, "inputs": {}, "fields": {"VALUE": ["epoch", None]},
            "shadow": True, "topLevel": False,
        }
        return new

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
            not blocks[call].get("topLevel") and passing(owners[call], found)
            for call in callers.get(name, [])
        )

    # A procedure that starts a decode, such as a forward's or step's check
    # that the weights are decoded, starts it in the epoch the call running
    # it started in, passed down to it; a call overtaken before it starts
    # the decode then decodes nothing.
    threaded: set[str] = set()
    for name in (root, *guarded):
        for call in callers.get(name, []):
            found: set[str] = set()
            if passing(owners[call], found):
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
    A forward that another call started during clears ``outputs``, every
    output list and shard, reports so in ``cattorch status``, sends no
    completion broadcast, and changes the epoch, so that a call it resumed
    during fails too. An init that another call started during, the forward's
    own included, leaves the weights for the next call to decode again.
    """
    from cattorch.util.scratch.dsl import (
        Program, broadcast_and_wait, change_var, clear, if_else, set_var,
    )

    forward = "cattorch forward"
    complete = f"{forward} complete"
    current = _epoch_current()
    # Sent at the end instead, only by a call that ran alone.
    _unlink_broadcasts(sprite, complete)
    _add_to_procedure(sprite, forward, Program(
        "plain_forward_end", variables=(_EPOCH, STATUS_VARIABLE), lists=outputs,
        body=(if_else(current, (broadcast_and_wait(complete),), (
            change_var(_EPOCH, 1), *(clear(name) for name in outputs),
            set_var(STATUS_VARIABLE, _CALL_OVERTAKEN),
        )),),
    ))
    _mark_init_decoding(sprite, "plain")
    _take_epoch_argument(sprite, forward)
    inner_init = _take_epoch_argument(sprite, "cattorch init")
    sprite = _add_epoch_entry(
        sprite, forward, "plain_forward_public", x=0, y=320, variables=(_EPOCH,),
    )
    sprite = _add_init_epoch_entry(sprite, inner_init, "plain", x=0, y=480)
    # A variable-length export renamed the loose forward block on the canvas
    # with the procedure it wraps, so it ran that unchecked.
    wrapped = ("_cattorch checked forward", "_cattorch forward unit")
    for block in sprite["blocks"].values():
        if (
            block.get("opcode") == "procedures_call" and block.get("topLevel")
            and block["mutation"].get("proccode") in wrapped
        ):
            block["mutation"] = _procedure_mutation(forward)
    _stop_overtaken_decoding(sprite)
    try:
        _find_procedure_definition(sprite, "cattorch prepare for save")
    except ValueError:
        return sprite
    # Re-encoding the weights takes them from a running call.
    _add_to_procedure(sprite, "cattorch prepare for save", Program(
        "plain_prepare_start", variables=(_EPOCH,), body=(change_var(_EPOCH, 1),),
    ), first=True)
    return sprite


# An empty slot of the top-k lists. Scratch compares a number with an empty
# string as text, and every number's text, -Infinity and NaN included, sorts
# after "", so the first ``top_k`` logits always fill the empty slots, with
# the same strict comparison that keeps the lower token ID first among
# equal logits. A -Infinity placeholder would never be replaced by a
# -Infinity logit, leaving an ID of -1. NaN's text also sorts after every
# number's, so a NaN logit ranks first, as in ``torch.topk``.
_TOP_K_EMPTY = ""


def _top_k_placeholders(top_k: int) -> list:
    from cattorch.util.scratch.dsl import append, clear

    initialize = [clear(_TOP_K_VALUES), clear(_TOP_K_IDS)]
    for _ in range(top_k):
        initialize.extend((append(_TOP_K_VALUES, _TOP_K_EMPTY), append(_TOP_K_IDS, -1)))
    return initialize


def _top_k_insertion(rank: int, top_k: int):
    """Shift a fixed sorted prefix and insert the current logit at ``rank``."""
    from cattorch.util.scratch.dsl import item, replace, sub, var

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
    from cattorch.util.scratch.dsl import gt, if_else, item, var

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
    from cattorch.util.scratch.dsl import (
        Program, for_each, gt, if_, item, length,
        set_var, var,
    )

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
    from cattorch.util.scratch.dsl import (
        Program, change_var, gt, if_, item, for_each, length,
        repeat, repeat_until, replace, set_var, sub, var,
    )

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
    step: str = "cattorch forward",
    reset: str = "cattorch reset cache",
    restore: str | None = None,
    tokens: str = "input",
    logits: str = "output",
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
    init check; it consumes the one token in ``tokens`` and writes ``logits``.
    A ``reset`` procedure is created that calls the ``restore`` procedure, if
    given, and clears ``state_lists``; the public ``cattorch reset`` runs it.
    Without a ``token_count``, ``cattorch cache length`` counts steps that
    finish with an ok status. A ``capacity`` of None sets no token limit.
    """
    from cattorch.util.scratch.dsl import (
        Program, append, call, change_var, clear, eq, for_each, gt, if_, if_else,
        item, length, lt, not_, round_, set_var, sub, var,
    )

    buffer = "cattorch prefill buffer"
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

    reset_lists = (*state_lists, logits, *top_k_lists)
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
    overtaken = (
        set_var(_BUSY, 1),
        change_var(_EPOCH, 1),
        # Another call may have finished decoding the weights while this one
        # was, and this one then left them half overwritten.
        set_var("cattorch initialized", 0),
        clear(logits), *(clear(name) for name in top_k_lists),
        set_var(STATUS_VARIABLE, _OVERTAKEN),
    )
    finish = if_else(current, (set_var(_BUSY, 0),), overtaken)
    lifecycle_variables = (_BUSY, _EPOCH, "cattorch initialized", STATUS_VARIABLE)
    lifecycle_lists = (logits, *top_k_lists)

    def add_public(name: str, body: Program, *, y: int, refusal=None) -> None:
        """Add ``name``, which takes the states over and runs ``body`` with its epoch."""
        nonlocal sprite
        sprite = _add_epoch_procedure(
            sprite, name, body, x=640, y=y, take_over=(set_var(_BUSY, 1),), refusal=refusal,
            variables=lifecycle_variables, lists=lifecycle_lists,
        )

    add_public("cattorch reset", Program(
        "generation_reset_in_epoch", variables=lifecycle_variables, lists=lifecycle_lists,
        body=(call(reset), finish),
    ), y=640)

    # A token that isn't a whole number in the vocabulary would read past the
    # embedding table and still report ok.
    valid_tokens = "_cattorch tokens valid"
    token = "_cattorch token"

    def check_tokens(source):
        if vocabulary is None:
            return (set_var(valid_tokens, 1),)
        return (
            set_var(valid_tokens, 1),
            for_each("_cattorch token index", length(source), (
                set_var(token, item(source, var("_cattorch token index"))),
                if_(not_(eq(var(token), round_(var(token)))), (set_var(valid_tokens, 0),)),
                if_(lt(var(token), 0), (set_var(valid_tokens, 0),)),
                if_(gt(var(token), vocabulary - 1), (set_var(valid_tokens, 0),)),
            )),
        )

    invalid_token = f"token ids must be whole numbers from 0 to {(vocabulary or 0) - 1}"
    invalid_decode = (
        clear(logits),
        *(clear(name) for name in top_k_lists),
        set_var(STATUS_VARIABLE, "decode requires exactly one token"),
    )
    token_variables = (valid_tokens, token, "_cattorch token index")
    decode = Program(
        "generation_decode",
        variables=(
            *lifecycle_variables, *token_variables,
            *(("cattorch project output",) if hidden_prefill else ()),
        ),
        lists=(tokens, *lifecycle_lists),
        body=(
            # A stopped hidden prefill can leave it 0.
            *((set_var("cattorch project output", 1),) if hidden_prefill else ()),
            if_else(
                eq(length(tokens), 1),
                (
                    *check_tokens(tokens),
                    if_else(
                        eq(var(valid_tokens), 1),
                        (call(step),),
                        (
                            clear(logits), *(clear(name) for name in top_k_lists),
                            set_var(STATUS_VARIABLE, invalid_token),
                        ),
                    ),
                ),
                invalid_decode,
            ),
            finish,
        ),
    )
    # A refusal sets the status, so a call still running couldn't report its
    # own; it is stopped too.
    add_public("cattorch decode", decode, y=160, refusal=(
        change_var(_EPOCH, 1),
        clear(logits), *(clear(name) for name in top_k_lists),
        set_var(STATUS_VARIABLE, _INTERRUPTED),
    ))

    # A prefill that another call took over skips the rest of its work.
    copy_and_reset = (
            clear(buffer),
            for_each("prefill index", length(tokens), (
                append(buffer, item(tokens, var("prefill index"))),
            )),
            if_(current, (call(reset),)),
            clear(tokens),
    )
    if hidden_prefill:
        valid_prefill: tuple = (
                if_(current, (set_var("cattorch project output", 0),)),
                for_each("prefill index", sub(length(buffer), 1), (if_(current, (
                    append(tokens, item(buffer, var("prefill index"))),
                    call(step),
                    clear(tokens),
                )),)),
                if_(current, (
                    set_var("cattorch project output", 1),
                    append(tokens, item(buffer, length(buffer))),
                    call(step),
                    clear(tokens),
                )),
        )
        prefill_variables: tuple[str, ...] = (
            "prefill index", "cattorch project output", STATUS_VARIABLE, *token_variables,
        )
    else:
        valid_prefill = (
            for_each("prefill index", length(buffer), (if_(current, (
                append(tokens, item(buffer, var("prefill index"))),
                call(step),
                clear(tokens),
            )),)),
        )
        prefill_variables = ("prefill index", STATUS_VARIABLE, *token_variables)
    checked_prefill = if_else(
        eq(var(valid_tokens), 1),
        valid_prefill,
        (set_var(STATUS_VARIABLE, invalid_token),),
    )
    if capacity is not None:
        checked_prefill = if_else(
            gt(length(buffer), capacity),
            (set_var(STATUS_VARIABLE, "prefill exceeds maximum context"),),
            (checked_prefill,),
        )
    prefill_body = (
        *copy_and_reset,
        *check_tokens(buffer),
        if_else(
            eq(length(buffer), 0),
            (set_var(STATUS_VARIABLE, "prefill requires at least one token"),),
            (checked_prefill,),
        ),
        finish,
    )
    prefill = Program(
        "generation_prefill",
        variables=(*prefill_variables, *lifecycle_variables),
        lists=(tokens, buffer, *lifecycle_lists),
        body=prefill_body,
    )
    add_public("cattorch prefill", prefill, y=320)
    if hidden_prefill:
        # Only prefill clears it, so a decode before any prefill must find 1.
        for entry in sprite["variables"].values():
            if entry[0] == "cattorch project output":
                entry[1] = 1
    if top_k_lists:
        _merge_lists_by_name(sprite, set(top_k_lists))

    # Splice the guard and bookkeeping in after the step's automatic init check.
    blocks = sprite["blocks"]
    step_def = _find_procedure_definition(sprite, step)
    init_check = blocks[step_def]["next"]
    old_root = blocks[init_check]["next"]
    logits_id = next(
        sid for sid, entry in sprite["lists"].items() if entry[0] == logits
    )
    suffix = uuid.uuid4().hex[:8]
    prefix = f"cattorch_generation_guard_{suffix}"
    status_id = f"{prefix}_status"
    cache_length_id = f"{prefix}_cache_length"
    sprite.setdefault("variables", {})[status_id] = [STATUS_VARIABLE, "ok"]
    sprite["variables"][cache_length_id] = ["cattorch cache length", 0]
    if capacity is not None:
        sprite["variables"][f"{prefix}_max_context"] = ["cattorch max context", capacity]

    def count_input(identifier: str, parent: str, shadow: list) -> list:
        if token_count is None:
            return [3, [12, "cattorch cache length", cache_length_id], shadow]
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
        clear_output = f"{prefix}_clear_output"
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
        "inputs": {"VALUE": [1, [10, "ok"]]},
        "fields": {"VARIABLE": [STATUS_VARIABLE, status_id]},
        "shadow": False, "topLevel": False,
    }
    blocks[old_root]["parent"] = success
    if capacity is not None:
        blocks[failure] = {
            "opcode": "data_setvariableto", "next": clear_output, "parent": guard,
            "inputs": {"VALUE": [1, [10, "maximum context exceeded"]]},
            "fields": {"VARIABLE": [STATUS_VARIABLE, status_id]},
            "shadow": False, "topLevel": False,
        }
        blocks[clear_output] = {
            "opcode": "data_deletealloflist", "next": None, "parent": failure,
            "inputs": {}, "fields": {"LIST": [logits, logits_id]},
            "shadow": False, "topLevel": False,
        }
        clear_failure_tail = clear_output
        for index, name in enumerate(top_k_lists):
            list_id = next(
                sid for sid, entry in sprite["lists"].items() if entry[0] == name
            )
            clear_top_k = f"{prefix}_clear_top_k_{index}"
            blocks[clear_failure_tail]["next"] = clear_top_k
            blocks[clear_top_k] = {
                "opcode": "data_deletealloflist", "next": None,
                "parent": clear_failure_tail,
                "inputs": {}, "fields": {"LIST": [name, list_id]},
                "shadow": False, "topLevel": False,
            }
            clear_failure_tail = clear_top_k

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
                "OPERAND2": [1, [10, "ok"]],
            },
            "fields": {}, "shadow": False, "topLevel": False,
        }
        blocks[increment] = {
            "opcode": "data_changevariableby", "next": None, "parent": tail,
            "inputs": {"VALUE": [1, [4, 1]]},
            "fields": {"VARIABLE": ["cattorch cache length", cache_length_id]},
            "shadow": False, "topLevel": False,
        }
    else:
        tail = f"{prefix}_set_cache_length"
        blocks[tail] = {
            "opcode": "data_setvariableto", "next": None, "parent": terminal,
            "inputs": {"VALUE": None},
            "fields": {"VARIABLE": ["cattorch cache length", cache_length_id]},
            "shadow": False, "topLevel": False,
        }
        blocks[tail]["inputs"]["VALUE"] = count_input(
            f"{prefix}_final_normalized_length", tail, [4, 0],
        )
    blocks[terminal]["next"] = tail
    if top_k is not None:
        top_k_guard = f"{prefix}_top_k_guard"
        top_k_condition = f"{prefix}_top_k_condition"
        top_k_output_length = f"{prefix}_top_k_output_length"
        top_k_call = f"{prefix}_top_k_call"
        blocks[tail]["next"] = top_k_guard
        blocks[top_k_guard] = {
            "opcode": "control_if", "next": None, "parent": tail,
            "inputs": {
                "CONDITION": [2, top_k_condition],
                "SUBSTACK": [2, top_k_call],
            },
            "fields": {}, "shadow": False, "topLevel": False,
        }
        blocks[top_k_condition] = {
            "opcode": "operator_gt", "next": None, "parent": top_k_guard,
            "inputs": {
                "OPERAND1": [3, top_k_output_length, [4, 0]],
                "OPERAND2": [1, [4, 0]],
            },
            "fields": {}, "shadow": False, "topLevel": False,
        }
        blocks[top_k_output_length] = {
            "opcode": "data_lengthoflist", "next": None,
            "parent": top_k_condition,
            "inputs": {}, "fields": {"LIST": [logits, logits_id]},
            "shadow": False, "topLevel": False,
        }
        blocks[top_k_call] = {
            "opcode": "procedures_call", "next": None, "parent": top_k_guard,
            "inputs": {}, "fields": {}, "shadow": False, "topLevel": False,
            "mutation": _procedure_mutation(_TOP_K_PROCEDURE),
        }

    # Reset also resets public state variables.
    reset_def = _find_procedure_definition(sprite, reset)
    reset_root = blocks[reset_def]["next"]
    reset_status = f"{prefix}_reset_status"
    reset_length = f"{prefix}_reset_length"
    blocks[reset_def]["next"] = reset_status
    blocks[reset_status] = {
        "opcode": "data_setvariableto", "next": reset_length, "parent": reset_def,
        "inputs": {"VALUE": [1, [10, "ok"]]},
        "fields": {"VARIABLE": [STATUS_VARIABLE, status_id]},
        "shadow": False, "topLevel": False,
    }
    blocks[reset_length] = {
        "opcode": "data_setvariableto", "next": reset_root, "parent": reset_status,
        "inputs": {"VALUE": [1, [4, 0]]},
        "fields": {"VARIABLE": ["cattorch cache length", cache_length_id]},
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
    )
