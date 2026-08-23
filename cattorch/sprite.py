"""Scratch sprite assembly, lifecycle, storage, and generation wiring."""

from __future__ import annotations

import json
import hashlib
import logging
import math
import re
import uuid
from typing import TYPE_CHECKING

import torch

from cattorch.graph import GenerationConfig, _GraphInfo, _get_shape
from cattorch.storage import (
    EncodedList,
    StorageConfig,
    build_decoder_programs,
    build_shared_unpack_program,
    encode_values,
    quantize_values,
    rounded_values,
    _encode_bytes,
    _float_bytes,
    _quantized_bytes,
)
from cattorch.util.scratch.block_combiner import combine
from cattorch.util.scratch.block_manager import BlockManager
from cattorch.util.scratch.remap import remap_ids
from cattorch.util.scratch.sharding import SCRATCH_LIST_LIMIT, ListLayout

if TYPE_CHECKING:
    from cattorch.transpiler import _Compiler

log = logging.getLogger(__name__)


# ── Sprite post-processing ───────────────────────────────────────────────────

def _rename_list(sprite, old_name, new_name):
    """Rename a list's display name and update all block references."""
    lists = sprite.get("lists", {})
    target_id = None
    for sid, entry in lists.items():
        if entry[0] == old_name:
            target_id = sid
            entry[0] = new_name
            break
    if target_id is None:
        return

    raw = json.dumps(sprite["blocks"])
    raw = raw.replace(
        json.dumps([old_name, target_id]),
        json.dumps([new_name, target_id]),
    )
    sprite["blocks"] = json.loads(raw)


def _refresh_data_display_name(blocks: dict, identifier: str, name: str) -> None:
    """Update Scratch's redundant display names after remapping a data ID."""
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


def _merge_duplicate_lists(sprite):
    """Merge local lists (names starting with '_') that have identical contents."""
    lists = sprite.get("lists", {})

    by_content: dict[tuple, list[tuple[str, str]]] = {}
    for sid, entry in lists.items():
        display_name = entry[0]
        if not display_name.startswith("_"):
            continue
        content_key = tuple(entry[1])
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

    sprite["blocks"] = remap_ids(sprite["blocks"], remap)

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
) -> dict[str, torch.Tensor | None]:
    """Share bit-identical flattened tensors before storage and sharding.

    Tensor shape is intentionally not part of the identity: Scratch kernels
    access the physical lists as flat arrays and have already compiled their
    shape-specific index arithmetic. Hash matches are confirmed byte-for-byte
    so this transform remains exact even for NaNs and signed zeroes.
    """
    winners: dict[tuple[str, int, bytes], tuple[str, bytes]] = {}
    aliases: dict[str, str] = {}
    kept: dict[str, torch.Tensor | None] = {}
    for key, tensor in static_tensors.items():
        if not isinstance(tensor, torch.Tensor):
            kept[key] = tensor
            continue
        contiguous = tensor.detach().cpu().contiguous()
        raw = contiguous.view(torch.uint8).numpy().tobytes()
        identity = (
            str(contiguous.dtype), contiguous.numel(), hashlib.sha256(raw).digest(),
        )
        previous = winners.get(identity)
        if previous is not None and previous[1] == raw:
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
        sprite["blocks"] = remap_ids(sprite["blocks"], id_remap)

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
) -> None:
    """Point aligned embedding shards at grouped output-head storage."""
    lists = sprite.get("lists", {})
    remap: dict[str, str] = {}
    display_names: dict[str, str] = {}
    for source_name, destination_names in aliases.items():
        source = layouts.get(source_name)
        destinations = [layouts.get(name) for name in destination_names]
        if (
            source is None
            or any(layout is None or len(layout.shards) != 1 for layout in destinations)
            or len(source.shards) != len(destinations)
        ):
            # Precision-aware storage shards can be smaller than the grouped
            # output-head layout. Retaining both representations is larger,
            # but remains correct and lets the normal storage path encode each
            # physical list within its temporary-byte budget.
            log.info(
                "Tied embedding/head sharing skipped because shard layouts "
                "do not match"
            )
            continue
        for source_shard, destination_layout in zip(source.shards, destinations):
            assert destination_layout is not None
            destination_shard = destination_layout.shards[0]
            if lists[source_shard.identifier][1] != lists[destination_shard.identifier][1]:
                raise ValueError("Tied embedding/head shard contents do not match")
            remap[source_shard.identifier] = destination_shard.identifier
            display_names[destination_shard.identifier] = destination_shard.name
            del lists[source_shard.identifier]
        del layouts[source_name]
        static_tensors.pop(source_name.removeprefix("W_"), None)

    if not remap:
        return
    sprite["blocks"] = remap_ids(sprite["blocks"], remap)
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


def _merge_lists_by_name(sprite, names: set[str]) -> None:
    """Unify repeated DSL declarations of persistent generation lists."""
    lists = sprite.get("lists", {})
    ids_by_name: dict[str, list[str]] = {}
    for identifier, entry in lists.items():
        if entry[0] in names:
            ids_by_name.setdefault(entry[0], []).append(identifier)
    remap = {}
    for ids in ids_by_name.values():
        if len(ids) < 2:
            continue
        winner = ids[0]
        for sid in ids[1:]:
            remap[sid] = winner
            del lists[sid]
    if remap:
        sprite["blocks"] = remap_ids(sprite["blocks"], remap)


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
            raise ValueError("Malformed generated command chain")
        chain.append(current)
        current = blocks[current].get("next")
    if stop is not None and current != stop:
        raise ValueError("Repeated layer roots are not consecutive")
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
    sprite: dict, compiler, storage: StorageConfig,
) -> bool:
    """Factor structurally identical ``blocks.N`` slabs into one warp block.

    Scratch cannot pass lists to custom blocks. The shared slab therefore uses
    the first layer's activation lists, copies only its final hidden state
    between calls, and selects concatenated weight banks through scalar bases.
    """
    def reject(reason: str) -> bool:
        log.info("Shared transformer layer skipped: %s", reason)
        return False

    if compiler.graph.generation.first_cache is not None:
        # Cached K/V lists are statically named in Scratch. Until attention is
        # split into thin per-layer cache wrappers, sharing the whole slab
        # would incorrectly direct every call to layer zero's cache.
        return reject("cached generation requires layer-specific cache wrappers")

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
    signatures = [
        [(str(nodes[name].target), tuple(_get_shape(nodes[name]))) for name, _ in group]
        for group in grouped
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
        names = []
        for identifier, block in blocks.items():
            if identifier not in owned:
                continue
            field = block.get("fields", {}).get("LIST")
            if field and field[0] in static_names and field[0] not in names:
                names.append(field[0])
        role_names.append(names)
    if not role_names[0] or any(len(names) != len(role_names[0]) for names in role_names):
        log.info("Repeated layer static roles: %s", role_names)
        return reject("static role counts differ")

    bank_roles = []
    for role, names in enumerate(zip(*role_names)):
        if len(set(names)) == 1:
            continue
        entries = [sprite["lists"][list_ids[name]] for name in names]
        sizes = [len(entry[1]) for entry in entries]
        if len(set(sizes)) != 1 or sum(sizes) > 200_000:
            return reject("a candidate weight bank is misaligned or oversized")
        bank_roles.append((role, names, sizes[0]))
    if not bank_roles:
        return reject("no layer-varying static weights were found")
    log.info("Repeated layer bank roles: %s", bank_roles)

    layer_zero_owned = owned_groups[0]
    base_variables = []
    for role, names, size in bank_roles:
        winner_id = list_ids[names[0]]
        winner = sprite["lists"][winner_id]
        winner[1] = [
            value
            for name in names
            for value in sprite["lists"][list_ids[name]][1]
        ]
        for name in names[1:]:
            sprite["lists"].pop(list_ids[name], None)

        tensors = [compiler.static_lists.pop(name.removeprefix("W_")) for name in names]
        all_matrix = all(isinstance(tensor, torch.Tensor) and tensor.ndim >= 2 for tensor in tensors)
        bank_tensor = torch.tensor(winner[1], dtype=torch.float32)
        if all_matrix and size >= storage.min_quantized_values:
            bank_tensor = bank_tensor.reshape(len(names), -1)
        compiler.static_lists[names[0].removeprefix("W_")] = bank_tensor

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
            if block["opcode"] != "data_itemoflist":
                return reject("a banked tensor uses a non-read list operation")
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

    # Resolve the carried hidden state from cross-layer data dependencies.
    layer_names = [
        {node.name for node in compiler.graph.nodes if _graph_layer(node) == layer}
        for layer in layers
    ]
    output_nodes = []
    for index, names in enumerate(layer_names):
        consumers = layer_names[index + 1] if index + 1 < len(layer_names) else {
            node.name for node in compiler.graph.nodes if _graph_layer(node) is None
        }
        used = {
            arg.name
            for consumer in compiler.graph.nodes if consumer.name in consumers
            for arg in consumer.all_input_nodes
            if arg.name in names and arg.name in compiler.scope.assignments
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
        if arg.name not in layer_names[0] and arg.name in compiler.scope.assignments
    ]
    if not first_inputs:
        return reject("could not identify the first layer input")
    input_list = f"T{compiler.scope.assignments[first_inputs[-1]]}"
    output_list = output_lists[0]
    log.info(
        "Repeated layer activation path: input=%s outputs=%s",
        input_list, output_lists,
    )

    predecessor = next(
        (identifier for identifier, block in blocks.items() if block.get("next") == starts[0]),
        None,
    )
    if predecessor is None:
        return reject("the repeated layer has no chain predecessor")
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
    body = []
    copy_index = "cattorch shared layer copy index"
    for layer in layers:
        body.extend(set_var(name, layer * size) for name, size in base_variables)
        body.append(call(procedure_name))
        if layer != layers[-1]:
            body.extend((
                clear(input_list),
                for_each(copy_index, length(output_list), (
                    append(input_list, item(output_list, var(copy_index))),
                )),
            ))
    wrapper = Program(
        "shared_layer_calls",
        variables=(copy_index, *(name for name, _ in base_variables)),
        lists=(input_list, output_list),
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

    final_id = next(
        identifier for identifier, entry in sprite["lists"].items()
        if entry[0] == output_lists[-1]
    )
    canonical_id = next(
        identifier for identifier, entry in sprite["lists"].items()
        if entry[0] == output_list
    )
    if final_id != canonical_id:
        sprite["blocks"] = remap_ids(sprite["blocks"], {final_id: canonical_id})
        _refresh_data_display_name(sprite["blocks"], canonical_id, output_list)
        sprite["lists"].pop(final_id, None)
    log.info(
        "Shared wrapper splice root=%s parent=%s top=%s predecessor_next=%s",
        wrapper_root,
        sprite["blocks"][wrapper_root].get("parent"),
        sprite["blocks"][wrapper_root].get("topLevel"),
        sprite["blocks"][predecessor].get("next"),
    )
    return True


def _merge_variables_by_name(sprite, names: set[str]) -> None:
    """Unify repeated DSL declarations of persistent generation variables."""
    variables = sprite.get("variables", {})
    ids_by_name: dict[str, list[str]] = {}
    for identifier, entry in variables.items():
        if entry[0] in names:
            ids_by_name.setdefault(entry[0], []).append(identifier)
    remap = {}
    for ids in ids_by_name.values():
        if len(ids) < 2:
            continue
        winner = ids[0]
        for sid in ids[1:]:
            remap[sid] = winner
            del variables[sid]
    if remap:
        sprite["blocks"] = remap_ids(sprite["blocks"], remap)


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

    for section in ("lists", "variables"):
        slots = sprite.get(section, {})
        unused = [
            sid for sid, entry in slots.items()
            if sid not in referenced
            and not entry[0].startswith("cattorch ")
        ]
        for sid in unused:
            log.info("Removing unused %s: %s", section.rstrip("s"), slots[sid][0])
            del slots[sid]


def _procedure_mutation(name: str, *, warp: bool = False) -> dict:
    mutation = {
        "tagName": "mutation",
        "children": [],
        "proccode": name,
        "argumentids": "[]",
    }
    if warp:
        mutation.update({
            "argumentnames": "[]",
            "argumentdefaults": "[]",
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
        raise ValueError(f"Expected one compiled root before lifecycle wrapping, got {roots}")

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

    call_base = {
        "next": None, "parent": None, "inputs": {}, "fields": {},
        "shadow": False, "topLevel": True,
    }
    blocks[forward_call] = {
        **call_base,
        "opcode": "procedures_call",
        "mutation": _procedure_mutation(forward_name),
        "x": 0, "y": 0,
    }
    blocks[init_call] = {
        **call_base,
        "opcode": "procedures_call",
        "mutation": _procedure_mutation(init_name),
        "x": 0, "y": 80,
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


def _add_warp_procedure(sprite: dict, name: str, program, *, x: int, y: int) -> dict:
    """Combine a DSL command stack and expose it as a no-argument warp block."""
    data = program.compile()
    roots = [
        bid for bid, block in data["blocks"].items()
        if block.get("topLevel") and block.get("parent") is None
    ]
    if len(roots) != 1:
        raise ValueError(f"Expected one root for procedure {name!r}")
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
        "mutation": _procedure_mutation(name, warp=True),
    }
    # Lifecycle wrapping gives the sprite several legitimate top-level roots,
    # whereas the ordinary block combiner expects one computation chain.  A
    # procedure is independent, so merge it directly after uniquifying every
    # generated ID; repeated list display names are unified afterward.
    mapping = {
        identifier: f"{identifier}_{suffix}"
        for section in ("blocks", "variables", "lists")
        for identifier in data.get(section, {})
    }
    data = remap_ids(data, mapping)
    sprite["blocks"].update(data.get("blocks", {}))
    sprite.setdefault("variables", {}).update(data.get("variables", {}))
    sprite.setdefault("lists", {}).update(data.get("lists", {}))
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
    raise ValueError(f"Procedure not found after generation wrapping: {name}")


def _add_procedure_completion_broadcast(
    sprite: dict,
    procedure: str,
    message: str,
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
        raise ValueError(f"Procedure {procedure!r} has no executable body")
    while blocks[terminal].get("next") is not None:
        terminal = blocks[terminal]["next"]

    notification = f"cattorch_completion_{uuid.uuid4().hex[:8]}"
    blocks[terminal]["next"] = notification
    blocks[notification] = {
        "opcode": "event_broadcastandwait",
        "next": None,
        "parent": terminal,
        "inputs": {
            "BROADCAST_INPUT": [1, [11, message, broadcast_id]],
        },
        "fields": {},
        "shadow": False,
        "topLevel": False,
    }


def _logical_list_sizes(
    compiler: _Compiler,
    graph: _GraphInfo,
    generation: GenerationConfig | None,
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


def _static_storage_precision(
    tensor: torch.Tensor | None,
    storage: StorageConfig,
) -> str:
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        return "float32"
    if storage.precision in {"int8", "int6", "int4"} and tensor.ndim < 2:
        return "float16"
    return storage.precision


def _static_storage_shard_limits(
    static_tensors: dict[str, torch.Tensor | None],
    storage: StorageConfig,
) -> dict[str, int]:
    """Cap static shards so every temporary decode stream fits in Scratch."""
    if not storage.compression:
        return {}
    limits: dict[str, int] = {}
    for key, tensor in static_tensors.items():
        precision = _static_storage_precision(tensor, storage)
        if precision == "float32":
            limit = SCRATCH_LIST_LIMIT // 4
        elif precision == "float16":
            limit = SCRATCH_LIST_LIMIT // 2
        else:
            scale_bytes = 4 if storage.scale_precision == "float32" else 2
            scale_groups = SCRATCH_LIST_LIMIT // scale_bytes
            limit = min(SCRATCH_LIST_LIMIT, scale_groups * storage.group_size)
            # A shard below the quantization threshold falls back to float16.
            # Keep every possible fallback shard within its byte budget too.
            if storage.min_quantized_values > SCRATCH_LIST_LIMIT // 2:
                limit = min(limit, SCRATCH_LIST_LIMIT // 2)
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
            raise ValueError(f"Static tensor {key!r} has unsupported complex values")
        original = tensor.detach().cpu()
        try:
            round_trip = original.to(scratch_dtype).to(original.dtype)
        except (RuntimeError, TypeError) as error:
            raise ValueError(
                f"Static tensor {key!r} cannot be represented by {label}"
            ) from error
        if not torch.equal(original, round_trip):
            suggestion = (
                "; use compression=False for integers exactly representable "
                "by Scratch doubles"
                if compressed else ""
            )
            raise ValueError(
                f"Static tensor {key!r} contains values not exactly "
                f"representable by {label}{suggestion}"
            )


def _apply_static_storage(
    sprite: dict,
    layouts: dict[str, ListLayout],
    static_tensors: dict[str, torch.Tensor | None],
    storage: StorageConfig,
) -> set[str]:
    """Encode static physical shards and install the one-time unpacker."""
    static_names = {f"W_{key}" for key in static_tensors}
    _validate_nonfloating_static_tensors(
        static_tensors, compressed=storage.compression,
    )
    if not storage.compression:
        if storage.precision in {"float16", "int8", "int6", "int4"}:
            for logical_name in static_names:
                tensor = static_tensors.get(logical_name.removeprefix("W_"))
                if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                    continue
                precision = _static_storage_precision(tensor, storage)
                for shard in layouts.get(logical_name, ListLayout(logical_name, 0, ())).shards:
                    contents = sprite["lists"][shard.identifier][1]
                    shard_precision = precision
                    if (
                        precision in {"int8", "int6", "int4"}
                        and len(contents) < storage.min_quantized_values
                    ):
                        shard_precision = "float16"
                    sprite["lists"][shard.identifier][1] = rounded_values(
                        contents,
                        shard_precision,
                        storage.group_size,
                        storage.scale_precision,
                    )
        return static_names

    specs = []
    for logical_name in sorted(static_names):
        layout = layouts.get(logical_name)
        if layout is None:
            continue
        tensor = static_tensors.get(logical_name.removeprefix("W_"))
        precision = _static_storage_precision(tensor, storage)
        for position, shard in enumerate(layout.shards, 1):
            contents = sprite["lists"][shard.identifier][1]
            if not contents:
                continue
            shard_precision = precision
            if (
                precision in {"int8", "int6", "int4"}
                and len(contents) < storage.min_quantized_values
            ):
                shard_precision = "float16"
            payload_name = f"cattorch payload {len(specs) + 1}"
            if shard_precision in {"int8", "int6", "int4"}:
                codes, scales = quantize_values(
                    contents,
                    shard_precision,
                    storage.group_size,
                    storage.scale_precision,
                )
                raw_payload = _quantized_bytes(codes, shard_precision)
                payload = _encode_bytes(raw_payload)
                scale_payload = encode_values(scales, storage.scale_precision)
                scale_number = len(specs) + 1
                specs.append(EncodedList(
                    shard.name,
                    payload_name,
                    payload,
                    shard_precision,
                    value_count=len(contents),
                    group_size=storage.group_size,
                    scale_list_name=f"cattorch quant scales {scale_number}",
                    scale_payload_name=f"cattorch scale payload {scale_number}",
                    scale_payload=scale_payload,
                    scale_precision=storage.scale_precision,
                    payload_bytes=raw_payload,
                    scale_values=tuple(scales),
                ))
            else:
                raw_payload = _float_bytes(contents, shard_precision)
                specs.append(EncodedList(
                    shard.name,
                    payload_name,
                    encode_values(contents, shard_precision),
                    shard_precision,
                    payload_bytes=raw_payload,
                ))
            sprite["lists"][shard.identifier][1] = []

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
        _insert_init_call(sprite, "cattorch unpack weights")
    return static_names


def _add_prepare_for_save(
    sprite: dict,
    layouts: dict[str, ListLayout],
    *,
    static_names: set[str],
    compressed: bool,
) -> None:
    """Add a public procedure that drops rebuildable runtime list contents."""
    from cattorch.util.scratch.dsl import Program, clear

    names = [
        shard.name
        for layout in layouts.values()
        if not layout.name.startswith("_")
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
    _merge_lists_by_name(sprite, set(names))

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


def _guard_generation_output(sprite: dict, output_root: str) -> None:
    """Make the final projection conditional for hidden-only prompt tokens."""
    blocks = sprite["blocks"]
    predecessors = [
        bid for bid, block in blocks.items() if block.get("next") == output_root
    ]
    if len(predecessors) != 1:
        raise ValueError(
            f"Expected one predecessor for output projection, got {predecessors}"
        )
    predecessor = predecessors[0]
    suffix = uuid.uuid4().hex[:8]
    guard = f"cattorch_hidden_prefill_{suffix}_guard"
    condition = f"cattorch_hidden_prefill_{suffix}_condition"
    variable_id = f"cattorch_hidden_prefill_{suffix}_project_output"
    sprite.setdefault("variables", {})[variable_id] = [
        "cattorch project output", 1,
    ]
    blocks[predecessor]["next"] = guard
    blocks[guard] = {
        "opcode": "control_if", "next": None, "parent": predecessor,
        "inputs": {
            "CONDITION": [2, condition],
            "SUBSTACK": [2, output_root],
        },
        "fields": {}, "shadow": False, "topLevel": False,
    }
    blocks[condition] = {
        "opcode": "operator_equals", "next": None, "parent": guard,
        "inputs": {
            "OPERAND1": [
                3,
                [12, "cattorch project output", variable_id],
                [10, ""],
            ],
            "OPERAND2": [1, [4, 1]],
        },
        "fields": {}, "shadow": False, "topLevel": False,
    }
    blocks[output_root]["parent"] = guard


def _guard_generation_suffix(sprite: dict, suffix_root: str) -> None:
    """Skip the final attention/MLP suffix after direct K/V cache emission."""
    blocks = sprite["blocks"]
    predecessors = [
        identifier
        for identifier, block in blocks.items()
        if block.get("next") == suffix_root
    ]
    if len(predecessors) != 1:
        raise ValueError(
            f"Expected one final QKV suffix predecessor, got {predecessors}"
        )
    predecessor = predecessors[0]
    variable_id = next(
        identifier
        for identifier, entry in sprite.get("variables", {}).items()
        if entry[0] == "cattorch project output"
    )
    token = uuid.uuid4().hex[:8]
    guard = f"cattorch_hidden_suffix_{token}_guard"
    condition = f"cattorch_hidden_suffix_{token}_condition"
    blocks[predecessor]["next"] = guard
    blocks[guard] = {
        "opcode": "control_if", "next": None, "parent": predecessor,
        "inputs": {
            "CONDITION": [2, condition],
            "SUBSTACK": [2, suffix_root],
        },
        "fields": {}, "shadow": False, "topLevel": False,
    }
    blocks[condition] = {
        "opcode": "operator_equals", "next": None, "parent": guard,
        "inputs": {
            "OPERAND1": [
                3,
                [12, "cattorch project output", variable_id],
                [10, ""],
            ],
            "OPERAND2": [1, [4, 1]],
        },
        "fields": {}, "shadow": False, "topLevel": False,
    }
    blocks[suffix_root]["parent"] = guard


_TOP_K_VALUES = "cattorch top k values"
_TOP_K_IDS = "cattorch top k ids"
_TOP_K_PROCEDURE = "cattorch select top k"


def _top_k_insertion(rank: int, top_k: int):
    """Shift a fixed sorted prefix and insert the current logit at ``rank``."""
    from cattorch.util.scratch.dsl import item, replace, sub, var

    shifts = []
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


def _build_top_k_program(top_k: int):
    """Select sorted top-k logits in one pass without mutating ``output``."""
    from cattorch.util.scratch.dsl import (
        Program, append, clear, for_each, gt, if_, item, length,
        set_var, var,
    )

    initialize = [clear(_TOP_K_VALUES), clear(_TOP_K_IDS)]
    for _ in range(top_k):
        initialize.extend((append(_TOP_K_VALUES, "-Infinity"), append(_TOP_K_IDS, -1)))
    return Program(
        "generation_top_k",
        variables=("top k index", "top k current"),
        lists=("output", _TOP_K_VALUES, _TOP_K_IDS),
        body=(
            *initialize,
            for_each("top k index", length("output"), (
                set_var("top k current", item("output", var("top k index"))),
                if_(
                    gt(var("top k current"), item(_TOP_K_VALUES, top_k)),
                    _top_k_decision(1, top_k),
                ),
            )),
        ),
    )


def _build_compact_top_k_program(top_k: int):
    """Select top-k with constant-size dynamic insertion loops."""
    from cattorch.util.scratch.dsl import (
        Program, append, change_var, clear, eq, gt, if_, if_else, item,
        for_each, length, lt, repeat, repeat_until, replace, set_var, sub, var,
    )

    initialize = [clear(_TOP_K_VALUES), clear(_TOP_K_IDS)]
    for _ in range(top_k):
        initialize.extend((append(_TOP_K_VALUES, "-Infinity"), append(_TOP_K_IDS, -1)))
    return Program(
        "generation_top_k_compact",
        variables=(
            "top k index", "top k current", "top k rank",
            "top k found", "top k shift",
        ),
        lists=("output", _TOP_K_VALUES, _TOP_K_IDS),
        body=(
            *initialize,
            for_each("top k index", length("output"), (
                set_var("top k current", item("output", var("top k index"))),
                set_var("top k rank", 1),
                set_var("top k found", 0),
                repeat_until(eq(var("top k found"), 1), (
                    if_else(
                        gt(
                            var("top k current"),
                            item(_TOP_K_VALUES, var("top k rank")),
                        ),
                        (set_var("top k found", 1),),
                        (
                            change_var("top k rank", 1),
                            if_(
                                gt(var("top k rank"), top_k),
                                (set_var("top k found", 1),),
                            ),
                        ),
                    ),
                )),
                if_(
                    lt(var("top k rank"), top_k + 1),
                    (
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

def _wrap_generation_lifecycle(
    sprite: dict,
    *,
    cache_names: list[str],
    first_cache: tuple[str, int],
    max_context: int,
    hidden_prefill: bool,
    top_k: int | None,
    compact_top_k: bool,
) -> dict:
    """Add reset/prefill/decode blocks and guard the cached forward procedure."""
    from cattorch.util.scratch.dsl import (
        Program, append, call, clear, eq, for_each, gt, if_else, item,
        length, set_var, sub, var,
    )

    top_k_lists = (_TOP_K_VALUES, _TOP_K_IDS) if top_k is not None else ()
    if top_k is not None:
        sprite = _add_warp_procedure(
            sprite,
            _TOP_K_PROCEDURE,
            (
                _build_compact_top_k_program(top_k)
                if compact_top_k
                else _build_top_k_program(top_k)
            ),
            x=640,
            y=480,
        )

    reset_lists = (*cache_names, "output", *top_k_lists)
    reset = Program(
        "generation_reset",
        lists=reset_lists,
        body=tuple(clear(name) for name in reset_lists),
    )
    sprite = _add_warp_procedure(
        sprite, "cattorch reset cache", reset, x=640, y=0,
    )

    invalid_decode = (
        clear("output"),
        *(clear(name) for name in top_k_lists),
        set_var("cattorch status", "decode requires exactly one token"),
    )
    decode = Program(
        "generation_decode",
        variables=("cattorch status",),
        lists=("input", "output", *top_k_lists),
        body=(
            if_else(
                eq(length("input"), 1),
                (call("cattorch forward"),),
                invalid_decode,
            ),
        ),
    )
    sprite = _add_warp_procedure(sprite, "cattorch decode", decode, x=640, y=160)

    copy_and_reset = (
            clear("cattorch prefill buffer"),
            for_each("prefill index", length("input"), (
                append(
                    "cattorch prefill buffer",
                    item("input", var("prefill index")),
                ),
            )),
            call("cattorch reset cache"),
            clear("input"),
    )
    if hidden_prefill:
        valid_prefill = (
                set_var("cattorch project output", 0),
                for_each("prefill index", sub(length("cattorch prefill buffer"), 1), (
                    append("input", item("cattorch prefill buffer", var("prefill index"))),
                    call("cattorch forward"),
                    clear("input"),
                )),
                set_var("cattorch project output", 1),
                append(
                    "input", item("cattorch prefill buffer", length("cattorch prefill buffer")),
                ),
                call("cattorch forward"),
                clear("input"),
        )
        prefill_body = (*copy_and_reset,
            if_else(
                eq(length("cattorch prefill buffer"), 0),
                (set_var("cattorch status", "prefill requires at least one token"),),
                (
                    if_else(
                        gt(length("cattorch prefill buffer"), max_context),
                        (
                            set_var(
                                "cattorch status",
                                "prefill exceeds maximum context",
                            ),
                        ),
                        valid_prefill,
                    ),
                ),
            ),
        )
        prefill_variables = (
            "prefill index", "cattorch project output", "cattorch status",
        )
    else:
        valid_prefill = (
            for_each("prefill index", length("cattorch prefill buffer"), (
                append("input", item("cattorch prefill buffer", var("prefill index"))),
                call("cattorch forward"),
                clear("input"),
            )),
        )
        prefill_body = (*copy_and_reset,
            if_else(
                eq(length("cattorch prefill buffer"), 0),
                (set_var("cattorch status", "prefill requires at least one token"),),
                (
                    if_else(
                        gt(length("cattorch prefill buffer"), max_context),
                        (
                            set_var(
                                "cattorch status",
                                "prefill exceeds maximum context",
                            ),
                        ),
                        valid_prefill,
                    ),
                ),
            ),
        )
        prefill_variables = ("prefill index", "cattorch status")
    prefill = Program(
        "generation_prefill",
        variables=prefill_variables,
        lists=("input", "cattorch prefill buffer"),
        body=prefill_body,
    )
    sprite = _add_warp_procedure(sprite, "cattorch prefill", prefill, x=640, y=320)
    if hidden_prefill:
        _merge_variables_by_name(sprite, {"cattorch project output"})
    if top_k_lists:
        _merge_lists_by_name(sprite, set(top_k_lists))

    # Insert a max-context guard after forward's automatic init check.
    blocks = sprite["blocks"]
    forward_def = _find_procedure_definition(sprite, "cattorch forward")
    init_check = blocks[forward_def]["next"]
    old_root = blocks[init_check]["next"]
    cache_name, cache_width = first_cache
    cache_id = next(
        sid for sid, entry in sprite["lists"].items() if entry[0] == cache_name
    )
    output_id = next(
        sid for sid, entry in sprite["lists"].items() if entry[0] == "output"
    )
    suffix = uuid.uuid4().hex[:8]
    prefix = f"cattorch_generation_guard_{suffix}"
    status_id = f"{prefix}_status"
    cache_length_id = f"{prefix}_cache_length"
    max_context_id = f"{prefix}_max_context"
    sprite.setdefault("variables", {})[status_id] = ["cattorch status", "ok"]
    sprite["variables"][cache_length_id] = ["cattorch cache length", 0]
    sprite["variables"][max_context_id] = ["cattorch max context", max_context]

    guard = f"{prefix}_guard"
    condition = f"{prefix}_condition"
    list_length = f"{prefix}_list_length"
    normalized_length = f"{prefix}_normalized_length"
    success = f"{prefix}_success"
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
        "inputs": {
            "OPERAND1": [3, normalized_length, [10, ""]],
            "OPERAND2": [1, [4, max_context]],
        },
        "fields": {}, "shadow": False, "topLevel": False,
    }
    blocks[normalized_length] = {
        "opcode": "operator_divide", "next": None, "parent": condition,
        "inputs": {
            "NUM1": [3, list_length, [4, 0]],
            "NUM2": [1, [4, cache_width]],
        },
        "fields": {}, "shadow": False, "topLevel": False,
    }
    blocks[list_length] = {
        "opcode": "data_lengthoflist", "next": None, "parent": normalized_length,
        "inputs": {}, "fields": {"LIST": [cache_name, cache_id]},
        "shadow": False, "topLevel": False,
    }
    blocks[success] = {
        "opcode": "data_setvariableto", "next": old_root, "parent": guard,
        "inputs": {"VALUE": [1, [10, "ok"]]},
        "fields": {"VARIABLE": ["cattorch status", status_id]},
        "shadow": False, "topLevel": False,
    }
    blocks[old_root]["parent"] = success
    blocks[failure] = {
        "opcode": "data_setvariableto", "next": clear_output, "parent": guard,
        "inputs": {"VALUE": [1, [10, "maximum context exceeded"]]},
        "fields": {"VARIABLE": ["cattorch status", status_id]},
        "shadow": False, "topLevel": False,
    }
    blocks[clear_output] = {
        "opcode": "data_deletealloflist", "next": None, "parent": failure,
        "inputs": {}, "fields": {"LIST": ["output", output_id]},
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

    # Update the public cache-length variable after a successful forward.
    terminal = old_root
    while blocks[terminal].get("next") is not None:
        terminal = blocks[terminal]["next"]
    set_cache_length = f"{prefix}_set_cache_length"
    length_reporter = f"{prefix}_final_length"
    normalized_reporter = f"{prefix}_final_normalized_length"
    blocks[terminal]["next"] = set_cache_length
    blocks[set_cache_length] = {
        "opcode": "data_setvariableto", "next": None, "parent": terminal,
        "inputs": {"VALUE": [3, normalized_reporter, [4, 0]]},
        "fields": {"VARIABLE": ["cattorch cache length", cache_length_id]},
        "shadow": False, "topLevel": False,
    }
    blocks[normalized_reporter] = {
        "opcode": "operator_divide", "next": None, "parent": set_cache_length,
        "inputs": {
            "NUM1": [3, length_reporter, [4, 0]],
            "NUM2": [1, [4, cache_width]],
        },
        "fields": {}, "shadow": False, "topLevel": False,
    }
    blocks[length_reporter] = {
        "opcode": "data_lengthoflist", "next": None, "parent": normalized_reporter,
        "inputs": {}, "fields": {"LIST": [cache_name, cache_id]},
        "shadow": False, "topLevel": False,
    }
    if top_k is not None:
        top_k_guard = f"{prefix}_top_k_guard"
        top_k_condition = f"{prefix}_top_k_condition"
        top_k_output_length = f"{prefix}_top_k_output_length"
        top_k_call = f"{prefix}_top_k_call"
        blocks[set_cache_length]["next"] = top_k_guard
        blocks[top_k_guard] = {
            "opcode": "control_if", "next": None, "parent": set_cache_length,
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
            "inputs": {}, "fields": {"LIST": ["output", output_id]},
            "shadow": False, "topLevel": False,
        }
        blocks[top_k_call] = {
            "opcode": "procedures_call", "next": None, "parent": top_k_guard,
            "inputs": {}, "fields": {}, "shadow": False, "topLevel": False,
            "mutation": _procedure_mutation(_TOP_K_PROCEDURE),
        }

    # Reset also resets public state variables.
    reset_def = _find_procedure_definition(sprite, "cattorch reset cache")
    reset_root = blocks[reset_def]["next"]
    reset_status = f"{prefix}_reset_status"
    reset_length = f"{prefix}_reset_length"
    blocks[reset_def]["next"] = reset_status
    blocks[reset_status] = {
        "opcode": "data_setvariableto", "next": reset_length, "parent": reset_def,
        "inputs": {"VALUE": [1, [10, "ok"]]},
        "fields": {"VARIABLE": ["cattorch status", status_id]},
        "shadow": False, "topLevel": False,
    }
    blocks[reset_length] = {
        "opcode": "data_setvariableto", "next": reset_root, "parent": reset_status,
        "inputs": {"VALUE": [1, [4, 0]]},
        "fields": {"VARIABLE": ["cattorch cache length", cache_length_id]},
        "shadow": False, "topLevel": False,
    }
    blocks[reset_root]["parent"] = reset_length
    _merge_variables_by_name(sprite, {"cattorch status"})
    return sprite
