"""Share the code of a transformer's repeated layers in one custom block.

When every ``blocks.N`` layer compiles to the same operations,
``_share_repeated_transformer_layers`` keeps layer zero's code as the warp
block ``cattorch shared transformer layer``, concatenates the weights that
differ between layers into banks, and calls the block once per layer with
the bank offsets (and, for banked K/V caches, the cache offset) set before
each call.
"""

from __future__ import annotations

import logging
import math
import re
import uuid

import numpy as np
import torch

from cattorch.procedures import _merge_variables_by_name, _procedure_mutation
from cattorch.storage import StorageConfig
from cattorch.util.scratch.dsl import (
    Program, append, call, clear, for_each, item, length, set_var, var,
)
from cattorch.util.scratch.remap import remap_ids
from cattorch.util.scratch.tensor_adder import StaticValues

log = logging.getLogger(__name__)


_LAYER_PATH = re.compile(r"(?:^|\.)blocks\.(\d+)(?:\.|$)")
# The offset of the current layer's rows in the banked K/V caches, which the
# shared layer's cached attention adds to its cache indices.
_LAYER_CACHE_BASE = "cattorch layer cache base"


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

    body: list = []
    copy_index = "cattorch shared layer copy index"
    cache_base = _LAYER_CACHE_BASE if compiler.graph.generation.banked_caches else None
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
