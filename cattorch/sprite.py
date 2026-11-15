"""Scratch sprite assembly, lifecycle, storage, and generation wiring."""

from __future__ import annotations

import logging
import math
import re
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, AbstractSet

import numpy as np
import torch

from cattorch.graph import _GenerationSpec, _GraphInfo, _get_shape
from cattorch.guards import _EPOCH, _epoch_current
from cattorch.procedures import _add_warp_procedure, _merge_variables_by_name, _procedure_mutation
from cattorch.storage import StorageConfig
from cattorch.util.scratch.interface import rename_procedure as _rename_procedure
from cattorch.util.scratch.remap import remap_ids
from cattorch.util.scratch.sharding import ListLayout
from cattorch.util.scratch.tensor_adder import StaticValues
from cattorch import interface
from cattorch.interface import STATUS_VARIABLE

if TYPE_CHECKING:
    from cattorch.transpiler import _Compiler

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
            and not entry[0].startswith("cattorch ")
            and not is_public_data(entry[0], is_list=section == "lists")
        ]
        for sid in unused:
            log.info("Removing unused %s: %s", section.rstrip("s"), slots[sid][0])
            del slots[sid]


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
        sizes[interface.PREFILL_BUFFER] = generation.max_context
        if generation.top_k is not None:
            sizes[interface.TOP_K_VALUES] = generation.top_k
            sizes[interface.TOP_K_IDS] = generation.top_k
    for name, size in compiler.logical_sizes.items():
        sizes[name] = max(sizes.get(name, 0), size)
    return sizes


def _add_prepare_for_save(
    sprite: dict,
    layouts: dict[str, ListLayout],
    *,
    static_names: set[str],
    compressed: bool,
    inputs: AbstractSet[str] = frozenset(),
) -> None:
    """Add the public procedure that drops rebuildable runtime list contents.

    It marks the weights undecoded and clears the decoded weights and the
    working lists. The ``inputs`` lists are kept: they hold the caller's
    values, or are public lists, and a program's persistent state is passed
    to its entry points as inputs.
    """
    from cattorch.util.scratch.dsl import Program, clear, set_var

    names = [
        shard.name
        for layout in layouts.values()
        if not layout.name.startswith("_") and layout.name not in inputs
        and (compressed or layout.name not in static_names)
        for shard in layout.shards
    ]
    initialized = interface.INITIALIZED_VARIABLE
    program = Program(
        "prepare_for_save",
        variables=(initialized,),
        lists=names,
        body=(set_var(initialized, 0), *(clear(name) for name in names)),
    )
    _add_warp_procedure(sprite, interface.PREPARE_FOR_SAVE, program, x=640, y=640)
    _merge_variables_by_name(sprite, {initialized})


def _gate_chain(
    sprite: dict, start: str, gate: str = interface.PROJECT_OUTPUT, stop: str | None = None,
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


# What ``cattorch forward`` becomes once a wrapper checks its inputs first.
_CHECKED_FORWARD = "_cattorch checked forward"


def _add_input_checks(sprite: dict, checks: Sequence[Sequence], lists: Sequence[str]) -> None:
    """Make ``cattorch forward`` set the status, and run only if ``checks`` leave it ok.

    ``checks`` are groups of statements, each run only while the status is
    still ok, so that it reports the first problem. The forward becomes the
    internal ``_cattorch checked forward``. ``lists`` are the lists the
    checks read.
    """
    from cattorch.util.scratch.dsl import Program, call, eq, if_, set_var, var

    ok = eq(var(STATUS_VARIABLE), interface.OK)
    _rename_procedure(sprite, interface.FORWARD, _CHECKED_FORWARD)
    wrapper = Program(
        "checked_forward",
        variables=(STATUS_VARIABLE, "_cattorch check index", _EPOCH),
        lists=tuple(dict.fromkeys(lists)),
        body=(
            set_var(STATUS_VARIABLE, interface.OK),
            *(if_(ok, tuple(group)) for group in checks),
            # A forward taken over while it checked computes nothing; the
            # forward later runs in an epoch procedure (_guard_plain_calls).
            if_(ok, (if_(_epoch_current(), (call(_CHECKED_FORWARD),)),)),
        ),
    )
    _add_warp_procedure(sprite, interface.FORWARD, wrapper, x=640, y=960)
    # The loose forward block on the canvas runs the public block, not the
    # procedure it wraps.
    for block in sprite["blocks"].values():
        if (
            block.get("opcode") == "procedures_call" and block.get("topLevel")
            and block["mutation"].get("proccode") == _CHECKED_FORWARD
        ):
            block["mutation"] = _procedure_mutation(interface.FORWARD)

