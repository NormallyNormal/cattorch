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
# The list a shared layer reads its input from when that input can't be
# overwritten between layers: a public input, or a value still read after
# the layers.
_LAYER_INPUT_COPY = "cattorch shared layer input"

# Why a graph's layers weren't shared, when it has nothing to share. A
# program warns about this only when none of its entry points has layers.
_SHARED_MODULES = (
    "(layer sharing shares submodules named `blocks.0`, `blocks.1`, ..., "
    "such as the items of an `nn.ModuleList` attribute called `blocks`)"
)
NO_REPEATED_LAYERS = f"it has no repeated layers to share {_SHARED_MODULES}"
NO_REPEATED_LAYERS_IN_PROGRAM = (
    f"none of its entry points has repeated layers to share {_SHARED_MODULES}"
)


def layer_sharing_warning(reason: str, entry_point: str | None = None) -> str:
    """The warning that ``layer_sharing=True`` didn't share layers, and why."""
    if entry_point is None:
        return (
            f"layer_sharing=True didn't share the model's layers: {reason}; "
            "it was exported without layer sharing"
        )
    return (
        f"layer_sharing=True didn't share the layers of entry point "
        f"{entry_point!r}: {reason}; that entry point was exported without "
        "layer sharing"
    )


def _layer_path(node) -> tuple[str, int] | None:
    """The ``blocks`` list a node's module sits in (its outermost), and its index."""
    for entry in reversed(tuple(node.meta.get("nn_module_stack", {}).values())):
        if not entry or not entry[0]:
            continue
        match = _LAYER_PATH.search(entry[0])
        if match:
            return entry[0][:match.start(1) - 1].lstrip("."), int(match.group(1))
    return None


def _graph_layer(node) -> int | None:
    path = _layer_path(node)
    return None if path is None else path[1]


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
    cache_reason: str | None = None,
) -> str | None:
    """Factor structurally identical ``blocks.N`` slabs into one warp block.

    Scratch cannot pass lists to custom blocks. The shared slab therefore uses
    the first layer's activation lists, copies only its final hidden state
    between calls, and selects concatenated weight banks through scalar bases.
    ``layer_tail`` is ``(gate, final)``: the gate variable is set to 1 before
    each call, and to the ``final`` variable's value before the last one.

    Returns ``None`` when the layers were shared, and otherwise why not, as
    a clause for a user-facing warning (``NO_REPEATED_LAYERS`` when there is
    nothing to share); the sprite is then unchanged. ``cache_reason`` is why
    a generation step's K/V caches couldn't be banked.
    """
    def reject(reason: str) -> str:
        log.info("Shared transformer layer skipped: %s", reason)
        return reason

    nodes = {node.name: node for node in compiler.graph.nodes}
    # Output copies follow the roots that compute nodes, under the last
    # node's name; they belong to no layer.
    emitted = [
        (name, root, _graph_layer(nodes[name]))
        for name, root in compiler.emitted_roots[:compiler.node_roots]
        if name in nodes
    ]
    layers = sorted({layer for _, _, layer in emitted if layer is not None})
    if len(layers) < 2:
        return reject(NO_REPEATED_LAYERS)
    lists = sorted({
        path[0] for path in map(_layer_path, compiler.graph.nodes) if path is not None
    })
    if len(lists) > 1:
        return reject(
            "its layers are in more than one `blocks` list "
            f"({', '.join(f'`{name}`' for name in lists)}), and only one can share"
        )
    if layers != list(range(len(layers))):
        return reject(
            "its `blocks.N` layers aren't numbered consecutively from `blocks.0` "
            f"(found {', '.join(f'`blocks.{layer}`' for layer in layers)})"
        )
    if (
        compiler.graph.generation.first_cache is not None
        and not compiler.graph.generation.banked_caches
    ):
        # Cached K/V lists are statically named in Scratch. Until attention is
        # split into thin per-layer cache wrappers, sharing the whole slab
        # would incorrectly direct every call to layer zero's cache.
        return reject(cache_reason or "its layers' K/V caches can't be shared")

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
    for layer, signature in zip(layers[1:], signatures[1:]):
        if signature != signatures[0]:
            return reject(
                f"layer `blocks.{layer}` doesn't compute the same operations as "
                "`blocks.0` (its shapes, constants or operations differ)"
            )

    blocks = sprite["blocks"]
    starts = [group[0][1] for group in grouped]
    emitted_roots = [root for _, root in compiler.emitted_roots]
    last_position = max(emitted_roots.index(root) for _, root in grouped[-1])
    successor = emitted_roots[last_position + 1] if last_position + 1 < len(emitted_roots) else None
    try:
        command_groups = [
            _chain_until(blocks, start, starts[index + 1] if index + 1 < len(starts) else successor)
            for index, start in enumerate(starts)
        ]
    except ValueError:
        return reject("code from outside the `blocks.N` layers runs between them")
    main_root = next(
        identifier for identifier, block in blocks.items()
        if block.get("topLevel") and block.get("parent") is None
        and block.get("opcode") != "procedures_definition"
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
    if not role_names[0]:
        return reject("its layers have no weights")
    if any(len(names) != len(role_names[0]) for names in role_names):
        log.info("Repeated layer static roles: %s", role_names)
        return reject(
            "its layers don't read the same number of distinct weight tensors "
            "(weights with identical values are stored once, so a layer whose "
            "weights repeat each other or another layer's reads fewer)"
        )

    bank_roles = []
    for role, names in enumerate(zip(*role_names)):
        if len(set(names)) == 1:
            continue
        entries = [sprite["lists"][list_ids[name]] for name in names]
        sizes = [len(entry[1]) for entry in entries]
        if len(set(sizes)) != 1:
            return reject("a weight tensor's stored size differs between layers")
        # Logical layer banks may exceed one physical Scratch list. The
        # ordinary precision-aware sharder runs after this transform and
        # rewrites indexed reads across row-aligned physical shards.
        bank_roles.append((role, names, sizes[0]))
    if not bank_roles:
        return reject("its layers have identical weights, so they have no per-layer weights to bank")
    banked = [name for _, names, _ in bank_roles for name in names]
    if len(set(banked)) != len(banked):
        # A deduplicated tensor read by only some layers, or in a different
        # role per layer, can't become one row of a bank.
        return reject(
            "a weight tensor with the same values is used by some layers but "
            "not others, or in a different place in different layers"
        )
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
                return reject(
                    "an operation in the layers uses a weight in a way other than "
                    "reading it item by item, so it can't read the weight from a "
                    "per-layer bank"
                )

    # Every call runs layer zero's code on layer zero's lists, and only one
    # tensor is carried from call to call. So each layer must read the
    # previous layer's output (layer zero: its input) and otherwise only
    # tensors from before the layers, in the same places as every other
    # layer, and only its output may be read outside it: by the next layer
    # alone, or, for the last layer, by the rest of the model.
    graph = compiler.graph

    def root(name: str) -> str:
        """The tensor whose list ``name`` is, through no-op aliases."""
        seen = set()
        while name in graph.aliases and name not in seen:
            seen.add(name)
            name = graph.aliases[name]
        return name

    def is_weight(name: str) -> bool:
        return (
            (name in nodes and nodes[name].op == "get_attr")
            or name in graph.generated_weights
            or graph.resolve_weight(name) is not None
        )

    position = {node.name: index for index, node in enumerate(graph.nodes)}
    layer_names = [
        {node.name for node in graph.nodes if _graph_layer(node) == layer}
        for layer in layers
    ]
    # Who reads each tensor; "" is the model's return.
    readers: dict[str, set[str]] = {}
    for node in graph.nodes:
        if node.name in graph.aliases:
            continue
        for argument in node.all_input_nodes:
            readers.setdefault(root(argument.name), set()).add(node.name)
    for name, _, _ in graph.outputs:
        readers.setdefault(root(name), set()).add("")

    layers_end = max(position[name] for name in layer_names[-1])
    output_nodes = []
    for index, members in enumerate(layer_names):
        layer = layers[index]
        last = index + 1 == len(layers)
        passed = sorted(
            (
                name for name in members
                if not is_weight(name) and readers.get(name, set()) - members
            ),
            key=position.__getitem__,
        )
        if not passed:
            return reject(
                f"layer `blocks.{layer}`'s output isn't read by the "
                + ("rest of the model" if last else "next layer")
            )
        if len(passed) > 1:
            return reject(
                f"more than one tensor of layer `blocks.{layer}` is read outside it "
                "(a shared layer passes on only its output)"
            )
        (output,) = passed
        # Who else reads it; the last layer's output is for the rest of the model.
        elsewhere = set() if last else readers[output] - members - layer_names[index + 1]
        if any(
            reader in nodes and _graph_layer(nodes[reader]) is None
            and position[reader] < layers_end
            for reader in elsewhere
        ):
            return reject("code from outside the `blocks.N` layers runs between them")
        if elsewhere:
            return reject(
                f"layer `blocks.{layer}`'s output is read after the next layer, "
                "such as by a skip connection (a shared layer's output is "
                "overwritten by the next call)"
            )
        if output not in compiler.scope.assignments:
            return reject(
                f"layer `blocks.{layer}`'s output is fused into the operation that "
                "reads it, so there is no list to pass between the calls"
            )
        output_nodes.append(output)

    def inputs_read(index: int, carried: str | None) -> list:
        """What each operation of a layer reads, by role."""
        own = {
            node.name: place
            for place, node in enumerate(node for node in graph.nodes
                                          if node.op == "call_function"
                                          and node.name in layer_names[index])
        }
        found: list = []
        for node in graph.nodes:
            if node.op != "call_function" or node.name not in layer_names[index]:
                continue
            for argument in node.all_input_nodes:
                name = root(argument.name)
                if is_weight(argument.name) or is_weight(name):
                    found.append("weight")
                elif name in own:
                    found.append(own[name])
                elif name == carried:
                    found.append("carried")
                else:
                    found.append(("before", name))
        return found

    expected = inputs_read(1, output_nodes[0])
    first_reads = inputs_read(0, None)
    entering = {
        role for role, wanted in zip(first_reads, expected) if wanted == "carried"
    } if len(first_reads) == len(expected) else set()
    first_input = None
    if len(entering) == 1 and isinstance(role := next(iter(entering)), tuple):
        first_input = role[1]
    for index, layer in enumerate(layers):
        carried_in = first_input if index == 0 else output_nodes[index - 1]
        if carried_in is None or inputs_read(index, carried_in) != expected:
            return reject(
                f"layer `blocks.{layer}` reads different tensors from the other "
                "layers (each layer may read only the previous layer's output, "
                "and the same tensors from before the layers)"
            )
    assert first_input is not None

    def list_of(name: str) -> str | None:
        if name in compiler.scope.assignments:
            return f"T{compiler.scope.assignments[name]}"
        return graph.input_names.get(name)

    for name in (first_input, *(role[1] for role in expected if isinstance(role, tuple))):
        if list_of(name) is None and isinstance(nodes[name].meta.get("val"), torch.Tensor):
            return reject(
                "layer `blocks.0` reads a tensor from before the layers that is "
                "fused into its code, so the other layers can't share that code"
            )
    input_list = list_of(first_input)
    if input_list is None:
        return reject("layer `blocks.0`'s input has no list of its own")
    output_lists = [f"T{compiler.scope.assignments[name]}" for name in output_nodes]
    output_list = output_lists[0]
    # The calls copy each layer's output into the first layer's input list.
    # A public input list must keep the caller's values, and a value read
    # again after the layers must survive them: the first layer then reads
    # a private copy instead.
    first_position = min(position[name] for name in layer_names[0])
    copy_input = first_input in graph.input_names or any(
        reader == "" or (reader not in layer_names[0] and position[reader] > first_position)
        for reader in readers.get(first_input, ())
    )
    # Without a predecessor the first layer starts the script, and the calls
    # take its place there.
    predecessor = next(
        (identifier for identifier, block in blocks.items() if block.get("next") == starts[0]),
        None,
    )
    if copy_input and input_list not in list_ids:
        return reject("layer `blocks.0`'s input list wasn't found")

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

    carried = input_list
    if copy_input:
        carried = _LAYER_INPUT_COPY
        source_id = list_ids[input_list]
        carried_id = f"cattorch_shared_layer_input_{uuid.uuid4().hex[:8]}"
        sprite["lists"][carried_id] = [carried, []]
        input_node = nodes.get(first_input)
        shape = tuple(input_node.meta["val"].shape) if input_node is not None else ()
        capacity = (
            compiler.graph.runtime_capacity(first_input, shape)
            if first_input in compiler.graph.dynamic_axes
            else math.prod(shape)
        )
        compiler.logical_sizes[carried] = max(compiler.logical_sizes.get(carried, 0), capacity)

        def point_at_copy(value):
            if isinstance(value, list):
                if len(value) >= 3 and value[0] == 13 and value[2] == source_id:
                    value[1:3] = [carried, carried_id]
                for part in value:
                    point_at_copy(part)

        for identifier in layer_zero_owned:
            block = blocks.get(identifier)
            if block is None:
                continue
            for field in block.get("fields", {}).values():
                if isinstance(field, list) and len(field) >= 2 and field[1] == source_id:
                    field[0:2] = [carried, carried_id]
            for value in block.get("inputs", {}).values():
                point_at_copy(value)

    log.info(
        "Repeated layer activation path: input=%s carried=%s outputs=%s",
        input_list, carried, output_lists,
    )

    first_end = command_groups[0][-1]
    first_position_xy = {
        key: blocks[starts[0]][key] for key in ("x", "y") if key in blocks[starts[0]]
    }
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

    def copy_list(target: str, source: str) -> tuple:
        return (
            clear(target),
            for_each(copy_index, length(source), (
                append(target, item(source, var(copy_index))),
            )),
        )

    if carried != input_list:
        body.extend(copy_list(carried, input_list))
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
        if layer != layers[-1] and carried != output_list:
            body.extend(copy_list(carried, output_list))
    final_output_list = output_lists[-1]
    if final_output_list != output_list:
        body.extend(copy_list(final_output_list, output_list))
    wrapper = Program(
        "shared_layer_calls",
        variables=(
            copy_index,
            *(name for name, _ in base_variables),
            *((cache_base,) if cache_base is not None else ()),
            *(layer_tail or ()),
        ),
        lists=tuple(dict.fromkeys((input_list, carried, output_list, final_output_list))),
        body=tuple(body),
    )
    wrapper_root, wrapper_tail = _program_stack(sprite, wrapper)
    # `_program_stack` may replace its temporary block dictionary while
    # remapping data IDs; refresh the shared reference before splicing.
    blocks = sprite["blocks"]
    if predecessor is None:
        # The calls start the script, where the first layer was.
        blocks[wrapper_root]["parent"] = None
        blocks[wrapper_root]["topLevel"] = True
        blocks[wrapper_root].pop("x", None)
        blocks[wrapper_root].pop("y", None)
        blocks[wrapper_root].update(first_position_xy)
    else:
        blocks[predecessor]["next"] = wrapper_root
        blocks[wrapper_root]["parent"] = predecessor
        blocks[wrapper_root]["topLevel"] = False
        blocks[wrapper_root].pop("x", None)
        blocks[wrapper_root].pop("y", None)
    blocks[wrapper_tail]["next"] = successor
    if successor in blocks:
        blocks[successor]["parent"] = wrapper_tail

    log.info(
        "Shared wrapper splice root=%s parent=%s top=%s",
        wrapper_root,
        sprite["blocks"][wrapper_root].get("parent"),
        sprite["blocks"][wrapper_root].get("topLevel"),
    )
    if cache_base is not None:
        _merge_variables_by_name(sprite, {cache_base})
    if layer_tail is not None:
        _merge_variables_by_name(sprite, set(layer_tail))
    return None
