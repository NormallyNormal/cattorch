import json
import logging
import uuid

import torch
from torch.export import export

from cattorch.util.argument import Argument
from cattorch.util.instruction import Instruction
from cattorch.util.scope import ScopeManager
from cattorch.util.scratch.block_combiner import combine
from cattorch.util.scratch.block_manager import BlockManager
from cattorch.util.scratch.finalize_scratch import finalize_sprite
from cattorch.util.scratch.tensor_adder import TensorAdder
from cattorch.util.scratch.tensor_replacer import TensorReplacer

log = logging.getLogger(__name__)

# Ops that change logical shape but don't move data in the flat array
_NOOP_OPS = {
    "aten.view.default",
    "aten.reshape.default",
    "aten._unsafe_view.default",
    "aten.contiguous.default",
    "aten.clone.default",
}


def _decompose_linear(graph):
    """Rewrite aten.linear.default nodes into transpose + matmul + optional add."""
    for node in list(graph.nodes):
        if node.op != 'call_function' or str(node.target) != 'aten.linear.default':
            continue

        args = node.args
        x, weight = args[0], args[1]
        has_bias = len(args) == 3

        with graph.inserting_before(node):
            # Transpose weight: [out, in] → [in, out]
            tp = graph.create_node('call_function', torch.ops.aten.numpy_T.default, (weight,))
            tp.meta = {'val': torch.empty(torch.Size(reversed(weight.meta['val'].shape)))}

            # Matmul: x @ weight.T
            mm = graph.create_node('call_function', torch.ops.aten.matmul.default, (x, tp))
            mm_shape = list(x.meta['val'].shape[:-1]) + [weight.meta['val'].shape[0]]
            mm.meta = {'val': torch.empty(mm_shape)}

            if has_bias:
                bias = args[2]
                add = graph.create_node('call_function', torch.ops.aten.add.Tensor, (mm, bias))
                add.meta = {'val': torch.empty(mm_shape)}
                replacement = add
            else:
                replacement = mm

        node.replace_all_uses_with(replacement)
        graph.erase_node(node)


def _get_shape(arg) -> torch.Size:
    if hasattr(arg, 'meta') and 'val' in arg.meta:
        return arg.meta['val'].shape
    return torch.Size([])


def _resolve_args(node, scope, input_names, resolve_weight, aliases=None):
    input_lists = []
    dynamic_lists = set()
    static_lists = {}
    constants = []

    for arg in node.args:
        if hasattr(arg, 'name'):
            arg_name = arg.name
            if aliases and arg_name in aliases:
                arg_name = aliases[arg_name]
            if arg_name in scope.assignments:
                name = f"T{scope.assignments[arg_name]}"
                input_lists.append(name)
                dynamic_lists.add(name)
            elif arg_name in input_names:
                input_lists.append(input_names[arg_name])
            else:
                name = f"W_{arg_name}"
                input_lists.append(name)
                static_lists[arg_name] = resolve_weight(arg_name)
        else:
            input_lists.append(f"C_{arg}")
            constants.append(arg)

    return input_lists, dynamic_lists, static_lists, constants


def _process_instruction(instruction, input_lists, target_list, block_manager):
    data = instruction.finalize()

    block_manager.new_context()
    data["blocks"] = block_manager.apply_to_blocks(data["blocks"])

    # Only tensor args go through list management; constants are compile-time only
    tensor_lists = [name for name in input_lists if not name.startswith("C_")]

    tensor_adder = TensorAdder()
    all_lists = tensor_lists + [target_list]
    data = tensor_adder.apply(data, all_lists)

    tensor_replacer = TensorReplacer(data, all_lists)
    data["blocks"] = tensor_replacer.apply(data["blocks"])

    return data


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
    # Replace the display name in LIST fields: ["old_name", "id"] -> ["new_name", "id"]
    raw = raw.replace(
        json.dumps([old_name, target_id]),
        json.dumps([new_name, target_id]),
    )
    sprite["blocks"] = json.loads(raw)


def _remove_unused(sprite):
    """Remove lists and variables that are not referenced by any block."""
    raw = json.dumps(sprite["blocks"])

    for section in ("lists", "variables"):
        slots = sprite.get(section, {})
        unused = [sid for sid in slots if f'"{sid}"' not in raw]
        for sid in unused:
            log.info("Removing unused %s: %s", section.rstrip("s"), slots[sid][0])
            del slots[sid]


def _uniquify_ids(sprite):
    """Add a UUID suffix to all list and variable IDs to prevent conflicts
    when multiple cattorch sprites are loaded into one Scratch project."""
    suffix = uuid.uuid4().hex[:12]
    remap = {}

    for section in ("lists", "variables"):
        slots = sprite.get(section, {})
        for sid in list(slots):
            new_id = f"{sid}_{suffix}"
            remap[sid] = new_id
            slots[new_id] = slots.pop(sid)

    if not remap:
        return

    raw = json.dumps(sprite["blocks"])
    for old, new in sorted(remap.items(), key=lambda x: -len(x[0])):
        raw = raw.replace(f'"{old}"', f'"{new}"')
    sprite["blocks"] = json.loads(raw)


def transpile(model: torch.nn.Module, input_shape: torch.Size, sprite_name: str):
    exported = export(model, (torch.randn(input_shape),))
    _decompose_linear(exported.graph)
    nodes = list(exported.graph.nodes)
    normalised_state = {k.lower().replace(".", "_"): v for k, v in exported.state_dict.items()}

    def resolve_weight(arg_name: str):
        key = arg_name.removeprefix("p_")
        return normalised_state.get(key)

    # Identify placeholder nodes: those not in state_dict are model inputs
    input_names = {}
    input_count = 0
    for node in nodes:
        if node.op == 'placeholder' and resolve_weight(node.name) is None:
            name = "input" if input_count == 0 else f"input_{input_count}"
            input_names[node.name] = name
            input_count += 1

    # Build alias map for no-op shape ops (view, reshape, contiguous, etc.)
    # These don't move data, so their output is the same list as their input.
    aliases = {}  # no-op node name -> source node name
    for node in nodes:
        if node.op != 'call_function':
            continue
        aten_op = str(node.target)
        if aten_op in _NOOP_OPS and hasattr(node.args[0], 'name'):
            source = node.args[0].name
            # Follow alias chains: if source is itself aliased, point to the root
            while source in aliases:
                source = aliases[source]
            aliases[node.name] = source
            log.info("Alias: %s -> %s (no-op %s)", node.name, source, aten_op)

    scope = ScopeManager()
    scope.analyze_lifetimes(nodes, skip=set(aliases.keys()))

    all_dynamic_lists = set()
    all_static_lists = {}
    all_constants = []

    scratch_output = None
    block_manager = BlockManager()
    last_target_list = None

    for node in nodes:
        if node.op != 'call_function':
            continue

        # Skip no-op nodes — they are aliases, not real computations
        if node.name in aliases:
            continue

        target_list = scope.get_list_for_node(node)
        last_target_list = target_list
        aten_op = str(node.target)

        input_lists, dynamic_lists, static_lists, constants = _resolve_args(
            node, scope, input_names, resolve_weight, aliases
        )

        all_dynamic_lists.update(dynamic_lists)
        all_static_lists.update(static_lists)
        all_constants.extend(constants)

        log.info("%s -> %s (%s) (Inputs: %s)", aten_op, target_list, node.name, input_lists)

        args = []
        for i in range(len(input_lists)):
            node_arg = node.args[i]
            if hasattr(node_arg, 'name'):
                args.append(Argument(input_lists[i], _get_shape(node_arg)))
            else:
                args.append(Argument(input_lists[i], torch.Size([]), value=node_arg))
        instruction = Instruction.create(aten_op, node.target, target_list, *args)

        data = _process_instruction(instruction, input_lists, target_list, block_manager)

        if scratch_output is None:
            scratch_output = data
        else:
            scratch_output = combine(scratch_output, data)

        scope.release_dependencies(node)

    # Add static weight lists with preloaded data
    tensor_adder = TensorAdder()
    scratch_output = tensor_adder.apply(
        scratch_output,
        [f"W_{k}" for k in all_static_lists.keys()],
        weights={f"W_{k}": v for k, v in all_static_lists.items() if v is not None}
    )

    # Add input lists (empty — populated at runtime)
    scratch_output = tensor_adder.apply(
        scratch_output,
        list(input_names.values()),
    )

    # If the last graph node is a no-op alias, resolve to the source list
    last_call = None
    for node in reversed(nodes):
        if node.op == 'call_function':
            last_call = node
            break
    if last_call and last_call.name in aliases:
        source = aliases[last_call.name]
        if source in scope.assignments:
            last_target_list = f"T{scope.assignments[source]}"
        elif source in input_names:
            last_target_list = input_names[source]

    # Rename the final output list
    if last_target_list is not None:
        _rename_list(scratch_output, last_target_list, "output")

    # Remove orphaned lists and variables
    _remove_unused(scratch_output)

    # Add unique suffixes to all internal list/variable IDs to avoid
    # conflicts when multiple cattorch sprites are loaded into one project
    _uniquify_ids(scratch_output)

    log.info("Dynamic lists: %s", list(all_dynamic_lists))
    log.info("Static lists: %s", list(all_static_lists.keys()))
    log.info("Constants: %s", list(all_constants))

    finalize_sprite(scratch_output, f"{sprite_name}.sprite3", sprite_name=sprite_name)
