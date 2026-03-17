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


def _get_shape(arg) -> torch.Size:
    if hasattr(arg, 'meta') and 'val' in arg.meta:
        return arg.meta['val'].shape
    return torch.Size([])


def _resolve_args(node, scope, input_names, resolve_weight):
    input_lists = []
    dynamic_lists = set()
    static_lists = {}
    constants = []

    for arg in node.args:
        if hasattr(arg, 'name'):
            if arg.name in scope.assignments:
                name = f"T{scope.assignments[arg.name]}"
                input_lists.append(name)
                dynamic_lists.add(name)
            elif arg.name in input_names:
                input_lists.append(input_names[arg.name])
            else:
                name = f"W_{arg.name}"
                input_lists.append(name)
                static_lists[arg.name] = resolve_weight(arg.name)
        else:
            input_lists.append(f"C_{arg}")
            constants.append(arg)

    return input_lists, dynamic_lists, static_lists, constants


def _process_instruction(instruction, input_lists, target_list, block_manager):
    data = instruction.finalize()

    block_manager.new_context()
    data["blocks"] = block_manager.apply_to_blocks(data["blocks"])

    tensor_adder = TensorAdder()
    all_lists = input_lists + [target_list]
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
    nodes = list(exported.graph.nodes)
    normalised_state = {k.lower(): v for k, v in exported.state_dict.items()}

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

    scope = ScopeManager()
    scope.analyze_lifetimes(nodes)

    all_dynamic_lists = set()
    all_static_lists = {}
    all_constants = []

    scratch_output = None
    block_manager = BlockManager()
    last_target_list = None

    for node in nodes:
        if node.op != 'call_function':
            continue

        target_list = scope.get_list_for_node(node)
        last_target_list = target_list
        aten_op = str(node.target)

        input_lists, dynamic_lists, static_lists, constants = _resolve_args(
            node, scope, input_names, resolve_weight
        )

        all_dynamic_lists.update(dynamic_lists)
        all_static_lists.update(static_lists)
        all_constants.extend(constants)

        log.info("%s -> %s (%s) (Inputs: %s)", aten_op, target_list, node.name, input_lists)

        args = [
            Argument(input_lists[i], _get_shape(node.args[i]))
            for i in range(len(input_lists))
        ]
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
