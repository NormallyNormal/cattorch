import json
import logging
import math
import operator
import uuid

import torch
from torch.export import export

from cattorch.util.argument import Argument
from cattorch.util.instruction import Instruction
from cattorch.util.instruction.cat import CatInstruction
from cattorch.util.instruction.getitem import GetItemInstruction
from cattorch.util.scope import ScopeManager
from cattorch.util.scratch.block_combiner import combine
from cattorch.util.scratch.block_manager import BlockManager
from cattorch.util.scratch.finalize_scratch import finalize_sprite
from cattorch.util.scratch.remap import remap_ids
from cattorch.util.scratch.tensor_adder import TensorAdder
from cattorch.util.scratch.tensor_replacer import TensorReplacer

log = logging.getLogger(__name__)

# Ops that change logical shape but don't move data in the flat array
_NOOP_OPS = {
    "aten.view.default",
    "aten.reshape.default",
    "aten._unsafe_view.default",
    "aten.flatten.using_ints",
    "aten.contiguous.default",
    "aten.clone.default",
}

# Ops that split a tensor into multiple outputs. The split itself is a no-op;
# each getitem on the result becomes the real copy operation.
_SPLIT_OPS = {
    "aten.split.Tensor",
    "aten.split_with_sizes.default",
    "aten.chunk.default",
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


def _merge_duplicate_lists(sprite):
    """Merge local lists (names starting with '_') that have identical contents.

    When multiple instructions produce identical data lists (e.g. two transposes
    with the same index map), we can share a single list and remap all block
    references to it.
    """
    lists = sprite.get("lists", {})

    # Group local lists by their contents
    by_content: dict[tuple, list[tuple[str, str]]] = {}  # (content_tuple) -> [(sid, display_name), ...]
    for sid, entry in lists.items():
        display_name = entry[0]
        if not display_name.startswith("_"):
            continue
        content_key = tuple(entry[1])
        by_content.setdefault(content_key, []).append((sid, display_name))

    # Build remap for groups with more than one list
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

    # Renumber local list display names sequentially
    base_counts: dict[str, int] = {}
    for entry in lists.values():
        name = entry[0]
        if not name.startswith("_"):
            continue
        # Strip existing suffix: "_index_map_4" -> "_index_map"
        base = name
        for i in range(len(name) - 1, 0, -1):
            if name[i] == '_' and name[i+1:].isdigit():
                base = name[:i]
                break
        count = base_counts.get(base, 0) + 1
        base_counts[base] = count
        entry[0] = base if count == 1 else f"{base}_{count}"


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
    mapping = {}

    for section in ("lists", "variables"):
        slots = sprite.get(section, {})
        for sid in list(slots):
            new_id = f"{sid}_{suffix}"
            mapping[sid] = new_id
            slots[new_id] = slots.pop(sid)

    if mapping:
        sprite["blocks"] = remap_ids(sprite["blocks"], mapping)


def transpile(model: torch.nn.Module, example_inputs: torch.Tensor | tuple[torch.Tensor, ...], sprite_name: str, sig_figs: int | None = None):
    if isinstance(example_inputs, torch.Tensor):
        example_inputs = (example_inputs,)
    exported = export(model, example_inputs)
    _decompose_linear(exported.graph)
    nodes = list(exported.graph.nodes)
    normalised_state = {k.lower().replace(".", "_"): v for k, v in exported.state_dict.items()}

    def resolve_weight(arg_name: str):
        key = arg_name.removeprefix("p_").removeprefix("b_")
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
    split_meta = {}  # split node name -> (input_shape, split_size, dim)
    for node in nodes:
        if node.op != 'call_function':
            continue
        aten_op = str(node.target)
        if aten_op in _NOOP_OPS and hasattr(node.args[0], 'name'):
            source = node.args[0].name
            while source in aliases:
                source = aliases[source]
            aliases[node.name] = source
            log.info("Alias: %s -> %s (no-op %s)", node.name, source, aten_op)
        elif aten_op in _SPLIT_OPS:
            source = node.args[0].name
            while source in aliases:
                source = aliases[source]
            aliases[node.name] = source
            input_shape = _get_shape(node.args[0])
            if aten_op == "aten.chunk.default":
                num_chunks = node.args[1]
                dim = node.args[2] if len(node.args) > 2 else 0
                if dim < 0:
                    dim = len(input_shape) + dim
                split_size = math.ceil(input_shape[dim] / num_chunks)
            else:
                split_size = node.args[1]
                dim = node.args[2] if len(node.args) > 2 else 0
                if dim < 0:
                    dim = len(input_shape) + dim
            split_meta[node.name] = (input_shape, split_size, dim)
            log.info("Split: %s -> %s (split_size=%s, dim=%s)", node.name, source, split_size, dim)

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

        # Handle cat (first arg is a list of tensors, not a single tensor)
        if aten_op == "aten.cat.default":
            tensor_list = node.args[0]
            dim = node.args[1] if len(node.args) > 1 else 0

            def _resolve_tensor(t):
                t_name = t.name
                if aliases and t_name in aliases:
                    t_name = aliases[t_name]
                if t_name in scope.assignments:
                    name = f"T{scope.assignments[t_name]}"
                    all_dynamic_lists.add(name)
                elif t_name in input_names:
                    name = input_names[t_name]
                else:
                    name = f"W_{t_name}"
                    all_static_lists[t_name] = resolve_weight(t_name)
                return name

            def _cat_shapes(s1, s2, d):
                """Compute the output shape of cat(s1, s2, dim=d)."""
                out = list(s1)
                out[d] = s1[d] + s2[d]
                return torch.Size(out)

            resolved = [(_resolve_tensor(t), _get_shape(t)) for t in tensor_list]

            if dim < 0:
                dim = len(resolved[0][1]) + dim

            # Chain pairwise: cat(a,b)→tmp, cat(tmp,c)→tmp2, ... cat(tmpN,last)→target
            cur_list, cur_shape = resolved[0]
            for pair_idx in range(1, len(resolved)):
                next_list, next_shape = resolved[pair_idx]
                is_last = pair_idx == len(resolved) - 1

                if is_last:
                    cat_target = target_list
                else:
                    # Allocate a temporary list from the scope
                    if scope.free_pool:
                        scope.free_pool.sort()
                        temp_id = scope.free_pool.pop(0)
                    else:
                        scope.peak_lists += 1
                        temp_id = scope.peak_lists
                    cat_target = f"T{temp_id}"
                    all_dynamic_lists.add(cat_target)

                num_outer = math.prod(cur_shape[:dim]) if dim > 0 else 1
                chunk_1 = math.prod(cur_shape[dim:])
                chunk_2 = math.prod(next_shape[dim:])
                cat_inputs = [cur_list, next_list]

                log.info("cat(dim=%d) [%d/%d] -> %s (%s) (Inputs: %s)",
                         dim, pair_idx, len(resolved) - 1, cat_target, node.name, cat_inputs)

                cat_args = [
                    Argument(cur_list, cur_shape),
                    Argument(next_list, next_shape),
                    Argument("C_num_outer", torch.Size([]), value=num_outer),
                    Argument("C_chunk_1", torch.Size([]), value=chunk_1),
                    Argument("C_chunk_2", torch.Size([]), value=chunk_2),
                ]
                cat_instr = CatInstruction(node.target, cat_target, *cat_args)
                data = _process_instruction(cat_instr, cat_inputs, cat_target, block_manager)

                if scratch_output is None:
                    scratch_output = data
                else:
                    scratch_output = combine(scratch_output, data)

                # The intermediate becomes the left input for the next pair
                out_shape = _cat_shapes(cur_shape, next_shape, dim)
                if not is_last:
                    prev_temp_id = temp_id
                cur_list, cur_shape = cat_target, out_shape

                # Free the intermediate after it's consumed (except the final target)
                if pair_idx > 1:
                    scope.free_pool.append(prev_temp_id)

            # Skip the shared _process_instruction below — already handled inline
            scope.release_dependencies(node)
            continue

        # Handle slice as a getitem-style copy (or no-op if full range)
        elif aten_op == "aten.slice.Tensor":
            source = node.args[0]
            dim = node.args[1] if len(node.args) > 1 else 0
            start = node.args[2] if len(node.args) > 2 else 0
            end = node.args[3] if len(node.args) > 3 else None
            input_shape = _get_shape(source)

            if dim < 0:
                dim = len(input_shape) + dim

            # Clamp end to actual size
            dim_size = input_shape[dim]
            if end is None or end > dim_size:
                end = dim_size

            # Full-range slice is a no-op
            if start == 0 and end == dim_size:
                src_name = source.name
                while src_name in aliases:
                    src_name = aliases[src_name]
                aliases[node.name] = src_name
                log.info("Alias: %s -> %s (full slice)", node.name, src_name)
                continue

            # Partial slice: resolve source and use getitem template
            src_name = source.name
            if aliases and src_name in aliases:
                src_name = aliases[src_name]
            if src_name in scope.assignments:
                src_list = f"T{scope.assignments[src_name]}"
                all_dynamic_lists.add(src_list)
            elif src_name in input_names:
                src_list = input_names[src_name]
            else:
                src_list = f"W_{src_name}"
                all_static_lists[src_name] = resolve_weight(src_name)

            trailing = math.prod(input_shape[dim + 1:]) if dim + 1 < len(input_shape) else 1
            chunk_size = (end - start) * trailing
            num_rows = math.prod(input_shape[:dim]) if dim > 0 else 1
            row_stride = math.prod(input_shape[dim:])
            skip = row_stride - chunk_size
            offset = start * trailing

            input_lists = [src_list]
            log.info("slice(dim=%d, %d:%d) -> %s (%s) (Inputs: %s)",
                     dim, start, end, target_list, node.name, input_lists)

            args = [
                Argument(src_list, input_shape),
                Argument("C_chunk_size", torch.Size([]), value=chunk_size),
                Argument("C_num_rows", torch.Size([]), value=num_rows),
                Argument("C_skip", torch.Size([]), value=skip),
                Argument("C_offset", torch.Size([]), value=offset),
            ]
            instruction = GetItemInstruction(node.target, target_list, *args)

        # Handle getitem on split/chunk results
        elif node.target is operator.getitem and hasattr(node.args[0], 'name') and node.args[0].name in split_meta:
            split_node = node.args[0]
            chunk_index = node.args[1]
            input_shape, split_size, dim = split_meta[split_node.name]

            # Resolve the split's input tensor through aliases
            source_name = aliases[split_node.name]
            if source_name in scope.assignments:
                src_list = f"T{scope.assignments[source_name]}"
                all_dynamic_lists.add(src_list)
            elif source_name in input_names:
                src_list = input_names[source_name]
            else:
                src_list = f"W_{source_name}"

            row_stride = input_shape[dim]
            chunk_size = min(split_size, row_stride - chunk_index * split_size)
            num_rows = math.prod(input_shape[:dim]) if dim > 0 else 1
            skip = row_stride - chunk_size
            offset = chunk_index * split_size

            input_lists = [src_list]
            log.info("getitem[%d] -> %s (%s) (Inputs: %s)", chunk_index, target_list, node.name, input_lists)

            args = [
                Argument(src_list, input_shape),
                Argument("C_chunk_size", torch.Size([]), value=chunk_size),
                Argument("C_num_rows", torch.Size([]), value=num_rows),
                Argument("C_skip", torch.Size([]), value=skip),
                Argument("C_offset", torch.Size([]), value=offset),
            ]
            instruction = GetItemInstruction(node.target, target_list, *args)
        else:
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
        weights={f"W_{k}": v for k, v in all_static_lists.items() if v is not None},
        sig_figs=sig_figs,
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

    # Merge local lists with identical contents to save space
    _merge_duplicate_lists(scratch_output)

    # Remove orphaned lists and variables
    _remove_unused(scratch_output)

    # Add unique suffixes to all internal list/variable IDs to avoid
    # conflicts when multiple cattorch sprites are loaded into one project
    _uniquify_ids(scratch_output)

    log.info("Dynamic lists: %s", list(all_dynamic_lists))
    log.info("Static lists: %s", list(all_static_lists.keys()))
    log.info("Constants: %s", list(all_constants))

    finalize_sprite(scratch_output, f"{sprite_name}.sprite3", sprite_name=sprite_name)
