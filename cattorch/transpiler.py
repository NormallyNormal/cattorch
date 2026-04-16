import json
import logging
import math
import operator
import uuid
from dataclasses import dataclass, field

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


# ── Op classification ────────────────────────────────────────────────────────

# Ops that change logical shape but don't move data in the flat array
_NOOP_OPS = {
    "aten.view.default",
    "aten.reshape.default",
    "aten._unsafe_view.default",
    "aten.flatten.using_ints",
    "aten.contiguous.default",
    "aten.clone.default",
    "aten.unsqueeze.default",
    "aten.alias.default",
    "aten.dropout.default",
    "aten._assert_tensor_metadata.default",
    "aten.to.dtype",
}

# Ops that split a tensor into multiple outputs. The split itself is a no-op;
# each getitem on the result becomes the real copy operation.
_SPLIT_OPS = {
    "aten.split.Tensor",
    "aten.split_with_sizes.default",
    "aten.chunk.default",
}

# Ops that materialise a compile-time constant tensor as a static weight.
_TENSOR_GENERATORS = {
    "aten.arange.default": lambda args: torch.arange(args[0], dtype=torch.float32),
    "aten.arange.start": lambda args: torch.arange(args[0], args[1], dtype=torch.float32),
    "aten.ones.default": lambda args: torch.ones(args[0], dtype=torch.float32),
    "aten.zeros.default": lambda args: torch.zeros(args[0], dtype=torch.float32),
    "aten.full.default": lambda args: torch.full(args[0], args[1], dtype=torch.float32),
}


# ── Graph preparation ────────────────────────────────────────────────────────

def _get_shape(arg) -> torch.Size:
    if hasattr(arg, 'meta') and 'val' in arg.meta:
        return arg.meta['val'].shape
    return torch.Size([])


def _decompose_linear(graph):
    """Rewrite aten.linear.default nodes into transpose + matmul + optional add."""
    for node in list(graph.nodes):
        if node.op != 'call_function' or str(node.target) != 'aten.linear.default':
            continue

        args = node.args
        x, weight = args[0], args[1]
        has_bias = len(args) == 3

        with graph.inserting_before(node):
            tp = graph.create_node('call_function', torch.ops.aten.numpy_T.default, (weight,))
            tp.meta = {'val': torch.empty(torch.Size(reversed(weight.meta['val'].shape)))}

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


@dataclass
class _GraphInfo:
    """Pre-processed PyTorch export graph, ready for compilation."""
    nodes: list
    resolve_weight: object  # callable
    input_names: dict
    aliases: dict            # node_name → source_name for no-op ops
    split_meta: dict         # split_node_name → (input_shape, split_size, dim)
    generated_weights: dict  # node_name → tensor


def _prepare_graph(model, example_inputs) -> _GraphInfo:
    """Export the model, decompose ops, and analyze the graph structure."""
    exported = export(model, example_inputs)
    _decompose_linear(exported.graph)
    nodes = list(exported.graph.nodes)
    normalised_state = {
        k.lower().replace(".", "_"): v
        for k, v in exported.state_dict.items()
    }

    # Materialise compile-time generated tensors as static weights
    generated_weights = {}
    for node in nodes:
        if node.op != 'call_function':
            continue
        target = str(node.target)
        if target in _TENSOR_GENERATORS:
            generated_weights[node.name] = _TENSOR_GENERATORS[target](node.args)
            log.info("Generated %s as static weight %s", target, node.name)
        elif target == "aten.ones_like.default":
            ref = node.args[0]
            shape = (generated_weights[ref.name].shape if ref.name in generated_weights
                     else tuple(ref.meta['tensor_meta'].shape))
            generated_weights[node.name] = torch.ones(shape, dtype=torch.float32)
        elif target == "aten.zeros_like.default":
            ref = node.args[0]
            shape = (generated_weights[ref.name].shape if ref.name in generated_weights
                     else tuple(ref.meta['tensor_meta'].shape))
            generated_weights[node.name] = torch.zeros(shape, dtype=torch.float32)

    def resolve_weight(arg_name):
        if arg_name in generated_weights:
            return generated_weights[arg_name]
        key = arg_name.removeprefix("p_").removeprefix("b_")
        return normalised_state.get(key)

    # Identify model inputs (placeholders not in state_dict)
    input_names = {}
    input_count = 0
    for node in nodes:
        if node.op == 'placeholder' and resolve_weight(node.name) is None:
            name = "input" if input_count == 0 else f"input_{input_count}"
            input_names[node.name] = name
            input_count += 1

    # Build alias map for no-op shape ops and splits
    aliases = {}
    split_meta = {}
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
            log.info("Split: %s -> %s (split_size=%s, dim=%s)",
                     node.name, source, split_size, dim)

    return _GraphInfo(
        nodes=nodes,
        resolve_weight=resolve_weight,
        input_names=input_names,
        aliases=aliases,
        split_meta=split_meta,
        generated_weights=generated_weights,
    )


# ── Compiler ─────────────────────────────────────────────────────────────────

class _Compiler:
    """Walk the graph, dispatch each node to the right handler, and produce
    a single Scratch sprite dict with all blocks chained together."""

    def __init__(self, graph: _GraphInfo):
        self.graph = graph
        self.scope = ScopeManager()
        self.block_manager = BlockManager()
        self.dynamic_lists: set[str] = set()
        self.static_lists: dict[str, torch.Tensor | None] = {}
        self.sprite: dict | None = None
        self.last_target_list: str | None = None

    # ── public ───────────────────────────────────────────────────────────

    def compile(self) -> dict:
        self.scope.analyze_lifetimes(
            self.graph.nodes,
            skip=set(self.graph.aliases) | set(self.graph.generated_weights),
        )
        for node in self.graph.nodes:
            if node.op != 'call_function':
                continue
            if node.name in self.graph.aliases:
                continue
            if node.name in self.graph.generated_weights:
                continue
            self._compile_node(node)
        return self.sprite

    @property
    def output_list(self) -> str | None:
        """Name of the list that holds the final output tensor."""
        last_call = None
        for node in reversed(self.graph.nodes):
            if node.op == 'call_function':
                last_call = node
                break
        if last_call and last_call.name in self.graph.aliases:
            source = self.graph.aliases[last_call.name]
            if source in self.scope.assignments:
                return f"T{self.scope.assignments[source]}"
            if source in self.graph.input_names:
                return self.graph.input_names[source]
        return self.last_target_list

    # ── node dispatch ────────────────────────────────────────────────────

    def _compile_node(self, node):
        aten_op = str(node.target)

        # Full-range slices become aliases — detect before allocating a list
        if aten_op == "aten.slice.Tensor" and self._is_full_slice(node):
            source = node.args[0]
            src_name = source.name
            while src_name in self.graph.aliases:
                src_name = self.graph.aliases[src_name]
            self.graph.aliases[node.name] = src_name
            log.info("Alias: %s -> %s (full slice)", node.name, src_name)
            return

        target_list = self.scope.get_list_for_node(node)
        self.last_target_list = target_list

        if aten_op == "aten.cat.default":
            self._compile_cat(node, target_list)
        elif aten_op == "aten.slice.Tensor":
            self._compile_slice(node, target_list)
        elif (node.target is operator.getitem
              and hasattr(node.args[0], 'name')
              and node.args[0].name in self.graph.split_meta):
            self._compile_getitem(node, target_list)
        else:
            self._compile_instruction(node, target_list, aten_op)

        self.scope.release_dependencies(node)

    # ── standard instruction ─────────────────────────────────────────────

    def _compile_instruction(self, node, target_list, aten_op):
        input_lists = self._resolve_args(node)
        log.info("%s -> %s (%s) (Inputs: %s)",
                 aten_op, target_list, node.name, input_lists)

        args = []
        for i, name in enumerate(input_lists):
            node_arg = node.args[i]
            if node_arg is None:
                args.append(Argument(name, torch.Size([]), value=None))
            elif hasattr(node_arg, 'name'):
                args.append(Argument(name, _get_shape(node_arg)))
            else:
                args.append(Argument(name, torch.Size([]), value=node_arg))

        instruction = Instruction.create(aten_op, node.target, target_list, *args)
        instruction.transform_weights(self.static_lists)
        self._emit(instruction, input_lists, target_list)

    # ── cat (pairwise chaining) ──────────────────────────────────────────

    def _compile_cat(self, node, target_list):
        tensor_list = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else 0

        resolved = [(self._resolve_name(t.name), _get_shape(t)) for t in tensor_list]
        if dim < 0:
            dim = len(resolved[0][1]) + dim

        cur_list, cur_shape = resolved[0]
        for pair_idx in range(1, len(resolved)):
            next_list, next_shape = resolved[pair_idx]
            is_last = pair_idx == len(resolved) - 1

            if is_last:
                cat_target = target_list
            else:
                if self.scope.free_pool:
                    self.scope.free_pool.sort()
                    temp_id = self.scope.free_pool.pop(0)
                else:
                    self.scope.peak_lists += 1
                    temp_id = self.scope.peak_lists
                cat_target = f"T{temp_id}"
                self.dynamic_lists.add(cat_target)

            num_outer = math.prod(cur_shape[:dim]) if dim > 0 else 1
            chunk_1 = math.prod(cur_shape[dim:])
            chunk_2 = math.prod(next_shape[dim:])
            cat_inputs = [cur_list, next_list]

            log.info("cat(dim=%d) [%d/%d] -> %s (%s) (Inputs: %s)",
                     dim, pair_idx, len(resolved) - 1,
                     cat_target, node.name, cat_inputs)

            cat_args = [
                Argument(cur_list, cur_shape),
                Argument(next_list, next_shape),
                Argument("C_num_outer", torch.Size([]), value=num_outer),
                Argument("C_chunk_1", torch.Size([]), value=chunk_1),
                Argument("C_chunk_2", torch.Size([]), value=chunk_2),
            ]
            instruction = CatInstruction(node.target, cat_target, *cat_args)
            self._emit(instruction, cat_inputs, cat_target)

            out_shape = list(cur_shape)
            out_shape[dim] = cur_shape[dim] + next_shape[dim]
            out_shape = torch.Size(out_shape)

            if not is_last:
                prev_temp_id = temp_id
            cur_list, cur_shape = cat_target, out_shape

            if pair_idx > 1:
                self.scope.free_pool.append(prev_temp_id)

    # ── slice ────────────────────────────────────────────────────────────

    def _compile_slice(self, node, target_list):
        source = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else 0
        start = node.args[2] if len(node.args) > 2 else 0
        end = node.args[3] if len(node.args) > 3 else None
        input_shape = _get_shape(source)

        if dim < 0:
            dim = len(input_shape) + dim
        dim_size = input_shape[dim]
        if end is None or end > dim_size:
            end = dim_size

        src_list = self._resolve_name(source.name)
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
        self._emit(instruction, input_lists, target_list)

    # ── getitem on split/chunk ───────────────────────────────────────────

    def _compile_getitem(self, node, target_list):
        split_node = node.args[0]
        chunk_index = node.args[1]
        input_shape, split_size, dim = self.graph.split_meta[split_node.name]

        source_name = self.graph.aliases[split_node.name]
        src_list = self._resolve_name(source_name)

        row_stride = input_shape[dim]
        chunk_size = min(split_size, row_stride - chunk_index * split_size)
        num_rows = math.prod(input_shape[:dim]) if dim > 0 else 1
        skip = row_stride - chunk_size
        offset = chunk_index * split_size

        input_lists = [src_list]
        log.info("getitem[%d] -> %s (%s) (Inputs: %s)",
                 chunk_index, target_list, node.name, input_lists)

        args = [
            Argument(src_list, input_shape),
            Argument("C_chunk_size", torch.Size([]), value=chunk_size),
            Argument("C_num_rows", torch.Size([]), value=num_rows),
            Argument("C_skip", torch.Size([]), value=skip),
            Argument("C_offset", torch.Size([]), value=offset),
        ]
        instruction = GetItemInstruction(node.target, target_list, *args)
        self._emit(instruction, input_lists, target_list)

    # ── helpers ──────────────────────────────────────────────────────────

    def _resolve_args(self, node) -> list[str]:
        """Map a node's args to Scratch list names."""
        input_lists = []
        for arg in node.args:
            if arg is None:
                input_lists.append("_none")
            elif hasattr(arg, 'name'):
                input_lists.append(self._resolve_name(arg.name))
            else:
                input_lists.append(f"C_{arg}")
        return input_lists

    def _resolve_name(self, arg_name: str) -> str:
        """Resolve a single node name to its Scratch list name."""
        if arg_name in self.graph.aliases:
            arg_name = self.graph.aliases[arg_name]
        if arg_name in self.scope.assignments:
            name = f"T{self.scope.assignments[arg_name]}"
            self.dynamic_lists.add(name)
            return name
        if arg_name in self.graph.input_names:
            return self.graph.input_names[arg_name]
        name = f"W_{arg_name}"
        self.static_lists[arg_name] = self.graph.resolve_weight(arg_name)
        return name

    def _emit(self, instruction, input_lists, target_list):
        """Generate Scratch blocks for an instruction and merge into sprite."""
        data = instruction.finalize()

        self.block_manager.new_context()
        data["blocks"] = self.block_manager.apply_to_blocks(data["blocks"])

        tensor_lists = [n for n in input_lists if not n.startswith("C_")]
        all_lists = tensor_lists + [target_list]

        data = TensorAdder().apply(data, all_lists)
        data["blocks"] = TensorReplacer(data, all_lists).apply(data["blocks"])

        if self.sprite is None:
            self.sprite = data
        else:
            self.sprite = combine(self.sprite, data)

    @staticmethod
    def _is_full_slice(node) -> bool:
        """True if the slice covers the full dimension (i.e. is a no-op)."""
        dim = node.args[1] if len(node.args) > 1 else 0
        start = node.args[2] if len(node.args) > 2 else 0
        end = node.args[3] if len(node.args) > 3 else None
        input_shape = _get_shape(node.args[0])
        if dim < 0:
            dim = len(input_shape) + dim
        dim_size = input_shape[dim]
        if end is None or end > dim_size:
            end = dim_size
        return start == 0 and end == dim_size


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


# ── Public API ───────────────────────────────────────────────────────────────

def transpile(
    model: torch.nn.Module,
    example_inputs: torch.Tensor | tuple[torch.Tensor, ...],
    sprite_name: str,
    sig_figs: int | None = None,
):
    if isinstance(example_inputs, torch.Tensor):
        example_inputs = (example_inputs,)

    # 1. Analyse the PyTorch graph
    graph = _prepare_graph(model, example_inputs)

    # 2. Compile graph nodes into Scratch blocks
    compiler = _Compiler(graph)
    sprite = compiler.compile()

    # 3. Attach static weights and model inputs
    tensor_adder = TensorAdder()
    sprite = tensor_adder.apply(
        sprite,
        [f"W_{k}" for k in compiler.static_lists],
        weights={f"W_{k}": v for k, v in compiler.static_lists.items() if v is not None},
        sig_figs=sig_figs,
    )
    sprite = tensor_adder.apply(sprite, list(graph.input_names.values()))

    # 4. Name the output list and clean up
    output = compiler.output_list
    if output is not None:
        _rename_list(sprite, output, "output")

    _merge_duplicate_lists(sprite)
    _remove_unused(sprite)
    _uniquify_ids(sprite)

    log.info("Dynamic lists: %s", list(compiler.dynamic_lists))
    log.info("Static lists: %s", list(compiler.static_lists.keys()))

    finalize_sprite(sprite, f"{sprite_name}.sprite3", sprite_name=sprite_name)
