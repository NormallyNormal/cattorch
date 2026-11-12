import logging
import math
import re
import operator
import dataclasses
import weakref
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, NamedTuple

import torch

from cattorch.codegen import CodegenConfig
from cattorch.errors import UnsupportedModelError
from cattorch.export_options import validate_export_options
from cattorch.fast import FastConfig, prepare_fast_model
from cattorch.frontend import flat_example_inputs, keep_operations
from cattorch.graph import (
    _NOOP_OPS,
    _GenerationSpec,
    _GraphInfo,
    _get_dtype,
    _get_shape,
    _prepare_graph,
    _unsupported_operation,
)
from cattorch.graph_transforms import fold_eval_batch_norms
from cattorch.results import (
    TranspileResult,
    _tensor_spec as boundary_spec,
    fallback_warning,
)
from cattorch.program import (
    ExportProgram,
    GenerationProgram,
    ProgramCall,
    RecurrentGenerationProgram,
)
from cattorch.adapters import ModuleAdapter
from cattorch.kernel_api import (
    Kernel,
    active_kernel_targets,
    validate_kernels,
    warn_unused_kernels,
)
from cattorch.quantization import (
    QuantizationConfig,
    _StaticSourceRecorder,
    _storage_quantization_report,
    effective_group_size,
    expert_bank_effective_ndims,
    quantize_model,
)
from cattorch.sprite import (
    _TOP_K_IDS,
    _TOP_K_VALUES,
    _GENERATION_RESET,
    _STEP_LOGITS,
    _STEP_TOKEN,
    _add_input_checks,
    _add_prepare_for_save,
    _apply_static_storage,
    _apply_tied_static_aliases,
    _deduplicate_static_tensors,
    _find_procedure_definition,
    _finish_generation_prepare,
    _gate_chain,
    _guard_generation_init,
    _guard_plain_calls,
    _logical_list_sizes,
    _merge_duplicate_lists,
    _remove_canvas_calls,
    _graph_layer,
    _merge_lists_by_name,
    _merge_variables_by_name,
    _remove_unused,
    _share_repeated_transformer_layers,
    _static_storage_shard_limits,
    _wrap_generation_lifecycle,
    _wrap_optimized_lifecycle,
)
from cattorch import interface
from cattorch.interface import (
    NUMBERS, OK, STATUS_VARIABLE, WHOLE_NUMBERS, final_data_name, input_name, item_checks,
    length_checks, output_name,
)
from cattorch.storage import StorageConfig, quantized_storage
from cattorch.util.argument import Argument
from cattorch.operator_registry import (
    INDEX_MAP_OPS,
    MATMUL_OPS,
    OperatorRegistry,
    default_registry,
)
from cattorch.util.instruction.optimized_elementwise import (
    ARITHMETIC_EXPRESSION_KERNEL,
    PAIRED_LINEAR_SWIGLU_KERNEL,
    SWIGLU_KERNEL,
)
from cattorch.util.instruction.optimized_linear import (
    CAUSAL_SCORE_KERNEL,
    CAUSAL_SOFTMAX_KERNEL,
    CAUSAL_VALUE_MATMUL_KERNEL,
    FAST_CAUSAL_SOFTMAX_KERNEL,
    LINEAR_KERNEL,
    LOW_RANK_MATMUL_KERNEL,
    QKV_LINEAR_KERNEL,
    grouped_linear,
    matmul_kernel,
    pruned_matmul_blocks,
    pruned_matmul_kernel,
)
from cattorch.util.instruction.optimized_moe import (
    EXPERT_FAMILY_MOE_KERNEL,
    ROUTED_EXPERTS_KERNEL,
)
from cattorch.util.instruction.generic import INDEX_MAP_KERNEL, index_map
from cattorch.util.instruction.optimized_tensor import (
    CAT_KERNEL,
    EMBEDDING_ADD_KERNEL,
    GETITEM_KERNEL,
    STRIDED_SLICE_KERNEL,
)
from cattorch.util.scope import ScopeManager
from cattorch.util.scratch.block_combiner import combine
from cattorch.util.scratch.block_manager import BlockManager
from cattorch.util.scratch.finalize_scratch import FinalizedSprite, finalize_sprite
from cattorch.util.scratch.ids import uniquify_data_ids
from cattorch.util.scratch.interface import (
    rename_lists as _rename_lists,
    rename_procedure as _rename_procedure,
)
from cattorch.util.scratch.dsl import (
    eq,
    if_,
    unroll_factor_limit,
    var,
)
from cattorch.util.scratch.sharding import SCRATCH_LIST_LIMIT, shard_sprite_lists
from cattorch.util.scratch.tensor_adder import TensorAdder
from cattorch.util.scratch.tensor_replacer import TensorReplacer

log = logging.getLogger(__name__)

_LAYER_CACHE_BASE = "cattorch layer cache base"
_PROJECT_OUTPUT = "cattorch project output"
# 1 when a shared layer computes past its K/V append: always, except in the
# final layer for a hidden prefill token.
_LAYER_TAIL = "cattorch layer tail"


def _bank_generation_caches(graph: _GraphInfo) -> bool:
    """Interleave layer caches so one shared layer body can address them.

    Rows are appended token-major and layer-minor. A shared transformer
    procedure selects its layer with ``_LAYER_CACHE_BASE``; cached attention
    advances between tokens by the complete per-token bank width.
    """
    generation = graph.generation
    score_entries = list(generation.score_caches.items())
    value_entries = list(generation.value_caches.items())
    if len(score_entries) < 2 or len(score_entries) != len(value_entries):
        return False
    widths = [generation.cache_widths[cache] for _, cache in (*score_entries, *value_entries)]
    if len(set(widths)) != 1:
        log.info("Banked generation caches skipped: per-layer widths differ")
        return False

    layer_width = widths[0]
    layers = len(score_entries)
    bank_width = layers * layer_width
    key_bank = "cattorch K cache"
    value_bank = "cattorch V cache"
    generation.score_caches = {name: key_bank for name, _ in score_entries}
    generation.value_caches = {name: value_bank for name, _ in value_entries}
    generation.cache_widths = {key_bank: bank_width, value_bank: bank_width}
    generation.first_cache = (key_bank, bank_width)
    generation.banked_caches = True
    generation.cache_layers = layers
    generation.cache_layer_width = layer_width
    return True


def _query_only_nodes(graph: _GraphInfo, attention: str) -> set[str]:
    """Nodes whose values reach nothing but ``attention``'s query argument."""
    nodes = {node.name: node for node in graph.nodes}
    target = nodes[attention]
    query_only: set[str] = set()
    for node in reversed(graph.nodes):
        if node.op != "call_function" or node is target or not node.users:
            continue
        if all(
            user.name in query_only
            or (user is target and node is target.args[0] and node not in target.args[1:])
            for user in node.users
        ):
            query_only.add(node.name)
    return query_only


def _kinds_of_items(graph: _GraphInfo, flat_inputs) -> dict[int, tuple[str | None, int | None]]:
    """What each input's items must be, by position, and the rows of the table it indexes.

    Numbers in a floating-point list, whole numbers in an integer one, and,
    for an index into an embedding table, whole numbers within its rows.
    Bool lists take any values.
    """
    kinds: dict[int, tuple[str | None, int | None]] = {}
    for index, example in enumerate(flat_inputs):
        if example.is_floating_point():
            kinds[index] = (NUMBERS, None)
        elif example.dtype == torch.bool or example.is_complex():
            kinds[index] = (None, None)
        else:
            kinds[index] = (WHOLE_NUMBERS, _index_rows(graph, input_name(index)))
    return kinds


def _input_checks(graph: _GraphInfo, layouts, flat_inputs, *, unit_spec=None) -> list[tuple]:
    """What ``cattorch forward`` checks in each input list before it runs.

    The number of items: a fixed list's, or a variable-length list's whole
    rows within its bounds (``unit_spec`` gives them for an export run once
    per item); then the items, as ``_kinds_of_items`` says. Returns one group of
    statements per input.
    """
    groups: list[tuple] = []
    nodes = {graph.input_names[node.name]: node for node in graph.nodes if node.name in graph.input_names}
    kinds = _kinds_of_items(graph, flat_inputs)
    for index, example in enumerate(flat_inputs):
        name = input_name(index)
        node = nodes.get(name)
        shards = [shard.name for shard in layouts[name].shards] if name in layouts else [name]
        statements: list = []
        if unit_spec is not None and index in unit_spec:
            _, minimum, maximum, _ = unit_spec[index]
            width = example.numel()
            statements += length_checks(
                name, width=width, min_items=width * max(minimum, 1),
                max_items=width * maximum, lists=shards,
            )
        elif node is not None and node.name in graph.dynamic_bounds:
            shape = _get_shape(node)
            width = max(math.prod(shape) // max(shape[graph.dynamic_axes[node.name]], 1), 1)
            _, minimum, _ = graph.dynamic_bounds[node.name]
            statements += length_checks(
                name, width=width, min_items=width * max(minimum, 1),
                max_items=graph.runtime_capacity(node.name, shape), lists=shards,
            )
        else:
            statements += length_checks(name, numel=example.numel(), lists=shards)
        items = item_checks(name, *kinds[index], lists=shards)
        if items:
            statements.append(if_(eq(var(STATUS_VARIABLE), OK), items))
        groups.append(tuple(statements))
    return groups


def _input_lists(layouts, flat_inputs) -> list[str]:
    """Every physical list of the inputs, shards included."""
    names = []
    for index in range(len(flat_inputs)):
        name = input_name(index)
        names += [shard.name for shard in layouts[name].shards] if name in layouts else [name]
    return names


def _index_rows(graph: _GraphInfo, list_name: str = "input") -> int | None:
    """Rows of the smallest embedding table an input list indexes, if it indexes one."""
    tokens = {name for name, value in graph.input_names.items() if value == list_name}
    rows: int | None = None
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if (
            str(node.target) == "aten.embedding.default"
            and getattr(node.args[1], "name", None) in tokens
        ):
            table = int(_get_shape(node.args[0])[0])
            rows = table if rows is None else min(rows, table)
        if (
            node.all_input_nodes
            and node.all_input_nodes[0].name in tokens
            and str(node.target) in _NOOP_OPS | {"aten.to.dtype", "aten._to_copy.default"}
        ):
            tokens.add(node.name)
    return rows


def _shared_layer_attention(graph: _GraphInfo) -> str:
    """The cached attention of layer zero, whose code the shared layer runs."""
    by_name = {node.name: node for node in graph.nodes}
    return next(name for name in graph.generation.score_caches if _graph_layer(by_name[name]) == 0)


def _guard_hidden_layer_tail(sprite: dict, graph: _GraphInfo, compiler, attention: str) -> None:
    """Skip what a hidden prefill token never reads in ``attention``'s layer.

    That is the query, which only feeds the attention output, and everything
    after the attention. Its K/V append still runs. In a shared layer, the
    work stops at the end of the layer's code and the gate is the layer tail;
    otherwise it runs to the end of the step.
    """
    gate = _LAYER_TAIL if graph.generation.banked_caches else _PROJECT_OUTPUT
    root_entries = compiler.emitted_roots
    root_names = [name for name, _ in root_entries]
    attention_position = root_names.index(attention)
    if attention_position + 1 < len(root_entries):
        following = root_entries[attention_position + 1][1]
        # In a shared layer, a following root of another layer was removed.
        if following in sprite["blocks"]:
            _gate_chain(sprite, following, gate)
    query_only = _query_only_nodes(graph, attention)
    position = 0
    while position < attention_position:
        if root_names[position] not in query_only:
            position += 1
            continue
        end = position
        while end < attention_position and root_names[end] in query_only:
            end += 1
        _gate_chain(sprite, root_entries[position][1], gate, stop=root_entries[end][1])
        position = end


def _gate_hidden_outputs(sprite: dict, graph: _GraphInfo, compiler, hidden: frozenset[int]) -> None:
    """Run the operations only the ``hidden`` outputs need only while projecting.

    They run while ``cattorch project output`` is 1, so a hidden prefill token
    skips them; everything another output reads still runs. If the compiled
    chain no longer holds every node's root in order, as after layer sharing,
    nothing is gated.
    """
    hidden_names = {name for index, (name, _, _) in enumerate(graph.outputs) if index in hidden}
    kept_names = {name for index, (name, _, _) in enumerate(graph.outputs) if index not in hidden}
    only_hidden: set[str] = set()
    for node in reversed(graph.nodes):
        if node.op != "call_function" or not node.users or node.name in kept_names:
            continue
        if all(
            user.name in only_hidden or (user.op == "output" and node.name in hidden_names)
            for user in node.users
        ):
            only_hidden.add(node.name)
    roots = compiler.emitted_roots
    gated = [
        position < compiler.node_roots and name in only_hidden
        for position, (name, _) in enumerate(roots)
    ]
    if not any(gated):
        return
    blocks = sprite["blocks"]
    definition = _find_procedure_definition(sprite, "cattorch forward")
    chain = []
    current = blocks[blocks[definition]["next"]]["next"]
    while current is not None:
        chain.append(current)
        current = blocks[current].get("next")
    emitted = [block for _, block in roots]
    if [block for block in chain if block in set(emitted)] != emitted:
        log.info("Hidden prefill computes every output: the compiled roots were rearranged")
        return
    position = 0
    while position < len(roots):
        if not gated[position]:
            position += 1
            continue
        end = position
        while end < len(roots) and gated[end]:
            end += 1
        _gate_chain(
            sprite,
            roots[position][1],
            _PROJECT_OUTPUT,
            stop=roots[end][1] if end < len(roots) else None,
        )
        position = end


# ── Compiler ─────────────────────────────────────────────────────────────────


class _LayerSharingUnavailable(Exception):
    """Banked caches were requested but the layers could not be factored."""


# A variable-length input: its name, minimum and maximum length, and the
# axis that varies when it isn't the first.
_DynamicBound = tuple[str, int, int] | tuple[str, int, int, int]


def _public_dynamic_bound(bound: _DynamicBound | None) -> tuple[int, tuple] | None:
    """A variable-length input's axis and ``(name, minimum, maximum)``."""
    if bound is None:
        return None
    return (bound[3] if len(bound) == 4 else 0, tuple(bound[:3]))


def _tensor_identity(tensor: torch.Tensor) -> tuple:
    """What makes two tensors the same stored values: one view of one storage.

    Quantization keys its parameter views the same way (without the dtype),
    in ``quantization._storage_key``.
    """
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


class _RowShards(NamedTuple):
    """How the library Linear kernel's weight is cut into row shards."""

    # Whether the kernel runs four outputs at a time; its shards then hold
    # whole four-row groups, stored interleaved.
    grouped: bool
    rows: int


def _layers_can_share_caches(graph: _GraphInfo) -> bool:
    """Whether each automatic attention block sits in its own ``blocks.N`` layer."""
    by_name = {node.name: node for node in graph.nodes}
    layers = [_graph_layer(by_name[name]) for name in graph.generation.score_caches]
    return None not in layers and layers == list(range(len(layers)))


class _Compiler:
    """Walk the graph, dispatch each node to the right handler, and produce
    a single Scratch sprite dict with all blocks chained together."""

    def __init__(self, graph: _GraphInfo, *, storage: StorageConfig):
        self.graph = graph
        # Operations lowered by an approximate kernel, for the result.
        self.approximate: Counter[str] = Counter(graph.approximate_fusions)
        self.scope = ScopeManager()
        self.block_manager = BlockManager()
        self.dynamic_lists: set[str] = set()
        self.static_lists: dict[str, torch.Tensor | None] = {}
        self.static_effective_ndims: dict[str, int] = {}
        self.sprite: dict | None = None
        self.last_target_list: str | None = None
        self._static_tensor_names: dict[tuple, str] = {}
        self.logical_sizes: dict[str, int] = {}
        self.active_node: str | None = None
        self.emitted_roots: list[tuple[str, str]] = []
        # How many of ``emitted_roots`` compute graph nodes; output copies follow.
        self.node_roots = 0
        self.static_shard_alignments: dict[str, int] = {}
        self.tied_storage_aliases: dict[str, tuple[str, ...]] = {}
        # A tied table that ends up stored a second time, with the reason.
        self.tied_storage_copies: dict[str, str] = {}
        self.storage = storage
        # Quantization groups each matrix along its rows, over its inputs.
        self.row_aligned_weights = storage.precision in {"int8", "int6", "int4"}
        # id(weight) -> (weak reference to it, its storage list cap).
        self._storage_shard_limits: dict[int, tuple[weakref.ref, int]] = {}
        self.interleaved_static_layouts: dict[str, tuple[tuple[int, int, int], ...]] = {}
        # A row shard's source weight size, for storage to decide the shard's
        # precision as the whole weight's.
        self.static_source_numels: dict[str, int] = {}
        self._tied_weights, self._grouped_tied_weights = self._find_tied_weights()
        self._compiled_output_lists: tuple[str, ...] = ()

    def _row_shard_limit(self, weight: torch.Tensor, *, grouped: bool, tied: bool = False) -> int:
        """The most values in each row shard of an output-major ``weight``.

        Storage caps each physical list by its encoded size, which is below
        Scratch's list limit for a weight it keeps in float16 or float32. A
        grouped weight under integer storage is cut to that cap, so no
        physical list ends inside a four-row group the codec must undo, and
        so is a tied weight larger than a list, so the table, sharded at the
        same row boundaries, can share the head's lists.
        """
        if weight.numel() <= SCRATCH_LIST_LIMIT // 4 or not (
            # No storage caps a list below a quarter of the limit.
            (grouped and self.row_aligned_weights) or (tied and weight.numel() > SCRATCH_LIST_LIMIT)
        ):
            return SCRATCH_LIST_LIMIT
        # Cached by the tensor object, not its address: a transposed copy is
        # a temporary, and the next one can reuse the address.
        cached = self._storage_shard_limits.get(id(weight))
        if cached is not None and cached[0]() is weight:
            limit = cached[1]
        else:
            limit = _static_storage_shard_limits({"weight": weight}, self.storage).get(
                "W_weight", SCRATCH_LIST_LIMIT
            )
            self._storage_shard_limits[id(weight)] = (weakref.ref(weight), limit)
        return limit if weight.numel() > limit else SCRATCH_LIST_LIMIT

    def _row_shards(self, weight: torch.Tensor, *, tied: bool = False) -> _RowShards | None:
        """How to cut the library Linear kernel's output-major ``weight`` into
        row shards, or ``None`` when no list holds a complete one.

        The kernel runs four outputs at a time exactly when ``grouped_linear``
        says so for the whole weight, however it is cut, so each shard of a
        grouped weight holds whole four-row groups. When not even one row (or
        one four-row group) fits, the caller keeps the weight whole and lets
        the generic sharder split it.
        """
        columns, inner = weight.shape
        grouped = grouped_linear(columns, inner)
        rows = self._row_shard_limit(weight, grouped=grouped, tied=tied) // inner
        if grouped:
            rows -= rows % 4
        if rows < (4 if grouped else 1):
            return None
        return _RowShards(grouped, rows)

    def _store_row_shards(
        self,
        weight: torch.Tensor,
        rows: int,
        *,
        interleaved: bool,
        key: Callable[[int], str],
    ) -> list[tuple[str, torch.Tensor]]:
        """Store output-major ``weight`` as static lists of ``rows`` rows each.

        Shard ``index`` (from 1) is stored as ``key(index)``. An
        ``interleaved`` shard is stored in the grouped kernel's four-row
        layout, and under integer storage that layout is recorded so storage
        can undo it to find the quantized grid. Returns each shard's list name
        and values.
        """
        inner = weight.shape[1]
        shards = []
        for index, start in enumerate(range(0, weight.shape[0], rows), 1):
            shard = weight[start : start + rows].detach().clone()
            if interleaved:
                shard = (
                    shard.reshape(-1, 4, inner).permute(0, 2, 1).contiguous().reshape(shard.shape)
                )
            name = key(index)
            self.static_lists[name] = shard
            self.static_source_numels[name] = weight.numel()
            if interleaved and self.row_aligned_weights:
                self.interleaved_static_layouts[name] = ((0, shard.shape[0], inner),)
            shards.append((f"W_{name}", shard))
        return shards

    def _find_tied_weights(self) -> tuple[set[tuple], set[tuple]]:
        """Weights a library Linear reads that something else reads too, and
        those of them read only by embeddings and grouped Linears, all built in.

        The second kind is stored once, in the Linear's four-row interleaved
        layout (see ``_prepare_linear_weight``), and the embedding kernel reads
        it that way. Any other reader, or a user's kernel for either
        operation, needs the plain layout.
        """
        from cattorch.operator_registry import default_registry

        graph = self.graph
        registry, fast, config = graph.registry, graph.fast, graph.fast_config
        linear = registry.kernel("aten.linear.default", fast=fast, config=config)
        embedding = registry.kernel("aten.embedding.default", fast=fast, config=config)
        interleaving = (
            # Weight pruning owns the Linear's layout.
            not (fast and config.weights.enabled)
            and getattr(linear, "kernel", None) is LINEAR_KERNEL
            and embedding
            is default_registry().kernel(
                "aten.embedding.default",
                fast=fast,
                config=config,
            )
        )

        def interleavable(tensor):
            shards = self._row_shards(tensor, tied=True)
            return shards is not None and shards.grouped

        readers: dict[tuple, set[str]] = {}
        tensors: dict[tuple, torch.Tensor] = {}
        for node in graph.nodes:
            if node.op not in {"call_function", "output"}:
                continue
            target = str(node.target)
            operands = [(position, value) for position, value in enumerate(node.args)] + [
                (None, value) for value in node.kwargs.values()
            ]
            for position, value in operands:
                for argument in value if isinstance(value, (list, tuple)) else (value,):
                    if not hasattr(argument, "name"):
                        continue
                    tensor = graph.resolve_weight(argument.name)
                    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
                        continue
                    if target == "aten.embedding.default" and position == 0:
                        kind = "embedding"
                    elif target == "aten.linear.default" and position == 1:
                        kind = "linear"
                    else:
                        kind = "other"
                    identity = _tensor_identity(tensor)
                    readers.setdefault(identity, set()).add(kind)
                    tensors[identity] = tensor
        tied = {
            identity for identity, kinds in readers.items() if "linear" in kinds and len(kinds) > 1
        }
        grouped = {
            identity
            for identity in tied
            if interleaving
            and readers[identity] == {"embedding", "linear"}
            and interleavable(tensors[identity])
        }
        return tied, grouped

    def _uses_grouped_tied_weight(self, node) -> bool:
        weight_index = 0 if str(node.target) == "aten.embedding.default" else 1
        if len(node.args) <= weight_index or not hasattr(node.args[weight_index], "name"):
            return False
        tensor = self.graph.resolve_weight(node.args[weight_index].name)
        return (
            isinstance(tensor, torch.Tensor)
            and _tensor_identity(tensor) in self._grouped_tied_weights
        )

    def _argument(self, name: str, node) -> Argument:
        return Argument(
            name,
            _get_shape(node),
            dynamic_axis=self.graph.dynamic_axes.get(node.name),
            dtype=_get_dtype(node),
            # Parameters, buffers and folded constants live in W_ lists.
            static=name.startswith("W_"),
        )

    @staticmethod
    def _kernel_instruction(kernel, output, *args, options=None):
        """An instruction running ``kernel``, a ``Kernel`` the compiler calls itself."""
        instruction = kernel.instruction_class()(kernel.target_name, output, *args)
        if options:
            instruction.options = options
        return instruction

    def _linear_instruction(self, output, args, *, interleaved=False, epilogue=(), fast=False):
        """The library Linear kernel on ``args``: the input, each weight
        shard, the bias, and a residual for the epilogue."""
        return self._kernel_instruction(
            LINEAR_KERNEL,
            output,
            *args,
            options={"interleaved": interleaved, "epilogue": tuple(epilogue), "fast": fast},
        )

    @staticmethod
    def _weight_shards(names, shard_rows, inner):
        """Arguments for weight lists of ``shard_rows`` whole rows each."""
        return [
            Argument(name, torch.Size((rows, inner)), static=name.startswith("W_"))
            for name, rows in zip(names, shard_rows)
        ]

    def _linear_arguments(self, args, input_lists, shard_rows):
        """Linear's arguments after ``_prepare_linear_weight`` laid out its weight.

        ``args`` are the input, weight, bias and residual; ``input_lists``
        holds one list per weight shard in place of the weight.
        """
        source, weight, *rest = args
        shard_count = len(input_lists) - len(rest) - 1
        names = input_lists[1 : 1 + shard_count]
        if len(names) == 1:
            shards = [Argument(names[0], weight.shape, static=names[0].startswith("W_"))]
        else:
            shards = self._weight_shards(names, shard_rows, weight.shape[-1])
        return [source, *shards, *rest]

    def _instantiate(self, kernel, lowering_target, node, target_list, args):
        """Create a kernel instruction; context kernels also get output info."""
        output = target_list
        if getattr(kernel, "wants_output_info", False):
            output = Argument(target_list, _get_shape(node), dtype=_get_dtype(node))
        instruction = kernel(lowering_target, output, *args)
        options = self._lowering_options(node)
        if options:
            instruction.options = options
        return instruction

    def _automatic_bindings(self, node) -> dict[int, str]:
        """Argument positions of an automatic-generation op bound to named lists."""
        generation = self.graph.generation
        target = str(node.target)
        if target == "cattorch.cached_attention.default":
            return {
                3: generation.score_caches[node.name],
                4: generation.value_caches[node.name],
                5: "_none",
            }
        if target == "cattorch.sequence_row.default":
            return {1: "_none"}
        return {}

    def _lowering_options(self, node) -> dict[str, object]:
        """Compile-time facts for a context kernel (``KernelContext.options``)."""
        generation = self.graph.generation
        if str(node.target) != "cattorch.cached_attention.default":
            return {}
        if generation.banked_caches:
            options: dict[str, object] = {
                "cache_stride": generation.cache_widths[generation.score_caches[node.name]],
                "cache_base_variable": _LAYER_CACHE_BASE,
            }
            if generation.hidden_prefill_attention is not None:
                # Every shared layer reads the gate; only the final one sets it
                # to skip work for hidden prefill tokens.
                options["hidden_prefill_gate"] = _LAYER_TAIL
            return options
        if node.name == generation.hidden_prefill_attention:
            return {"hidden_prefill_gate": _PROJECT_OUTPUT}
        return {}

    # ── public ───────────────────────────────────────────────────────────

    def compile(self) -> dict:
        self.scope.analyze_lifetimes(
            self.graph.nodes,
            skip=set(self.graph.aliases) | set(self.graph.generated_weights),
            aliases=self.graph.aliases,
        )
        for node in self.graph.nodes:
            if node.op != "call_function":
                continue
            if node.name in self.graph.aliases:
                continue
            if node.name in self.graph.generated_weights:
                continue
            if node.name in self.graph.fused_nodes:
                continue
            self._compile_node(node)
        self.node_roots = len(self.emitted_roots)
        self._materialize_output_if_needed()
        # The output is always materialized into the sprite.
        assert self.sprite is not None
        return self.sprite

    @property
    def output_list(self) -> str | None:
        """Name of the list that holds the declared model output tensor."""
        return self._compiled_output_lists[0] if self._compiled_output_lists else None

    @property
    def output_lists(self) -> tuple[str, ...]:
        """Scratch lists holding tensor return leaves in pytree order."""
        return self._compiled_output_lists

    def _materialize_output_if_needed(self) -> None:
        """Copy input/static alias outputs into a distinct writable output list."""
        result = []
        seen: set[str] = set()
        for graph_output, output_shape, _output_dtype in self.graph.outputs:
            source = graph_output
            while source in self.graph.aliases:
                source = self.graph.aliases[source]
            if source in self.scope.assignments:
                source_list = f"T{self.scope.assignments[source]}"
                needs_copy = source_list in seen
            elif source in self.graph.input_names:
                source_list = self.graph.input_names[source]
                needs_copy = True
            elif isinstance(self.graph.resolve_weight(source), torch.Tensor):
                source_list = self._resolve_name(source)
                needs_copy = True
            else:
                raise UnsupportedModelError(
                    f"unable to materialize exported output node {graph_output!r}"
                )

            if needs_copy:
                self.scope.peak_lists += 1
                target_list = f"T{self.scope.peak_lists}"
                self.dynamic_lists.add(target_list)
                size = math.prod(output_shape)
                args = (
                    Argument(
                        source_list,
                        output_shape,
                        dynamic_axis=self.graph.dynamic_axes.get(graph_output),
                    ),
                    Argument("C_chunk_size", torch.Size([]), value=size),
                    Argument("C_num_rows", torch.Size([]), value=1),
                    Argument("C_skip", torch.Size([]), value=0),
                    Argument("C_offset", torch.Size([]), value=0),
                )
                instruction = self._kernel_instruction(GETITEM_KERNEL, target_list, *args)
                self._emit(instruction, [source_list], target_list)
            else:
                target_list = source_list
            result.append(target_list)
            seen.add(target_list)
            self.last_target_list = target_list
        self._compiled_output_lists = tuple(result)

    # ── node dispatch ────────────────────────────────────────────────────

    def _compile_node(self, node):
        self.active_node = node.name
        aten_op = str(node.target)

        if node.name in self.graph.causal_score_fusions:
            self._compile_causal_score(node)
            return
        if node.name in self.graph.qkv_linear_fusions:
            self._compile_qkv_linear(node)
            return
        if node.name in self.graph.embedding_add_fusions:
            self._compile_embedding_add(node)
            return
        if node.name in self.graph.elementwise_fusions:
            self._compile_fused_arithmetic(node)
            return
        if node.name in self.graph.linear_fusions:
            self._compile_fused_linear(node)
            return
        if node.name in self.graph.paired_swiglu_fusions:
            self._compile_paired_swiglu(node)
            return
        if node.name in self.graph.silu_mul_fusions:
            self._compile_silu_mul(node)
            return

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
        elif (
            node.target is operator.getitem
            and hasattr(node.args[0], "name")
            and node.args[0].name in self.graph.split_meta
        ):
            self._compile_getitem(node, target_list)
        elif aten_op in INDEX_MAP_OPS and not self._runtime_gather(node):
            self._compile_index_map(node, target_list)
        else:
            self._compile_instruction(node, target_list, aten_op)
        self.scope.release_dependencies(node)

    def _runtime_gather(self, node) -> bool:
        """A ``gather`` whose indices are computed at run time."""
        if str(node.target) != "aten.gather.default":
            return False
        index = node.args[2] if len(node.args) > 2 else node.kwargs.get("index")
        name = getattr(index, "name", None)
        while name in self.graph.aliases:
            name = self.graph.aliases[name]
        return name is not None and not isinstance(
            self.graph.resolve_weight(name),
            torch.Tensor,
        )

    def _compile_index_map(self, node, target_list):
        """Lower a data-movement operation through a static index table."""
        source = node.args[0]

        def static(value):
            if hasattr(value, "name"):
                name = value.name
                while name in self.graph.aliases:
                    name = self.graph.aliases[name]
                tensor = self.graph.resolve_weight(name)
                if not isinstance(tensor, torch.Tensor):
                    raise _unsupported_operation(
                        node,
                        "only the first argument can vary at runtime",
                    )
                # Aliases are views such as reshapes; keep this value's shape.
                return tensor.reshape(_get_shape(value))
            if isinstance(value, (list, tuple)):
                return type(value)(static(entry) for entry in value)
            return value

        if not hasattr(source, "name"):
            raise _unsupported_operation(node, "the first argument must be a tensor")
        if node.name not in self.graph.row_loops and (
            node.name in self.graph.dynamic_axes or source.name in self.graph.dynamic_axes
        ):
            raise _unsupported_operation(
                node,
                "data movement on a runtime extent is not supported",
            )
        try:
            indices, fill = index_map(
                node.target,
                tuple(_get_shape(source)),
                static(tuple(node.args[1:])),
                static(dict(node.kwargs)),
            )
        except ValueError as error:
            raise _unsupported_operation(node, str(error)) from error
        if len(indices) != math.prod(_get_shape(node)):
            raise _unsupported_operation(node, "unexpected output size")
        input_list = self._resolve_name(source.name)
        instruction = INDEX_MAP_KERNEL.instruction_class()(
            INDEX_MAP_KERNEL.target_name,
            target_list,
            self._argument(input_list, source),
        )
        instruction.options = {"indices": indices, "fill": fill}
        self._emit(instruction, [input_list], target_list)

    def _compile_causal_score(self, node):
        fusion = self.graph.causal_score_fusions[node.name]
        final_node, q, k = fusion.mask, fusion.query, fusion.key
        scale, transposed_k, scale_node = fusion.scale, fusion.transposed_key, fusion.scale_node
        target_list = self.scope.get_list_for_node(final_node)
        self.last_target_list = target_list
        input_lists = [self._resolve_name(q.name), self._resolve_name(k.name)]
        args = [
            Argument(input_lists[0], _get_shape(q)),
            Argument(input_lists[1], _get_shape(k)),
            Argument("C_scale", torch.Size([]), value=scale),
            Argument(
                "C_query_offset",
                torch.Size([]),
                value=self.graph.qkv_offsets.get(q.name, 0),
            ),
            Argument(
                "C_key_offset",
                torch.Size([]),
                value=self.graph.qkv_offsets.get(k.name, 0),
            ),
        ]
        instruction = self._kernel_instruction(CAUSAL_SCORE_KERNEL, target_list, *args)
        self._emit(instruction, input_lists, target_list)

        # Mirror the dependency releases of the materialized chain that this
        # single instruction replaces.
        self.scope.release_dependencies(transposed_k)
        self.scope.release_dependencies(node)
        if scale_node is not None:
            self.scope.release_dependencies(scale_node)
        self.scope.release_dependencies(final_node)

    def _compile_qkv_linear(self, node):
        projection = self.graph.qkv_linear_fusions[node.name]
        query_heads, kv_heads = projection.query_heads, projection.kv_heads
        target_list = self.scope.get_list_for_node(node)
        self.last_target_list = target_list
        node_args = list(node.args)
        input_lists = self._resolve_args(node)
        if len(node_args) == 2:
            node_args.append(None)
            input_lists.append("_none")
        args = []
        for value, name in zip(node_args, input_lists):
            if value is None:
                args.append(Argument(name, torch.Size([]), value=None))
            else:
                args.append(self._argument(name, value))
        args.extend(
            (
                Argument("C_query_heads", torch.Size([]), value=query_heads),
                Argument("C_kv_heads", torch.Size([]), value=kv_heads),
            )
        )
        instruction = self._kernel_instruction(QKV_LINEAR_KERNEL, target_list, *args)
        self._emit(instruction, input_lists, target_list)
        self.scope.release_dependencies(node)

    def _compile_embedding_add(self, node):
        first, second = self.graph.embedding_add_fusions[node.name]
        target_list = self.scope.get_list_for_node(node)
        self.last_target_list = target_list
        tensor_nodes = (first.args[0], first.args[1], second.args[0], second.args[1])
        input_lists = [self._resolve_name(value.name) for value in tensor_nodes]
        args = [self._argument(name, value) for name, value in zip(input_lists, tensor_nodes)]
        instruction = self._kernel_instruction(EMBEDDING_ADD_KERNEL, target_list, *args)
        instruction.options = {
            "interleaved_sides": (
                self._uses_grouped_tied_weight(first),
                self._uses_grouped_tied_weight(second),
            )
        }
        self._emit(instruction, input_lists, target_list)
        self.scope.release_dependencies(first)
        self.scope.release_dependencies(second)
        self.scope.release_dependencies(node)

    def _compile_silu_mul(self, node):
        silu, gate, value = self.graph.silu_mul_fusions[node.name]
        target_list = self.scope.get_list_for_node(node)
        self.last_target_list = target_list
        input_lists = [self._resolve_name(gate.name), self._resolve_name(value.name)]
        args = [self._argument(input_lists[0], gate), self._argument(input_lists[1], value)]
        instruction = self._kernel_instruction(SWIGLU_KERNEL, target_list, *args)
        self._emit(instruction, input_lists, target_list)
        self.scope.release_dependencies(silu)
        self.scope.release_dependencies(node)

    def _compile_paired_swiglu(self, node):
        silu, gate, value = self.graph.paired_swiglu_fusions[node.name]
        target_list = self.scope.get_list_for_node(node)
        self.last_target_list = target_list

        source = gate.args[0]
        gate_bias = gate.args[2] if len(gate.args) > 2 else None
        value_bias = value.args[2] if len(value.args) > 2 else None
        node_args = (source, gate.args[1], gate_bias, value.args[1], value_bias)
        input_lists = [
            self._resolve_name(argument.name) if hasattr(argument, "name") else "_none"
            for argument in node_args
        ]
        args = [
            self._argument(name, argument)
            if hasattr(argument, "name")
            else Argument(name, torch.Size([]), value=None)
            for argument, name in zip(node_args, input_lists)
        ]
        input_lists, shard_rows = self._prepare_paired_linear_weights(
            node,
            gate,
            value,
            input_lists,
        )
        # The kernel takes the input, each shard's gate and value weights,
        # then both biases, in the order of ``input_lists``.
        if len(shard_rows) > 1:
            inner = args[1].shape[-1]
            shapes = [torch.Size([rows, inner]) for rows in shard_rows for _ in "gv"]
        else:
            shapes = [args[1].shape, args[3].shape]
        weights = [Argument(name, shape) for name, shape in zip(input_lists[1:-2], shapes)]
        instruction = self._kernel_instruction(
            PAIRED_LINEAR_SWIGLU_KERNEL,
            target_list,
            args[0],
            *weights,
            args[2],
            args[4],
            options={"shard_rows": shard_rows},
        )
        self._emit(instruction, input_lists, target_list)
        self.scope.release_dependencies(gate)
        self.scope.release_dependencies(value)
        self.scope.release_dependencies(silu)
        self.scope.release_dependencies(node)

    def _compile_fused_arithmetic(self, node):
        chain = self.graph.elementwise_fusions[node.name]
        target_list = self.scope.get_list_for_node(node)
        self.last_target_list = target_list

        external_nodes = []
        external_slots = {}
        steps = {}

        def operand(value):
            if not hasattr(value, "name"):
                return ("constant", value)
            if value.name in steps:
                return ("step", steps[value.name])
            if value.name not in external_slots:
                external_slots[value.name] = len(external_nodes)
                external_nodes.append(value)
            return ("input", external_slots[value.name])

        chain_steps = []
        for fused_node in chain:
            arguments = (
                fused_node.args[:1]
                if str(fused_node.target) == "aten.neg.default"
                else fused_node.args[:2]
            )
            chain_steps.append(
                (str(fused_node.target), tuple(operand(value) for value in arguments))
            )
            steps[fused_node.name] = len(chain_steps) - 1

        input_lists = [self._resolve_name(value.name) for value in external_nodes]
        args = [self._argument(name, value) for name, value in zip(input_lists, external_nodes)]
        instruction = self._kernel_instruction(
            ARITHMETIC_EXPRESSION_KERNEL,
            target_list,
            *args,
            options={"steps": tuple(chain_steps), "size": math.prod(_get_shape(node))},
        )
        self._emit(instruction, input_lists, target_list)
        for fused_node in chain:
            self.scope.release_dependencies(fused_node)

    def _compile_fused_linear(self, node):
        fusion = self.graph.linear_fusions[node.name]
        final_node, epilogue, chain = fusion.final, fusion.epilogue, fusion.chain
        target_list = self.scope.get_list_for_node(final_node)
        self.last_target_list = target_list

        input_lists = self._resolve_args(node)
        node_args = list(node.args)
        if len(node_args) == 2:
            node_args.append(None)
            input_lists.append("_none")

        residual_arg = next(
            (operand for operation, operand in epilogue if operation == "tensor_add"),
            None,
        )
        if residual_arg is None:
            node_args.append(None)
            input_lists.append("_none")
        else:
            node_args.append(residual_arg)
            input_lists.append(self._resolve_name(residual_arg.name))
        epilogue = tuple(
            (operation, None if operation == "tensor_add" else operand)
            for operation, operand in epilogue
        )

        args = []
        for node_arg, name in zip(node_args, input_lists):
            if node_arg is None:
                args.append(Argument(name, torch.Size([]), value=None))
            elif hasattr(node_arg, "name"):
                args.append(self._argument(name, node_arg))
            else:
                args.append(Argument(name, torch.Size([]), value=node_arg))

        input_lists, weight_shard_rows, interleaved = self._prepare_linear_weight(
            node,
            input_lists,
        )
        instruction = self._linear_instruction(
            target_list,
            self._linear_arguments(args, input_lists, weight_shard_rows),
            interleaved=interleaved,
            epilogue=epilogue,
            fast=self.graph.fast and self.graph.fast_config.activations,
        )
        self._emit(instruction, input_lists, target_list)
        self.scope.release_dependencies(node)
        for fused_node in chain:
            self.scope.release_dependencies(fused_node)

    # ── standard instruction ─────────────────────────────────────────────

    def _compile_instruction(self, node, target_list, aten_op):
        rule = self.graph.registry.operation(aten_op)
        detail = (
            None
            if node.name in self.graph.causal_value_matmuls
            else self.graph.registry.validate(
                node,
                fast=self.graph.fast,
                config=self.graph.fast_config,
                dynamic_axes=self.graph.dynamic_axes,
                resolve_weight=self.graph.resolve_weight,
            )
        )
        if detail is not None:
            raise _unsupported_operation(node, detail)
        if aten_op in {
            "cattorch.expert_family_moe.default",
            "cattorch.routed_experts.default",
        }:
            value_nodes = list(node.args[:3])
            bank_nodes = list(node.args[3])
            tensor_nodes = [*value_nodes, *bank_nodes]
            input_lists = [self._resolve_name(value.name) for value in tensor_nodes]
            args = [self._argument(name, value) for name, value in zip(input_lists, tensor_nodes)]
            handle = int(node.args[4])
            options = {
                "family": self.graph.registry.expert_family(handle),
                "fast_activations": self.graph.fast and self.graph.fast_config.activations,
                "fast_layer_norm": self.graph.fast and self.graph.fast_config.layer_norm,
            }
            if aten_op == "cattorch.routed_experts.default":
                kernel = ROUTED_EXPERTS_KERNEL
            else:
                kernel = EXPERT_FAMILY_MOE_KERNEL
                options.update(top_k=int(node.args[5]), normalize_selected=bool(node.args[6]))
            instruction = self._kernel_instruction(kernel, target_list, *args, options=options)
            family = self.graph.registry.expert_family(handle)
            template_state = {}
            template_state.update(
                dict(
                    family.template.named_parameters(remove_duplicate=False),
                )
            )
            template_state.update(
                dict(
                    family.template.named_buffers(remove_duplicate=False),
                )
            )
            locations = tuple(dict.fromkeys(family._bank_names.values()))
            first_names = {
                location: next(
                    name for name in family._state_order if family._bank_names[name] == location
                )
                for location in locations
            }
            for position, location in enumerate(locations):
                key = input_lists[3 + position].removeprefix("W_")
                self.static_effective_ndims[key] = template_state[first_names[location]].ndim
            self._emit(instruction, input_lists, target_list)
            for approximated in instruction.context.approximated:
                self.approximate[approximated] += 1
            return
        lowering_target = (
            node.target if rule is None or rule.lowering_target is None else rule.lowering_target
        )
        input_lists = self._resolve_args(node)
        # Automatic generation binds a step module's cache buffers to the
        # persistent K/V lists and drops its position input (the kernels read
        # `cattorch cache length`), so none of them becomes a static weight.
        for position, bound in self._automatic_bindings(node).items():
            input_lists[position] = bound
        node_args = list(node.args)
        # The library Linear kernel takes a bias and then a residual slot
        # for a fused add, both empty here. A user's kernel gets the call as
        # traced.
        if aten_op == "aten.linear.default" and aten_op not in self.graph.protected:
            if len(node_args) == 2:
                node_args.append(None)
                input_lists.append("_none")
            node_args.append(None)
            input_lists.append("_none")
        # GELU's form is keyword-only in the captured graph.
        if aten_op == "aten.gelu.default" and len(node_args) == 1:
            approximate = node.kwargs.get("approximate", "none")
            node_args.append(approximate)
            input_lists.append(f"C_{approximate}")
        if aten_op in {"aten.conv1d.default", "aten.conv2d.default"}:
            dimensions = 1 if aten_op == "aten.conv1d.default" else 2
            defaults = (None, [1] * dimensions, [0] * dimensions)
            while len(node_args) < 5:
                value = defaults[len(node_args) - 2]
                node_args.append(value)
                input_lists.append("_none" if value is None else f"C_{value}")
        # Other keyword arguments, such as ``include_self``, take their schema
        # positions; skipped ones before them get their defaults.
        schema = getattr(node.target, "_schema", None)
        if (
            schema is not None
            and node.kwargs
            and not any(hasattr(value, "name") for value in node.kwargs.values())
        ):
            names = [argument.name for argument in schema.arguments]
            last = max(
                (names.index(key) for key in node.kwargs if key in names),
                default=-1,
            )
            for argument in schema.arguments[len(node_args) : last + 1]:
                value = node.kwargs.get(
                    argument.name,
                    argument.default_value if argument.has_default_value() else None,
                )
                node_args.append(value)
                input_lists.append("_none" if value is None else f"C_{value}")
        log.info("%s -> %s (%s) (Inputs: %s)", aten_op, target_list, node.name, input_lists)

        args = []
        for i, name in enumerate(input_lists):
            node_arg = node_args[i]
            if node_arg is None:
                args.append(Argument(name, torch.Size([]), value=None))
            elif hasattr(node_arg, "name"):
                args.append(self._argument(name, node_arg))
            else:
                args.append(Argument(name, torch.Size([]), value=node_arg))

        if (
            aten_op in MATMUL_OPS
            and aten_op not in self.graph.protected
            and len(node.args) > 1
            and hasattr(node.args[1], "name")
            and not (self.graph.fast and self.graph.fast_config.weights.enabled)
        ):
            rhs = self.graph.resolve_weight(node.args[1].name)
            if (
                isinstance(rhs, torch.Tensor)
                and rhs.ndim > 2
                and math.prod(rhs.shape[:-2]) == 1
                and self.row_aligned_weights
            ):
                # A batch of one matrix, as ``torch.einsum`` lowers a product
                # with a weight on the right: one matrix, stored like one.
                rhs = rhs.reshape(rhs.shape[-2:])
            if (
                isinstance(rhs, torch.Tensor)
                and rhs.ndim == 2
                and rhs.is_floating_point()
                # Quantization groups every matrix over its input width, so
                # store quantized weights output-major like linear weights.
                and (rhs.numel() > SCRATCH_LIST_LIMIT or self.row_aligned_weights)
            ):
                large = self._large_static_matmul(node, target_list, args[0], rhs)
                if large is not None:
                    instruction, matrix_inputs = large
                    self._emit(instruction, matrix_inputs, target_list)
                    return
            if (
                isinstance(rhs, torch.Tensor)
                and rhs.ndim > 2
                and rhs.is_floating_point()
                and self.row_aligned_weights
                and len(args[0].shape) == rhs.ndim
                and tuple(args[0].shape[:-2]) == tuple(rhs.shape[:-2])
            ):
                # A stored batch of matrices on the right, such as per-head
                # einsum weights: keep each one transposed, output-major,
                # so integer storage groups it over its inputs like any
                # other quantized matrix.
                key = f"matmul_{node.name}_weight_t"
                weight = rhs.transpose(-1, -2).contiguous()
                self.static_lists[key] = weight
                instruction = self._kernel_instruction(
                    matmul_kernel(aten_op),
                    target_list,
                    args[0],
                    Argument(f"W_{key}", weight.shape, static=True),
                    options={"transposed_right": True},
                )
                self._emit(instruction, [input_lists[0], f"W_{key}"], target_list)
                return
        kernel = self.graph.registry.kernel(
            aten_op,
            fast=self.graph.fast,
            config=self.graph.fast_config,
        )
        library_linear = getattr(kernel, "kernel", None) is LINEAR_KERNEL
        if library_linear:
            input_lists, weight_shard_rows, interleaved = self._prepare_linear_weight(
                node,
                input_lists,
            )
            args = self._linear_arguments(args, input_lists, weight_shard_rows)
        static_rhs = None
        if (
            self.graph.fast
            and aten_op in MATMUL_OPS
            and aten_op not in self.graph.protected
            and len(node.args) > 1
            and hasattr(node.args[1], "name")
            and self.graph.fast_config.weights.enabled
        ):
            candidate = self.graph.resolve_weight(node.args[1].name)
            if isinstance(candidate, torch.Tensor) and candidate.ndim == 2:
                static_rhs = candidate

        if static_rhs is not None:
            from cattorch.fast import prepare_fast_matmul_weights

            factors = prepare_fast_matmul_weights(
                static_rhs,
                self.graph.fast_config.weights,
            )
            if len(factors) == 2:
                factor_lists = []
                factor_args = []
                for index, factor in enumerate(factors):
                    key = f"fast_{node.name}_factor_{index}"
                    self.static_lists[key] = factor
                    factor_lists.append(f"W_{key}")
                    factor_args.append(Argument(f"W_{key}", factor.shape, static=True))
                options = {}
                if args[0].dynamic_axis is not None:
                    left = node.args[0]
                    options["most_rows"] = (
                        self.graph.runtime_capacity(
                            left.name,
                            _get_shape(left),
                        )
                        // _get_shape(left)[-1]
                    )
                instruction = self._kernel_instruction(
                    LOW_RANK_MATMUL_KERNEL,
                    target_list,
                    args[0],
                    *factor_args,
                    options=options,
                )
                self._emit(
                    instruction,
                    [input_lists[0], *factor_lists],
                    target_list,
                )
                return

            # A pruned copy: other operations may read the original.
            key = f"fast_{node.name}_pruned"
            self.static_lists[key] = factors[0]
            input_lists[1] = f"W_{key}"
            args[1] = Argument(f"W_{key}", factors[0].shape, static=True)
            pruned = pruned_matmul_blocks(factors[0]) is not None
            instruction = self._kernel_instruction(
                (pruned_matmul_kernel if pruned else matmul_kernel)(aten_op),
                target_list,
                *args,
            )
        elif node.name in self.graph.causal_softmaxes:
            if self.graph.fast and self.graph.fast_config.softmax:
                instruction = self._kernel_instruction(
                    FAST_CAUSAL_SOFTMAX_KERNEL,
                    target_list,
                    *args,
                )
                self.approximate[aten_op] += 1
            else:
                instruction = self._kernel_instruction(CAUSAL_SOFTMAX_KERNEL, target_list, *args)
        elif aten_op == "aten.embedding.default":
            kernel = self.graph.registry.kernel(
                aten_op,
                fast=self.graph.fast,
                config=self.graph.fast_config,
            )
            instruction = self._instantiate(
                kernel,
                lowering_target,
                node,
                target_list,
                args,
            )
            instruction.options = {
                **instruction.options,
                "interleaved_weight": self._uses_grouped_tied_weight(node),
            }
        elif node.name in self.graph.causal_value_matmuls:
            value_offset = self.graph.qkv_offsets.get(node.args[1].name, 0)
            instruction = self._kernel_instruction(
                CAUSAL_VALUE_MATMUL_KERNEL,
                target_list,
                *args,
                Argument("C_value_offset", torch.Size([]), value=value_offset),
            )
        else:
            if kernel is None:
                raise _unsupported_operation(node)
            if self.graph.fast and rule is not None and kernel is rule.fast_kernel:
                self.approximate[aten_op] += 1
            if library_linear:
                instruction = self._linear_instruction(target_list, args, interleaved=interleaved)
            else:
                instruction = self._instantiate(
                    kernel,
                    lowering_target,
                    node,
                    target_list,
                    args,
                )
        instruction.transform_weights(self.static_lists)
        self._emit(instruction, input_lists, target_list)
        # Context kernels report the scratch lists they allocated, so the
        # sharding pass can size them like any other tensor list.
        for list_name, size in getattr(instruction, "temporary_sizes", {}).items():
            self.logical_sizes[list_name] = max(
                self.logical_sizes.get(list_name, 0),
                size,
            )

    # ── cat (pairwise chaining) ──────────────────────────────────────────

    def _compile_cat(self, node, target_list):
        tensor_list = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else 0

        resolved = [(self._resolve_name(t.name), _get_shape(t)) for t in tensor_list]
        if dim < 0:
            dim = len(resolved[0][1]) + dim

        if len(resolved) == 1:
            source_list, source_shape = resolved[0]
            size = math.prod(source_shape)
            args = (
                Argument(source_list, source_shape),
                Argument("C_chunk_size", torch.Size([]), value=size),
                Argument("C_num_rows", torch.Size([]), value=1),
                Argument("C_skip", torch.Size([]), value=0),
                Argument("C_offset", torch.Size([]), value=0),
            )
            instruction = self._kernel_instruction(GETITEM_KERNEL, target_list, *args)
            self._emit(instruction, [source_list], target_list)
            return

        cur_list, cur_shape = resolved[0]
        accumulator_temp_id = None
        for pair_idx in range(1, len(resolved)):
            next_list, next_shape = resolved[pair_idx]
            is_last = pair_idx == len(resolved) - 1

            if is_last:
                cat_target = target_list
            else:
                temp_id = self.scope.acquire_list()
                cat_target = f"T{temp_id}"
                self.dynamic_lists.add(cat_target)

            num_outer = math.prod(cur_shape[:dim]) if dim > 0 else 1
            chunk_1 = math.prod(cur_shape[dim:])
            chunk_2 = math.prod(next_shape[dim:])
            output_size = math.prod(cur_shape) + math.prod(next_shape)
            self.logical_sizes[cat_target] = max(
                self.logical_sizes.get(cat_target, 0),
                output_size,
            )
            cat_inputs = [cur_list, next_list]

            log.info(
                "cat(dim=%d) [%d/%d] -> %s (%s) (Inputs: %s)",
                dim,
                pair_idx,
                len(resolved) - 1,
                cat_target,
                node.name,
                cat_inputs,
            )

            cat_args = [
                Argument(cur_list, cur_shape),
                Argument(next_list, next_shape),
                Argument("C_num_outer", torch.Size([]), value=num_outer),
                Argument("C_chunk_1", torch.Size([]), value=chunk_1),
                Argument("C_chunk_2", torch.Size([]), value=chunk_2),
            ]
            instruction = self._kernel_instruction(CAT_KERNEL, cat_target, *cat_args)
            self._emit(instruction, cat_inputs, cat_target)

            out_shape = list(cur_shape)
            out_shape[dim] = cur_shape[dim] + next_shape[dim]
            out_shape = torch.Size(out_shape)

            previous_temp_id = accumulator_temp_id
            if not is_last:
                accumulator_temp_id = temp_id
            cur_list, cur_shape = cat_target, out_shape

            if previous_temp_id is not None:
                self.scope.release_list(previous_temp_id)

    # ── slice ────────────────────────────────────────────────────────────

    def _compile_slice(self, node, target_list):
        source = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else 0
        start = node.args[2] if len(node.args) > 2 else 0
        end = node.args[3] if len(node.args) > 3 else None
        step = node.args[4] if len(node.args) > 4 else 1
        input_shape = _get_shape(source)

        if dim < 0:
            dim = len(input_shape) + dim
        dim_size = input_shape[dim]
        start, end, step = slice(start, end, step).indices(dim_size)
        src_list = self._resolve_name(source.name)
        trailing = math.prod(input_shape[dim + 1 :]) if dim + 1 < len(input_shape) else 1
        if step != 1:
            selected = len(range(start, end, step))
            num_rows = math.prod(input_shape[:dim]) if dim > 0 else 1
            row_stride = math.prod(input_shape[dim:])
            input_lists = [src_list]
            args = [
                Argument(src_list, input_shape),
                Argument("C_num_rows", torch.Size([]), value=num_rows),
                Argument("C_selected", torch.Size([]), value=selected),
                Argument("C_trailing", torch.Size([]), value=trailing),
                Argument("C_row_stride", torch.Size([]), value=row_stride),
                Argument("C_offset", torch.Size([]), value=start * trailing),
                Argument("C_step_skip", torch.Size([]), value=(step - 1) * trailing),
            ]
            instruction = self._kernel_instruction(STRIDED_SLICE_KERNEL, target_list, *args)
            self._emit(instruction, input_lists, target_list)
            return

        chunk_size = max(0, end - start) * trailing
        num_rows = math.prod(input_shape[:dim]) if dim > 0 else 1
        row_stride = math.prod(input_shape[dim:])
        skip = row_stride - chunk_size
        offset = start * trailing

        input_lists = [src_list]
        log.info(
            "slice(dim=%d, %d:%d) -> %s (%s) (Inputs: %s)",
            dim,
            start,
            end,
            target_list,
            node.name,
            input_lists,
        )

        args = [
            Argument(src_list, input_shape),
            Argument("C_chunk_size", torch.Size([]), value=chunk_size),
            Argument("C_num_rows", torch.Size([]), value=num_rows),
            Argument("C_skip", torch.Size([]), value=skip),
            Argument("C_offset", torch.Size([]), value=offset),
        ]
        instruction = self._kernel_instruction(GETITEM_KERNEL, target_list, *args)
        self._emit(instruction, input_lists, target_list)

    # ── getitem on split/chunk ───────────────────────────────────────────

    def _compile_getitem(self, node, target_list):
        split_node = node.args[0]
        chunk_index = node.args[1]
        input_shape, split_size, dim = self.graph.split_meta[split_node.name]

        source_name = self.graph.aliases[split_node.name]
        src_list = self._resolve_name(source_name)

        trailing = math.prod(input_shape[dim + 1 :]) if dim + 1 < len(input_shape) else 1
        row_stride = math.prod(input_shape[dim:])
        if isinstance(split_size, (list, tuple)):
            output_count = len(split_size)
            if chunk_index < 0:
                chunk_index += output_count
            chunk_size = split_size[chunk_index]
            offset = sum(split_size[:chunk_index])
        else:
            dimension_size = input_shape[dim]
            output_count = math.ceil(dimension_size / split_size)
            if chunk_index < 0:
                chunk_index += output_count
            chunk_size = min(
                split_size,
                dimension_size - chunk_index * split_size,
            )
            offset = chunk_index * split_size
        num_rows = math.prod(input_shape[:dim]) if dim > 0 else 1
        chunk_size *= trailing
        offset *= trailing
        skip = row_stride - chunk_size

        input_lists = [src_list]
        log.info(
            "getitem[%d] -> %s (%s) (Inputs: %s)", chunk_index, target_list, node.name, input_lists
        )

        args = [
            Argument(src_list, input_shape),
            Argument("C_chunk_size", torch.Size([]), value=chunk_size),
            Argument("C_num_rows", torch.Size([]), value=num_rows),
            Argument("C_skip", torch.Size([]), value=skip),
            Argument("C_offset", torch.Size([]), value=offset),
        ]
        instruction = self._kernel_instruction(GETITEM_KERNEL, target_list, *args)
        self._emit(instruction, input_lists, target_list)

    # ── helpers ──────────────────────────────────────────────────────────

    def _resolve_args(self, node) -> list[str]:
        """Map a node's args to Scratch list names."""
        input_lists = []
        for arg in node.args:
            if arg is None:
                input_lists.append("_none")
            elif hasattr(arg, "name"):
                input_lists.append(self._resolve_name(arg.name))
            else:
                input_lists.append(f"C_{arg}")
        return input_lists

    def _prepare_linear_weight(self, node, input_lists):
        """Lay out dense Linear rows for grouped execution and safe sharding.

        Four-row groups are interleaved at export time so the hot kernel needs
        one running weight index. Oversized matrices are split only between
        complete output rows (and complete four-row groups), keeping shard
        selection outside every dot-product loop.
        """
        weight = self.graph.resolve_weight(node.args[1].name)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            return input_lists, (), False
        if self.graph.fast and self.graph.fast_config.weights.enabled:
            # Structured pruning owns this weight's layout and sparse metadata;
            # low-rank replacement may also rewrite the surrounding graph.
            return input_lists, (), False
        columns, inner = weight.shape
        identity = _tensor_identity(weight)
        table = input_lists[1] if input_lists[1].startswith("W_") else None
        tied = table is not None and identity in self._tied_weights
        plan = self._row_shards(weight, tied=tied)
        if plan is None:
            # Not one complete row (four-row group) fits in a list. The kernel
            # reads the whole weight through the generic sharder instead.
            return input_lists, (), False
        grouped, rows_per_shard = plan
        oversized = rows_per_shard < columns
        if not grouped and not oversized:
            return input_lists, (), False

        interleaved_table = tied and grouped and identity in self._grouped_tied_weights
        if interleaved_table and not oversized:
            # Keep one physical copy of a tied token embedding/output head.
            # The embedding kernel reads this four-row interleaved layout with
            # a stride-four inner loop.
            original_key = table.removeprefix("W_")
            self._store_row_shards(weight, columns, interleaved=True, key=lambda _: original_key)
            return input_lists, (columns,), True

        shards = self._store_row_shards(
            weight,
            rows_per_shard,
            interleaved=grouped,
            key=lambda index: f"linear_{node.name}_weight_{index}",
        )
        names = [name for name, _ in shards]
        tensors = [tensor for _, tensor in shards]
        row_counts = [tensor.shape[0] for tensor in tensors]
        if interleaved_table:
            # The table becomes the shards end to end, sharded at the same
            # row boundaries, so each table shard can be the head's list.
            original_key = table.removeprefix("W_")
            self.static_lists[original_key] = torch.cat(tensors, dim=0)
            self.static_shard_alignments[table] = 4 * inner
            self.tied_storage_aliases[table] = tuple(names)
            if self.row_aligned_weights:
                # Storage must see the four-row interleave to recognize the
                # quantized grid and choose the head's precision and shard
                # sizes.
                self.interleaved_static_layouts[original_key] = tuple(
                    (start * inner, count, inner)
                    for start, count in zip(range(0, columns, rows_per_shard), row_counts)
                )
        elif tied and not grouped:
            # Plain row shards: the table itself, sharded at their boundaries.
            self.static_shard_alignments[table] = inner
            self.tied_storage_aliases[table] = tuple(names)
        elif tied:
            self.tied_storage_copies[table] = (
                "a Linear reads it in four-row groups, and another operation "
                "or a custom kernel reads it in rows"
            )
        return [input_lists[0], *names, *input_lists[2:]], tuple(row_counts), grouped

    def _prepare_paired_linear_weights(self, node, gate, value, input_lists):
        gate_weight = self.graph.resolve_weight(gate.args[1].name)
        value_weight = self.graph.resolve_weight(value.args[1].name)
        if (
            not isinstance(gate_weight, torch.Tensor)
            or not isinstance(value_weight, torch.Tensor)
            or gate_weight.ndim != 2
            or gate_weight.shape != value_weight.shape
        ):
            return [
                input_lists[0],
                input_lists[1],
                input_lists[3],
                input_lists[2],
                input_lists[4],
            ], ()
        hidden, inner = gate_weight.shape
        rows_per_shard = SCRATCH_LIST_LIMIT // inner
        if rows_per_shard < 1:
            return [
                input_lists[0],
                input_lists[1],
                input_lists[3],
                input_lists[2],
                input_lists[4],
            ], ()
        if gate_weight.numel() <= SCRATCH_LIST_LIMIT:
            return [
                input_lists[0],
                input_lists[1],
                input_lists[3],
                input_lists[2],
                input_lists[4],
            ], (hidden,)

        names = []
        row_counts = []
        for index, start in enumerate(range(0, hidden, rows_per_shard), 1):
            end = min(hidden, start + rows_per_shard)
            gate_key = f"swiglu_{node.name}_gate_weight_{index}"
            value_key = f"swiglu_{node.name}_value_weight_{index}"
            self.static_lists[gate_key] = gate_weight[start:end].detach().clone()
            self.static_lists[value_key] = value_weight[start:end].detach().clone()
            self.static_source_numels[gate_key] = gate_weight.numel()
            self.static_source_numels[value_key] = value_weight.numel()
            names.extend((f"W_{gate_key}", f"W_{value_key}"))
            row_counts.append(end - start)
        return [input_lists[0], *names, input_lists[2], input_lists[4]], tuple(row_counts)

    def _large_static_matmul(self, node, target_list, left_arg, rhs):
        """Use row-aligned output-major weights only when sharding dominates."""
        weight = rhs.T.contiguous()
        columns, inner = weight.shape
        plan = self._row_shards(weight)
        if plan is not None:
            interleaved, rows_per_shard = plan
        elif self.row_aligned_weights:
            # Not one complete row (four-row group) fits in a list. Integer
            # storage needs the weight output-major, as it was quantized, so
            # keep it whole and plain, as a Linear's weight of this shape is
            # kept, for the generic sharder to split.
            interleaved, rows_per_shard = False, columns
        else:
            # The plain matmul reads the stored weight through the generic
            # sharder instead.
            return None

        shards = self._store_row_shards(
            weight,
            rows_per_shard,
            interleaved=interleaved,
            key=lambda index: f"matmul_{node.name}_weight_{index}",
        )
        names = [name for name, _ in shards]
        row_counts = [tensor.shape[0] for _, tensor in shards]

        matrix_inputs = [left_arg.name, *names, "_none", "_none"]
        empty = Argument("_none", torch.Size([]), value=None)
        instruction = self._linear_instruction(
            target_list,
            (left_arg, *self._weight_shards(names, row_counts, inner), empty, empty),
            interleaved=interleaved,
        )
        return instruction, matrix_inputs

    def _resolve_name(self, arg_name: str) -> str:
        """Resolve a single node name to its Scratch list name."""
        visited = set()
        while arg_name in self.graph.aliases:
            if arg_name in visited:
                raise UnsupportedModelError(
                    f"alias cycle while resolving exported node {arg_name!r}"
                )
            visited.add(arg_name)
            arg_name = self.graph.aliases[arg_name]
        if arg_name in self.scope.assignments:
            name = f"T{self.scope.assignments[arg_name]}"
            self.dynamic_lists.add(name)
            return name
        if arg_name in self.graph.input_names:
            return self.graph.input_names[arg_name]
        tensor = self.graph.resolve_weight(arg_name)
        if isinstance(tensor, torch.Tensor):
            identity = _tensor_identity(tensor)
            if identity in self._static_tensor_names:
                return self._static_tensor_names[identity]
        name = f"W_{arg_name}"
        self.static_lists[arg_name] = tensor
        if isinstance(tensor, torch.Tensor):
            self._static_tensor_names[identity] = name
        return name

    def _emit(self, instruction, input_lists, target_list):
        """Generate Scratch blocks for an instruction and merge into sprite."""
        if self.active_node in self.graph.row_loops:
            data = self._row_looped(instruction, input_lists, target_list)
        else:
            data = instruction.finalize()

        self.block_manager.new_context()
        data["blocks"] = self.block_manager.apply_to_blocks(data["blocks"])
        roots = [
            bid
            for bid, block in data["blocks"].items()
            if block.get("topLevel") and block.get("parent") is None
        ]
        if len(roots) != 1:
            raise ValueError(f"expected one instruction root, got {roots}")
        if self.active_node is not None:
            self.emitted_roots.append((self.active_node, roots[0]))

        tensor_lists = [n for n in input_lists if not n.startswith("C_")]
        all_lists = tensor_lists + [target_list]

        data = TensorAdder().apply(data, all_lists)
        data["blocks"] = TensorReplacer(data, all_lists).apply(data["blocks"])

        if self.sprite is None:
            self.sprite = data
        else:
            self.sprite = combine(self.sprite, data, in_place=True)

    def _row_looped(self, instruction, input_lists, target_list) -> dict:
        """Run a one-row kernel once per row of its runtime-length inputs."""
        from cattorch.util.instruction.row_loop import captured_programs, row_loop

        node = next(node for node in self.graph.nodes if node.name == self.active_node)
        with captured_programs() as programs:
            instruction.finalize()
        if len(programs) != 1:
            raise _unsupported_operation(
                node,
                "its kernel can't run once per row of a variable-length input",
            )
        # Variable-length inputs, found by list name, or by argument for a
        # fusion, whose inputs (such as the SiLU of a SiLU-times product)
        # aren't the node's own.
        dynamic = {
            self._resolve_name(arg.name): math.prod(_get_shape(arg))
            for arg in node.all_input_nodes
            if arg.name in self.graph.dynamic_axes
        }
        tensor_lists = [name for name in input_lists if not name.startswith("C_")]
        tensor_args = [
            argument for argument in instruction.args if not str(argument.name).startswith("C_")
        ]
        rows = {}
        for position, name in enumerate(tensor_lists, 1):
            argument = tensor_args[position - 1] if position <= len(tensor_args) else None
            if name in dynamic:
                rows[f"T{position}"] = dynamic[name]
            elif argument is not None and getattr(argument, "dynamic_axis", None) is not None:
                rows[f"T{position}"] = math.prod(argument.shape)
        if not rows:
            raise _unsupported_operation(
                node,
                "its kernel can't run once per row of a variable-length input",
            )
        output = f"T{len(tensor_lists) + 1}"
        return row_loop(programs[0], rows, output, math.prod(_get_shape(node))).compile()

    @staticmethod
    def _is_full_slice(node) -> bool:
        """True if the slice covers the full dimension (i.e. is a no-op)."""
        dim = node.args[1] if len(node.args) > 1 else 0
        start = node.args[2] if len(node.args) > 2 else 0
        end = node.args[3] if len(node.args) > 3 else None
        step = node.args[4] if len(node.args) > 4 else 1
        input_shape = _get_shape(node.args[0])
        if dim < 0:
            dim = len(input_shape) + dim
        dim_size = input_shape[dim]
        start, end, step = slice(start, end, step).indices(dim_size)
        return start == 0 and end == dim_size and step == 1


# ── Public API ───────────────────────────────────────────────────────────────


def _size_numbered_private_lists(sprite: dict, sizes: dict[str, int]) -> None:
    """Give ``_name_2``-style copies of a private list its recorded capacity.

    The block combiner keeps every instruction's private list separate by
    numbering later duplicates, so a size recorded for ``_name`` applies to
    all of them.
    """
    for name, _ in sprite.get("lists", {}).values():
        base = re.sub(r"_\d+$", "", name)
        if base != name and base.startswith("_") and base in sizes:
            sizes[name] = max(sizes.get(name, 0), sizes[base])


def _select_unroll_factor(
    model: torch.nn.Module,
    storage: StorageConfig,
    codegen: CodegenConfig,
) -> int:
    if codegen.unrolling == "compact":
        return 1
    if codegen.unrolling == "speed" or codegen.target_json_bytes is None:
        return 8

    seen = set()
    effective_ndims = expert_bank_effective_ndims(model)
    payload_chars = 0.0
    density = {
        "float32": 64 / 13,
        "float16": 32 / 13,
        "int8": 16 / 13,
        "int6": 12 / 13,
        # Conservatively budget fixed q4. Huffman coding can only improve it.
        "int4": 8 / 13,
    }
    # An automatic generation step module also holds caches and a position
    # counter, which are never stored; it names the tensors that are.
    static_tensors = getattr(model, "static_tensors", None)
    for tensor in (
        static_tensors() if static_tensors is not None else (*model.parameters(), *model.buffers())
    ):
        identity = _tensor_identity(tensor)
        if identity in seen:
            continue
        seen.add(identity)
        precision = storage.precision
        if not tensor.is_floating_point():
            precision = "float32"
        elif precision.startswith("int") and (
            effective_ndims.get(tensor.untyped_storage().data_ptr(), tensor.ndim) < 2
            or tensor.numel() < storage.min_quantized_values
        ):
            precision = "float16"
        payload_chars += tensor.numel() * density[precision]
        if precision.startswith("int"):
            group_size = effective_group_size(
                max(1, int(tensor.shape[-1])) if tensor.ndim else 1, storage.group_size,
            )
            scale_chars = 32 / 13 if storage.scale_precision == "float16" else 64 / 13
            payload_chars += math.ceil(tensor.numel() / group_size) * scale_chars

    # Shared decoders and lifecycle/interface code form a relatively stable
    # floor. Spend remaining JSON budget on reducing hot-loop dispatch.
    available = codegen.target_json_bytes - int(payload_chars) - 650_000
    if available >= 1_500_000:
        return 8
    if available >= 750_000:
        return 4
    if available >= 350_000:
        return 2
    return 1


def transpile(
    model: torch.nn.Module,
    example_inputs: (
        torch.Tensor
        | tuple[torch.Tensor, ...]
        | ExportProgram
        | GenerationProgram
        | RecurrentGenerationProgram
        | None
    ),
    output_path: str | Path,
    sig_figs: int | None = None,
    *,
    name: str | None = None,
    optimization: Literal["exact", "fast"] = "exact",
    fast_config: FastConfig | None = None,
    storage: StorageConfig | None = None,
    quantization: QuantizationConfig | None = None,
    calibration_inputs: Iterable[torch.Tensor | tuple[torch.Tensor, ...]] | None = None,
    codegen: CodegenConfig | None = None,
    adapters: tuple[ModuleAdapter, ...] = (),
    calibration_calls: tuple[ProgramCall, ...] | None = None,
    calibration_sequences: Iterable[torch.Tensor | tuple[torch.Tensor, ...]] | None = None,
    example_kwargs: dict[str, Any] | None = None,
    kernels: tuple[Kernel, ...] = (),
    dynamic: dict | None = None,
):
    """Export a PyTorch module as a Scratch ``.sprite3`` file.

    Parameters
    ----------
    model:
        Module to trace. Export-time weight transforms operate on an
        evaluation-mode copy and do not mutate this object; modules left in
        training mode are exported in evaluation mode with a warning.
    example_inputs:
        One tensor or a tuple defining a fixed interface, an ``ExportProgram``
        defining named methods and state, a ``GenerationProgram`` defining
        the cached decoder interface, or a ``RecurrentGenerationProgram``
        defining the same interface for a recurrent step. ``None`` takes the interface from the
        model's experimental markers: its ``@entrypoint`` methods, or the
        ``@example`` inputs of ``forward``.
    output_path:
        Destination path. ``.sprite3`` is appended when absent and missing
        parent directories are created.
    sig_figs:
        Optional positive significant-figure limit for static values. This is
        an independent lossy control.
    name:
        Scratch sprite display name. Defaults to the output filename stem.
    optimization:
        ``"exact"`` (default) or ``"fast"``. Exact describes kernel semantics;
        separately selected storage and weight transforms may still be lossy.
    fast_config:
        Approximation and weight-transform policy. Valid only in fast mode.
    storage:
        Serialized static-tensor compression policy.
    quantization:
        Symmetric or GPTQ integer matrix-weight policy.
    calibration_inputs:
        One-pass iterable of eager-model input tensors or tuples. Required by
        GPTQ fixed-interface exports and invalid for other exports.
    calibration_calls:
        Named whole-program GPTQ calls for an ``ExportProgram``.
    calibration_sequences:
        Token sequences used to calibrate a ``GenerationProgram`` or
        ``RecurrentGenerationProgram`` with GPTQ.
    codegen:
        Expanded-JSON budget, loop-unrolling, compact-ID, and optional
        structural-compaction policy.
    adapters:
        Scoped module adapters for an ``ExportProgram``,
        ``GenerationProgram`` or ``RecurrentGenerationProgram``. They apply to
        an export-owned model copy.
    example_kwargs:
        Keyword arguments for the traced call, like ``torch.export``'s
        ``kwargs``. Tensor values become inputs after the positional ones
        (``input_k`` in order); other values, such as ``use_cache=False``, are
        constants of the trace. Not valid for programs, and tensor values
        cannot be combined with GPTQ ``calibration_inputs``.
    kernels:
        Experimental ``cattorch.experimental.Kernel`` lowerings for this
        export. A kernel for an operation cattorch already lowers replaces the
        built-in one; an approximate kernel applies only in fast mode.
    dynamic:
        Bounded variable-length inputs, as ``{input: (minimum, maximum)}`` for
        an input's first dimension or ``{input: {dimension: (minimum,
        maximum)}}``; an input is its position or tensor keyword name. The
        sprite then accepts any length in the bounds. The export is checked
        against the model at several lengths and fails if the model uses the
        length in a way the sprite can't follow. Not valid for programs.

    Returns
    -------
    TranspileResult or ProgramResult
        Artifact path, size/count metadata, warnings, tensor interface,
        procedures, and any quantization report.

    Notes
    -----
    Operations without a fast implementation use the exact kernel. Unsupported
    operations raise an error instead of silently changing the model's
    behavior. See ``docs/api-reference.md`` and ``docs/supported-models.md``.
    """
    validate_export_options(
        name=name,
        optimization=optimization,
        fast_config=fast_config,
        storage=storage,
        quantization=quantization,
        codegen=codegen,
        sig_figs=sig_figs,
        adapters=adapters,
        example_kwargs=example_kwargs,
    )
    kernels = validate_kernels(kernels)
    kept = active_kernel_targets(
        kernels,
        fast=optimization == "fast",
        config=fast_config or FastConfig(),
    )
    with keep_operations(kept), warn_unused_kernels(kernels, fast=optimization == "fast"):
        return _transpile_any(
            model,
            example_inputs,
            output_path,
            sig_figs,
            name=name,
            optimization=optimization,
            fast_config=fast_config,
            storage=storage,
            quantization=quantization,
            calibration_inputs=calibration_inputs,
            codegen=codegen,
            adapters=adapters,
            calibration_calls=calibration_calls,
            calibration_sequences=calibration_sequences,
            example_kwargs=example_kwargs,
            kernels=kernels,
            dynamic=dynamic,
        )


def _transpile_dynamic(
    model,
    example_inputs,
    output_path,
    sig_figs,
    dynamic,
    *,
    example_kwargs,
    kernels,
    adapters=(),
    **options,
):
    """Trace one row of the dynamic input, export, and check other lengths."""
    import copy

    from cattorch.dynamic import (
        at_length,
        check_lengths,
        normalize_dynamic,
        per_unit_result,
        run_per_unit,
        unit_output_axes,
    )
    from cattorch.verify import verify

    if isinstance(example_inputs, torch.Tensor):
        example_inputs = (example_inputs,)
    example_inputs = tuple(example_inputs)
    kwarg_names = [
        key for key, value in (example_kwargs or {}).items() if isinstance(value, torch.Tensor)
    ]
    flat = flat_example_inputs(example_inputs, example_kwargs)
    spec = normalize_dynamic(dynamic, flat, kwarg_names)
    ((index, (_, minimum, maximum, axis)),) = spec.items()
    representative = list(flat)
    representative[index] = at_length(flat[index], axis, 1)
    positional = len(example_inputs)
    traced_kwargs = dict(example_kwargs or {})
    traced_kwargs.update(zip(kwarg_names, representative[positional:]))
    arguments = dict(
        generation=None,
        example_kwargs=traced_kwargs or None,
        _registry=_registry_with(kernels, adapters) if kernels or adapters else None,
        **options,
    )
    try:
        result = _transpile(
            model,
            tuple(representative[:positional]),
            output_path,
            sig_figs,
            _dynamic_inputs=spec,
            **arguments,
        )
    except UnsupportedModelError as error:
        # Operations that can't follow the length directly: run the whole
        # model once per item instead, if items are independent.
        try:
            axes = unit_output_axes(model, flat, kwarg_names, spec, example_kwargs)
        except Exception:  # noqa: BLE001 - the model may not run on one item at all
            # The graph is traced with one item along the dimension, whatever
            # the declared minimum.
            try:
                with torch.no_grad():
                    copy.deepcopy(model)(*representative[:positional], **traced_kwargs)
            except Exception as reason:  # noqa: BLE001 - reported to the user
                raise UnsupportedModelError(
                    "a variable-length export traces the model with one item along "
                    f"the dimension, and this model can't run on one "
                    f"({type(reason).__name__}: {reason})"
                ) from None
            raise error from None
        input_shapes = [tuple(value.shape) for value in representative]
        result = _transpile(
            model,
            tuple(representative[:positional]),
            output_path,
            sig_figs,
            **arguments,
            _edit_sprite=lambda sprite: run_per_unit(sprite, spec, input_shapes, len(axes)),
            _unit_inputs=spec,
        )
        result = per_unit_result(
            result,
            spec,
            axes,
            input_shapes,
            [tuple(item for item in spec_value.shape) for spec_value in result.outputs],
        )
    check_lengths(
        model, flat, kwarg_names, spec, result, verify, example_kwargs,
        optimization=options.get("optimization", "exact"), storage=options.get("storage"),
        quantization=options.get("quantization"), sig_figs=sig_figs, adapters=adapters,
        kernels=kernels,
    )
    return result


def _registry_with(kernels, adapters=()) -> OperatorRegistry:
    return default_registry().for_export(adapters=adapters, kernels=kernels)


def _transpile_any(
    model,
    example_inputs,
    output_path,
    sig_figs,
    *,
    name,
    optimization,
    fast_config,
    storage,
    quantization,
    calibration_inputs,
    codegen,
    adapters,
    calibration_calls,
    calibration_sequences,
    example_kwargs,
    kernels,
    dynamic=None,
):
    if example_inputs is None:
        # The model describes its own interface with markers.
        from cattorch.markers import (
            example_inputs as marked_example,
            has_entrypoints,
            marked_program,
        )

        if has_entrypoints(model):
            model, example_inputs = marked_program(model)
        else:
            example_inputs = marked_example(model)
            if example_inputs is None:
                raise ValueError(
                    "example_inputs is None, but the model has no @example or "
                    "@entrypoint marker to take them from"
                )
    if isinstance(example_inputs, GenerationProgram) and example_inputs.method == "forward":
        from cattorch.markers import generation_method

        method = generation_method(model)
        if method is not None:
            example_inputs = dataclasses.replace(example_inputs, method=method)
    if dynamic is not None and isinstance(example_inputs, ExportProgram):
        raise ValueError(
            "dynamic= applies to forward; give an entry point's bounds with "
            "Input(..., dynamic=...) or @entrypoint(dynamic=...)"
        )
    if dynamic is not None and isinstance(example_inputs, GenerationProgram):
        raise ValueError(
            "dynamic= can't be used with a GenerationProgram, whose prompts already "
            "vary up to max_context"
        )
    if dynamic is not None and isinstance(example_inputs, RecurrentGenerationProgram):
        raise ValueError(
            "dynamic= can't be used with a RecurrentGenerationProgram, which takes "
            "prompts of any length"
        )
    if example_kwargs is not None and isinstance(
        example_inputs,
        (ExportProgram, GenerationProgram),
    ):
        raise ValueError("example_kwargs cannot be used with an ExportProgram or GenerationProgram")
    if example_kwargs is not None and isinstance(example_inputs, RecurrentGenerationProgram):
        raise ValueError("example_kwargs cannot be used with a RecurrentGenerationProgram")
    if isinstance(example_inputs, ExportProgram):
        if calibration_inputs is not None:
            raise ValueError("ExportProgram uses calibration_calls instead of calibration_inputs")
        if calibration_sequences is not None:
            raise ValueError(
                "calibration_sequences require a GenerationProgram or RecurrentGenerationProgram"
            )
        from cattorch.program_export import transpile_program

        return transpile_program(
            model,
            example_inputs,
            output_path,
            sig_figs,
            name=name,
            optimization=optimization,
            fast_config=fast_config,
            storage=storage,
            quantization=quantization,
            calibration_calls=calibration_calls,
            codegen=codegen,
            adapters=adapters,
            kernels=kernels,
        )
    if isinstance(example_inputs, GenerationProgram):
        if calibration_calls is not None:
            raise ValueError("calibration_calls require an ExportProgram")
        if calibration_inputs is not None:
            raise ValueError(
                "GenerationProgram uses calibration_sequences instead of calibration_inputs"
            )
        if not hasattr(model, example_inputs.method) or not callable(
            getattr(model, example_inputs.method)
        ):
            raise ValueError(f"model has no callable generation method {example_inputs.method!r}")
        from cattorch.huggingface_recurrent import contains_recurrent_model

        family = contains_recurrent_model(model)
        if family is not None:
            raise UnsupportedModelError(
                f"GenerationProgram caches attention keys and values, and this {family} "
                "model has no attention; export it with RecurrentGenerationProgram() instead"
            )
        generation_program = example_inputs
        registry = _registry_with(kernels, adapters)
        return _transpile_automatic_generation(
            model,
            generation_program,
            output_path,
            sig_figs,
            name=name,
            optimization=optimization,
            fast_config=fast_config,
            storage=storage,
            quantization=quantization,
            calibration_sequences=calibration_sequences,
            codegen=codegen,
            registry=registry,
        )
    if isinstance(example_inputs, RecurrentGenerationProgram):
        if calibration_calls is not None:
            raise ValueError("calibration_calls require an ExportProgram")
        if calibration_inputs is not None:
            raise ValueError(
                "RecurrentGenerationProgram uses calibration_sequences instead of "
                "calibration_inputs"
            )
        from cattorch.recurrent_generation import transpile_recurrent_generation

        return transpile_recurrent_generation(
            model,
            example_inputs,
            output_path,
            sig_figs,
            name=name,
            optimization=optimization,
            fast_config=fast_config,
            storage=storage,
            quantization=quantization,
            calibration_sequences=calibration_sequences,
            codegen=codegen,
            adapters=adapters,
            kernels=kernels,
        )
    if calibration_calls is not None:
        raise ValueError("calibration_calls require an ExportProgram")
    if calibration_sequences is not None:
        raise ValueError(
            "calibration_sequences require a GenerationProgram or RecurrentGenerationProgram"
        )
    if dynamic is not None:
        return _transpile_dynamic(
            model,
            example_inputs,
            output_path,
            sig_figs,
            dynamic,
            name=name,
            optimization=optimization,
            fast_config=fast_config,
            storage=storage,
            quantization=quantization,
            calibration_inputs=calibration_inputs,
            codegen=codegen,
            example_kwargs=example_kwargs,
            kernels=kernels,
            adapters=adapters,
        )
    return _transpile(
        model,
        example_inputs,
        output_path,
        sig_figs,
        name=name,
        optimization=optimization,
        fast_config=fast_config,
        generation=None,
        storage=storage,
        quantization=quantization,
        calibration_inputs=calibration_inputs,
        codegen=codegen,
        example_kwargs=example_kwargs,
        _registry=_registry_with(kernels, adapters) if kernels or adapters else None,
    )


def _transpile_automatic_generation(
    model: torch.nn.Module,
    program: GenerationProgram,
    output_path: str | Path,
    sig_figs: int | None,
    *,
    name: str | None,
    optimization: Literal["exact", "fast"],
    fast_config: FastConfig | None,
    storage: StorageConfig | None,
    quantization: QuantizationConfig | None,
    calibration_sequences: Iterable[torch.Tensor | tuple[torch.Tensor, ...]] | None,
    codegen: CodegenConfig | None,
    registry: OperatorRegistry,
) -> TranspileResult:
    """Export a stateless ``forward(tokens)`` model as a cached decoder.

    The model is prepared exactly like an ordinary export (adapters, fast
    transforms, batch-norm folding, quantization), so quantization still sees
    the parameters. ``build_generation_step`` then derives the one-token step
    module whose attention reads and writes per-layer K/V caches, and the
    step is lowered through the generation machinery.
    """
    import copy
    from cattorch.autoregressive import (
        EmbeddingLookup,
        GenerationStepModule,
        MethodLeaves,
        build_generation_step,
        check_calibration_tokens,
        lookup_vocabulary,
    )

    config = fast_config or FastConfig()
    storage = storage or StorageConfig()
    codegen = codegen or CodegenConfig()
    if calibration_sequences is not None and (
        quantization is None or quantization.method != "gptq"
    ):
        raise ValueError("calibration_sequences can only be used with GPTQ quantization")
    max_context = program.max_context
    owned = False
    if any(
        value.is_floating_point() and value.dtype != torch.float32
        for value in (*model.parameters(), *model.buffers())
    ):
        # Scratch computes in doubles either way. In float32 the cached step
        # can be checked against the model without bfloat16 rounding noise.
        model = copy.deepcopy(model).float()
        owned = True
    module = model if program.method == "forward" else MethodLeaves(model, program.method)
    # A copy made above is this export's own; adapt it in place.
    prepared = registry.adapt_model(module, _copy=not owned)
    if optimization == "fast":
        prepared = prepare_fast_model(prepared, config)
    prepared = fold_eval_batch_norms(prepared)
    report = None
    if quantization is not None:
        if calibration_sequences is not None:
            token = torch.zeros(1, 1, dtype=torch.long)
            lookup = EmbeddingLookup(token)
            with torch.no_grad(), lookup:
                prepared(token)
            calibration_sequences = check_calibration_tokens(
                calibration_sequences,
                lookup_vocabulary(prepared, lookup),
            )
        report = quantize_model(
            prepared,
            quantization,
            calibration_sequences,
            orientation_inputs=[(torch.zeros(1, max_context, dtype=torch.long),)],
            _registry=registry,
            _calibration_name="calibration_sequences",
        )
    step = GenerationStepModule(
        build_generation_step(
            prepared,
            max_context,
            prepared=True,
            registry=registry,
        )
    )
    spec = _GenerationSpec(
        max_context,
        program.hidden_prefill,
        program.top_k,
    )
    options = dict(
        name=name,
        optimization=optimization,
        fast_config=fast_config,
        generation=spec,
        storage=storage,
        quantization=quantization,
        _registry=registry,
        _prepared_model=True,
        _prepared_quantization_report=report,
    )
    example = torch.zeros(1, 1, dtype=torch.long)
    try:
        # Tracing runs the step, so every attempt gets its own copy. Without
        # layer sharing there is no second attempt.
        return _transpile(
            copy.deepcopy(step) if codegen.layer_sharing == "auto" else step,
            (example,),
            output_path,
            sig_figs,
            codegen=codegen,
            **options,
        )
    except _LayerSharingUnavailable:
        log.warning(
            "layer_sharing='auto' could not factor this model's layers; "
            "exporting without layer sharing"
        )
        return _transpile(
            copy.deepcopy(step),
            (example,),
            output_path,
            sig_figs,
            codegen=replace(codegen, layer_sharing="off"),
            **options,
        )


def _transpile(
    model: torch.nn.Module,
    example_inputs: torch.Tensor | tuple[torch.Tensor, ...],
    output_path: str | Path,
    sig_figs: int | None = None,
    *,
    name: str | None = None,
    optimization: Literal["exact", "fast"] = "exact",
    fast_config: FastConfig | None = None,
    generation: _GenerationSpec | None = None,
    storage: StorageConfig | None = None,
    quantization: QuantizationConfig | None = None,
    calibration_inputs: Iterable[torch.Tensor | tuple[torch.Tensor, ...]] | None = None,
    codegen: CodegenConfig | None = None,
    example_kwargs: dict[str, Any] | None = None,
    _registry: OperatorRegistry | None = None,
    _prepared_model: bool = False,
    _prepared_quantization_report=None,
    _dynamic_inputs: dict[int, _DynamicBound] | None = None,
    _defer_interface_sharding: bool = False,
    _edit_sprite: Callable[[dict], dict] | None = None,
    _finalize: Callable[..., FinalizedSprite] = finalize_sprite,
    _hidden_outputs: frozenset[int] = frozenset(),
    _deferred_sizes: dict[str, int] | None = None,
    _unit_inputs: dict | None = None,
    _item_kinds: dict | None = None,
) -> TranspileResult:
    """Export one graph as a sprite. Private hooks, for the other export paths:

    ``_unit_inputs`` gives the bounds of the input an ``_edit_sprite`` runs
    once per item, which ``cattorch forward`` checks. With
    ``_defer_interface_sharding`` (a program's entry point), the inputs
    aren't checked; ``_item_kinds`` is filled with what their items must
    be, for the program to check them.
    """
    if isinstance(example_inputs, torch.Tensor):
        example_inputs = (example_inputs,)
    for index, value in enumerate(example_inputs):
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"example input {index} is a {type(value).__name__}, not a tensor; "
                "each positional input becomes one Scratch list, so pass nested "
                "values (such as an LSTM's (hidden, cell) state) through a "
                "wrapper that takes them as separate tensors"
            )
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".sprite3":
        output_path = Path(f"{output_path}.sprite3")
    sprite_name = output_path.stem if name is None else name
    if not isinstance(sprite_name, str) or not sprite_name.strip():
        raise ValueError("name must be a non-empty string")
    config = fast_config or FastConfig()
    storage = storage or StorageConfig()
    codegen = codegen or CodegenConfig()
    registry = _registry or default_registry().clone()
    if calibration_inputs is not None and (quantization is None or quantization.method != "gptq"):
        raise ValueError("calibration_inputs can only be used with GPTQ quantization")
    if (
        quantization is not None
        and quantization.method == "gptq"
        and any(isinstance(value, torch.Tensor) for value in (example_kwargs or {}).values())
    ):
        raise ValueError(
            "tensor-valued example_kwargs cannot be combined with GPTQ "
            "calibration_inputs; pass those tensors positionally"
        )
    quantization_report = _prepared_quantization_report
    if not _prepared_model:
        model = registry.adapt_model(model)
        if optimization == "fast":
            model = prepare_fast_model(model, config)
        model = fold_eval_batch_norms(model)
        if quantization is not None:
            quantization_report = quantize_model(
                model,
                quantization,
                calibration_inputs,
                orientation_inputs=[tuple(example_inputs)],
                call_kwargs=example_kwargs,
                _registry=registry,
            )
    elif calibration_inputs is not None:
        raise ValueError("prepared model calibration has already been consumed")
    if quantization is not None:
        storage = quantized_storage(storage, quantization)

    # 1. Analyse the PyTorch graph
    graph = _prepare_graph(
        model,
        example_inputs,
        optimization=optimization,
        fast_config=config,
        generation=generation,
        _registry=registry,
        dynamic_inputs=_dynamic_inputs,
        example_kwargs=example_kwargs,
    )
    if generation is not None and codegen.layer_sharing == "auto":
        if not _layers_can_share_caches(graph):
            log.info(
                "Banked generation caches skipped: attention blocks are not in "
                "consecutive `blocks.N` layers"
            )
        elif _bank_generation_caches(graph):
            log.info(
                "Banked %s generation cache layers for structural sharing",
                graph.generation.cache_layers,
            )

    # 2. Compile graph nodes into Scratch blocks
    compiler = _Compiler(graph, storage=storage)
    static_sources = (
        _StaticSourceRecorder(compiler) if quantization_report is not None else None
    )
    selected_unroll = _select_unroll_factor(model, storage, codegen)
    log.info(
        "Code generation unroll cap: %s (%s mode)",
        selected_unroll,
        codegen.unrolling,
    )
    with unroll_factor_limit(selected_unroll):
        sprite = compiler.compile()
    if static_sources is not None:
        static_sources.finish()
    before_sharing = dict(compiler.static_lists)
    lists_before_sharing = {entry[0] for entry in sprite.get("lists", {}).values()}
    compiler.static_lists = _deduplicate_static_tensors(
        sprite,
        compiler.static_lists,
        effective_ndims=compiler.static_effective_ndims,
    )

    # Layout-specialized kernels may replace an original static tensor with
    # one or more derived lists. Keep only weights that survived into blocks;
    # otherwise compressed storage would retain an unreachable payload.
    used_lists = {entry[0] for entry in sprite.get("lists", {}).values()}
    compiler.static_lists = {
        key: tensor for key, tensor in compiler.static_lists.items() if f"W_{key}" in used_lists
    }
    compiler.interleaved_static_layouts = {
        key: value
        for key, value in compiler.interleaved_static_layouts.items()
        if key in compiler.static_lists
    }

    # 3. Attach static weights and model inputs
    tensor_adder = TensorAdder()
    sprite = tensor_adder.apply(
        sprite,
        [f"W_{k}" for k in compiler.static_lists],
        weights={f"W_{k}": v for k, v in compiler.static_lists.items() if v is not None},
        sig_figs=sig_figs,
    )
    sprite = tensor_adder.apply(sprite, list(graph.input_names.values()))
    if codegen.layer_sharing == "auto":
        hidden_layers = (
            generation is not None
            and graph.generation.banked_caches
            and graph.generation.hidden_prefill_attention is not None
        )
        shared = _share_repeated_transformer_layers(
            sprite,
            compiler,
            storage,
            layer_tail=(_LAYER_TAIL, _PROJECT_OUTPUT) if hidden_layers else None,
        )
        if shared:
            log.info("Factored repeated transformer layers into one shared procedure")
        elif generation is not None and graph.generation.banked_caches:
            # Banked caches are addressed through the shared procedure's base
            # variable; without it every layer would read layer zero's rows.
            raise _LayerSharingUnavailable()

    # 4. Name the output list and clean up
    output_lists = compiler.output_lists
    output_names = tuple(output_name(index) for index in range(len(output_lists)))
    _rename_lists(sprite, dict(zip(output_lists, output_names)))
    output = output_lists[0] if output_lists else None

    logical_sizes = _logical_list_sizes(
        compiler,
        graph,
        generation,
        output,
    )
    _size_numbered_private_lists(sprite, logical_sizes)
    for node in graph.nodes:
        if node.name not in graph.dynamic_axes:
            continue
        maximum = graph.runtime_capacity(node.name, _get_shape(node))
        if node.name in graph.input_names:
            # Program input/state lists are renamed before their one global
            # interface-sharding pass.
            continue
        elif node.name in compiler.scope.assignments:
            list_name = f"T{compiler.scope.assignments[node.name]}"
        else:
            continue
        logical_sizes[list_name] = max(logical_sizes.get(list_name, 0), maximum)
    for old_name, public_name, (node_name, shape, _) in zip(
        output_lists,
        output_names,
        graph.outputs,
    ):
        # Output copies (including returned inputs and repeated leaves) have
        # no scope assignment of their own. Size every public output from its
        # contract so those copies are sharded too, using runtime capacity.
        capacity = (
            graph.runtime_capacity(node_name, shape)
            if node_name in graph.dynamic_axes
            else math.prod(shape)
        )
        logical_sizes[public_name] = max(
            capacity,
            logical_sizes.get(public_name, 0),
            logical_sizes.pop(old_name, 0),
        )
    if generation is not None and generation.top_k is not None:
        # The output list may be a reused, larger scratch list; the logits
        # row is the output's own shape.
        output_size = math.prod(graph.outputs[0][1])
        if generation.top_k > output_size:
            raise ValueError(f"top_k ({generation.top_k}) exceeds output size ({output_size})")

    cache_names = set(graph.generation.score_caches.values()) | set(
        graph.generation.value_caches.values()
    )
    _merge_lists_by_name(sprite, cache_names)
    _merge_duplicate_lists(sprite)
    _wrap_optimized_lifecycle(sprite)
    if _hidden_outputs:
        _gate_hidden_outputs(sprite, graph, compiler, _hidden_outputs)
    if generation is not None:
        if generation.hidden_prefill:
            root_entries = compiler.emitted_roots
            root_names = [name for name, _ in root_entries]
            hidden_attention = graph.generation.hidden_prefill_attention
            if graph.generation.banked_caches:
                # The shared layer is layer zero's code, gated by the layer
                # tail variable the calls set.
                _guard_hidden_layer_tail(sprite, graph, compiler, _shared_layer_attention(graph))
                # After the layers, only the logits remain.
                by_name = {node.name: node for node in graph.nodes}
                last_layer = max(
                    position
                    for position, name in enumerate(root_names)
                    if name in by_name and _graph_layer(by_name[name]) is not None
                )
                if last_layer + 1 >= len(root_entries):
                    raise ValueError("hidden prefill requires a compiled output projection")
                _gate_chain(
                    sprite,
                    root_entries[last_layer + 1][1],
                    _PROJECT_OUTPUT,
                )
                _merge_variables_by_name(sprite, {_PROJECT_OUTPUT, _LAYER_TAIL})
            elif hidden_attention in root_names and root_names.index(hidden_attention) + 1 < len(
                root_entries
            ):
                # Keep the final layer's K/V append, skip its attention output
                # and everything after it for hidden prompt tokens.
                for entry in sprite.get("variables", {}).values():
                    if entry[0] == _PROJECT_OUTPUT:
                        entry[1] = 1
                _guard_hidden_layer_tail(sprite, graph, compiler, hidden_attention)
            else:
                if not compiler.emitted_roots:
                    raise ValueError("hidden prefill requires a compiled output projection")
                _gate_chain(sprite, compiler.emitted_roots[-1][1])
        first_cache = graph.generation.first_cache
        assert first_cache is not None
        sprite = _wrap_generation_lifecycle(
            sprite,
            cache_names=sorted(cache_names),
            first_cache=first_cache,
            max_context=generation.max_context,
            hidden_prefill=generation.hidden_prefill,
            top_k=generation.top_k,
            compact_top_k=codegen.unrolling != "speed",
            vocabulary=_index_rows(graph),
        )
        # Kernels declare `cattorch cache length` themselves; unify them with
        # the lifecycle's variable.
        _merge_variables_by_name(sprite, {interface.CACHE_LENGTH})
        generation_lists = {
            "input",
            "output",
            "cattorch prefill buffer",
            interface.TOKEN_IDS,
            interface.LOGITS,
        }
        if generation.top_k is not None:
            generation_lists.update({_TOP_K_VALUES, _TOP_K_IDS})
        _merge_lists_by_name(
            sprite,
            cache_names | generation_lists,
        )
        _merge_duplicate_lists(sprite)
        # The step is private; only reset, prefill and decode run it.
        _remove_canvas_calls(sprite, "cattorch forward")
        _rename_procedure(sprite, "cattorch forward", "_cattorch token step")
        private_lists = {
            "input": _STEP_TOKEN,
            "output": _STEP_LOGITS,
            "cattorch prefill buffer": "_cattorch prefill buffer",
        }
        _rename_lists(sprite, private_lists)
        for old, new in private_lists.items():
            if old in logical_sizes:
                logical_sizes[new] = logical_sizes.pop(old)
        # The public logits are a copy of the step's.
        logical_sizes[interface.LOGITS] = logical_sizes.get(_STEP_LOGITS, 0)

    deferred_names = (
        frozenset((*graph.input_names.values(), *output_names))
        if _defer_interface_sharding
        else frozenset()
    )
    layouts = shard_sprite_lists(
        sprite,
        logical_sizes,
        deferred_names=deferred_names,
        alignments=compiler.static_shard_alignments,
        limits=_static_storage_shard_limits(
            compiler.static_lists,
            storage,
            effective_ndims=compiler.static_effective_ndims,
            interleaved_layouts=compiler.interleaved_static_layouts,
            source_numels=compiler.static_source_numels,
        ),
    )
    if _deferred_sizes is not None:
        # An output list can be a reused scratch list larger than the output;
        # the caller shards it later and needs its peak size.
        _deferred_sizes.update(
            {name: layouts[name].logical_size for name in deferred_names if name in layouts}
        )
    tied_copies = _apply_tied_static_aliases(
        sprite,
        layouts,
        compiler.static_lists,
        compiler.tied_storage_aliases,
    )
    tied_copies.update(
        (table, reason)
        for table, reason in compiler.tied_storage_copies.items()
        if table in layouts
    )
    misgrouped: tuple[str, ...] = ()
    if quantization_report is not None and static_sources is not None:
        misgrouped, quantization_report = _storage_quantization_report(
            quantization_report,
            static_tensors=compiler.static_lists,
            deduplicated={
                key: tensor for key, tensor in before_sharing.items()
                if key not in compiler.static_lists and f"W_{key}" in lists_before_sharing
            },
            storage=storage,
            sprite=sprite,
            layouts=layouts,
            effective_ndims=compiler.static_effective_ndims,
            interleaved_layouts=compiler.interleaved_static_layouts,
            recorder=static_sources,
            model=model,
        )
    static_names = _apply_static_storage(
        sprite,
        layouts,
        compiler.static_lists,
        storage,
        interleaved_layouts=compiler.interleaved_static_layouts,
        effective_ndims=compiler.static_effective_ndims,
        source_numels=compiler.static_source_numels,
    )
    # Public lists keep their values; a generation sprite's prepare for save
    # also resets it, since its cache is internal data it can't keep.
    _add_prepare_for_save(
        sprite,
        layouts,
        static_names=static_names,
        compressed=storage.compression,
        inputs=(
            {interface.TOKEN_IDS, interface.LOGITS, _TOP_K_VALUES, _TOP_K_IDS}
            if generation is not None else set(graph.input_names.values())
        ),
    )
    if generation is not None:
        _finish_generation_prepare(sprite, _GENERATION_RESET)
    sprite.setdefault("variables", {}).update(
        {
            "cattorch_storage_compression": [
                "cattorch storage compression",
                (
                    "base92-huffman-int4"
                    if storage.compression and storage.precision == "int4"
                    else "base92"
                    if storage.compression
                    else "none"
                ),
            ],
            "cattorch_storage_precision": [
                "cattorch storage precision",
                storage.precision,
            ],
            "cattorch_storage_scale_precision": [
                "cattorch storage scale precision",
                storage.scale_precision,
            ],
            "cattorch_quantization_method": [
                "cattorch quantization method",
                quantization.method if quantization is not None else "none",
            ],
            "cattorch_quantization_bits": [
                "cattorch quantization bits",
                quantization.bits if quantization is not None else 0,
            ],
        }
    )
    _remove_unused(sprite)
    uniquify_data_ids(sprite)

    log.info("Dynamic lists: %s", list(compiler.dynamic_lists))
    log.info("Static lists: %s", list(compiler.static_lists.keys()))

    # A caller's last changes, such as running the forward once per item,
    # come before finalization, so it sees the sprite whole.
    if _edit_sprite is not None:
        sprite = _edit_sprite(sprite)
    flat_inputs = flat_example_inputs(example_inputs, example_kwargs)
    if generation is not None:
        sprite = _guard_generation_init(sprite)
    elif not _defer_interface_sharding:
        _add_input_checks(sprite, _input_checks(
            graph, layouts, flat_inputs, unit_spec=_unit_inputs,
        ), _input_lists(layouts, flat_inputs))
        sprite = _guard_plain_calls(sprite, [
            shard
            for name in dict.fromkeys(output_names)
            for shard in (
                [shard.name for shard in layouts[name].shards] if name in layouts else [name]
            )
        ])
    elif _item_kinds is not None:
        _item_kinds.update(_kinds_of_items(graph, flat_inputs))
    finalized = _finalize(
        sprite,
        output_path,
        sprite_name=sprite_name,
        codegen=codegen,
        # A program finishes its entry points' interface itself.
        public_interface=not _defer_interface_sharding,
    )

    input_specs = tuple(
        boundary_spec(
            interface.TOKEN_IDS if generation is not None else input_name(index),
            tuple(value.shape),
            value.dtype,
            _public_dynamic_bound((_dynamic_inputs or {}).get(index)),
        )
        for index, value in enumerate(flat_example_inputs(example_inputs, example_kwargs))
    )
    procedures = ["cattorch init", "cattorch forward", "cattorch prepare for save"]
    if generation is not None:
        procedures = [
            "cattorch init", "cattorch reset", "cattorch prefill", "cattorch decode",
            "cattorch prepare for save",
        ]
    return TranspileResult(
        path=finalized.path,
        sprite_name=sprite_name,
        archive_bytes=finalized.archive_bytes,
        expanded_json_bytes=finalized.expanded_json_bytes,
        block_count=len(sprite.get("blocks", {})),
        list_count=len(sprite.get("lists", {})),
        sharded_lists=tuple(
            (layout.name if _defer_interface_sharding else final_data_name(layout.name))
            for layout in layouts.values()
            if len(layout.shards) > 1
        ),
        inputs=input_specs,
        output=boundary_spec(
            interface.LOGITS if generation is not None else "output",
            tuple(graph.output_shape),
            graph.output_dtype,
            (
                (graph.dynamic_axes[graph.output_name], graph._public_bounds(graph.output_name))
                if graph.output_name in graph.dynamic_axes
                else None
            ),
        ),
        additional_outputs=tuple(
            boundary_spec(
                f"output_{index}",
                tuple(shape),
                dtype,
                (
                    (graph.dynamic_axes[node_name], graph._public_bounds(node_name))
                    if node_name in graph.dynamic_axes
                    else None
                ),
            )
            for index, (node_name, shape, dtype) in enumerate(graph.outputs[1:], 1)
        ),
        procedures=tuple(procedures),
        warnings=finalized.warnings
        + tuple(
            f"GPTQ fallback for {item.name}: {item.fallback_reason}"
            for item in (quantization_report.tensors if quantization_report else ())
            if item.method == "symmetric" and item.fallback_reason is not None
        )
        + misgrouped
        + tuple(
            f"tied weight {table.removeprefix('W_')!r} is stored twice: {reason}"
            for table, reason in tied_copies.items()
        )
        + tuple(filter(None, (fallback_warning(graph.fallbacks()),))),
        quantization=quantization_report,
        fallback_operations=graph.fallbacks(),
        approximate_operations=tuple(sorted(compiler.approximate.items())),
    )
