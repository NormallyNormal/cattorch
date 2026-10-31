import logging
import math
import re
import operator
import dataclasses
import warnings
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Literal

import torch

from cattorch.codegen import CodegenConfig
from cattorch.errors import UnsupportedModelError
from cattorch.export_options import validate_export_options
from cattorch.fast import FastConfig, prepare_fast_model
from cattorch.frontend import Frontend, flat_example_inputs, keep_operations
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
    TranspileResult, _tensor_spec as boundary_spec, fallback_warning,
)
from cattorch.program import ExportProgram, GenerationProgram, ProgramCall
from cattorch.adapters import ModuleAdapter
from cattorch.kernel_api import Kernel, active_kernel_targets, validate_kernels
from cattorch.quantization import (
    QuantizationConfig, expert_bank_effective_ndims, quantize_model,
)
from cattorch.sprite import (
    _TOP_K_IDS,
    _TOP_K_VALUES,
    _add_prepare_for_save,
    _add_procedure_completion_broadcast,
    _add_warp_procedure,
    _apply_static_storage,
    _apply_tied_static_aliases,
    _deduplicate_static_tensors,
    _guard_generation_output,
    _guard_generation_range,
    _guard_generation_suffix,
    _logical_list_sizes,
    _merge_duplicate_lists,
    _graph_layer,
    _merge_lists_by_name,
    _merge_variables_by_name,
    _remove_unused,
    _share_repeated_transformer_layers,
    _static_storage_group_size,
    _static_storage_shard_limits,
    _wrap_generation_lifecycle,
    _wrap_optimized_lifecycle,
)
from cattorch.interface import STATUS_VARIABLE, final_data_name, input_name, output_name
from cattorch.storage import StorageConfig, quantized_storage
from cattorch.util.argument import Argument
from cattorch.operator_registry import (
    INDEX_MAP_OPS, MATMUL_OPS, OperatorRegistry, default_registry,
)
from cattorch.util.instruction.generic import IndexMapInstruction, index_map
from cattorch.util.scope import ScopeManager
from cattorch.util.scratch.block_combiner import combine
from cattorch.util.scratch.block_manager import BlockManager
from cattorch.util.scratch.finalize_scratch import finalize_sprite
from cattorch.util.scratch.ids import uniquify_data_ids
from cattorch.util.scratch.interface import (
    rename_list as _rename_list, rename_procedure as _rename_procedure,
)
from cattorch.util.scratch.dsl import (
    Program, add, call, clear, delete, div, eq, gt, if_, length, mul, not_,
    repeat, set_var, sub, unroll_factor_limit, var,
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
    widths = [
        generation.cache_widths[cache]
        for _, cache in (*score_entries, *value_entries)
    ]
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


def _check_input_lengths(sprite, graph: _GraphInfo, layouts, output_names) -> None:
    """Reject a variable-length input of the wrong length before running.

    ``cattorch forward`` becomes a wrapper that checks each variable-length
    input holds whole items within its bounds, as program entry points do;
    otherwise it clears the outputs and reports why in ``cattorch status``.
    """
    from cattorch.util.scratch.dsl import broadcast_and_wait, if_else, lt, mod

    checks = []
    for node in graph.nodes:
        name = graph.input_names.get(node.name)
        if name is None or node.name not in graph.dynamic_bounds:
            continue
        if name in layouts and len(layouts[name].shards) > 1:
            continue  # a split list's length isn't one list's length
        shape = _get_shape(node)
        axis = graph.dynamic_axes[node.name]
        width = max(math.prod(shape) // max(shape[axis], 1), 1)
        _, minimum, _ = graph.dynamic_bounds[node.name]
        checks.extend((
            if_(gt(mod(length(name), width), 0), (
                set_var(STATUS_VARIABLE, f"invalid dynamic length: {name}"),
            )),
            if_(lt(length(name), width * max(minimum, 1)), (
                set_var(STATUS_VARIABLE, f"too few items: {name}"),
            )),
            if_(gt(length(name), graph.runtime_capacity(node.name, shape)), (
                set_var(STATUS_VARIABLE, f"dynamic capacity exceeded: {name}"),
            )),
        ))
    if not checks:
        return
    internal = "_cattorch checked forward"
    _rename_procedure(sprite, "cattorch forward", internal)
    wrapper = Program(
        "checked_forward",
        variables=(STATUS_VARIABLE,),
        lists=tuple(dict.fromkeys((*graph.input_names.values(), *output_names))),
        body=(
            set_var(STATUS_VARIABLE, "ok"),
            *checks,
            if_else(
                eq(var(STATUS_VARIABLE), "ok"),
                (call(internal),),
                (
                    *(clear(name) for name in output_names),
                    broadcast_and_wait("cattorch forward complete"),
                ),
            ),
        ),
    )
    _add_warp_procedure(sprite, "cattorch forward", wrapper, x=640, y=960)


def _token_vocabulary(graph: _GraphInfo) -> int | None:
    """Rows of the embedding table the token input is looked up in."""
    tokens = {name for name, list_name in graph.input_names.items() if list_name == "input"}
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if str(node.target) == "aten.embedding.default" and getattr(
            node.args[1], "name", None,
        ) in tokens:
            return int(_get_shape(node.args[0])[0])
        if node.all_input_nodes and node.all_input_nodes[0].name in tokens and str(
            node.target,
        ) in _NOOP_OPS | {"aten.to.dtype", "aten._to_copy.default"}:
            tokens.add(node.name)
    return None


def _shared_layer_attention(graph: _GraphInfo) -> str:
    """The cached attention of layer zero, whose code the shared layer runs."""
    by_name = {node.name: node for node in graph.nodes}
    return next(
        name for name in graph.generation.score_caches if _graph_layer(by_name[name]) == 0
    )


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
            _guard_generation_suffix(sprite, following, gate)
    query_only = _query_only_nodes(graph, attention)
    position = 0
    while position < attention_position:
        if root_names[position] not in query_only:
            position += 1
            continue
        end = position
        while end < attention_position and root_names[end] in query_only:
            end += 1
        _guard_generation_range(
            sprite, root_entries[position][1], root_entries[end][1], gate,
        )
        position = end


def _wrap_transactional_generation_step(
    sprite: dict,
    cache_widths: dict[str, int],
    *,
    top_k: bool,
) -> dict:
    """Commit all KV rows together or restore every cache to its old length."""
    _rename_procedure(sprite, "cattorch forward", "_cattorch token compute")
    ordered = sorted(cache_widths.items())
    first_cache, first_width = ordered[0]
    old_tokens = "_cattorch old cache tokens"
    variables = {"_cattorch transaction ok", old_tokens, STATUS_VARIABLE}
    body = [
        set_var("_cattorch transaction ok", 1),
        set_var(old_tokens, div(length(first_cache), first_width)),
    ]
    body.append(call("_cattorch token compute"))
    for cache_name, width in ordered:
        body.append(if_(
            not_(eq(length(cache_name), mul(add(var(old_tokens), 1), width))),
            (set_var("_cattorch transaction ok", 0),),
        ))
    rollback = []
    for cache_name, width in ordered:
        old_length = mul(var(old_tokens), width)
        rollback.append(if_(
            gt(length(cache_name), old_length),
            (repeat(
                sub(length(cache_name), old_length),
                (delete(cache_name, length(cache_name)),),
            ),),
        ))
    rollback.extend((
        clear("output"),
        if_(
            eq(var(STATUS_VARIABLE), "ok"),
            (set_var(STATUS_VARIABLE, "cache transaction failed"),),
        ),
    ))
    if top_k:
        rollback.extend((clear(_TOP_K_VALUES), clear(_TOP_K_IDS)))
    body.append(if_(eq(var("_cattorch transaction ok"), 0), tuple(rollback)))
    return _add_warp_procedure(
        sprite,
        "cattorch forward",
        Program(
            "transactional_generation_step",
            variables=variables,
            lists=("output", *cache_widths, *(
                (_TOP_K_VALUES, _TOP_K_IDS) if top_k else ()
            )),
            body=tuple(body),
        ),
        x=640,
        y=800,
    )


# ── Compiler ─────────────────────────────────────────────────────────────────

class _LayerSharingUnavailable(Exception):
    """Banked caches were requested but the layers could not be factored."""


def _layers_can_share_caches(graph: _GraphInfo) -> bool:
    """Whether each automatic attention block sits in its own ``blocks.N`` layer."""
    by_name = {node.name: node for node in graph.nodes}
    layers = [
        _graph_layer(by_name[name]) for name in graph.generation.score_caches
    ]
    return None not in layers and layers == list(range(len(layers)))


class _Compiler:
    """Walk the graph, dispatch each node to the right handler, and produce
    a single Scratch sprite dict with all blocks chained together."""

    def __init__(self, graph: _GraphInfo, *, row_aligned_weights: bool = False):
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
        self.static_shard_alignments: dict[str, int] = {}
        self.tied_storage_aliases: dict[str, tuple[str, ...]] = {}
        self.row_aligned_weights = row_aligned_weights
        self.interleaved_static_layouts: dict[
            str, tuple[tuple[int, int, int], ...]
        ] = {}
        self._grouped_tied_weights = self._find_grouped_tied_weights()
        self._compiled_output_lists: tuple[str, ...] = ()

    @staticmethod
    def _weight_identity(tensor: torch.Tensor) -> tuple[int, int, torch.dtype]:
        return (
            tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tensor.dtype,
        )

    def _find_grouped_tied_weights(self) -> set[tuple[int, int, torch.dtype]]:
        """Find small Linear weights whose grouped layout can serve embeddings."""
        from cattorch.util.instruction.optimized import LinearInstruction

        embeddings = set()
        linears = set()
        for node in self.graph.nodes:
            target = str(node.target)
            if target not in {"aten.embedding.default", "aten.linear.default"}:
                continue
            weight_index = 0 if target == "aten.embedding.default" else 1
            if len(node.args) <= weight_index or not hasattr(node.args[weight_index], "name"):
                continue
            tensor = self.graph.resolve_weight(node.args[weight_index].name)
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
                continue
            identity = self._weight_identity(tensor)
            if target == "aten.embedding.default":
                embeddings.add(identity)
            elif (
                tensor.shape[0] % 4 == 0
                and LinearInstruction.grouped_min_macs <= tensor.numel()
            ):
                linears.add(identity)
        return embeddings & linears

    def _uses_grouped_tied_weight(self, node) -> bool:
        weight_index = 0 if str(node.target) == "aten.embedding.default" else 1
        if len(node.args) <= weight_index or not hasattr(node.args[weight_index], "name"):
            return False
        tensor = self.graph.resolve_weight(node.args[weight_index].name)
        return (
            isinstance(tensor, torch.Tensor)
            and self._weight_identity(tensor) in self._grouped_tied_weights
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
            if node.op != 'call_function':
                continue
            if node.name in self.graph.aliases:
                continue
            if node.name in self.graph.generated_weights:
                continue
            if node.name in self.graph.fused_nodes:
                continue
            self._compile_node(node)
        self._materialize_output_if_needed()
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
                        source_list, output_shape,
                        dynamic_axis=self.graph.dynamic_axes.get(graph_output),
                    ),
                    Argument("C_chunk_size", torch.Size([]), value=size),
                    Argument("C_num_rows", torch.Size([]), value=1),
                    Argument("C_skip", torch.Size([]), value=0),
                    Argument("C_offset", torch.Size([]), value=0),
                )
                from cattorch.util.instruction.optimized import OptimizedGetItemInstruction
                instruction = OptimizedGetItemInstruction(
                    "cattorch.output.copy", target_list, *args,
                )
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
        elif (node.target is operator.getitem
              and hasattr(node.args[0], 'name')
              and node.args[0].name in self.graph.split_meta):
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
            self.graph.resolve_weight(name), torch.Tensor,
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
                        node, "only the first argument can vary at runtime",
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
                node, "data movement on a runtime extent is not supported",
            )
        try:
            indices, fill = index_map(
                node.target, tuple(_get_shape(source)),
                static(tuple(node.args[1:])), static(dict(node.kwargs)),
            )
        except ValueError as error:
            raise _unsupported_operation(node, str(error)) from error
        if len(indices) != math.prod(_get_shape(node)):
            raise _unsupported_operation(node, "unexpected output size")
        input_list = self._resolve_name(source.name)
        instruction = IndexMapInstruction(
            node.target, target_list, self._argument(input_list, source),
            indices=indices, fill=fill,
        )
        self._emit(instruction, [input_list], target_list)

    def _compile_causal_score(self, node):
        final_node, q, k, scale, transposed_k, scale_node = (
            self.graph.causal_score_fusions[node.name]
        )
        target_list = self.scope.get_list_for_node(final_node)
        self.last_target_list = target_list
        input_lists = [self._resolve_name(q.name), self._resolve_name(k.name)]
        args = [
            Argument(input_lists[0], _get_shape(q)),
            Argument(input_lists[1], _get_shape(k)),
            Argument("C_scale", torch.Size([]), value=scale),
            Argument(
                "C_query_offset", torch.Size([]),
                value=self.graph.qkv_offsets.get(q.name, 0),
            ),
            Argument(
                "C_key_offset", torch.Size([]),
                value=self.graph.qkv_offsets.get(k.name, 0),
            ),
        ]
        from cattorch.util.instruction.optimized import CausalScoreInstruction
        instruction = CausalScoreInstruction(node.target, target_list, *args)
        self._emit(instruction, input_lists, target_list)

        # Mirror the dependency releases of the materialized chain that this
        # single instruction replaces.
        self.scope.release_dependencies(transposed_k)
        self.scope.release_dependencies(node)
        if scale_node is not None:
            self.scope.release_dependencies(scale_node)
        self.scope.release_dependencies(final_node)

    def _compile_qkv_linear(self, node):
        query_heads, kv_heads, *_ = self.graph.qkv_linear_fusions[node.name]
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
        args.extend((
            Argument("C_query_heads", torch.Size([]), value=query_heads),
            Argument("C_kv_heads", torch.Size([]), value=kv_heads),
        ))
        from cattorch.util.instruction.optimized import QKVLinearInstruction
        instruction = QKVLinearInstruction(node.target, target_list, *args)
        self._emit(instruction, input_lists, target_list)
        self.scope.release_dependencies(node)

    def _compile_embedding_add(self, node):
        first, second = self.graph.embedding_add_fusions[node.name]
        target_list = self.scope.get_list_for_node(node)
        self.last_target_list = target_list
        tensor_nodes = (first.args[0], first.args[1], second.args[0], second.args[1])
        input_lists = [self._resolve_name(value.name) for value in tensor_nodes]
        args = [
            self._argument(name, value)
            for name, value in zip(input_lists, tensor_nodes)
        ]
        from cattorch.util.instruction.optimized import EmbeddingAddInstruction
        instruction = EmbeddingAddInstruction(node.target, target_list, *args)
        instruction.interleaved_sides = (
            self._uses_grouped_tied_weight(first),
            self._uses_grouped_tied_weight(second),
        )
        self._emit(instruction, input_lists, target_list)
        self.scope.release_dependencies(first)
        self.scope.release_dependencies(second)
        self.scope.release_dependencies(node)

    def _compile_silu_mul(self, node):
        silu, gate, value = self.graph.silu_mul_fusions[node.name]
        target_list = self.scope.get_list_for_node(node)
        self.last_target_list = target_list
        input_lists = [self._resolve_name(gate.name), self._resolve_name(value.name)]
        args = [
            Argument(input_lists[0], _get_shape(gate)),
            Argument(input_lists[1], _get_shape(value)),
        ]
        from cattorch.util.instruction.optimized import OptimizedSwiGLUInstruction
        instruction = OptimizedSwiGLUInstruction(node.target, target_list, *args)
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
            node, gate, value, input_lists,
        )
        args[1] = Argument(input_lists[1], args[1].shape)
        args[3] = Argument(input_lists[2], args[3].shape)
        from cattorch.util.instruction.optimized import PairedLinearSwiGLUInstruction
        instruction = PairedLinearSwiGLUInstruction(
            node.target, target_list, *args, shard_rows=shard_rows,
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

        from cattorch.util.scratch.dsl import add, div, item, mul, sub, var

        operations = {
            "aten.add.Tensor": add,
            "aten.sub.Tensor": sub,
            "aten.mul.Tensor": mul,
            "aten.div.Tensor": div,
        }
        external_nodes = []
        external_slots = {}
        expressions = {}

        def operand_expression(operand):
            if not hasattr(operand, "name"):
                return operand
            if operand.name in expressions:
                return expressions[operand.name]
            if operand.name not in external_slots:
                external_slots[operand.name] = len(external_nodes) + 1
                external_nodes.append(operand)
            return item(f"T{external_slots[operand.name]}", var("index"))

        for fused_node in chain:
            target = str(fused_node.target)
            if target == "aten.neg.default":
                expression = mul(-1, operand_expression(fused_node.args[0]))
            else:
                expression = operations[target](
                    operand_expression(fused_node.args[0]),
                    operand_expression(fused_node.args[1]),
                )
            expressions[fused_node.name] = expression

        input_lists = [self._resolve_name(value.name) for value in external_nodes]
        args = [
            self._argument(name, value)
            for name, value in zip(input_lists, external_nodes)
        ]
        from cattorch.util.instruction.optimized import FusedArithmeticInstruction
        instruction = FusedArithmeticInstruction(
            node.target,
            target_list,
            *args,
            expression=expressions[node.name],
            size=math.prod(_get_shape(node)),
        )
        self._emit(instruction, input_lists, target_list)
        for fused_node in chain:
            self.scope.release_dependencies(fused_node)

    def _compile_fused_linear(self, node):
        final_node, epilogue, chain = self.graph.linear_fusions[node.name]
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
            node, input_lists,
        )
        args[1] = Argument(input_lists[1], args[1].shape)

        from cattorch.util.instruction.optimized import LinearInstruction
        instruction = LinearInstruction(
            node.target, target_list, *args, epilogue=epilogue,
            fast=self.graph.fast and self.graph.fast_config.activations,
            weight_shard_rows=weight_shard_rows,
            interleaved=interleaved,
        )
        self._emit(instruction, input_lists, target_list)
        self.scope.release_dependencies(node)
        for fused_node in chain:
            self.scope.release_dependencies(fused_node)

    # ── standard instruction ─────────────────────────────────────────────

    def _compile_instruction(self, node, target_list, aten_op):
        rule = self.graph.registry.operation(aten_op)
        detail = (
            None if node.name in self.graph.causal_value_matmuls
            else self.graph.registry.validate(
                node, fast=self.graph.fast, config=self.graph.fast_config,
            )
        )
        if detail is not None:
            raise _unsupported_operation(node, detail)
        if aten_op in {
            "cattorch.expert_family_moe.default", "cattorch.routed_experts.default",
        }:
            value_nodes = list(node.args[:3])
            bank_nodes = list(node.args[3])
            tensor_nodes = [*value_nodes, *bank_nodes]
            input_lists = [self._resolve_name(value.name) for value in tensor_nodes]
            args = [
                self._argument(name, value)
                for name, value in zip(input_lists, tensor_nodes)
            ]
            handle = int(node.args[4])
            from cattorch.util.instruction.optimized_moe import (
                ExpertFamilyMoEInstruction, RoutedExpertsInstruction,
            )
            fast_options = {
                "fast_activations": self.graph.fast and self.graph.fast_config.activations,
                "fast_layer_norm": self.graph.fast and self.graph.fast_config.layer_norm,
            }
            if aten_op == "cattorch.routed_experts.default":
                instruction = RoutedExpertsInstruction(
                    node.target, target_list, *args,
                    family=self.graph.registry.expert_family(handle),
                    **fast_options,
                )
            else:
                instruction = ExpertFamilyMoEInstruction(
                    node.target, target_list, *args,
                    family=self.graph.registry.expert_family(handle),
                    top_k=int(node.args[5]),
                    normalize_selected=bool(node.args[6]),
                    **fast_options,
                )
            family = self.graph.registry.expert_family(handle)
            template_state = {}
            template_state.update(dict(
                family.template.named_parameters(remove_duplicate=False),
            ))
            template_state.update(dict(
                family.template.named_buffers(remove_duplicate=False),
            ))
            locations = tuple(dict.fromkeys(family._bank_names.values()))
            first_names = {
                location: next(
                    name for name in family._state_order
                    if family._bank_names[name] == location
                )
                for location in locations
            }
            for position, location in enumerate(locations):
                key = input_lists[3 + position].removeprefix("W_")
                self.static_effective_ndims[key] = template_state[
                    first_names[location]
                ].ndim
            self._emit(instruction, input_lists, target_list)
            return
        lowering_target = (
            node.target if rule is None or rule.lowering_target is None
            else rule.lowering_target
        )
        input_lists = self._resolve_args(node)
        # Automatic generation binds a step module's cache buffers to the
        # persistent K/V lists and drops its position input (the kernels read
        # `cattorch cache length`), so none of them becomes a static weight.
        for position, bound in self._automatic_bindings(node).items():
            input_lists[position] = bound
        node_args = list(node.args)
        # Capture frontends omit the optional bias for bias-free linear.
        # Keep the production kernel's tensor slots stable with an empty list.
        if aten_op == "aten.linear.default" and len(node_args) == 2:
            node_args.append(None)
            input_lists.append("_none")
        # The production LinearInstruction has a fourth tensor slot for an
        # optional fused residual. Unfused calls keep that slot explicitly
        # empty; this is separate from the optional PyTorch bias argument.
        if aten_op == "aten.linear.default":
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
        if schema is not None and node.kwargs and not any(
            hasattr(value, "name") for value in node.kwargs.values()
        ):
            names = [argument.name for argument in schema.arguments]
            last = max(
                (names.index(key) for key in node.kwargs if key in names), default=-1,
            )
            for argument in schema.arguments[len(node_args):last + 1]:
                value = node.kwargs.get(
                    argument.name,
                    argument.default_value if argument.has_default_value() else None,
                )
                node_args.append(value)
                input_lists.append("_none" if value is None else f"C_{value}")
        log.info("%s -> %s (%s) (Inputs: %s)",
                 aten_op, target_list, node.name, input_lists)

        args = []
        for i, name in enumerate(input_lists):
            node_arg = node_args[i]
            if node_arg is None:
                args.append(Argument(name, torch.Size([]), value=None))
            elif hasattr(node_arg, 'name'):
                args.append(self._argument(name, node_arg))
            else:
                args.append(Argument(name, torch.Size([]), value=node_arg))

        if (
            aten_op in MATMUL_OPS
            and len(node.args) > 1
            and hasattr(node.args[1], "name")
            and not (
                self.graph.fast and self.graph.fast_config.weights.enabled
            )
        ):
            rhs = self.graph.resolve_weight(node.args[1].name)
            if (
                isinstance(rhs, torch.Tensor)
                and rhs.ndim == 2
                and rhs.is_floating_point()
                # Quantization groups every matrix over its input width, so
                # store quantized weights output-major like linear weights.
                and (rhs.numel() > SCRATCH_LIST_LIMIT or self.row_aligned_weights)
            ):
                instruction, matrix_inputs = self._large_static_matmul(
                    node, target_list, args[0], rhs,
                )
                self._emit(instruction, matrix_inputs, target_list)
                return
        linear_layout = None
        if aten_op == "aten.linear.default":
            input_lists, weight_shard_rows, interleaved = self._prepare_linear_weight(
                node, input_lists,
            )
            args[1] = Argument(input_lists[1], args[1].shape)
            linear_layout = (weight_shard_rows, interleaved)
        static_rhs = None
        if (
            self.graph.fast
            and aten_op in MATMUL_OPS
            and len(node.args) > 1
            and hasattr(node.args[1], "name")
            and self.graph.fast_config.weights.enabled
        ):
            candidate = self.graph.resolve_weight(node.args[1].name)
            if isinstance(candidate, torch.Tensor) and candidate.ndim == 2:
                static_rhs = candidate

        if static_rhs is not None:
            from cattorch.fast import prepare_fast_matmul_weights
            from cattorch.util.instruction.optimized import (
                FastLowRankMatMulInstruction,
                FastPrunedMatMulInstruction,
            )
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
                    factor_args.append(Argument(f"W_{key}", factor.shape))
                instruction = FastLowRankMatMulInstruction(
                    node.target, target_list, args[0], *factor_args,
                )
                self._emit(
                    instruction,
                    [input_lists[0], *factor_lists],
                    target_list,
                )
                return

            raw_key = input_lists[1].removeprefix("W_")
            self.static_lists[raw_key] = factors[0]
            instruction = FastPrunedMatMulInstruction(
                node.target, target_list, *args,
                weight=factors[0],
            )
        elif node.name in self.graph.causal_softmaxes:
            if self.graph.fast and self.graph.fast_config.softmax:
                from cattorch.util.instruction.optimized import FastCausalSoftmaxInstruction
                instruction = FastCausalSoftmaxInstruction(node.target, target_list, *args)
                self.approximate[aten_op] += 1
            else:
                from cattorch.util.instruction.optimized import CausalSoftmaxInstruction
                instruction = CausalSoftmaxInstruction(node.target, target_list, *args)
        elif aten_op == "aten.embedding.default":
            kernel = self.graph.registry.kernel(
                aten_op,
                fast=self.graph.fast,
                config=self.graph.fast_config,
            )
            instruction = self._instantiate(
                kernel, lowering_target, node, target_list, args,
            )
            instruction.interleaved_weight = self._uses_grouped_tied_weight(node)
        elif node.name in self.graph.causal_value_matmuls:
            from cattorch.util.instruction.optimized import CausalValueMatMulInstruction
            value_offset = self.graph.qkv_offsets.get(node.args[1].name, 0)
            instruction = CausalValueMatMulInstruction(
                node.target, target_list, *args,
                Argument("C_value_offset", torch.Size([]), value=value_offset),
            )
        else:
            kernel = self.graph.registry.kernel(
                aten_op,
                fast=self.graph.fast,
                config=self.graph.fast_config,
            )
            if kernel is None:
                raise _unsupported_operation(node)
            if self.graph.fast and rule is not None and kernel is rule.fast_kernel:
                self.approximate[aten_op] += 1
            if aten_op == "aten.linear.default" and linear_layout is not None:
                weight_shard_rows, interleaved = linear_layout
                instruction = kernel(
                    lowering_target,
                    target_list,
                    *args,
                    weight_shard_rows=weight_shard_rows,
                    interleaved=interleaved,
                )
            else:
                instruction = self._instantiate(
                    kernel, lowering_target, node, target_list, args,
                )
        instruction.transform_weights(self.static_lists)
        self._emit(instruction, input_lists, target_list)
        # Context kernels report the scratch lists they allocated, so the
        # sharding pass can size them like any other tensor list.
        for list_name, size in getattr(instruction, "temporary_sizes", {}).items():
            self.logical_sizes[list_name] = max(
                self.logical_sizes.get(list_name, 0), size,
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
            from cattorch.util.instruction.optimized import OptimizedGetItemInstruction
            instruction = OptimizedGetItemInstruction(node.target, target_list, *args)
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
                self.logical_sizes.get(cat_target, 0), output_size,
            )
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
            from cattorch.util.instruction.optimized import OptimizedCatInstruction
            instruction = OptimizedCatInstruction(node.target, cat_target, *cat_args)
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
        trailing = math.prod(input_shape[dim + 1:]) if dim + 1 < len(input_shape) else 1
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
            from cattorch.util.instruction.optimized import StridedSliceInstruction
            instruction = StridedSliceInstruction(node.target, target_list, *args)
            self._emit(instruction, input_lists, target_list)
            return

        chunk_size = max(0, end - start) * trailing
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
        from cattorch.util.instruction.optimized import OptimizedGetItemInstruction
        instruction = OptimizedGetItemInstruction(node.target, target_list, *args)
        self._emit(instruction, input_lists, target_list)

    # ── getitem on split/chunk ───────────────────────────────────────────

    def _compile_getitem(self, node, target_list):
        split_node = node.args[0]
        chunk_index = node.args[1]
        input_shape, split_size, dim = self.graph.split_meta[split_node.name]

        source_name = self.graph.aliases[split_node.name]
        src_list = self._resolve_name(source_name)

        trailing = (
            math.prod(input_shape[dim + 1:])
            if dim + 1 < len(input_shape) else 1
        )
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
                split_size, dimension_size - chunk_index * split_size,
            )
            offset = chunk_index * split_size
        num_rows = math.prod(input_shape[:dim]) if dim > 0 else 1
        chunk_size *= trailing
        offset *= trailing
        skip = row_stride - chunk_size

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
        from cattorch.util.instruction.optimized import OptimizedGetItemInstruction
        instruction = OptimizedGetItemInstruction(node.target, target_list, *args)
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

    def _prepare_linear_weight(self, node, input_lists):
        """Lay out dense Linear rows for grouped execution and safe sharding.

        Four-row groups are interleaved at export time so the hot kernel needs
        one running weight index. Oversized matrices are split only between
        complete output rows (and complete four-row groups), keeping shard
        selection outside every dot-product loop.
        """
        from cattorch.util.instruction.optimized import LinearInstruction

        weight = self.graph.resolve_weight(node.args[1].name)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            return input_lists, (), False
        if self.graph.fast and self.graph.fast_config.weights.enabled:
            # Structured pruning owns this weight's layout and sparse metadata;
            # low-rank replacement may also rewrite the surrounding graph.
            return input_lists, (), False
        columns, inner = weight.shape
        grouped = (
            columns % 4 == 0
            and inner * columns >= LinearInstruction.grouped_min_macs
        )
        oversized = weight.numel() > SCRATCH_LIST_LIMIT
        if not grouped and not oversized:
            return input_lists, (), False

        rows_per_shard = SCRATCH_LIST_LIMIT // inner
        if rows_per_shard < 1:
            # A single output row crosses a shard boundary. The generic
            # sharder remains the correctness fallback for this rare shape.
            return input_lists, (), False
        if grouped:
            rows_per_shard -= rows_per_shard % 4
            if rows_per_shard < 4:
                return input_lists, (), False
        elif not oversized:
            rows_per_shard = columns

        tensors = []
        row_counts = []
        for start in range(0, columns, rows_per_shard):
            shard = weight[start:start + rows_per_shard].detach().clone()
            if grouped:
                shard = (
                    shard.reshape(-1, 4, inner)
                    .permute(0, 2, 1)
                    .contiguous()
                    .reshape(shard.shape)
                )
            tensors.append(shard)
            row_counts.append(shard.shape[0])

        if (
            grouped
            and not oversized
            and self._weight_identity(weight) in self._grouped_tied_weights
            and input_lists[1].startswith("W_")
        ):
            # Keep one physical copy of a tied token embedding/output head.
            # The embedding kernel reads this four-row interleaved layout with
            # a stride-four inner loop.
            original_key = input_lists[1].removeprefix("W_")
            self.static_lists[original_key] = tensors[0]
            if self.row_aligned_weights:
                self.interleaved_static_layouts[original_key] = ((0, columns, inner),)
            return input_lists, (columns,), True

        names = []
        for index, tensor in enumerate(tensors, 1):
            key = f"linear_{node.name}_weight_{index}"
            self.static_lists[key] = tensor
            if grouped and self.row_aligned_weights:
                self.interleaved_static_layouts[key] = ((
                    0, tensor.shape[0], inner,
                ),)
            names.append(f"W_{key}")
        if (
            grouped
            and oversized
            and self._weight_identity(weight) in self._grouped_tied_weights
            and input_lists[1].startswith("W_")
        ):
            original_name = input_lists[1]
            original_key = original_name.removeprefix("W_")
            self.static_lists[original_key] = torch.cat(tensors, dim=0)
            self.static_shard_alignments[original_name] = 4 * inner
            self.tied_storage_aliases[original_name] = tuple(names)
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
                input_lists[0], input_lists[1], input_lists[3],
                input_lists[2], input_lists[4],
            ], ()
        hidden, inner = gate_weight.shape
        rows_per_shard = SCRATCH_LIST_LIMIT // inner
        if rows_per_shard < 1:
            return [
                input_lists[0], input_lists[1], input_lists[3],
                input_lists[2], input_lists[4],
            ], ()
        if gate_weight.numel() <= SCRATCH_LIST_LIMIT:
            return [
                input_lists[0], input_lists[1], input_lists[3],
                input_lists[2], input_lists[4],
            ], (hidden,)

        names = []
        row_counts = []
        for index, start in enumerate(range(0, hidden, rows_per_shard), 1):
            end = min(hidden, start + rows_per_shard)
            gate_key = f"swiglu_{node.name}_gate_weight_{index}"
            value_key = f"swiglu_{node.name}_value_weight_{index}"
            self.static_lists[gate_key] = gate_weight[start:end].detach().clone()
            self.static_lists[value_key] = value_weight[start:end].detach().clone()
            names.extend((f"W_{gate_key}", f"W_{value_key}"))
            row_counts.append(end - start)
        return [input_lists[0], *names, input_lists[2], input_lists[4]], tuple(row_counts)

    def _large_static_matmul(self, node, target_list, left_arg, rhs):
        """Use row-aligned output-major weights only when sharding dominates."""
        from cattorch.util.instruction.optimized import LinearInstruction

        weight = rhs.T.contiguous()
        columns, inner = weight.shape
        rows_per_shard = SCRATCH_LIST_LIMIT // inner
        grouped = (
            columns % 4 == 0
            and columns * inner >= LinearInstruction.grouped_min_macs
            and rows_per_shard >= 4
        )
        if grouped:
            rows_per_shard -= rows_per_shard % 4
        if rows_per_shard < 1:
            raise ValueError(
                "static matmul reduction dimension exceeds one Scratch list"
            )

        names = []
        row_counts = []
        for index, start in enumerate(range(0, columns, rows_per_shard), 1):
            shard = weight[start:start + rows_per_shard].detach().clone()
            if grouped:
                shard = (
                    shard.reshape(-1, 4, inner)
                    .permute(0, 2, 1)
                    .contiguous()
                    .reshape(shard.shape)
                )
            key = f"matmul_{node.name}_weight_{index}"
            self.static_lists[key] = shard
            if grouped and self.row_aligned_weights:
                self.interleaved_static_layouts[key] = ((
                    0, shard.shape[0], inner,
                ),)
            names.append(f"W_{key}")
            row_counts.append(shard.shape[0])

        matrix_inputs = [left_arg.name, *names, "_none", "_none"]
        instruction_args = (
            left_arg,
            Argument(names[0], torch.Size((columns, inner))),
            Argument("_none", torch.Size([]), value=None),
            Argument("_none", torch.Size([]), value=None),
        )
        instruction = LinearInstruction(
            node.target,
            target_list,
            *instruction_args,
            weight_shard_rows=tuple(row_counts),
            interleaved=grouped,
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
            signature = (
                tensor.untyped_storage().data_ptr(), tensor.storage_offset(),
                tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype,
            )
            if signature in self._static_tensor_names:
                return self._static_tensor_names[signature]
        name = f"W_{arg_name}"
        self.static_lists[arg_name] = tensor
        if isinstance(tensor, torch.Tensor):
            self._static_tensor_names[signature] = name
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
            bid for bid, block in data["blocks"].items()
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
                node, "its kernel can't run once per row of a variable-length input",
            )
        dynamic = {
            self._resolve_name(arg.name): math.prod(_get_shape(arg))
            for arg in node.all_input_nodes if arg.name in self.graph.dynamic_axes
        }
        tensor_lists = [name for name in input_lists if not name.startswith("C_")]
        rows = {
            f"T{position}": dynamic[name]
            for position, name in enumerate(tensor_lists, 1) if name in dynamic
        }
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
        static_tensors() if static_tensors is not None
        else (*model.parameters(), *model.buffers())
    ):
        identity = (
            tensor.untyped_storage().data_ptr(), tensor.storage_offset(),
            tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype,
        )
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
            group_size = _static_storage_group_size(tensor, storage)
            scale_chars = (
                32 / 13 if storage.scale_precision == "float16" else 64 / 13
            )
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
    example_inputs: torch.Tensor | tuple[torch.Tensor, ...] | ExportProgram | GenerationProgram | None,
    output_path: str | Path,
    sig_figs: int | None = None,
    *,
    name: str | None = None,
    optimization: Literal["exact", "fast"] = "exact",
    fast_config: FastConfig | None = None,
    storage: StorageConfig | None = None,
    quantization: QuantizationConfig | None = None,
    calibration_inputs: Iterable[
        torch.Tensor | tuple[torch.Tensor, ...]
    ] | None = None,
    codegen: CodegenConfig | None = None,
    frontend: Frontend = "fx",
    adapters: tuple[ModuleAdapter, ...] = (),
    calibration_calls: tuple[ProgramCall, ...] | None = None,
    calibration_sequences: Iterable[
        torch.Tensor | tuple[torch.Tensor, ...]
    ] | None = None,
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
        defining named methods and state, or a ``GenerationProgram`` defining
        the cached decoder interface. ``None`` takes the interface from the
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
        Token sequences used to calibrate a ``GenerationProgram`` with GPTQ.
    codegen:
        Expanded-JSON budget, loop-unrolling, compact-ID, and optional
        structural-compaction policy.
    frontend:
        How the model is traced. ``"fx"`` (default) uses FX tracing.
        ``"export"`` uses ``torch.export``; it is deprecated, supports fewer
        operations, and will be removed in 2.0. Programs require ``"fx"``.
    adapters:
        Scoped module adapters for an ``ExportProgram`` or
        ``GenerationProgram``. They apply to an export-owned model copy.
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
        name=name, optimization=optimization, fast_config=fast_config,
        storage=storage, quantization=quantization, codegen=codegen,
        sig_figs=sig_figs, frontend=frontend, adapters=adapters,
        example_kwargs=example_kwargs,
    )
    if frontend == "export":
        warnings.warn(
            "frontend='export' is deprecated and will be removed in cattorch 2.0; "
            "the default frontend='fx' supports every model it does, and more",
            DeprecationWarning,
            stacklevel=2,
        )
    kernels = validate_kernels(kernels)
    kept = active_kernel_targets(
        kernels, fast=optimization == "fast", config=fast_config or FastConfig(),
    )
    with keep_operations(kept):
        return _transpile_any(
            model, example_inputs, output_path, sig_figs,
            name=name, optimization=optimization, fast_config=fast_config,
            storage=storage, quantization=quantization,
            calibration_inputs=calibration_inputs, codegen=codegen,
            frontend=frontend, adapters=adapters,
            calibration_calls=calibration_calls,
            calibration_sequences=calibration_sequences,
            example_kwargs=example_kwargs, kernels=kernels, dynamic=dynamic,
        )


def _transpile_dynamic(
    model, example_inputs, output_path, sig_figs, dynamic, *, example_kwargs, kernels,
    adapters=(), **options,
):
    """Trace one row of the dynamic input, export, and check other lengths."""
    from cattorch.dynamic import (
        at_length, check_lengths, exact_export, loop_over_units, normalize_dynamic,
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
        generation=None, example_kwargs=traced_kwargs or None,
        _registry=_registry_with(kernels, adapters) if kernels or adapters else None,
        **options,
    )
    try:
        result = _transpile(
            model, tuple(representative[:positional]), output_path, sig_figs,
            _dynamic_inputs=spec, **arguments,
        )
    except UnsupportedModelError as error:
        # Operations that can't follow the length directly: run the whole
        # model once per item instead, if items are independent.
        try:
            axes = unit_output_axes(model, flat, kwarg_names, spec, example_kwargs)
        except UnsupportedModelError:
            raise error from None
        result = _transpile(
            model, tuple(representative[:positional]), output_path, sig_figs, **arguments,
        )
        result = loop_over_units(
            result, spec, axes,
            [tuple(value.shape) for value in representative],
            [tuple(item for item in spec_value.shape) for spec_value in result.outputs],
        )
    exact = exact_export(
        optimization=options.get("optimization", "exact"), storage=options.get("storage"),
        quantization=options.get("quantization"), sig_figs=sig_figs,
    )
    check_lengths(model, flat, kwarg_names, spec, result, verify, example_kwargs, exact=exact)
    return result


def _registry_with(kernels, adapters=()) -> OperatorRegistry:
    registry = default_registry().clone()
    for adapter in adapters:
        registry.register_adapter(adapter)
    for kernel in kernels:
        registry.register_kernel(kernel)
    return registry


def _transpile_any(
    model, example_inputs, output_path, sig_figs, *, name, optimization,
    fast_config, storage, quantization, calibration_inputs, codegen, frontend,
    adapters, calibration_calls, calibration_sequences, example_kwargs, kernels,
    dynamic=None,
):
    if example_inputs is None:
        # The model describes its own interface with markers.
        from cattorch.markers import example_inputs as marked_example, has_entrypoints, marked_program
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
    if example_kwargs is not None and isinstance(
        example_inputs, (ExportProgram, GenerationProgram),
    ):
        raise ValueError(
            "example_kwargs cannot be used with an ExportProgram or GenerationProgram"
        )
    if isinstance(example_inputs, ExportProgram):
        if frontend != "fx":
            raise ValueError("ExportProgram is supported only by the FX frontend")
        if calibration_inputs is not None:
            raise ValueError(
                "ExportProgram uses calibration_calls instead of calibration_inputs"
            )
        if calibration_sequences is not None:
            raise ValueError("calibration_sequences require a GenerationProgram")
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
        if frontend != "fx":
            raise ValueError("GenerationProgram is supported only by the FX frontend")
        if any(not isinstance(adapter, ModuleAdapter) for adapter in adapters):
            raise TypeError("adapters must contain ModuleAdapter values")
        if calibration_calls is not None:
            raise ValueError("calibration_calls require an ExportProgram")
        if calibration_inputs is not None:
            raise ValueError(
                "GenerationProgram uses calibration_sequences instead of calibration_inputs"
            )
        if not hasattr(model, example_inputs.method) or not callable(
            getattr(model, example_inputs.method)
        ):
            raise ValueError(
                f"model has no callable generation method {example_inputs.method!r}"
            )
        generation_program = example_inputs
        registry = _registry_with(kernels, adapters)
        return _transpile_automatic_generation(
            model, generation_program, output_path, sig_figs,
            name=name, optimization=optimization, fast_config=fast_config,
            storage=storage, quantization=quantization,
            calibration_sequences=calibration_sequences,
            codegen=codegen, registry=registry,
        )
    if calibration_calls is not None:
        raise ValueError("calibration_calls require an ExportProgram")
    if calibration_sequences is not None:
        raise ValueError("calibration_sequences require a GenerationProgram")
    if dynamic is not None:
        return _transpile_dynamic(
            model, example_inputs, output_path, sig_figs, dynamic,
            name=name, optimization=optimization, fast_config=fast_config,
            storage=storage, quantization=quantization,
            calibration_inputs=calibration_inputs, codegen=codegen,
            frontend=frontend, example_kwargs=example_kwargs, kernels=kernels,
            adapters=adapters,
        )
    return _transpile(
        model, example_inputs, output_path, sig_figs,
        name=name,
        optimization=optimization, fast_config=fast_config, generation=None,
        storage=storage, quantization=quantization,
        calibration_inputs=calibration_inputs,
        codegen=codegen,
        frontend=frontend,
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
        GenerationStepModule, MethodLeaves, build_generation_step,
    )

    config = fast_config or FastConfig()
    storage = storage or StorageConfig()
    codegen = codegen or CodegenConfig()
    if calibration_sequences is not None and (
        quantization is None or quantization.method != "gptq"
    ):
        raise ValueError("calibration_inputs can only be used with GPTQ quantization")
    max_context = program.max_context
    if any(
        value.is_floating_point() and value.dtype != torch.float32
        for value in (*model.parameters(), *model.buffers())
    ):
        # Scratch computes in doubles either way. In float32 the cached step
        # can be checked against the model without bfloat16 rounding noise.
        model = copy.deepcopy(model).float()
    module = model if program.method == "forward" else MethodLeaves(model, program.method)
    prepared = registry.adapt_model(module)
    if optimization == "fast":
        prepared = prepare_fast_model(prepared, config)
    prepared = fold_eval_batch_norms(prepared)
    report = None
    if quantization is not None:
        report = quantize_model(
            prepared,
            quantization,
            calibration_sequences,
            orientation_inputs=[(torch.zeros(1, max_context, dtype=torch.long),)],
            _registry=registry,
        )
    step = GenerationStepModule(build_generation_step(
        prepared, max_context, prepared=True, registry=registry,
    ))
    spec = _GenerationSpec(
        max_context, program.hidden_prefill, program.top_k,
    )
    options = dict(
        name=name, optimization=optimization, fast_config=fast_config,
        generation=spec, storage=storage, quantization=quantization,
        frontend="fx", _generation_public_interface=True, _registry=registry,
        _prepared_model=True, _prepared_quantization_report=report,
    )
    example = torch.zeros(1, 1, dtype=torch.long)
    try:
        # Tracing runs the step, so every attempt gets its own copy.
        return _transpile(
            copy.deepcopy(step), (example,), output_path, sig_figs,
            codegen=codegen, **options,
        )
    except _LayerSharingUnavailable:
        log.warning(
            "layer_sharing='auto' could not factor this model's layers; "
            "exporting without layer sharing"
        )
        return _transpile(
            copy.deepcopy(step), (example,), output_path, sig_figs,
            codegen=replace(codegen, layer_sharing="off"), **options,
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
    calibration_inputs: Iterable[
        torch.Tensor | tuple[torch.Tensor, ...]
    ] | None = None,
    codegen: CodegenConfig | None = None,
    frontend: Frontend = "fx",
    example_kwargs: dict[str, Any] | None = None,
    _registry: OperatorRegistry | None = None,
    _generation_public_interface: bool = False,
    _prepared_model: bool = False,
    _prepared_quantization_report=None,
    _dynamic_inputs: dict[int, tuple[str, int, int]] | None = None,
    _defer_interface_sharding: bool = False,
) -> TranspileResult:
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
    if calibration_inputs is not None and (
        quantization is None or quantization.method != "gptq"
    ):
        raise ValueError("calibration_inputs can only be used with GPTQ quantization")
    if (
        quantization is not None and quantization.method == "gptq"
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
        model, example_inputs,
        optimization=optimization,
        fast_config=config,
        generation=generation,
        frontend=frontend,
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
    compiler = _Compiler(graph, row_aligned_weights=storage.grouping == "row")
    selected_unroll = _select_unroll_factor(model, storage, codegen)
    log.info(
        "Code generation unroll cap: %s (%s mode)",
        selected_unroll,
        codegen.unrolling,
    )
    with unroll_factor_limit(selected_unroll):
        sprite = compiler.compile()
    compiler.static_lists = _deduplicate_static_tensors(
        sprite, compiler.static_lists,
        effective_ndims=compiler.static_effective_ndims,
    )

    # Layout-specialized kernels may replace an original static tensor with
    # one or more derived lists. Keep only weights that survived into blocks;
    # otherwise compressed storage would retain an unreachable payload.
    used_lists = {entry[0] for entry in sprite.get("lists", {}).values()}
    compiler.static_lists = {
        key: tensor
        for key, tensor in compiler.static_lists.items()
        if f"W_{key}" in used_lists
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
            generation is not None and graph.generation.banked_caches
            and graph.generation.hidden_prefill_attention is not None
        )
        shared = _share_repeated_transformer_layers(
            sprite, compiler, storage,
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
    output_names = tuple(
        output_name(index)
        for index in range(len(output_lists))
    )
    for old_name, public_name in zip(output_lists, output_names):
        _rename_list(sprite, old_name, public_name)
    output = output_lists[0] if output_lists else None

    logical_sizes = _logical_list_sizes(
        compiler, graph, generation, output,
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
        logical_sizes[list_name] = maximum
    for old_name, public_name, (node_name, shape, _) in zip(
        output_lists, output_names, graph.outputs,
    ):
        # Output copies (including returned inputs and repeated leaves) have
        # no scope assignment of their own. Size every public output from its
        # contract so those copies are sharded too, using runtime capacity.
        capacity = (
            graph.runtime_capacity(node_name, shape) if node_name in graph.dynamic_axes
            else math.prod(shape)
        )
        logical_sizes[public_name] = max(
            capacity, logical_sizes.get(public_name, 0), logical_sizes.pop(old_name, 0),
        )
    if generation is not None and generation.top_k is not None:
        # The output list may be a reused, larger scratch list; the logits
        # row is the output's own shape.
        output_size = math.prod(graph.outputs[0][1])
        if generation.top_k > output_size:
            raise ValueError(
                f"top_k ({generation.top_k}) exceeds output size ({output_size})"
            )

    cache_names = set(graph.generation.score_caches.values()) | set(
        graph.generation.value_caches.values()
    )
    _merge_lists_by_name(sprite, cache_names)
    _merge_duplicate_lists(sprite)
    _wrap_optimized_lifecycle(sprite)
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
                    position for position, name in enumerate(root_names)
                    if name in by_name and _graph_layer(by_name[name]) is not None
                )
                if last_layer + 1 >= len(root_entries):
                    raise ValueError("hidden prefill requires a compiled output projection")
                _guard_generation_suffix(
                    sprite, root_entries[last_layer + 1][1], _PROJECT_OUTPUT,
                )
                _merge_variables_by_name(sprite, {_PROJECT_OUTPUT, _LAYER_TAIL})
            elif (
                hidden_attention in root_names
                and root_names.index(hidden_attention) + 1 < len(root_entries)
            ):
                # Keep the final layer's K/V append, skip its attention output
                # and everything after it for hidden prompt tokens.
                for entry in sprite.get("variables", {}).values():
                    if entry[0] == _PROJECT_OUTPUT:
                        entry[1] = 1
                _guard_hidden_layer_tail(sprite, graph, compiler, hidden_attention)
            else:
                if not compiler.emitted_roots:
                    raise ValueError(
                        "hidden prefill requires a compiled output projection"
                    )
                _guard_generation_output(sprite, compiler.emitted_roots[-1][1])
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
            vocabulary=_token_vocabulary(graph),
        )
        # Kernels declare `cattorch cache length` themselves; unify them with
        # the lifecycle's variable.
        _merge_variables_by_name(sprite, {"cattorch cache length"})
        sprite = _wrap_transactional_generation_step(
            sprite,
            {
                cache_name: graph.generation.cache_widths[cache_name]
                for cache_name in cache_names
            },
            top_k=generation.top_k is not None,
        )
        # The step wrapper declares its own copy of the status variable.
        _merge_variables_by_name(sprite, {STATUS_VARIABLE})
        generation_lists = {
            "input", "output", "cattorch prefill buffer",
        }
        if generation.top_k is not None:
            generation_lists.update({_TOP_K_VALUES, _TOP_K_IDS})
        _merge_lists_by_name(
            sprite,
            cache_names | generation_lists,
        )
        _merge_duplicate_lists(sprite)
        if _generation_public_interface:
            _rename_procedure(sprite, "cattorch forward", "_cattorch token step")
            _rename_procedure(
                sprite, "cattorch reset cache", "_cattorch reset state",
            )
            sprite = _add_warp_procedure(
                sprite,
                "cattorch reset",
                Program(
                    "generation_public_reset",
                    body=(call("_cattorch reset state"),),
                ),
                x=640,
                y=640,
            )
            _rename_list(sprite, "input", "cattorch tokens")
            _rename_list(sprite, "output", "cattorch logits")
            _rename_list(
                sprite, "cattorch prefill buffer", "_cattorch prefill buffer",
            )
            for old, new in (
                ("input", "cattorch tokens"),
                ("output", "cattorch logits"),
                ("cattorch prefill buffer", "_cattorch prefill buffer"),
            ):
                if old in logical_sizes:
                    logical_sizes[new] = logical_sizes.pop(old)

    layouts = shard_sprite_lists(
        sprite,
        logical_sizes,
        deferred_names=(
            frozenset((*graph.input_names.values(), *output_names))
            if _defer_interface_sharding else frozenset()
        ),
        alignments=compiler.static_shard_alignments,
        limits=_static_storage_shard_limits(
            compiler.static_lists, storage,
            effective_ndims=compiler.static_effective_ndims,
            interleaved_layouts=compiler.interleaved_static_layouts,
        ),
    )
    _apply_tied_static_aliases(
        sprite,
        layouts,
        compiler.static_lists,
        compiler.tied_storage_aliases,
    )
    static_names = _apply_static_storage(
        sprite,
        layouts,
        compiler.static_lists,
        storage,
        interleaved_layouts=compiler.interleaved_static_layouts,
        effective_ndims=compiler.static_effective_ndims,
    )
    # GenerationProgram deliberately exposes only its four lifecycle methods.
    # Its processor sprite is intended to remain initialized while a project
    # runs, and emitting one clear block for every decoded weight shard is a
    # material JSON cost for the large language models this interface serves.
    if not _generation_public_interface:
        _add_prepare_for_save(
            sprite,
            layouts,
            static_names=static_names,
            compressed=storage.compression,
            inputs=set(graph.input_names.values()),
        )
    if generation is not None:
        _add_procedure_completion_broadcast(
            sprite, "cattorch init", "cattorch init complete", status="ok",
        )
        _add_procedure_completion_broadcast(
            sprite, "cattorch prefill", "cattorch prefill complete",
        )
        if _generation_public_interface:
            _add_procedure_completion_broadcast(
                sprite, "cattorch reset", "cattorch reset complete",
            )
            _add_procedure_completion_broadcast(
                sprite, "cattorch decode", "cattorch decode complete",
            )
    if generation is None and not _defer_interface_sharding:
        # Like generation and programs: a status and a notification when each
        # public block finishes. Program entry points get theirs from the
        # program.
        variables = sprite.setdefault("variables", {})
        if not any(entry[0] == STATUS_VARIABLE for entry in variables.values()):
            variables["cattorch_status"] = [STATUS_VARIABLE, ""]
        for procedure in ("cattorch init", "cattorch forward"):
            _add_procedure_completion_broadcast(
                sprite, procedure, f"{procedure} complete", status="ok",
            )
        _check_input_lengths(sprite, graph, layouts, output_names)
    sprite.setdefault("variables", {}).update({
        "cattorch_storage_compression": [
            "cattorch storage compression",
            (
                "costume-base92-huffman-int4"
                if storage.compression and storage.precision == "int4"
                else "costume-base92" if storage.compression else "none"
            ),
        ],
        "cattorch_storage_precision": [
            "cattorch storage precision", storage.precision,
        ],
        "cattorch_storage_scale_precision": [
            "cattorch storage scale precision", storage.scale_precision,
        ],
        "cattorch_quantization_method": [
            "cattorch quantization method",
            quantization.method if quantization is not None else "none",
        ],
        "cattorch_quantization_bits": [
            "cattorch quantization bits",
            quantization.bits if quantization is not None else 0,
        ],
    })
    _remove_unused(sprite)
    uniquify_data_ids(sprite)

    log.info("Dynamic lists: %s", list(compiler.dynamic_lists))
    log.info("Static lists: %s", list(compiler.static_lists.keys()))

    finalized = finalize_sprite(
        sprite, output_path, sprite_name=sprite_name, codegen=codegen,
        # A program finishes its entry points' interface itself.
        public_interface=not _defer_interface_sharding,
    )

    input_specs = tuple(
        boundary_spec(
            (
                "cattorch tokens"
                if generation is not None and _generation_public_interface
                else (input_name(index))
            ),
            tuple(value.shape),
            value.dtype,
            (
                (
                    _dynamic_inputs[index][3] if len(_dynamic_inputs[index]) > 3 else 0,
                    tuple(_dynamic_inputs[index][:3]),
                )
                if _dynamic_inputs is not None and index in _dynamic_inputs
                else None
            ),
        )
        for index, value in enumerate(flat_example_inputs(example_inputs, example_kwargs))
    )
    procedures = ["cattorch init", "cattorch forward", "cattorch prepare for save"]
    if generation is not None:
        if _generation_public_interface:
            procedures = [
                "cattorch init", "cattorch reset", "cattorch prefill",
                "cattorch decode",
            ]
        else:
            procedures.extend(("cattorch reset cache", "cattorch prefill", "cattorch decode"))
    return TranspileResult(
        path=finalized.path,
        sprite_name=sprite_name,
        archive_bytes=finalized.archive_bytes,
        expanded_json_bytes=finalized.expanded_json_bytes,
        block_count=len(sprite.get("blocks", {})),
        list_count=len(sprite.get("lists", {})),
        sharded_lists=tuple(
            (layout.name if _defer_interface_sharding else final_data_name(layout.name))
            for layout in layouts.values() if len(layout.shards) > 1
        ),
        inputs=input_specs,
        output=boundary_spec(
            (
                "cattorch logits"
                if generation is not None and _generation_public_interface
                else "output"
            ),
            tuple(graph.output_shape),
            graph.output_dtype,
            (
                (graph.dynamic_axes[graph.output_name], graph._public_bounds(graph.output_name))
                if graph.output_name in graph.dynamic_axes else None
            ),
        ),
        additional_outputs=tuple(
            boundary_spec(
                f"output_{index}",
                tuple(shape),
                dtype,
                (
                    (graph.dynamic_axes[node_name], graph._public_bounds(node_name))
                    if node_name in graph.dynamic_axes else None
                ),
            )
            for index, (node_name, shape, dtype) in enumerate(graph.outputs[1:], 1)
        ),
        procedures=tuple(procedures),
        warnings=finalized.warnings + tuple(
            f"GPTQ fallback for {item.name}: {item.fallback_reason}"
            for item in (quantization_report.tensors if quantization_report else ())
            if item.method == "symmetric" and item.fallback_reason is not None
        ) + tuple(filter(None, (fallback_warning(graph.fallbacks()),))),
        quantization=quantization_report,
        fallback_operations=graph.fallbacks(),
        approximate_operations=tuple(sorted(compiler.approximate.items())),
    )
