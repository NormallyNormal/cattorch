"""Captured PyTorch graph preparation and fusion analysis."""

from __future__ import annotations

import logging
import math
import operator
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import torch

from cattorch.errors import UnsupportedModelError, UnsupportedOperationError
from cattorch.fast import FastConfig
from cattorch.frontend import FALLBACK_META, CapturedGraph, Frontend, capture_model
from cattorch.results import _tensor_spec
from cattorch.operator_registry import (
    ELEMENTWISE_OPS, MATMUL_OPS, OperatorRegistry, default_registry,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _GenerationSpec:
    """Internal lowering policy for a validated generation program.

    ``max_context`` fixes the positive token capacity. ``hidden_prefill`` uses
    a recognized hidden-only prompt path to avoid projecting logits at every
    prompt position. ``top_k`` optionally materializes the best 1--16 logits
    and zero-based token IDs after prefill and decode.
    """

    max_context: int
    hidden_prefill: bool = True
    top_k: int | None = None

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
    "aten.detach.default",
    "aten.detach_.default",
    "aten.lift_fresh_copy.default",
    "aten._assert_tensor_metadata.default",
    "aten.unflatten.int",
    "aten.squeeze.dim",
    "aten.squeeze.dims",
    "aten.squeeze.default",
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


def _graph_handled_operations() -> frozenset[str]:
    """Operations graph analysis lowers itself, without a registry kernel."""
    return frozenset(
        _NOOP_OPS | _SPLIT_OPS | set(_TENSOR_GENERATORS) | {
            "aten.to.dtype", "aten.dropout.default", "aten.slice.Tensor",
            "aten.cat.default", "aten.ones_like.default", "aten.zeros_like.default",
            "aten.transpose.int", "aten.permute.default", "aten.numpy_T.default",
        }
    )


# Folding a static subgraph never stores a result larger than this. A larger
# tensor would usually come from an expand that is cheaper to compute in Scratch.
_FOLD_ELEMENT_LIMIT = 1 << 20


_CACHED_ATTENTION = "cattorch.cached_attention.default"
_SEQUENCE_ROW = "cattorch.sequence_row.default"
_ADVANCE_POSITION = "cattorch.advance_position.default"


def _generation_state_names(nodes, state: dict[str, torch.Tensor]) -> set[str]:
    """Names of the state tensors a step module changes while it runs.

    These are the caches and position scalar passed to the step module's
    custom ops. make_fx gives every use its own ``get_attr`` name, so the
    tensors are matched by identity.
    """
    runtime: set[int] = set()
    for node in nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        if target == _CACHED_ATTENTION:
            roles = node.args[3:6]
        elif target == _SEQUENCE_ROW:
            roles = node.args[1:2]
        else:
            continue
        for argument in roles:
            value = state.get(getattr(argument, "name", None))
            if value is not None:
                runtime.add(id(value))
    return {name for name, value in state.items() if id(value) in runtime}


def _is_foldable(node) -> bool:
    """Whether ``node`` may be evaluated at export time on static inputs."""
    if node.target is operator.getitem:
        return True
    schema = getattr(node.target, "_schema", None)
    if schema is None or schema.is_mutable:
        return False
    tags = getattr(node.target, "tags", ())
    return torch.Tag.nondeterministic_seeded not in tags


def fold_static_values(
    nodes,
    state: dict[str, torch.Tensor],
    *,
    seed: dict[str, torch.Tensor] | None = None,
    element_limit: int | None = _FOLD_ELEMENT_LIMIT,
    passthrough: frozenset[str] = frozenset(),
) -> dict[str, torch.Tensor]:
    """Evaluate every foldable node whose tensor inputs are all static.

    ``state`` maps get_attr names to tensors and ``seed`` holds values already
    generated for other nodes. Returns the newly folded tensors by node name,
    without ``seed``. Results above ``element_limit`` elements are skipped
    (``None`` folds them all). A node whose target is in ``passthrough``
    takes its first argument's value without being called, for aliasing
    operations such as ``detach_`` that are not otherwise foldable.
    """
    folded: dict[str, torch.Tensor] = {}
    known = dict(seed or {})
    missing = object()
    static_results: dict[str, object] = {}

    def static_value(arg):
        if hasattr(arg, "name"):
            if arg.name in known:
                return known[arg.name]
            if arg.name in static_results:
                return static_results[arg.name]
            found = state.get(arg.name)
            return found if found is not None else missing
        if isinstance(arg, (list, tuple)):
            values = [static_value(item) for item in arg]
            if any(item is missing for item in values):
                return missing
            return type(arg)(values)
        return arg

    with torch.no_grad():
        for node in nodes:
            if node.op != "call_function":
                continue
            if str(node.target) in passthrough:
                value = static_value(node.args[0])
                if isinstance(value, torch.Tensor):
                    folded[node.name] = known[node.name] = value
                continue
            if not _is_foldable(node):
                continue
            args = static_value(node.args)
            kwargs = static_value(node.kwargs)
            if args is missing or kwargs is missing:
                continue
            try:
                result = node.target(*args, **kwargs)
            except (RuntimeError, TypeError, ValueError, IndexError):
                continue
            if isinstance(result, torch.Tensor):
                if element_limit is not None and result.numel() > element_limit:
                    continue
                folded[node.name] = known[node.name] = result.detach().contiguous()
                log.info("Constant-folded %s as static weight %s", node.target, node.name)
            elif isinstance(result, (tuple, list)) and all(
                isinstance(value, torch.Tensor) for value in result
            ):
                # Keep multi-output results (split, max.dim) only as
                # intermediates; their getitem users become the weights.
                static_results[node.name] = result
    return folded


def _is_value_preserving_cast(source: torch.dtype, target: torch.dtype) -> bool:
    """Whether a conversion leaves every Scratch number unchanged.

    Scratch numbers are doubles, so conversions to a floating type, and from
    bool or integer to integer, keep the stored value. Float-to-integer
    truncation and conversion to bool need a kernel.
    """
    if source == target or target.is_floating_point:
        return True
    if target == torch.bool:
        return source == torch.bool
    return not source.is_floating_point


def _gelu_approximation(node) -> str:
    if len(node.args) > 1:
        return node.args[1]
    return node.kwargs.get("approximate", "none")


# ── Graph preparation ────────────────────────────────────────────────────────

def _get_shape(arg) -> torch.Size:
    if hasattr(arg, 'meta') and 'val' in arg.meta:
        return arg.meta['val'].shape
    return torch.Size([])


def _get_dtype(arg) -> torch.dtype | None:
    if hasattr(arg, 'meta') and 'val' in arg.meta:
        return getattr(arg.meta['val'], 'dtype', None)
    return None


def _unsupported_operation(node, detail: str | None = None) -> UnsupportedOperationError:
    module_stack = node.meta.get("nn_module_stack", {})
    module_path = next(
        (entry[0] for entry in reversed(tuple(module_stack.values())) if entry[0]),
        None,
    )
    return UnsupportedOperationError(
        str(node.target),
        node_name=node.name,
        module_path=module_path,
        detail=detail,
    )


@dataclass
class _GenerationInfo:
    """Generation-only graph analysis kept out of the common graph contract."""

    score_caches: dict[str, str] = field(default_factory=dict)
    value_caches: dict[str, str] = field(default_factory=dict)
    cache_widths: dict[str, int] = field(default_factory=dict)
    first_cache: tuple[str, int] | None = None
    hidden_prefill_attention: str | None = None
    banked_caches: bool = False
    cache_layers: int = 1
    cache_layer_width: int = 0


@dataclass
class _GraphInfo:
    """Pre-processed captured PyTorch graph, ready for compilation."""
    nodes: list
    resolve_weight: object  # callable
    input_names: dict
    outputs: tuple[tuple[str, torch.Size, torch.dtype], ...]
    aliases: dict            # node_name → source_name for no-op ops
    split_meta: dict         # split_node_name → (input_shape, split_size, dim)
    generated_weights: dict  # node_name → tensor
    fast: bool
    fast_config: FastConfig
    generation: _GenerationInfo
    linear_fusions: dict
    paired_swiglu_fusions: dict
    silu_mul_fusions: dict
    embedding_add_fusions: dict
    elementwise_fusions: dict
    qkv_linear_fusions: dict
    qkv_offsets: dict[str, int]
    causal_score_fusions: dict
    causal_softmaxes: dict
    causal_value_matmuls: set[str]
    fused_nodes: set[str]
    registry: OperatorRegistry
    dynamic_axes: dict[str, int]
    dynamic_bounds: dict[str, tuple[str, int, int]]

    @property
    def output_name(self) -> str:
        return self.outputs[0][0]

    def fallbacks(self) -> tuple[tuple[str, int], ...]:
        """Operations lowered through PyTorch's generic decomposition.

        Each entry is an operation and how many of its calls still run in
        Scratch. Calls folded into constants at export, and decompositions
        that only rename the operation (``expand_as`` to ``expand``), aren't
        counted.
        """
        sizes = Counter(
            node.meta[FALLBACK_META] for node in self.nodes
            if node.op == "call_function" and FALLBACK_META in node.meta
            and node.name not in self.generated_weights
        )
        return tuple(sorted(Counter(
            name for (name, _), size in sizes.items() if size > 1
        ).items()))

    @property
    def output_shape(self) -> torch.Size:
        return self.outputs[0][1]

    @property
    def output_dtype(self) -> torch.dtype:
        return self.outputs[0][2]

    def output_specs(self):
        return tuple(
            _tensor_spec(
                "output" if index == 0 else f"output_{index}", shape, dtype,
                (self.dynamic_axes[name], self.dynamic_bounds[name])
                if name in self.dynamic_axes else None,
            )
            for index, (name, shape, dtype) in enumerate(self.outputs)
        )


def _drop_empty_cat_inputs(graph) -> None:
    """Remove zero-element tensors from ``cat`` and the nodes only they fed.

    Caches start as empty tensors that are concatenated with the first keys
    and values. Scratch has nothing to copy for them, so they never reach
    lowering.
    """
    def empty(value) -> bool:
        meta = getattr(value, "meta", {}).get("val")
        return isinstance(meta, torch.Tensor) and meta.numel() == 0

    for node in list(graph.nodes):
        if node.op != "call_function" or str(node.target) != "aten.cat.default":
            continue
        pieces = list(node.args[0])
        kept = [value for value in pieces if not empty(value)]
        if kept and len(kept) != len(pieces):
            node.args = (kept,) + tuple(node.args[1:])
    for node in reversed(list(graph.nodes)):
        if node.op in {"call_function", "get_attr"} and not node.users and empty(node):
            graph.erase_node(node)


def _prepare_graph(
    model,
    example_inputs,
    *,
    optimization: Literal["exact", "fast"] = "exact",
    fast_config: FastConfig | None = None,
    generation: _GenerationSpec | None = None,
    frontend: Frontend = "fx",
    _registry: OperatorRegistry | None = None,
    dynamic_inputs: dict[int, tuple[str, int, int]] | None = None,
    example_kwargs: dict | None = None,
    _captured: CapturedGraph | None = None,
) -> _GraphInfo:
    """Capture the model, decompose ops, and analyze the graph structure."""
    use_fusions = optimization in {"exact", "fast"}
    fast = optimization == "fast"
    fast_config = fast_config or FastConfig()
    registry = _registry or default_registry()
    from cattorch.moe import _expert_family_context
    with _expert_family_context(registry.expert_families):
        captured = _captured or capture_model(
            model, example_inputs, frontend=frontend, example_kwargs=example_kwargs,
        )
    _drop_empty_cat_inputs(captured.graph)
    if generation is not None:
        # The wrapper tracks the length, so the step module's position
        # increment has nothing to lower.
        for node in list(captured.graph.nodes):
            if (
                node.op == "call_function"
                and str(node.target) == _ADVANCE_POSITION
            ):
                captured.graph.erase_node(node)
    nodes = list(captured.graph.nodes)
    dynamic_inputs = dynamic_inputs or {}
    placeholders = [
        node for node in nodes
        if node.op == "placeholder" and node.name in captured.user_input_names
    ]
    dynamic_axes: dict[str, int] = {}
    dynamic_bounds: dict[str, tuple[str, int, int]] = {}
    for index, bound in dynamic_inputs.items():
        if index >= len(placeholders):
            raise UnsupportedModelError("dynamic input index is outside the captured signature")
        node = placeholders[index]
        if len(_get_shape(node)) < 1:
            raise UnsupportedModelError("a dynamic input must have a leading dimension")
        dynamic_axes[node.name] = 0
        dynamic_bounds[node.name] = bound

    shape_only = _NOOP_OPS | {"aten.to.dtype", "aten.dropout.default"}
    elementwise = {
        "aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor", "aten.div.Tensor",
        "aten.neg.default", "aten.relu.default", "aten.sigmoid.default",
        "aten.tanh.default", "aten.gelu.default", "aten.silu.default",
        "aten.leaky_relu.default", "aten.elu.default", "aten.pow.Tensor_Scalar",
        "aten.rsqrt.default", "aten.linear.default", "aten.layer_norm.default",
        "aten.native_layer_norm.default", "aten._softmax.default", "aten.softmax.int",
        "aten.mean.dim", "aten.embedding.default",
    } | ELEMENTWISE_OPS
    for node in nodes:
        if node.op != "call_function":
            continue
        dynamic_parents = [
            value for value in node.all_input_nodes if value.name in dynamic_axes
        ]
        if not dynamic_parents:
            continue
        origins = {dynamic_bounds[value.name][0] for value in dynamic_parents}
        if len(origins) != 1:
            raise UnsupportedModelError(
                f"{node.target} combines multiple independent runtime extents"
            )
        parent = dynamic_parents[0]
        axis = dynamic_axes[parent.name]
        target = str(node.target)
        input_shape = tuple(_get_shape(parent))
        output_shape = tuple(_get_shape(node))
        if target == "aten.transpose.int":
            left = int(node.args[1]) % len(input_shape)
            right = int(node.args[2]) % len(input_shape)
            if axis == left:
                axis = right
            elif axis == right:
                axis = left
        elif target == "aten.permute.default":
            permutation = [int(value) % len(input_shape) for value in node.args[1]]
            axis = permutation.index(axis)
        elif target == "aten.numpy_T.default":
            axis = len(input_shape) - axis - 1
        elif target in {
            "aten.view.default", "aten.reshape.default", "aten._unsafe_view.default",
            "aten.unflatten.int",
        }:
            suffix = math.prod(input_shape[axis + 1:])
            candidates = [
                position for position, extent in enumerate(output_shape)
                if extent == 1 and math.prod(output_shape[position + 1:]) == suffix
            ]
            if len(candidates) != 1:
                raise UnsupportedModelError(
                    f"{target} folds or splits the runtime extent; only a movable extent is supported"
                )
            axis = candidates[0]
        elif target == "aten.unsqueeze.default":
            inserted = int(node.args[1]) % len(output_shape)
            if inserted <= axis:
                axis += 1
        elif target in {"aten.squeeze.dim", "aten.squeeze.dims", "aten.squeeze.default"}:
            if target == "aten.squeeze.default":
                candidates = range(len(input_shape))
            elif target == "aten.squeeze.dim":
                candidates = (int(node.args[1]),)
            else:
                candidates = tuple(int(value) for value in node.args[1])
            removed = {
                value % len(input_shape) for value in candidates
                if input_shape[value % len(input_shape)] == 1
            }
            if axis in removed:
                raise UnsupportedModelError("cannot squeeze the runtime extent")
            axis -= sum(value < axis for value in removed)
        elif target in {
            "aten.conv1d.default", "aten.conv2d.default",
            "aten.max_pool1d.default", "aten.max_pool2d.default",
            "aten.avg_pool1d.default", "aten.avg_pool2d.default",
            "aten.adaptive_avg_pool2d.default",
            "aten.batch_norm.default",
        }:
            if axis != 0:
                raise UnsupportedModelError(
                    "convolution and pooling support only a dynamic batch extent"
                )
        elif target in {
            "aten._softmax.default", "aten.softmax.int",
            "aten._log_softmax.default", "aten.log_softmax.int",
        }:
            dimension = int(node.args[1]) % len(input_shape)
            if axis >= dimension or math.prod(input_shape[dimension + 1:]) != 1:
                raise UnsupportedModelError(
                    "dynamic softmax requires the runtime extent before a contiguous reduction"
                )
        elif target == "aten.mean.dim":
            raw_dims = node.args[1]
            dims = {
                int(value) % len(input_shape)
                for value in (raw_dims if isinstance(raw_dims, (list, tuple)) else (raw_dims,))
            }
            if axis in dims or axis > min(dims):
                raise UnsupportedModelError(
                    "mean cannot reduce or move a runtime extent into its reduction geometry"
                )
            axis -= sum(value < axis for value in dims)
        elif target in {"aten.layer_norm.default", "aten.native_layer_norm.default"}:
            normalized_rank = len(node.args[1])
            if axis >= len(input_shape) - normalized_rank:
                raise UnsupportedModelError("layer normalization cannot normalize the runtime extent")
        elif target in MATMUL_OPS:
            if parent is not node.args[0] or axis == len(input_shape) - 1:
                raise UnsupportedModelError("matmul cannot use the runtime extent as a contraction dimension")
        elif target == "aten.slice.Tensor":
            dimension = int(node.args[1]) % len(input_shape)
            start = int(node.args[2]) if len(node.args) > 2 else 0
            end = node.args[3] if len(node.args) > 3 else None
            if dimension == axis and (start != 0 or end not in {None, 9223372036854775807}):
                raise UnsupportedModelError("partial slicing of a runtime extent is not supported")
        elif target not in shape_only and target not in elementwise:
            raise UnsupportedModelError(
                f"{target} does not yet define runtime-extent propagation"
            )
        dynamic_axes[node.name] = axis
        dynamic_bounds[node.name] = dynamic_bounds[parent.name]
    node_order = {node.name: index for index, node in enumerate(nodes)}
    state_inputs = captured.state_inputs
    user_input_placeholders = captured.user_input_names

    def resolve_state(arg_name):
        return state_inputs.get(arg_name)

    if not captured.user_output_names:
        raise UnsupportedModelError(
            "cattorch requires at least one tensor output"
        )
    by_name = {node.name: node for node in nodes}
    outputs = []
    for index, output_name in enumerate(captured.user_output_names):
        output_node = by_name.get(output_name) if output_name is not None else None
        value = None if output_node is None else output_node.meta.get("val")
        if output_node is None or not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise UnsupportedModelError(
                f"model output leaf {index} is not a tensor"
            )
        outputs.append((output_name, _get_shape(output_node), value.dtype))
    output_name, output_shape, output_dtype = outputs[0]

    # Materialise compile-time generated tensors as static weights.
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

    # Evaluate every deterministic, non-mutating operation whose tensor inputs
    # are all static during export. Registered-buffer masks (`bias == 0`),
    # RoPE tables (`torch.outer(arange, inv_freq)`), and similar setup code
    # then never reach Scratch, whatever operations they happen to use.
    if use_fusions:
        # Position and cache buffers change at run time: nothing derived from
        # them is a constant.
        runtime_state = (
            _generation_state_names(nodes, state_inputs) if generation is not None else set()
        )
        # A zero-element tensor's value is known from its shape alone, so
        # it and everything computed only from such tensors (an empty sum,
        # ``mm`` over an empty dimension) fold too.
        for node in nodes:
            value = node.meta.get("val")
            if (
                node.op in {"placeholder", "call_function", "get_attr"}
                and isinstance(value, torch.Tensor) and value.numel() == 0
                and node.name not in generated_weights
            ):
                generated_weights[node.name] = torch.zeros(value.shape, dtype=value.dtype)
        generated_weights.update(fold_static_values(
            nodes,
            {
                name: value for name, value in state_inputs.items()
                if name not in runtime_state
            },
            seed=generated_weights,
        ))

    def resolve_weight(arg_name):
        if arg_name in generated_weights:
            return generated_weights[arg_name]
        return resolve_state(arg_name)

    # Identify model inputs (placeholders not in state_dict)
    input_names = {}
    input_count = 0
    for node in nodes:
        if node.op != 'placeholder':
            continue
        if node.name in user_input_placeholders:
            name = "input" if input_count == 0 else f"input_{input_count}"
            input_names[node.name] = name
            input_count += 1
        elif resolve_weight(node.name) is None:
            raise UnsupportedModelError(
                f"unable to resolve exported parameter or buffer {node.name!r} "
                "from the PyTorch export signature"
            )

    # Build alias map for no-op shape ops and splits
    aliases = {}
    split_meta = {}
    for node in nodes:
        if node.op != 'call_function':
            continue
        aten_op = str(node.target)

        if node.name in generated_weights:
            continue

        if aten_op == "aten.dropout.default":
            training = bool(node.args[2]) if len(node.args) > 2 else True
            if training:
                raise _unsupported_operation(
                    node,
                    "training-mode dropout is nondeterministic and cannot be removed; "
                    "call model.eval() before exporting",
                )
            source = node.args[0].name
            while source in aliases:
                source = aliases[source]
            aliases[node.name] = source
            log.info("Alias: %s -> %s (evaluation dropout)", node.name, source)

        elif aten_op == "aten.to.dtype":
            source_dtype = node.args[0].meta.get("val").dtype
            target_dtype = node.args[1]
            if source_dtype.is_complex or target_dtype.is_complex:
                raise _unsupported_operation(
                    node, "complex tensors are not supported",
                )
            if not _is_value_preserving_cast(source_dtype, target_dtype):
                # Truncation to an integer or conversion to bool runs as an
                # elementwise kernel.
                continue
            source = node.args[0].name
            while source in aliases:
                source = aliases[source]
            aliases[node.name] = source
            log.info("Alias: %s -> %s (dtype conversion)", node.name, source)

        elif aten_op in _NOOP_OPS and hasattr(node.args[0], 'name'):
            source = node.args[0].name
            while source in aliases:
                source = aliases[source]
            aliases[node.name] = source
            log.info("Alias: %s -> %s (no-op %s)", node.name, source, aten_op)

        elif aten_op == "aten.transpose.int" and hasattr(node.args[0], "name"):
            input_shape = _get_shape(node.args[0])
            # A 0-d tensor accepts dimension 0 or -1, like a 1-d one.
            dim0 = node.args[1] % max(len(input_shape), 1)
            dim1 = node.args[2] % max(len(input_shape), 1)
            low, high = sorted((dim0, dim1))
            singleton_move_is_flat = not input_shape or (
                (input_shape[dim0] == 1 or input_shape[dim1] == 1)
                and all(size == 1 for size in input_shape[low + 1:high])
            )
            runtime_axis = dynamic_axes.get(node.args[0].name)
            if runtime_axis in {dim0, dim1} and dim0 != dim1:
                # The representative extent is one only for capture. It may
                # not be erased by the ordinary singleton-transpose alias.
                singleton_move_is_flat = False
            if dim0 == dim1 or singleton_move_is_flat:
                source = node.args[0].name
                while source in aliases:
                    source = aliases[source]
                aliases[node.name] = source
                log.info(
                    "Alias: %s -> %s (singleton transpose %s)",
                    node.name, source, aten_op,
                )

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

    # Reject valid ATen variants whose extra arguments the Scratch kernels do
    # not implement.  Silently discarding these options is much worse than an
    # explicit unsupported-operation error in exact mode.
    for node in nodes:
        if node.op != "call_function" or node.name in generated_weights:
            continue
        aten_op = str(node.target)

        # Matmul legality can depend on a later whole-graph fusion. It is
        # checked by the ordinary compiler path after those fusions are known.
        detail = (
            None if aten_op in {"aten.matmul.default", "aten.mm.default", "aten.bmm.default"}
            else registry.validate(node, fast=fast, config=fast_config)
        )
        if detail is not None:
            raise _unsupported_operation(node, detail)

    linear_fusions = {}
    fused_nodes = set()
    if use_fusions:
        scalar_ops = {
            "aten.mul.Tensor": "mul", "aten.div.Tensor": "div",
            "aten.add.Tensor": "add", "aten.sub.Tensor": "sub",
        }
        activation_ops = {
            "aten.relu.default": "relu", "aten.sigmoid.default": "sigmoid",
            "aten.tanh.default": "tanh", "aten.silu.default": "silu",
            "aten.gelu.default": "gelu",
        }
        for node in nodes:
            if node.op != "call_function" or str(node.target) != "aten.linear.default":
                continue
            current = node
            epilogue = []
            chain = []
            while len(current.users) == 1:
                user = next(iter(current.users))
                if user.name in fused_nodes or user.op != "call_function" or not user.args:
                    break
                target = str(user.target)
                if user.args[0] is current and target in scalar_ops and len(user.args) > 1 and not hasattr(user.args[1], "name"):
                    operand = user.args[1]
                    if not isinstance(operand, (int, float)):
                        break
                    if target == "aten.div.Tensor" and operand == 0:
                        break
                    epilogue.append((scalar_ops[target], operand))
                elif target == "aten.add.Tensor" and len(user.args) > 1:
                    if user.args[0] is current:
                        other = user.args[1]
                    elif user.args[1] is current:
                        other = user.args[0]
                    else:
                        break
                    if (
                        not hasattr(other, "name")
                        or _get_shape(other) != _get_shape(user)
                        or node_order.get(other.name, -1) >= node_order[node.name]
                        # The linear kernel reads one residual list.
                        or any(operation == "tensor_add" for operation, _ in epilogue)
                    ):
                        break
                    epilogue.append(("tensor_add", other))
                elif target in activation_ops:
                    if user.args[0] is not current:
                        break
                    # The fused linear epilogue only implements tanh GELU;
                    # exact GELU runs as its own elementwise kernel.
                    if target == "aten.gelu.default" and _gelu_approximation(user) != "tanh" and not (
                        fast and fast_config.activations
                    ):
                        break
                    # Leave SiLU materialization to the broader SiLU-times-
                    # tensor fusion below when it feeds a single multiply.
                    if target == "aten.silu.default" and len(user.users) == 1:
                        next_user = next(iter(user.users))
                        if str(next_user.target) == "aten.mul.Tensor":
                            break
                    epilogue.append((activation_ops[target], None))
                    chain.append(user)
                    current = user
                    break  # Activations terminate the simple expression epilogue.
                else:
                    break
                chain.append(user)
                current = user
            if chain:
                linear_fusions[node.name] = (current, tuple(epilogue), tuple(chain))
                fused_nodes.update(item.name for item in chain)

    paired_swiglu_fusions = {}
    silu_mul_fusions = {}
    if use_fusions:
        for node in nodes:
            if node.op != "call_function" or str(node.target) != "aten.silu.default":
                continue
            if len(node.users) != 1:
                continue
            multiply = next(iter(node.users))
            if multiply.op != "call_function" or str(multiply.target) != "aten.mul.Tensor":
                continue
            if multiply.args[0] is node:
                other = multiply.args[1]
            elif multiply.args[1] is node:
                other = multiply.args[0]
            else:
                continue
            if not hasattr(other, "name") or _get_shape(other) != _get_shape(multiply):
                continue
            gate = node.args[0]
            if not hasattr(gate, "name") or _get_shape(gate) != _get_shape(multiply):
                continue
            if (
                gate.op == "call_function"
                and str(gate.target) == "aten.linear.default"
                and other.op == "call_function"
                and str(other.target) == "aten.linear.default"
                and gate.args[0] is other.args[0]
                and len(gate.users) == 1
                and len(other.users) == 1
            ):
                paired_swiglu_fusions[multiply.name] = (node, gate, other)
                fused_nodes.update((node.name, gate.name, other.name))
            else:
                silu_mul_fusions[multiply.name] = (node, gate, other)
                fused_nodes.add(node.name)

    embedding_add_fusions = {}
    if use_fusions:
        for node in nodes:
            if (
                node.op != "call_function"
                or str(node.target) != "aten.add.Tensor"
                or len(node.args) < 2
            ):
                continue
            first, second = node.args[:2]
            if not all(
                hasattr(value, "name")
                and value.op == "call_function"
                and str(value.target) == "aten.embedding.default"
                and len(value.users) == 1
                for value in (first, second)
            ):
                continue
            if (
                math.prod(_get_shape(first)) != math.prod(_get_shape(node))
                or math.prod(_get_shape(second)) != math.prod(_get_shape(node))
                or math.prod(_get_shape(first.args[1]))
                != math.prod(_get_shape(second.args[1]))
                or _get_shape(first.args[0])[1:] != _get_shape(second.args[0])[1:]
            ):
                continue
            embedding_add_fusions[node.name] = (first, second)
            fused_nodes.update((first.name, second.name))

    # Collapse straight-line, same-shape arithmetic into one traversal.  This
    # is deliberately conservative: broadcasting still uses the regular
    # elementwise kernel, and a value with another consumer terminates a chain.
    # The fusion is keyed by its final node so branched operands have already
    # been materialized when the compiler emits it.
    elementwise_fusions = {}
    if use_fusions:
        arithmetic_ops = {
            "aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor",
            "aten.div.Tensor", "aten.neg.default",
        }
        claimed = (
            set(fused_nodes)
            | set(linear_fusions)
            | set(paired_swiglu_fusions)
            | set(silu_mul_fusions)
            | set(embedding_add_fusions)
        )
        for root in nodes:
            if (
                root.name in claimed
                or root.op != "call_function"
                or str(root.target) not in arithmetic_ops
                or root.kwargs
            ):
                continue
            shape = _get_shape(root)
            if not shape:
                continue
            chain = []
            current = root
            while (
                current.name not in claimed
                and current.op == "call_function"
                and str(current.target) in arithmetic_ops
                and not current.kwargs
                and _get_shape(current) == shape
            ):
                tensor_args = [arg for arg in current.args if hasattr(arg, "name")]
                scalar_args = [arg for arg in current.args if not hasattr(arg, "name")]
                target = str(current.target)
                expected_args = 1 if target == "aten.neg.default" else 2
                if len(current.args) != expected_args:
                    break
                if any(_get_shape(arg) != shape for arg in tensor_args):
                    break
                if any(not isinstance(arg, (int, float)) for arg in scalar_args):
                    break
                chain.append(current)
                if len(current.users) != 1:
                    break
                user = next(iter(current.users))
                if current not in user.args:
                    break
                current = user

            if len(chain) < 2:
                continue
            final = chain[-1]
            elementwise_fusions[final.name] = tuple(chain)
            claimed.update(item.name for item in chain)
            fused_nodes.update(item.name for item in chain[:-1])

    # Recognize the standard exact causal-attention chain.  The specialized
    # score kernel consumes K before its final transpose, folds the scalar
    # scale and -inf mask, and exposes enough metadata for softmax and A@V to
    # skip the known upper triangle as well.
    causal_score_fusions = {}
    causal_softmaxes = {}
    causal_value_matmuls = set()
    qkv_linear_fusions = {}
    qkv_offsets = {}
    if use_fusions:
        def static_tensor(candidate):
            if not hasattr(candidate, "name"):
                return None
            name = candidate.name
            while name in aliases:
                name = aliases[name]
            value = resolve_weight(name)
            return value if isinstance(value, torch.Tensor) else None

        for mask_node in nodes:
            if (
                mask_node.op != "call_function"
                or str(mask_node.target) != "aten.masked_fill.Scalar"
                or len(mask_node.args) < 3
                or mask_node.args[2] != float("-inf")
            ):
                continue
            score_value = mask_node.args[0]
            mask = static_tensor(mask_node.args[1])
            output_shape = _get_shape(mask_node)
            if (
                mask is None or mask.dtype != torch.bool or mask.ndim != 2
                or len(output_shape) < 3
                or output_shape[-1] != output_shape[-2]
                or tuple(mask.shape) != tuple(output_shape[-2:])
            ):
                continue
            length = output_shape[-1]
            expected_mask = torch.triu(
                torch.ones((length, length), dtype=torch.bool, device=mask.device),
                diagonal=1,
            )
            if not torch.equal(mask, expected_mask):
                continue

            scale = 1.0
            scale_node = None
            if hasattr(score_value, "name") and score_value.op == "call_function":
                target = str(score_value.target)
                if (
                    target in {"aten.mul.Tensor", "aten.div.Tensor"}
                    and len(score_value.args) > 1
                    and not hasattr(score_value.args[1], "name")
                ):
                    operand = score_value.args[1]
                    if not isinstance(operand, (int, float)) or operand == 0:
                        continue
                    scale = float(operand) if target == "aten.mul.Tensor" else 1.0 / float(operand)
                    scale_node = score_value
                    score_value = score_value.args[0]
            if (
                not hasattr(score_value, "name")
                or score_value.op != "call_function"
                or str(score_value.target) not in {
                    "aten.matmul.default", "aten.mm.default", "aten.bmm.default",
                }
                or len(score_value.users) != 1
            ):
                continue
            qk = score_value
            q, transposed_k = qk.args[:2]
            if (
                not hasattr(q, "name") or not hasattr(transposed_k, "name")
                or transposed_k.op != "call_function"
                or str(transposed_k.target) != "aten.transpose.int"
                or len(transposed_k.users) != 1
            ):
                continue
            k = transposed_k.args[0]
            k_shape = _get_shape(k)
            q_shape = _get_shape(q)
            if (
                len(q_shape) < 3 or len(k_shape) != len(q_shape)
                or q_shape[:-3] != k_shape[:-3]
                or q_shape[-1] != k_shape[-1]
                or q_shape[-2] != length or k_shape[-2] != length
                or k_shape[-3] < 1 or q_shape[-3] % k_shape[-3] != 0
                or _get_shape(qk)[-2:] != torch.Size((length, length))
            ):
                continue
            ndim = len(k_shape)
            dim0 = transposed_k.args[1] % ndim
            dim1 = transposed_k.args[2] % ndim
            if {dim0, dim1} != {ndim - 2, ndim - 1}:
                continue
            if scale_node is not None and len(scale_node.users) != 1:
                continue

            causal_score_fusions[qk.name] = (
                mask_node, q, k, scale, transposed_k, scale_node,
            )
            fused_nodes.add(mask_node.name)
            fused_nodes.add(transposed_k.name)
            if scale_node is not None:
                fused_nodes.add(scale_node.name)

            for softmax in mask_node.users:
                if (
                    softmax.op == "call_function"
                    and str(softmax.target) == "aten.softmax.int"
                    and softmax.args[0] is mask_node
                    and softmax.args[1] in {-1, len(output_shape) - 1}
                ):
                    causal_softmaxes[softmax.name] = length

        for node in nodes:
            if (
                node.op == "call_function"
                and str(node.target) in {
                    "aten.matmul.default", "aten.mm.default", "aten.bmm.default",
                }
                and hasattr(node.args[0], "name")
                and node.args[0].name in causal_softmaxes
            ):
                left_shape = _get_shape(node.args[0])
                right_shape = _get_shape(node.args[1])
                if (
                    len(left_shape) >= 3 and len(right_shape) == len(left_shape)
                    and left_shape[:-3] == right_shape[:-3]
                    and right_shape[-3] >= 1
                    and left_shape[-3] % right_shape[-3] == 0
                    and left_shape[-1] == right_shape[-2]
                ):
                    causal_value_matmuls.add(node.name)

        # Collapse the canonical combined-QKV split/view/transposes into one
        # projection whose output is [Q head-major, K head-major, V head-major].
        # The causal kernels use offsets into that shared list.
        by_name = {node.name: node for node in nodes}

        def alias_source(candidate):
            while hasattr(candidate, "name") and candidate.name in aliases:
                candidate = by_name.get(aliases[candidate.name], candidate)
                if candidate.name not in aliases:
                    break
            return candidate

        for qk_name, fusion in causal_score_fusions.items():
            mask_node, q, k, _, _, _ = fusion
            # Reordered QKV emission adds outer-loop dispatch.  Official-VM
            # A/B runs show the removed copies pay for that only once the
            # sequence is moderately long.
            if _get_shape(q)[-2] < 16:
                continue
            q_item = alias_source(q.args[0]) if str(q.target) == "aten.transpose.int" else None
            k_item = alias_source(k.args[0]) if str(k.target) == "aten.transpose.int" else None
            if (
                q_item is None or k_item is None
                or q_item.target is not operator.getitem
                or k_item.target is not operator.getitem
                or q_item.args[1] != 0 or k_item.args[1] != 1
                or q_item.args[0] is not k_item.args[0]
            ):
                continue
            split = q_item.args[0]
            softmax = next(
                (candidate for candidate in mask_node.users if candidate.name in causal_softmaxes),
                None,
            )
            value_matmul = next(
                (
                    candidate for candidate in nodes
                    if candidate.name in causal_value_matmuls
                    and softmax is not None and candidate.args[0] is softmax
                ),
                None,
            )
            if value_matmul is None:
                continue
            v = value_matmul.args[1]
            if not hasattr(v, "name") or str(v.target) != "aten.transpose.int":
                continue
            v_item = alias_source(v.args[0])
            if (
                v_item.target is not operator.getitem
                or v_item.args[1] != 2
                or v_item.args[0] is not split
            ):
                continue
            linear_name = aliases.get(split.name)
            linear = by_name.get(linear_name)
            q_shape = _get_shape(q)
            k_shape = _get_shape(k)
            v_shape = _get_shape(v)
            q_embed = q_shape[-3] * q_shape[-1]
            k_embed = k_shape[-3] * k_shape[-1]
            v_embed = v_shape[-3] * v_shape[-1]
            if (
                linear is None or str(linear.target) != "aten.linear.default"
                or len(linear.users) != 1
                or q_shape[-1] != k_shape[-1] or q_shape[-1] != v_shape[-1]
                or k_shape[-3] != v_shape[-3]
                or _get_shape(linear)[-1] != q_embed + k_embed + v_embed
            ):
                continue
            q_heads = q_shape[-3]
            kv_heads = k_shape[-3]
            q_size = math.prod(q_shape)
            k_size = math.prod(k_shape)
            qkv_linear_fusions[linear.name] = (
                q_heads, kv_heads, q, k, v, q_item, k_item, v_item,
            )
            qkv_offsets.update({q.name: 0, k.name: q_size, v.name: q_size + k_size})
            aliases[q.name] = linear.name
            aliases[k.name] = linear.name
            aliases[v.name] = linear.name
            fused_nodes.update((q_item.name, k_item.name, v_item.name))

    generation_score_caches = {}
    generation_value_caches = {}
    generation_cache_widths = {}
    generation_first_cache = None
    generation_hidden_prefill_attention = None
    if generation is not None:
        if len(example_inputs) != 1 or tuple(example_inputs[0].shape) != (1, 1):
            raise ValueError("generation export example input must have shape [1, 1]")
        for node in nodes:
            if node.op != "call_function" or str(node.target) != _CACHED_ATTENTION:
                continue
            layer = len(generation_score_caches) + 1
            k_cache = f"cattorch K cache {layer}"
            v_cache = f"cattorch V cache {layer}"
            q_shape, k_shape, v_shape = (_get_shape(node.args[i]) for i in range(3))
            kv_heads = 1 if len(q_shape) == 3 else k_shape[-3]
            generation_score_caches[node.name] = k_cache
            generation_value_caches[node.name] = v_cache
            generation_cache_widths[k_cache] = kv_heads * k_shape[-1]
            generation_cache_widths[v_cache] = kv_heads * v_shape[-1]
            if generation_first_cache is None:
                generation_first_cache = (k_cache, generation_cache_widths[k_cache])
        if not generation_score_caches:
            raise UnsupportedModelError(
                "automatic generation found no causal attention block to cache"
            )
        if generation.hidden_prefill:
            generation_hidden_prefill_attention = next(reversed(generation_score_caches))

    graph = _GraphInfo(
        nodes=nodes,
        resolve_weight=resolve_weight,
        input_names=input_names,
        outputs=tuple(outputs),
        aliases=aliases,
        split_meta=split_meta,
        generated_weights=generated_weights,
        fast=fast,
        fast_config=fast_config,
        generation=_GenerationInfo(
            score_caches=generation_score_caches,
            value_caches=generation_value_caches,
            cache_widths=generation_cache_widths,
            first_cache=generation_first_cache,
            hidden_prefill_attention=generation_hidden_prefill_attention,
        ),
        linear_fusions=linear_fusions,
        paired_swiglu_fusions=paired_swiglu_fusions,
        silu_mul_fusions=silu_mul_fusions,
        embedding_add_fusions=embedding_add_fusions,
        elementwise_fusions=elementwise_fusions,
        qkv_linear_fusions=qkv_linear_fusions,
        qkv_offsets=qkv_offsets,
        causal_score_fusions=causal_score_fusions,
        causal_softmaxes=causal_softmaxes,
        causal_value_matmuls=causal_value_matmuls,
        fused_nodes=fused_nodes,
        registry=registry,
        dynamic_axes=dynamic_axes,
        dynamic_bounds=dynamic_bounds,
    )
    _validate_lowering_operations(graph)
    return graph


def _validate_lowering_operations(graph: _GraphInfo) -> None:
    """Check unfused operation contracts before either analysis or emission."""
    skipped = set(graph.aliases) | set(graph.generated_weights) | graph.fused_nodes
    for roots in (
        graph.causal_score_fusions, graph.qkv_linear_fusions,
        graph.embedding_add_fusions,
        graph.elementwise_fusions, graph.linear_fusions,
        graph.paired_swiglu_fusions, graph.silu_mul_fusions,
    ):
        skipped.update(roots)
    for node in graph.nodes:
        if node.op != "call_function" or node.name in skipped:
            continue
        target = str(node.target)
        if target in {"aten.cat.default", "aten.slice.Tensor"}:
            continue
        if node.target is operator.getitem and getattr(node.args[0], "name", None) in graph.split_meta:
            continue
        if graph.registry.operation(target) is None:
            raise _unsupported_operation(node)
        if node.name not in graph.causal_value_matmuls:
            detail = graph.registry.validate(node, fast=graph.fast, config=graph.fast_config)
            if detail is not None:
                raise _unsupported_operation(node, detail)
