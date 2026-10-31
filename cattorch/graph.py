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
from cattorch.interface import input_name, output_name

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
    # Operations a fusion approximates in fast mode, such as exact GELU in a
    # linear epilogue.
    approximate_fusions: tuple[str, ...] = ()
    # Nodes whose one-row kernel runs once per row of a runtime extent.
    row_loops: frozenset[str] = frozenset()
    # Fixed items along a node's runtime axis beyond the runtime length.
    dynamic_offsets: dict[str, int] = field(default_factory=dict)

    def runtime_capacity(self, name: str, shape) -> int:
        """The most items ``name``'s list can hold."""
        axis = self.dynamic_axes[name]
        offset = self.dynamic_offsets.get(name, 0)
        width = math.prod(extent for index, extent in enumerate(shape) if index != axis)
        return width * ((shape[axis] - offset) * self.dynamic_bounds[name][2] + offset)

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

    def _public_bounds(self, name: str) -> tuple[str, int, int]:
        origin, minimum, maximum = self.dynamic_bounds[name]
        offset = self.dynamic_offsets.get(name, 0)
        return origin, max(minimum + offset, 0), maximum + offset

    def output_specs(self):
        return tuple(
            _tensor_spec(
                output_name(index), shape, dtype,
                (self.dynamic_axes[name], self._public_bounds(name))
                if name in self.dynamic_axes else None,
            )
            for index, (name, shape, dtype) in enumerate(self.outputs)
        )


def _row_separable_axis(node, dynamic_axes, dynamic_units) -> tuple[int, int] | None:
    """``(output axis, items per unit)`` of a unit-by-unit operation, or ``None``.

    Each runtime-length input holds its units contiguously: only size-1
    dimensions come before its axis. The operation runs on random inputs with
    two units and with each unit alone. It treats units independently if the
    two-unit output is the one-unit outputs joined along an axis that leads
    the flat layout.
    """
    if not callable(node.target):
        return None
    for arg in node.all_input_nodes:
        if arg.name in dynamic_axes:
            if math.prod(_get_shape(arg)[:dynamic_axes[arg.name]]) != 1:
                return None
    found = _separable_with(node, dynamic_axes, dynamic_units, varied=False)
    if found is None or not any(
        arg.name in dynamic_axes and (dtype := _get_dtype(arg)) is not None
        and not dtype.is_floating_point and dtype != torch.bool
        for arg in node.all_input_nodes
    ):
        return found
    # Zeros are safe for any index, but every unit of them looks alike, so a
    # running sum over integer units would pass. Small varied integers show
    # it, when the operation accepts them.
    varied = _separable_with(node, dynamic_axes, dynamic_units, varied=True, strict=True)
    return found if varied in {found, "failed"} else None


def _separable_with(node, dynamic_axes, dynamic_units, *, varied, strict=False):
    """The separable axis for one choice of integer samples.

    ``varied`` fills runtime-length integer inputs with zeros and ones
    instead of zeros. With ``strict``, an operation that fails on every axis
    returns ``"failed"`` rather than ``None``.
    """
    import torch.utils._pytree as pytree

    generator = torch.Generator().manual_seed(0)

    def unit_of(name):
        return dynamic_units.get(name, 1)

    def sample(arg):
        value = arg.meta.get("val") if hasattr(arg, "meta") else None
        if not isinstance(value, torch.Tensor):
            return None
        shape = list(value.shape)
        if arg.name in dynamic_axes:
            shape[dynamic_axes[arg.name]] = 2 * unit_of(arg.name)
        if value.dtype == torch.bool:
            return torch.rand(shape, generator=generator) > 0.5
        if not value.dtype.is_floating_point:
            if varied and arg.name in dynamic_axes:
                return torch.randint(0, 2, shape, generator=generator, dtype=value.dtype)
            return torch.zeros(shape, dtype=value.dtype)
        return torch.randn(shape, generator=generator, dtype=value.dtype)

    output_shape = list(_get_shape(node))
    samples = {}
    for arg in node.all_input_nodes:
        value = sample(arg)
        if value is None:
            return None
        samples[arg.name] = value

    def call(pick=None, sizes=None):
        def convert(value):
            if not isinstance(value, torch.fx.Node):
                return value
            tensor = samples[value.name]
            if pick is not None and value.name in dynamic_axes:
                unit = unit_of(value.name)
                tensor = tensor.narrow(dynamic_axes[value.name], pick * unit, unit)
            return tensor

        def size_argument(value):
            # A size for the whole output, such as expand's, grows with it.
            if sizes is not None and isinstance(value, (list, tuple)) and list(value) == output_shape:
                return type(value)(sizes)
            return pytree.tree_map(convert, value)

        args = tuple(size_argument(arg) for arg in node.args)
        kwargs = pytree.tree_map(convert, node.kwargs)
        return node.target(*args, **kwargs)

    ran = False
    for axis in range(len(output_shape)):
        if math.prod(output_shape[:axis]) != 1:
            break
        sizes = list(output_shape)
        sizes[axis] *= 2
        try:
            with torch.no_grad():
                both, again = call(sizes=sizes), call(sizes=sizes)
                first, second = call(0), call(1)
        except Exception:  # noqa: BLE001 - an operation that can't run per unit isn't separable
            continue
        ran = True
        if not all(isinstance(value, torch.Tensor) for value in (both, first, second, again)):
            continue
        if not torch.equal(both.nan_to_num(), again.nan_to_num()) or first.shape != second.shape:
            continue
        if (
            both.ndim > axis and list(first.shape) == output_shape
            and both.shape[axis] == 2 * first.shape[axis]
            and torch.allclose(both, torch.cat([first, second], axis), equal_nan=True)
        ):
            return axis, first.shape[axis]
    return "failed" if strict and not ran else None


def _broadcasts_along(value, axis, output_shape) -> bool:
    """Whether a fixed operand repeats along a runtime axis of the output.

    Elementwise kernels index such an operand from the flat output position,
    which is only right when it is one value or matches dimensions after the
    axis.
    """
    shape = tuple(_get_shape(value))
    if math.prod(shape) == 1:
        return False
    trimmed = shape
    while trimmed and trimmed[0] == 1:
        trimmed = trimmed[1:]
    suffix = len(output_shape) - axis - 1
    return not (
        shape != tuple(output_shape) and len(trimmed) <= suffix
        and trimmed == tuple(output_shape[len(output_shape) - len(trimmed):])
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


def _broadcast_matmul_batches(graph) -> None:
    """Give both operands of a broadcasting ``matmul`` the same batch dimensions.

    The matmul kernel needs identical batches. A vector becomes a one-row or
    one-column matrix, both operands are expanded to the broadcast batch
    shape, and the result is reshaped back to what ``matmul`` returns.
    """
    for node in list(graph.nodes):
        if node.op != "call_function" or str(node.target) != "aten.matmul.default":
            continue
        left, right = node.args[0], node.args[1]
        if not (isinstance(left, torch.fx.Node) and isinstance(right, torch.fx.Node)):
            continue
        left_shape, right_shape = tuple(_get_shape(left)), tuple(_get_shape(right))
        if len(right_shape) <= 2 and len(left_shape) >= 1:
            continue  # The kernel handles any left batch over a matrix.
        if len(left_shape) >= 2 and left_shape[:-2] == right_shape[:-2]:
            continue
        dtype = node.meta["val"].dtype
        operands = []
        matrices = (
            left_shape if len(left_shape) >= 2 else (1, *left_shape),
            right_shape if len(right_shape) >= 2 else (*right_shape, 1),
        )
        batch = tuple(torch.broadcast_shapes(matrices[0][:-2], matrices[1][:-2]))
        for operand, shape, matrix in zip((left, right), (left_shape, right_shape), matrices):
            if shape != matrix:
                operand = _insert(
                    graph, node, torch.ops.aten.reshape.default, (operand, list(matrix)),
                    matrix, _get_dtype(operand),
                )
            full = (*batch, *matrix[-2:])
            if matrix != full:
                operand = _insert(
                    graph, node, torch.ops.aten.expand.default, (operand, list(full)),
                    full, _get_dtype(operand),
                )
            operands.append(operand)
        product_shape = (*batch, matrices[0][-2], matrices[1][-1])
        product = _insert(
            graph, node, torch.ops.aten.matmul.default, tuple(operands), product_shape, dtype,
        )
        node.target = torch.ops.aten.reshape.default
        node.args, node.kwargs = (product, list(_get_shape(node))), {}


def _fuse_rotate_half(graph) -> None:
    """Rewrite Hugging Face RoPE, ``x * cos + rotate_half(x) * sin``, to one kernel.

    ``rotate_half(x)`` traces as ``cat([-x[..., h:], x[..., :h]], -1)``. The
    match needs ``cos`` and ``sin`` to broadcast to ``x`` without enlarging it.
    Anything else is left to the general kernels, which give the same values.
    """
    import cattorch.ops  # noqa: F401 - registers cattorch.rotary_half

    def call(value, *targets):
        return (
            isinstance(value, torch.fx.Node) and value.op == "call_function"
            and str(value.target) in targets and not value.kwargs
        )

    def last_dim(value, dim) -> bool:
        rank = len(_get_shape(value))
        return isinstance(dim, int) and dim % rank == rank - 1

    def half_slice(value, source, start, end) -> bool:
        if not call(value, "aten.slice.Tensor") or value.args[0] is not source:
            return False
        defaults = [0, None, None, 1]
        dim, low, high, step = [*value.args[1:], *defaults[len(value.args) - 1:]]
        width = _get_shape(source)[-1]
        low = low or 0
        high = width if high is None or high >= width else high
        return last_dim(source, dim) and low == start and high == end and step == 1

    def rotated(value):
        """``x`` if ``value`` is ``rotate_half(x)``, else ``None``."""
        if not call(value, "aten.cat.default") or len(value.args[0]) != 2:
            return None
        if len(value.args) > 1 and not last_dim(value, value.args[1]):
            return None
        if len(value.args) == 1 and len(_get_shape(value)) != 1:
            return None
        negated, first = value.args[0]
        if not call(negated, "aten.neg.default"):
            return None
        second = negated.args[0]
        if not call(second, "aten.slice.Tensor"):
            return None
        source = second.args[0]
        width = _get_shape(source)[-1]
        half = width // 2
        if width % 2 or not (
            half_slice(second, source, half, width) and half_slice(first, source, 0, half)
        ):
            return None
        return source

    def product(value):
        """``(tensor, factor)`` pairs of a ``mul`` node, both ways round."""
        if not call(value, "aten.mul.Tensor"):
            return []
        left, right = value.args
        return [(left, right), (right, left)]

    for node in list(graph.nodes):
        if not call(node, "aten.add.Tensor") or len(node.args) != 2:
            continue
        match = None
        for plain, turned in (node.args, node.args[::-1]):
            for value, cosine in product(plain):
                for rotation, sine in product(turned):
                    source = rotated(rotation)
                    if source is not None and source is value:
                        match = value, cosine, sine
        if match is None:
            continue
        value, cosine, sine = match
        if not all(isinstance(table, torch.fx.Node) for table in (cosine, sine)):
            continue
        shape = tuple(_get_shape(value))
        try:
            broadcast = torch.broadcast_shapes(shape, _get_shape(cosine), _get_shape(sine))
        except RuntimeError:
            continue
        dtypes = {_get_dtype(item) for item in (value, cosine, sine, node)}
        if tuple(broadcast) != shape or tuple(_get_shape(node)) != shape or len(dtypes) != 1:
            continue
        replaced = list(node.all_input_nodes)
        node.target = torch.ops.cattorch.rotary_half.default
        node.args = (value, cosine, sine)
        while replaced:
            unused = replaced.pop()
            if unused.graph is graph and unused.op == "call_function" and not unused.users:
                replaced.extend(unused.all_input_nodes)
                graph.erase_node(unused)


def _user_kernel(registry, target: str, fast: bool, config) -> bool:
    """Whether ``target`` lowers through a kernel passed to this export."""
    from cattorch.builtin_kernels import BUILTIN_KERNELS

    kernel = registry.kernel(target, fast=fast, config=config)
    context = getattr(kernel, "kernel", None)
    return context is not None and not any(context is builtin for builtin in BUILTIN_KERNELS)


def _drop_identity_permutes(graph) -> None:
    """Remove ``permute`` calls that keep every dimension in place.

    Decompositions such as ``x[index]`` emit them. Lowered, each would copy
    its whole input into a new list.
    """
    for node in list(graph.nodes):
        if node.op != "call_function" or str(node.target) != "aten.permute.default":
            continue
        if any(user.op == "output" for user in node.users):
            continue  # outputs are recorded by node name
        dims = [int(dim) % max(len(_get_shape(node)), 1) for dim in node.args[1]]
        if dims == list(range(len(dims))):
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)


def _normalize_graph(graph) -> None:
    """Rewrites every captured graph gets before analysis.

    The traces at other lengths get them too, so their nodes line up.
    """
    _drop_empty_cat_inputs(graph)
    _drop_identity_permutes(graph)
    _broadcast_matmul_batches(graph)
    _fuse_rotate_half(graph)


_RUNTIME_END = 2**63 - 1


def _length_literals(
    model, example_inputs, example_kwargs, dynamic_inputs, captured, frontend, items=2,
):
    """Literal arguments that change when each dynamic input has ``items`` items.

    The graph is traced with one item, where ``x[-1]`` and ``x[0]`` are the
    same slice. A second trace with two items tells them apart. Returns
    ``{node name: {location: value}}``, with the value at ``items`` items, or
    ``None`` if the trace fails or has another structure.
    """
    from cattorch.dynamic import at_length
    from cattorch.frontend import flat_example_inputs

    if isinstance(example_inputs, torch.Tensor):
        example_inputs = (example_inputs,)
    example_inputs = tuple(example_inputs)
    values = list(flat_example_inputs(example_inputs, example_kwargs))
    for index, bound in dynamic_inputs.items():
        axis = bound[3] if len(bound) > 3 else 0
        if index >= len(values) or values[index].shape[axis] != 1:
            return None
        values[index] = at_length(values[index], axis, items)
    keywords = dict(example_kwargs or {})
    names = [key for key, value in keywords.items() if isinstance(value, torch.Tensor)]
    keywords.update(zip(names, values[len(example_inputs):]))
    try:
        longer = capture_model(
            model, tuple(values[:len(example_inputs)]), frontend=frontend,
            example_kwargs=keywords or None,
        )
    except Exception:  # noqa: BLE001 - without a second trace, literals stay as traced
        return None
    _normalize_graph(longer.graph)
    first, second = list(captured.graph.nodes), list(longer.graph.nodes)
    if len(first) != len(second) or any(
        (a.op, str(a.target)) != (b.op, str(b.target)) for a, b in zip(first, second)
    ):
        return None
    result: dict[str, dict[tuple, object]] = {}

    def walk(node, a, b, location) -> bool:
        if isinstance(a, torch.fx.Node) or isinstance(b, torch.fx.Node):
            return isinstance(a, torch.fx.Node) and isinstance(b, torch.fx.Node)
        if isinstance(a, (list, tuple)):
            return (
                isinstance(b, (list, tuple)) and len(a) == len(b)
                and all(walk(node, x, y, (*location, i)) for i, (x, y) in enumerate(zip(a, b)))
            )
        if isinstance(a, (int, float)) and not isinstance(a, bool) and a != b:
            result.setdefault(node.name, {})[location] = b
        return True

    for a, b in zip(first, second):
        if a.op == "call_function" and not (
            walk(a, a.args, b.args, ("args",))
            and walk(a, dict(a.kwargs), dict(b.kwargs), ("kwargs",))
        ):
            return None
    return result


def _reduction(node) -> tuple[str, tuple[int, ...] | None, bool] | None:
    """``(kind, dimensions or None for all, keepdim)`` of a reduction node."""
    target = str(node.target)
    kinds = {
        "aten.sum.dim_IntList": "sum", "aten.sum.default": "sum",
        "aten.mean.dim": "mean", "aten.mean.default": "mean",
        "aten.amax.default": "amax", "aten.amin.default": "amin",
        "aten.prod.dim_int": "prod", "aten.prod.default": "prod",
    }
    if target not in kinds:
        return None
    if target.endswith(".default") and target not in {"aten.amax.default", "aten.amin.default"}:
        return kinds[target], None, False
    raw = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    keepdim = bool(node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False))
    if raw is None or (isinstance(raw, (list, tuple)) and not raw):
        return kinds[target], None, keepdim
    dims = tuple(raw) if isinstance(raw, (list, tuple)) else (raw,)
    return kinds[target], tuple(int(value) for value in dims), keepdim


def _insert(graph, before, target, args, shape, dtype):
    with graph.inserting_before(before):
        node = graph.call_function(target, args)
    node.meta["val"] = torch.empty(tuple(shape), dtype=dtype)
    return node


_REDUCE_ONE = {
    "sum": (torch.ops.aten.sum.dim_IntList, True),
    "mean": (torch.ops.aten.mean.dim, True),
    "amax": (torch.ops.aten.amax.default, True),
    "amin": (torch.ops.aten.amin.default, True),
    "prod": (torch.ops.aten.prod.dim_int, False),
}
_SOFTMAXES = {
    "aten._softmax.default": False, "aten.softmax.int": False,
    "aten._log_softmax.default": True, "aten.log_softmax.int": True,
}


def _mix_runtime_axis(
    node, target, parent, axis, offset, input_shape, output_shape, *,
    graph, runtime, dynamic_axes, dynamic_units, dynamic_offsets, literals, shortest, mark,
) -> bool:
    """Rewrite an operation that mixes values along the runtime axis.

    Reductions, softmax, matmul contraction, ``cat`` and slices along the
    axis become ``cattorch::runtime_*`` operations, whose kernels read the
    extent from the list's length. Returns whether ``node`` was handled.
    """
    dtype = node.meta["val"].dtype
    reshape = torch.ops.aten.reshape.default
    reduction = _reduction(node)
    if reduction is not None:
        kind, dims, keepdim = reduction
        rank = max(len(input_shape), 1)
        dims = tuple(range(len(input_shape))) if dims is None else tuple(
            sorted({value % rank for value in dims})
        )
        if axis not in dims:
            return False
        kept = list(input_shape)
        kept[axis] = 1
        last = _insert(graph, node, runtime.runtime_reduce.default, (parent, axis, kind), kept, dtype)
        # Other dimensions are static after the runtime one, one at a time.
        operation, listed = _REDUCE_ONE[kind]
        for dim in dims:
            if dim != axis:
                kept[dim] = 1
                last = _insert(graph, node, operation, (last, [dim] if listed else dim, True), kept, dtype)
        node.target, node.args, node.kwargs = reshape, (last, list(output_shape)), {}
        return True
    if target in _SOFTMAXES:
        if int(node.args[1]) % len(input_shape) != axis:
            return False
        node.target = runtime.runtime_softmax.default
        node.args, node.kwargs = (parent, axis, _SOFTMAXES[target]), {}
        mark(node.name, axis, parent.name, offset)
        return True
    if target in MATMUL_OPS:
        left, right = node.args[0], node.args[1]
        left_shape, right_shape = _get_shape(left), _get_shape(right)
        left_axis = dynamic_axes.get(left.name)
        right_axis = dynamic_axes.get(right.name)
        contraction = (
            left_axis == len(left_shape) - 1 and right_axis == max(len(right_shape) - 2, 0)
        )
        columns = left_axis is None and len(right_shape) >= 2 and right_axis == len(right_shape) - 1
        if not (contraction or columns):
            return False
        if any(dynamic_units.get(value.name, 1) != 1 for value in (left, right)):
            raise UnsupportedModelError(
                "matmul over a runtime extent folded into another dimension isn't supported"
            )
        batches = [tuple(shape[:-2]) for shape in (left_shape, right_shape) if len(shape) > 2]
        if len(batches) == 2 and batches[0] != batches[1]:
            raise UnsupportedModelError(
                "a matmul over a runtime extent needs operands with the same batch dimensions"
            )
        node.target, node.args, node.kwargs = runtime.runtime_matmul.default, (left, right), {}
        if columns:
            mark(node.name, len(output_shape) - 1, right.name, dynamic_offsets.get(right.name, 0))
        return True
    if target == "aten.cat.default":
        pieces = list(node.args[0])
        dim = int(node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0))
        dim %= len(output_shape)
        if dim != axis or len(pieces) < 2:
            return False
        moving = [piece for piece in pieces if piece.name in dynamic_axes]
        if len(moving) != 1 or dynamic_units.get(moving[0].name, 1) != 1:
            raise UnsupportedModelError(
                "cat along the runtime extent can join one variable-length tensor "
                "with fixed-size ones, not several variable-length tensors"
            )
        (source,) = moving

        def extent(piece):
            return dynamic_offsets.get(source.name, 0) if piece is source else _get_shape(piece)[dim]

        # Joined in pairs; ``extra`` counts the fixed items joined so far,
        # which become the offset once the variable-length piece is in.
        joined, holds, extra = pieces[0], pieces[0] is source, extent(pieces[0])
        for position in range(1, len(pieces)):
            piece = pieces[position]
            holds = holds or piece is source
            extra += extent(piece)
            if position == len(pieces) - 1:
                node.target = runtime.runtime_cat.default
                node.args, node.kwargs = (joined, piece, dim), {}
                mark(node.name, dim, source.name, extra)
                break
            shape = list(_get_shape(joined))
            shape[dim] += _get_shape(piece)[dim]
            joined = _insert(graph, node, runtime.runtime_cat.default, (joined, piece, dim), shape, dtype)
            if holds:
                mark(joined.name, dim, source.name, extra)
        return True
    if target in {"aten.slice.Tensor", "aten.select.int"}:
        if int(node.args[1]) % len(input_shape) != axis:
            return False
        traced = input_shape[axis]

        def place(value, location):
            """``(counts from the end, position)`` for a traced bound."""
            if value is None or value >= _RUNTIME_END:
                return True, 0
            if value < 0:
                return True, value
            if location in literals and literals[location] != value:
                return True, value - traced
            return False, value

        if target == "aten.select.int":
            first_from_end, first = place(int(node.args[2]), ("args", 2))
            last_from_end, last = first_from_end, first + 1
        else:
            start = node.args[2] if len(node.args) > 2 else 0
            end = node.args[3] if len(node.args) > 3 else None
            if start in {0, None} and (end is None or end >= _RUNTIME_END):
                return False
            if (node.args[4] if len(node.args) > 4 else 1) != 1:
                raise UnsupportedModelError(
                    "slicing a runtime extent with a step isn't supported"
                )
            first_from_end, first = place(start or 0, ("args", 2))
            last_from_end, last = place(end, ("args", 3))
        if first_from_end and not last_from_end:
            raise UnsupportedModelError(
                "a slice from the end of a runtime extent to a fixed position isn't supported"
            )
        fewest = shortest(parent.name)
        end = last
        if first_from_end == last_from_end:
            size = end - first
            reach = -first if first_from_end else last
            if reach > traced:
                raise UnsupportedModelError(
                    f"a fixed slice of a runtime extent reaches {reach} items, but the "
                    f"export traces it with {traced}"
                )
            if size <= 0 or reach > fewest:
                raise UnsupportedModelError(
                    f"a fixed slice of a runtime extent reaches {reach} items, but it can "
                    f"hold as few as {fewest}"
                )
            extent = None
        else:
            extent = traced - first + end
            if extent <= 0 or fewest - first + end < 0:
                raise UnsupportedModelError(
                    "slicing the runtime extent leaves no items in the one-item trace; "
                    "it can't follow the length"
                )
        expected = size if extent is None else extent
        sliced_shape = list(input_shape)
        sliced_shape[axis] = expected
        if target == "aten.slice.Tensor" and tuple(sliced_shape) != output_shape:
            raise UnsupportedModelError(
                "a slice of the runtime extent doesn't match its traced shape"
            )
        encoded_end = _RUNTIME_END if last_from_end and last == 0 else last
        args = (parent, axis, first, encoded_end)
        if target == "aten.select.int":
            sliced = _insert(graph, node, runtime.runtime_slice.default, args, sliced_shape, dtype)
            node.target, node.args, node.kwargs = reshape, (sliced, list(output_shape)), {}
        else:
            node.target, node.args, node.kwargs = runtime.runtime_slice.default, args, {}
            if extent is not None:
                mark(node.name, axis, parent.name, offset - first + end)
        return True
    return False


_ARITHMETIC = {"aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor", "aten.div.Tensor"}


def _follow_length_literals(node, at_two, at_three, graph, runtime, source, axis) -> None:
    """Replace arithmetic constants computed from the length by its runtime value.

    A constant that changes between the traces with one, two and three items
    is rebuilt from ``n``, read from the list's length, when it is ``a + b n``
    (the ``n - 1`` in a variance, ``x.shape[0]`` in a hand-written mean) or
    ``a + b / n`` (``1 / x.shape[0]``). Without a third trace it is taken to
    be ``a + b n``.
    """
    for position in (0, 1):
        location = ("args", position)
        one = node.args[position] if position < len(node.args) else None
        if location not in at_two or isinstance(one, (bool, torch.fx.Node)):
            continue
        two, three = at_two[location], at_three.get(location)
        slope = two - one
        # ``a + b / n`` through one and two items.
        scale = 2 * (one - two)
        if three is None or math.isclose(three, one + 2 * slope, rel_tol=1e-9, abs_tol=1e-12):
            reciprocal = False
        elif math.isclose(three, one - scale + scale / 3, rel_tol=1e-9, abs_tol=1e-12):
            reciprocal = True
        else:
            raise UnsupportedModelError(
                f"{node.target} uses a value computed from the variable length that "
                "cattorch can't follow; it follows a + b * n and a + b / n"
            )
        count = _insert(
            graph, node, runtime.runtime_length.default, (source, axis), (), torch.float32,
        )
        if reciprocal:
            count = _insert(
                graph, node, torch.ops.aten.pow.Tensor_Scalar, (count, -1), (), torch.float32,
            )
            factor, constant = scale, one - scale
        else:
            factor, constant = slope, one - slope
        if factor != 1:
            count = _insert(graph, node, torch.ops.aten.mul.Tensor, (count, factor), (), torch.float32)
        if constant:
            count = _insert(
                graph, node, torch.ops.aten.add.Tensor, (count, constant), (), torch.float32,
            )
        arguments = list(node.args)
        arguments[position] = count
        node.args = tuple(arguments)


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
    fast = optimization == "fast"
    fast_config = fast_config or FastConfig()
    registry = _registry or default_registry()
    from cattorch.experts import _expert_family_context
    with _expert_family_context(registry.expert_families):
        captured = _captured or capture_model(
            model, example_inputs, frontend=frontend, example_kwargs=example_kwargs,
        )
    _normalize_graph(captured.graph)
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
    row_loops: set[str] = set()
    # Items along a node's dynamic axis per unit of runtime length: 1, or
    # more once a view folds the length into a larger dimension.
    dynamic_units: dict[str, int] = {}
    for index, bound in dynamic_inputs.items():
        if index >= len(placeholders):
            raise UnsupportedModelError("dynamic input index is outside the captured signature")
        node = placeholders[index]
        # ``(name, minimum, maximum)``, with an optional fourth item for the
        # axis; the leading one by default.
        axis = bound[3] if len(bound) > 3 else 0
        if len(_get_shape(node)) <= axis:
            raise UnsupportedModelError("a dynamic input must have the dynamic dimension")
        dynamic_axes[node.name] = axis
        dynamic_bounds[node.name] = tuple(bound[:3])
    # Items along a node's dynamic axis beyond the runtime length, from a
    # ``cat`` with static tensors along it; a slice can make it negative.
    dynamic_offsets: dict[str, int] = {}
    length_literals = None
    third_literals: dict = {}
    if dynamic_inputs:
        with _expert_family_context(registry.expert_families):
            length_literals = _length_literals(
                model, example_inputs, example_kwargs, dynamic_inputs, captured, frontend,
            )
            # A third trace tells how arithmetic constants depend on the length.
            if length_literals and len(dynamic_inputs) == 1 and any(
                str(node.target) in _ARITHMETIC and node.name in length_literals
                for node in nodes
            ):
                third_literals = _length_literals(
                    model, example_inputs, example_kwargs, dynamic_inputs, captured,
                    frontend, items=3,
                ) or {}
    graph = captured.graph
    import cattorch.ops  # noqa: F401 - registers the runtime operations
    runtime = torch.ops.cattorch

    def mark(name, axis, origin, offset):
        dynamic_axes[name] = axis
        dynamic_bounds[name] = dynamic_bounds[origin]
        if offset:
            dynamic_offsets[name] = offset

    def shortest(name) -> int:
        """The fewest items the runtime axis of ``name`` can hold."""
        return dynamic_bounds[name][1] + dynamic_offsets.get(name, 0)

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
    pointwise = elementwise - {
        "aten.linear.default", "aten.layer_norm.default", "aten.native_layer_norm.default",
        "aten._softmax.default", "aten.softmax.int", "aten.mean.dim", "aten.embedding.default",
    }
    for node in nodes:
        if node.op != "call_function":
            continue
        if len(dynamic_inputs) == 1 and str(node.target) in _ARITHMETIC:
            _follow_length_literals(
                node, (length_literals or {}).get(node.name, {}),
                third_literals.get(node.name, {}), graph, runtime,
                placeholders[next(iter(dynamic_inputs))], next(iter(dynamic_axes.values())),
            )
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
        unit = dynamic_units.get(parent.name, 1)
        target = str(node.target)
        if not isinstance(node.meta.get("val"), torch.Tensor):
            raise UnsupportedModelError(
                f"{target} returns several tensors from a runtime-length input"
            )
        input_shape = tuple(_get_shape(parent))
        output_shape = tuple(_get_shape(node))
        offset = dynamic_offsets.get(parent.name, 0)
        if unit == 1 and _mix_runtime_axis(
            node, target, parent, axis, offset, input_shape, output_shape,
            graph=graph, runtime=runtime, dynamic_axes=dynamic_axes,
            dynamic_units=dynamic_units, dynamic_offsets=dynamic_offsets,
            literals=(length_literals or {}).get(node.name, {}),
            shortest=shortest, mark=mark,
        ):
            continue
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
            if offset or math.prod(input_shape[:axis]) != 1:
                # With fixed items along the axis, or items before it, the
                # view must keep the axis whole, with the same items before
                # and after it.
                before = math.prod(input_shape[:axis])
                after = math.prod(input_shape[axis + 1:])
                axis = next((
                    position for position, extent in enumerate(output_shape)
                    if extent == input_shape[axis]
                    and math.prod(output_shape[:position]) == before
                    and math.prod(output_shape[position + 1:]) == after
                ), None)
                if axis is None or unit != 1:
                    raise UnsupportedModelError(
                        f"{target} reshapes across a runtime extent that isn't leading"
                    )
                mark(node.name, axis, parent.name, offset)
                continue
            # One unit of runtime length is the whole traced tensor; a view
            # keeps it as a contiguous block, on a dimension of its own when
            # one fits, or folded into the first dimension that holds it.
            block = math.prod(output_shape)
            candidates = [
                position for position, extent in enumerate(output_shape)
                if extent == 1 and math.prod(output_shape[position + 1:]) == block
            ]
            if candidates:
                axis, unit = candidates[-1], 1
            else:
                axis = next(
                    (position for position, extent in enumerate(output_shape) if extent != 1), 0,
                )
                unit = output_shape[axis] if output_shape else 1
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
            if len(dynamic_parents) > 1:
                raise UnsupportedModelError(
                    "matmul between two tensors that both vary along the runtime extent, "
                    "as in self-attention over it, gives two runtime dimensions, which "
                    "aren't supported"
                )
            if parent is not node.args[0] or axis == len(input_shape) - 1:
                raise UnsupportedModelError("matmul cannot use the runtime extent as a contraction dimension")
        elif target in {"cattorch.recurrent_scan.default", "cattorch.recurrent_final.default"}:
            if len(dynamic_parents) != 1 or parent is not node.args[0] or axis != 0 or unit != 1:
                raise UnsupportedModelError(
                    "a recurrent layer takes a runtime extent only along its sequence"
                )
            if target == "cattorch.recurrent_final.default":
                # The final state has no runtime extent.
                continue
        elif target == "aten.slice.Tensor":
            dimension = int(node.args[1]) % len(input_shape)
            start = int(node.args[2]) if len(node.args) > 2 else 0
            end = node.args[3] if len(node.args) > 3 else None
            if dimension == axis and (start != 0 or end not in {None, _RUNTIME_END}):
                raise UnsupportedModelError("partial slicing of a runtime extent is not supported")
            if output_shape != input_shape:
                # A fixed slice of another dimension.
                size = input_shape[dimension]
                first, last, _ = slice(
                    int(node.args[2]) if len(node.args) > 2 and node.args[2] is not None else 0,
                    min(int(end), size) if end is not None else size,
                ).indices(size)
                if (node.args[4] if len(node.args) > 4 else 1) != 1:
                    raise UnsupportedModelError(
                        "slicing a variable-length tensor with a step isn't supported"
                    )
                node.target = runtime.runtime_slice.default
                node.args, node.kwargs = (parent, dimension, first, last), {}
        elif target in elementwise and _user_kernel(registry, target, fast, fast_config):
            # A user's kernel for it may handle only the traced row.
            if offset or any(
                math.prod(_get_shape(value)[:dynamic_axes[value.name]]) != 1
                for value in dynamic_parents
            ):
                raise UnsupportedModelError(
                    f"the kernel passed for {target} gets a variable-length input "
                    "that can't be split into items; read the length from "
                    "ctx.inputs[...].dynamic_axis, or export without the kernel"
                )
            row_loops.add(node.name)
        elif target in elementwise and not (
            # These kernels size their rows from the list's length.
            target == "aten.linear.default" and dynamic_parents == [node.args[0]]
            and axis < len(input_shape) - 1
            or target == "aten.embedding.default" and dynamic_parents == [node.args[1]]
        ) and (
            any(tuple(_get_shape(value)) != output_shape for value in dynamic_parents)
            # Elementwise kernels size their loop from the first operand.
            or not (node.args and getattr(node.args[0], "name", None) in dynamic_axes)
            or (target in pointwise and any(
                _broadcasts_along(value, axis, output_shape)
                for value in node.all_input_nodes if value.name not in dynamic_axes
            ))
        ):
            # Broadcasting the runtime-length operand: the kernel's broadcast
            # tables describe one row.
            row_loops.add(node.name)
        elif target not in shape_only and target not in elementwise:
            # Any other operation that treats rows independently runs its
            # one-row kernel once per row.
            axis, unit = _row_separable_axis(node, dynamic_axes, dynamic_units) or (None, 1)
            if axis is None:
                raise UnsupportedModelError(
                    f"{target} mixes values across the runtime extent, or doesn't "
                    "keep it as a dimension of its own, so it can't take a "
                    "variable-length input"
                )
            row_loops.add(node.name)
        if unit != 1 and math.prod(output_shape[:axis]) != 1:
            raise UnsupportedModelError(
                f"{target} moves a folded runtime extent away from the front"
            )
        if node.name in row_loops and (offset or any(
            math.prod(_get_shape(value)[:dynamic_axes[value.name]]) != 1
            for value in dynamic_parents
        )):
            raise UnsupportedModelError(
                f"{target} runs once per item of the runtime extent, which needs "
                "the extent to lead its tensor, without fixed items joined to it"
            )
        mark(node.name, axis, parent.name, offset)
        if unit != 1:
            dynamic_units[node.name] = unit
    nodes = list(graph.nodes)
    for name in captured.user_output_names:
        if dynamic_units.get(name, 1) != 1:
            raise UnsupportedModelError(
                "an output keeps the runtime extent folded into another dimension; "
                "reshape it back so the extent has a dimension of its own"
            )
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
    for index, traced_name in enumerate(captured.user_output_names):
        output_node = by_name.get(traced_name) if traced_name is not None else None
        value = None if output_node is None else output_node.meta.get("val")
        if output_node is None or not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise UnsupportedModelError(
                f"model output leaf {index} is not a tensor"
            )
        outputs.append((traced_name, _get_shape(output_node), value.dtype))
    traced_name, output_shape, output_dtype = outputs[0]

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
            name = input_name(input_count)
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
            if runtime_axis is not None and low <= runtime_axis <= high and dim0 != dim1:
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
    approximate_fusions: list[str] = []
    fused_nodes = set()
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
        if (
            node.op != "call_function" or str(node.target) != "aten.linear.default"
            # A folded linear is never compiled, so it can't carry an epilogue.
            or node.name in generated_weights
        ):
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
                exact_gelu = (
                    target == "aten.gelu.default" and _gelu_approximation(user) != "tanh"
                )
                if exact_gelu and not (fast and fast_config.activations):
                    break
                # Leave SiLU materialization to the broader SiLU-times-
                # tensor fusion below when it feeds a single multiply.
                if target == "aten.silu.default" and len(user.users) == 1:
                    next_user = next(iter(user.users))
                    if str(next_user.target) == "aten.mul.Tensor":
                        break
                epilogue.append((activation_ops[target], None))
                if exact_gelu:
                    approximate_fusions.append(target)
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
            # A row-looped step broadcasts over the runtime extent.
            and current.name not in row_loops
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
        approximate_fusions=tuple(approximate_fusions),
        row_loops=frozenset(row_loops),
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
        dynamic_offsets=dynamic_offsets,
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
