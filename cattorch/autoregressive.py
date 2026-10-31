"""Sequence analysis for automatic cached generation.

A stateless ``forward(tokens) -> logits`` model is traced twice, at
``max_context`` and at ``max_context - 1``. Pairing the two graphs node by node
shows which tensor axes grow with the sequence, which literals depend on its
length, and where tokens interact. :func:`analyze_sequence` verifies that the
interaction is confined to causal attention, so a later step can rewrite the
graph into a one-row step that reads and writes per-layer K/V caches.

The analysis needs the export copy of the model, as produced by
``OperatorRegistry.adapt_model``: it applies the Hugging Face defaults (SDPA
attention, ``use_cache=False``) and the registered adapters.
"""

from __future__ import annotations

import operator
import copy
from dataclasses import dataclass
from math import prod
from types import MappingProxyType
from typing import Any, Mapping

import torch
import torch.fx
import torch.nn as nn

from cattorch.errors import UnsupportedModelError
from cattorch.frontend import (
    FALLBACK_META, CapturedGraph, capture_model, tensor_output_leaves,
)
from cattorch.graph import _unsupported_operation, fold_static_values
from cattorch.program_runtime import MethodModule

__all__ = [
    "AttentionBlock", "SequenceStepModule", "build_generation_step",
    "build_step_module", "check_step_equivalence", "LiteralDiff", "NodeSequenceInfo", "SequenceAxis",
    "SequencePlan", "SequenceTable", "analyze_sequence", "prepare_sequence_model",
]

# A float mask entry at or below this blocks attention. Hugging Face uses the
# dtype minimum and PyTorch uses -inf.
_MASKED_BELOW = -1e4


# ── Plan ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SequenceAxis:
    """An axis whose size is ``stride * length + offset``.

    ``stride`` is greater than 1 where a view merged the sequence into a larger
    axis (``[B, L, H]`` viewed as ``[B * L, H]`` has stride ``B``).
    """

    axis: int
    stride: int = 1
    offset: int = 0

    def size(self, length: int) -> int:
        return self.stride * length + self.offset


@dataclass(frozen=True)
class NodeSequenceInfo:
    """One live node of the L-graph and its counterpart in the (L-1)-graph.

    ``shape`` and ``short_shape`` are ``None`` for nodes with several outputs
    (``split``); their ``getitem`` users carry the shapes. ``static`` marks
    nodes with no placeholder ancestor.
    """

    name: str
    short_name: str
    op: str
    target: str
    module_path: str | None
    shape: tuple[int, ...] | None
    short_shape: tuple[int, ...] | None
    sequence_axes: tuple[SequenceAxis, ...]
    static: bool


@dataclass(frozen=True)
class LiteralDiff:
    """A non-tensor argument that differs between the two traces.

    ``location`` indexes into the node: ``("args", 1, 0)`` is
    ``node.args[1][0]`` and ``("kwargs", "dtype")`` is ``node.kwargs["dtype"]``.
    """

    node: str
    location: tuple[int | str, ...]
    long: Any
    short: Any

    def value_at(self, length: int, max_context: int) -> Any:
        """The literal extrapolated linearly to a sequence of ``length`` tokens.

        ``length`` counts tokens, so a one-row step uses ``value_at(1, L)``.
        Integers, floats, and lists or tuples of them are supported.
        """
        return _extrapolate(self.long, self.short, length - max_context)


@dataclass(frozen=True, eq=False)
class SequenceTable:
    """A static, input-independent tensor with a sequence axis, used outside attention.

    Examples are RoPE cos/sin, learned position embeddings and position ids.
    ``value`` is the table at ``max_context`` tokens. A row is its slice
    along ``axis``. ``kind`` is ``"additive"`` if an add consumes it (learned
    positions), ``"multiplicative"`` for a mul (RoPE cos/sin), else ``"other"``.
    """

    node: str
    axis: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    value: torch.Tensor
    consumers: tuple[str, ...]
    kind: str
    module_path: str | None


@dataclass(frozen=True)
class AttentionBlock:
    """A causal attention pattern in the L-graph.

    Matched nodes are ``scores = matmul(q, transpose(key))``, optional scale,
    optional causal mask, ``softmax(-1)``, and ``output = matmul(p, value)``.
    ``query`` is the matmul operand. ``key`` and ``value`` are the operands
    after any GQA expansion, ``key_source`` and ``value_source`` the tensors
    before it, with ``n_kv_heads`` heads. ``internal`` holds every matched node
    and the GQA expansion nodes: the nodes a one-row rewrite removes. All
    tensors have the sequence on axis ``-2`` and the heads on ``head_axis``
    (``None`` if the tensors are rank 3).
    """

    index: int
    module_path: str | None
    query: str
    key: str
    value: str
    key_source: str
    value_source: str
    scores: str
    scale: float
    scale_nodes: tuple[str, ...]
    mask_nodes: tuple[str, ...]
    softmax: str
    output: str
    head_axis: int | None
    n_heads: int
    n_kv_heads: int
    kv_groups: int
    head_dim: int
    value_dim: int
    internal: frozenset[str]
    window: int = 0  # keys within the last ``window`` positions; 0 for all
    slopes: tuple[float, ...] | None = None  # ALiBi: slope * (key - query)
    sinks: tuple[float, ...] | None = None  # one extra softmax logit per head


@dataclass(frozen=True, eq=False)
class SequencePlan:
    """Everything a one-row rewrite of the L-graph needs.

    ``long`` is the capture at ``max_context`` tokens (the graph to rewrite)
    and ``short`` the capture at ``max_context - 1``. Node names are those of
    the long graph. ``nodes`` holds the live nodes (ancestors of the outputs)
    in graph order. ``literal_diffs`` covers live nodes. ``static_nodes`` are
    live nodes without a placeholder ancestor. ``token_input`` is the
    ``[1, L]`` placeholder. ``outputs`` lists the output nodes whose sequence
    axis is on the given ``SequenceAxis``.
    """

    max_context: int
    long: CapturedGraph
    short: CapturedGraph
    token_input: str
    outputs: tuple[tuple[str, SequenceAxis], ...]
    nodes: Mapping[str, NodeSequenceInfo]
    literal_diffs: tuple[LiteralDiff, ...]
    attention: tuple[AttentionBlock, ...]
    tables: tuple[SequenceTable, ...]
    static_nodes: frozenset[str]

    @property
    def n_layers(self) -> int:
        return len(self.attention)

    def literal_diffs_for(self, node: str) -> tuple[LiteralDiff, ...]:
        return tuple(diff for diff in self.literal_diffs if diff.node == node)


# ── Entry point ──────────────────────────────────────────────────────────────

def analyze_sequence(
    model: torch.nn.Module,
    max_context: int,
    *,
    example_kwargs: dict | None = None,
) -> SequencePlan:
    """Analyse an export copy of a stateless ``forward(tokens)`` model.

    ``model`` takes one ``[1, length]`` integer token tensor. Pass the copy made
    by ``OperatorRegistry.adapt_model`` in evaluation mode (see
    :func:`prepare_sequence_model`).
    """
    if max_context < 3:
        raise UnsupportedModelError(
            f"automatic generation needs max_context >= 3, got {max_context}"
        )
    length, short_length = max_context, max_context - 1
    try:
        long = _capture(model, length, example_kwargs)
    except UnsupportedModelError as error:
        _explain_context_failure(model, length, example_kwargs, error)
        raise
    short = _capture(model, short_length, example_kwargs)
    long_nodes = list(long.graph.nodes)
    short_nodes = list(short.graph.nodes)
    long_static = _static_nodes(long_nodes)
    short_static = _static_nodes(short_nodes)
    # Static nodes depend on no input. Their structure may differ between the
    # traces (a buffer sliced to its own full size is not sliced at all), so only
    # their values count, and only those that dynamic nodes consume.
    dynamic = [n for n in long_nodes if n.name not in long_static]
    short_dynamic = [n for n in short_nodes if n.name not in short_static]
    _require_same_structure(dynamic, short_dynamic)
    pair = dict(zip((n.name for n in dynamic), short_dynamic))

    live = _live_nodes(long, long_nodes)
    static = long_static
    diffs: list[LiteralDiff] = []
    for node in dynamic:
        if node.name in live and node.op == "call_function":
            _pair_arguments(node, pair[node.name], pair, static, short_static, diffs)
    infos = {}
    for node in long_nodes:
        if node.name in live:
            infos[node.name] = _node_info(node, pair.get(node.name), static, length)
    token = _token_input(long, infos)
    outputs = _outputs(long, infos)

    frontier = {name for name in pair if name in static}
    values = _static_values(long, long_nodes, live & static)
    short_values = _static_values(short, short_nodes, {
        pair[name].name for name in frontier
    })

    blocks = _find_attention(long_nodes, live, static, infos, values, short_values, pair)
    internal = {name for block in blocks for name in block.internal}
    _check_row_wise(long_nodes, live, static, infos, internal)
    tables = _find_tables(
        long_nodes, live, static, infos, values, short_values, pair, internal,
        length,
    )
    return SequencePlan(
        max_context=max_context,
        long=long,
        short=short,
        token_input=token,
        outputs=outputs,
        nodes=MappingProxyType({name: info for name, info in infos.items()}),
        literal_diffs=tuple(diffs),
        attention=tuple(blocks),
        tables=tuple(tables),
        static_nodes=frozenset(live & static),
    )


def prepare_sequence_model(model: torch.nn.Module, registry=None) -> torch.nn.Module:
    """The export copy :func:`analyze_sequence` expects.

    Pass the export's registry; without one, a clone of the default registry
    holds the copy's expert families, so they never leak into the default.
    """
    from cattorch.operator_registry import default_registry

    return (registry or default_registry().clone()).adapt_model(model)


# ── Tracing and pairing ──────────────────────────────────────────────────────

def _capture(model, length: int, example_kwargs) -> CapturedGraph:
    from cattorch.experts import _ACTIVE_EXPERT_FAMILIES, _expert_family_context
    from cattorch.operator_registry import default_registry

    tokens = torch.zeros(1, length, dtype=torch.long)
    # ``build_generation_step`` activates the export's expert families.
    families = _ACTIVE_EXPERT_FAMILIES.get() or default_registry().expert_families
    with _expert_family_context(families):
        return capture_model(model, (tokens,), example_kwargs=example_kwargs)


def _explain_context_failure(model, length: int, example_kwargs, error) -> None:
    """Name ``max_context`` when ``forward`` itself fails at that length."""
    from cattorch.experts import _ACTIVE_EXPERT_FAMILIES, _expert_family_context
    from cattorch.operator_registry import default_registry

    families = _ACTIVE_EXPERT_FAMILIES.get() or default_registry().expert_families
    try:
        with torch.no_grad(), _expert_family_context(families):
            model(torch.zeros(1, length, dtype=torch.long), **(example_kwargs or {}))
    except IndexError as eager:
        raise UnsupportedModelError(
            f"the model's forward fails on {length} tokens ({eager}); "
            f"max_context ({length}) is probably larger than a position table"
        ) from error
    except Exception:  # noqa: BLE001 - the trace error is the better report
        return


def _module_path(node) -> str | None:
    stack = node.meta.get("nn_module_stack", {}) or {}
    return next((entry[0] for entry in reversed(tuple(stack.values())) if entry[0]), None)


def _where(node) -> str:
    path = _module_path(node)
    return f"{node.target}" + (f" at {path}" if path else "")


def _require_same_structure(long_nodes, short_nodes) -> None:
    for position, (a, b) in enumerate(zip(long_nodes, short_nodes)):
        if (a.op, str(a.target)) != (b.op, str(b.target)):
            raise UnsupportedModelError(
                "the model's graph changes with the sequence length: node "
                f"{position} is {a.op} {a.target} at the longest length and "
                f"{b.op} {b.target} at one token fewer ({_where(a)}); "
                "length-dependent Python branches (such as an `if` on "
                "tokens.shape[1]) cannot be cached"
            )
    if len(long_nodes) != len(short_nodes):
        extra = (long_nodes if len(long_nodes) > len(short_nodes) else short_nodes)[
            min(len(long_nodes), len(short_nodes))
        ]
        raise UnsupportedModelError(
            "the model's graph changes with the sequence length: "
            f"{len(long_nodes)} nodes at the longest length against "
            f"{len(short_nodes)} at one token fewer, first extra node "
            f"{extra.target} ({_where(extra)}); length-dependent Python "
            "branches cannot be cached"
        )


def _live_nodes(captured: CapturedGraph, nodes) -> set[str]:
    live: set[str] = set()
    stack = [node for node in nodes if node.op == "output"]
    while stack:
        node = stack.pop()
        if node.name in live:
            continue
        live.add(node.name)
        stack.extend(node.all_input_nodes)
    # Keep every placeholder visible even if unused.
    live.update(node.name for node in nodes if node.op == "placeholder")
    return live


def _static_nodes(nodes) -> set[str]:
    static: set[str] = set()
    for node in nodes:
        if node.op == "get_attr":
            static.add(node.name)
        elif node.op == "call_function" and all(
            arg.name in static for arg in node.all_input_nodes
        ):
            static.add(node.name)
    return static


def _shapes(node) -> list[tuple[int, ...]] | None:
    """Per-output shapes of ``node`` or ``None`` when it has no tensor value."""
    value = node.meta.get("val")
    if isinstance(value, torch.Tensor):
        return [tuple(value.shape)]
    if isinstance(value, (tuple, list)) and value and all(
        isinstance(item, torch.Tensor) for item in value
    ):
        return [tuple(item.shape) for item in value]
    return None


def _node_info(node, short_node, static: set[str], length: int) -> NodeSequenceInfo:
    shapes = _shapes(node)
    short_shapes = _shapes(short_node) if short_node is not None else None
    shape = short_shape = None
    axes: tuple[SequenceAxis, ...] = ()
    if short_node is None:
        # A static node no dynamic node consumes directly: its shape at
        # max_context is all anyone needs.
        if shapes is not None and len(shapes) == 1:
            shape = shapes[0]
    elif shapes is not None and short_shapes is not None and len(shapes) == len(short_shapes):
        if len(shapes) == 1:
            shape, short_shape = shapes[0], short_shapes[0]
            axes = _sequence_axes(node, shape, short_shape, length)
    elif (shapes is None) != (short_shapes is None):
        raise UnsupportedModelError(
            f"{_where(node)}: the value kind changes with the sequence length"
        )
    return NodeSequenceInfo(
        name=node.name,
        short_name=short_node.name if short_node is not None else "",
        op=node.op,
        target=str(node.target),
        module_path=_module_path(node),
        shape=shape,
        short_shape=short_shape,
        sequence_axes=axes,
        static=node.name in static,
    )


def _sequence_axes(node, shape, short_shape, length) -> tuple[SequenceAxis, ...]:
    if len(shape) != len(short_shape):
        raise UnsupportedModelError(
            f"{_where(node)}: the tensor rank changes with the sequence length "
            f"({len(shape)} against {len(short_shape)})"
        )
    axes = []
    for axis, (a, b) in enumerate(zip(shape, short_shape)):
        if a != b:
            stride = a - b
            axes.append(SequenceAxis(axis, stride, a - stride * length))
    return tuple(axes)


def _token_input(captured: CapturedGraph, infos) -> str:
    names = [name for name in captured.user_input_names if name in infos]
    if len(names) != 1:
        raise UnsupportedModelError(
            "automatic generation needs a model with one token input, "
            f"got {len(names)} tensor inputs"
        )
    info = infos[names[0]]
    if info.sequence_axes != (SequenceAxis(1, 1, 0),) or info.shape[0] != 1:
        raise UnsupportedModelError(
            f"the token input {names[0]!r} must be [1, length], got shape {info.shape}"
        )
    return names[0]


def _outputs(captured: CapturedGraph, infos) -> tuple[tuple[str, SequenceAxis], ...]:
    result = []
    for name in captured.user_output_names:
        info = infos.get(name) if name is not None else None
        if info is not None and len(info.sequence_axes) == 1:
            result.append((name, info.sequence_axes[0]))
    first = captured.user_output_names[0] if captured.user_output_names else None
    if not result or result[0][0] != first:
        info = infos.get(first) if first is not None else None
        raise UnsupportedModelError(
            "the model's first output has no sequence axis"
            + ("" if info is None else f" (shape {info.shape})")
            + ": automatic generation needs one output row per token"
        )
    axis = result[0][1]
    if axis.stride != 1 or axis.offset != 0:
        raise UnsupportedModelError(
            f"the model's output axis {axis.axis} is not one row per token"
        )
    return tuple(result)


# ── Literals ─────────────────────────────────────────────────────────────────

def _pair_arguments(node, short_node, pair, static, short_static, out) -> None:
    """Pair static arguments with their counterparts and record literal diffs."""
    def walk(a, b, location):
        if isinstance(a, torch.fx.Node):
            if not isinstance(b, torch.fx.Node):
                raise UnsupportedModelError(
                    f"{_where(node)}: the graph wiring changes with the "
                    "sequence length"
                )
            if a.name in static:
                if b.name not in short_static:
                    raise UnsupportedModelError(
                        f"{_where(node)}: an input is a constant at the longest "
                        "length but depends on the tokens one token fewer"
                    )
                pair[a.name] = b
            elif pair.get(a.name) is not b:
                raise UnsupportedModelError(
                    f"{_where(node)}: the graph wiring changes with the "
                    "sequence length"
                )
        elif isinstance(a, (list, tuple)):
            if not isinstance(b, (list, tuple)) or len(a) != len(b):
                raise UnsupportedModelError(
                    f"{_where(node)}: argument {location} has a different "
                    "length at one token fewer; length-dependent Python "
                    "branches cannot be cached"
                )
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, (*location, i))
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                raise UnsupportedModelError(
                    f"{_where(node)}: keyword arguments change with the sequence length"
                )
            for key in a:
                walk(a[key], b[key], (*location, key))
        elif isinstance(a, torch.Tensor):
            return
        else:
            if isinstance(b, (torch.fx.Node, list, tuple, dict)) or a != b:
                out.append(LiteralDiff(node.name, location, a, b))

    walk(node.args, short_node.args, ("args",))
    walk(node.kwargs, short_node.kwargs, ("kwargs",))


def _extrapolate(long, short, steps: int):
    """``short + (long - short) * (steps + 1)`` element-wise (steps from long)."""
    if isinstance(long, bool) or not isinstance(long, (int, float, list, tuple)):
        return long
    if isinstance(long, (list, tuple)):
        return type(long)(_extrapolate(a, b, steps) for a, b in zip(long, short))
    return long + (long - short) * steps


# ── Static values ────────────────────────────────────────────────────────────

def _static_values(captured, nodes, names) -> dict[str, torch.Tensor]:
    """Values of the static nodes, folding without a size limit."""
    values = {
        name: tensor for name, tensor in captured.state_inputs.items() if name in names
    }
    folded = fold_static_values(
        nodes, captured.state_inputs, seed=values, element_limit=None,
        passthrough=frozenset({"aten.detach_.default"}),
    )
    values.update(folded)
    return {name: value for name, value in values.items() if name in names}


# ── Row-wise classification ──────────────────────────────────────────────────

_POINTWISE = frozenset({
    "add", "sub", "mul", "div", "rsub", "neg", "relu", "sigmoid", "tanh", "gelu",
    "silu", "leaky_relu", "elu", "pow", "rsqrt", "exp", "log", "log10", "log2",
    "log1p", "expm1", "sqrt", "abs", "floor", "ceil", "round", "trunc",
    "reciprocal", "erf", "logical_not", "logical_and", "logical_or", "clamp",
    "clamp_min", "clamp_max", "hardtanh", "relu6", "bitwise_not", "bitwise_and",
    "bitwise_or", "to", "_to_copy", "eq", "ne", "gt", "lt", "ge", "le", "maximum",
    "minimum", "where", "sin", "cos", "masked_fill", "square", "sign", "softplus",
    "hardswish", "hardsigmoid", "mish", "nan_to_num", "dropout", "contiguous",
    "clone", "alias", "detach", "detach_", "lift_fresh_copy", "type_as", "copy",
    "isnan", "isinf", "full_like", "ones_like", "zeros_like", "rotary_embedding", "rotary_half",
    "_assert_tensor_metadata",
})
# Operations that only relabel axes: each input axis survives in the output.
_RELABEL = frozenset({
    "transpose", "permute", "t", "movedim", "squeeze", "unsqueeze", "expand",
    "expand_as", "broadcast_to", "stack", "numpy_T",
})
_VIEWS = frozenset({
    "view", "reshape", "_unsafe_view", "flatten", "unflatten",
})
# name -> (argument index of the dimension(s), keyword name, takes a list)
_DIM_ARGUMENT: dict[str, tuple[int, str, bool]] = {
    "sum": (1, "dim", True), "mean": (1, "dim", True), "amax": (1, "dim", True),
    "amin": (1, "dim", True), "var": (1, "dim", True), "std": (1, "dim", True),
    "logsumexp": (1, "dim", True), "norm": (2, "dim", True),
    "softmax": (1, "dim", False), "_softmax": (1, "dim", False),
    "log_softmax": (1, "dim", False), "_log_softmax": (1, "dim", False),
    "cumsum": (1, "dim", False), "cumprod": (1, "dim", False),
    "cummax": (1, "dim", False), "cummin": (1, "dim", False),
    "logcumsumexp": (1, "dim", False),
    "argmax": (1, "dim", False), "argmin": (1, "dim", False),
    "max": (1, "dim", False), "min": (1, "dim", False),
    "prod": (1, "dim", False), "any": (1, "dim", False), "all": (1, "dim", False),
    "median": (1, "dim", False), "mode": (1, "dim", False),
    "sort": (1, "dim", False), "argsort": (1, "dim", False),
    "topk": (2, "dim", False), "kthvalue": (2, "dim", False),
    "topk_values": (2, "dim", False), "topk_indices": (2, "dim", False),
    "flip": (1, "dims", True), "roll": (2, "dims", True), "diff": (2, "dim", False),
    "glu": (1, "dim", False),
}
_SPLITS = {"split": 2, "chunk": 2, "split_with_sizes": 2, "unbind": 1, "tensor_split": 2}


def _op_name(node) -> str:
    if node.target is operator.getitem:
        return "getitem"
    parts = str(node.target).split(".")
    return parts[1] if len(parts) >= 2 else parts[0]


def _normalize(dim: int, ndim: int) -> int:
    return dim % ndim if ndim else 0


def _argument(node, index: int, keyword: str, default=None):
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get(keyword, default)


def _dims(node, ndim: int, spec) -> set[int]:
    index, keyword, _ = spec
    value = _argument(node, index, keyword)
    if value is None or (isinstance(value, (list, tuple)) and not value):
        return set(range(ndim))
    if isinstance(value, int):
        value = [value]
    return {_normalize(int(dim), ndim) for dim in value}


def _mixing(node, detail: str):
    return _unsupported_operation(
        node, f"{detail}; automatic generation allows mixing tokens only in causal attention"
    )


def _check_row_wise(nodes, live, static, infos, internal) -> None:
    for node in nodes:
        if node.name not in live or node.name in static or node.name in internal:
            continue
        if node.op != "call_function":
            continue
        info = infos[node.name]
        inputs = [infos[arg.name] for arg in node.all_input_nodes if arg.name in infos]
        sequence_inputs = [item for item in inputs if item.sequence_axes]
        if not sequence_inputs and not info.sequence_axes:
            continue
        for item in (*sequence_inputs, info):
            for axis in item.sequence_axes:
                if axis.offset != 0:
                    raise _mixing(
                        node,
                        f"a tensor axis has {axis.size(10)} elements for 10 tokens "
                        "(a fixed offset from the sequence length)",
                    )
        _check_node(node, info, inputs, infos)


def _check_node(node, info, inputs, infos) -> None:
    name = _op_name(node)
    first = _first_input(node, infos)
    ndim = len(first.shape) if first is not None and first.shape is not None else 0
    axes = {a.axis for a in first.sequence_axes} if first is not None else set()

    if name == "getitem":
        return
    if name in _POINTWISE:
        out_ndim = len(info.shape)
        out_axes = {a.axis for a in info.sequence_axes}
        for item in inputs:
            if item.shape is None:
                continue
            shift = out_ndim - len(item.shape)
            for axis in item.sequence_axes:
                if axis.axis + shift not in out_axes:
                    raise _mixing(node, "the operation drops the sequence axis")
        return
    if name in _RELABEL:
        if len(info.sequence_axes) != len(first.sequence_axes):
            raise _mixing(node, "the operation removes the sequence axis")
        return
    if name in _VIEWS:
        _check_view(node, info, first)
        return
    if name in _DIM_ARGUMENT:
        spec = _DIM_ARGUMENT[name]
        reduced = _dims(node, ndim, spec)
        if axes & reduced:
            raise _mixing(node, f"`{name}` over the sequence")
        return
    if name in {"layer_norm", "native_layer_norm", "rms_norm"}:
        shape = node.args[1]
        count = len(shape) if isinstance(shape, (list, tuple)) else 1
        if axes & set(range(ndim - count, ndim)):
            raise _mixing(node, f"`{name}` normalises over the sequence")
        return
    if name in _SPLITS:
        dim = _normalize(int(_argument(node, _SPLITS[name], "dim", 0)), ndim)
        if dim in axes:
            raise _mixing(node, f"`{name}` along the sequence")
        return
    if name == "cat":
        dim = _normalize(int(_argument(node, 1, "dim", 0)), len(info.shape))
        if dim in {a.axis for a in info.sequence_axes}:
            raise _mixing(node, "`cat` along the sequence")
        return
    if name in {"slice", "narrow", "select", "index_select", "gather", "scatter", "unfold"}:
        dim = _normalize(int(_argument(node, 1, "dim", 0)), ndim)
        if dim not in axes:
            return
        if name == "slice":
            start = _argument(node, 2, "start", 0) or 0
            step = _argument(node, 4, "step", 1) or 1
            if start == 0 and step == 1 and any(
                a.axis == dim and a.stride == 1 for a in info.sequence_axes
            ):
                return
        raise _mixing(node, f"`{name}` selects along the sequence")
    if name in {"linear", "mm", "matmul", "bmm", "addmm"}:
        _check_matmul(node, name, info, infos)
        return
    if name == "embedding":
        if first is not None and first.sequence_axes:
            raise _mixing(node, "the embedding table has a sequence axis")
        return
    if name in {"routed_experts", "expert_family_moe", "stacked_swiglu_moe"}:
        # Experts and their routes act on each token's last-axis features.
        if ndim - 1 in axes:
            raise _mixing(node, "experts mix the sequence as features")
        return
    raise _unsupported_operation(
        node,
        "not known to act on each token independently, but its input has a "
        "sequence axis; automatic generation allows mixing tokens only in "
        "causal attention",
    )


def _first_input(node, infos):
    """Info of the first tensor argument (the first list element for ``cat``)."""
    arg = node.args[0] if node.args else None
    if isinstance(arg, (list, tuple)) and arg:
        arg = arg[0]
    return infos.get(arg.name) if isinstance(arg, torch.fx.Node) else None


def _check_matmul(node, name, info, infos) -> None:
    args = node.args

    def info_of(position):
        arg = args[position] if len(args) > position else None
        return infos.get(arg.name) if isinstance(arg, torch.fx.Node) else None

    if name == "linear":
        left, right = info_of(0), None
        extras = [info_of(1), info_of(2)]
    elif name == "addmm":
        left, right = info_of(1), info_of(2)
        extras = [info_of(0)]
    else:
        left, right, extras = info_of(0), info_of(1), []
    if left is not None and left.shape is not None:
        last = len(left.shape) - 1
        if any(a.axis == last for a in left.sequence_axes):
            raise _mixing(node, f"`{name}` contracts the sequence axis")
    if right is not None and right.shape is not None and right.sequence_axes:
        batch = max(len(right.shape) - 2, 0)
        if any(a.axis >= batch for a in right.sequence_axes):
            raise _mixing(node, f"`{name}` contracts or produces the sequence axis of its second operand")
    for extra in extras:
        if extra is not None and extra.sequence_axes and name == "linear":
            raise _mixing(node, "linear weight or bias has a sequence axis")
    if (left is not None and left.sequence_axes or right is not None and right.sequence_axes) \
            and not info.sequence_axes:
        raise _mixing(node, f"`{name}` drops the sequence axis")


def _check_view(node, info, source) -> None:
    for shape_in, shape_out, axes_in, axes_out in (
        (source.shape, info.shape, source.sequence_axes, info.sequence_axes),
        (source.short_shape, info.short_shape, source.sequence_axes, info.sequence_axes),
    ):
        if shape_in is None or shape_out is None:
            raise _mixing(node, "the view has no static shape")
        for axis in axes_in:
            before = prod(shape_in[:axis.axis])
            if not any(prod(shape_out[:out.axis]) == before for out in axes_out):
                raise _mixing(
                    node,
                    f"the view {tuple(shape_in)} -> {tuple(shape_out)} merges "
                    "the sequence axis into a larger axis that is not "
                    "sequence-major",
                )


# ── Attention ────────────────────────────────────────────────────────────────

def _find_attention(nodes, live, static, infos, values, short_values, pair):
    blocks = []
    for node in nodes:
        if node.name not in live or node.name in static or node.op != "call_function":
            continue
        if _op_name(node) not in {"softmax", "_softmax"}:
            continue
        source = infos[node.args[0].name]
        dim = _normalize(int(_argument(node, 1, "dim", -1)), len(source.shape or ()))
        if dim not in {axis.axis for axis in source.sequence_axes}:
            continue
        blocks.append(_match_attention(
            len(blocks), node, infos, values, short_values, pair,
        ))
    return blocks


def _skip_casts(node, internal):
    while (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and _op_name(node) in {"to", "_to_copy", "dropout"}
    ):
        internal.append(node.name)
        node = node.args[0]
    return node


def _scalar(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _match_attention(index, softmax, infos, values, short_values, pair) -> AttentionBlock:
    path = _module_path(softmax)

    def reject(detail):
        return _unsupported_operation(
            softmax,
            f"softmax over the sequence is not a recognised attention pattern "
            f"({detail}); expected matmul(q, k^T) -> [scale] -> causal mask -> "
            "softmax -> matmul(p, v)",
        )

    internal: list[str] = [softmax.name]
    scores_in = _skip_casts(softmax.args[0], internal)
    scores_in = _skip_row_shift(scores_in, internal)
    sink_node = None
    if (
        isinstance(scores_in, torch.fx.Node) and scores_in.op == "call_function"
        and _op_name(scores_in) == "cat"
    ):
        # GPT-OSS: cat([scores, sinks], -1), softmax, then drop the sink column.
        pieces = scores_in.args[0]
        ndim = len(scores_in.meta["val"].shape)
        if (
            len(pieces) == 2
            and _normalize(int(_argument(scores_in, 1, "dim", 0)), ndim) == ndim - 1
            and isinstance(pieces[1], torch.fx.Node) and pieces[1].name in values
            and pieces[1].meta["val"].shape[-1] == 1
        ):
            sink_node = pieces[1]
            internal.append(scores_in.name)
            scores_in = _skip_casts(pieces[0], internal)
    # Static masks and biases, applied in any order, become one additive table.
    masks: list[tuple[str, torch.fx.Node]] = []
    while isinstance(scores_in, torch.fx.Node) and scores_in.op == "call_function":
        op = _op_name(scores_in)
        if op == "masked_fill":
            mask_node, fill = scores_in.args[1], scores_in.args[2]
            if not (isinstance(mask_node, torch.fx.Node) and mask_node.name in values):
                raise reject("the mask is not a static tensor")
            if not (_scalar(fill) and fill <= _MASKED_BELOW):
                raise reject(f"masked_fill value {fill!r} is not -inf")
            masks.append(("masked_fill", mask_node))
        elif op == "add":
            operands = [scores_in.args[0], scores_in.args[1]]
            static_operands = [
                a for a in operands
                if isinstance(a, torch.fx.Node) and a.name in values
                and infos[a.name].sequence_axes
            ]
            if len(static_operands) != 1:
                break
            masks.append(("add", static_operands[0]))
        else:
            break
        internal.append(scores_in.name)
        scores_in = _skip_casts(
            next(a for a in scores_in.args[:2] if a is not masks[-1][1]), internal,
        )

    scale = 1.0
    scale_nodes = []
    while (
        isinstance(scores_in, torch.fx.Node)
        and scores_in.op == "call_function"
        and _op_name(scores_in) in {"mul", "div"}
        and _scalar(scores_in.args[1] if len(scores_in.args) > 1 else None)
    ):
        factor = float(scores_in.args[1])
        scale = scale * factor if _op_name(scores_in) == "mul" else scale / factor
        scale_nodes.append(scores_in.name)
        scores_in = _skip_casts(scores_in.args[0], internal)
    internal.extend(scale_nodes)

    if not (
        isinstance(scores_in, torch.fx.Node) and scores_in.op == "call_function"
        and _op_name(scores_in) in {"matmul", "bmm"}
    ):
        raise reject("the scores are not a matmul of q and k^T")
    scores = scores_in
    internal.append(scores.name)
    query, key_t = scores.args[0], scores.args[1]
    if not (
        isinstance(key_t, torch.fx.Node) and key_t.op == "call_function"
        and _is_last_two_transpose(key_t)
    ):
        raise reject("the keys are not transposed on their last two axes")
    internal.append(key_t.name)
    key = key_t.args[0]

    # softmax -> (casts) -> matmul(p, v)
    p_node = softmax
    users = list(p_node.users)
    sink_dropped = False
    while len(users) == 1 and (
        _op_name(users[0]) in {"to", "_to_copy", "dropout"}
        or (sink_node is not None and not sink_dropped and _drops_last_column(users[0]))
    ):
        sink_dropped = sink_dropped or _op_name(users[0]) == "slice"
        internal.append(users[0].name)
        p_node = users[0]
        users = list(p_node.users)
    if len(users) != 1 or _op_name(users[0]) not in {"matmul", "bmm"} or users[0].args[0] is not p_node:
        raise reject("the probabilities are not used only by matmul(p, v)")
    output = users[0]
    internal.append(output.name)
    value = output.args[1]

    for label, tensor in (("q", query), ("k", key), ("v", value)):
        item = infos[tensor.name]
        nd = len(item.shape)
        if nd not in (3, 4) or item.sequence_axes != (SequenceAxis(nd - 2, 1, 0),):
            raise reject(
                f"{label} {item.shape} must be [..., heads, tokens, dim] with "
                "the sequence on axis -2 and no other growing axis"
            )
    q_info = infos[query.name]
    nd = len(q_info.shape)
    head_axis = nd - 3 if nd == 4 else None

    key_source, key_groups, key_nodes = _unwrap_gqa(key, head_axis, infos)
    value_source, value_groups, value_nodes = _unwrap_gqa(value, head_axis, infos)
    if key_groups != value_groups:
        raise reject(f"keys repeat {key_groups}x but values {value_groups}x")
    internal.extend(key_nodes + value_nodes)
    n_heads = q_info.shape[head_axis] if head_axis is not None else 1
    n_kv = infos[key_source.name].shape[head_axis] if head_axis is not None else 1
    if key_groups == 1 and n_kv == 1 and n_heads > 1:
        # Broadcast multi-query attention: matmul broadcasts the single kv head
        # across the query heads, with no repeat_kv.
        key_groups = n_heads
    if n_heads != n_kv * key_groups:
        raise reject(f"{n_heads} query heads against {n_kv} kv heads x {key_groups}")

    if not masks:
        raise UnsupportedModelError(
            f"attention over the sequence isn't causal at {path}: the softmax node "
            f"{softmax.name} has no mask, so every token attends to later "
            "tokens (a bidirectional model such as BERT)"
        )
    if sink_node is not None and not sink_dropped:
        raise reject("the sink column is not dropped before matmul(p, v)")
    window, slopes = _check_causal(
        masks, values, short_values, pair, path, int(n_heads),
    )
    sinks = None
    if sink_node is not None:
        sinks = _per_head(values[sink_node.name][..., 0], int(n_heads), head_axis)
        if sinks is None:
            raise reject("the sink logits differ between tokens")
    return AttentionBlock(
        index=index,
        module_path=path,
        query=query.name,
        key=key.name,
        value=value.name,
        key_source=key_source.name,
        value_source=value_source.name,
        scores=scores.name,
        scale=scale,
        scale_nodes=tuple(scale_nodes),
        mask_nodes=tuple(node.name for _, node in masks),
        softmax=softmax.name,
        output=output.name,
        head_axis=head_axis,
        n_heads=int(n_heads),
        n_kv_heads=int(n_kv),
        kv_groups=int(key_groups),
        head_dim=int(q_info.shape[-1]),
        value_dim=int(infos[value.name].shape[-1]),
        internal=frozenset(internal),
        window=window,
        slopes=slopes,
        sinks=sinks,
    )


def _is_last_two_transpose(node) -> bool:
    name = _op_name(node)
    ndim = len(node.meta["val"].shape)
    if name == "transpose":
        dims = {_normalize(int(node.args[1]), ndim), _normalize(int(node.args[2]), ndim)}
        return dims == {ndim - 1, ndim - 2}
    if name == "permute":
        dims = [_normalize(int(d), ndim) for d in node.args[1]]
        return dims == [*range(ndim - 2), ndim - 1, ndim - 2]
    return False


def _unwrap_gqa(node, head_axis, infos):
    """Return ``(source, groups, expansion_nodes)`` for a K or V operand."""
    if head_axis is None:
        return node, 1, []
    # repeat_interleave decomposition: cat of per-head slices along the head axis.
    if node.op == "call_function" and _op_name(node) == "cat":
        pieces = node.args[0]
        dim = _normalize(int(_argument(node, 1, "dim", 0)), len(node.meta["val"].shape))
        if dim == head_axis and all(
            isinstance(p, torch.fx.Node) and _op_name(p) == "slice" for p in pieces
        ):
            sources = {p.args[0] for p in pieces}
            if len(sources) == 1:
                source = next(iter(sources))
                heads = source.meta["val"].shape[head_axis]
                starts = [p.args[2] for p in pieces]
                groups = len(pieces) // heads if heads else 0
                if (
                    groups >= 1 and groups * heads == len(pieces)
                    and all(_normalize(int(p.args[1]), len(source.meta["val"].shape)) == head_axis
                            and p.args[3] == p.args[2] + 1 for p in pieces)
                    and starts == [i // groups for i in range(len(pieces))]
                ):
                    return source, groups, [node.name, *{p.name for p in pieces}]
    # Hugging Face repeat_kv: unsqueeze -> expand -> reshape.
    if node.op == "call_function" and _op_name(node) in {"reshape", "view", "_unsafe_view"}:
        expanded = node.args[0]
        if isinstance(expanded, torch.fx.Node) and _op_name(expanded) == "expand":
            unsqueezed = expanded.args[0]
            if isinstance(unsqueezed, torch.fx.Node) and _op_name(unsqueezed) == "unsqueeze":
                source = unsqueezed.args[0]
                source_shape = source.meta["val"].shape
                groups = expanded.meta["val"].shape[head_axis + 1]
                if (
                    _normalize(int(unsqueezed.args[1]), len(source_shape) + 1) == head_axis + 1
                    and node.meta["val"].shape[head_axis] == source_shape[head_axis] * groups
                ):
                    return source, int(groups), [node.name, expanded.name, unsqueezed.name]
    return node, 1, []


def _skip_row_shift(node, internal):
    """Skip ``x - x.max(-1, keepdim=True)``: softmax doesn't change with it."""
    if not (
        isinstance(node, torch.fx.Node) and node.op == "call_function"
        and _op_name(node) == "sub" and isinstance(node.args[1], torch.fx.Node)
    ):
        return node
    source, maximum = node.args[0], node.args[1]
    chain = [maximum]
    if maximum.target is operator.getitem and maximum.args[1] == 0:
        maximum = maximum.args[0]
        chain.append(maximum)
    ndim = len(source.meta["val"].shape)
    if (
        _op_name(maximum) in {"amax", "max"} and maximum.args[0] is source
        and _dims(maximum, ndim, _DIM_ARGUMENT[_op_name(maximum)]) == {ndim - 1}
    ):
        internal.extend([node.name, *(item.name for item in chain)])
        return _skip_casts(source, internal)
    return node


def _drops_last_column(node) -> bool:
    """``p[..., :-1]``: the slice that removes an attention sink's column."""
    if _op_name(node) != "slice":
        return False
    ndim = len(node.meta["val"].shape)
    size = node.args[0].meta["val"].shape[-1]
    start = _argument(node, 2, "start", 0) or 0
    end = _argument(node, 3, "end", None)
    end = size if end is None else (end + size if end < 0 else min(end, size))
    return (
        _normalize(int(_argument(node, 1, "dim", 0)), ndim) == ndim - 1
        and start == 0 and end == size - 1 and (_argument(node, 4, "step", 1) or 1) == 1
    )


def _per_head(table: torch.Tensor, heads: int, head_axis) -> tuple[float, ...] | None:
    """Values that vary only by head (``[..., heads, tokens]``), one per head."""
    table = table.reshape(-1, *table.shape[-2:]) if table.ndim >= 2 else table.reshape(1, 1, -1)
    if table.shape[-2] not in (1, heads) or head_axis is None and table.shape[-2] != 1:
        return None
    if not bool((table == table[:1, :, :1]).all()):
        return None
    return tuple(float(v) for v in table[0, :, 0].expand(heads))


def _combined_bias(masks, tables) -> torch.Tensor:
    total = None
    for (kind, _), table in zip(masks, tables):
        if kind == "masked_fill":
            table = torch.zeros(table.shape).masked_fill(table.to(torch.bool), float("-inf"))
        table = table.to(torch.float64)
        total = table if total is None else total + table
    return total


def _check_causal(masks, values, short_values, pair, path, heads) -> tuple[int, tuple | None]:
    """The window and ALiBi slopes of an attention block's masks and biases.

    The combined table must block every later token, may block tokens more
    than ``window`` back, and may add ``slope * (key - query)`` per head plus
    any per-row constant (a softmax ignores it). Returns ``(window, slopes)``.
    """
    results = []
    for label, lookup in (("", values), ("one token fewer: ", short_values)):
        tables = []
        for _, node in masks:
            name = node.name if lookup is values else pair[node.name].name
            tables.append(lookup.get(name))
        if any(table is None for table in tables):
            continue
        bias = _combined_bias(masks, tables)
        if bias.ndim < 2:
            raise UnsupportedModelError(
                f"attention over the sequence isn't causal at {path}: the "
                f"mask has shape {tuple(bias.shape)}, not [tokens, tokens]"
            )
        flat = bias.reshape(-1, *bias.shape[-2:])
        blocked = flat <= _MASKED_BELOW
        if not bool((blocked == blocked[:1]).all()):
            raise UnsupportedModelError(
                f"attention at {path} uses a mask that differs between "
                "batch rows or heads"
            )
        rows, columns = blocked.shape[-2:]
        allowed = ~blocked[0]
        reach = int(allowed[-1].sum())
        window = 0 if reach >= rows else reach
        row = torch.arange(rows)[:, None]
        column = torch.arange(columns)[None, :]
        expected = column <= row
        if window:
            expected &= row - column < window
        wrong = (allowed != expected).nonzero()
        if len(wrong):
            r, c = (int(v) for v in wrong[0])
            what = "attends to" if bool(allowed[r, c]) else "cannot attend to"
            raise UnsupportedModelError(
                f"attention over the sequence isn't causal at {path}: {label}"
                f"token {r} {what} token {c} (each token may attend to itself and "
                "earlier tokens only, optionally within a fixed window)"
            )
        slopes = _fit_slopes(flat, allowed, heads, path)
        results.append((window, slopes))
    if not results:
        raise UnsupportedModelError(f"attention at {path}: the mask could not be evaluated")
    window, slopes = results[0]
    for other_window, other_slopes in results[1:]:
        # A window as long as the shorter trace looks unlimited there.
        if other_window not in (window, 0) or (other_slopes is None) != (slopes is None):
            raise UnsupportedModelError(
                f"attention at {path} uses a mask whose pattern changes with the "
                "sequence length"
            )
    return window, slopes


def _fit_slopes(flat, allowed, heads, path) -> tuple[float, ...] | None:
    """Per-head ALiBi slopes of a bias, or ``None`` when it is only a mask."""
    finite = torch.where(allowed, flat, torch.zeros_like(flat))
    # Softmax ignores a constant per row: measure every entry from the diagonal.
    diagonal = torch.diagonal(finite, dim1=-2, dim2=-1)[..., :, None]
    relative = torch.where(allowed, finite - diagonal, torch.zeros_like(finite))
    if not bool((relative.abs() > 1e-9).any()):
        return None
    rows = flat.shape[-2]
    if rows < 2 or flat.shape[0] not in (1, heads):
        raise UnsupportedModelError(
            f"attention at {path} adds a bias that is not one ALiBi slope per head"
        )
    slopes = -relative[:, 1, 0]
    distance = (
        torch.arange(flat.shape[-1], dtype=flat.dtype)[None, :]
        - torch.arange(rows, dtype=flat.dtype)[:, None]
    )
    fitted = slopes[:, None, None] * distance
    error = torch.where(allowed, relative - fitted, torch.zeros_like(relative)).abs()
    if bool((error > 1e-5 * (1 + relative.abs())).any()):
        raise UnsupportedModelError(
            f"attention at {path} adds a float bias that isn't a causal mask or "
            "an ALiBi bias (slope * (key - query) per head)"
        )
    return tuple(float(v) for v in slopes.expand(heads))


# ── Static tables ────────────────────────────────────────────────────────────

def _find_tables(nodes, live, static, infos, values, short_values, pair, internal, length):
    tables = []
    users: dict[str, list] = {}
    for node in nodes:
        if node.name not in live or node.name in static:
            continue
        for arg in node.all_input_nodes:
            users.setdefault(arg.name, []).append(node)
    for node in nodes:
        if node.name not in live or node.name not in static or node.op == "output":
            continue
        info = infos[node.name]
        consumers = [
            user for user in users.get(node.name, []) if user.name not in internal
        ]
        if not info.sequence_axes or not consumers:
            continue
        if len(info.sequence_axes) != 1 or info.sequence_axes[0].stride != 1:
            raise _unsupported_operation(
                node,
                f"a static tensor of shape {info.shape} is used outside attention "
                "and has more than one growing axis or merges the sequence "
                "with other axes",
            )
        value = values.get(node.name)
        short_value = short_values.get(pair[node.name].name)
        if value is None or short_value is None:
            raise _unsupported_operation(
                node,
                "a sequence-length-dependent constant could not be evaluated at "
                "export time",
            )
        axis = info.sequence_axes[0].axis
        prefix = value.narrow(axis, 0, length - 1)
        close = (
            torch.allclose(prefix, short_value, rtol=1e-5, atol=1e-6)
            if prefix.dtype.is_floating_point else torch.equal(prefix, short_value)
        )
        if prefix.shape != short_value.shape or not close:
            raise _unsupported_operation(
                node,
                f"the table's first {length - 1} rows at {length} tokens differ "
                f"from the table at {length - 1} tokens (axis {axis}); rows must "
                "not depend on the sequence length",
            )
        operations = {_op_name(user) for user in consumers}
        kind = (
            "additive" if "add" in operations
            else "multiplicative" if "mul" in operations else "other"
        )
        tables.append(SequenceTable(
            node=node.name,
            axis=axis,
            shape=tuple(value.shape),
            dtype=value.dtype,
            value=value,
            consumers=tuple(user.name for user in consumers),
            kind=kind,
            module_path=info.module_path,
        ))
    return tables



# ── Step module ──────────────────────────────────────────────────────────────

class SequenceStepModule(torch.fx.GraphModule):
    """The one-token step of a stateless causal model.

    ``forward(token)`` takes a ``[1, 1]`` token tensor and returns the model's
    output leaves with one row on each sequence axis. Buffers: ``position`` (a
    long scalar, the number of tokens consumed), ``k_cache_{n}`` and
    ``v_cache_{n}`` (``n_kv_heads`` heads, ``max_context`` rows) for attention
    block ``n``, and the position tables ``table_{i}``. Stepping past
    ``max_context`` raises ``RuntimeError``.
    """

    def __init__(self, root, graph):
        super().__init__(root, graph, "SequenceStepModule")

    def cache_buffers(self) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        """``(block index, k_cache, v_cache)`` per attention block, in order."""
        layers = sorted(
            int(name[len("k_cache_"):]) for name, _ in self.named_buffers()
            if name.startswith("k_cache_")
        )
        return [
            (str(layer), getattr(self, f"k_cache_{layer}"), getattr(self, f"v_cache_{layer}"))
            for layer in layers
        ]

    @property
    def max_context(self) -> int:
        caches = self.cache_buffers()
        return int(caches[0][1].shape[-2]) if caches else 0

    def reset(self) -> None:
        """Forget every token: position 0 and empty caches."""
        with torch.no_grad():
            self.position.zero_()
            for _, k_cache, v_cache in self.cache_buffers():
                k_cache.zero_()
                v_cache.zero_()


def _replace_at(value, path, new):
    """``value`` with the element at ``path`` replaced (tuples and lists rebuilt)."""
    if not path:
        return new
    head, rest = path[0], path[1:]
    if isinstance(value, dict):
        return {**value, head: _replace_at(value[head], rest, new)}
    items = list(value)
    items[head] = _replace_at(items[head], rest, new)
    return type(value)(items)


def build_step_module(
    adapted: nn.Module, plan: SequencePlan,
) -> SequenceStepModule:
    """Rewrite the ``max_context``-token graph of ``plan`` into a one-token step.

    Literals that depend on the length take their one-row value, every
    sequence table becomes ``cattorch::sequence_row`` over a buffer, and every
    attention block becomes ``cattorch::cached_attention`` over per-layer K/V
    cache buffers. Nodes left without users (masks, ``arange``) are removed.
    ``adapted`` is the model ``plan`` was made from. It is not used by the
    rewrite itself, which works on ``plan.long``.
    """
    long = plan.long
    max_context = plan.max_context
    old_nodes = {node.name: node for node in long.graph.nodes}
    root = nn.Module()
    graph = torch.fx.Graph()
    env: dict[str, torch.fx.Node] = {}
    removed: dict[str, str] = {}

    def lookup(node):
        if node.name not in env:
            owner = removed.get(node.name)
            raise UnsupportedModelError(
                f"{_where(node)}: the value is used outside the attention block "
                f"{owner} it belongs to, so the block cannot be replaced by a cached step"
            ) if owner else UnsupportedModelError(
                f"{_where(node)}: the node was not carried into the step graph"
            )
        return env[node.name]

    def copy_meta(new, old):
        new.meta = {
            key: value for key, value in old.meta.items()
            if key not in {"val", "tensor_meta"}
        }
        return new

    # Buffers and the position scalar.
    root.register_buffer("position", torch.zeros((), dtype=torch.long))
    position_node = None  # created after the placeholder, below

    blocks = {block.output: block for block in plan.attention}
    internal: dict[str, str] = {}
    for block in plan.attention:
        for name in block.internal:
            internal.setdefault(name, block.module_path or f"#{block.index}")
    tables = {table.node: table for table in plan.tables}
    table_buffers: list[tuple[torch.Tensor, str]] = []

    def table_buffer(table: SequenceTable) -> str:
        for value, name in table_buffers:
            if (
                value.dtype == table.value.dtype
                and value.shape == table.value.shape
                and torch.equal(value, table.value)
            ):
                return name
        name = f"table_{len(table_buffers)}"
        table_buffers.append((table.value, name))
        root.register_buffer(name, table.value.detach().clone())
        return name

    diffs: dict[str, list[LiteralDiff]] = {}
    for diff in plan.literal_diffs:
        diffs.setdefault(diff.node, []).append(diff)

    for node in long.graph.nodes:
        if node.op == "output":
            continue
        if node.name not in plan.nodes:
            continue
        if node.op == "placeholder":
            if node.name != plan.token_input:
                raise UnsupportedModelError(
                    f"placeholder {node.name!r} is not the token input"
                )
            env[node.name] = copy_meta(graph.placeholder("token"), node)
            position_node = graph.get_attr("position")
            continue
        if node.name in blocks:
            block = blocks[node.name]
            key_info = old_nodes[block.key_source].meta["val"]
            value_info = old_nodes[block.value_source].meta["val"]
            for kind, source in (("k", key_info), ("v", value_info)):
                shape = list(source.shape)
                shape[-2] = max_context
                root.register_buffer(
                    f"{kind}_cache_{block.index}",
                    torch.zeros(shape, dtype=source.dtype),
                )
            per_head = []
            for kind, numbers in (("slopes", block.slopes), ("sinks", block.sinks)):
                if numbers is None:
                    per_head.append(None)
                    continue
                root.register_buffer(
                    f"{kind}_{block.index}", torch.tensor(numbers, dtype=torch.float32),
                )
                per_head.append(graph.get_attr(f"{kind}_{block.index}"))
            new = graph.call_function(
                torch.ops.cattorch.cached_attention.default,
                (
                    lookup(old_nodes[block.query]),
                    lookup(old_nodes[block.key_source]),
                    lookup(old_nodes[block.value_source]),
                    graph.get_attr(f"k_cache_{block.index}"),
                    graph.get_attr(f"v_cache_{block.index}"),
                    position_node,
                    float(block.scale),
                    int(block.kv_groups),
                    int(block.window),
                    *per_head,
                ),
            )
            env[node.name] = copy_meta(new, node)
            continue
        if node.name in internal:
            removed[node.name] = internal[node.name]
            continue
        if node.name in tables:
            table = tables[node.name]
            new = graph.call_function(
                torch.ops.cattorch.sequence_row.default,
                (graph.get_attr(table_buffer(table)), position_node, table.axis),
            )
            env[node.name] = copy_meta(new, node)
            continue
        if node.op == "get_attr":
            value = long.state_inputs.get(node.name)
            if value is None:
                raise UnsupportedModelError(f"{_where(node)}: unknown constant {node.name!r}")
            root.register_buffer(node.name, value.detach())
            env[node.name] = copy_meta(graph.get_attr(node.name), node)
            continue
        args = torch.fx.map_arg(node.args, lookup)
        kwargs = torch.fx.map_arg(node.kwargs, lookup)
        for diff in diffs.get(node.name, ()):
            new_value = diff.value_at(1, max_context)
            if diff.location[0] == "args":
                args = _replace_at(args, diff.location[1:], new_value)
            else:
                kwargs = _replace_at(kwargs, diff.location[1:], new_value)
        new = graph.create_node(node.op, node.target, tuple(args), dict(kwargs), node.name)
        env[node.name] = copy_meta(new, node)

    output = next(node for node in long.graph.nodes if node.op == "output")
    if position_node is None:
        raise UnsupportedModelError("the graph has no token input")
    advance = graph.call_function(
        torch.ops.cattorch.advance_position.default, (position_node,),
    )
    graph.output(torch.fx.map_arg(output.args[0], lookup))

    # Remove everything the rewrite left without users.
    for node in reversed(list(graph.nodes)):
        if node.op in {"output", "placeholder"} or node is advance:
            continue
        if not node.users:
            graph.erase_node(node)
    # The advance must come after every use of position, i.e. right before the
    # output node; node order is already that.
    graph.lint()
    step = SequenceStepModule(root, graph)
    step.eval()
    return step


# ── Equivalence ──────────────────────────────────────────────────────────────

def _vocabulary_size(plan: SequencePlan) -> int:
    """Rows of the embedding table indexed by the token input."""
    graph = plan.long.graph
    frontier = {plan.token_input}
    for node in graph.nodes:
        if node.op != "call_function" or node.name not in plan.nodes:
            continue
        name = _op_name(node)
        if name == "embedding" and len(node.args) > 1 and node.args[1].name in frontier:
            weight = node.args[0]
            shape = plan.nodes[weight.name].shape if weight.name in plan.nodes else None
            if shape is None and weight.name in plan.long.state_inputs:
                shape = tuple(plan.long.state_inputs[weight.name].shape)
            if shape:
                return int(shape[0])
        elif name in {"to", "_to_copy", "view", "reshape", "_unsafe_view", "clone",
                      "contiguous", "alias", "detach", "unsqueeze", "squeeze"} and (
            node.args and getattr(node.args[0], "name", None) in frontier
        ):
            frontier.add(node.name)
    raise UnsupportedModelError(
        "cannot find the vocabulary size: the token input is not looked up in an "
        "embedding table; automatic generation needs token ids"
    )


def check_step_equivalence(
    adapted: nn.Module,
    step: SequenceStepModule,
    plan: SequencePlan,
    *,
    seed: int = 0,
    rtol: float = 1e-4,
) -> None:
    """Run the step module token by token against the stateless model.

    Every output row must match ``adapted(tokens)`` within ``rtol`` (relative to
    the largest value in the row). Raises :class:`UnsupportedModelError`
    otherwise. The step module is reset before and after.
    """
    length = min(plan.max_context, 32)
    generator = torch.Generator().manual_seed(seed)
    vocabulary = _vocabulary_size(plan)
    tokens = torch.randint(0, vocabulary, (1, length), generator=generator)
    names = plan.long.user_output_names
    with torch.no_grad():
        reference = tensor_output_leaves(adapted(tokens), warn=False)
        # Shorter prompts too: a model whose positions depend on the prompt
        # length, such as dynamic NTK RoPE, matches only at the full length.
        prefixes = {
            count: tensor_output_leaves(adapted(tokens[:, :count]), warn=False)
            for count in sorted({1, length // 2, length - 1}) if 0 < count < length
        }
        step.reset()
        try:
            rows = [step(tokens[:, i:i + 1]) for i in range(length)]
        finally:
            step.reset()
    for name, axis in plan.outputs:
        leaf = names.index(name)
        for row in range(length):
            expected = reference[leaf].narrow(axis.axis, row, 1)
            actual = rows[row][leaf]
            if actual.shape != expected.shape:
                raise UnsupportedModelError(
                    f"the cached step gives output {leaf} the shape "
                    f"{tuple(actual.shape)} but the model gives {tuple(expected.shape)} "
                    f"for token {row}"
                )
            # Equal values, such as -inf for a banned token, match exactly.
            same = actual.double() == expected.double()
            error = torch.where(same, 0.0, (actual.double() - expected.double()).abs())
            largest = float(error.max()) if error.numel() else 0.0
            finite = expected.double()[torch.isfinite(expected)].abs()
            scale = float(finite.max()) if finite.numel() else 0.0
            if not bool(torch.isfinite(actual).all() == torch.isfinite(expected).all()) \
                    or largest > rtol * max(scale, 1e-3) or largest != largest:
                overall = max(
                    float((rows[r][leaf].double() - reference[leaf].narrow(axis.axis, r, 1).double()).abs().max())
                    for r in range(length)
                )
                raise UnsupportedModelError(
                    f"the cached step disagrees with the model at output row {row} "
                    f"(output {leaf}): max error {largest:.3g} against values up to "
                    f"{scale:.3g}, tolerance {rtol:g} relative; max error over all "
                    f"rows {overall:.3g}. The rewrite assumed the model computes each "
                    "token from its own row and earlier tokens only"
                )

    # The cached step matches the full prompt; check it is the model's answer
    # for shorter prompts too.
    for count, shorter in prefixes.items():
        for name, axis in plan.outputs:
            leaf = names.index(name)
            expected = shorter[leaf].narrow(axis.axis, count - 1, 1)
            actual = rows[count - 1][leaf]
            scale = float(expected.double().abs().max()) if expected.numel() else 0.0
            if actual.shape != expected.shape or not torch.allclose(
                actual.double(), expected.double(), rtol=0, atol=rtol * max(scale, 1e-3),
                equal_nan=True,
            ):
                raise UnsupportedModelError(
                    f"the model's output for token {count} changes when more tokens "
                    "follow it, so it can't be generated one token at a time; for "
                    "example, its positions are scaled by the prompt length"
                )


def build_generation_step(
    model: nn.Module, max_context: int, *, example_kwargs: dict | None = None,
    prepared: bool = False, registry=None,
) -> SequenceStepModule:
    """Prepare, analyse, rewrite and verify a stateless ``forward(tokens)`` model.

    ``model`` should be in evaluation mode. ``prepared`` says it is already the
    export copy (adapters applied), as when ``transpile`` has also quantized it;
    ``registry`` is the one that made that copy. Returns the verified, reset
    step module.
    """
    from cattorch.experts import _expert_family_context
    from cattorch.operator_registry import default_registry

    registry = registry or default_registry().clone()
    adapted = model if prepared else prepare_sequence_model(model, registry)
    # Tracing and the equivalence check both run the copy's expert families.
    with _expert_family_context(registry.expert_families):
        plan = analyze_sequence(adapted, max_context, example_kwargs=example_kwargs)
        step = build_step_module(adapted, plan)
        check_step_equivalence(adapted, step, plan)
    return step


class MethodLeaves(MethodModule):
    """``MethodModule`` that returns its output's tensor leaves as a tuple.

    The generation pipeline treats a model as ``forward(tokens) -> leaves``;
    this gives a named method (and a Hugging Face output object) that shape.
    """

    def forward(self, *args):
        return tuple(tensor_output_leaves(super().forward(*args), warn=False))


class GenerationStepModule(nn.Module):
    """What ``transpile`` lowers: a step module that returns only the logits.

    ``forward(token)`` gives the first tensor leaf of the model's output (the
    logits of a Hugging Face model) flattened to one vocabulary row, so that
    leaf is the generation interface and nothing else is exported.
    """

    def __init__(self, step: SequenceStepModule):
        super().__init__()
        self.step = step

    def forward(self, token):
        return self.step(token)[0].reshape(-1)

    def static_tensors(self):
        """Parameters and buffers that are stored: everything but caches and position."""
        yield from self.parameters()
        for name, buffer in self.named_buffers():
            leaf = name.rsplit(".", 1)[-1]
            if leaf != "position" and not leaf.startswith(("k_cache_", "v_cache_")):
                yield buffer

    def restore_module_stacks(self, graph: torch.fx.Graph) -> int:
        """Copy module stacks and fallback tags from the step graph onto a traced graph.

        Tracing a ``GraphModule`` records no module calls, so the traced graph
        has lost which layer each operation came from. The step graph kept it
        from the model's own trace. Operations are paired in order by target;
        an operation the tracer added (the final flatten) is skipped. Returns
        the number of nodes that received a stack.
        """
        source = [node for node in self.step.graph.nodes if node.op == "call_function"]
        restored = 0
        cursor = 0
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            for candidate in range(cursor, min(cursor + 4, len(source))):
                if source[candidate].target == node.target:
                    stack = source[candidate].meta.get("nn_module_stack")
                    if stack:
                        node.meta["nn_module_stack"] = dict(stack)
                        restored += 1
                    # Which operations ran as decompositions, for reporting.
                    if FALLBACK_META in source[candidate].meta:
                        node.meta[FALLBACK_META] = source[candidate].meta[FALLBACK_META]
                    cursor = candidate + 1
                    break
        return restored
