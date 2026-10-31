"""Graph capture backends normalized into cattorch's internal contract."""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Literal

import contextlib
import functools
import itertools
import logging
import math
import threading
import warnings

import torch
from torch.fx import Graph, GraphModule, Node
import torch.fx.experimental.proxy_tensor as proxy_tensor
from torch.fx.experimental.proxy_tensor import make_fx
from torch.export import export
from torch.utils._pytree import tree_flatten
from torch._decomp import core_aten_decompositions, get_decompositions

from cattorch.decompositions import POINTWISE_DECOMPOSITIONS, traced_from_input
from cattorch.errors import UnsupportedModelError


log = logging.getLogger(__name__)

Frontend = Literal["fx", "export"]
_CAPTURE_LOCK = RLock()


@dataclass(frozen=True)
class CapturedGraph:
    """Frontend-neutral graph and model interface consumed by analysis.

    Parameters and buffers are normalized into ``state_inputs`` regardless of
    whether the frontend represented them as placeholders or ``get_attr``
    nodes. ``user_output_names`` contains one entry per returned pytree leaf;
    ``None`` marks a non-node leaf so the common analyzer can report the same
    unsupported-output error for every frontend.
    """

    graph: Graph
    graph_module: GraphModule
    state_inputs: dict[str, torch.Tensor]
    user_input_names: frozenset[str]
    user_output_names: tuple[str | None, ...]
    frontend: Frontend


def flat_example_inputs(example_inputs, example_kwargs=None) -> tuple[torch.Tensor, ...]:
    """Return the tensors a traced graph takes as inputs, in placeholder order.

    Positional tensors come first, then tensor-valued keyword arguments in
    dict order. Non-tensor keyword arguments are constants of the trace.
    """
    return (*example_inputs, *(
        value for value in (example_kwargs or {}).values()
        if isinstance(value, torch.Tensor)
    ))


def tensor_output_leaves(value, *, warn: bool) -> list[torch.Tensor]:
    """Return the tensor leaves of a model output, in traversal order.

    Mappings (including Hugging Face ``ModelOutput``), tuples, lists, and
    namedtuples are searched. Every other leaf, such as ``None``, a Python
    number, or a ``DynamicCache``, is dropped; ``warn`` reports the dropped
    paths once. Raises when no tensor is found.
    """
    leaves: list[torch.Tensor] = []
    dropped: list[tuple[str, str]] = []

    def visit(item, path: str) -> None:
        if isinstance(item, torch.Tensor):
            leaves.append(item)
        elif isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f"{path}[{key!r}]")
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        else:
            dropped.append((path, type(item).__name__))

    visit(value, "output")
    if not leaves:
        raise UnsupportedModelError("cattorch requires at least one tensor output")
    if warn and dropped:
        warnings.warn(
            "ignoring non-tensor model outputs: "
            + ", ".join(f"{path} ({kind})" for path, kind in dropped),
            UserWarning,
            stacklevel=2,
        )
    return leaves


def _export_model(model, example_inputs, example_kwargs=None):
    """Export without inheriting old TorchDynamo's tiny process-wide limit."""
    config = getattr(getattr(torch, "_dynamo", None), "config", None)
    overrides = {}
    if config is not None:
        for name in (
            "cache_size_limit",
            "recompile_limit",
            "accumulated_cache_size_limit",
        ):
            value = getattr(config, name, None)
            if isinstance(value, int) and value < 1024:
                overrides[name] = 1024
    if overrides and hasattr(config, "patch"):
        with config.patch(**overrides):
            return export(model, example_inputs, example_kwargs or None)
    return export(model, example_inputs, example_kwargs or None)


def _capture_export(model, example_inputs, example_kwargs=None) -> CapturedGraph:
    try:
        exported = _export_model(model, example_inputs, example_kwargs)
    except Exception as exc:
        raise UnsupportedModelError(
            f"torch.export could not trace this model ({type(exc).__name__}: "
            f"{str(exc).splitlines()[0] if str(exc) else ''}); a common cause is "
            "control flow that depends on tensor values. The default frontend='fx' "
            "supports more models"
        ) from exc
    symbolic = (torch.SymInt, torch.SymFloat, torch.SymBool)
    if any(isinstance(node.meta.get("val"), symbolic) for node in exported.graph.nodes):
        raise UnsupportedModelError(
            "torch.export traced a value read from a tensor, such as .item(); "
            "Scratch can't use it as a number at export. Keep the computation in "
            "tensor operations"
        )
    state_inputs: dict[str, torch.Tensor] = {}
    user_inputs: set[str] = set()
    constants = getattr(exported, "constants", {})
    for spec in exported.graph_signature.input_specs:
        argument_name = getattr(spec.arg, "name", None)
        if argument_name is None:
            continue
        kind = getattr(spec.kind, "name", str(spec.kind).rsplit(".", 1)[-1])
        if kind == "USER_INPUT":
            user_inputs.add(argument_name)
            continue
        target = spec.target
        if target in exported.state_dict:
            state_inputs[argument_name] = exported.state_dict[target]
        elif target in constants:
            state_inputs[argument_name] = constants[target]

    user_outputs = tuple(
        getattr(spec.arg, "name", None)
        for spec in exported.graph_signature.output_specs
        if getattr(spec.kind, "name", str(spec.kind).rsplit(".", 1)[-1])
        == "USER_OUTPUT"
    )
    return CapturedGraph(
        graph=exported.graph,
        graph_module=exported.graph_module,
        state_inputs=state_inputs,
        user_input_names=frozenset(user_inputs),
        user_output_names=user_outputs,
        frontend="export",
    )


def _resolve_attr(module: GraphModule, target: str):
    value = module
    for component in target.split("."):
        value = getattr(value, component)
    return value


def _repeat_interleave_decomposition(value, repeats, dim=None, *, output_size=None):
    if not isinstance(repeats, int) or repeats < 1:
        raise RuntimeError("cattorch repeat_interleave requires a positive static repeat")
    if dim is None:
        value = value.reshape(-1)
        dim = 0
    dim %= value.ndim
    pieces = []
    for index in range(value.shape[dim]):
        piece = torch.ops.aten.slice.Tensor(value, dim, index, index + 1)
        pieces.extend((piece,) * repeats)
    return torch.ops.aten.cat.default(pieces, dim)


def _select_decomposition(value, dim, index):
    if not -value.ndim <= dim < value.ndim:
        raise IndexError(f"select dimension {dim} is out of range for rank {value.ndim}")
    dim %= value.ndim
    size = value.shape[dim]
    if not -size <= index < size:
        raise IndexError(f"select index {index} is out of range for dimension of size {size}")
    index = index % size
    selected = torch.ops.aten.slice.Tensor(value, dim, index, index + 1)
    shape = tuple(value.shape[:dim]) + tuple(value.shape[dim + 1:])
    return torch.ops.aten.view.default(selected, shape)


def _sdpa_decomposition(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    enable_gqa=False,
):
    """Fixed-shape math SDPA using operations with proven Scratch kernels."""
    if dropout_p != 0:
        raise RuntimeError("cattorch SDPA decomposition requires dropout_p=0")
    if enable_gqa:
        repeats = query.shape[-3] // key.shape[-3]
        key = _repeat_interleave_decomposition(key, repeats, -3)
        value = _repeat_interleave_decomposition(value, repeats, -3)
    factor = (1.0 / math.sqrt(query.shape[-1])) if scale is None else scale
    scores = torch.ops.aten.matmul.default(
        query, torch.ops.aten.transpose.int(key, -2, -1),
    )
    scores = torch.ops.aten.mul.Tensor(scores, factor)
    if is_causal:
        rows, columns = scores.shape[-2:]
        mask = torch.tensor(
            [[column > row for column in range(columns)] for row in range(rows)],
            device=query.device,
            dtype=torch.bool,
        )
        scores = torch.ops.aten.masked_fill.Scalar(scores, mask, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = torch.ops.aten.masked_fill.Scalar(
                scores, torch.ops.aten.logical_not.default(attn_mask), float("-inf"),
            )
        else:
            scores = torch.ops.aten.add.Tensor(scores, attn_mask)
    probabilities = torch.ops.aten.softmax.int(scores, -1)
    return torch.ops.aten.matmul.default(probabilities, value)


def _extreme_with_index(extreme, argument):
    def decomposition(value, dim, keepdim=False):
        return (
            extreme(value, [dim], keepdim),
            argument(value, dim, keepdim),
        )
    return decomposition


def _prims_var(value, dims, correction, *, output_dtype=None):
    dims = list(range(value.ndim)) if dims is None else list(dims)
    count = 1
    for dim in dims:
        count *= value.shape[dim]
    centered = torch.ops.aten.sub.Tensor(
        value, torch.ops.aten.mean.dim(value, dims, True),
    )
    squares = torch.ops.aten.mul.Tensor(centered, centered)
    total = torch.ops.aten.sum.dim_IntList(squares, dims)
    return torch.ops.aten.div.Tensor(total, max(count - correction, 0))


def _prims_broadcast_in_dim(value, shape, broadcast_dimensions):
    view_shape = [1] * len(shape)
    for source, target in enumerate(broadcast_dimensions):
        view_shape[target] = value.shape[source]
    return torch.ops.aten.expand.default(
        torch.ops.aten.reshape.default(value, view_shape), list(shape),
    )


def _prims_prod(value, dims, *, output_dtype=None):
    for dim in sorted((dim % value.ndim for dim in dims), reverse=True):
        value = torch.ops.aten.prod.dim_int(value, dim)
    return value


# Some PyTorch decompositions are written against primitives. Map the ones
# that reach models back to ATen operations cattorch lowers.
_PRIMITIVE_DECOMPOSITIONS = {
    "var": _prims_var,
    "broadcast_in_dim": _prims_broadcast_in_dim,
    "sum": lambda value, dims, *, output_dtype=None: torch.ops.aten.sum.dim_IntList(
        value, list(dims) if dims is not None else None,
    ),
    "log1p": lambda value: torch.ops.aten.log.default(
        torch.ops.aten.add.Tensor(value, 1),
    ),
    "expm1": lambda value: torch.ops.aten.sub.Tensor(
        torch.ops.aten.exp.default(value), 1,
    ),
    "signbit": lambda value: torch.ops.aten.lt.Scalar(value, 0),
    "prod": _prims_prod,
}


_aten = torch.ops.aten


def _addmm(bias, left, right, *, beta=1, alpha=1):
    """GPT-2's ``Conv1D``: ``bias + left @ right`` is a linear layer.

    Other calls, with scaling or a bias that isn't one value per output
    column, use PyTorch's decomposition.
    """
    if beta != 1 or alpha != 1 or bias.shape != (right.shape[-1],):
        generic = _available_decompositions()[_aten.addmm.default]
        return _recorded_fallback("aten.addmm.default", generic)(
            bias, left, right, beta=beta, alpha=alpha,
        )
    return _aten.linear.default(left, _aten.t.default(right), bias)


# Pointwise operations whose PyTorch decompositions end in primitives
# cattorch can't lower, written with ones it can.
_POINTWISE_DECOMPOSITIONS = {**POINTWISE_DECOMPOSITIONS, _aten.addmm.default: _addmm}


def _topk_decomposition(value, k, dim=-1, largest=True, sorted=True):
    # Scratch always produces sorted results; unsorted order is unspecified.
    return (
        torch.ops.cattorch.topk_values(value, k, dim, largest),
        torch.ops.cattorch.topk_indices(value, k, dim, largest),
    )


def _static_scalar_decomposition(operator):
    """Evaluate ``.item()`` on values that don't depend on the model inputs.

    Hugging Face mask construction branches on tensors built from ``arange``.
    Those are fixed for the traced shapes, so the Python value is specialised.
    Values reachable from an input placeholder return ``NotImplemented`` and
    keep the data-dependent error.
    """
    def decomposition(value, *args, **kwargs):
        if traced_from_input(value) is not False:
            return NotImplemented
        with proxy_tensor.disable_proxy_modes_tracing():
            result = operator(value, *args, **kwargs)
        log.info("specialised %s on input-independent value: %r", operator, result)
        return result
    return decomposition


def _resolve_operator(name: str):
    """Return the ``OpOverload`` for a dispatcher name like ``aten::outer``."""
    qualified, _, overload = name.partition(".")
    packet_name = qualified.split("::", 1)[1]
    try:
        packet = getattr(torch.ops.aten, packet_name)
        return getattr(packet, overload or "default")
    except (AttributeError, RuntimeError):
        return None


@functools.lru_cache(maxsize=1)
def _available_decompositions() -> dict:
    """Every ATen decomposition PyTorch provides, by operator overload.

    Registered decompositions come first. Composite operations, which PyTorch
    implements in terms of other operations but keeps whole when capturing
    before autograd, decompose through ``OpOverload.decompose``.
    """
    table: dict[torch._ops.OpOverload, Callable] = {}
    for source in (core_aten_decompositions(), torch._decomp.decomposition_table):
        for operator, function in source.items():
            if isinstance(operator, torch._ops.OpOverload):
                table.setdefault(operator, function)
    for name in torch._C._dispatch_get_all_op_names():
        if not name.startswith("aten::") or not torch._C._dispatch_has_kernel_for_dispatch_key(
            name, "CompositeImplicitAutograd",
        ):
            continue
        operator = _resolve_operator(name)
        if operator is not None and operator not in table:
            table[operator] = operator.decompose
    return table


class _TensorView:
    """Minimal ``fx.Node`` stand-in so validators can inspect live arguments."""

    name = "argument"

    def __init__(self, value):
        self.meta = {"val": value}


class _CallView:
    def __init__(self, operator, args, kwargs):
        self.target = operator
        self.args = tuple(
            _TensorView(value) if isinstance(value, torch.Tensor) else value
            for value in args
        )
        self.kwargs = kwargs


# Operations an export's own kernels lower, so tracing keeps them whole.
_KEPT_OPERATIONS: ContextVar[frozenset[str]] = ContextVar(
    "cattorch_kept_operations", default=frozenset(),
)


@contextlib.contextmanager
def keep_operations(names):
    """Don't decompose ``names`` in captures made inside this block."""
    token = _KEPT_OPERATIONS.set(_KEPT_OPERATIONS.get() | frozenset(names))
    try:
        yield
    finally:
        _KEPT_OPERATIONS.reset(token)


# ``node.meta`` key naming the operation whose generic PyTorch decomposition
# produced the node, and which call of it, as ``(operation, call)``.
FALLBACK_META = "cattorch_fallback"
_fallback_calls = itertools.count()


def _recorded_fallback(name: str, decomposition):
    """Tag the nodes ``decomposition`` traces with the operation it replaces.

    Lowering reports these as operations without a kernel of their own. An
    outer decomposition runs after its inner ones and overwrites their tags,
    so each node names the operation the model called.
    """
    def record(*args, **kwargs):
        mode = proxy_tensor.get_proxy_mode()
        if mode is None:
            return decomposition(*args, **kwargs)
        graph = mode.tracer.graph
        last = graph._root.prev
        result = decomposition(*args, **kwargs)
        call = (name, next(_fallback_calls))
        node = last.next
        while node is not graph._root:
            node.meta[FALLBACK_META] = call
            node = node.next
        return result
    return record


def _decompose_when_rejected(operator, validator, decomposition):
    from cattorch.fast import FastConfig

    def conditional(*args, **kwargs):
        if validator(_CallView(operator, args, kwargs), FastConfig(), False) is None:
            return NotImplemented
        return decomposition(*args, **kwargs)
    return conditional


# Operations with a kernel whose validator can reject some calls, and whose
# PyTorch decomposition leads to operations cattorch supports. Other rejected
# calls keep their operation so the error names it and gives the reason.
# ``matmul`` stays whole even when rejected: the cached-attention recognizer
# depends on seeing broadcasting matmuls as written.
_DECOMPOSE_WHEN_REJECTED = frozenset({"aten.batch_norm.default"})


def _fx_decompositions() -> dict:
    """Decompose every operation cattorch can't lower directly.

    Operations with a registered kernel, or that graph analysis handles itself,
    are kept whole. Everything else becomes PyTorch's decomposition, repeated
    until the graph only contains kept operations or ones with no
    decomposition.
    """
    from cattorch.graph import _graph_handled_operations
    from cattorch.operator_registry import default_registry

    registry = default_registry()
    kept = _graph_handled_operations()
    primitives = _primitive_decompositions()
    table = {}
    for operator, decomposition in _available_decompositions().items():
        name = str(operator)
        rule = registry.operation(name)
        # An operation whose reference is a same-named primitive with no
        # mapping back to ATen would only rename the unsupported operation.
        packet = operator.overloadpacket.__name__
        dead_end = hasattr(torch.ops.prims, packet) and packet not in primitives
        if rule is None and name not in kept and not dead_end:
            table[operator] = _recorded_fallback(name, decomposition)
        elif (
            rule is not None and rule.validator is not None
            and name in _DECOMPOSE_WHEN_REJECTED
        ):
            table[operator] = _recorded_fallback(name, _decompose_when_rejected(
                operator, rule.validator, decomposition,
            ))
    table.update(get_decompositions((torch.ops.aten.stack.default,)))
    table.update({
        torch.ops.aten.repeat_interleave.self_int: _repeat_interleave_decomposition,
        torch.ops.aten.select.int: _select_decomposition,
        torch.ops.aten.scaled_dot_product_attention.default: _sdpa_decomposition,
        torch.ops.aten.max.dim: _extreme_with_index(
            torch.ops.aten.amax.default, torch.ops.aten.argmax.default,
        ),
        torch.ops.aten.min.dim: _extreme_with_index(
            torch.ops.aten.amin.default, torch.ops.aten.argmin.default,
        ),
        torch.ops.aten.topk.default: _topk_decomposition,
    })
    table.update(_POINTWISE_DECOMPOSITIONS)
    for operator in (
        torch.ops.aten._local_scalar_dense.default, torch.ops.aten.item.default,
    ):
        table[operator] = _static_scalar_decomposition(operator)
    for name, decomposition in primitives.items():
        table[getattr(torch.ops.prims, name).default] = decomposition
    kept_by_export = _KEPT_OPERATIONS.get()
    # Operations declared with ``cattorch.experimental.op`` and no kernel in
    # this export run as their eager function.
    from cattorch.custom_ops import declared_without_kernel
    for operator, operation in declared_without_kernel(kept_by_export):
        table[operator] = _recorded_fallback(str(operator), operation.eager)
    return {
        operator: _without_cycles(operator, decomposition)
        for operator, decomposition in table.items()
        if str(operator) not in kept_by_export
    }


_decomposing = threading.local()


def _without_cycles(operator, decomposition):
    """Keep ``operator`` whole when its decomposition leads back to it.

    ``aten.sign`` decomposes to ``prims.sign``, which maps back to
    ``aten.sign``. The inner call is traced as written, so lowering names the
    operation instead of the trace failing with a recursion error.
    """
    def decompose(*args, **kwargs):
        active = _decomposing.__dict__.setdefault("operators", set())
        if operator in active:
            return NotImplemented
        active.add(operator)
        try:
            return decomposition(*args, **kwargs)
        finally:
            active.discard(operator)
    return decompose


# Elementwise primitives with an ATen operation of the same name and meaning.
_ELEMENTWISE_PRIMITIVES = (
    "add", "sub", "mul", "neg", "exp", "log", "sqrt", "rsqrt", "abs", "sin",
    "cos", "tanh", "erf", "floor", "ceil", "reciprocal", "maximum", "minimum",
    "where", "eq", "ne", "lt", "le", "gt", "ge", "bitwise_not", "sign",
)


def _primitive_decompositions() -> dict:
    aten = torch.ops.aten

    def same_named(name):
        packet = getattr(aten, name)
        overload = "Tensor" if "Tensor" in packet.overloads() else (
            "self" if "self" in packet.overloads() else "default"
        )
        return getattr(packet, overload)

    def divide(left, right):
        integers = all(
            not isinstance(value, torch.Tensor) or not value.is_floating_point()
            for value in (left, right)
        ) and not any(isinstance(value, float) for value in (left, right))
        if integers:
            return aten.div.Tensor_mode(left, right, rounding_mode="trunc")
        return aten.div.Tensor(left, right)

    def power(base, exponent):
        if isinstance(exponent, torch.Tensor):
            return aten.pow.Tensor_Tensor(base, exponent)
        return aten.pow.Tensor_Scalar(base, exponent)

    table = {}
    for name in _ELEMENTWISE_PRIMITIVES:
        try:
            table[name] = same_named(name)
        except AttributeError:
            continue
    table.update(_PRIMITIVE_DECOMPOSITIONS)
    table.update({
        "div": divide,
        "pow": power,
        "convert_element_type": lambda value, dtype: aten.to.dtype(value, dtype),
        "reshape": lambda value, shape: aten.reshape.default(value, list(shape)),
        "squeeze": lambda value, dims: aten.squeeze.dims(value, list(dims)),
        "transpose": lambda value, permutation: aten.permute.default(
            value, list(permutation),
        ),
        "view_of": lambda value: aten.alias.default(value),
        "amax": lambda value, dims, *, output_dtype=None: aten.amax.default(
            value, list(dims) if dims is not None else [],
        ),
        "amin": lambda value, dims, *, output_dtype=None: aten.amin.default(
            value, list(dims) if dims is not None else [],
        ),
    })
    return table


@contextlib.contextmanager
def _onednn_disabled():
    enabled = torch._C._get_mkldnn_enabled()
    torch._C._set_mkldnn_enabled(False)
    try:
        yield
    finally:
        torch._C._set_mkldnn_enabled(enabled)


_PYTHON_CONVERSIONS = ("tolist", "numpy", "__array__")


@contextlib.contextmanager
def _input_conversions_rejected():
    """Reject ``tolist()`` and NumPy conversions of values that depend on inputs.

    Unlike ``.item()``, they read the data without the tracer seeing it, so
    the example's values would be baked into the sprite as constants.
    """
    originals = {name: getattr(torch.Tensor, name) for name in _PYTHON_CONVERSIONS}

    def guarded(name):
        original = originals[name]

        @functools.wraps(original)
        def conversion(self, *args, **kwargs):
            if traced_from_input(self):
                raise UnsupportedModelError(
                    f"the model calls Tensor.{name}() on a value computed from its "
                    "inputs; Scratch would keep the example's values as constants. "
                    "Keep the computation in tensor operations"
                )
            return original(self, *args, **kwargs)
        return conversion

    for name in _PYTHON_CONVERSIONS:
        setattr(torch.Tensor, name, guarded(name))
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(torch.Tensor, name, original)


# Mode switches that make_fx records but that have no effect on a traced
# inference graph. Enter-autocast results are consumed by the matching exit.
_IGNORED_CALLS = tuple(
    target for target in (
        getattr(torch._C, "_set_grad_enabled", None),
        getattr(getattr(torch.amp, "autocast_mode", None), "_enter_autocast", None),
        getattr(getattr(torch.amp, "autocast_mode", None), "_exit_autocast", None),
    ) if target is not None
)


def _erase_mode_switches(graph: Graph) -> None:
    for node in reversed(list(graph.nodes)):
        if node.op == "call_function" and any(node.target is t for t in _IGNORED_CALLS):
            node.replace_all_uses_with(None)
            graph.erase_node(node)


def _returns_alias(node: Node) -> bool:
    """Whether ``node``'s result may share storage with one of its inputs."""
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return True
    return any(value.alias_info is not None for value in schema.returns)


_VIEW_SLICE = torch.ops.aten.slice.Tensor
_VIEW_SELECT = torch.ops.aten.select.int
# Views that keep every element and only change the shape. Assigning through
# one assigns the whole parent, viewed back to the parent's shape.
_SHAPE_VIEWS = {
    torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default,
    torch.ops.aten.unsqueeze.default, torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.dims, torch.ops.aten.squeeze.default,
}
# Uninitialized buffers that a model fills by slice assignment. They become
# zeros, so no unspecified values are stored in the sprite.
_EMPTY_AS_ZEROS = {
    torch.ops.aten.empty.memory_format: torch.ops.aten.zeros.default,
    torch.ops.aten.new_empty.default: torch.ops.aten.new_zeros.default,
    torch.ops.aten.empty_like.default: torch.ops.aten.zeros_like.default,
}


def _region_view(node: Node) -> tuple[int, int, int] | None:
    """``(dim, start, end)`` written through a step-one slice or a select.

    A shape-only view writes all of its parent and gives ``(-1, 0, 0)``.
    """
    parent = node.args[0]
    shape = parent.meta.get("val").shape if isinstance(parent, Node) else None
    if shape is None:
        return None
    if node.target in _SHAPE_VIEWS:
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor) and value.numel() == math.prod(shape):
            return -1, 0, 0
        return None
    if node.target is _VIEW_SLICE:
        given = tuple(node.args[1:])
        dim, start, end, step = given + (0, None, None, 1)[len(given):]
        dim = node.kwargs.get("dim", dim)
        start = node.kwargs.get("start", start)
        end = node.kwargs.get("end", end)
        if node.kwargs.get("step", step) != 1:
            return None
        dim %= len(shape)
        size = shape[dim]
        start = 0 if start is None else start
        end = size if end is None else end
        start = max(0, min(size, start + size if start < 0 else start))
        end = max(start, min(size, end + size if end < 0 else end))
        return dim, start, end
    if node.target is _VIEW_SELECT:
        dim, index = node.args[1:3]
        dim %= len(shape)
        index = index + shape[dim] if index < 0 else index
        return dim, index, index + 1
    return None


def _replace_intermediate_copies(graph: Graph) -> None:
    """Rewrite in-place updates of intermediate tensors as ordinary dataflow.

    The capture records ``x /= y`` on a computed tensor as ``div`` followed by
    ``copy_(x, quotient)``, and ``x[..., :4] = y`` as ``copy_`` into a slice
    view of ``x``. When ``x`` is a fresh tensor rather than a view, and no
    other view of it taken before the copy is read afterwards, the copy
    becomes a new tensor: the source itself, or ``x`` with the sliced region
    replaced (a ``cat`` of the untouched parts and the source). Later reads of
    ``x`` take that tensor. Copies into inputs, state, strided slices, or other
    views are left for the compiler to reject.
    """
    def read_after(node: Node, position: int) -> bool:
        return any(
            order[user] > position or (_returns_alias(user) and read_after(user, position))
            for user in node.users
        )

    for copy in [node for node in graph.nodes if node.target is torch.ops.aten.copy_.default]:
        order = {node: index for index, node in enumerate(graph.nodes)}
        target, source = copy.args[:2]
        if (
            not isinstance(target, Node) or not isinstance(source, Node)
            or target.op != "call_function" or copy.kwargs.get("non_blocking", False)
        ):
            continue
        # Walk slice and select views up to the tensor that owns the storage.
        chain: list[tuple[Node, tuple[int, int, int], bool]] = []
        base = target
        while _returns_alias(base):
            region = _region_view(base) if base.op == "call_function" else None
            if region is None:
                break
            chain.append((base, region, base.target is _VIEW_SELECT))
            base = base.args[0]
        if base.op != "call_function" or _returns_alias(base):
            continue
        if chain and copy.users:
            continue
        position = order[copy]
        chain_nodes = {node for node, _, _ in chain}
        if any(
            user is not copy and user not in chain_nodes and order[user] < position
            and _returns_alias(user) and read_after(user, position)
            for user in base.users
        ):
            continue
        expected, actual = target.meta.get("val"), source.meta.get("val")
        if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
            continue

        def call(operation, args, meta_value):
            node = graph.call_function(operation, args)
            node.meta.update(copy.meta)
            node.meta["val"] = meta_value
            return node

        with graph.inserting_before(copy):
            # ``copy_`` casts and broadcasts its source to the target.
            value = source
            if actual.dtype != expected.dtype:
                value = call(torch.ops.aten.to.dtype, (value, expected.dtype),
                             actual.to(expected.dtype))
            if tuple(actual.shape) != tuple(expected.shape):
                value = call(torch.ops.aten.expand.default, (value, list(expected.shape)),
                             expected)
            for view, (dim, start, end), selected in chain:
                parent = view.args[0]
                whole = parent.meta["val"]
                if dim < 0:
                    value = call(torch.ops.aten.view.default, (value, list(whole.shape)),
                                 whole)
                    continue
                if selected:
                    value = call(torch.ops.aten.unsqueeze.default, (value, dim),
                                 whole.narrow(dim, start, 1))
                pieces = []
                if start > 0:
                    pieces.append(call(_VIEW_SLICE, (parent, dim, 0, start),
                                       whole.narrow(dim, 0, start)))
                pieces.append(value)
                if end < whole.shape[dim]:
                    pieces.append(call(_VIEW_SLICE, (parent, dim, end, whole.shape[dim]),
                                       whole.narrow(dim, end, whole.shape[dim] - end)))
                if len(pieces) > 1:
                    value = call(torch.ops.aten.cat.default, (pieces, dim), whole)
        if chain and base.target in _EMPTY_AS_ZEROS:
            base.target = _EMPTY_AS_ZEROS[base.target]
            base.kwargs = {
                key: item for key, item in base.kwargs.items() if key != "memory_format"
            }
        for user in list(base.users):
            if user is not copy and user not in chain_nodes and order.get(user, -1) > position:
                user.replace_input_with(base, value)
        copy.replace_all_uses_with(value)
        graph.erase_node(copy)
        for view, _, _ in chain:
            if not view.users:
                graph.erase_node(view)


def _slice_cat_pieces(graph: Graph) -> None:
    """Replace a slice of a ``cat`` that starts at a piece boundary by the pieces.

    ``torch.cat([history, row])[len(history):]`` is ``row`` whatever
    ``history``'s length, which is how append state commits its new rows.
    """
    for node in list(graph.nodes):
        if node.op != "call_function" or str(node.target) != "aten.slice.Tensor":
            continue
        source = node.args[0]
        if not isinstance(source, Node) or str(source.target) != "aten.cat.default":
            continue
        pieces = list(source.args[0])
        rank = len(source.meta["val"].shape)
        cat_dim = (source.args[1] if len(source.args) > 1 else 0) % rank
        dim = (node.args[1] if len(node.args) > 1 else 0) % rank
        start = node.args[2] if len(node.args) > 2 else 0
        end = node.args[3] if len(node.args) > 3 else None
        step = node.args[4] if len(node.args) > 4 else 1
        total = source.meta["val"].shape[cat_dim]
        if dim != cat_dim or step != 1 or not isinstance(start, int) or (
            end is not None and end < total
        ):
            continue
        offset = 0
        for index, piece in enumerate(pieces):
            if offset == start and 0 < index:
                kept = pieces[index:]
                with graph.inserting_before(node):
                    replacement = kept[0] if len(kept) == 1 else graph.call_function(
                        source.target, (kept, cat_dim),
                    )
                if replacement is not kept[0]:
                    replacement.meta = dict(node.meta)
                node.replace_all_uses_with(replacement)
                graph.erase_node(node)
                break
            offset += piece.meta["val"].shape[cat_dim]


def _capture_fx(model, example_inputs, example_kwargs=None) -> CapturedGraph:
    example_kwargs = dict(example_kwargs or {})
    tensor_names = [
        key for key, value in example_kwargs.items()
        if isinstance(value, torch.Tensor)
    ]
    positional_count = len(example_inputs)
    # Program entrypoints keep their explicit contract: every returned leaf
    # must be a tensor, so nothing is dropped for them.
    from cattorch.program_runtime import MethodModule
    keep_all_leaves = isinstance(model, MethodModule)

    # make_fx only records module ownership when the traced callable exposes
    # the original module through this conventional attribute.
    def forward(*flat):
        # Tensor keywords arrive after the positional tensors; every other
        # keyword is bound as a constant of the trace.
        kwargs = dict(example_kwargs)
        kwargs.update(zip(tensor_names, flat[positional_count:]))
        output = model(*flat[:positional_count], **kwargs)
        if keep_all_leaves:
            return output
        return tuple(tensor_output_leaves(output, warn=True))

    forward._orig_mod = model  # type: ignore[attr-defined]
    try:
        # With oneDNN enabled, PyTorch's LSTM decomposition emits an
        # oneDNN-only operation instead of the portable cell arithmetic.
        with _onednn_disabled(), _input_conversions_rejected():
            graph_module = make_fx(
                forward,
                decomposition_table=_fx_decompositions(),
                tracing_mode="real",
                pre_dispatch=True,
                record_module_stack=True,
                _error_on_data_dependent_ops=True,
            )(*flat_example_inputs(example_inputs, example_kwargs))
    except UnsupportedModelError:
        raise
    except Exception as exc:
        raise UnsupportedModelError(
            "the FX frontend could not trace this model "
            f"({type(exc).__name__}: {exc}); a common cause is control flow that "
            "depends on tensor values or converting tensors to Python numbers, see "
            "https://github.com/NormallyNormal/cattorch/blob/main/docs/"
            "troubleshooting.md#keep-one-inference-return-path"
        ) from exc

    # A wrapper around an already-traced graph (automatic generation) knows
    # which modules its operations came from; the tracer saw only the graph.
    restore = getattr(model, "restore_module_stacks", None)
    if callable(restore):
        restore(graph_module.graph)

    _erase_mode_switches(graph_module.graph)
    _replace_intermediate_copies(graph_module.graph)
    _slice_cat_pieces(graph_module.graph)
    # Unused results, such as pooling indices nobody reads, would otherwise
    # still run in Scratch. Mutations are kept.
    graph_module.graph.eliminate_dead_code()
    graph_module.recompile()

    nodes = list(graph_module.graph.nodes)
    state_inputs: dict[str, torch.Tensor] = {}
    for node in nodes:
        if node.op != "get_attr":
            continue
        value = _resolve_attr(graph_module, str(node.target))
        if isinstance(value, torch.Tensor):
            state_inputs[node.name] = value
            # PyTorch 2.6 does not attach ``val``/``tensor_meta`` to make_fx
            # get_attr nodes, while newer releases do. The analyzer only
            # needs tensor shape/dtype, so normalize the stable information
            # from the resolved state tensor instead of version-branching.
            node.meta.setdefault("val", value)

    output = next((node for node in nodes if node.op == "output"), None)
    if output is None:
        output_names: tuple[str | None, ...] = ()
    else:
        value = output.args[0] if output.args else None
        leaves, _ = tree_flatten(value)
        output_names = tuple(leaf.name if isinstance(leaf, Node) else None for leaf in leaves)

    return CapturedGraph(
        graph=graph_module.graph,
        graph_module=graph_module,
        state_inputs=state_inputs,
        user_input_names=frozenset(
            node.name for node in nodes if node.op == "placeholder"
        ),
        user_output_names=output_names,
        frontend="fx",
    )


def capture_model(
    model, example_inputs, *, frontend: Frontend = "fx", example_kwargs=None,
) -> CapturedGraph:
    """Capture ``model`` and normalize frontend-specific state and outputs."""
    # ProxyTensor and TorchDynamo both manipulate process-global tracing state.
    # Concurrent captures can therefore corrupt one another even for unrelated
    # models. Serializing this short export-only phase makes the public API safe
    # to call from worker threads; RLock also permits internal nested captures.
    with _CAPTURE_LOCK:
        if frontend == "fx":
            return _capture_fx(model, example_inputs, example_kwargs)
        if frontend == "export":
            return _capture_export(model, example_inputs, example_kwargs)
        raise ValueError(f"frontend must be 'fx' or 'export', got {frontend!r}")


__all__ = [
    "CapturedGraph", "Frontend", "capture_model", "flat_example_inputs",
    "tensor_output_leaves",
]
