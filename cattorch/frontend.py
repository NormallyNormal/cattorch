"""Graph capture backends normalized into cattorch's internal contract."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Literal

import torch
import math
from torch.fx import Graph, GraphModule, Node
from torch.fx.experimental.proxy_tensor import make_fx
from torch.export import export
from torch.utils._pytree import tree_flatten
from torch._decomp import get_decompositions

from cattorch.errors import UnsupportedModelError


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


def _export_model(model, example_inputs):
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
            return export(model, example_inputs)
    return export(model, example_inputs)


def _capture_export(model, example_inputs) -> CapturedGraph:
    exported = _export_model(model, example_inputs)
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
    if attn_mask is not None:
        raise RuntimeError(
            "cattorch SDPA decomposition currently requires attn_mask=None; "
            "express a static mask with masked_fill outside SDPA"
        )
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
    probabilities = torch.ops.aten.softmax.int(scores, -1)
    return torch.ops.aten.matmul.default(probabilities, value)


def _fx_decompositions() -> dict:
    table = get_decompositions((torch.ops.aten.stack.default,))
    table.update({
        torch.ops.aten.repeat_interleave.self_int: _repeat_interleave_decomposition,
        torch.ops.aten.select.int: _select_decomposition,
        torch.ops.aten.scaled_dot_product_attention.default: _sdpa_decomposition,
    })
    return table


def _capture_fx(model, example_inputs) -> CapturedGraph:
    # make_fx only records module ownership when the traced callable exposes
    # the original module through this conventional attribute.
    def forward(*args):
        return model(*args)

    forward._orig_mod = model  # type: ignore[attr-defined]
    try:
        graph_module = make_fx(
            forward,
            decomposition_table=_fx_decompositions(),
            tracing_mode="real",
            pre_dispatch=True,
            record_module_stack=True,
            _error_on_data_dependent_ops=True,
        )(*example_inputs)
    except Exception as exc:
        raise UnsupportedModelError(
            "the FX frontend could not trace this model "
            f"({type(exc).__name__}: {exc}); a common cause is control flow that "
            "depends on tensor values or converting tensors to Python numbers, see "
            "https://github.com/NormallyNormal/cattorch/blob/main/docs/"
            "troubleshooting.md#keep-one-inference-return-path"
        ) from exc

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


def capture_model(model, example_inputs, *, frontend: Frontend = "fx") -> CapturedGraph:
    """Capture ``model`` and normalize frontend-specific state and outputs."""
    # ProxyTensor and TorchDynamo both manipulate process-global tracing state.
    # Concurrent captures can therefore corrupt one another even for unrelated
    # models. Serializing this short export-only phase makes the public API safe
    # to call from worker threads; RLock also permits internal nested captures.
    with _CAPTURE_LOCK:
        if frontend == "fx":
            return _capture_fx(model, example_inputs)
        if frontend == "export":
            return _capture_export(model, example_inputs)
        raise ValueError(f"frontend must be 'fx' or 'export', got {frontend!r}")


__all__ = ["CapturedGraph", "Frontend", "capture_model"]
