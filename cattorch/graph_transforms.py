"""Export-only PyTorch graph and module transforms."""

from __future__ import annotations

import copy
import logging

import torch
from torch.fx import Tracer


log = logging.getLogger(__name__)


def _set_submodule(
    model: torch.nn.Module,
    path: str,
    replacement: torch.nn.Module,
) -> None:
    parent_path, _, name = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    setattr(parent, name, replacement)


def fold_eval_batch_norms(
    model: torch.nn.Module, *, methods: tuple[str, ...] = ("forward",),
) -> torch.nn.Module:
    """Fold dataflow-connected eval Conv/Linear + BatchNorm pairs on a copy."""
    eligible = (
        (torch.nn.Conv1d, torch.nn.BatchNorm1d),
        (torch.nn.Conv2d, torch.nn.BatchNorm2d),
        (torch.nn.Conv3d, torch.nn.BatchNorm3d),
        (torch.nn.Linear, torch.nn.BatchNorm1d),
    )
    eligible_types = tuple(kind for pair in eligible for kind in pair)
    eligible_modules = {
        name: module
        for name, module in model.named_modules()
        if name and isinstance(module, eligible_types)
    }
    if not eligible_modules:
        return model

    try:
        graphs = []
        for method in dict.fromkeys(methods):
            tracer = Tracer()
            tracer.traced_func_name = method
            graphs.append(tracer.trace(model))
    except Exception:
        log.info("Skipping BatchNorm folding because FX tracing was not available")
        return model

    calls_by_target: dict[str, list] = {}
    attributes = set()
    for graph in graphs:
        for node in graph.nodes:
            if node.op == "call_module":
                calls_by_target.setdefault(str(node.target), []).append(node)
            elif node.op == "get_attr":
                attributes.add(str(node.target))

    pairs = []
    for batch_norm_path, batch_norm_calls in calls_by_target.items():
        batch_norm = eligible_modules.get(batch_norm_path)
        if (
            len(batch_norm_calls) != 1
            or not isinstance(
                batch_norm,
                (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
            )
            or batch_norm.training
        ):
            continue
        batch_norm_node = batch_norm_calls[0]
        if not batch_norm_node.args or not hasattr(batch_norm_node.args[0], "op"):
            continue
        producer = batch_norm_node.args[0]
        if producer.op != "call_module" or len(producer.users) != 1:
            continue
        producer_path = str(producer.target)
        layer = eligible_modules.get(producer_path)
        if (
            len(calls_by_target.get(producer_path, ())) != 1
            or layer is None
            or layer.training
        ):
            continue
        if not any(
            isinstance(layer, left) and isinstance(batch_norm, right)
            for left, right in eligible
        ):
            continue
        # A different method may consume a layer's parameters directly.
        if any(
            target.startswith((f"{producer_path}.", f"{batch_norm_path}."))
            for target in attributes
        ):
            continue
        pairs.append((producer_path, batch_norm_path))

    if not pairs:
        return model

    transformed = copy.deepcopy(model)
    from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval

    for layer_path, batch_norm_path in pairs:
        layer = transformed.get_submodule(layer_path)
        batch_norm = transformed.get_submodule(batch_norm_path)
        if isinstance(layer, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            folded = fuse_conv_bn_eval(layer, batch_norm)
        else:
            folded = fuse_linear_bn_eval(layer, batch_norm)
        _set_submodule(transformed, layer_path, folded)
        _set_submodule(transformed, batch_norm_path, torch.nn.Identity())
    return transformed


__all__ = ["fold_eval_batch_norms"]
