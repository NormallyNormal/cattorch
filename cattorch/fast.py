"""Configuration and export-time model transforms for ``optimization='fast'``."""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Mapping

import torch
import torch.nn as nn
from torch.fx import symbolic_trace


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FastLayerConfig:
    """Approximation policy for an eligible static-weight layer.

    ``rank`` is an explicit truncated-SVD rank. ``rank_ratio`` selects a
    fraction of the matrix's full rank. ``pruning`` removes that fraction of
    Scratch-friendly weight blocks by magnitude. Module overrides replace the
    global policy rather than merging with it.
    """

    pruning: float = 0.0
    rank: int | None = None
    rank_ratio: float | None = None

    def __post_init__(self):
        if (
            isinstance(self.pruning, bool)
            or not isinstance(self.pruning, (int, float))
            or not 0 <= self.pruning < 1
        ):
            raise ValueError("pruning must be in [0, 1)")
        if self.rank is not None and (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 1
        ):
            raise ValueError("rank must be a positive integer")
        if self.rank_ratio is not None and (
            isinstance(self.rank_ratio, bool)
            or not isinstance(self.rank_ratio, (int, float))
            or not 0 < self.rank_ratio <= 1
        ):
            raise ValueError("rank_ratio must be in (0, 1]")
        if self.rank is not None and self.rank_ratio is not None:
            raise ValueError("rank and rank_ratio are mutually exclusive")

    @property
    def enabled(self) -> bool:
        return self.pruning > 0 or self.rank is not None or self.rank_ratio is not None


@dataclass(frozen=True)
class FastConfig:
    """Controls approximate kernels and optional static-weight transforms."""

    activations: bool = True
    layer_norm: bool = True
    softmax: bool = True
    neuron_pruning: float = 0.0
    weights: FastLayerConfig = field(default_factory=FastLayerConfig)
    overrides: Mapping[str, FastLayerConfig] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("activations", "layer_norm", "softmax"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if (
            isinstance(self.neuron_pruning, bool)
            or not isinstance(self.neuron_pruning, (int, float))
            or not 0 <= self.neuron_pruning < 1
        ):
            raise ValueError("neuron_pruning must be in [0, 1)")
        if not isinstance(self.weights, FastLayerConfig):
            raise TypeError("weights must be a FastLayerConfig")
        normalized = dict(self.overrides)
        if any(not isinstance(name, str) or not name for name in normalized):
            raise ValueError("fast override names must be non-empty module names")
        if any(not isinstance(policy, FastLayerConfig) for policy in normalized.values()):
            raise TypeError("fast overrides must contain FastLayerConfig values")
        object.__setattr__(self, "overrides", normalized)

    @property
    def has_weight_transforms(self) -> bool:
        return (
            self.neuron_pruning > 0
            or self.weights.enabled
            or any(policy.enabled for policy in self.overrides.values())
        )

    def policy_for(self, module_name: str) -> FastLayerConfig:
        return self.overrides.get(module_name, self.weights)


def _selected_rank(weight: torch.Tensor, policy: FastLayerConfig) -> int | None:
    matrix = weight.flatten(1)
    full_rank = min(matrix.shape)
    if policy.rank is not None:
        rank = policy.rank
    elif policy.rank_ratio is not None:
        rank = max(1, math.floor(full_rank * policy.rank_ratio))
    else:
        return None
    if rank >= full_rank:
        return None
    return rank


def _factor_linear(module: nn.Linear, rank: int) -> nn.Sequential | None:
    original_macs = module.in_features * module.out_features
    factor_macs = rank * (module.in_features + module.out_features)
    if factor_macs >= original_macs:
        return None
    factory_kwargs = {
        "device": module.weight.device,
        "dtype": module.weight.dtype,
    }
    with torch.no_grad():
        u, singular, vh = torch.linalg.svd(module.weight.detach(), full_matrices=False)
        first = nn.Linear(module.in_features, rank, bias=False, **factory_kwargs)
        second = nn.Linear(
            rank, module.out_features, bias=module.bias is not None, **factory_kwargs,
        )
        first.weight.copy_(vh[:rank])
        second.weight.copy_(u[:, :rank] * singular[:rank])
        if module.bias is not None:
            second.bias.copy_(module.bias.detach())
    return nn.Sequential(first, second).train(module.training)


def _factor_conv(module: nn.Conv1d | nn.Conv2d, rank: int) -> nn.Sequential | None:
    if module.groups != 1:
        return None
    kernel_macs = math.prod(module.kernel_size)
    original_macs = module.out_channels * module.in_channels * kernel_macs
    factor_macs = rank * module.in_channels * kernel_macs + module.out_channels * rank
    if factor_macs >= original_macs:
        return None

    conv_type = type(module)
    one = (1,) * len(module.kernel_size)
    factory_kwargs = {
        "device": module.weight.device,
        "dtype": module.weight.dtype,
    }
    with torch.no_grad():
        matrix = module.weight.detach().flatten(1)
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        first = conv_type(
            module.in_channels, rank, module.kernel_size,
            stride=module.stride, padding=module.padding, dilation=module.dilation,
            bias=False, padding_mode=module.padding_mode, **factory_kwargs,
        )
        second = conv_type(
            rank, module.out_channels, one, bias=module.bias is not None,
            **factory_kwargs,
        )
        first.weight.copy_(vh[:rank].reshape_as(first.weight))
        second.weight.copy_((u[:, :rank] * singular[:rank]).reshape_as(second.weight))
        if module.bias is not None:
            second.bias.copy_(module.bias.detach())
    return nn.Sequential(first, second).train(module.training)


def _prune_linear_blocks(weight: torch.Tensor, fraction: float) -> torch.Tensor:
    """Prune contiguous groups of four inputs so Scratch can skip whole blocks."""
    result = weight.detach().clone()
    rows, columns = result.shape
    blocks = []
    for row in range(rows):
        for start in range(0, columns - columns % 4, 4):
            blocks.append((float(result[row, start:start + 4].abs().sum()), row, start))
    count = math.floor(len(blocks) * fraction)
    for _, row, start in sorted(blocks)[:count]:
        result[row, start:start + 4] = 0
    return result


def _prune_conv_blocks(weight: torch.Tensor, fraction: float) -> torch.Tensor:
    """Prune complete input-channel kernels for efficient Scratch skipping."""
    result = weight.detach().clone()
    blocks = [
        (float(result[output, channel].abs().sum()), output, channel)
        for output in range(result.shape[0])
        for channel in range(result.shape[1])
    ]
    count = math.floor(len(blocks) * fraction)
    for _, output, channel in sorted(blocks)[:count]:
        result[output, channel] = 0
    return result


def _prune_matmul_blocks(weight: torch.Tensor, fraction: float) -> torch.Tensor:
    """Prune four-wide reduction blocks from a ``[K, N]`` RHS matrix."""
    result = weight.detach().clone()
    inner, columns = result.shape
    blocks = []
    for column in range(columns):
        for start in range(0, inner - inner % 4, 4):
            blocks.append((float(result[start:start + 4, column].abs().sum()), start, column))
    count = math.floor(len(blocks) * fraction)
    for _, start, column in sorted(blocks)[:count]:
        result[start:start + 4, column] = 0
    return result


def prepare_fast_matmul_weights(
    weight: torch.Tensor,
    policy: FastLayerConfig,
) -> tuple[torch.Tensor, ...]:
    """Prepare one or two ``[K, N]`` matrices for a static-RHS matmul."""
    if weight.ndim != 2:
        return (weight,)
    inner, columns = weight.shape
    rank = _selected_rank(weight, policy)
    factors: tuple[torch.Tensor, ...]
    if rank is not None and rank * (inner + columns) < inner * columns:
        matrix = weight.detach()
        svd_input = (
            matrix.float()
            if matrix.dtype in {torch.float16, torch.bfloat16}
            else matrix
        )
        with torch.no_grad():
            u, singular, vh = torch.linalg.svd(svd_input, full_matrices=False)
            first = (u[:, :rank] * singular[:rank]).to(
                dtype=matrix.dtype, device=matrix.device,
            )
            second = vh[:rank].to(dtype=matrix.dtype, device=matrix.device)
        factors = (first, second)
    else:
        factors = (weight.detach().clone(),)
    if policy.pruning:
        factors = tuple(_prune_matmul_blocks(factor, policy.pruning) for factor in factors)
    return factors


def _apply_pruning(module: nn.Module, fraction: float) -> None:
    if not fraction:
        return
    for child in module.modules():
        if isinstance(child, nn.Linear):
            pruned = _prune_linear_blocks(child.weight, fraction)
        elif isinstance(child, (nn.Conv1d, nn.Conv2d)) and child.groups == 1:
            pruned = _prune_conv_blocks(child.weight, fraction)
        else:
            continue
        child.weight = nn.Parameter(pruned, requires_grad=child.weight.requires_grad)


def _prune_sequential_neurons(model: nn.Module, fraction: float) -> None:
    """Remove neurons from dataflow-connected Linear/activation/Linear MLPs."""
    if not fraction:
        return
    activations = (
        nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid,
        nn.ELU, nn.LeakyReLU, nn.Identity,
    )
    try:
        traced = symbolic_trace(model)
    except Exception:
        log.warning("Skipping whole-neuron pruning because FX tracing was not available")
        return

    modules = dict(model.named_modules())
    calls_by_target: dict[str, list] = {}
    for node in traced.graph.nodes:
        if node.op == "call_module":
            calls_by_target.setdefault(str(node.target), []).append(node)

    triples = []
    claimed: set[str] = set()
    for activation_path, calls in calls_by_target.items():
        activation = modules.get(activation_path)
        if len(calls) != 1 or not isinstance(activation, activations):
            continue
        activation_node = calls[0]
        if not activation_node.args or not hasattr(activation_node.args[0], "op"):
            continue
        up_node = activation_node.args[0]
        if up_node.op != "call_module" or len(up_node.users) != 1:
            continue
        users = tuple(activation_node.users)
        if len(users) != 1:
            continue
        down_node = users[0]
        if down_node.op != "call_module" or not down_node.args or down_node.args[0] is not activation_node:
            continue
        up_path, down_path = str(up_node.target), str(down_node.target)
        up, down = modules.get(up_path), modules.get(down_path)
        paths = {up_path, activation_path, down_path}
        if (
            len(paths) != 3
            or paths & claimed
            or len(calls_by_target.get(up_path, ())) != 1
            or len(calls_by_target.get(down_path, ())) != 1
            or not isinstance(up, nn.Linear)
            or not isinstance(down, nn.Linear)
            or up.out_features != down.in_features
            or up.out_features < 2
        ):
            continue
        triples.append((up_path, down_path))
        claimed.update(paths)

    for up_path, down_path in triples:
        up = model.get_submodule(up_path)
        down = model.get_submodule(down_path)
        keep_count = max(1, up.out_features - math.floor(up.out_features * fraction))
        if keep_count == up.out_features:
            continue
        importance = (
            up.weight.detach().abs().sum(dim=1)
            + down.weight.detach().abs().sum(dim=0)
        )
        keep = torch.topk(
            importance, keep_count, largest=True, sorted=False,
        ).indices.sort().values
        factory = {"device": up.weight.device, "dtype": up.weight.dtype}
        new_up = nn.Linear(
            up.in_features, keep_count, bias=up.bias is not None, **factory,
        ).train(up.training)
        new_down = nn.Linear(
            keep_count, down.out_features, bias=down.bias is not None,
            device=down.weight.device, dtype=down.weight.dtype,
        ).train(down.training)
        with torch.no_grad():
            new_up.weight.copy_(up.weight[keep])
            new_down.weight.copy_(down.weight[:, keep])
            if up.bias is not None:
                new_up.bias.copy_(up.bias[keep])
            if down.bias is not None:
                new_down.bias.copy_(down.bias)
        up_parent_path, _, up_name = up_path.rpartition(".")
        down_parent_path, _, down_name = down_path.rpartition(".")
        up_parent = model.get_submodule(up_parent_path) if up_parent_path else model
        down_parent = model.get_submodule(down_parent_path) if down_parent_path else model
        setattr(up_parent, up_name, new_up)
        setattr(down_parent, down_name, new_down)
        log.info(
            "Fast pruned MLP hidden width %d -> %d at %s/%s",
            up.out_features, keep_count, up_path, down_path,
        )


def prepare_fast_model(model: nn.Module, config: FastConfig) -> nn.Module:
    """Return an independently transformed model for fast export."""
    if not config.has_weight_transforms:
        return model

    transformed = copy.deepcopy(model)
    _prune_sequential_neurons(transformed, config.neuron_pruning)
    if isinstance(transformed, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        unknown = sorted(set(config.overrides) - {"<root>"})
        if unknown:
            raise ValueError(
                f"Unknown or ineligible fast module override(s): {', '.join(unknown)}"
            )
        policy = config.policy_for("<root>")
        rank = _selected_rank(transformed.weight, policy)
        replacement = None
        if rank is not None:
            replacement = (
                _factor_linear(transformed, rank)
                if isinstance(transformed, nn.Linear)
                else _factor_conv(transformed, rank)
            )
        target = replacement if replacement is not None else transformed
        _apply_pruning(target, policy.pruning)
        return target

    eligible = [
        (name, module)
        for name, module in transformed.named_modules()
        if name and isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d))
    ]
    known_names = {name for name, _ in eligible}
    unknown = sorted(set(config.overrides) - known_names)
    if unknown:
        raise ValueError(f"Unknown or ineligible fast module override(s): {', '.join(unknown)}")

    for name, module in eligible:
        policy = config.policy_for(name)
        if not policy.enabled:
            continue
        rank = _selected_rank(module.weight, policy)
        replacement = None
        if rank is not None:
            if isinstance(module, nn.Linear):
                replacement = _factor_linear(module, rank)
            else:
                replacement = _factor_conv(module, rank)
            if replacement is None:
                log.warning("Skipping non-beneficial low-rank transform for %s", name)
            else:
                parent_name, _, child_name = name.rpartition(".")
                parent = transformed.get_submodule(parent_name) if parent_name else transformed
                setattr(parent, child_name, replacement)
                log.info("Fast factorized %s at rank %d", name, rank)
        target = replacement if replacement is not None else module
        _apply_pruning(target, policy.pruning)
        if policy.pruning:
            log.info("Fast pruned %.1f%% of Scratch blocks in %s", policy.pruning * 100, name)
    return transformed
