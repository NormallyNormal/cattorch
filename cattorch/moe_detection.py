"""Recognize hand-written sparse mixture-of-experts modules.

A typical MoE block holds a ``ModuleList`` of identical experts and a linear
router, picks experts with ``torch.topk``, and loops over them with
``torch.where`` and ``index_add_``. That loop depends on tensor values, so it
can't be traced. A module shaped like that is a candidate: cattorch runs it on
probe inputs next to ``SparseMoE`` with each plausible routing configuration
and swaps in the first one that gives the same output.
"""

from __future__ import annotations

import copy
import itertools
import logging
import warnings

import torch
from torch import nn

from cattorch.errors import UnsupportedModelError

log = logging.getLogger(__name__)

# Attribute names MoE blocks commonly use for the number of routed experts.
_TOP_K_NAMES = (
    "top_k", "topk", "k", "num_experts_per_tok", "num_selected", "experts_per_token",
    "n_activated_experts", "num_activated_experts",
)
_SCORINGS = ("softmax", "sigmoid", "raw")
_PROBE_SEED = 1234
# Operations whose output size depends on tensor values, as an expert loop's
# ``torch.where(ids == expert)`` does; the graph traces but can't be lowered.
_DATA_DEPENDENT_OPS = frozenset({
    "aten.nonzero.default", "aten.masked_select.default",
    "aten._local_scalar_dense.default", "aten.unique_dim.default",
    "aten._unique2.default",
})


class CustomMoEWarning(UserWarning):
    """A hand-written MoE block was exported as a sparse mixture of experts."""


def _candidate_parts(module: nn.Module) -> tuple[nn.Linear, nn.ModuleList] | None:
    """The router and experts of a module that looks like a sparse MoE block."""
    expert_lists = [
        child for child in module.children()
        if isinstance(child, nn.ModuleList) and len(child) >= 2
        and len({type(expert) for expert in child}) == 1
    ]
    if len(expert_lists) != 1:
        return None
    experts = expert_lists[0]
    routers = [
        child for child in module.children()
        if isinstance(child, nn.Linear) and child.out_features == len(experts)
    ]
    if len(routers) != 1:
        return None
    return routers[0], experts


def _top_k_order(module: nn.Module, experts: int) -> list[int]:
    preferred = []
    for name in _TOP_K_NAMES:
        value = getattr(module, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= experts:
            preferred.append(value)
    rest = [value for value in range(1, experts + 1) if value not in preferred]
    return list(dict.fromkeys(preferred)) + rest


def _router(linear: nn.Linear, scoring: str) -> nn.Module:
    if scoring == "softmax":
        return nn.Sequential(linear, nn.Softmax(dim=-1))
    if scoring == "sigmoid":
        return nn.Sequential(linear, nn.Sigmoid())
    return nn.Sequential(linear)


def _probe_outputs(module: nn.Module, width: int, dtype: torch.dtype):
    """Inputs the module accepts, with its outputs, or ``None``."""
    generator = torch.Generator().manual_seed(_PROBE_SEED)
    for shape in ((2, 6, width), (12, width)):
        inputs = [torch.randn(shape, generator=generator, dtype=dtype) for _ in range(2)]
        try:
            with torch.no_grad():
                outputs = [module(value) for value in inputs]
        except Exception:  # noqa: BLE001 - any failure means "not this shape"
            continue
        if all(
            isinstance(output, torch.Tensor) and output.shape == value.shape
            for output, value in zip(outputs, inputs)
        ):
            return inputs, outputs
    return None


def _describe(top_k: int, scoring: str, normalize: bool) -> str:
    scores = {"softmax": "softmax", "sigmoid": "sigmoid", "raw": "raw router"}[scoring]
    weights = "renormalized" if normalize else "unnormalized"
    return f"top_k={top_k}, {scores} scores, {weights} weights"


def detect_custom_moe(module: nn.Module, path: str) -> nn.Module | None:
    """Return a ``SparseMoE`` equivalent to ``module``, or ``None``.

    Raises ``UnsupportedModelError`` for a candidate that matches no routing
    configuration and can't be traced as it is.
    """
    from cattorch.moe import ExpertFamily, SparseMoE

    parts = _candidate_parts(module)
    if parts is None:
        return None
    linear, expert_list = parts
    width = linear.in_features
    dtype = linear.weight.dtype
    probes = _probe_outputs(module, width, dtype)
    if probes is None:
        return None
    try:
        family = ExpertFamily.from_modules(
            list(expert_list), example_input=torch.zeros(1, width, dtype=dtype),
        )
    except (ValueError, TypeError, RuntimeError):
        return None
    family.train(module.training)
    inputs, expected = probes
    where = path or "<root>"
    name = type(module).__name__
    count = len(expert_list)
    for top_k, scoring, normalize in itertools.product(
        _top_k_order(module, count), _SCORINGS, (False, True),
    ):
        candidate = SparseMoE(
            _router(linear, scoring), family, top_k=top_k, normalize_selected=normalize,
        ).train(module.training)
        try:
            with torch.no_grad():
                matches = all(
                    torch.allclose(candidate(value), output, rtol=1e-4, atol=1e-5)
                    for value, output in zip(inputs, expected)
                )
        except Exception:  # noqa: BLE001 - a configuration that can't run doesn't match
            continue
        if matches:
            description = _describe(top_k, scoring, normalize)
            warnings.warn(
                f"{name} at {where} is exported as a sparse mixture of {count} "
                f"experts ({description}), which matched it on probe inputs; "
                "run verify on real inputs, or declare the block with "
                "ExpertFamily and SparseMoE",
                CustomMoEWarning,
                stacklevel=2,
            )
            log.info("custom MoE %s at %s: %s", name, where, description)
            return candidate
    _require_traceable(module, inputs[0], name, where)
    return None


def _require_traceable(module: nn.Module, example: torch.Tensor, name: str, where: str) -> None:
    from cattorch.frontend import capture_model

    try:
        captured = capture_model(copy.deepcopy(module), (example,))
        dynamic = next(
            (
                str(node.target) for node in captured.graph.nodes
                if str(node.target) in _DATA_DEPENDENT_OPS
            ),
            None,
        )
        if dynamic is not None:
            raise UnsupportedModelError(f"{dynamic} has a data-dependent output size")
    except UnsupportedModelError as error:
        raise UnsupportedModelError(
            f"{name} at {where} looks like a mixture of experts, but no top-k, "
            "softmax or sigmoid routing reproduces it and it can't be traced as "
            "it is; rebuild it with ExpertFamily and SparseMoE, or keep its "
            "router and call routed_experts (see docs/programs-and-moe.md)"
        ) from error
