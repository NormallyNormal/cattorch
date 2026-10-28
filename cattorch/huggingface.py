"""Export-time defaults for Hugging Face ``PreTrainedModel`` instances."""

from __future__ import annotations

import copy
import logging
import sys

import torch

log = logging.getLogger(__name__)



def prepare_huggingface_models(model: torch.nn.Module) -> bool:
    """Configure outermost Hugging Face models inside an export-owned copy.

    Disables the KV cache and extra outputs and selects a traceable attention
    implementation. Returns whether any model was found. Does nothing unless
    the caller already imported ``transformers``.
    """
    transformers = sys.modules.get("transformers")
    if transformers is None:
        return False
    found = False
    claimed: set[int] = set()
    for module in model.modules():
        if id(module) in claimed or not isinstance(module, transformers.PreTrainedModel):
            continue
        found = True
        # Inner models of an outermost one share its treatment.
        claimed.update(id(inner) for inner in module.modules())
        _configure(module)
    return found


def _configure(model: torch.nn.Module) -> None:
    config = model.config
    config.use_cache = False
    config.output_attentions = False
    config.output_hidden_states = False
    change = ""
    # SDPA lowers to smaller sprites than eager attention; anything else,
    # such as flash or flex attention, doesn't trace.
    current = getattr(config, "_attn_implementation", None)
    target = "sdpa" if getattr(model, "_supports_sdpa", False) else "eager"
    if current is not None and current != target:
        model.set_attn_implementation(target)
        change = f", attention {current} -> {target}"
    log.info("prepared %s for export: use_cache=False%s", type(model).__name__, change)
    _configure_experts(model)


def _configure_experts(model: torch.nn.Module) -> None:
    """Experts are replaced module by module; see ``convert_huggingface_experts``."""


# ── Experts ──────────────────────────────────────────────────────────────────
# transformers marks most MoE expert classes with ``@use_experts_implementation``.
# They store stacked weights and are called as ``experts(hidden, top_k_index,
# top_k_weights)``; only the expert loop depends on tensor values. Instead of
# registering a global ``ExpertsInterface`` entry, the export copy swaps each
# such module for one that runs ``routed_experts`` on an ``ExpertFamily``.

_EXPERTS_FLAGS = ("has_gate", "has_bias", "is_transposed", "has_post_expert_norm")


def is_huggingface_experts(module: torch.nn.Module) -> bool:
    """Whether ``module`` is a ``@use_experts_implementation`` experts class."""
    down = getattr(module, "down_proj", None)
    return (
        "transformers" in sys.modules
        and all(hasattr(module, flag) for flag in _EXPERTS_FLAGS)
        and hasattr(module, "_apply_gate")
        and isinstance(down, torch.Tensor) and down.ndim == 3
    )


class _HuggingFaceExpert(torch.nn.Module):
    """One expert of a Hugging Face experts class, as an ``ExpertFamily`` template."""

    def __init__(self, gated, width, hidden, bias, act_fn, norm):
        super().__init__()
        self.gate = torch.nn.Linear(width, hidden, bias=bias) if gated else None
        self.up = torch.nn.Linear(width, hidden, bias=bias)
        self.down = torch.nn.Linear(hidden, width, bias=bias)
        self.act_fn = act_fn
        self.norm = norm

    def forward(self, value):
        if self.gate is not None:
            hidden = self.act_fn(self.gate(value)) * self.up(value)
        else:
            hidden = self.act_fn(self.up(value))
        output = self.down(hidden)
        return output if self.norm is None else self.norm(output)


class _HuggingFaceGate(torch.nn.Module):
    """A class's own ``_apply_gate`` with the settings it reads (limits, alpha)."""

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self._gate = type(module)._apply_gate
        for name, value in vars(module).items():
            if not name.startswith("_") and isinstance(value, (bool, int, float, str)):
                setattr(self, name, value)
        if isinstance(getattr(module, "act_fn", None), torch.nn.Module):
            self.act_fn = copy.deepcopy(module.act_fn)

    def forward(self, gate_up):
        return self._gate(self, gate_up)


class _HuggingFaceCustomGateExpert(torch.nn.Module):
    """One expert whose combined gate/up output goes through a custom gate."""

    def __init__(self, width, combined, hidden, bias, gate, norm):
        super().__init__()
        self.gate_up = torch.nn.Linear(width, combined, bias=bias)
        self.down = torch.nn.Linear(hidden, width, bias=bias)
        self.gate = gate
        self.norm = norm

    def forward(self, value):
        output = self.down(self.gate(self.gate_up(value)))
        return output if self.norm is None else self.norm(output)


class HuggingFaceExperts(torch.nn.Module):
    """Export form of a Hugging Face experts module: routed banked experts."""

    def __init__(self, family):
        super().__init__()
        self.experts = family

    def forward(self, hidden_states, top_k_index, top_k_weights):
        from cattorch.experts import routed_experts

        output = routed_experts(hidden_states, top_k_index, top_k_weights, self.experts)
        return output.to(hidden_states.dtype)


def convert_huggingface_experts(module: torch.nn.Module, path: str) -> HuggingFaceExperts:
    """Build the ``routed_experts`` form of a Hugging Face experts module."""
    from transformers.integrations.moe import _default_apply_gate

    from cattorch.errors import UnsupportedModelError

    where = path or "<root>"
    name = type(module).__name__

    def matrix(stack):
        # Linear layout is [experts, out, in]; transposed classes store [experts, in, out].
        return stack.detach().transpose(1, 2) if module.is_transposed else stack.detach()

    if module.has_gate and getattr(type(module), "_apply_gate", None) is not _default_apply_gate:
        return _convert_custom_gate(module, matrix, where, name)

    up_name = "gate_up_proj" if module.has_gate else "up_proj"
    up = matrix(getattr(module, up_name))
    down = matrix(module.down_proj)
    experts, width = down.shape[0], down.shape[1]
    hidden = down.shape[2]
    banks = {}
    biases = {}
    if module.has_bias:
        up_bias = getattr(module, f"{up_name}_bias").detach()
        biases["down"] = module.down_proj_bias.detach()
    if module.has_gate:
        # The default gate is ``act_fn(gate) * up`` over ``gate_up.chunk(2, -1)``.
        banks["gate.weight"], banks["up.weight"] = up[:, :hidden], up[:, hidden:]
        if module.has_bias:
            biases["gate"], biases["up"] = up_bias[:, :hidden], up_bias[:, hidden:]
    else:
        banks["up.weight"] = up
        if module.has_bias:
            biases["up"] = up_bias
    banks["down.weight"] = down
    banks.update({f"{layer}.bias": value for layer, value in biases.items()})
    if banks["up.weight"].shape[1:] != (hidden, width):
        raise UnsupportedModelError(
            f"{name} at {where} has expert weights shaped {tuple(getattr(module, up_name).shape)} "
            f"and {tuple(module.down_proj.shape)}, which don't form an up/down projection pair"
        )

    norm = (
        copy.deepcopy(module.post_expert_norm) if module.has_post_expert_norm else None
    )
    template = _HuggingFaceExpert(
        module.has_gate, width, hidden, module.has_bias, copy.deepcopy(module.act_fn), norm,
    )
    return _family(module, template, banks, experts, width, down.dtype, where, name)


def _family(module, template, banks, experts, width, dtype, where, name) -> HuggingFaceExperts:
    from cattorch.experts import ExpertFamily

    # Every template tensor needs a bank; the shared norm and activation state
    # is the same for every expert.
    for state_name, value in (
        *template.named_parameters(remove_duplicate=False),
        *template.named_buffers(remove_duplicate=False),
    ):
        if state_name not in banks:
            banks[state_name] = value.detach().unsqueeze(0).expand(experts, *value.shape)
    family = ExpertFamily(
        template.to(dtype), banks, example_input=torch.zeros(1, width, dtype=dtype),
    )
    family.train(module.training)
    log.info("exporting %s at %s through routed_experts (%d experts)", name, where, experts)
    return HuggingFaceExperts(family)


def _convert_custom_gate(module, matrix, where, name) -> HuggingFaceExperts:
    """Experts whose class defines ``_apply_gate``, such as GPT-OSS's clamped GLU."""
    from cattorch.errors import UnsupportedModelError

    gate_up = matrix(module.gate_up_proj)
    down = matrix(module.down_proj)
    experts, width, hidden = down.shape
    banks = {"gate_up.weight": gate_up, "down.weight": down}
    if module.has_bias:
        banks["gate_up.bias"] = module.gate_up_proj_bias.detach()
        banks["down.bias"] = module.down_proj_bias.detach()
    gate = _HuggingFaceGate(module)
    with torch.no_grad():
        probe = gate(torch.zeros(1, gate_up.shape[1], dtype=down.dtype))
    if gate_up.shape[2] != width or probe.shape[-1] != hidden:
        raise UnsupportedModelError(
            f"{name} at {where} has expert weights shaped {tuple(module.gate_up_proj.shape)} "
            f"and {tuple(module.down_proj.shape)}, which its _apply_gate doesn't connect"
        )
    norm = copy.deepcopy(module.post_expert_norm) if module.has_post_expert_norm else None
    template = _HuggingFaceCustomGateExpert(
        width, gate_up.shape[1], hidden, module.has_bias, gate, norm,
    )
    return _family(module, template, banks, experts, width, down.dtype, where, name)
