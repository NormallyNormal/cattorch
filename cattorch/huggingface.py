"""Export-time defaults for Hugging Face ``PreTrainedModel`` instances."""

from __future__ import annotations

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
    """Hook for feature 6: register the "cattorch" experts implementation."""
