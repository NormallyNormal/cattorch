"""One-token steps for Hugging Face recurrent causal language models.

``RecurrentGenerationProgram()`` without states exports a Mamba, Mamba-2 or
RWKV causal LM through the step built here. Each step computes one token from
the module weights and returns the new states as plain tensors; it depends
only on parameter, attribute and config names, not on the model's cache
objects, whose API differs between transformers versions. Every export checks
the step against the model's stateless forward, which catches upstream
changes these steps don't follow.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from cattorch.errors import UnsupportedModelError
from cattorch.program import State

#: The transformers version the adapters were written and tested against.
TESTED_TRANSFORMERS = "5.20.2"

_FAMILIES = {
    "mamba2": ("Mamba-2", "backbone"),
    "mamba": ("Mamba", "backbone"),
    "rwkv": ("RWKV", "rwkv"),
}


def recurrent_family(model: nn.Module) -> str | None:
    """The name of the supported recurrent family of a Hugging Face model, if any.

    Looks only at models from an already imported ``transformers``.
    """
    transformers = sys.modules.get("transformers")
    if transformers is None or not isinstance(model, transformers.PreTrainedModel):
        return None
    family = _FAMILIES.get(str(getattr(model.config, "model_type", "")))
    return None if family is None else family[0]


def contains_recurrent_model(model: nn.Module) -> str | None:
    """The family of the first supported recurrent Hugging Face model inside ``model``."""
    if "transformers" not in sys.modules:
        return None
    for module in model.modules():
        family = recurrent_family(module)
        if family is not None:
            return family
    return None


def version_hint() -> str:
    """Where an adapted model's mismatch most likely comes from."""
    transformers = sys.modules.get("transformers")
    installed = getattr(transformers, "__version__", "unknown")
    return (
        f"; the installed transformers version ({installed}) may be unsupported: "
        f"cattorch's Hugging Face recurrent adapters are tested with transformers "
        f"{TESTED_TRANSFORMERS}"
    )


@dataclass(frozen=True)
class RecurrentAdapter:
    """An export-owned step module, its initial states and method names."""

    module: nn.Module
    states: tuple[State, ...]
    step: str = "step"
    reference: str = "reference"


def adapt_recurrent_model(model: nn.Module) -> RecurrentAdapter | None:
    """Build the step of a supported Hugging Face recurrent causal LM.

    The step owns a float32 copy of the model, so the caller's model is
    never changed. Returns None for any other model.
    """
    family = recurrent_family(model)
    if family is None:
        return None
    model_type = model.config.model_type
    name = type(model).__name__
    prefix = _FAMILIES[model_type][1]
    head = "head" if model_type == "rwkv" else "lm_head"
    if not isinstance(getattr(model, prefix, None), nn.Module) or not isinstance(
        getattr(model, head, None), nn.Module,
    ):
        raise UnsupportedModelError(
            f"{name} is a {family} model without a language-model head; "
            f"RecurrentGenerationProgram exports causal LMs such as "
            f"{model_type.capitalize()}ForCausalLM"
        )
    builder = {"mamba2": _Mamba2Step, "mamba": _MambaStep, "rwkv": _RwkvStep}[model_type]
    builder.check(model)
    owned = copy.deepcopy(model).float()
    step = builder(owned)
    return RecurrentAdapter(step, step.initial_states())


def _require(owner: object, names: tuple[str, ...], where: str) -> None:
    missing = [name for name in names if not hasattr(owner, name)]
    if missing:
        raise UnsupportedModelError(
            f"{where} has no {', '.join(missing)}, which the recurrent step reads"
            + version_hint()
        )


def _config_error(model: nn.Module, field: str, detail: str) -> UnsupportedModelError:
    value = getattr(model.config, field, None)
    return UnsupportedModelError(
        f"{type(model).__name__} with config {field}={value!r} isn't supported: {detail}"
    )


class _HuggingFaceStep(nn.Module):
    """A causal LM's one-token step beside its stateless forward."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.training = model.training

    def reference(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model(tokens, use_cache=False, return_dict=True).logits


class _Mamba2Step(_HuggingFaceStep):
    """Mamba-2: per layer a convolution window and an SSM state per head.

    ``conv i`` is ``[conv_dim, conv_kernel - 1]``, the last inputs of the
    depthwise convolution, and ``ssm i`` is
    ``[num_heads, head_dim, state_size]``.
    """

    _MIXER = (
        "in_proj", "conv1d", "act", "dt_bias", "A_log", "D", "norm", "out_proj",
        "num_heads", "head_dim", "n_groups", "ssm_state_size", "intermediate_size",
        "conv_dim", "conv_kernel_size", "time_step_limit",
    )

    @classmethod
    def check(cls, model: nn.Module) -> None:
        _require(model.backbone, ("embeddings", "layers", "norm_f"), "backbone")
        for index, block in enumerate(model.backbone.layers):
            _require(block, ("norm", "mixer"), f"backbone.layers.{index}")
            _require(block.mixer, cls._MIXER, f"backbone.layers.{index}.mixer")
        config = model.config
        if config.num_heads % config.n_groups:
            raise _config_error(
                model, "n_groups", f"num_heads ({config.num_heads}) must be a multiple of it",
            )
        limit = config.time_step_limit
        if (
            not isinstance(limit, (tuple, list)) or len(limit) != 2
            or not all(isinstance(value, (int, float)) for value in limit)
        ):
            raise _config_error(model, "time_step_limit", "expected (minimum, maximum)")

    def initial_states(self) -> tuple[State, ...]:
        states = []
        for index, block in enumerate(self.model.backbone.layers):
            mixer = block.mixer
            if mixer.conv_kernel_size > 1:
                states.append(State(
                    f"conv {index}", torch.zeros(mixer.conv_dim, mixer.conv_kernel_size - 1),
                ))
            states.append(State(
                f"ssm {index}",
                torch.zeros(mixer.num_heads, mixer.head_dim, mixer.ssm_state_size),
            ))
        return tuple(states)

    def step(self, token: torch.Tensor, *states: torch.Tensor):
        backbone = self.model.backbone
        hidden = backbone.embeddings(token).reshape(1, -1)
        remaining = list(states)
        updates = []
        for block in backbone.layers:
            mixer = block.mixer
            heads, width, size = mixer.num_heads, mixer.head_dim, mixer.ssm_state_size
            gate, xbc, dt = mixer.in_proj(block.norm(hidden)).split(
                [mixer.intermediate_size, mixer.conv_dim, heads], dim=-1,
            )
            column = xbc.reshape(mixer.conv_dim, 1)
            if mixer.conv_kernel_size > 1:
                window = torch.cat([remaining.pop(0), column], dim=-1)
                updates.append(window[:, 1:])
            else:
                window = column
            xbc = (window * mixer.conv1d.weight.reshape(mixer.conv_dim, -1)).sum(-1)
            if mixer.conv1d.bias is not None:
                xbc = xbc + mixer.conv1d.bias
            group = mixer.n_groups * size
            x, B, C = mixer.act(xbc).split([mixer.intermediate_size, group, group], dim=-1)
            x = x.reshape(heads, width)
            # Each group's B and C serve num_heads / n_groups consecutive heads.
            repeat = heads // mixer.n_groups
            B = B.reshape(mixer.n_groups, 1, size).expand(-1, repeat, -1).reshape(heads, size)
            C = C.reshape(mixer.n_groups, 1, size).expand(-1, repeat, -1).reshape(heads, size)
            dt = F.softplus(dt.reshape(heads) + mixer.dt_bias)
            low, high = mixer.time_step_limit
            if (low, high) != (0.0, float("inf")):
                dt = dt.clamp(low, high)
            decay = torch.exp(dt * -torch.exp(mixer.A_log))
            ssm = remaining.pop(0) * decay[:, None, None] + (dt[:, None] * x)[:, :, None] * B[:, None, :]
            updates.append(ssm)
            y = (ssm * C[:, None, :]).sum(-1) + mixer.D[:, None] * x
            hidden = hidden + mixer.out_proj(mixer.norm(y.reshape(1, -1), gate))
        logits = self.model.lm_head(backbone.norm_f(hidden))
        return (logits, *updates)


class _MambaStep(_HuggingFaceStep):
    """Mamba: per layer a convolution window and an SSM state per channel.

    ``conv i`` is ``[intermediate_size, conv_kernel - 1]`` and ``ssm i`` is
    ``[intermediate_size, state_size]``.
    """

    _MIXER = (
        "in_proj", "conv1d", "act", "x_proj", "dt_proj", "A_log", "D", "out_proj",
        "intermediate_size", "ssm_state_size", "time_step_rank", "conv_kernel_size",
    )

    @classmethod
    def check(cls, model: nn.Module) -> None:
        _require(model.backbone, ("embeddings", "layers", "norm_f"), "backbone")
        for index, block in enumerate(model.backbone.layers):
            _require(block, ("norm", "mixer"), f"backbone.layers.{index}")
            _require(block.mixer, cls._MIXER, f"backbone.layers.{index}.mixer")

    def initial_states(self) -> tuple[State, ...]:
        states = []
        for index, block in enumerate(self.model.backbone.layers):
            mixer = block.mixer
            if mixer.conv_kernel_size > 1:
                states.append(State(
                    f"conv {index}",
                    torch.zeros(mixer.intermediate_size, mixer.conv_kernel_size - 1),
                ))
            states.append(State(
                f"ssm {index}", torch.zeros(mixer.intermediate_size, mixer.ssm_state_size),
            ))
        return tuple(states)

    def step(self, token: torch.Tensor, *states: torch.Tensor):
        backbone = self.model.backbone
        hidden = backbone.embeddings(token).reshape(1, -1)
        remaining = list(states)
        updates = []
        for block in backbone.layers:
            mixer = block.mixer
            inner, size = mixer.intermediate_size, mixer.ssm_state_size
            x, gate = mixer.in_proj(block.norm(hidden)).chunk(2, dim=-1)
            column = x.reshape(inner, 1)
            if mixer.conv_kernel_size > 1:
                window = torch.cat([remaining.pop(0), column], dim=-1)
                updates.append(window[:, 1:])
            else:
                window = column
            x = (window * mixer.conv1d.weight.reshape(inner, -1)).sum(-1)
            if mixer.conv1d.bias is not None:
                x = x + mixer.conv1d.bias
            x = mixer.act(x)
            dt, B, C = mixer.x_proj(x).split([mixer.time_step_rank, size, size], dim=-1)
            dt = F.softplus(mixer.dt_proj(dt))
            decay = torch.exp(dt[:, None] * -torch.exp(mixer.A_log))
            ssm = remaining.pop(0) * decay + (dt * x)[:, None] * B[None, :]
            updates.append(ssm)
            y = ((ssm * C[None, :]).sum(-1) + mixer.D * x) * F.silu(gate.reshape(inner))
            hidden = hidden + mixer.out_proj(y.reshape(1, inner))
        logits = self.model.lm_head(backbone.norm_f(hidden))
        return (logits, *updates)


def _time_mix(current: torch.Tensor, previous: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.reshape(1, -1)
    return current * weight + previous * (1 - weight)


class _RwkvStep(_HuggingFaceStep):
    """RWKV-4: per layer five states, as in transformers' ``state=`` list.

    ``attention shift i`` and ``feed forward shift i`` (``[hidden_size]``) are
    the previous token's normalized inputs; ``numerator i``, ``denominator i``
    and ``maximum i`` (``[attention_hidden_size]``) are the WKV sums, stored
    relative to ``exp(maximum)``.
    """

    _ATTENTION = (
        "time_decay", "time_first", "time_mix_key", "time_mix_value", "time_mix_receptance",
        "key", "value", "receptance", "output",
    )
    _FEED_FORWARD = ("time_mix_key", "time_mix_receptance", "key", "receptance", "value")

    @classmethod
    def check(cls, model: nn.Module) -> None:
        _require(model.rwkv, ("embeddings", "blocks", "ln_out", "layers_are_rescaled"), "rwkv")
        for index, block in enumerate(model.rwkv.blocks):
            where = f"rwkv.blocks.{index}"
            _require(block, ("ln1", "ln2", "attention", "feed_forward"), where)
            if index == 0:
                _require(block, ("pre_ln",), where)
            _require(block.attention, cls._ATTENTION, f"{where}.attention")
            _require(block.feed_forward, cls._FEED_FORWARD, f"{where}.feed_forward")
        if not isinstance(model.config.rescale_every, int):
            raise _config_error(model, "rescale_every", "expected an integer")

    def initial_states(self) -> tuple[State, ...]:
        states = []
        for index, block in enumerate(self.model.rwkv.blocks):
            hidden = block.attention.time_mix_key.numel()
            width = block.attention.time_decay.numel()
            states += [
                State(f"attention shift {index}", torch.zeros(1, hidden)),
                State(f"feed forward shift {index}", torch.zeros(1, hidden)),
                State(f"numerator {index}", torch.zeros(1, width)),
                State(f"denominator {index}", torch.zeros(1, width)),
                # As transformers starts its state; any value far below the
                # keys makes the first token's weight exp(-inf) = 0.
                State(f"maximum {index}", torch.full((1, width), -1e30)),
            ]
        return tuple(states)

    def step(self, token: torch.Tensor, *states: torch.Tensor):
        rwkv = self.model.rwkv
        every = self.model.config.rescale_every
        hidden = rwkv.embeddings(token).reshape(1, -1)
        updates = []
        for index, block in enumerate(rwkv.blocks):
            attention_shift, feed_forward_shift, numerator, denominator, maximum = (
                states[5 * index:5 * index + 5]
            )
            # In evaluation, transformers divides these layers' output
            # weights by 2 ** (index // rescale_every) once, in place, and
            # halves the hidden state after every rescale_every blocks.
            scale = 1.0
            if every > 0 and not rwkv.layers_are_rescaled:
                scale = 2.0 ** -(index // every)
            if index == 0:
                hidden = block.pre_ln(hidden)

            attention = block.attention
            attention_input = block.ln1(hidden)
            key, value, receptance = (
                _time_mix(attention_input, attention_shift, weight)
                for weight in (
                    attention.time_mix_key, attention.time_mix_value, attention.time_mix_receptance,
                )
            )
            key = attention.key(key)
            value = attention.value(value)
            receptance = torch.sigmoid(attention.receptance(receptance))
            first = key + attention.time_first
            top = torch.maximum(maximum, first)
            old, new = torch.exp(maximum - top), torch.exp(first - top)
            wkv = (old * numerator + new * value) / (old * denominator + new)
            decayed = maximum - torch.exp(attention.time_decay)
            maximum = torch.maximum(decayed, key)
            old, new = torch.exp(decayed - maximum), torch.exp(key - maximum)
            numerator = old * numerator + new * value
            denominator = old * denominator + new
            output = attention.output(receptance * wkv)
            hidden = hidden + (output if scale == 1.0 else output * scale)

            feed_forward = block.feed_forward
            feed_forward_input = block.ln2(hidden)
            key, receptance = (
                _time_mix(feed_forward_input, feed_forward_shift, weight)
                for weight in (feed_forward.time_mix_key, feed_forward.time_mix_receptance)
            )
            value = feed_forward.value(torch.square(torch.relu(feed_forward.key(key))))
            if scale != 1.0:
                value = value * scale
            hidden = hidden + torch.sigmoid(feed_forward.receptance(receptance)) * value
            if every > 0 and (index + 1) % every == 0:
                hidden = hidden / 2
            updates += [attention_input, feed_forward_input, numerator, denominator, maximum]
        logits = self.model.head(rwkv.ln_out(hidden))
        return (logits, *updates)


__all__ = [
    "RecurrentAdapter", "TESTED_TRANSFORMERS", "adapt_recurrent_model",
    "contains_recurrent_model", "recurrent_family", "version_hint",
]
