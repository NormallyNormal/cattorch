"""Hugging Face ``@use_experts_implementation`` experts export through routed_experts."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cattorch import UnsupportedModelError, transpile, verify
from cattorch.huggingface import HuggingFaceExperts
from cattorch.moe import _expert_family_context
from cattorch.operator_registry import default_registry

transformers = pytest.importorskip("transformers")
moe = pytest.importorskip("transformers.integrations.moe")
if not hasattr(moe, "use_experts_implementation"):
    pytest.skip("transformers has no experts interface", allow_module_level=True)
from transformers.activations import ACT2FN  # noqa: E402

EXPERTS, WIDTH, HIDDEN = 4, 6, 5


class Config:
    _experts_implementation = "eager"


def experts_class(*, gated=True, transposed=False, bias=False, norm=False, act="silu"):
    """An experts class in the transformers style, with a reference eager loop."""

    @moe.use_experts_implementation(
        has_gate=gated, is_transposed=transposed, has_bias=bias, has_post_expert_norm=norm,
    )
    class Experts(nn.Module):
        def __init__(self, config):
            super().__init__()
            up = 2 * HIDDEN if gated else HIDDEN
            shapes = {
                "up": (EXPERTS, WIDTH, up) if transposed else (EXPERTS, up, WIDTH),
                "down": (EXPERTS, HIDDEN, WIDTH) if transposed else (EXPERTS, WIDTH, HIDDEN),
            }
            self.up_name = "gate_up_proj" if gated else "up_proj"
            setattr(self, self.up_name, nn.Parameter(torch.randn(shapes["up"]) / 2))
            self.down_proj = nn.Parameter(torch.randn(shapes["down"]) / 2)
            if bias:
                setattr(self, f"{self.up_name}_bias", nn.Parameter(torch.randn(EXPERTS, up)))
                self.down_proj_bias = nn.Parameter(torch.randn(EXPERTS, WIDTH))
            self.act_fn = ACT2FN[act]
            if norm:
                # Expert programs lower nn.RMSNorm, not a hand-written norm.
                self.post_expert_norm = nn.RMSNorm(WIDTH, eps=1e-6)
                nn.init.uniform_(self.post_expert_norm.weight, 0.5, 1.5)

        def forward(self, hidden_states, top_k_index, top_k_weights):
            output = torch.zeros_like(hidden_states)
            for token in range(hidden_states.shape[0]):
                for route in range(top_k_index.shape[-1]):
                    expert = int(top_k_index[token, route])
                    value = hidden_states[token]

                    def project(name, value):
                        weight = getattr(self, name)[expert]
                        result = value @ weight if transposed else weight @ value
                        if bias:
                            result = result + getattr(self, f"{name}_bias")[expert]
                        return result

                    projected = project(self.up_name, value)
                    if gated:
                        hidden = self._apply_gate(projected)
                    else:
                        hidden = self.act_fn(projected)
                    result = project("down_proj", hidden)
                    if norm:
                        result = self.post_expert_norm(result)
                    output[token] += top_k_weights[token, route] * result
            return output

    return Experts


class Block(nn.Module):
    """A Mixtral-style block around an experts module: softmax, top-k, renormalize."""

    def __init__(self, experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(WIDTH, EXPERTS, bias=False)
        self.experts = experts
        self.top_k = top_k

    def forward(self, value):
        flat = value.reshape(-1, WIDTH)
        weights, ids = torch.topk(torch.softmax(self.gate(flat), dim=-1), self.top_k, dim=-1)
        weights /= weights.sum(dim=-1, keepdim=True)
        return self.experts(flat, ids, weights).reshape(value.shape)


VARIANTS = {
    "default": {},
    "transposed": {"transposed": True},
    "bias": {"bias": True},
    "transposed bias": {"transposed": True, "bias": True},
    "ungated relu2": {"gated": False, "act": "relu2"},
    "post-expert norm": {"norm": True},
    "gelu tanh": {"act": "gelu_pytorch_tanh"},
}


@torch.no_grad()
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_converted_experts_match_the_reference_loop(variant):
    torch.manual_seed(0)
    experts = experts_class(**VARIANTS[variant])(Config()).eval()
    registry = default_registry().clone()
    converted = registry.adapt_model(experts)
    assert isinstance(converted, HuggingFaceExperts)
    value = torch.randn(5, WIDTH)
    ids = torch.tensor([[0, 3], [2, 2], [1, 0], [3, 1], [2, 0]])
    weights = torch.rand(5, 2)
    with _expert_family_context(registry.expert_families):
        actual = converted(value, ids, weights)
    assert torch.allclose(actual, experts(value, ids, weights), atol=1e-5)


@torch.no_grad()
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_experts_block_lowers_to_scratch(tmp_path, variant):
    torch.manual_seed(1)
    model = Block(experts_class(**VARIANTS[variant])(Config())).eval()
    value = torch.randn(1, 3, WIDTH)
    result = transpile(model, value, tmp_path / "hf-experts")
    comparison = verify(model, value, result, atol=1e-4)
    assert comparison.passed, comparison


def gpt_oss_gate(self, gate_up):
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=self.limit)
    up = up.clamp(min=-self.limit, max=self.limit)
    return (up + 1) * (gate * torch.sigmoid(gate * self.alpha))


def clamped_swiglu_gate(self, gate_up):
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=self.limit)
    up = up.clamp(min=-self.limit, max=self.limit)
    return (self.act_fn(gate.float()) * up.float()).to(gate.dtype)


CUSTOM_GATES = {
    "gpt-oss": (gpt_oss_gate, {"bias": True, "transposed": True}),
    "clamped swiglu": (clamped_swiglu_gate, {}),
}


def custom_gate_class(case):
    gate, options = CUSTOM_GATES[case]
    experts = experts_class(**options)
    experts._apply_gate = gate
    original = experts.__init__

    def init(self, config):
        original(self, config)
        # Small limits so the clamps change some values.
        self.limit, self.alpha = 0.6, 1.702

    experts.__init__ = init
    return experts


@torch.no_grad()
@pytest.mark.parametrize("case", sorted(CUSTOM_GATES))
def test_custom_gates_export_through_the_class_gate(tmp_path, case):
    torch.manual_seed(4)
    experts = custom_gate_class(case)(Config()).eval()
    value = torch.randn(5, WIDTH)
    ids = torch.tensor([[0, 3], [2, 2], [1, 0], [3, 1], [2, 0]])
    weights = torch.rand(5, 2)
    expected = torch.zeros(5, WIDTH)
    for token in range(5):
        for route in range(2):
            expert = int(ids[token, route])
            up = experts.gate_up_proj[expert]
            projected = value[token] @ up if experts.is_transposed else up @ value[token]
            if experts.has_bias:
                projected = projected + experts.gate_up_proj_bias[expert]
            hidden = experts._apply_gate(projected)
            down = experts.down_proj[expert]
            result = hidden @ down if experts.is_transposed else down @ hidden
            if experts.has_bias:
                result = result + experts.down_proj_bias[expert]
            expected[token] += weights[token, route] * result
    registry = default_registry().clone()
    converted = registry.adapt_model(experts)
    with _expert_family_context(registry.expert_families):
        torch.testing.assert_close(converted(value, ids, weights), expected, atol=1e-5, rtol=0)
    model = Block(custom_gate_class(case)(Config())).eval()
    block_value = torch.randn(1, 3, WIDTH)
    result = transpile(model, block_value, tmp_path / "custom-gate")
    comparison = verify(model, block_value, result, atol=1e-4)
    assert comparison.passed, comparison


def test_the_original_experts_module_is_unchanged():
    experts = experts_class()(Config()).eval()
    default_registry().clone().adapt_model(Block(experts).eval())
    assert type(experts).__name__ == "Experts" and experts.gate_up_proj.ndim == 3


TINY = dict(
    vocab_size=64, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
    num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32,
)
MODELS = {
    "mixtral": ("MixtralConfig", dict(num_local_experts=4, num_experts_per_tok=2)),
    "qwen2_moe": ("Qwen2MoeConfig", dict(
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=12,
        shared_expert_intermediate_size=20,
    )),
    "qwen3_moe": ("Qwen3MoeConfig", dict(
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=12, head_dim=8,
    )),
    "olmoe": ("OlmoeConfig", dict(num_experts=4, num_experts_per_tok=2)),
    # Interleaved gate/up with a clamped GLU in its own _apply_gate.
    "gpt_oss": ("GptOssConfig", dict(
        num_local_experts=4, num_experts_per_tok=2, head_dim=8, sliding_window=4,
    )),
    # Multi-head latent attention fills its keys by slice assignment, and the
    # router picks expert groups with scatter and gather.
    "deepseek_v3": ("DeepseekV3Config", dict(
        n_routed_experts=4, num_experts_per_tok=2, n_group=2, topk_group=1,
        moe_intermediate_size=12, first_k_dense_replace=0, n_shared_experts=1,
        num_key_value_heads=2, q_lora_rank=8, kv_lora_rank=8,
        qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=8,
    )),
}


@torch.no_grad()
@pytest.mark.parametrize("kind", sorted(MODELS))
def test_tiny_huggingface_moe_models_export_unwrapped(tmp_path, kind):
    config_name, extra = MODELS[kind]
    config_class = getattr(transformers, config_name, None)
    if config_class is None:
        pytest.skip(f"transformers has no {config_name}")
    torch.manual_seed(2)
    model = transformers.AutoModelForCausalLM.from_config(config_class(**{**TINY, **extra})).eval()
    tokens = torch.tensor([[3, 17, 5, 40]])
    result = transpile(model, tokens, tmp_path / kind)
    comparison = verify(model, tokens, result, atol=1e-4)
    assert comparison.passed, comparison


@torch.no_grad()
@pytest.mark.parametrize("kind", ["mixtral", "qwen2_moe", "deepseek_v3"])
def test_tiny_huggingface_moe_models_generate_with_a_cache(tmp_path, kind):
    import json
    import zipfile

    from cattorch import GenerationProgram
    from cattorch.util.scratch.emulator import ScratchEmulator

    config_name, extra = MODELS[kind]
    config_class = getattr(transformers, config_name, None)
    if config_class is None:
        pytest.skip(f"transformers has no {config_name}")
    torch.manual_seed(3)
    model = transformers.AutoModelForCausalLM.from_config(
        config_class(**{**TINY, "num_hidden_layers": 2, **extra}),
    ).eval()
    before = dict(default_registry().expert_families)
    result = transpile(model, GenerationProgram(max_context=8), tmp_path / kind)
    assert default_registry().expert_families == before
    with zipfile.ZipFile(result.path) as archive:
        emulator = ScratchEmulator(json.loads(archive.read("sprite.json")))

    def assert_logits(prefix):
        expected = model(torch.tensor([prefix])).logits[0, -1]
        actual = torch.tensor(emulator.lists["cattorch logits"])
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)

    emulator.run_procedure("cattorch init")
    prefix = [3, 17, 5]
    emulator.lists["cattorch tokens"] = list(prefix)
    emulator.run_procedure("cattorch prefill")
    assert emulator.variables["cattorch status"] == "ok"
    assert_logits(prefix)
    for token in (40, 9):
        prefix.append(token)
        emulator.lists["cattorch tokens"] = [token]
        emulator.run_procedure("cattorch decode")
        assert emulator.variables["cattorch status"] == "ok"
        assert_logits(prefix)
