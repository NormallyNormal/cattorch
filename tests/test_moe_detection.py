"""Hand-written sparse MoE blocks are recognized and exported as SparseMoE."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cattorch import GenerationProgram, UnsupportedModelError, transpile, verify
from cattorch import CustomMoEWarning
from cattorch.operator_registry import default_registry

sys.path.insert(0, str(Path(__file__).parent))
from test_generation_automatic import load_sprite, prefill_and_decode  # noqa: E402
from cattorch.util.scratch.emulator import ScratchEmulator  # noqa: E402

WIDTH, EXPERTS = 8, 4


class SwiGLUExpert(nn.Module):
    def __init__(self, width=WIDTH, hidden=12):
        super().__init__()
        self.w1 = nn.Linear(width, hidden, bias=False)
        self.w2 = nn.Linear(hidden, width, bias=False)
        self.w3 = nn.Linear(width, hidden, bias=False)

    def forward(self, value):
        return self.w2(F.silu(self.w1(value)) * self.w3(value))


class GeluExpert(nn.Module):
    def __init__(self, width=WIDTH, hidden=10):
        super().__init__()
        self.up = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, value):
        return self.down(F.gelu(self.up(value)))


class LoopMoE(nn.Module):
    """The usual hand-written block: score, top-k, then loop over experts."""

    def __init__(self, scoring, normalize, top_k=2, expert=SwiGLUExpert, scale=1.0):
        super().__init__()
        self.gate = nn.Linear(WIDTH, EXPERTS, bias=False)
        self.experts = nn.ModuleList(expert() for _ in range(EXPERTS))
        self.num_experts_per_tok = top_k
        self.scoring = scoring
        self.normalize = normalize
        self.scale = scale

    def forward(self, value):
        shape = value.shape
        flat = value.reshape(-1, shape[-1])
        logits = self.gate(flat)
        if self.scoring == "softmax_after_topk":
            weights, ids = torch.topk(logits, self.num_experts_per_tok, dim=-1)
            weights = F.softmax(weights, dim=-1)
        else:
            scores = torch.sigmoid(logits) if self.scoring == "sigmoid" else logits.softmax(-1)
            weights, ids = torch.topk(scores, self.num_experts_per_tok, dim=-1)
            if self.normalize:
                weights = weights / weights.sum(dim=-1, keepdim=True)
        output = torch.zeros_like(flat)
        for index, expert in enumerate(self.experts):
            token, slot = torch.where(ids == index)
            if token.numel():
                output.index_add_(0, token, expert(flat[token]) * weights[token, slot, None])
        return (output * self.scale).reshape(shape)


CASES = {
    "mixtral": dict(scoring="softmax_after_topk", normalize=True),
    "switch top-1": dict(scoring="softmax", normalize=False, top_k=1),
    "softmax unnormalized": dict(scoring="softmax", normalize=False),
    "sigmoid normalized": dict(scoring="sigmoid", normalize=True, expert=GeluExpert),
    "sigmoid unnormalized": dict(scoring="sigmoid", normalize=False, top_k=3),
}
DESCRIPTIONS = {
    "mixtral": "top_k=2, softmax scores, renormalized weights",
    "switch top-1": "top_k=1, softmax scores, unnormalized weights",
    "softmax unnormalized": "top_k=2, softmax scores, unnormalized weights",
    "sigmoid normalized": "top_k=2, sigmoid scores, renormalized weights",
    "sigmoid unnormalized": "top_k=3, sigmoid scores, unnormalized weights",
}


@torch.no_grad()
@pytest.mark.parametrize("case", sorted(CASES))
def test_hand_written_moe_blocks_export_as_sparse_moe(tmp_path, case):
    torch.manual_seed(0)
    model = LoopMoE(**CASES[case]).eval()
    value = torch.randn(1, 5, WIDTH)
    with pytest.warns(CustomMoEWarning, match=DESCRIPTIONS[case]):
        result = transpile(model, value, tmp_path / "moe")
    comparison = verify(model, value, result, atol=1e-4)
    assert comparison.passed, comparison


def test_detection_leaves_the_original_model_alone():
    model = LoopMoE("softmax", normalize=True).eval()
    with pytest.warns(CustomMoEWarning):
        default_registry().clone().adapt_model(model)
    assert isinstance(model.experts, nn.ModuleList) and model.num_experts_per_tok == 2


def test_unmatched_moe_that_cannot_trace_points_to_the_explicit_api(tmp_path):
    # A routed scaling factor is outside the configurations cattorch tries.
    model = LoopMoE("softmax", normalize=True, scale=2.5).eval()
    with pytest.raises(UnsupportedModelError, match="ExpertFamily and SparseMoE"):
        transpile(model, torch.randn(1, 5, WIDTH), tmp_path / "unmatched")


class DenseEnsemble(nn.Module):
    """Same parts as an MoE block, but every expert runs: it traces as it is."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(WIDTH, EXPERTS)
        self.experts = nn.ModuleList(GeluExpert() for _ in range(EXPERTS))

    def forward(self, value):
        mix = torch.tanh(self.gate(value))
        stacked = torch.stack([expert(value) for expert in self.experts], dim=-1)
        return (stacked * mix.unsqueeze(-2)).sum(-1)


def test_traceable_lookalike_exports_unchanged(tmp_path, recwarn):
    torch.manual_seed(1)
    model = DenseEnsemble().eval()
    value = torch.randn(1, 3, WIDTH)
    result = transpile(model, value, tmp_path / "dense")
    assert verify(model, value, result).passed
    assert not [w for w in recwarn if issubclass(w.category, CustomMoEWarning)]


class MoEDecoder(nn.Module):
    def __init__(self, vocab=24, context=16):
        super().__init__()
        self.wte = nn.Embedding(vocab, WIDTH)
        self.wpe = nn.Embedding(context, WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.norm = nn.LayerNorm(WIDTH)
        self.moe = LoopMoE("softmax_after_topk", normalize=True)
        self.head = nn.Linear(WIDTH, vocab, bias=False)

    def forward(self, tokens):
        x = self.wte(tokens) + self.wpe(torch.arange(tokens.shape[1]))
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        x = x + F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.moe(self.norm(x))
        return self.head(x)


@torch.no_grad()
def test_detected_moe_works_in_cached_generation(tmp_path):
    torch.manual_seed(2)
    model = MoEDecoder().eval()
    with pytest.warns(CustomMoEWarning):
        result = transpile(model, GenerationProgram(max_context=8), tmp_path / "decoder")
    emulator = ScratchEmulator(load_sprite(result.path))
    prefill_and_decode(emulator, model, [3, 1, 4], [1, 5])
