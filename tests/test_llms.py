"""
End-to-end tests for LLM-style architectures.

Each test transpiles a small language model block, runs the resulting Scratch
sprite through the emulator, and compares the output against PyTorch.
"""

import json
import os
import zipfile

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cattorch.transpiler import transpile
from cattorch.util.scratch.emulator import ScratchEmulator

TOLERANCE = 1e-4


def _run_sprite(model: nn.Module, x: torch.Tensor) -> tuple[list[float], list[float]]:
    """Transpile a model, run through emulator, return (expected, actual)."""
    with torch.no_grad():
        expected = model(x).flatten().tolist()

    path = os.path.join(os.path.dirname(__file__), "_test_llm_output")
    transpile(model, x, path)

    sprite_path = path + ".sprite3"
    try:
        with zipfile.ZipFile(sprite_path, "r") as z:
            project = json.loads(z.read("sprite.json"))
    finally:
        os.remove(sprite_path)

    emu = ScratchEmulator(project)
    emu.lists["input"] = [float(v) for v in x.flatten().tolist()]
    emu.run()
    actual = [float(v) for v in emu.lists.get("output", [])]
    return expected, actual


def _assert_close(expected: list[float], actual: list[float], tolerance=TOLERANCE):
    assert len(expected) == len(actual), (
        f"Length mismatch: expected {len(expected)}, got {len(actual)}"
    )
    for i, (e, a) in enumerate(zip(expected, actual)):
        if e == a:
            continue
        assert abs(e - a) < tolerance, (
            f"Index {i}: expected {e:.6f}, got {a:.6f}"
        )


# ── Models ────────────────────────────────────────────────────────────────────


class SimpleCausalLM(nn.Module):
    """Embedding + learned positional embedding + causal attention + FFN.

    Tests: embedding, tensor add (positional + residual), masked_fill,
    softmax, matmul, layernorm, relu, linear.
    """
    def __init__(self, vocab_size=16, d_model=8, seq_len=4, d_ff=16):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.register_buffer('positions', torch.arange(seq_len))
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer('mask', mask)
        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        x = self.token_emb(tokens) + self.pos_emb(self.positions)
        h = self.ln1(x)
        q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        attn = (q @ k.T) / self.scale
        attn = attn.masked_fill(self.mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        x = x + self.out_proj(attn @ v)
        h = self.ln2(x)
        x = x + self.ff2(F.relu(self.ff1(h)))
        return self.head(x)


class MultiHeadAttentionBlock(nn.Module):
    """Two-head self-attention with separate Q/K/V projections.

    Tests: chunk (split into heads), cat (combine heads), multiple
    independent attention computations.
    """
    def __init__(self, d_model=8, n_heads=2):
        super().__init__()
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** 0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Split into 2 heads
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        v1, v2 = v.chunk(2, dim=-1)
        # Per-head attention
        a1 = F.softmax(q1 @ k1.T / self.scale, dim=-1) @ v1
        a2 = F.softmax(q2 @ k2.T / self.scale, dim=-1) @ v2
        return self.out_proj(torch.cat([a1, a2], dim=-1))


class CombinedQKVBlock(nn.Module):
    """Single QKV projection split three ways.

    Tests: 3-way chunk on a single large projection, efficient fused pattern.
    """
    def __init__(self, d_model=6):
        super().__init__()
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

    def forward(self, x):
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        attn = F.softmax(q @ k.T / self.scale, dim=-1)
        return self.out_proj(attn @ v)


class RoPEAttentionBlock(nn.Module):
    """Self-attention with rotary position embeddings.

    Tests: slice (cos/sin buffer slicing), chunk, sub, mul, cat (rotation),
    matmul, softmax.
    """
    def __init__(self, d_model=8, max_len=16):
        super().__init__()
        self.scale = d_model ** 0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        # Precomputed RoPE tables
        freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        t = torch.arange(max_len).float()
        angles = torch.outer(t, freqs)
        self.register_buffer('cos_cached', angles.cos())
        self.register_buffer('sin_cached', angles.sin())

    def _apply_rope(self, x):
        seq_len = x.shape[0]
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x):
        q = self._apply_rope(self.q_proj(x))
        k = self._apply_rope(self.k_proj(x))
        v = self.v_proj(x)
        attn = F.softmax(q @ k.T / self.scale, dim=-1)
        return self.out_proj(attn @ v)


class LLaMABlock(nn.Module):
    """LLaMA-style transformer block: RMSNorm + RoPE + SwiGLU FFN.

    Tests: rms_norm, RoPE (slice + chunk + cat + sub + mul), silu,
    tensor multiply (gate), no bias linears.
    """
    def __init__(self, d_model=8, d_ff=16, max_len=16):
        super().__init__()
        self.rn1 = nn.RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model ** 0.5
        # RoPE
        freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        t = torch.arange(max_len).float()
        angles = torch.outer(t, freqs)
        self.register_buffer('cos_cached', angles.cos())
        self.register_buffer('sin_cached', angles.sin())
        # SwiGLU FFN
        self.rn2 = nn.RMSNorm(d_model)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def _apply_rope(self, x):
        seq_len = x.shape[0]
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x):
        # Attention with RoPE
        h = self.rn1(x)
        q = self._apply_rope(self.q_proj(h))
        k = self._apply_rope(self.k_proj(h))
        v = self.v_proj(h)
        attn = F.softmax(q @ k.T / self.scale, dim=-1)
        x = x + self.out_proj(attn @ v)
        # SwiGLU FFN
        h = self.rn2(x)
        x = x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
        return x


class GPTBlock(nn.Module):
    """GPT-style block: LayerNorm + multi-head attention + GELU FFN.

    Tests: layernorm, multi-head (chunk + cat), gelu, residual connections.
    """
    def __init__(self, d_model=8, n_heads=2, d_ff=16):
        super().__init__()
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** 0.5
        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, x):
        h = self.ln1(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        v1, v2 = v.chunk(2, dim=-1)
        a1 = F.softmax(q1 @ k1.T / self.scale, dim=-1) @ v1
        a2 = F.softmax(q2 @ k2.T / self.scale, dim=-1) @ v2
        x = x + self.out_proj(torch.cat([a1, a2], dim=-1))
        h = self.ln2(x)
        x = x + self.ff2(self.gelu(self.ff1(h)))
        return x


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


def test_simple_causal_lm():
    model = SimpleCausalLM(vocab_size=16, d_model=8, seq_len=4, d_ff=16)
    x = torch.tensor([0, 5, 12, 3])
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_multi_head_attention():
    model = MultiHeadAttentionBlock(d_model=8, n_heads=2)
    x = torch.randn(4, 8)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_combined_qkv():
    model = CombinedQKVBlock(d_model=6)
    x = torch.randn(3, 6)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_rope_attention():
    model = RoPEAttentionBlock(d_model=8, max_len=16)
    x = torch.randn(4, 8)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_llama_block():
    model = LLaMABlock(d_model=8, d_ff=16, max_len=16)
    x = torch.randn(4, 8)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_gpt_block():
    model = GPTBlock(d_model=8, n_heads=2, d_ff=16)
    x = torch.randn(4, 8)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)
