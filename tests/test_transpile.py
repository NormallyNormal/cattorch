"""
End-to-end transpilation tests.

Each test transpiles a PyTorch model, runs the resulting Scratch sprite
through the emulator, and compares the output against PyTorch.
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

    path = os.path.join(os.path.dirname(__file__), "_test_output")
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


class SingleMatMul(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(4, 3))

    def forward(self, x):
        return x @ self.W


class ChainedMatMul(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(4, 8))
        self.W2 = nn.Parameter(torch.randn(8, 2))

    def forward(self, x):
        return (x @ self.W1) @ self.W2


class MatMulWithBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(4, 3))
        self.b = nn.Parameter(torch.randn(3))

    def forward(self, x):
        return x @ self.W + self.b


class MatMulReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(4, 3))
        self.b = nn.Parameter(torch.randn(3))

    def forward(self, x):
        return F.relu(x @ self.W + self.b)


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(4, 8))
        self.b1 = nn.Parameter(torch.randn(8))
        self.W2 = nn.Parameter(torch.randn(8, 2))
        self.b2 = nn.Parameter(torch.randn(2))

    def forward(self, x):
        hidden = F.relu(x @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3)

    def forward(self, x):
        return self.linear(x)


class LinearNoBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3, bias=False)

    def forward(self, x):
        return self.linear(x)


class StackedLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Softmax1D(nn.Module):
    def forward(self, x):
        return F.softmax(x, dim=0)


class Softmax2D(nn.Module):
    def forward(self, x):
        return F.softmax(x, dim=-1)


class Softmax3D(nn.Module):
    def forward(self, x):
        return F.softmax(x, dim=1)


class ScalarMul(nn.Module):
    def forward(self, x):
        return x * 0.5


class ScalarMulChained(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(4, 3))

    def forward(self, x):
        return (x @ self.W) * 0.125


class LayerNorm1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(4)

    def forward(self, x):
        return self.ln(x)


class LayerNorm2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm([3, 4])

    def forward(self, x):
        return self.ln(x)


class ViewReshape(nn.Module):
    """View/reshape as a no-op (same flat data, different logical shape)."""
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(4, 6))

    def forward(self, x):
        y = x @ self.W            # [2, 6]
        y = y.view(3, 4)          # reshape, same flat data
        return y * 0.5


class FlattenModel(nn.Module):
    """Flatten uses reshape internally."""
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(4, 3))

    def forward(self, x):
        y = x @ self.W            # [2, 3]
        return y.reshape(-1)      # flatten to [6]


class ScalarDiv(nn.Module):
    def forward(self, x):
        return x / 4.0


class SigmoidModel(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)


class TanhModel(nn.Module):
    def forward(self, x):
        return torch.tanh(x)


class GELUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, x):
        return self.gelu(x)


class SiLUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(x)


class LeakyReLUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        return self.leaky_relu(x)


class ELUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.elu = nn.ELU(alpha=1.0)

    def forward(self, x):
        return self.elu(x)


class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(8, 4)

    def forward(self, x):
        return self.emb(x)


class EmbeddingLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(8, 4)
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(self.emb(x))


class TensorMultiply(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4, bias=False)
        self.fc2 = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.fc1(x) * self.fc2(x)


class MaskedFillModel(nn.Module):
    def __init__(self):
        super().__init__()
        mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
        self.register_buffer('mask', mask)

    def forward(self, x):
        return x.masked_fill(self.mask, float('-inf'))


class RMSNorm1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.rms = nn.RMSNorm(4)

    def forward(self, x):
        return self.rms(x)


class RMSNorm2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.rms = nn.RMSNorm([3, 4])

    def forward(self, x):
        return self.rms(x)


class SplitAdd(nn.Module):
    def forward(self, x):
        a, b, c = x.split(4, dim=-1)
        return a + b + c


class ChunkAdd(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a + b


class CatDim0(nn.Module):
    """Concatenate two linear outputs along dim 0."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3, bias=False)
        self.fc2 = nn.Linear(4, 3, bias=False)

    def forward(self, x):
        a = self.fc1(x)  # [2, 3]
        b = self.fc2(x)  # [2, 3]
        return torch.cat([a, b], dim=0)  # [4, 3]


class CatDim1(nn.Module):
    """Concatenate two linear outputs along dim 1."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3, bias=False)
        self.fc2 = nn.Linear(4, 5, bias=False)

    def forward(self, x):
        a = self.fc1(x)  # [2, 3]
        b = self.fc2(x)  # [2, 5]
        return torch.cat([a, b], dim=1)  # [2, 8]


class CatThreeWay(nn.Module):
    """Concatenate three linear outputs along dim 1 (chained cat)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 2, bias=False)
        self.fc2 = nn.Linear(4, 3, bias=False)
        self.fc3 = nn.Linear(4, 5, bias=False)

    def forward(self, x):
        a = self.fc1(x)  # [2, 2]
        b = self.fc2(x)  # [2, 3]
        c = self.fc3(x)  # [2, 5]
        return torch.cat([a, b, c], dim=1)  # [2, 10]


class SplitCat(nn.Module):
    """Split then cat back together (round-trip)."""
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        a = a * 2.0
        return torch.cat([a, b], dim=-1)


class NegateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return -self.fc(x)


class TensorSubtract(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4, bias=False)
        self.fc2 = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.fc1(x) - self.fc2(x)


class RoPEModel(nn.Module):
    """Rotary position embeddings with precomputed cos/sin buffers."""
    def __init__(self, dim=8, max_len=16):
        super().__init__()
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        angles = torch.outer(t, freqs)
        self.register_buffer('cos_cached', angles.cos())
        self.register_buffer('sin_cached', angles.sin())

    def forward(self, x):
        seq_len = x.shape[0]
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class ArangeAdd(nn.Module):
    """Uses torch.arange inside forward (common for positional embeddings)."""
    def forward(self, x):
        pos = torch.arange(x.shape[-1]).float()
        return x + pos


class SingleHeadAttention(nn.Module):
    """Minimal single-head self-attention block."""
    def __init__(self, d_model=8):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = F.softmax((q @ k.T) / self.scale, dim=-1)
        return self.out_proj(attn @ v)


class TransformerBlock(nn.Module):
    """Single transformer block: self-attention + FFN with residuals and layernorm."""
    def __init__(self, d_model=8, d_ff=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Self-attention with residual
        h = self.ln1(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        attn = F.softmax((q @ k.T) / self.scale, dim=-1)
        x = x + self.out_proj(attn @ v)
        # FFN with residual
        h = self.ln2(x)
        x = x + self.ff2(F.relu(self.ff1(h)))
        return x

class OnesAdd(nn.Module):
    def forward(self, x):
        t = torch.ones(x.shape).float()
        return x + t


class ZerosAdd(nn.Module):
    def forward(self, x):
        t = torch.zeros(x.shape).float()
        return x + t


class FullAdd(nn.Module):
    def forward(self, x):
        t = torch.full(x.shape, 2.5).float()
        return x + t


class OnesLikeAdd(nn.Module):
    def forward(self, x):
        t = torch.ones_like(x).float()
        return x + t


class ZerosLikeAdd(nn.Module):
    def forward(self, x):
        t = torch.zeros_like(x).float()
        return x + t


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


def test_single_matmul():
    model = SingleMatMul()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_chained_matmul():
    model = ChainedMatMul()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_matmul_with_bias():
    model = MatMulWithBias()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_matmul_relu():
    model = MatMulReLU()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_simple_neural_net():
    model = SimpleNN()
    x = torch.randn(3, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_linear():
    model = LinearModel()
    x = torch.randn(1, 2)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_linear_no_bias():
    model = LinearNoBias()
    x = torch.randn(1, 2)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_stacked_linear():
    model = StackedLinear()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_softmax_1d():
    model = Softmax1D()
    x = torch.randn(4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_softmax_2d():
    model = Softmax2D()
    x = torch.randn(2, 3)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_softmax_3d():
    model = Softmax3D()
    x = torch.randn(3, 4, 5)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_scalar_multiply():
    model = ScalarMul()
    x = torch.randn(2, 3)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_scalar_multiply_chained():
    model = ScalarMulChained()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_layernorm():
    model = LayerNorm1D()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_layernorm_2d():
    model = LayerNorm2D()
    x = torch.randn(2, 3, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_view_reshape():
    model = ViewReshape()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_flatten():
    model = FlattenModel()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_scalar_divide():
    model = ScalarDiv()
    x = torch.randn(2, 3)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_embedding():
    model = EmbeddingModel()
    x = torch.tensor([0, 3, 7, 1])
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_embedding_linear():
    model = EmbeddingLinear()
    x = torch.tensor([2, 5, 0])
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_tensor_multiply():
    model = TensorMultiply()
    x = torch.randn(3, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_masked_fill():
    model = MaskedFillModel()
    x = torch.randn(3, 3)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_rms_norm():
    model = RMSNorm1D()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_rms_norm_2d():
    model = RMSNorm2D()
    x = torch.randn(2, 3, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_split():
    model = SplitAdd()
    x = torch.randn(2, 12)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_chunk():
    model = ChunkAdd()
    x = torch.randn(2, 8)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_negate():
    model = NegateModel()
    x = torch.randn(3, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_tensor_subtract():
    model = TensorSubtract()
    x = torch.randn(3, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_cat_dim0():
    model = CatDim0()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_cat_dim1():
    model = CatDim1()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_cat_three_way():
    model = CatThreeWay()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_split_cat():
    model = SplitCat()
    x = torch.randn(2, 8)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_arange():
    model = ArangeAdd()
    x = torch.randn(4, 3)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_rope():
    model = RoPEModel(dim=8, max_len=16)
    x = torch.randn(4, 8)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_single_head_attention():
    model = SingleHeadAttention(d_model=8)
    x = torch.randn(3, 8)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_transformer_block():
    model = TransformerBlock(d_model=8, d_ff=16)
    x = torch.randn(3, 8)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_sigmoid():
    model = SigmoidModel()
    x = torch.randn(2, 3)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_tanh():
    model = TanhModel()
    x = torch.randn(2, 3)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_gelu():
    model = GELUModel()
    x = torch.randn(2, 3)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_silu():
    model = SiLUModel()
    x = torch.randn(2, 3)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_leaky_relu():
    model = LeakyReLUModel()
    x = torch.randn(2, 3)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_elu():
    model = ELUModel()
    x = torch.randn(2, 3)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


# ── Edge case / chained op tests ─────────────────────────────────────────────


class ChainedNoOps(nn.Module):
    """View → reshape → contiguous → scalar op. Tests alias chain resolution."""
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(4, 6))

    def forward(self, x):
        y = x @ self.W            # [2, 6]
        y = y.view(3, 4)
        y = y.reshape(12)
        y = y.contiguous()
        return y * 0.5


class FanOut(nn.Module):
    """One intermediate used by two downstream ops. Tests ref counting."""
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(4, 4))

    def forward(self, x):
        h = x @ self.W
        return h + h              # same tensor added to itself


class FanOutDivergent(nn.Module):
    """One intermediate feeds two different op types. Tests list not freed early."""
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(4, 4))
        self.W2 = nn.Parameter(torch.randn(4, 4))

    def forward(self, x):
        h = F.relu(x)
        a = h @ self.W1
        b = h @ self.W2
        return a + b


class NoOpOutput(nn.Module):
    """Final node is a no-op (view). Tests output alias resolution."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 6)

    def forward(self, x):
        return self.linear(x).view(2, 3)


class SoftmaxThenTranspose(nn.Module):
    """Softmax and transpose both have local lists. Tests no ID collision."""
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(4, 3))

    def forward(self, x):
        h = x @ self.W           # [2, 3]
        h = F.softmax(h, dim=-1)
        return h.T               # [3, 2]


class ResidualWithActivations(nn.Module):
    """Residual + two different activations. Tests scope reuse under pressure."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x):
        h = torch.sigmoid(self.fc1(x))
        return x + torch.tanh(self.fc2(h))


class DeepChain(nn.Module):
    """6 operations chained. Tests sustained list reuse without corruption."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return self.fc3(x)


class LayerNormSoftmaxChain(nn.Module):
    """LayerNorm → linear → softmax. Both LN and softmax have group logic."""
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(4)
        self.W = nn.Parameter(torch.randn(4, 3))

    def forward(self, x):
        h = self.ln(x)
        return F.softmax(h @ self.W, dim=-1)


class ScalarChain(nn.Module):
    """Chained scalar ops. Tests multiple scalar templates don't collide."""
    def forward(self, x):
        return (x * 2.0) / 3.0 * 0.5


class ActivationSandwich(nn.Module):
    """Different activation on each layer. Tests template reuse with variety."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 3)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = F.silu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class DoubleResidual(nn.Module):
    """Two residual connections. Tests scope with multiple fan-out/fan-in."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x):
        x = x + F.relu(self.fc1(x))
        x = x + F.relu(self.fc2(x))
        return x


class Conv1dSimple(nn.Module):
    """Basic 1D convolution."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 3, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


class Conv2dSimple(nn.Module):
    """Basic 2D convolution."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


class Conv2dNoBias(nn.Module):
    """Conv2d without bias."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class Conv2dStride(nn.Module):
    """Conv2d with stride > 1."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Conv2dReLU(nn.Module):
    """Conv2d followed by ReLU activation."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)

    def forward(self, x):
        return F.relu(self.conv(x))


class TinyConvNet(nn.Module):
    """Two conv layers with ReLU and a linear head."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        self.fc = nn.Linear(2 * 4 * 4, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        return self.fc(x)


class PowSquared(nn.Module):
    """x^2 via torch.pow — tests the optimised x*x path."""
    def forward(self, x):
        return torch.pow(x, 2)


class PowGeneral(nn.Module):
    """x^1.5 via torch.pow — tests the general e^(n*ln(x)) path."""
    def forward(self, x):
        return torch.pow(x, 1.5)


class RSqrtModel(nn.Module):
    """1/sqrt(x) — tests rsqrt instruction."""
    def forward(self, x):
        return torch.rsqrt(x)


class MeanLastDim(nn.Module):
    """Mean over last dimension."""
    def forward(self, x):
        return x.mean(dim=-1)


class MeanFirstDim(nn.Module):
    """Mean over first dimension of a 2D tensor."""
    def forward(self, x):
        return x.mean(dim=0)


class PowMeanRsqrt(nn.Module):
    """pow → mean → rsqrt chain (core of manual RMSNorm, without broadcasting)."""
    def forward(self, x):
        ms = torch.pow(x, 2).mean(dim=-1)
        return torch.rsqrt(ms + 1e-6)


def test_chained_no_ops():
    model = ChainedNoOps()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_fan_out():
    model = FanOut()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_fan_out_divergent():
    model = FanOutDivergent()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_no_op_output():
    model = NoOpOutput()
    x = torch.randn(1, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_softmax_then_transpose():
    model = SoftmaxThenTranspose()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_residual_with_activations():
    model = ResidualWithActivations()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_deep_chain():
    model = DeepChain()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_layernorm_softmax_chain():
    model = LayerNormSoftmaxChain()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_scalar_chain():
    model = ScalarChain()
    x = torch.randn(2, 3)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_activation_sandwich():
    model = ActivationSandwich()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_double_residual():
    model = DoubleResidual()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_conv1d():
    model = Conv1dSimple()
    x = torch.randn(1, 2, 8)  # (batch, channels, length)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_conv2d():
    model = Conv2dSimple()
    x = torch.randn(1, 1, 5, 5)  # (batch, channels, H, W)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_conv2d_no_bias():
    model = Conv2dNoBias()
    x = torch.randn(1, 1, 5, 5)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_conv2d_stride():
    model = Conv2dStride()
    x = torch.randn(1, 1, 6, 6)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_conv2d_relu():
    model = Conv2dReLU()
    x = torch.randn(1, 1, 5, 5)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_tiny_conv_net():
    model = TinyConvNet()
    x = torch.randn(1, 1, 4, 4)  # small spatial dims to keep it fast
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_pow_squared():
    model = PowSquared()
    x = torch.randn(2, 3)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_pow_general():
    model = PowGeneral()
    x = torch.abs(torch.randn(2, 3)) + 0.1  # positive inputs for ln
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_rsqrt():
    model = RSqrtModel()
    x = torch.abs(torch.randn(2, 3)) + 0.1  # positive inputs
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_mean_last_dim():
    model = MeanLastDim()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_mean_first_dim():
    model = MeanFirstDim()
    x = torch.randn(3, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_pow_mean_rsqrt():
    model = PowMeanRsqrt()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_ones():
    model = OnesAdd()
    x = torch.randn(4, 3)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_zeros():
    model = ZerosAdd()
    x = torch.randn(4, 3)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_full():
    model = FullAdd()
    x = torch.randn(4, 3)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_ones_like():
    model = OnesLikeAdd()
    x = torch.randn(4, 3)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)


def test_zeros_like():
    model = ZerosLikeAdd()
    x = torch.randn(4, 3)
    expected, actual = _run_sprite(model, x)
    _assert_close(expected, actual)