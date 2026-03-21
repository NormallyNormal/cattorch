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


def _run_sprite(model: nn.Module, input_shape: torch.Size, x: torch.Tensor) -> tuple[list[float], list[float]]:
    """Transpile a model, run through emulator, return (expected, actual)."""
    with torch.no_grad():
        expected = model(x).flatten().tolist()

    path = os.path.join(os.path.dirname(__file__), "_test_output")
    transpile(model, input_shape, path)

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


def _assert_close(expected: list[float], actual: list[float]):
    assert len(expected) == len(actual), (
        f"Length mismatch: expected {len(expected)}, got {len(actual)}"
    )
    for i, (e, a) in enumerate(zip(expected, actual)):
        assert abs(e - a) < TOLERANCE, (
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


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


def test_single_matmul():
    model = SingleMatMul()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, torch.Size([2, 4]), x)
    _assert_close(expected, actual)


def test_chained_matmul():
    model = ChainedMatMul()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, torch.Size([2, 4]), x)
    _assert_close(expected, actual)


def test_matmul_with_bias():
    model = MatMulWithBias()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, torch.Size([2, 4]), x)
    _assert_close(expected, actual)


def test_matmul_relu():
    model = MatMulReLU()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, torch.Size([2, 4]), x)
    _assert_close(expected, actual)


def test_simple_neural_net():
    model = SimpleNN()
    x = torch.randn(3, 4)
    expected, actual = _run_sprite(model, torch.Size([3, 4]), x)
    _assert_close(expected, actual)


def test_linear():
    model = LinearModel()
    x = torch.randn(1, 2)
    expected, actual = _run_sprite(model, torch.Size([1, 2]), x)
    _assert_close(expected, actual)


def test_linear_no_bias():
    model = LinearNoBias()
    x = torch.randn(1, 2)
    expected, actual = _run_sprite(model, torch.Size([1, 2]), x)
    _assert_close(expected, actual)


def test_stacked_linear():
    model = StackedLinear()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, torch.Size([2, 4]), x)
    _assert_close(expected, actual)


def test_softmax_1d():
    model = Softmax1D()
    x = torch.randn(4)
    expected, actual = _run_sprite(model, torch.Size([4]), x)
    _assert_close(expected, actual)


def test_softmax_2d():
    model = Softmax2D()
    x = torch.randn(2, 3)
    expected, actual = _run_sprite(model, torch.Size([2, 3]), x)
    _assert_close(expected, actual)


def test_softmax_3d():
    model = Softmax3D()
    x = torch.randn(3, 4, 5)
    expected, actual = _run_sprite(model, torch.Size([3, 4, 5]), x)
    _assert_close(expected, actual)


def test_scalar_multiply():
    model = ScalarMul()
    x = torch.randn(2, 3)
    expected, actual = _run_sprite(model, torch.Size([2, 3]), x)
    _assert_close(expected, actual)


def test_scalar_multiply_chained():
    model = ScalarMulChained()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, torch.Size([2, 4]), x)
    _assert_close(expected, actual)


def test_layernorm():
    model = LayerNorm1D()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, torch.Size([2, 4]), x)
    _assert_close(expected, actual)


def test_layernorm_2d():
    model = LayerNorm2D()
    x = torch.randn(2, 3, 4)
    expected, actual = _run_sprite(model, torch.Size([2, 3, 4]), x)
    _assert_close(expected, actual)


def test_view_reshape():
    model = ViewReshape()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, torch.Size([2, 4]), x)
    _assert_close(expected, actual)


def test_flatten():
    model = FlattenModel()
    x = torch.randn(2, 4)
    expected, actual = _run_sprite(model, torch.Size([2, 4]), x)
    _assert_close(expected, actual)


def test_scalar_divide():
    model = ScalarDiv()
    x = torch.randn(2, 3)
    expected, actual = _run_sprite(model, torch.Size([2, 3]), x)
    _assert_close(expected, actual)


def test_single_head_attention():
    model = SingleHeadAttention(d_model=8)
    x = torch.randn(3, 8)
    expected, actual = _run_sprite(model, torch.Size([3, 8]), x)
    _assert_close(expected, actual)


def test_transformer_block():
    model = TransformerBlock(d_model=8, d_ff=16)
    x = torch.randn(3, 8)
    expected, actual = _run_sprite(model, torch.Size([3, 8]), x)
    _assert_close(expected, actual)
