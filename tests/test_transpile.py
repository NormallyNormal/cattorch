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
