"""Reporting operations that run as PyTorch decompositions."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cattorch import UnsupportedOperationError, transpile, verify
from cattorch.analysis import analyze
from cattorch.program import EntryPoint, ExportProgram, Input, Output


class Mixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer("table", torch.arange(4.0))

    def forward(self, x):
        # Computed from a buffer at export, so it never runs in Scratch.
        table = F.hardsigmoid(self.table)
        return F.mish(self.linear(x)) + F.mish(x) + F.hardsigmoid(x) + table


def test_transpile_reports_decomposed_operations(tmp_path):
    torch.manual_seed(0)
    model = Mixed().eval()
    inputs = (torch.randn(2, 4),)
    result = transpile(model, inputs, tmp_path / "mixed")
    assert result.fallback_operations == (
        ("aten.hardsigmoid.default", 1), ("aten.mish.default", 2),
    )
    assert any("aten.mish.default (2 calls)" in warning for warning in result.warnings)
    assert verify(model, inputs, result).passed


def test_supported_models_report_no_fallbacks(tmp_path):
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.LayerNorm(8)).eval()
    result = transpile(model, (torch.randn(1, 4),), tmp_path / "plain")
    assert result.fallback_operations == ()
    assert not any("decomposition" in warning for warning in result.warnings)


def test_analyze_reports_fallbacks():
    program = ExportProgram(entrypoints=(EntryPoint(
        "forward", arguments=(Input("value", torch.randn(2, 4)),),
        returns=(Output("value"),),
    ),))
    report = analyze(Mixed().eval(), program)
    assert dict(report.entrypoints[0].fallback_operations)["aten.mish.default"] == 2
    assert any("aten.mish.default" in warning for warning in report.warnings)


def test_addmm_with_a_bias_vector_is_a_linear_layer(tmp_path):
    """GPT-2's Conv1D calls ``addmm``; it should use the linear kernel."""
    torch.manual_seed(1)

    class Conv1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 6))
            self.bias = nn.Parameter(torch.randn(6))

        def forward(self, x):
            return torch.addmm(self.bias, x.view(-1, 4), self.weight)

    model = Conv1D().eval()
    inputs = (torch.randn(3, 4),)
    result = transpile(model, inputs, tmp_path / "conv1d")
    assert result.fallback_operations == ()
    assert verify(model, inputs, result).passed


@pytest.mark.parametrize("operation", [torch.sign, torch.round, lambda x: x % 1.5])
def test_pointwise_decompositions_match(tmp_path, operation):
    model = type("Model", (nn.Module,), {"forward": lambda self, x: operation(x)})().eval()
    inputs = (torch.tensor([[-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 3.7]]),)
    assert verify(model, inputs, transpile(model, inputs, tmp_path / "pointwise")).passed


def test_a_decomposition_cycle_names_the_operation(tmp_path):
    """``aten.pow.Tensor_Tensor`` decomposes to a primitive mapped back to it."""
    model = type("Model", (nn.Module,), {"forward": lambda self, x, y: x ** y})().eval()
    with pytest.raises(UnsupportedOperationError, match="aten.pow.Tensor_Tensor"):
        transpile(model, (torch.rand(3) + 1, torch.rand(3)), tmp_path / "pow")
