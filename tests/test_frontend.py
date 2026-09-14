"""Differential coverage for the FX/export capture transition."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time

import pytest
import torch
import torch.nn as nn

import cattorch.frontend as frontend_module
from cattorch import UnsupportedModelError, UnsupportedOperationError, transpile, verify
from cattorch.frontend import capture_model
from cattorch.operator_registry import OperationRule, default_registry
from cattorch.transpiler import _transpile
from cattorch.util.instruction.optimized import OptimizedElementwiseInstruction


class StatefulModel(nn.Module):
    def __init__(self):
        super().__init__()
        weight = nn.Parameter(torch.randn(4, 4))
        self.weight = weight
        self.tied_weight = weight
        self.register_buffer("bias", torch.randn(4))
        self.constant = torch.arange(4, dtype=torch.float32)

    def forward(self, value, residual):
        return torch.relu(value @ self.weight + self.bias + self.constant + residual)


@pytest.mark.parametrize("frontend", ["fx", "export"])
def test_frontends_lower_parameters_buffers_constants_and_multiple_inputs(tmp_path, frontend):
    torch.manual_seed(123)
    model = StatefulModel().eval()
    inputs = (torch.randn(2, 4), torch.randn(2, 4))
    result = transpile(model, inputs, tmp_path / frontend, frontend=frontend)

    comparison = verify(model, inputs, result, atol=2e-4)
    assert comparison.passed, comparison
    assert tuple(item.list_name for item in result.inputs) == ("input", "input_1")
    assert result.output.shape == (2, 4)


def test_frontends_have_equivalent_public_interfaces(tmp_path):
    model = StatefulModel().eval()
    inputs = (torch.randn(2, 4), torch.randn(2, 4))
    fx = transpile(model, inputs, tmp_path / "fx", frontend="fx")
    exported = transpile(model, inputs, tmp_path / "export", frontend="export")

    assert fx.inputs == exported.inputs
    assert fx.output == exported.output
    assert set(fx.procedures) == set(exported.procedures)


class NestedLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(nn.Linear(3, 2))

    def forward(self, value):
        return self.stack(value)


def test_fx_capture_preserves_nested_module_context():
    captured = capture_model(NestedLinear().eval(), (torch.randn(1, 3),), frontend="fx")
    linear = next(node for node in captured.graph.nodes if str(node.target) == "aten.linear.default")
    paths = [entry[0] for entry in linear.meta["nn_module_stack"].values()]
    assert "stack.0" in paths


def test_public_capture_serializes_process_global_tracing_state(monkeypatch):
    guard = Lock()
    active = 0
    maximum = 0

    def fake_capture(model, example_inputs):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        with guard:
            active -= 1
        return model, example_inputs

    monkeypatch.setattr(frontend_module, "_capture_fx", fake_capture)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(
            lambda index: capture_model(index, (index,), frontend="fx"),
            range(8),
        ))

    assert results == [(index, (index,)) for index in range(8)]
    assert maximum == 1


@pytest.mark.parametrize("frontend", ["fx", "export"])
def test_frontends_preserve_multiple_tensor_outputs(tmp_path, frontend):
    class Multiple(nn.Module):
        def forward(self, value):
            return value, -value

    value = torch.ones(2)
    result = transpile(
        Multiple(), value, tmp_path / f"multiple-{frontend}", frontend=frontend,
    )
    assert tuple(output.list_name for output in result.outputs) == ("output", "output_1")
    assert all(output.shape == (2,) for output in result.outputs)


def test_fx_rejects_data_dependent_python_control_flow(tmp_path):
    class DataDependent(nn.Module):
        def forward(self, value):
            if value.sum().item() > 0:
                return value + 1
            return value - 1

    with pytest.raises(UnsupportedModelError, match="FX frontend could not trace"):
        transpile(DataDependent(), torch.ones(2), tmp_path / "data-dependent")


def test_public_transpile_captures_an_export_owned_copy(tmp_path):
    class StatefulForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            return -value

    model = StatefulForward()
    transpile(model, torch.ones(2), tmp_path / "owned")
    assert model.calls == 0


@torch.library.custom_op("cattorch_test::neg", mutates_args=())
def _custom_neg(value: torch.Tensor) -> torch.Tensor:
    return -value


@_custom_neg.register_fake
def _custom_neg_fake(value):
    return torch.empty_like(value)


class AdaptedNeg(nn.Module):
    def forward(self, value):
        return -value


class OpaqueNeg(nn.Module):
    def forward(self, value):
        return _custom_neg(value)


def test_private_registry_supports_module_adapters_and_semantic_ops(tmp_path):
    registry = default_registry().clone()
    registry.register_module_adapter(AdaptedNeg, lambda _module: OpaqueNeg())
    registry.register_operation(OperationRule(
        target="cattorch_test.neg.default",
        exact_kernel=OptimizedElementwiseInstruction,
        lowering_target=torch.ops.aten.neg.default,
    ))
    model = AdaptedNeg()
    result = _transpile(
        model,
        torch.tensor([1.0, -2.0]),
        tmp_path / "registered",
        _registry=registry,
    )
    assert verify(model, torch.tensor([1.0, -2.0]), result).passed


def test_unknown_semantic_op_still_fails_closed(tmp_path):
    with pytest.raises(UnsupportedOperationError, match="cattorch_test.neg.default"):
        transpile(OpaqueNeg(), torch.ones(2), tmp_path / "unknown")


def test_runtime_full_slice_followed_by_view_resolves_complete_alias_chain(tmp_path):
    class FinalTokenProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(5, 4))

        def forward(self, value):
            selected = value[:, 0:1]
            return torch.nn.functional.linear(selected.view(1, 4), self.weight)

    model = FinalTokenProjection().eval()
    value = torch.randn(1, 1, 4)
    result = transpile(model, value, tmp_path / "alias-chain")
    assert verify(model, value, result).passed


def test_fx_decomposes_causal_sdpa_and_mqa_repeat_interleave(tmp_path):
    class Attention(nn.Module):
        def forward(self, query, key, value):
            key = key.repeat_interleave(2, dim=1)
            value = value.repeat_interleave(2, dim=1)
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=True,
            )

    inputs = (
        torch.randn(1, 2, 3, 2),
        torch.randn(1, 1, 3, 2),
        torch.randn(1, 1, 3, 2),
    )
    model = Attention().eval()
    result = transpile(model, inputs, tmp_path / "sdpa")
    assert verify(model, inputs, result, atol=2e-4).passed


def test_positive_strided_slice_lowers_without_an_index_map(tmp_path):
    class EvenFeatures(nn.Module):
        def forward(self, value):
            return value[:, ::2]

    model = EvenFeatures()
    value = torch.randn(3, 7)
    result = transpile(model, value, tmp_path / "strided")
    assert verify(model, value, result).passed


def test_frontend_argument_is_validated(tmp_path):
    with pytest.raises(ValueError, match="frontend must be"):
        transpile(nn.Identity(), torch.ones(1), tmp_path / "bad", frontend="other")
