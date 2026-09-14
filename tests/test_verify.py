"""Verification of ordinary tensor and pytree model outputs."""

import pytest
import torch

from cattorch import (
    CodegenConfig, MultiOutputVerifyResult, UnsupportedModelError, VerifyResult,
    transpile, verify,
)


class Outputs(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, value):
        return self.function(value)


@pytest.mark.parametrize("frontend", ["fx", "export"])
@pytest.mark.parametrize("use_path", [False, True])
def test_verify_nested_outputs_in_export_order(tmp_path, frontend, use_path):
    model = Outputs(lambda x: {"z": x, "a": (-x, [x.mean(dim=1), x])})
    value = torch.tensor([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    artifact = transpile(
        model, value, tmp_path / "nested", frontend=frontend,
        codegen=CodegenConfig(compact_internal_names=True),
    )
    result = verify(model, value, artifact.path if use_path else artifact)
    assert isinstance(result, MultiOutputVerifyResult)
    assert result.passed
    assert [output.expected_shape for output in result.outputs] == [
        (2, 3), (2, 3), (2,), (2, 3),
    ]
    assert all(output.passed and output.max_abs_error < 1e-6 for output in result.outputs)


@pytest.mark.parametrize("wrapped", [False, True])
def test_single_tensor_verification_keeps_existing_result(tmp_path, wrapped):
    model = Outputs(lambda x: (x,) if wrapped else x)
    value = torch.ones(2)
    artifact = transpile(model, value, tmp_path / "single")
    result = verify(model, value, artifact)
    assert isinstance(result, VerifyResult)
    assert result.passed
    assert result.expected_shape == (2,)


@pytest.mark.parametrize("length_mismatch", [False, True])
def test_verify_checks_later_outputs_independently(tmp_path, length_mismatch):
    value = torch.tensor([1.0, 2.0, 4.0])
    artifact = transpile(Outputs(lambda x: (x, -x)), value, tmp_path / "pair")
    reference = Outputs(lambda x: (x, -x[:1] if length_mismatch else -x + 1))
    result = verify(reference, value, artifact)
    assert not result.passed
    assert result.outputs[0].passed
    assert not result.outputs[1].passed
    assert result.outputs[1].max_abs_error == (float("inf") if length_mismatch else 1.0)


@pytest.mark.parametrize("reference", [
    lambda x: x,
    lambda x: (x, -x, x),
    lambda x: (x, 3),
    lambda x: (),
])
def test_verify_rejects_incompatible_output_contract(tmp_path, reference):
    value = torch.ones(2)
    artifact = transpile(Outputs(lambda x: (x, -x)), value, tmp_path / "pair")
    with pytest.raises(UnsupportedModelError, match="output"):
        verify(Outputs(reference), value, artifact.path)


def test_verify_empty_output_and_sharded_later_output(tmp_path):
    model = Outputs(lambda x: (x[:0], x))
    value = torch.arange(200_001, dtype=torch.float32)
    artifact = transpile(
        model, value, tmp_path / "sharded",
        codegen=CodegenConfig(compact_internal_names=True),
    )
    assert "output_1" in artifact.sharded_lists
    result = verify(model, value, artifact.path)
    assert result.passed
    assert result.outputs[0].values_compared == 0
    assert result.outputs[0].worst_index is None
    assert result.outputs[1].values_compared == value.numel()
