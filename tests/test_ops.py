"""Tests for semantic operations with dedicated Scratch lowerings."""

from __future__ import annotations

import torch
import torch.nn as nn

from cattorch import StorageConfig, rotary_embedding, transpile, verify
from cattorch.frontend import capture_model


class RotaryModel(nn.Module):
    def forward(self, value, cosine, sine):
        return rotary_embedding(value, cosine, sine)


def _reference(value, cosine, sine):
    pairs = value.reshape(*value.shape[:-1], -1, 2)
    even, odd = pairs.unbind(-1)
    rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
    return value * cosine + rotated * sine


def test_rotary_embedding_eager_contract_and_validation():
    value = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)
    cosine = torch.linspace(0.2, 0.8, 4).reshape(1, 1, 1, 4)
    sine = torch.linspace(-0.3, 0.4, 4).reshape(1, 1, 1, 4)
    torch.testing.assert_close(
        rotary_embedding(value, cosine, sine),
        _reference(value, cosine, sine),
        rtol=0,
        atol=0,
    )

    try:
        rotary_embedding(torch.ones(1, 3), torch.ones(3), torch.ones(3))
    except ValueError as error:
        assert "even final dimension" in str(error)
    else:
        raise AssertionError("odd rotary width should be rejected")


def test_rotary_embedding_is_preserved_as_a_semantic_fx_operation():
    inputs = (
        torch.randn(1, 4, 1, 14),
        torch.randn(1, 1, 1, 14),
        torch.randn(1, 1, 1, 14),
    )
    captured = capture_model(RotaryModel(), inputs, frontend="fx")
    assert sum(
        str(node.target) == "cattorch.rotary_embedding.default"
        for node in captured.graph.nodes
    ) == 1


def test_rotary_embedding_scratch_lowering_matches_reference(tmp_path):
    torch.manual_seed(81)
    inputs = (
        torch.randn(1, 4, 1, 14),
        torch.randn(1, 1, 1, 14),
        torch.randn(1, 1, 1, 14),
    )
    model = RotaryModel().eval()
    result = transpile(
        model,
        inputs,
        tmp_path / "rotary",
        storage=StorageConfig(compression=False),
    )
    report = verify(model, inputs, result, atol=1e-5, rtol=0)
    assert report.passed
