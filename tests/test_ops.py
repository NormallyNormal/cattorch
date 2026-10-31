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


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class HalfRotaryModel(nn.Module):
    """Hugging Face style RoPE, written the way Llama's modeling code is."""

    def forward(self, value, cosine, sine):
        value = value.transpose(1, 2)
        return _rotate_half(value) * sine + cosine * value


def _half_inputs():
    return (torch.randn(1, 5, 2, 8), torch.randn(1, 1, 5, 8), torch.randn(5, 8))


def test_rotate_half_is_recognized_as_one_operation():
    from cattorch.graph import _fuse_rotate_half

    captured = capture_model(HalfRotaryModel(), _half_inputs(), frontend="fx")
    _fuse_rotate_half(captured.graph)
    targets = [str(node.target) for node in captured.graph.nodes if node.op == "call_function"]
    assert targets == ["aten.transpose.int", "cattorch.rotary_half.default"]


def test_rotate_half_lookalikes_are_left_alone():
    from cattorch.graph import _fuse_rotate_half

    class Swapped(nn.Module):
        def forward(self, value, cosine, sine):
            half = value.shape[-1] // 2
            turned = torch.cat((value[..., :half], -value[..., half:]), dim=-1)
            return value * cosine + turned * sine

    class Expanding(HalfRotaryModel):
        pass

    for model, inputs in (
        (Swapped(), (torch.randn(2, 8), torch.randn(8), torch.randn(8))),
        (Expanding(), (torch.randn(1, 5, 1, 8), torch.randn(3, 5, 8), torch.randn(5, 8))),
    ):
        captured = capture_model(model, inputs, frontend="fx")
        _fuse_rotate_half(captured.graph)
        assert not any(
            str(node.target) == "cattorch.rotary_half.default" for node in captured.graph.nodes
        )


def test_rotate_half_scratch_lowering_matches_pytorch(tmp_path):
    torch.manual_seed(82)
    model = HalfRotaryModel().eval()
    for index, inputs in enumerate((
        _half_inputs(),
        (torch.randn(1, 1, 3, 6), torch.randn(6), torch.randn(1, 6)),
    )):
        result = transpile(
            model, inputs, tmp_path / f"half{index}", storage=StorageConfig(compression=False),
        )
        assert verify(model, inputs, result, atol=1e-5, rtol=0).passed
