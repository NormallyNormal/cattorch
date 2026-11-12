"""Every calibration observer has a storage layout that keeps its weights in integers."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from cattorch import QuantizationConfig, transpile
from cattorch.experimental import ExpertFamily, SparseMoE, StackedSwiGLUMoE, routed_experts
from cattorch.operator_registry import _DECOMPOSED_OBSERVERS, default_registry


class _Model(nn.Module):
    def __init__(self, forward, **parts):
        super().__init__()
        self._forward = forward
        for name, value in parts.items():
            setattr(
                self, name,
                value if isinstance(value, nn.Module)
                else nn.Parameter(torch.randn(*value) * 0.3),
            )

    def forward(self, value):
        return self._forward(self, value)


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(16, 8)
        self.down = nn.Linear(8, 16)

    def forward(self, value):
        return self.down(torch.relu(self.up(value)))


def _family():
    return ExpertFamily.from_modules([_Expert(), _Expert()], example_input=torch.ones(1, 16))


class _Routed(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(16, 2, bias=False)
        self.experts = _family()

    def forward(self, value):
        weights, ids = torch.topk(torch.softmax(self.gate(value), -1), 1, dim=-1)
        return routed_experts(value, ids, weights, self.experts)


def _conv(dims, transposed=False, padding=0):
    layer = (
        (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d) if transposed
        else (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    )[dims - 1](4, 8, 3, padding=padding)
    shape = (1, 4) + (4,) * dims
    return _Model(
        lambda s, x: s.c(x[:1, :4 ** (dims + 1)].reshape(shape)).flatten()[None], c=layer,
    ), 4 ** (dims + 1)


def _models():
    """A small model per observed operation, and its input width."""
    convolution = _Model(
        lambda s, x: torch.convolution(
            x[:1].reshape(1, 4, 4), s.w, None, [1], [0], [1], False, [0], 1,
        ).flatten()[None],
        w=(8, 4, 3),
    )
    return {
        "aten.linear.default": (_Model(lambda s, x: F.linear(x, s.w), w=(8, 16)), 16),
        "aten.mm.default": (_Model(lambda s, x: torch.mm(x, s.w), w=(16, 8)), 16),
        "aten.matmul.default": (
            _Model(lambda s, x: torch.matmul(x.view(2, 1, 16), s.w), w=(16, 8)), 16,
        ),
        "aten.bmm.default": (
            _Model(lambda s, x: torch.bmm(x.view(2, 1, 16), s.w), w=(2, 16, 8)), 16,
        ),
        "aten.addmm.default": (
            _Model(lambda s, x: torch.addmm(s.b, x, s.w), w=(16, 8), b=(8,)), 16,
        ),
        "aten.baddbmm.default": (
            _Model(lambda s, x: torch.baddbmm(s.b, x.view(2, 1, 16), s.w), w=(2, 16, 8), b=(8,)),
            16,
        ),
        "aten.mv.default": (_Model(lambda s, x: torch.mv(s.w, x[0])[None], w=(8, 16)), 16),
        "aten.addmv.default": (
            _Model(lambda s, x: torch.addmv(s.b, s.w, x[0])[None], w=(8, 16), b=(8,)), 16,
        ),
        "aten.einsum.default": (
            _Model(lambda s, x: torch.einsum("bi,oi->bo", x, s.w), w=(8, 16)), 16,
        ),
        "aten.tensordot.default": (
            _Model(lambda s, x: torch.tensordot(x, s.w, dims=([1], [0])), w=(16, 8)), 16,
        ),
        "aten.bilinear.default": (
            _Model(lambda s, x: s.b(x, x), b=nn.Bilinear(16, 16, 4)), 16,
        ),
        "aten._native_multi_head_attention.default": (
            _Model(
                lambda s, x: s.a(*(x.view(1, 2, 16),) * 3, need_weights=False)[0],
                a=nn.MultiheadAttention(16, 2, batch_first=True),
            ),
            16,
        ),
        "aten.lstm.input": (_Model(lambda s, x: s.r(x)[0], r=nn.LSTM(16, 8)), 16),
        "aten.gru.input": (_Model(lambda s, x: s.r(x)[0], r=nn.GRU(16, 8)), 16),
        "aten.rnn_tanh.input": (_Model(lambda s, x: s.r(x)[0], r=nn.RNN(16, 8)), 16),
        "aten.rnn_relu.input": (
            _Model(lambda s, x: s.r(x)[0], r=nn.RNN(16, 8, nonlinearity="relu")), 16,
        ),
        "aten.conv1d.default": _conv(1),
        "aten.conv2d.default": _conv(2),
        "aten.conv3d.default": _conv(3),
        "aten.conv1d.padding": _conv(1, padding="same"),
        "aten.conv2d.padding": _conv(2, padding="same"),
        "aten.conv3d.padding": _conv(3, padding="same"),
        "aten.conv_transpose1d.default": _conv(1, transposed=True),
        "aten.conv_transpose2d.input": _conv(2, transposed=True),
        "aten.conv_transpose3d.input": _conv(3, transposed=True),
        "aten.convolution.default": (convolution, 16),
        "cattorch.stacked_swiglu_moe.default": (
            StackedSwiGLUMoE(
                torch.randn(2, 16), torch.randn(2, 8, 16), torch.randn(2, 8, 16),
                torch.randn(2, 16, 8), top_k=1,
            ),
            16,
        ),
        "cattorch.expert_family_moe.default": (
            SparseMoE(nn.Linear(16, 2, bias=False), _family()), 16,
        ),
        "cattorch.routed_experts.default": (_Routed(), 16),
    }


def _observed_targets():
    registry = default_registry()
    return set(_DECOMPOSED_OBSERVERS) | {
        target for target, rule in registry._operations.items() if rule.observer is not None
    }


def test_every_observer_has_a_model():
    # An observer added without a model here fails, so the parametrized test
    # below covers it.
    assert set(_models()) == _observed_targets()


@pytest.mark.parametrize("target", sorted(_observed_targets()))
@torch.no_grad()
def test_observed_weights_are_stored_as_integers_and_reported_so(tmp_path, target):
    torch.manual_seed(5)
    model, width = _models()[target]
    model = model.eval()
    config = QuantizationConfig(
        bits=4, method="gptq", group_size=8, min_quantized_values=1, min_calibration_rows=1,
    )
    result = transpile(
        model, torch.randn(2, width), tmp_path / "model", quantization=config,
        calibration=[torch.randn(2, width) for _ in range(4)],
    )
    report = result.quantization
    assert any(item.method == "gptq" for item in report.tensors), report
    assert not [item for item in report.tensors if item.method == "float16"], report
    assert not [
        warning for warning in result.warnings
        if warning.startswith(("quantized weight", "weight computed from"))
    ]
