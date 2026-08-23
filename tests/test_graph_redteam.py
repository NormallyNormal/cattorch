"""Adversarial graph/lowering regressions found during the 0.4 audit."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from cattorch import FastConfig, UnsupportedOperationError, transpile, verify
from cattorch.graph import _prepare_graph


class FunctionModel(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, *inputs):
        return self.function(*inputs)


def _assert_exact(tmp_path, name, function, inputs, *, optimization="exact"):
    model = FunctionModel(function)
    result = transpile(
        model, inputs, tmp_path / name, optimization=optimization,
    )
    comparison = verify(model, inputs, result, atol=2e-4, rtol=1e-5)
    assert comparison.passed, comparison


@pytest.mark.parametrize("optimization", ["exact", "fast"])
def test_slices_normalize_negative_and_empty_bounds(tmp_path, optimization):
    value = torch.randn(2, 5, 3)
    _assert_exact(
        tmp_path, f"negative_{optimization}", lambda x: x[:, -3:, :],
        (value,), optimization=optimization,
    )
    _assert_exact(
        tmp_path, f"empty_{optimization}", lambda x: x[:, 4:1, :],
        (value,), optimization=optimization,
    )


def test_stepped_slice_is_rejected(tmp_path):
    with pytest.raises(UnsupportedOperationError, match="step of 1"):
        transpile(
            FunctionModel(lambda x: x[:, ::2, :]),
            torch.randn(2, 5, 3),
            tmp_path / "stepped",
        )


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_split_uses_flat_trailing_geometry(tmp_path, dim):
    value = torch.randn(6, 6, 6)
    _assert_exact(
        tmp_path, f"split_{dim}",
        lambda x: torch.split(x, [1, 3, 2], dim=dim)[1],
        (value,),
    )


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("operation", ["split", "chunk"])
def test_uneven_split_and_chunk_select_last_output(tmp_path, dim, operation):
    value = torch.randn(5, 5, 5)

    def select_last(tensor):
        if operation == "split":
            return torch.split(tensor, 2, dim=dim)[-1]
        return torch.chunk(tensor, 3, dim=dim)[-1]

    _assert_exact(
        tmp_path, f"uneven_{operation}_{dim}", select_last, (value,),
    )


@pytest.mark.parametrize("count", [1, 5, 6])
def test_cat_keeps_multiway_accumulator_alive(tmp_path, count):
    values = tuple(torch.randn(2, index % 3 + 1, 2) for index in range(count))
    _assert_exact(
        tmp_path, f"cat_{count}", lambda *items: torch.cat(items, dim=1), values,
    )


def test_nonadjacent_singleton_transpose_is_not_aliased(tmp_path):
    value = torch.arange(24.0).reshape(1, 2, 3, 4)
    model = FunctionModel(lambda x: x.transpose(0, 2))
    graph = _prepare_graph(model, (value,))
    transpose = next(
        node for node in graph.nodes if str(node.target) == "aten.transpose.int"
    )
    assert transpose.name not in graph.aliases
    _assert_exact(tmp_path, "singleton_transpose", model.function, (value,))


def test_broadcast_swiglu_and_embedding_add_do_not_fuse(tmp_path):
    gate = torch.randn(2, 1, 4)
    value = torch.randn(2, 3, 4)
    swiglu = FunctionModel(lambda x, y: F.silu(x) * y)
    graph = _prepare_graph(swiglu, (gate, value))
    assert not graph.silu_mul_fusions
    _assert_exact(tmp_path, "broadcast_swiglu", swiglu.function, (gate, value))

    class EmbeddingAdd(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Embedding(8, 4)
            self.second = nn.Embedding(8, 4)

        def forward(self, first, second):
            return self.first(first) + self.second(second)

    embedding_add = EmbeddingAdd().eval()
    indices = (torch.tensor([[1], [2]]), torch.tensor([[3, 4]]))
    graph = _prepare_graph(embedding_add, indices)
    assert not graph.embedding_add_fusions
    result = transpile(embedding_add, indices, tmp_path / "broadcast_embeddings")
    comparison = verify(embedding_add, indices, result, atol=2e-4)
    assert comparison.passed, comparison


def test_linear_division_by_zero_falls_back_from_epilogue(tmp_path):
    class LinearDivision(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 4)

        def forward(self, value):
            return self.linear(value) / 0.0

    model = LinearDivision().eval()
    value = torch.randn(2, 3)
    graph = _prepare_graph(model, (value,))
    assert not graph.linear_fusions
    result = transpile(model, value, tmp_path / "linear_division")
    assert verify(model, value, result).passed


@pytest.mark.parametrize(
    ("model", "inputs", "message"),
    [
        (
            FunctionModel(lambda left, right: left @ right),
            (torch.randn(1, 3, 4), torch.randn(2, 4, 5)),
            "identical batch dimensions",
        ),
        (
            FunctionModel(lambda left, right: left @ right),
            (torch.randn(4), torch.randn(2, 4, 5)),
            "vector-by-batched",
        ),
        (
            FunctionModel(lambda left, right: torch.add(left, right, alpha=2)),
            (torch.randn(2, 3), torch.randn(2, 3)),
            "alpha=2",
        ),
        (
            FunctionModel(lambda value: value.mean((0, 2))),
            (torch.randn(2, 3, 4),),
            "consecutive",
        ),
        (
            FunctionModel(lambda value: torch.pow(value, 3)),
            (torch.randn(2, 3),),
            "exponents 0 and 2",
        ),
    ],
)
def test_unsupported_semantic_variants_are_rejected(
    tmp_path, model, inputs, message,
):
    with pytest.raises(UnsupportedOperationError, match=message):
        transpile(model, inputs, tmp_path / "unsupported")


@pytest.mark.parametrize(
    "model",
    [
        nn.Conv2d(2, 3, 3, dilation=2),
        nn.Conv2d(4, 4, 3, groups=2),
        nn.MaxPool2d(2, dilation=2),
        nn.MaxPool2d(2, ceil_mode=True),
        nn.AvgPool2d(3, padding=1, count_include_pad=False),
        nn.AvgPool2d(2, divisor_override=7),
    ],
)
def test_unimplemented_convolution_and_pooling_options_are_rejected(
    tmp_path, model,
):
    channels = 4 if isinstance(model, nn.Conv2d) and model.groups == 2 else 2
    with pytest.raises(UnsupportedOperationError):
        transpile(model, torch.randn(1, channels, 8, 8), tmp_path / "options")


def test_batchnorm_without_running_statistics_is_rejected(tmp_path):
    model = nn.BatchNorm2d(3, track_running_stats=False).eval()
    with pytest.raises(UnsupportedOperationError, match="running statistics"):
        transpile(model, torch.randn(2, 3, 4, 4), tmp_path / "batchnorm")


def test_default_gelu_requires_fast_activations(tmp_path):
    model = nn.GELU().eval()
    value = torch.randn(2, 3)
    with pytest.raises(UnsupportedOperationError, match="approximate='tanh'"):
        transpile(model, value, tmp_path / "exact_gelu")
    with pytest.raises(UnsupportedOperationError, match="approximate='tanh'"):
        transpile(
            model, value, tmp_path / "disabled_fast_gelu",
            optimization="fast", fast_config=FastConfig(activations=False),
        )
    transpile(model, value, tmp_path / "fast_gelu", optimization="fast")
