"""Regressions found by adversarial bug hunts against the public API."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from cattorch import FastConfig, FastLayerConfig, transpile, verify
from cattorch.experimental import ExpertFamily, SparseMoE


def _check(model, inputs, path, **options):
    model = model.eval()
    inputs = inputs if isinstance(inputs, tuple) else (inputs,)
    result = transpile(model, inputs, path, **options)
    report = verify(model, inputs, result)
    assert report.passed, report
    return result


class CombinedQKV(nn.Module):
    """nanoGPT-style attention: one projection split into Q, K and V."""

    def __init__(self, extra=None, sequence=16, width=16, heads=4):
        super().__init__()
        self.extra, self.heads, self.width = extra, heads, width
        self.c_attn = nn.Linear(width, 3 * width)
        self.register_buffer("mask", torch.triu(torch.ones(sequence, sequence, dtype=torch.bool), 1))

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.c_attn(x).split(width, dim=2)
        q, k, v = (
            value.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
            for value in (q, k, v)
        )
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        scores = scores.masked_fill(self.mask, float("-inf"))
        y = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, length, width)
        if self.extra == "kv":
            return y, k, v
        if self.extra == "scaled_k":
            return y, k * 2
        return y


@pytest.mark.parametrize(("extra", "batch"), [(None, 2), ("kv", 1), ("scaled_k", 1)])
def test_combined_qkv_attention_with_batches_or_other_readers(tmp_path, extra, batch):
    torch.manual_seed(0)
    _check(CombinedQKV(extra), torch.randn(batch, 16, 16), tmp_path / "attention")


class Tied(nn.Module):
    def __init__(self, vocabulary=256, width=192, index=False):
        super().__init__()
        self.index = index
        self.embedding = nn.Embedding(vocabulary, width)
        self.head = nn.Linear(width, vocabulary, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, tokens):
        logits = self.head(self.embedding(tokens))
        return (logits, self.embedding.weight[tokens]) if self.index else logits


def test_tied_table_read_by_a_third_operation(tmp_path):
    torch.manual_seed(0)
    _check(Tied(index=True), torch.tensor([[7, 3]]), tmp_path / "tied")


def test_tied_table_with_fast_weights_that_change_nothing(tmp_path):
    torch.manual_seed(0)
    fast = FastConfig(weights=FastLayerConfig(rank=100_000))
    _check(Tied(), torch.tensor([[7, 3]]), tmp_path / "tied", optimization="fast", fast_config=fast)


def test_fast_pruning_of_a_matmul_leaves_other_readers_alone(tmp_path):
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.randn(32, 32))

        def forward(self, tokens, x):
            return x @ self.weight, F.embedding(tokens, self.weight)

    torch.manual_seed(0)
    model = Shared().eval()
    inputs = (torch.tensor([1, 5]), torch.randn(2, 32))
    fast = FastConfig(weights=FastLayerConfig(pruning=0.5))
    result = transpile(model, inputs, tmp_path / "shared", optimization="fast", fast_config=fast)
    report = verify(model, inputs, result)
    assert report.outputs[1].max_abs_error == 0


def test_residual_add_that_broadcasts_the_linear_output(tmp_path):
    class Broadcast(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 8)

        def forward(self, x, residual):
            return self.linear(x) + residual

    torch.manual_seed(0)
    _check(Broadcast(), (torch.randn(1, 16), torch.randn(4, 8)), tmp_path / "residual")


class ParameterExpert(nn.Module):
    """Operations on the expert's own parameters, and a computed weight."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.gate = nn.Parameter(torch.randn(4))
        self.weight = nn.Parameter(torch.randn(4, 4))

    def forward(self, x):
        gated = self.linear(x) * torch.sigmoid(self.gate) + self.gate[:4].clamp(-0.5, 0.5) ** 2
        return gated + F.linear(x, self.weight.t())


class Fixed(nn.Module):
    """Routes every token to one expert."""

    def __init__(self, expert):
        super().__init__()
        self.expert = expert

    def forward(self, x):
        scores = torch.zeros(3)
        scores[self.expert] = 1.0
        return x.new_tensor(scores.tolist()).expand(*x.shape[:-1], 3)


@pytest.mark.parametrize("expert", [1, 2])
def test_expert_operations_on_parameters_read_the_chosen_expert(tmp_path, expert):
    torch.manual_seed(0)
    family = ExpertFamily.from_modules(
        [ParameterExpert() for _ in range(3)], example_input=torch.ones(1, 4),
    )
    _check(SparseMoE(Fixed(expert), family, top_k=1), torch.randn(5, 4), tmp_path / "moe")


def test_boolean_add_is_a_logical_or(tmp_path):
    class Either(nn.Module):
        def forward(self, x):
            return ((x > 0) + (x > 1)).float(), ((x > 0) + True).float()

    _check(Either(), torch.tensor([[-1.0, 0.5, 2.0]]), tmp_path / "or")


@pytest.mark.parametrize("exponent", [5, -7, 13])
def test_integer_powers_are_exact(tmp_path, exponent):
    class Power(nn.Module):
        def forward(self, x):
            return x ** exponent

    model = Power().eval()
    value = torch.tensor([[3.0, -2.0, 6.0, 1.5]])
    result = transpile(model, value, tmp_path / "power")
    report = verify(model, value, result, atol=0, rtol=1e-6)
    assert report.passed, report


@pytest.mark.parametrize(("weight", "bias"), [((5,), None), ((3, 5), (1,)), ((3, 5), ())])
def test_functional_linear_with_a_vector_weight_or_broadcast_bias(tmp_path, weight, bias):
    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(weight))
            self.bias = None if bias is None else nn.Parameter(torch.randn(bias))

        def forward(self, x):
            return F.linear(x, self.weight, self.bias)

    torch.manual_seed(0)
    _check(Linear(), torch.randn(2, 5), tmp_path / "linear")
