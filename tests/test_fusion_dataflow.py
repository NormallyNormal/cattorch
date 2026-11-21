"""Fusions only absorb values nothing else reads, and every export checks its dataflow."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cattorch import transpile, verify
from cattorch.graph import (
    _check_emission_plan,
    _check_fusions,
    _FusionRecord,
    _prepare_graph,
)
from cattorch.util.scratch.dataflow import check_list_dataflow


class FunctionModel(nn.Module):
    def __init__(self, function, width=8):
        super().__init__()
        self.function = function
        self.lin = nn.Linear(width, width)
        self.other = nn.Linear(width, width)
        self.eval()

    def forward(self, *inputs):
        return self.function(self, *inputs)


def _silu_square(m, x):
    s = F.silu(x)
    return s * s


def _linear_silu_square(m, x):
    s = F.silu(m.lin(x))
    return s * s + 1


def _two_silus(m, x):
    return F.silu(x) * F.silu(2 * x)


def _two_silus_same_input(m, x):
    return F.silu(x) * F.silu(x)


def _linear_silus(m, x):
    return F.silu(m.lin(x)) * F.silu(m.other(x))


def _silu_of_linear_times_linear(m, x):
    z = m.lin(x)
    return F.silu(z) * z


def _linear_square(m, x):
    z = m.lin(x) * 2
    return z * z + (z + z)


def _chain_square(m, x):
    z = x * 2 + 1
    return z * z


def _self_where(m, x):
    return torch.where(x > 0, x, x) + torch.cat([x, x], dim=-1)[..., :8]


@pytest.mark.parametrize("function", [
    _silu_square, _linear_silu_square, _two_silus, _two_silus_same_input, _linear_silus,
    _silu_of_linear_times_linear, _linear_square, _chain_square, _self_where,
])
def test_operations_reading_one_value_twice_are_exact(tmp_path, function):
    model = FunctionModel(function)
    value = torch.randn(2, 3, 8)
    result = transpile(model, value, tmp_path / function.__name__)
    comparison = verify(model, value, result, atol=2e-4, rtol=1e-5)
    assert comparison.passed, comparison


def test_silu_product_absorbs_one_of_two_silus():
    model = FunctionModel(_two_silus)
    graph = _prepare_graph(model, (torch.randn(2, 8),))
    ((silu, _, other),) = graph.silu_mul_fusions.values()
    assert silu.name in graph.fused_nodes
    assert other.name not in graph.fused_nodes


def test_silu_square_isnt_fused():
    graph = _prepare_graph(FunctionModel(_silu_square), (torch.randn(2, 8),))
    assert not graph.silu_mul_fusions


class SelfScores(nn.Module):
    """Causal attention whose scores are ``t @ t`` for one transposed tensor."""

    def forward(self, k, v):
        t = k.transpose(-2, -1)
        scores = (t @ t) * 0.5
        mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
        return torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1) @ v


def test_causal_scores_of_one_transposed_tensor_are_exact(tmp_path):
    model = SelfScores().eval()
    inputs = (torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4))
    graph = _prepare_graph(model, inputs)
    assert not graph.causal_score_fusions
    result = transpile(model, inputs, tmp_path / "self_scores")
    assert verify(model, inputs, result, atol=2e-4, rtol=1e-5).passed


def test_fusion_check_rejects_a_kernel_reading_an_absorbed_value():
    graph = _prepare_graph(FunctionModel(_silu_square), (torch.randn(2, 8),))
    silu = next(node for node in graph.nodes if str(node.target) == "aten.silu.default")
    (multiply,) = silu.users
    record = _FusionRecord("SiLU product", multiply, multiply, (multiply, silu), (silu.args[0], silu))
    with pytest.raises(RuntimeError, match="internal error.*reads 'silu'"):
        _check_fusions((record,), {silu.name}, {})


def test_fusion_check_rejects_two_fusions_absorbing_one_node():
    graph = _prepare_graph(FunctionModel(_linear_silu_square), (torch.randn(2, 8),))
    (linear,) = (node for node in graph.nodes if str(node.target) == "aten.linear.default")
    silu = next(node for node in graph.nodes if str(node.target) == "aten.silu.default")
    first = _FusionRecord("linear", linear, silu, (linear, silu), tuple(linear.all_input_nodes))
    second = _FusionRecord("SiLU product", silu, silu, (silu, linear), ())
    with pytest.raises(RuntimeError, match="internal error"):
        _check_fusions((first, second), {silu.name, linear.name}, {})


def test_fusion_check_rejects_skipped_nodes_no_fusion_computes():
    graph = _prepare_graph(FunctionModel(_chain_square), (torch.randn(2, 8),))
    with pytest.raises(RuntimeError, match="internal error.*skip"):
        _check_fusions(graph.fusion_records, {*graph.fused_nodes, "phantom"}, graph.aliases)


def test_emission_plan_check_rejects_a_value_whose_code_is_gone():
    graph = _prepare_graph(FunctionModel(_linear_square), (torch.randn(2, 8),))
    order = [
        node.name for node in graph.nodes
        if node.op == "call_function" and node.name not in graph.fused_nodes
        and node.name not in graph.aliases
    ]
    _check_emission_plan(graph, order, order)
    with pytest.raises(RuntimeError, match="internal error.*before anything computes it"):
        _check_emission_plan(graph, order, order[1:])


def _sprite(blocks, lists):
    return {"blocks": blocks, "lists": lists, "variables": {}}


def _block(opcode, list_id, name):
    return {"opcode": opcode, "fields": {"LIST": [name, list_id]}, "inputs": {}}


def test_list_check_rejects_a_block_reading_a_missing_list():
    sprite = _sprite({"a": _block("data_itemoflist", "gone", "W_head")}, {})
    with pytest.raises(RuntimeError, match="internal error.*doesn't have.*W_head"):
        check_list_dataflow(sprite)


def test_list_check_rejects_a_tensor_list_nothing_fills():
    blocks = {
        "a": _block("data_itemoflist", "t", "T3"),
        "b": _block("data_deletealloflist", "t", "T3"),
    }
    sprite = _sprite(blocks, {"t": ["T3", []]})
    with pytest.raises(RuntimeError, match="internal error.*nothing fills.*T3"):
        check_list_dataflow(sprite)


@pytest.mark.parametrize("other", [
    _block("data_addtolist", "t", "T3"),
    _block("data_lengthoflist", "t", "T3"),
])
def test_list_check_accepts_a_filled_or_measured_list(other):
    sprite = _sprite({"a": _block("data_itemoflist", "t", "T3"), "b": other}, {"t": ["T3", []]})
    check_list_dataflow(sprite)


def test_list_check_accepts_stored_public_and_non_tensor_lists():
    blocks = {
        "a": _block("data_itemoflist", "w", "W_weight"),
        "b": _block("data_itemoflist", "i", "input"),
        "c": _block("data_itemoflist", "m", "merge_ids"),
    }
    lists = {"w": ["W_weight", [1.0]], "i": ["input", []], "m": ["merge_ids", []]}
    check_list_dataflow(_sprite(blocks, lists))


class EmbeddingSums(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.tokens = nn.Embedding(10, 8)
        self.positions = nn.Embedding(10, 8)

    def forward(self, index):
        if self.kind == "same lookup":
            value = self.tokens(index)
            return value + value
        if self.kind == "same table":
            return self.tokens(index) + self.tokens(index)
        return self.tokens(index) + self.positions(index)


@pytest.mark.parametrize("kind", ["same lookup", "same table", "two tables"])
def test_embedding_sums_reading_one_lookup_twice_are_exact(tmp_path, kind):
    model = EmbeddingSums(kind).eval()
    index = torch.tensor([[1, 2, 3]])
    graph = _prepare_graph(model, (index,))
    for first, second in graph.embedding_add_fusions.values():
        assert first is not second
    result = transpile(model, index, tmp_path / kind.replace(" ", "_"))
    assert verify(model, index, result).passed
