"""Experts routed by ordinary router code through ``routed_experts``."""

from __future__ import annotations

import json
import zipfile

import pytest
import torch
import torch.nn as nn

from cattorch import FastConfig, FastLayerConfig, QuantizationConfig, transpile, verify
from cattorch.experimental import ExpertFamily, SparseMoE, routed_experts
from cattorch.operator_registry import default_registry
from cattorch.quantization import quantize_model


class Expert(nn.Module):
    def __init__(self, width=4, hidden=6):
        super().__init__()
        self.up = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, value):
        return self.down(torch.relu(self.up(value)))


def family(experts=4, width=4, hidden=6):
    return ExpertFamily.from_modules(
        [Expert(width, hidden) for _ in range(experts)],
        example_input=torch.ones(1, width),
    )


class TopKBlock(nn.Module):
    """A Mixtral-style block: softmax router, top-k, renormalized weights."""

    def __init__(self, experts=4, width=4, top_k=2, normalize=True):
        super().__init__()
        self.gate = nn.Linear(width, experts, bias=False)
        self.experts = family(experts, width)
        self.top_k = top_k
        self.normalize = normalize

    def forward(self, value):
        probabilities = torch.softmax(self.gate(value), dim=-1)
        weights, ids = torch.topk(probabilities, self.top_k, dim=-1)
        if self.normalize:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return routed_experts(value, ids, weights, self.experts)


def reference(value, ids, weights, experts):
    flat = value.reshape(-1, value.shape[-1])
    flat_ids = ids.reshape(-1, ids.shape[-1])
    flat_weights = weights.reshape(-1, ids.shape[-1])
    rows = []
    for token in range(flat.shape[0]):
        row = 0
        for route in range(flat_ids.shape[-1]):
            row = row + flat_weights[token, route] * experts(
                int(flat_ids[token, route]), flat[token:token + 1],
            )[0]
        rows.append(row)
    return torch.stack(rows).reshape(*value.shape[:-1], -1)


@torch.no_grad()
def test_eager_sums_weighted_outputs_including_repeated_routes():
    torch.manual_seed(0)
    experts = family()
    value = torch.randn(2, 3, 4)
    ids = torch.tensor([[[0, 2], [3, 3], [1, 0]], [[2, 1], [0, 0], [3, 2]]])
    weights = torch.rand(2, 3, 2)
    assert torch.allclose(
        routed_experts(value, ids, weights, experts),
        reference(value, ids, weights, experts), atol=1e-6,
    )


@torch.no_grad()
def test_eager_matches_sparse_moe_for_the_same_routes():
    torch.manual_seed(1)
    block = TopKBlock(normalize=False).eval()
    sparse = SparseMoE(
        nn.Sequential(block.gate, nn.Softmax(dim=-1)), block.experts, top_k=2,
    ).eval()
    value = torch.randn(5, 4)
    assert torch.allclose(block(value), sparse(value), atol=1e-6)


@pytest.mark.parametrize(
    ("ids", "weights", "error"),
    [
        (torch.zeros(3, 2, dtype=torch.long), torch.ones(3, 3), ValueError),
        (torch.zeros(2, 2, dtype=torch.long), torch.ones(2, 2), ValueError),
        (torch.zeros(3, 2), torch.ones(3, 2), TypeError),
        (torch.full((3, 2), 4), torch.ones(3, 2), IndexError),
        (torch.full((3, 2), -1), torch.ones(3, 2), IndexError),
        (torch.zeros(3, 0, dtype=torch.long), torch.ones(3, 0), ValueError),
    ],
)
def test_eager_rejects_malformed_routes(ids, weights, error):
    with pytest.raises(error):
        routed_experts(torch.ones(3, 4), ids, weights, family())


def test_rejects_non_family_experts():
    with pytest.raises(TypeError, match="ExpertFamily"):
        routed_experts(
            torch.ones(1, 4), torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1),
            nn.Linear(4, 4),
        )


@torch.no_grad()
@pytest.mark.parametrize("shape", [(1, 4), (3, 4), (1, 3, 4)])
@pytest.mark.parametrize("top_k", [1, 2])
def test_topk_router_block_lowers_to_scratch(tmp_path, shape, top_k):
    torch.manual_seed(2)
    model = TopKBlock(top_k=top_k).eval()
    value = torch.randn(*shape)
    result = transpile(model, value, tmp_path / "routed")
    comparison = verify(model, value, result, atol=1e-4)
    assert comparison.passed, comparison


@torch.no_grad()
def test_sigmoid_router_with_shared_expert_lowers_to_scratch(tmp_path):
    """DeepSeek-style: sigmoid scores, unnormalized weights, an always-on expert."""

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(4, 6)
            self.experts = family(6)
            self.shared = Expert()

        def forward(self, value):
            scores = torch.sigmoid(self.gate(value))
            weights, ids = torch.topk(scores, 3, dim=-1)
            return self.shared(value) + routed_experts(value, ids, weights, self.experts)

    torch.manual_seed(3)
    model = Block().eval()
    value = torch.randn(2, 4)
    result = transpile(model, value, tmp_path / "shared")
    comparison = verify(model, value, result, atol=1e-4)
    assert comparison.passed, comparison


class RoutesAsInputs(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.experts = family(experts)

    def forward(self, value, ids, weights):
        return routed_experts(value, ids, weights, self.experts)


@torch.no_grad()
def test_routes_from_inputs_lower_to_scratch(tmp_path):
    torch.manual_seed(4)
    model = RoutesAsInputs(4).eval()
    inputs = (torch.randn(3, 4), torch.tensor([[3, 0], [1, 1], [2, 3]]), torch.rand(3, 2))
    result = transpile(model, inputs, tmp_path / "inputs")
    comparison = verify(model, inputs, result, atol=1e-4)
    assert comparison.passed, comparison


@torch.no_grad()
def test_block_count_does_not_grow_with_expert_count(tmp_path):
    # Routes come in as inputs, so only the expert kernel sees the expert count.
    counts = []
    for experts in (2, 8):
        torch.manual_seed(4)
        model = RoutesAsInputs(experts).eval()
        inputs = (torch.ones(1, 4), torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1))
        result = transpile(model, inputs, tmp_path / f"experts-{experts}")
        with zipfile.ZipFile(result.path) as archive:
            counts.append(len(json.loads(archive.read("sprite.json"))["blocks"]))
    assert counts[1] == counts[0]


@torch.no_grad()
def test_export_does_not_touch_the_default_registry_or_the_model(tmp_path):
    model = TopKBlock().eval()
    before = dict(default_registry().expert_families)
    transpile(model, torch.ones(1, 4), tmp_path / "isolated")
    assert default_registry().expert_families == before
    assert model.experts._cattorch_handle is None


def test_gptq_calibrates_routed_expert_banks():
    torch.manual_seed(5)
    model = TopKBlock().eval()
    registry = default_registry().clone()
    owned = registry.adapt_model(model)
    report = quantize_model(
        owned,
        QuantizationConfig(
            bits=8, method="gptq", min_quantized_values=1, min_calibration_rows=1,
        ),
        [torch.randn(2, 4, 4)],
        _registry=registry,
    )
    expert_matrices = [
        item for item in report.tensors
        if "experts._parameter_banks" in item.name and "offset=" in item.name
    ]
    assert any(item.method == "gptq" and item.calibration_rows > 0 for item in expert_matrices)
    # Eight tokens, two routes each, reach both of their expert's matrices.
    assert sum(item.calibration_rows for item in expert_matrices) == 32


def test_fast_topology_override_inside_routed_family_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="generic ExpertFamily"):
        transpile(
            TopKBlock().eval(), torch.ones(1, 4), tmp_path / "invalid",
            optimization="fast", fast_config=FastConfig(overrides={
                "experts": FastLayerConfig(rank=1),
            }),
        )
