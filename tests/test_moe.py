"""Generic eager MoE semantics used by FX semantic lowering."""

from __future__ import annotations

import json
import zipfile

import torch
import torch.nn as nn

from cattorch import (
    CodegenConfig, FastConfig, FastLayerConfig, QuantizationConfig, transpile,
    verify,
)
from cattorch.quantization import quantize_model
from cattorch.fast import prepare_fast_model
from cattorch.operator_registry import default_registry
from cattorch.experimental import (
    ExpertFamily, RoutingScores, SparseMoE, StackedSwiGLUMoE, select_routes,
)
from cattorch.util.scratch.emulator import ScratchEmulator


class Scale(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, value):
        return value * self.weight


def test_route_selection_has_stable_ties_and_independent_combination_scores():
    selection = torch.tensor([[2.0, 2.0, 1.0]])
    combination = torch.tensor([[0.2, 0.3, 0.5]])
    ids, weights = select_routes(RoutingScores(selection, combination), 2)
    assert ids.tolist() == [[0, 1]]
    assert torch.allclose(weights, torch.tensor([[0.2, 0.3]]))


def test_selected_normalization_zero_is_defined():
    ids, weights = select_routes(RoutingScores(
        torch.tensor([[3.0, 2.0]]), torch.zeros(1, 2), normalize_selected=True,
    ), 2)
    assert ids.tolist() == [[0, 1]]
    assert weights.tolist() == [[0.0, 0.0]]


def test_sparse_moe_top_two_accumulates_only_selected_experts():
    experts = ExpertFamily.from_modules(
        [Scale(1), Scale(2), Scale(10)], example_input=torch.ones(1, 2),
    )

    class Router(nn.Module):
        def forward(self, value):
            shape = (*value.shape[:-1], 3)
            selection = value.new_tensor([3.0, 2.0, 1.0]).expand(shape)
            combination = value.new_tensor([0.25, 0.75, 100.0]).expand(shape)
            return selection, combination

    moe = SparseMoE(Router(), experts, top_k=2)
    value = torch.tensor([[2.0, 4.0], [1.0, 3.0]])
    # expert 0 * .25 + expert 1 * .75 = 1.75x; expert 2 is never selected.
    assert torch.allclose(moe(value), value * 1.75)


def test_from_stacked_builds_uniform_expert_family():
    template = nn.Linear(2, 1, bias=False)
    bank = {"weight": torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])}
    family = ExpertFamily.from_stacked(
        template, bank, example_input=torch.ones(1, 2),
    )
    value = torch.tensor([[1.0, 1.0]])
    assert family(0, value).item() == 3.0
    assert family(1, value).item() == 7.0


def test_expert_family_is_bank_native_and_preserves_buffers_and_tied_aliases():
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(2, 2, bias=False)
            self.right = nn.Linear(2, 2, bias=False)
            self.right.weight = self.left.weight
            self.register_buffer("offset", torch.ones(2))

        def forward(self, value):
            return self.left(value) + self.right(value) + self.offset

    template = Tied()
    banks = {
        "left.weight": torch.stack((template.left.weight, template.left.weight + 1)),
        "right.weight": torch.stack((template.left.weight, template.left.weight + 1)),
        "offset": torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
    }
    family = ExpertFamily(template, banks, example_input=torch.ones(1, 2))
    assert family.expert_count == 2
    assert len(family.unique_banks()) == 2
    assert family.stacked_state["left.weight"] is family.stacked_state["right.weight"]
    assert not hasattr(family, "experts")


@torch.no_grad()
def test_generic_biased_gelu_residual_moe_lowers_to_scratch(tmp_path):
    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Linear(3, 5)
            self.down = nn.Linear(5, 3)

        def forward(self, value):
            return value + self.down(torch.nn.functional.gelu(
                self.up(value), approximate="tanh",
            ))

    class Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(3, 3)

        def forward(self, value):
            selection = self.projection(value)
            return RoutingScores(selection, torch.sigmoid(selection), True)

    torch.manual_seed(13)
    model = SparseMoE(
        Router(),
        ExpertFamily.from_modules(
            [Expert(), Expert(), Expert()], example_input=torch.ones(1, 3),
        ),
        top_k=2,
    ).eval()
    value = torch.randn(2, 2, 3)
    result = transpile(model, value, tmp_path / "generic-moe")
    comparison = verify(model, value, result, atol=3e-4)
    assert comparison.passed, comparison


def test_exact_swiglu_family_auto_specializes():
    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(3, 4, bias=False)
            self.up = nn.Linear(3, 4, bias=False)
            self.down = nn.Linear(4, 3, bias=False)

        def forward(self, value):
            return self.down(torch.nn.functional.silu(self.gate(value)) * self.up(value))

    model = SparseMoE(
        nn.Sequential(nn.Linear(3, 2, bias=False), nn.Softmax(dim=-1)),
        ExpertFamily.from_modules(
            [Expert(), Expert()], example_input=torch.ones(1, 3),
        ),
    ).eval()
    assert isinstance(default_registry().clone().adapt_model(model), StackedSwiGLUMoE)


@torch.no_grad()
def test_swiglu_specialization_rejects_extra_expert_operations():
    class PreNormExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(3)
            self.gate = nn.Linear(3, 4, bias=False)
            self.up = nn.Linear(3, 4, bias=False)
            self.down = nn.Linear(4, 3, bias=False)

        def forward(self, value):
            hidden = self.norm(value)
            return self.down(torch.nn.functional.silu(self.gate(hidden)) * self.up(hidden))

    class ScaledInputExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(3, 4, bias=False)
            self.up = nn.Linear(3, 4, bias=False)
            self.down = nn.Linear(4, 3, bias=False)

        def forward(self, value):
            hidden = value / 2
            return self.down(torch.nn.functional.silu(self.gate(hidden)) * self.up(hidden))

    torch.manual_seed(3)
    for expert_type in (PreNormExpert, ScaledInputExpert):
        model = SparseMoE(
            nn.Sequential(nn.Linear(3, 2, bias=False), nn.Softmax(dim=-1)),
            ExpertFamily.from_modules(
                [expert_type(), expert_type()], example_input=torch.ones(1, 3),
            ),
        ).eval()
        adapted = default_registry().clone().adapt_model(model)
        assert not isinstance(adapted, StackedSwiGLUMoE), expert_type.__name__


def test_generic_gptq_observes_selected_linear_bank_views():
    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.up_weight = nn.Parameter(torch.randn(8, 3))
            self.up_bias = nn.Parameter(torch.randn(8))
            self.down_weight = nn.Parameter(torch.randn(3, 8))
            self.down_bias = nn.Parameter(torch.randn(3))

        def forward(self, value):
            hidden = torch.relu(torch.nn.functional.linear(
                value, self.up_weight, self.up_bias,
            ))
            return torch.nn.functional.linear(
                hidden, self.down_weight, self.down_bias,
            )

    model = SparseMoE(
        nn.Linear(3, 2, bias=False),
        ExpertFamily.from_modules(
            [Expert(), Expert()], example_input=torch.ones(1, 3),
        ),
    ).eval()
    registry = default_registry().clone()
    owned = registry.adapt_model(model)
    report = quantize_model(
        owned,
        QuantizationConfig(
            bits=8, method="gptq", min_quantized_values=1,
            min_calibration_rows=1,
        ),
        [torch.randn(2, 4, 3)],
        _registry=registry,
    )
    expert_matrices = [
        item for item in report.tensors
        if "expert_family._parameter_banks" in item.name and "offset=" in item.name
    ]
    assert expert_matrices
    assert any(item.method == "gptq" and item.calibration_rows > 0 for item in expert_matrices)
    # Every one of the eight rank-3 token rows reaches both expert matrices.
    assert sum(item.calibration_rows for item in expert_matrices) == 16


@torch.no_grad()
def test_generic_expert_block_graph_does_not_grow_with_expert_count(tmp_path):
    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Linear(2, 3)
            self.down = nn.Linear(3, 2)

        def forward(self, value):
            return self.down(torch.relu(self.up(value)))

    counts = []
    for experts in (2, 8):
        model = SparseMoE(
            nn.Linear(2, experts, bias=False),
            ExpertFamily.from_modules(
                [Expert() for _ in range(experts)],
                example_input=torch.ones(1, 2),
            ),
        ).eval()
        result = transpile(model, torch.ones(1, 2), tmp_path / f"experts-{experts}")
        with zipfile.ZipFile(result.path) as archive:
            counts.append(len(json.loads(archive.read("sprite.json"))["blocks"]))
    assert counts[1] == counts[0]


def test_generic_expert_rejects_path_scoped_topology_transform(tmp_path):
    model = SparseMoE(
        nn.Linear(2, 2),
        ExpertFamily.from_modules(
            [nn.Linear(2, 2), nn.Linear(2, 2)],
            example_input=torch.ones(1, 2),
        ),
    ).eval()
    try:
        transpile(
            model, torch.ones(1, 2), tmp_path / "invalid-fast-moe",
            optimization="fast", fast_config=FastConfig(overrides={
                "expert_family": FastLayerConfig(rank=1),
            }),
        )
    except ValueError as error:
        assert "generic ExpertFamily" in str(error)
    else:
        raise AssertionError("topology-changing generic-expert pruning was accepted")


def test_expert_family_rejects_state_role_swaps_and_mutation():
    class Swapped(nn.Module):
        def __init__(self, reverse=False):
            super().__init__()
            self.left = nn.Parameter(torch.tensor(2.0))
            self.right = nn.Parameter(torch.tensor(10.0))
            self.reverse = reverse

        def forward(self, value):
            if self.reverse:
                return value * self.right + self.left
            return value * self.left + self.right

    try:
        ExpertFamily.from_modules(
            [Swapped(), Swapped(True)], example_input=torch.ones(1, 1),
        )
    except ValueError as error:
        assert "dataflow" in str(error)
    else:
        raise AssertionError("state-role swap was accepted")

    class Mutating(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.register_buffer("counter", torch.tensor(0.0))

        def forward(self, value):
            self.counter.add_(1)
            return value * self.weight

    try:
        ExpertFamily.from_modules(
            [Mutating(), Mutating()], example_input=torch.ones(1, 1),
        )
    except ValueError as error:
        assert "must not mutate" in str(error)
    else:
        raise AssertionError("stateful expert was accepted")


def test_expert_family_propagates_mode_dtype_and_buffer_persistence():
    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)
            self.dropout = nn.Dropout(0.5)
            self.register_buffer("scratch", torch.ones(1), persistent=False)

        def forward(self, value):
            return self.dropout(self.linear(value)) + self.scratch

    family = ExpertFamily.from_modules(
        [Expert(), Expert()], example_input=torch.ones(1, 2),
    )
    family.eval().double()
    assert not family.template.training
    assert next(family.template.parameters()).dtype == torch.float64
    assert all("buffer_bank" not in name for name in family.state_dict())


@torch.no_grad()
def test_generic_multirow_linear_and_grouped_norm_lowering(tmp_path):
    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 3)
            self.norm = nn.LayerNorm(3, elementwise_affine=False)

        def forward(self, value):
            rows = self.linear(value.reshape(2, 2))
            return self.norm(rows).reshape(1, 6)

    model = SparseMoE(
        nn.Linear(4, 2),
        ExpertFamily.from_modules(
            [Expert(), Expert()], example_input=torch.ones(1, 4),
        ),
    ).eval()
    value = torch.randn(1, 4)
    result = transpile(model, value, tmp_path / "multirow-expert")
    comparison = verify(model, value, result, atol=3e-4)
    assert comparison.passed, comparison


@torch.no_grad()
def test_generic_exact_tanh_activations_stay_finite_at_extremes(tmp_path):
    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=False)
            self.linear.weight.copy_(torch.eye(2))

        def forward(self, value):
            projected = self.linear(value)
            return torch.tanh(projected) + torch.nn.functional.gelu(
                projected, approximate="tanh",
            )

    model = SparseMoE(
        nn.Linear(2, 2, bias=False),
        ExpertFamily.from_modules(
            [Expert(), Expert()], example_input=torch.ones(1, 2),
        ),
    ).eval()
    value = torch.tensor([[-1000.0, 1000.0]])
    result = transpile(model, value, tmp_path / "extreme-tanh-expert")
    comparison = verify(model, value, result, atol=1e-4)
    assert comparison.passed, comparison


@torch.no_grad()
def test_stacked_swiglu_semantic_module_lowers_to_scratch(tmp_path):
    torch.manual_seed(42)
    model = StackedSwiGLUMoE(
        torch.randn(3, 2),
        torch.randn(3, 3, 2),
        torch.randn(3, 3, 2),
        torch.randn(3, 2, 3),
        top_k=2,
        normalize_selected=True,
    ).eval()
    value = torch.randn(2, 2)
    result = transpile(model, value, tmp_path / "moe")
    comparison = verify(model, value, result, atol=2e-4)
    assert comparison.passed, comparison


@torch.no_grad()
def test_stacked_swiglu_grouped_top_one_lowering_matches_eager(tmp_path):
    torch.manual_seed(43)
    model = StackedSwiGLUMoE(
        torch.randn(3, 4),
        torch.randn(3, 8, 4),
        torch.randn(3, 8, 4),
        torch.randn(3, 4, 8),
        top_k=1,
    ).eval()
    value = torch.randn(2, 4)
    result = transpile(model, value, tmp_path / "moe-grouped-top1")
    comparison = verify(model, value, result, atol=2e-4)
    assert comparison.passed, comparison


def test_gptq_observes_selected_expert_banks_and_down_activations():
    torch.manual_seed(7)
    model = StackedSwiGLUMoE(
        torch.randn(2, 3),
        torch.randn(2, 4, 3),
        torch.randn(2, 4, 3),
        torch.randn(2, 3, 4),
    ).eval()
    report = quantize_model(
        model,
        QuantizationConfig(
            bits=8, method="gptq", min_quantized_values=1,
            min_calibration_rows=1,
        ),
        [torch.randn(16, 3)],
    )
    names = {item.name for item in report.tensors if item.method == "gptq"}
    assert any(name.startswith("router_weight") for name in names)
    assert any(name.startswith("gate_weight") for name in names)
    assert any(name.startswith("up_weight") for name in names)
    assert any(name.startswith("down_weight") for name in names)


def test_fast_neuron_pruning_reduces_stacked_swiglu_work():
    torch.manual_seed(9)
    model = StackedSwiGLUMoE(
        torch.randn(2, 4),
        torch.randn(2, 8, 4),
        torch.randn(2, 8, 4),
        torch.randn(2, 4, 8),
    ).eval()
    transformed = prepare_fast_model(model, FastConfig(neuron_pruning=0.5))

    assert transformed is not model
    assert transformed.gate_weight.shape == (2, 4, 4)
    assert transformed.up_weight.shape == (2, 4, 4)
    assert transformed.down_weight.shape == (2, 4, 4)
    assert model.gate_weight.shape == (2, 8, 4)


@torch.no_grad()
def test_shared_quantized_moe_banks_keep_groups_aligned_across_shards(tmp_path):
    width, experts, hidden = 48, 8, 300

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.moe = StackedSwiGLUMoE(
                torch.randn(experts, width) * 0.02,
                torch.randn(experts, hidden, width) * 0.02,
                torch.randn(experts, hidden, width) * 0.02,
                torch.randn(experts, width, hidden) * 0.02,
            )

        def forward(self, value):
            return value + self.moe(value)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList((Block(), Block()))

        def forward(self, value):
            value = value + 0.0
            for block in self.blocks:
                value = block(value)
            return value

    torch.manual_seed(23)
    model = Model().eval()
    value = torch.randn(1, width)
    outputs = {}
    for mode in ("off", "auto"):
        result = transpile(
            model,
            value,
            tmp_path / f"moe-bank-{mode}",
            quantization=QuantizationConfig(
                bits=4, method="symmetric", min_quantized_values=1,
            ),
            codegen=CodegenConfig(
                id_namespace="", unrolling="compact", layer_sharing=mode,
            ),
        )
        with zipfile.ZipFile(result.path) as archive:
            emulator = ScratchEmulator(json.loads(archive.read("sprite.json")))
        emulator.lists["input"] = value.flatten().tolist()
        emulator.run_procedure("cattorch forward")
        outputs[mode] = torch.tensor(emulator.lists["output"])
        if mode == "auto":
            assert "cattorch shared transformer layer" in emulator._procedures

    torch.testing.assert_close(outputs["auto"], outputs["off"], atol=1e-6, rtol=0)
