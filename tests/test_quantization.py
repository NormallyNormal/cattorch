import copy
import json
import zipfile

import pytest
import torch
import torch.nn.functional as F

from cattorch import CodegenConfig, QuantizationConfig, StorageConfig, transpile
from cattorch.quantization import effective_group_size, quantize_model
from cattorch.util.scratch.emulator import ScratchEmulator


def _load(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("sprite.json"))


def test_quantization_config_validation_and_row_aligned_groups():
    assert effective_group_size(112, 64) == 56
    assert effective_group_size(192, 64) == 64
    for bits in (4, 6, 8):
        assert QuantizationConfig(bits=bits).bits == bits
    with pytest.raises(ValueError, match="bits"):
        QuantizationConfig(bits=3)
    with pytest.raises(ValueError, match="method"):
        QuantizationConfig(bits=4, method="awq")
    with pytest.raises(ValueError, match="damp_percent"):
        QuantizationConfig(bits=4, damp_percent=0)


def test_gptq_improves_activation_weighted_error_over_symmetric():
    torch.manual_seed(9)
    source = torch.nn.Linear(12, 7, bias=False).eval()
    with torch.no_grad():
        source.weight.mul_(3)
    calibration = torch.randn(128, 12)
    calibration[:, 0] *= 20
    calibration[:, 1] *= 0.05
    symmetric = copy.deepcopy(source)
    gptq = copy.deepcopy(source)

    common = dict(bits=4, group_size=6, min_quantized_values=1)
    quantize_model(symmetric, QuantizationConfig(method="symmetric", **common), None)
    report = quantize_model(
        gptq,
        QuantizationConfig(
            method="gptq", min_calibration_rows=1, **common,
        ),
        [calibration],
    )

    symmetric_error = (symmetric(calibration) - source(calibration)).square().mean()
    gptq_error = (gptq(calibration) - source(calibration)).square().mean()
    assert gptq_error < symmetric_error
    assert report.gptq_values == source.weight.numel()
    assert report.symmetric_values == 0


class _FunctionalLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(6, 5))

    def forward(self, value):
        return F.linear(value, self.weight)


class _StackedExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(3, 4, 5))

    def forward(self, value):
        return sum(F.linear(value, self.weight[index]) for index in range(3))


def test_gptq_captures_functional_linear_and_stacked_weight_views():
    functional = _FunctionalLinear().eval()
    functional_report = quantize_model(
        functional,
        QuantizationConfig(
            bits=4, method="gptq", min_quantized_values=1,
            min_calibration_rows=1,
        ),
        [torch.randn(4, 5)],
    )
    assert [item.name for item in functional_report.tensors] == ["weight"]
    assert functional_report.gptq_values == 30

    experts = _StackedExperts().eval()
    expert_report = quantize_model(
        experts,
        QuantizationConfig(
            bits=4, method="gptq", min_quantized_values=1,
            min_calibration_rows=1,
        ),
        [torch.randn(4, 5)],
    )
    assert expert_report.gptq_values == experts.weight.numel()
    assert len(expert_report.tensors) == 3


def test_unobserved_stacked_experts_fall_back_independently():
    class OneExpert(_StackedExperts):
        def forward(self, value):
            return F.linear(value, self.weight[0])

    model = OneExpert().eval()
    report = quantize_model(
        model,
        QuantizationConfig(
            bits=4, method="gptq", min_quantized_values=1,
            min_calibration_rows=1,
        ),
        [torch.randn(4, 5)],
    )
    assert report.gptq_values == 20
    assert report.symmetric_values == 40
    assert [item.method for item in report.tensors] == [
        "gptq", "symmetric", "symmetric",
    ]


def test_overlapping_transposed_weight_views_use_symmetric_fallback():
    class Bidirectional(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(6, 6))

        def forward(self, value):
            return F.linear(F.linear(value, self.weight), self.weight.T)

    model = Bidirectional().eval()
    report = quantize_model(
        model,
        QuantizationConfig(
            bits=4, method="gptq", min_quantized_values=1,
            min_calibration_rows=1,
        ),
        [torch.randn(4, 6)],
    )
    assert len(report.tensors) == 1
    assert report.tensors[0].method == "symmetric"
    assert "overlapping matrix views" in report.tensors[0].fallback_reason


def test_missing_gptq_statistics_use_reported_symmetric_fallback():
    class EmbeddingOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(32, 8)

        def forward(self, tokens):
            return self.embedding(tokens)

    model = EmbeddingOnly().eval()
    report = quantize_model(
        model,
        QuantizationConfig(
            bits=4, method="gptq", min_quantized_values=1,
            min_calibration_rows=1,
        ),
        [torch.tensor([[1, 2, 3]])],
    )
    assert report.gptq_values == 0
    assert report.symmetric_values == model.embedding.weight.numel()
    assert report.tensors[0].fallback_reason == "no compatible linear activation statistics"


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_transpile_gptq_reconstructs_expected_weights_in_scratch(tmp_path, bits):
    torch.manual_seed(bits)
    model = torch.nn.Linear(8, 4, bias=False).eval()
    value = torch.randn(2, 8)
    calibration = [torch.randn(8, 8)]
    config = QuantizationConfig(
        bits=bits,
        method="gptq",
        group_size=4,
        min_quantized_values=1,
        min_calibration_rows=1,
    )
    expected_model = copy.deepcopy(model)
    quantize_model(expected_model, config, calibration)
    original_weight = model.weight.detach().clone()

    result = transpile(
        model,
        value,
        tmp_path / f"gptq-{bits}",
        quantization=config,
        calibration_inputs=calibration,
        codegen=CodegenConfig(unrolling="compact", id_namespace=""),
    )
    emulator = ScratchEmulator(_load(result.path))
    emulator.lists["input"] = value.flatten().tolist()
    emulator.run()

    torch.testing.assert_close(
        torch.tensor(emulator.lists["output"]),
        expected_model(value).flatten(),
        atol=1e-5,
        rtol=0,
    )
    assert torch.equal(model.weight, original_weight)
    assert result.quantization is not None
    assert result.quantization.bits == bits
    assert emulator.variables["cattorch quantization method"] == "gptq"
    assert emulator.variables["cattorch quantization bits"] == bits


def test_transpile_quantization_validation_and_legacy_alias(tmp_path):
    model = torch.nn.Linear(3, 2).eval()
    value = torch.randn(1, 3)
    gptq = QuantizationConfig(
        bits=4, method="gptq", min_quantized_values=1,
    )
    with pytest.raises(ValueError, match="nonempty calibration_inputs"):
        transpile(model, value, tmp_path / "missing", quantization=gptq)
    with pytest.raises(ValueError, match="cannot be combined"):
        transpile(
            model,
            value,
            tmp_path / "conflict",
            storage=StorageConfig(precision="int4"),
            quantization=gptq,
            calibration_inputs=[value],
        )
    with pytest.warns(DeprecationWarning, match="QuantizationConfig"):
        transpile(
            model,
            value,
            tmp_path / "legacy",
            storage=StorageConfig(precision="int4"),
        )


def test_row_aligned_quantization_preserves_large_grouped_linear(tmp_path):
    torch.manual_seed(21)
    model = torch.nn.Linear(384, 128, bias=False).eval()
    value = torch.randn(1, 384)
    config = QuantizationConfig(
        bits=4, method="symmetric", group_size=64,
        min_quantized_values=1,
    )
    expected_model = copy.deepcopy(model)
    quantize_model(expected_model, config, None)
    result = transpile(
        model,
        value,
        tmp_path / "grouped-row-aligned",
        quantization=config,
        codegen=CodegenConfig(unrolling="compact", id_namespace=""),
    )
    emulator = ScratchEmulator(_load(result.path))
    assert "cattorch interleave grouped weights" in emulator._procedures
    emulator.lists["input"] = value.flatten().tolist()
    emulator.run()
    torch.testing.assert_close(
        torch.tensor(emulator.lists["output"]),
        expected_model(value).flatten(),
        atol=1e-5,
        rtol=1e-5,
    )


@torch.no_grad()
def test_quantization_keeps_float_buffers_out_of_integer_storage(tmp_path):
    from cattorch import verify

    class TableModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            generator = torch.Generator().manual_seed(5)
            self.register_buffer(
                "table", torch.rand((4, 8), generator=generator) * 2 - 1,
            )

        def forward(self, value):
            return value + self.table

    model = TableModel().eval()
    example = torch.zeros(4, 8)
    result = transpile(
        model, example, tmp_path / "table",
        quantization=QuantizationConfig(bits=4, min_quantized_values=1),
    )
    comparison = verify(model, example, result, atol=1e-3)
    assert comparison.passed, comparison


@torch.no_grad()
def test_quantization_exports_nonfinite_float_mask_buffers(tmp_path):
    from cattorch import verify

    class MaskModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "mask", torch.full((8, 8), float("-inf")).triu(1),
            )

        def forward(self, value):
            return torch.softmax(value + self.mask, dim=-1)

    model = MaskModel().eval()
    example = torch.randn(8, 8)
    result = transpile(
        model, example, tmp_path / "mask",
        quantization=QuantizationConfig(bits=4, min_quantized_values=1),
    )
    comparison = verify(model, example, result, atol=1e-4)
    assert comparison.passed, comparison


class _RightHandMatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(64, 48))

    def forward(self, value):
        return value @ self.weight


class _TiedHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(96, 32)
        self.mix = torch.nn.Linear(32, 32, bias=False)

    def forward(self, tokens):
        hidden = torch.relu(self.mix(self.embedding(tokens)))
        return hidden @ self.embedding.weight.T


@pytest.mark.parametrize("method", ["symmetric", "gptq"])
@pytest.mark.parametrize("model_type", [_RightHandMatMul, _TiedHead])
@torch.no_grad()
def test_right_hand_matmul_weights_use_integer_storage_that_matches_eager(
    tmp_path, monkeypatch, method, model_type,
):
    import cattorch.sprite as sprite_module

    torch.manual_seed(4)
    model = model_type().eval()
    if model_type is _TiedHead:
        example = torch.randint(0, 96, (4,))
        calibration = [torch.randint(0, 96, (16,)) for _ in range(20)]
    else:
        example = torch.randn(4, 64)
        calibration = [torch.randn(16, 64) for _ in range(20)]
    config = QuantizationConfig(
        bits=4, method=method, min_quantized_values=1, min_calibration_rows=64,
    )
    expected_model = copy.deepcopy(model)
    quantize_model(
        expected_model, config, calibration if method == "gptq" else None,
        orientation_inputs=[(example,)],
    )

    precisions = []
    original = sprite_module._static_storage_precision

    def record(tensor, storage, *args):
        precision = original(tensor, storage, *args)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            precisions.append(precision)
        return precision

    monkeypatch.setattr(sprite_module, "_static_storage_precision", record)
    result = transpile(
        model, example, tmp_path / f"{model_type.__name__}-{method}",
        quantization=config,
        calibration_inputs=calibration if method == "gptq" else None,
    )
    assert precisions and set(precisions) == {"int4"}
    names = [tensor.name for tensor in result.quantization.tensors]
    assert len(names) == len(set(names)), names

    emulator = ScratchEmulator(_load(result.path))
    emulator.lists["input"] = example.flatten().tolist()
    emulator.run_procedure("cattorch forward")
    expected = expected_model(example).flatten()
    torch.testing.assert_close(
        torch.tensor([float(value) for value in emulator.lists["output"]]),
        expected, atol=1e-4, rtol=1e-5,
    )
