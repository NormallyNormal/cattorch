"""The quantization report, warnings and verify's reference say what storage stored."""

import warnings

import pytest

import torch
import torch.nn.functional as F
from sprite_helpers import load_sprite

from cattorch import (
    EntryPoint, ExportProgram, Input, Output, ProgramCall, QuantizationConfig, transpile, verify,
)
from cattorch.util.scratch.emulator import ScratchEmulator


def _transpile(*args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = transpile(*args, **kwargs)
    return result, [str(item.message) for item in caught]


def _entries(report):
    return [(item.name, item.method, item.bits, item.values) for item in report.tensors]


class _TwoLayouts(torch.nn.Module):
    """``w`` read as a linear weight in ``enc`` and on the right in ``dec``."""

    def __init__(self, rows=64, columns=32):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(rows, columns) * 0.1)

    def enc(self, value):
        return F.linear(value, self.w)

    def dec(self, value):
        return value @ self.w


def _two_layout_program(order, rows=64, columns=32):
    entries = {
        "enc": EntryPoint(
            "enc", method="enc", arguments=(Input("x", torch.randn(1, columns)),),
            returns=(Output("h"),),
        ),
        "dec": EntryPoint(
            "dec", method="dec", arguments=(Input("h", torch.randn(1, rows)),),
            returns=(Output("y"),),
        ),
    }
    return ExportProgram(entrypoints=tuple(entries[name] for name in order))


def test_program_reports_a_later_entry_points_float16_copy(tmp_path):
    torch.manual_seed(0)
    model = _TwoLayouts().eval()
    config = QuantizationConfig(bits=4, min_quantized_values=1024)
    for order in (("enc", "dec"), ("dec", "enc")):
        result, caught = _transpile(
            model, _two_layout_program(order), tmp_path / "_".join(order), quantization=config,
        )
        # Whichever entry point comes first, one entry point stores w in
        # integers and the other a float16 copy of them.
        assert _entries(result.quantization) == [
            ("w", "symmetric", 4, 2048), ("w[float16]", "float16", None, 2048),
        ]
        assert result.quantization.quantized_values == 2048
        copies = [item for item in caught if item.startswith("quantized weight")]
        assert copies == [
            "quantized weight 'w': 2048 of its values are stored as float16, not int4 "
            "(a second copy): storage reads the values in groups along a different axis "
            "than they were quantized in"
        ]


class _MaskedInMethods(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(64, 32) * 0.1)
        self.register_buffer("mask", (torch.rand(64, 32) > 0.3).float())

    def a(self, value):
        return F.linear(value, self.w * self.mask)

    def b(self, value):
        return -F.linear(value, self.w * self.mask)


def test_program_lists_a_weight_its_entry_points_compute_once(tmp_path):
    torch.manual_seed(0)
    program = ExportProgram(entrypoints=tuple(
        EntryPoint(
            name, method=name, arguments=(Input(f"x{name}", torch.randn(1, 32)),),
            returns=(Output(f"y{name}"),),
        )
        for name in ("a", "b")
    ))
    result, caught = _transpile(
        _MaskedInMethods().eval(), program, tmp_path / "masked",
        quantization=QuantizationConfig(bits=4, min_quantized_values=1024),
    )
    names = [item.name for item in result.quantization.tensors]
    assert names.count("weight computed from w") == 1
    assert sum(item.startswith("weight computed from w") for item in caught) == 1


def test_short_last_row_shard_is_reported_as_stored(tmp_path, small_list_limit):
    small_list_limit(3000)
    torch.manual_seed(0)
    model = torch.nn.Linear(64, 100, bias=False).eval()
    # Row shards of 46 rows; the last one, 8 rows (512 values), is shorter
    # than min_quantized_values but stored in int8 like the rest.
    result, caught = _transpile(
        model, torch.randn(1, 64), tmp_path / "shards",
        quantization=QuantizationConfig(bits=8, min_quantized_values=1024),
    )
    assert _entries(result.quantization) == [("weight", "symmetric", 8, 6400)]
    assert not [item for item in caught if "float16" in item]
    assert verify(
        model, torch.randn(1, 64), result,
        quantization=QuantizationConfig(bits=8, min_quantized_values=1024),
    ).passed


class _PositionTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wpe = torch.nn.Embedding(512, 64)
        self.head = torch.nn.Linear(64, 4)

    def forward(self, value):
        return self.head(value + self.wpe(torch.arange(value.shape[1])))


def test_position_table_read_at_constant_positions_is_reported(tmp_path):
    torch.manual_seed(0)
    result, _ = _transpile(
        _PositionTable().eval(), torch.randn(1, 8, 64), tmp_path / "positions",
        quantization=QuantizationConfig(bits=8),
    )
    (entry,) = [item for item in result.quantization.tensors if item.name == "wpe.weight"]
    # The eight rows read are stored, as they are.
    assert (entry.method, entry.values) == ("float16", 8 * 64)
    assert "positions fixed at export" in entry.fallback_reason


def test_generation_position_table_is_reported_once(tmp_path):
    transformers = pytest.importorskip("transformers")
    from cattorch import GenerationProgram

    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=64, n_positions=512, n_embd=32, n_layer=1, n_head=2,
        bos_token_id=0, eos_token_id=0,
    )).eval()
    for minimum in (1, 16384):
        result, _ = _transpile(
            model, GenerationProgram(max_context=8), tmp_path / f"gpt2_{minimum}",
            quantization=QuantizationConfig(bits=8, min_quantized_values=minimum),
        )
        entries = [
            item for item in result.quantization.tensors
            if item.name == "transformer.wpe.weight"
        ]
        assert [(item.method, item.values) for item in entries] == [("float16", 8 * 32)]


class _TiedCopy(torch.nn.Module):
    """A table gathered directly and read as the head through a copy."""

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(128, 32)
        self.mix = torch.nn.Linear(32, 32)

    def forward(self, tokens):
        hidden = torch.tanh(self.mix(self.emb(tokens)))
        return hidden @ self.emb.weight.t().contiguous()


class _LinearAndCopy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(128, 32) * 0.1)

    def forward(self, value):
        return F.linear(value, self.w) + F.linear(value, self.w.clone())


@pytest.mark.parametrize("method", ["symmetric", "gptq"])
@pytest.mark.parametrize("kind", ["tied", "clone"])
def test_weight_read_directly_and_through_a_copy_is_rounded_once(tmp_path, method, kind):
    torch.manual_seed(0)
    model = (_TiedCopy() if kind == "tied" else _LinearAndCopy()).eval()
    name = "emb.weight" if kind == "tied" else "w"
    example = torch.randint(0, 128, (1, 8)) if kind == "tied" else torch.randn(1, 32)
    calibration = (
        [torch.randint(0, 128, (1, 8)) for _ in range(20)] if kind == "tied"
        else [torch.randn(16, 32) for _ in range(10)]
    ) if method == "gptq" else None
    config = QuantizationConfig(
        bits=4, method=method, min_quantized_values=1024,
        **({"min_calibration_rows": 8} if method == "gptq" else {}),
    )
    result, caught = _transpile(
        model, example, tmp_path / "copy", quantization=config, calibration=calibration,
    )
    entries = [item for item in result.quantization.tensors if item.name == name]
    assert [(item.method, item.values) for item in entries] == [(method, 4096)]
    assert not [item for item in caught if name in item]
    # The parameter is stored in one list.
    emulator = ScratchEmulator(load_sprite(result.path))
    emulator.run_procedure("cattorch init")
    assert sum(len(values) == 4096 for values in emulator.lists._by_id.values()) == 1
    if method == "symmetric":
        assert verify(model, example, result, quantization=config).passed


class _PlainMask(torch.nn.Module):
    """A mask held as a plain attribute, not a buffer."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(64, 32) * 0.1)
        self.mask = (torch.rand(64, 32) > 0.3).float()

    def forward(self, value):
        return F.linear(value, self.w * self.mask)


class _MaskInForward(torch.nn.Module):
    """A mask built in forward from constants alone."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(64, 32) * 0.1)

    def forward(self, value):
        return F.linear(value, self.w * torch.tril(torch.ones(64, 32)))


@pytest.mark.parametrize("model_type", [_PlainMask, _MaskInForward])
def test_weight_masked_by_a_known_constant_is_quantized_as_computed(tmp_path, model_type):
    torch.manual_seed(0)
    model = model_type().eval()
    example = torch.randn(1, 32)
    config = QuantizationConfig(bits=4, min_quantized_values=1024)
    result, caught = _transpile(model, example, tmp_path / "mask", quantization=config)
    # w itself is not rounded: the masked weight is, once, as it is read.
    assert _entries(result.quantization) == [("weight computed from w", "symmetric", 4, 2048)]
    assert not [item for item in caught if "float16" in item]
    assert verify(model, example, result, quantization=config).passed


class _TwoLayoutsForward(torch.nn.Module):
    def __init__(self, computed=False):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(64, 32))
        self.computed = computed

    def forward(self, value):
        weight = self.w * 0.5 if self.computed else self.w
        return torch.tanh(F.linear(value, weight)) @ weight


@pytest.mark.parametrize("computed", [False, True])
def test_verify_reads_the_float16_copy_a_use_reads(tmp_path, computed):
    # F.linear reads w in integers, `@ w` a float16 copy of them: verify's
    # reference must read each, or it fails on a correct sprite.
    torch.manual_seed(0)
    model = _TwoLayoutsForward(computed).eval()
    example = torch.randn(1, 32)
    config = QuantizationConfig(bits=4, min_quantized_values=1024)
    result, _ = _transpile(model, example, tmp_path / "two", quantization=config)
    assert any(item.name.endswith("[float16]") for item in result.quantization.tensors)
    outcome = verify(model, example, result, quantization=config)
    assert outcome.passed and outcome.max_abs_error < 1e-4


def test_program_verify_reads_each_entry_points_list(tmp_path):
    torch.manual_seed(0)
    model = _TwoLayouts().eval()
    config = QuantizationConfig(bits=4, min_quantized_values=1024)
    for order in (("enc", "dec"), ("dec", "enc")):
        program = _two_layout_program(order)
        result, _ = _transpile(model, program, tmp_path / "_".join(order), quantization=config)
        calls = [
            ProgramCall("enc", (torch.randn(1, 32),)), ProgramCall("dec", (torch.randn(1, 64),)),
        ]
        assert verify(model, program, result, quantization=config, calls=calls).passed


def test_dynamic_length_check_accepts_a_float16_copy(tmp_path):
    torch.manual_seed(0)
    model = _TwoLayoutsForward().eval()
    _transpile(
        model, torch.randn(1, 32), tmp_path / "dynamic", dynamic={0: (1, 8)},
        quantization=QuantizationConfig(bits=4, min_quantized_values=1024),
    )


class _FactorizedPair(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(64, 256)
        self.b = torch.nn.Linear(256, 64)

    def forward(self, value):
        return self.b(torch.relu(self.a(value)))


def test_override_names_a_parameter_fast_mode_factorizes(tmp_path):
    from cattorch import FastConfig, FastLayerConfig

    torch.manual_seed(0)
    result, _ = _transpile(
        _FactorizedPair().eval(), torch.randn(1, 64), tmp_path / "fast",
        optimization="fast",
        fast_config=FastConfig(
            activations=False, layer_norm=False, softmax=False,
            layers=FastLayerConfig(rank=16),
        ),
        quantization=QuantizationConfig(
            bits=4, min_quantized_values=1024, overrides={"a.weight": 8},
        ),
    )
    found = {item.name: (item.bits, item.override) for item in result.quantization.tensors}
    # The factors of a.weight take its override, under the key as written.
    assert found["a.0.weight"] == found["a.1.weight"] == (8, "a.weight")
    assert found["b.0.weight"] == (4, None)


class _ConvBatchNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c = torch.nn.Conv2d(32, 32, 3)
        self.bn = torch.nn.BatchNorm2d(32)

    def forward(self, value):
        return self.bn(self.c(value))


@pytest.mark.parametrize("key", ["bn", "bn.weight"])
def test_override_naming_a_folded_batch_norm_names_its_layer(tmp_path, key):
    with pytest.raises(ValueError, match="folds into the layer before it, 'c'; override 'c'"):
        transpile(
            _ConvBatchNorm().eval(), torch.randn(1, 32, 6, 6), tmp_path / "bn",
            quantization=QuantizationConfig(bits=4, min_quantized_values=1024, overrides={key: 8}),
        )



class _BigramLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = torch.nn.Embedding(64, 32)
        self.fc = torch.nn.Linear(32, 64)

    def forward(self, tokens):
        return self.fc(self.wte(tokens))


_GPTQ = QuantizationConfig(bits=4, method="gptq", min_quantized_values=1024)


@pytest.mark.parametrize(("bad", "message"), [
    (torch.randint(0, 64, (1, 9)), r"calibration\[1\] has shape \(1, 9\).*1 to 8 tokens"),
    (torch.randint(0, 64, (8,)), r"calibration\[1\] has shape \(8,\)"),
    (torch.full((1, 4), 64), r"calibration\[1\] has token 64"),
])
def test_generation_calibration_items_are_checked_up_front(tmp_path, bad, message):
    from cattorch import GenerationProgram

    good = torch.randint(0, 64, (1, 8))
    with pytest.raises(ValueError, match=message):
        transpile(
            _BigramLM().eval(), GenerationProgram(max_context=8), tmp_path / "lm",
            quantization=_GPTQ, calibration=[good, bad, good],
        )


@pytest.mark.parametrize(("bad", "message"), [
    (torch.full((1, 8), 64), r"failed on calibration\[1\] \(inputs \(1, 8\) int64\)"),
    (torch.randint(0, 64, (8,)), r"calibration\[1\] \(inputs \(8,\) int64\).*1 dimensions, not 2"),
    (torch.randn(1, 8), r"calibration\[1\] .*input 0 is floating-point, not integer"),
    ((torch.randint(0, 64, (1, 8)),) * 2, r"calibration\[1\] .*2 tensors, not 1"),
])
def test_tensor_calibration_items_name_the_item(tmp_path, bad, message):
    good = torch.randint(0, 64, (1, 8))
    with pytest.raises(ValueError, match=message):
        transpile(
            _BigramLM().eval(), good, tmp_path / "lm",
            quantization=_GPTQ, calibration=[good, bad, good],
        )


class _LargeActivations(torch.nn.Module):
    """Four residual MLP blocks whose activations reach about 1e5."""

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 64),
            )
            for _ in range(4)
        )
        with torch.no_grad():
            for block in self.blocks:
                block[0].weight.mul_(8)
                block[2].weight.mul_(8)
        self.out = torch.nn.Linear(64, 8)

    def forward(self, value):
        for block in self.blocks:
            value = value + block(value)
        return self.out(value)


def test_dynamic_length_check_tolerates_float32_noise_of_large_outputs(tmp_path):
    # The check's reference runs in float32: an output near zero among
    # others near 1e5 differs from the sprite's by noise of their size.
    torch.manual_seed(0)
    model = _LargeActivations().eval()
    _transpile(
        model, torch.randn(3, 64), tmp_path / "large", dynamic={0: (1, 8)},
        quantization=QuantizationConfig(bits=4, overrides={"blocks.3.2": 8}),
    )
