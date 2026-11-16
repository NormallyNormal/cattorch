"""The quantization report, warnings and verify's reference say what storage stored."""

import warnings

import pytest

import torch
import torch.nn.functional as F
from sprite_helpers import load_sprite

from cattorch import (
    EntryPoint, ExportProgram, Input, Output, QuantizationConfig, transpile, verify,
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
        bits=4, method=method, min_quantized_values=1024, min_calibration_rows=8,
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
