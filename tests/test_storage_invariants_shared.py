"""The storage invariants of ``test_storage_invariants`` with layer sharing on.

Shared layers keep each per-layer weight in one bank, stored under the first
layer's list. For plain and generation exports, a bank within one list and
one split over several, and a model with one layer quantized differently
(which isn't shared, with a warning):

* the independently decoded payloads are the lists the emulator holds;
* the report has one entry per parameter, at the precision stored, and its
  values per precision are the values the payloads hold;
* ``verify``'s reference, which rounds as storage stores, matches the sprite.
"""

import math
import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from cattorch import (
    CattorchWarning, CodegenConfig, GenerationProgram, QuantizationConfig, transpile, verify,
)
from cattorch.util.scratch.emulator import ScratchEmulator
from sprite_helpers import load_sprite
from test_storage_invariants import decode_payloads

V, D, H, LAYERS = 32, 32, 64, 3
SHARED = CodegenConfig(layer_sharing=True)
PROCEDURE = "cattorch shared transformer layer"


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.up = nn.Linear(D, H, bias=False)
        self.down = nn.Linear(H, D, bias=False)

    def forward(self, value):
        q, k, v = self.qkv(value).chunk(3, dim=-1)
        value = value + F.scaled_dot_product_attention(
            q[:, None], k[:, None], v[:, None], is_causal=True,
        )[:, 0]
        return value + self.down(torch.tanh(self.up(value)))


class _Layered(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.blocks = nn.ModuleList(_Block() for _ in range(LAYERS))
        self.head = nn.Linear(D, V, bias=False)

    def forward(self, tokens):
        value = self.emb(tokens)
        for block in self.blocks:
            value = block(value)
        return self.head(value)


class _TwoLayouts(_Block):
    """A layer reading one weight as ``F.linear``'s and as a matmul's right side.

    Storage keeps it in integers for one layout and a float16 copy for the
    other, in a bank per layout.
    """

    def __init__(self):
        super().__init__()
        del self.up, self.down
        self.w = nn.Parameter(torch.randn(H, D) * 0.2)

    def forward(self, value):
        q, k, v = self.qkv(value).chunk(3, dim=-1)
        value = value + F.scaled_dot_product_attention(
            q[:, None], k[:, None], v[:, None], is_causal=True,
        )[:, 0]
        return value + torch.tanh(torch.tanh(F.linear(value, self.w)) @ self.w)


class _Copies(_Layered):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(_TwoLayouts() for _ in range(LAYERS))


def _parameters(model):
    return {name for name, _ in model.named_parameters()}


def _shared(result):
    return any(
        block.get("mutation", {}).get("proccode", "").endswith(PROCEDURE)
        for block in load_sprite(result.path)["blocks"].values()
    )


def _check_invariants(result, check, model, bits, *, head_bits=None):
    sprite = load_sprite(result.path)
    lists, stored = decode_payloads(sprite)
    assert lists
    emulator = ScratchEmulator(sprite)
    emulator.run_procedure("cattorch init")
    names = {entry[0]: identifier for identifier, entry in sprite["lists"].items()}
    for name, values in lists.items():
        found = np.array([float(item) for item in emulator.lists._by_id[names[name]]])
        np.testing.assert_array_equal(found, values, err_msg=name)

    report = result.quantization
    entries: dict[str, list] = {}
    for item in report.tensors:
        entries.setdefault(item.name, []).append(item)
    assert all(len(found) == 1 for found in entries.values()), report
    # Every stored parameter once, and a float16 copy of one under its own
    # ``name[float16]`` entry.
    parameters = _parameters(model)
    assert parameters <= set(entries), report
    assert set(entries) <= parameters | {f"{name}[float16]" for name in parameters}, report
    contents: dict[str, set] = {}
    held: dict[str, int] = {}
    for name, (precision, _) in stored.items():
        values = lists[name].tobytes()
        if values not in contents.setdefault(precision, set()):
            contents[precision].add(values)
            held[precision] = held.get(precision, 0) + len(lists[name])
    reported: dict[str, int] = {}
    for item in report.tensors:
        precision = "float16" if item.method == "float16" else f"int{item.bits}"
        reported[precision] = reported.get(precision, 0) + item.values
    assert reported == held, (reported, held, report)

    outcome = check()
    assert outcome.passed, outcome
    worst = max(
        (
            value.max_abs_error
            for value in getattr(outcome, "outputs", None) or [
                output for call in getattr(outcome, "calls", ()) for output in call.outputs
            ] or [outcome]
        ),
        default=0.0,
    )
    assert math.isfinite(worst) and worst < 1e-4
    return entries


def _export(kind, model, path, config):
    tokens = torch.randint(0, V, (1, 4), generator=torch.Generator().manual_seed(1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if kind == "tensor":
            result = transpile(model, tokens, path, quantization=config, codegen=SHARED)
            check = lambda: verify(model, tokens, result, quantization=config)  # noqa: E731
        else:
            program = GenerationProgram(max_context=4)
            result = transpile(model, program, path, quantization=config, codegen=SHARED)
            check = lambda: verify(model, program, result, quantization=config)  # noqa: E731
    emitted = [str(item.message) for item in caught if issubclass(item.category, CattorchWarning)]
    return result, check, emitted


@pytest.mark.parametrize("limit", [None, 4000], ids=["whole", "sharded"])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("kind", ["tensor", "generation"])
def test_shared_banks_store_report_and_verify_alike(tmp_path, small_list_limit, kind, bits, limit):
    if limit is not None:
        # Each bank of three layers' up weights (6,144 values) spans two lists.
        small_list_limit(limit)
    torch.manual_seed(bits)
    model = _Layered().eval()
    config = QuantizationConfig(bits=bits, min_quantized_values=512)
    result, check, emitted = _export(kind, model, tmp_path / "sprite", config)
    assert _shared(result), emitted
    assert not any("layer_sharing" in message for message in emitted), emitted
    entries = _check_invariants(result, check, model, bits)
    for name, (item,) in entries.items():
        if name.startswith("blocks."):
            assert (item.method, item.bits) == ("symmetric", bits), item


@pytest.mark.parametrize("kind", ["tensor", "generation"])
def test_a_differently_quantized_layer_keeps_its_storage_unshared(tmp_path, kind):
    torch.manual_seed(0)
    model = _Layered().eval()
    config = QuantizationConfig(bits=4, min_quantized_values=512, overrides={"blocks.2.down": 8})
    result, check, emitted = _export(kind, model, tmp_path / "sprite", config)
    assert not _shared(result)
    assert any("are stored in different formats" in message for message in emitted), emitted
    entries = _check_invariants(result, check, model, 4)
    assert entries["blocks.2.down.weight"][0].bits == 8
    assert entries["blocks.1.down.weight"][0].bits == 4


class _WideBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(160, 160)

    def forward(self, value):
        return value + torch.tanh(self.linear(value))


class _Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(_WideBlock() for _ in range(8))

    def forward(self, value):
        for block in self.blocks:
            value = block(value)
        return value


def test_a_bank_over_the_list_limit_reports_no_float16(tmp_path):
    # Eight 160 x 160 weights bank 204,800 values, past Scratch's 200,000-item
    # list limit; storage keeps both lists in int8, and so does the report.
    torch.manual_seed(0)
    model = _Wide().eval()
    value = torch.randn(1, 160)
    config = QuantizationConfig(bits=8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = transpile(model, value, tmp_path / "wide", quantization=config, codegen=SHARED)
    emitted = [str(item.message) for item in caught if issubclass(item.category, CattorchWarning)]
    assert _shared(result)
    assert not any("float16" in message for message in emitted), emitted
    weights = [item for item in result.quantization.tensors if item.name.endswith(".weight")]
    assert len(weights) == 8
    assert all((item.method, item.bits, item.values) == ("symmetric", 8, 25600) for item in weights)
    assert not any(item.method == "float16" for item in result.quantization.tensors)
    sprite = load_sprite(result.path)
    lists, stored = decode_payloads(sprite)
    assert sum(len(lists[name]) for name, (precision, _) in stored.items() if precision == "int8") \
        >= 204_800
    assert verify(model, value, result, quantization=config).passed


@pytest.mark.parametrize("kind", ["tensor", "generation"])
def test_float16_copies_in_shared_banks_are_reported_per_layer(tmp_path, kind):
    torch.manual_seed(0)
    model = _Copies().eval()
    config = QuantizationConfig(bits=8, min_quantized_values=512)
    result, check, emitted = _export(kind, model, tmp_path / "sprite", config)
    assert _shared(result), emitted
    entries = _check_invariants(result, check, model, 8)
    for layer in range(LAYERS):
        (item,) = entries[f"blocks.{layer}.w[float16]"]
        assert item.values == H * D, item
