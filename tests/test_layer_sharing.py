"""``CodegenConfig(layer_sharing=True)``: what it shares, and the warning when it can't."""

import warnings

import pytest
import torch
from torch import nn

from cattorch import (
    CattorchWarning, CodegenConfig, EntryPoint, ExportProgram, Input, Output,
    QuantizationConfig, QuantizationOverride, transpile, verify,
)
from cattorch.quantization import _StaticSourceRecorder
from sprite_helpers import load_emulator, load_sprite

SHARED = CodegenConfig(layer_sharing=True)
PROCEDURE = "cattorch shared transformer layer"


class Block(nn.Module):
    def __init__(self, width=8, hidden=16):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, value):
        return value + self.down(torch.relu(self.up(self.norm(value))))


class Stack(nn.Module):
    """Repeated ``blocks.N`` layers, with what runs around them chosen by ``kind``."""

    def __init__(self, kind="input", hiddens=(16, 16, 16)):
        super().__init__()
        self.kind = kind
        self.inp = nn.Linear(8, 8)
        self.blocks = nn.ModuleList(Block(8, hidden) for hidden in hiddens)

    def forward(self, value):
        raw = value
        if self.kind.startswith("input"):
            value = self.inp(value)
        before = value
        for block in self.blocks:
            value = block(value)
        if self.kind.endswith("skip"):
            return value + 3 * before
        if self.kind.endswith("both"):
            return value, before
        if self.kind.endswith("raw"):
            # An output copied from the input, after the last layer.
            return value, raw
        return value

    def clear(self, value):
        return value * 0


def shared_procedure(result):
    return any(
        block.get("mutation", {}).get("proccode", "").endswith(PROCEDURE)
        for block in load_sprite(result.path)["blocks"].values()
    )


def export(model, inputs, path, codegen=SHARED):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = transpile(model, inputs, path, codegen=codegen)
    emitted = [
        str(item.message) for item in caught if issubclass(item.category, CattorchWarning)
    ]
    return result, emitted


@pytest.mark.parametrize("kind", [
    # The layers start the model, and read its input list.
    "plain", "skip", "both",
    # The layers follow a Linear whose output is read again after them.
    "input", "input_skip", "input_both", "input_raw",
])
def test_layers_share_wherever_they_sit(tmp_path, kind):
    torch.manual_seed(3)
    model = Stack(kind).eval()
    value = torch.randn(1, 4, 8)
    plain, _ = export(model, value, tmp_path / "plain", CodegenConfig())
    result, emitted = export(model, value, tmp_path / "shared")
    assert not result.warnings and not emitted
    assert shared_procedure(result)
    assert result.block_count < plain.block_count
    assert verify(model, value, result).passed


def test_shared_layers_leave_the_input_list_unchanged(tmp_path):
    from cattorch.util.scratch.emulator import ScratchEmulator

    torch.manual_seed(4)
    model = Stack("plain").eval()
    value = torch.randn(1, 4, 8)
    result, _ = export(model, value, tmp_path / "shared")
    emulator = ScratchEmulator(load_sprite(result.path))
    emulator.lists["input"] = value.flatten().tolist()
    emulator.run_procedure("cattorch forward")
    assert emulator.lists["input"] == value.flatten().tolist()


def test_unshareable_layers_warn_with_the_reason(tmp_path):
    model = Stack("input", hiddens=(16, 16, 12)).eval()
    with pytest.warns(CattorchWarning, match=r"`blocks\.2` doesn't compute the same operations"):
        result = transpile(model, torch.randn(1, 4, 8), tmp_path / "mixed", codegen=SHARED)
    (message,) = result.warnings
    assert message.startswith("layer_sharing=True didn't share the model's layers")
    assert message.endswith("it was exported without layer sharing")
    assert not shared_procedure(result)


def test_a_model_without_repeated_layers_warns_only_when_sharing_was_asked_for(tmp_path):
    model = nn.Linear(4, 3).eval()
    result, emitted = export(model, torch.randn(1, 4), tmp_path / "asked")
    assert result.warnings == tuple(emitted)
    assert len(emitted) == 1 and "no repeated layers to share" in emitted[0]
    assert "`blocks.0`" in emitted[0]
    # layer_sharing is off by default: nothing to say.
    result, emitted = export(model, torch.randn(1, 4), tmp_path / "default", CodegenConfig())
    assert not result.warnings and not emitted


def _program(*names):
    example = torch.randn(1, 4, 8)
    return ExportProgram(entrypoints=tuple(
        EntryPoint(
            name, method=name,
            arguments=(Input("value", example),), returns=(Output("out"),),
        )
        for name in names
    ))


@pytest.mark.parametrize("kind", ["input", "skip"])
def test_program_entry_points_without_layers_are_not_reported(tmp_path, kind):
    torch.manual_seed(5)
    model = Stack(kind).eval()
    program = _program("forward", "clear")
    result, emitted = export(model, program, tmp_path / "program")
    assert not result.warnings and not emitted
    assert shared_procedure(result)
    assert verify(model, program, result).passed


def test_program_reports_the_entry_point_whose_layers_did_not_share(tmp_path):
    model = Stack("input", hiddens=(16, 12)).eval()
    result, emitted = export(model, _program("forward", "clear"), tmp_path / "program")
    (message,) = result.warnings
    assert emitted == [message]
    assert "layers of entry point 'forward'" in message and "`blocks.1`" in message


def test_program_without_any_layers_warns_once(tmp_path):
    class Plain(nn.Module):
        def forward(self, value):
            return value + 1

        def clear(self, value):
            return value * 0

    result, emitted = export(Plain().eval(), _program("forward", "clear"), tmp_path / "program")
    (message,) = result.warnings
    assert emitted == [message]
    assert "none of its entry points has repeated layers" in message


class Pair(nn.Module):
    """A layer that takes and returns two tensors."""

    def __init__(self):
        super().__init__()
        self.f = nn.Linear(8, 8)
        self.g = nn.Linear(8, 8)

    def forward(self, a, b):
        return a + self.f(b), b + self.g(a)


class Dataflow(nn.Module):
    """Layers whose tensors are read in ways one carried tensor can't serve."""

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.blocks = nn.ModuleList(
            (Pair() if kind in ("pairs", "embeddings") else Block()) for _ in range(3)
        )

    def forward(self, value):
        if self.kind == "pairs":
            # Each layer passes two tensors to the next.
            a, b = value, value * 2
            for block in self.blocks:
                a, b = block(a, b)
            return a + b
        if self.kind == "embeddings":
            # Every layer reads the model's input as well as the previous layer.
            hidden = value
            for block in self.blocks:
                hidden, _ = block(hidden, value)
            return hidden
        outputs = []
        for block in self.blocks:
            value = block(value)
            outputs.append(value)
        if self.kind == "skip":
            # Layer 0's output, read again after layer 1 overwrote it.
            return value + outputs[0]
        if self.kind == "all":
            return tuple(outputs)
        assert self.kind == "between"
        return value


class Between(Dataflow):
    def forward(self, value):
        for block in self.blocks:
            value = block(value) * 0.9
        return value


class TwoLists(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.blocks = nn.ModuleList(Block() for _ in range(2))
        self.decoder = nn.Module()
        self.decoder.blocks = nn.ModuleList(Block() for _ in range(2))

    def forward(self, value):
        for block in (*self.encoder.blocks, *self.decoder.blocks):
            value = block(value)
        return value


@pytest.mark.parametrize("model, reason", [
    (Dataflow("pairs"), "more than one tensor of layer `blocks.0` is read outside it"),
    (Dataflow("embeddings"), "layer `blocks.0` reads different tensors from the other layers"),
    (Dataflow("skip"), "layer `blocks.0`'s output is read after the next layer"),
    (Dataflow("all"), "layer `blocks.0`'s output is read after the next layer"),
    (Between("between"), "code from outside the `blocks.N` layers runs between them"),
    (TwoLists(), "more than one `blocks` list (`decoder.blocks`, `encoder.blocks`)"),
], ids=["pairs", "embeddings", "skip", "all", "between", "two-lists"])
def test_layers_whose_tensors_one_call_cant_carry_are_not_shared(tmp_path, model, reason):
    # Shared, these computed wrong numbers: every call runs layer 0's code on
    # layer 0's lists and carries only one tensor to the next call.
    torch.manual_seed(6)
    model.eval()
    value = torch.randn(1, 4, 8)
    result, emitted = export(model, value, tmp_path / "shared")
    (message,) = result.warnings
    assert emitted == [message] and reason in message
    assert not shared_procedure(result)
    assert verify(model, value, result).passed


def test_cached_generation_with_a_skip_across_layers_is_not_shared(tmp_path):
    from cattorch import GenerationProgram
    from shared_models import CachedLM
    from sprite_helpers import load_emulator, prefill_and_decode

    class Skip(CachedLM):
        def forward(self, tokens):
            x = self.token(tokens) + self.position(torch.arange(tokens.shape[1]))
            first = x = self.blocks[0](x)
            for block in self.blocks[1:]:
                x = block(x)
            return self.head(self.norm(x + first))

    torch.manual_seed(8)
    model = Skip(context=8, layers=3).eval()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = transpile(model, GenerationProgram(max_context=8), tmp_path / "skip", codegen=SHARED)
    emitted = [str(item.message) for item in caught if issubclass(item.category, CattorchWarning)]
    (message,) = result.warnings
    assert emitted == [message] and "read after the next layer" in message
    assert not shared_procedure(result)
    prefill_and_decode(load_emulator(result), model, [1, 3, 5], [2, 4])


class OutsideReader(nn.Module):
    """Three layers, and code outside them reading one layer's weight.

    ``where`` is ``"head"`` (after the layers) or ``"before"``; ``how`` is
    ``"tied"`` (the same parameter) or ``"equal"`` (another parameter with
    the same values, which storage keeps once).
    """

    def __init__(self, layer, where="head", how="tied"):
        super().__init__()
        self.where = where
        self.blocks = nn.ModuleList(Block() for _ in range(3))
        self.head = nn.Linear(8, 16, bias=False)
        if how == "tied":
            self.head.weight = self.blocks[layer].up.weight
        else:
            with torch.no_grad():
                self.head.weight.copy_(self.blocks[layer].up.weight)

    def forward(self, value):
        before = self.head(value) if self.where == "before" else None
        for block in self.blocks:
            value = block(value)
        out = self.head(value) if self.where == "head" else value
        return (out, before) if before is not None else out


@pytest.mark.parametrize("layer", [0, 1, 2])
@pytest.mark.parametrize("where,how", [("head", "tied"), ("head", "equal"), ("before", "tied")])
def test_code_outside_the_layers_reading_a_layer_weight_reads_its_values(
    tmp_path, layer, where, how,
):
    torch.manual_seed(layer)
    model = OutsideReader(layer, where, how).eval()
    value = torch.randn(2, 8)
    result, emitted = export(model, value, tmp_path / f"{where}{how}{layer}")
    assert emitted == []
    assert shared_procedure(result)
    assert verify(model, value, result).passed


@pytest.mark.parametrize("layer", [1, 2])
def test_a_quantized_tied_head_reads_the_same_stored_values_as_unshared(tmp_path, layer):
    torch.manual_seed(layer)
    model = OutsideReader(layer).eval()
    value = torch.randn(2, 8)
    quantization = QuantizationConfig(bits=8, min_quantized_values=16)
    outputs = []
    for codegen in (CodegenConfig(), SHARED):
        result = transpile(
            model, value, tmp_path / f"q{codegen.layer_sharing}", codegen=codegen,
            quantization=quantization,
        )
        emulator = load_emulator(result)
        emulator.lists["input"] = value.flatten().tolist()
        emulator.run_procedure("cattorch forward")
        outputs.append([float(item) for item in emulator.lists["output"]])
    assert shared_procedure(result)
    assert outputs[0] == outputs[1]


class CodeBetween(nn.Module):
    """``side`` is computed in the layer loop after ``blocks.where``, from no layer output."""

    def __init__(self, where, weight=False):
        super().__init__()
        self.where = where
        self.inp = nn.Linear(8, 8)
        self.side = nn.Linear(8, 8) if weight else None
        self.blocks = nn.ModuleList(Block() for _ in range(3))

    def forward(self, value):
        start = self.inp(value)
        hidden = start
        side = None
        for index, block in enumerate(self.blocks):
            hidden = block(hidden)
            if index == self.where:
                side = self.side(value) if self.side is not None else torch.tanh(start)
        return hidden + side


@pytest.mark.parametrize("where", [0, 1])
@pytest.mark.parametrize("weight", [False, True])
def test_code_between_layers_that_reads_no_layer_output_is_not_shared(tmp_path, where, weight):
    torch.manual_seed(where)
    model = CodeBetween(where, weight).eval()
    value = torch.randn(1, 8)
    result, emitted = export(model, value, tmp_path / f"between{where}{weight}")
    assert not shared_procedure(result)
    assert len(emitted) == 1
    assert "code from outside the `blocks.N` layers runs between them" in emitted[0]
    assert verify(model, value, result).passed


class FusedLayer(nn.Module):
    """``blocks.0`` is only an activation, which the Linear before it absorbs."""

    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(8, 8)
        self.blocks = nn.ModuleList([nn.ReLU(), nn.Linear(8, 8), nn.Linear(8, 8)])

    def forward(self, value):
        value = self.inp(value)
        for block in self.blocks:
            value = block(value)
        return value


def test_a_layer_fused_into_outside_code_gets_an_accurate_reason(tmp_path):
    model = FusedLayer().eval()
    value = torch.randn(1, 8)
    result, emitted = export(model, value, tmp_path / "fused")
    assert len(emitted) == 1
    assert "aren't numbered consecutively" not in emitted[0]
    assert "`blocks.0` is compiled together with code from outside the layers" in emitted[0]
    assert verify(model, value, result).passed


class MLPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 16)

    def forward(self, value):
        return value + self.fc2(torch.relu(self.fc(value)))


class MLPStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(MLPBlock() for _ in range(4))

    def forward(self, value):
        for block in self.blocks:
            value = block(value)
        return value


@pytest.mark.parametrize("overrides", [
    {"blocks.3.fc2": 8},
    {"blocks.1": None},
    {"blocks.2.fc2": QuantizationOverride(bits=4, group_size=32)},
], ids=["bits", "unquantized", "group"])
def test_layers_quantized_differently_are_not_banked_together(tmp_path, overrides):
    torch.manual_seed(0)
    model = MLPStack().eval()
    value = torch.randn(1, 3, 16)
    quantization = QuantizationConfig(bits=4, min_quantized_values=16, overrides=overrides)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = transpile(
            model, value, tmp_path / "override", codegen=SHARED, quantization=quantization,
        )
    emitted = [str(item.message) for item in caught if issubclass(item.category, CattorchWarning)]
    assert not shared_procedure(result)
    assert len(emitted) == 1 and "is stored differently from" in emitted[0], emitted
    # Every layer keeps the storage the configuration asks for.
    assert not any(item.name.endswith("[float16]") for item in result.quantization.tensors)
    assert verify(model, value, result, quantization=quantization).passed


def test_a_shared_bank_is_traced_to_every_layer():
    """The bank is stored under layer 0's key but holds every layer's weights."""
    recorder = object.__new__(_StaticSourceRecorder)
    weights = {f"w{index}": torch.randn(4, 4) for index in range(3)}
    recorder.resolve = weights.get
    recorder.sources = {}
    recorder.banks = {"w0": tuple(weights.items())}
    bank = torch.cat(list(weights.values()))
    assert recorder.graph_values("w0", bank) == ("w0", "w1", "w2")
    assert recorder.graph_values("w1", weights["w1"]) == ("w1",)
