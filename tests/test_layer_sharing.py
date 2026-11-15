"""``CodegenConfig(layer_sharing=True)``: what it shares, and the warning when it can't."""

import warnings

import pytest
import torch
from torch import nn

from cattorch import (
    CattorchWarning, CodegenConfig, EntryPoint, ExportProgram, Input, Output, transpile, verify,
)
from sprite_helpers import load_sprite

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
