"""Experimental declarative markers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from cattorch import GenerationProgram, UnsupportedModelError, transpile, verify
from cattorch import (
    ProgramCall, adapter, entrypoint, example, export_forward, generate, marked_program,
    moe,
)
from cattorch.util.scratch.emulator import ScratchEmulator

sys.path.insert(0, str(Path(__file__).parent))
from test_generation_automatic import load_sprite, nano, prefill_and_decode  # noqa: E402


class Accumulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("total", torch.zeros(2))
        self.register_buffer("history", torch.zeros(0, 2))
        self.project = nn.Linear(2, 2)

    @entrypoint(example=lambda: (torch.ones(2),), outputs=("doubled",))
    def add(self, value):
        self.total += value
        return self.total * 2

    @entrypoint(example=lambda: (torch.ones(2),), capacity={"history": 4})
    def remember(self, row):
        projected = self.project(row).reshape(1, 2)
        self.history = torch.cat([self.history, projected])
        return projected.sum()

    @entrypoint(name="recall", example=lambda: (torch.ones(2),))
    def read_history(self, weights):
        return torch.relu(self.history @ weights)

    def helper(self, value):
        """Not an entry point."""
        return value


def test_entrypoints_with_state_export_and_verify(tmp_path):
    torch.manual_seed(0)
    model = Accumulator().eval()
    result = transpile(model, None, tmp_path / "accumulator")
    entries = {entry.name: entry for entry in result.entrypoints}
    assert set(entries) == {"add", "remember", "recall"}
    assert [spec.list_name for spec in entries["add"].inputs] == ["cattorch add value"]
    assert [spec.list_name for spec in entries["add"].outputs] == ["cattorch add doubled"]
    assert entries["add"].state_updates == ("total",)
    assert entries["remember"].state_updates == ("history",)
    assert {state.name: state.mode for state in result.states} == {
        "history": "append", "total": "replace",
    }
    calls = (
        ProgramCall("add", (torch.tensor([1.0, 2.0]),)),
        ProgramCall("remember", (torch.tensor([3.0, 4.0]),)),
        ProgramCall("add", (torch.tensor([1.0, 1.0]),)),
        ProgramCall("remember", (torch.tensor([-1.0, 0.5]),)),
        ProgramCall("recall", (torch.tensor([1.0, -1.0]),)),
    )
    assert verify(model, None, result, calls=calls).passed
    # The model itself is untouched and still runs in PyTorch.
    assert torch.equal(model.total, torch.zeros(2))
    assert model.add(torch.ones(2)).tolist() == [2.0, 2.0]


def test_marked_program_builds_the_explicit_form():
    exported, program = marked_program(Accumulator().eval())
    assert {entry.name for entry in program.entrypoints} == {"add", "recall", "remember"}
    assert hasattr(exported, "_cattorch_entry_add")
    assert not hasattr(Accumulator, "_cattorch_entry_add")


def test_growing_a_buffer_needs_a_capacity():
    class Grows(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("rows", torch.zeros(0, 2))

        @entrypoint(example=lambda: (torch.ones(2),))
        def push(self, row):
            self.rows = torch.cat([self.rows, row.reshape(1, 2)])
            return row

    with pytest.raises(UnsupportedModelError, match="capacity="):
        marked_program(Grows())


def test_a_capacity_for_a_buffer_that_never_grows_is_an_error():
    class Still(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("rows", torch.zeros(1, 2))

        @entrypoint(example=lambda: (torch.ones(2),), capacity={"rows": 4})
        def read(self, value):
            return self.rows.sum(0) + value

    with pytest.raises(UnsupportedModelError, match="doesn't grow"):
        marked_program(Still())


def test_example_marker_supplies_forward_inputs(tmp_path):
    @example(lambda: (torch.randn(2, 4),))
    class Net(nn.Sequential):
        pass

    torch.manual_seed(1)
    model = Net(nn.Linear(4, 3), nn.ReLU()).eval()
    assert verify(model, None, transpile(model, None, tmp_path / "net")).passed
    with pytest.raises(ValueError, match="no @example"):
        transpile(nn.Linear(2, 2).eval(), None, tmp_path / "plain")


class Gate(nn.Module):
    """Clamps with a branch on a tensor value, which can't be traced."""

    def forward(self, value):
        if float(value.max()) > 1:
            return value.clamp(max=1)
        return value

    @export_forward
    def clamped(self, value):
        return value.clamp(max=1)


def test_export_forward_replaces_forward_wherever_the_class_is_used(tmp_path):
    model = nn.Sequential(nn.Linear(3, 3), Gate(), nn.Linear(3, 2), Gate()).eval()
    value = torch.randn(2, 3) * 4
    assert verify(model, value, transpile(model, value, tmp_path / "gate")).passed


class HandWrittenMoE(nn.Module):
    """Top-2 routing over three experts, plus a shared expert detection can't see."""

    def __init__(self, width=8, experts=3):
        super().__init__()
        self.gate = nn.Linear(width, experts, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(width, 12), nn.SiLU(), nn.Linear(12, width))
            for _ in range(experts)
        )
        self.shared = nn.Linear(width, width)

    def forward(self, x):
        weights = torch.softmax(self.gate(x), dim=-1)
        top, ids = torch.topk(weights, 2, dim=-1)
        top = top / top.sum(-1, keepdim=True)
        routed = torch.zeros_like(x)
        for index, expert in enumerate(self.experts):
            chosen = (ids == index).to(x.dtype)
            routed = routed + expert(x) * (top * chosen).sum(-1, keepdim=True)
        return 0.5 * routed + self.shared(x)


def test_moe_marker_declares_routing_detection_cant_infer(tmp_path):
    marked = moe(router="gate", experts="experts", top_k=2, normalize=True,
                 shared="shared", scale=0.5)(type("Marked", (HandWrittenMoE,), {}))
    torch.manual_seed(2)
    model = marked().eval()
    value = torch.randn(1, 5, 8)
    result = transpile(model, value, tmp_path / "moe")
    assert verify(model, value, result).passed


@pytest.mark.parametrize(("options", "message"), [
    (dict(router="gate", experts="experts", top_k=1, normalize=True, shared="shared", scale=0.5),
     "doesn't reproduce"),
    (dict(router="router", experts="experts", top_k=2), "missing submodule 'router'"),
])
def test_a_wrong_moe_marker_fails_the_export(tmp_path, options, message):
    marked = moe(**options)(type("Marked", (HandWrittenMoE,), {}))
    with pytest.raises(UnsupportedModelError, match=message):
        transpile(marked().eval(), torch.randn(1, 5, 8), tmp_path / "wrong")


def test_generate_marker_picks_the_generation_method(tmp_path):
    class Chat(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nano()

        def forward(self, tokens):
            raise RuntimeError("training entry point")

        @generate
        def logits(self, tokens):
            return self.decoder(tokens)

    model = Chat().eval()
    result = transpile(model, GenerationProgram(max_context=8), tmp_path / "chat")
    with torch.no_grad():
        prefill_and_decode(
            ScratchEmulator(load_sprite(result.path)), model.decoder, [1, 2, 3], [4],
        )


def test_adapter_builds_a_scoped_module_adapter(tmp_path):
    from cattorch import ExportProgram, EntryPoint, Input, ModuleAdapter, Output

    class ThirdParty(nn.Module):
        def forward(self, value):
            return value.numpy().sum()  # not traceable

    @adapter(ThirdParty)
    def replace(module, context):
        return type("Summed", (nn.Module,), {"forward": lambda self, value: value.sum(-1)})()

    assert isinstance(replace, ModuleAdapter)
    program = ExportProgram(entrypoints=(EntryPoint(
        "sum", arguments=(Input("value", torch.ones(2, 3)),), returns=(Output("total"),),
    ),))
    model = nn.Sequential(nn.Linear(3, 3), ThirdParty()).eval()
    transpile(model, program, tmp_path / "adapted", adapters=[replace])


def test_markers_are_inherited_and_can_be_overridden():
    @example(lambda: (torch.zeros(1),))
    class Base(nn.Module):
        pass

    @example(lambda: (torch.zeros(2),))
    class Child(Base):
        pass

    from cattorch.markers import example_inputs

    assert example_inputs(Base())[0].shape == (1,)
    assert example_inputs(Child())[0].shape == (2,)
    assert example_inputs(type("Grandchild", (Child,), {})())[0].shape == (2,)
