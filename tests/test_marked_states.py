"""Which tensors of a marked model become state, and how verify checks them."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from cattorch import (
    ProgramCall, UnsupportedModelError, UnsupportedOperationError, entrypoint, transpile, verify,
)
from cattorch import markers

from sprite_helpers import load_emulator

EXAMPLE = lambda: (torch.ones(3),)  # noqa: E731


class Views(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(2, 3))
        self.register_buffer("k", self.cache[0])
        self.register_buffer("v", self.cache[1])

    @entrypoint(example_inputs=EXAMPLE)
    def write(self, x):
        self.k.mul_(0.5).add_(x)
        self.v.add_(1.0)
        return self.cache.sum(0)


class Overlapping(nn.Module):
    """Two buffers sharing memory, neither of them registered whole."""

    def __init__(self):
        super().__init__()
        memory = torch.zeros(4)
        self.register_buffer("low", memory[:3])
        self.register_buffer("high", memory[1:])

    @entrypoint(example_inputs=EXAMPLE)
    def write(self, x):
        self.low += x
        return self.high * 1


class ReadView(nn.Module):
    """A buffer that is a view of a state, only read."""

    def __init__(self):
        super().__init__()
        self.register_buffer("total", torch.zeros(2, 3))
        self.register_buffer("first", self.total[0])

    @entrypoint(example_inputs=EXAMPLE)
    def add(self, x):
        self.total += x
        return self.first * 1


@pytest.mark.parametrize(("model", "inner", "outer"), [
    (Views, "k", "cache"), (Overlapping, "high", "low"), (ReadView, "first", "total"),
])
def test_buffers_sharing_memory_are_rejected(tmp_path, model, inner, outer):
    with pytest.raises(UnsupportedModelError, match=rf"'{inner}' is a view of '{outer}'.*Register only '{outer}'"):
        transpile(model().eval(), None, tmp_path / "views")


class Sub(nn.Module):
    def __init__(self, memory, in_place):
        super().__init__()
        self.in_place = in_place
        self.register_buffer("mem", memory)

    def forward(self, x):
        if self.in_place:
            self.mem.mul_(0.5).add_(x)
        else:
            self.mem = self.mem * 0.5 + x
        return self.mem * 1


class Shared(nn.Module):
    """One tensor registered by the model and by a submodule."""

    def __init__(self, in_place):
        super().__init__()
        memory = torch.zeros(3)
        self.register_buffer("mem", memory)
        self.sub = Sub(memory, in_place)

    @entrypoint(example_inputs=EXAMPLE)
    def step(self, x):
        return self.sub(x) + self.mem


def test_a_buffer_reassigned_under_one_of_its_names_is_rejected(tmp_path):
    with pytest.raises(UnsupportedModelError, match=r"reassigns buffer 'mem' under only some of the names"):
        transpile(Shared(in_place=False).eval(), None, tmp_path / "shared")


def test_a_buffer_updated_in_place_under_one_of_its_names_is_one_state(tmp_path):
    model = Shared(in_place=True).eval()
    result = transpile(model, None, tmp_path / "shared")
    assert [state.name for state in result.states] == ["mem"]
    calls = [ProgramCall("step", (torch.tensor([1.0, -2.0, 0.5]) * index,)) for index in range(1, 5)]
    assert verify(model, None, result, calls=calls).passed
    emulator = load_emulator(result)
    eager = copy.deepcopy(model)
    for call in calls:
        emulator.lists["cattorch step x"] = call.inputs[0].tolist()
        emulator.run_procedure("cattorch step")
        actual = [float(value) for value in emulator.lists["cattorch step output"]]
        assert actual == pytest.approx(eager.step(call.inputs[0]).tolist())


class Tally(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("total", torch.zeros(3))
        self.register_buffer("last", torch.zeros(3))

    @entrypoint(example_inputs=EXAMPLE)
    def add(self, x):
        self.total += x
        return self.total * 2

    @entrypoint(example_inputs=EXAMPLE)
    def remember(self, x):
        self.last = self.last * 0.5 + x
        return self.last + self.total


def test_marked_verify_runs_the_users_own_methods(tmp_path, monkeypatch):
    model = Tally().eval()
    calls = [
        ProgramCall("add", (torch.tensor([1.0, 2.0, 3.0]),)),
        ProgramCall("remember", (torch.tensor([0.5, 0.0, -1.0]),)),
        ProgramCall("add", (torch.tensor([-1.0, 1.0, 0.0]),)),
        ProgramCall("remember", (torch.tensor([2.0, 2.0, 2.0]),)),
    ]
    result = transpile(model, None, tmp_path / "tally")
    assert verify(model, None, result, calls=calls).passed

    # A generated method that drops its state updates gives a wrong sprite;
    # verify, whose reference doesn't go through it, catches that.
    original = markers._entry_method

    def forgetful(method_name, inputs, states, changed, aliases, unused):
        return original(method_name, inputs, states, set(), aliases, unused)

    def no_updates(model):
        marked = original_program(model)
        entries = tuple(
            type(entry)(entry.name, entry.method, entry.arguments, tuple(
                binding for binding in entry.returns if type(binding).__name__ == "Output"
            ))
            for entry in marked.program.entrypoints
        )
        program = type(marked.program)(entrypoints=entries, states=marked.program.states)
        return markers.MarkedProgram(marked.model, program)

    original_program = markers.marked_program
    monkeypatch.setattr(markers, "_entry_method", forgetful)
    monkeypatch.setattr(markers, "marked_program", no_updates)
    broken = transpile(model, None, tmp_path / "broken")
    report = verify(model, None, broken, calls=calls)
    assert not report.passed
    assert not report.calls[1].outputs[0].passed or not report.calls[2].outputs[0].passed


class Changes(nn.Module):
    """An entry point changing something other than a buffer."""

    def __init__(self, change):
        super().__init__()
        self.change = change
        self.total = torch.zeros(3)
        self.weight = nn.Parameter(torch.ones(3))
        self.inner = nn.Module()
        self.inner.total = torch.zeros(3)
        self.tensors = [torch.zeros(3)]
        self.table = {"sum": torch.zeros(3)}
        self.count = 0
        self.history: list[float] = []
        self.cache = None

    @entrypoint(example_inputs=EXAMPLE)
    def add(self, x):
        change = self.change
        if change == "attribute":
            self.total = self.total + x
        elif change == "attribute in place":
            self.total += x
        elif change == "parameter":
            self.weight.add_(x)
        elif change == "submodule attribute":
            self.inner.total = self.inner.total + x
        elif change == "list entry":
            self.tensors[0].add_(x)
        elif change == "dict entry":
            self.table["sum"] = self.table["sum"] + x
        elif change == "int":
            self.count += 1
        elif change == "list":
            self.history.append(1.0)
        elif change == "cache":
            if self.cache is None:
                self.cache = 2.0
        return x * self.weight * (self.count + 1) * (self.cache or 1.0)


@pytest.mark.parametrize(("change", "message"), [
    ("attribute", r"changes self.total, a tensor attribute that isn't a buffer.*register_buffer\('total'"),
    ("attribute in place", r"changes self.total, a tensor attribute"),
    ("parameter", r"changes the parameter self.weight.*register_buffer\('weight'"),
    ("submodule attribute", r"changes self.inner.total, a tensor attribute"),
    ("list entry", r"changes self.tensors"),
    ("dict entry", r"changes self.table"),
    ("int", r"changes self.count on every call \(from 0 to 1 on the first\).*tensor buffer"),
    ("list", r"changes self.history on every call"),
])
def test_changes_outside_buffers_are_rejected(tmp_path, change, message):
    with pytest.raises(UnsupportedModelError, match=message):
        transpile(Changes(change).eval(), None, tmp_path / "changes")


def test_a_value_set_on_the_first_call_only_warns(tmp_path):
    with pytest.warns(UserWarning, match=r"changes self.cache on its first call only"):
        transpile(Changes("cache").eval(), None, tmp_path / "cache")


def test_the_in_place_message_in_an_entry_point_says_how_to_make_state(tmp_path):
    from cattorch import EntryPoint, ExportProgram, Input, Output

    class Stepper(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("total", torch.zeros(3))

        def step(self, x):
            self.total.add_(x)
            return x * 1

    program = ExportProgram(entrypoints=(
        EntryPoint("step", method="step", arguments=(Input("x", torch.ones(3)),), returns=(Output("y"),)),
    ))
    with pytest.raises(UnsupportedOperationError, match=r"updates the buffer self.total in place, but it isn.t one of its states.*StateInput"):
        transpile(Stepper().eval(), program, tmp_path / "stepper")


class Writes(nn.Module):
    """A state written in place by position, slice, mask and index."""

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.register_buffer("buffer", torch.zeros(4, 3))
        self.register_buffer("position", torch.tensor(0))

    @entrypoint(example_inputs=EXAMPLE)
    def write(self, x):
        kind = self.kind
        if kind == "ring":
            self.buffer[self.position.reshape(1)] = x.unsqueeze(0)
            self.position.add_(1).remainder_(4)
        elif kind == "shift":
            self.buffer[1:] = self.buffer[:-1].clone()
            self.buffer[0] = x
        elif kind == "mask":
            self.buffer[self.buffer > 0.5] = 0.0
            self.buffer[2] = x
        elif kind == "column":
            self.buffer[:, 2].mul_(0.5)
            self.buffer[1].add_(x)
        elif kind == "index tensor":
            self.buffer[torch.tensor([1, 3])] = x
        elif kind == "0-d index":
            self.buffer[self.position] = x
            self.position.add_(1).remainder_(4)
        elif kind == "transposed":
            self.buffer.t()[0] = 1.0
            self.buffer[1] += x[0]
        elif kind == "view read after":
            # The view sees the update, as in PyTorch.
            row = self.buffer[0]
            self.buffer[0] = x
            return row * 1 + self.buffer.sum(0)
        elif kind == "strided view read after":
            row = self.buffer.as_strided((3,), (1,))
            self.buffer[0] = x
            return row * 1
        return self.buffer.sum(0)


@pytest.mark.parametrize(
    "kind", ["ring", "shift", "mask", "column", "index tensor", "view read after"],
)
def test_states_written_in_place(tmp_path, kind):
    model = Writes(kind).eval()
    result = transpile(model, None, tmp_path / "writes")
    generator = torch.Generator().manual_seed(0)
    calls = [ProgramCall("write", (torch.rand(3, generator=generator),)) for _ in range(7)]
    assert verify(model, None, result, calls=calls).passed


@pytest.mark.parametrize(("kind", "error", "message"), [
    ("0-d index", UnsupportedModelError, r"indexing with a 0-d integer tensor.*i.reshape\(1\)"),
    ("transposed", UnsupportedOperationError, r"in-place update through a view.*out of place"),
    ("strided view read after", UnsupportedOperationError, r"a view read again after the update"),
])
def test_state_writes_that_fail_say_what_to_write(tmp_path, kind, error, message):
    with pytest.raises(error, match=message):
        transpile(Writes(kind).eval(), None, tmp_path / "writes")


class TwoCapacities(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("seen", torch.zeros(0, 3))

    @entrypoint(example_inputs=EXAMPLE, capacity={"seen": 4})
    def remember(self, x):
        self.seen = torch.cat([self.seen, x.reshape(1, 3)])
        return x * 1

    @entrypoint(example_inputs=EXAMPLE, capacity={"seen": 6})
    def remember_twice(self, x):
        self.seen = torch.cat([self.seen, x.reshape(1, 3), x.reshape(1, 3)])
        return x * 1


def test_a_buffer_given_two_capacities_is_rejected(tmp_path):
    with pytest.raises(UnsupportedModelError, match=r"different capacities \(4 and 6 in @entrypoint remember_twice\)"):
        transpile(TwoCapacities().eval(), None, tmp_path / "capacities")
