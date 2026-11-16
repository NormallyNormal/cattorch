"""Length arithmetic in marked programs with several variable-length inputs.

Each model is exported, then driven through a long random sequence of calls
and resets that is compared, call by call, with the same model run eagerly
on live buffers. A pattern the sprite can't follow must fail the export
instead.
"""

from __future__ import annotations

import copy
import random

import pytest
import torch
import torch.nn as nn

from cattorch import UnsupportedModelError, entrypoint, marked_program, transpile
from cattorch.program import StateInput

from sprite_helpers import load_emulator


def _drive(model, result, calls, seed=0, steps=40):
    """Run random calls on the sprite and eagerly on a copy, comparing each."""
    emulator = load_emulator(result)
    entries = {entry.name: entry for entry in result.entrypoints}
    initial = copy.deepcopy(model)
    eager = copy.deepcopy(model)
    capacities = {state.name: state.capacity for state in result.states if state.capacity}
    rng = random.Random(seed)
    for _ in range(steps):
        name = rng.choice([*calls, "reset"])
        if name == "reset":
            emulator.run_procedure("cattorch reset")
            eager = copy.deepcopy(initial)
            continue
        inputs = calls[name](rng)
        entry = entries[name]
        for spec, value in zip(entry.inputs, inputs):
            emulator.lists[spec.list_name] = value.flatten().tolist()
        emulator.run_procedure(entry.broadcast)
        status = emulator.variables["cattorch status"]
        before = copy.deepcopy(eager)
        with torch.no_grad():
            returned = getattr(eager, name)(*(value.clone() for value in inputs))
        buffers = dict(eager.named_buffers())
        if any(buffers[state].shape[0] > capacity for state, capacity in capacities.items()):
            eager = before
            assert status.startswith("state full"), status
            continue
        assert status == "ok", status
        returned = returned if isinstance(returned, tuple) else (returned,)
        for spec, expected in zip(entry.outputs, returned):
            actual = torch.tensor([float(value) for value in emulator.lists[spec.list_name]])
            torch.testing.assert_close(
                actual, expected.flatten().float(), atol=1e-4, rtol=1e-4, equal_nan=True,
            )
        for state in result.states:
            actual = torch.tensor([float(value) for value in emulator.lists[state.list_name]])
            torch.testing.assert_close(
                actual, buffers[state.name].flatten().float(), atol=1e-4, rtol=1e-4,
            )


def _rows(width):
    return lambda rng: (torch.tensor([rng.uniform(-2, 2) for _ in range(width)]),)


# Readings of the growing buffer ``a``, which another growing buffer ``b`` sits beside.
READINGS = {
    "mean": lambda a, b: a.mean(0),
    "hand-written mean": lambda a, b: a.sum(0) / a.shape[0],
    "var": lambda a, b: a.var(0),
    "std of b": lambda a, b: b.std(0),
    "length times a value": lambda a, b: a[-1] * a.shape[0] + b.shape[0],
    "reciprocal of the length": lambda a, b: a.sum(0) * (1.0 / b.shape[0]),
    "lengths of both, separately": lambda a, b: a.sum(0) / a.shape[0] + b.sum(0) / b.shape[0],
}


class TwoGrowing(nn.Module):
    def __init__(self, reading, rows_a=1, rows_b=1):
        super().__init__()
        self.reading = reading
        self.register_buffer("a", torch.arange(rows_a * 2, dtype=torch.float32).reshape(rows_a, 2))
        self.register_buffer("b", torch.ones(rows_b, 2) * 0.5)

    @entrypoint(example_inputs=lambda: (torch.ones(2),), capacity={"a": 6})
    def grow_a(self, x):
        self.a = torch.cat([self.a, x.reshape(1, 2)])
        return READINGS[self.reading](self.a, self.b)

    @entrypoint(example_inputs=lambda: (torch.ones(2),), capacity={"b": 5})
    def grow_b(self, y):
        self.b = torch.cat([self.b, y.reshape(1, 2)])
        return y * 1

    @entrypoint(example_inputs=lambda: (torch.ones(2),))
    def look(self, z):
        return READINGS[self.reading](self.a, self.b) + z


@pytest.mark.parametrize("reading", sorted(READINGS))
def test_length_arithmetic_with_two_growing_buffers(tmp_path, reading):
    # A variance of one row is NaN, which Scratch arithmetic turns into 0.
    rows_a, rows_b = (2, 2) if "var" in reading or "std" in reading else (1, 1)
    model = TwoGrowing(reading, rows_a, rows_b).eval()
    result = transpile(model, None, tmp_path / "two")
    _drive(model, result, {"grow_a": _rows(2), "grow_b": _rows(2), "look": _rows(2)}, steps=45)


@pytest.mark.parametrize("rows", [1, 2, 3])
def test_initial_lengths_of_a_growing_buffer(tmp_path, rows):
    model = TwoGrowing("hand-written mean", rows, 4 - rows).eval()
    result = transpile(model, None, tmp_path / "rows")
    _drive(model, result, {"grow_a": _rows(2), "grow_b": _rows(2), "look": _rows(2)}, seed=rows)


class WithRows(nn.Module):
    """A ``dynamic=`` input beside a growing buffer it reads the length of."""

    def __init__(self, rows=1, scale_by="mem"):
        super().__init__()
        self.scale_by = scale_by
        self.register_buffer("mem", torch.ones(rows, 3))

    @entrypoint(example_inputs=lambda: (torch.ones(3),), capacity={"mem": 5})
    def remember(self, x):
        self.mem = torch.cat([self.mem, x.reshape(1, 3)])
        return x * 1

    @entrypoint(example_inputs=lambda: (torch.ones(2, 3),), dynamic={"rows": (1, 4)})
    def batch(self, rows):
        if self.scale_by == "mem":
            return rows.sum(0) * self.mem.shape[0] + self.mem.mean(0)
        return rows.mean(0) - self.mem.sum(0) / self.mem.shape[0] + 1 / rows.shape[0]


@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("scale_by", ["mem", "both"])
def test_length_arithmetic_with_a_dynamic_input_and_a_growing_buffer(tmp_path, rows, scale_by):
    model = WithRows(rows, scale_by).eval()
    result = transpile(model, None, tmp_path / "rows")

    def batch(rng):
        return (torch.randn(rng.randint(1, 4), 3, generator=torch.Generator().manual_seed(rng.randint(0, 99))),)

    _drive(model, result, {"remember": _rows(3), "batch": batch}, seed=rows)


class Rejected(nn.Module):
    def __init__(self, reading, rows=1):
        super().__init__()
        self.reading = reading
        self.register_buffer("a", torch.ones(rows, 2))
        self.register_buffer("b", torch.ones(1, 2))

    @entrypoint(example_inputs=lambda: (torch.ones(2),), capacity={"a": 6, "b": 6})
    def grow(self, x):
        self.a = torch.cat([self.a, x.reshape(1, 2)])
        self.b = torch.cat([self.b, x.reshape(1, 2) * 2])
        return x * 1

    @entrypoint(example_inputs=lambda: (torch.ones(2),))
    def look(self, z):
        a, b = self.a, self.b
        if self.reading == "joint":
            return z / (a.shape[0] + b.shape[0])
        if self.reading == "arange":
            return z + torch.arange(a.shape[0]).float().sum()
        if self.reading == "comparison":
            return z * float(a.shape[0] > 2)
        return z[: 1] * 0 + b[: a.shape[0]].sum()


def test_a_number_from_two_lengths_together_is_rejected(tmp_path):
    with pytest.raises(UnsupportedModelError, match=r"entry point 'look' computes a number from the lengths of 'a', 'b' together"):
        transpile(Rejected("joint").eval(), None, tmp_path / "joint")


@pytest.mark.parametrize("reading", ["arange", "comparison", "slice to another length"])
@pytest.mark.parametrize("rows", [1, 2])
def test_lengths_the_sprite_cant_follow_fail_the_export(tmp_path, reading, rows):
    with pytest.raises(UnsupportedModelError, match="entry point 'look'.*state '[ab]'"):
        transpile(Rejected(reading, rows).eval(), None, tmp_path / "rejected")


class Unrelated(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("total", torch.zeros(2))
        self.register_buffer("seen", torch.zeros(0, 2))
        self.register_buffer("scale", torch.full((2,), 3.0))

    @entrypoint(example_inputs=lambda: (torch.ones(2),))
    def add(self, x):
        self.total += x * self.scale
        return self.total.sum()

    @entrypoint(example_inputs=lambda: (torch.ones(2),), capacity={"seen": 4})
    def remember(self, x):
        self.seen = torch.cat([self.seen, x.reshape(1, 2)])
        return self.seen.sum(0) / self.seen.shape[0]

    @entrypoint(example_inputs=())
    def peek(self):
        return self.total * 1

    @entrypoint(example_inputs=lambda: (torch.ones(2),))
    def double(self, x):
        return x * 2


def test_entry_points_take_only_the_states_they_use(tmp_path):
    _, program = marked_program(Unrelated())
    taken = {
        entry.name: [binding.name for binding in entry.arguments if isinstance(binding, StateInput)]
        for entry in program.entrypoints
    }
    assert taken == {"add": ["total"], "remember": ["seen"], "peek": ["total"], "double": []}
    model = Unrelated().eval()
    result = transpile(model, None, tmp_path / "unrelated")
    _drive(
        model, result,
        {"add": _rows(2), "remember": _rows(2), "peek": lambda rng: (), "double": _rows(2)},
    )
    # An entry point that doesn't use a state doesn't check its list either.
    emulator = load_emulator(result)
    emulator.lists["cattorch state seen"] = [1.0]
    emulator.lists["cattorch double x"] = [1.0, 2.0]
    emulator.run_procedure("cattorch double")
    assert emulator.variables["cattorch status"] == "ok"
    emulator.lists["cattorch remember x"] = [1.0, 2.0]
    emulator.run_procedure("cattorch remember")
    assert emulator.variables["cattorch status"] == "wrong number of items: unrelated state seen"


class BranchOnShape(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("seen", torch.zeros(0, 2))
        self.register_buffer("other", torch.zeros(2))

    @entrypoint(example_inputs=lambda: (torch.ones(2),), capacity={"seen": 4})
    def remember(self, x):
        self.seen = torch.cat([self.seen, x.reshape(1, 2)])
        return x * 1

    @entrypoint(example_inputs=lambda: (torch.ones(2),))
    def bump(self, x):
        self.other += x
        return x

    @entrypoint(example_inputs=lambda: (torch.ones(2),))
    def look(self, x):
        # Reads ``other`` only once ``seen`` has rows, which it has when traced.
        if self.seen.shape[0]:
            return x + self.other
        return x * 1


def test_a_state_read_only_when_traced_fails_the_export(tmp_path):
    with pytest.raises(UnsupportedModelError, match="look read buffer 'other' when cattorch traced it"):
        transpile(BranchOnShape().eval(), None, tmp_path / "branch")
