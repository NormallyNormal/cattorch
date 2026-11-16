"""Programs and their state: what an entry point changes, and the length checks.

Each test is a model the export once got wrong: exported with wrong numbers,
refused although valid, or refused with a message that blamed the wrong
thing.
"""

from __future__ import annotations

import random

import pytest
import torch
import torch.nn as nn

from cattorch import (
    EntryPoint, ExportProgram, Input, Output, ProgramCall, UnsupportedModelError,
    UnsupportedOperationError, entrypoint, transpile, verify,
)
from cattorch.quantization import QuantizationConfig

from sprite_helpers import load_emulator

THREE = (torch.ones(3),)


def _first_call(result, entry_name, inputs):
    """The outputs of one call of ``entry_name`` on a freshly loaded sprite."""
    emulator = load_emulator(result)
    entry = next(entry for entry in result.entrypoints if entry.name == entry_name)
    for spec, value in zip(entry.inputs, inputs):
        emulator.lists[spec.list_name] = value.flatten().tolist()
    emulator.run_procedure(entry.procedure)
    assert emulator.variables["cattorch status"] == "ok"
    return [
        torch.tensor([float(value) for value in emulator.lists[spec.list_name]])
        for spec in entry.outputs
    ]


def _calls(name, count, width=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [ProgramCall(name, (torch.randn(width, generator=generator),)) for _ in range(count)]


# ── A buffer that is, or becomes, None ──────────────────────────────────────

class LazyStart(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        self.register_buffer("prev", None)

    @entrypoint(example_inputs=THREE)
    def step(self, x):
        out = x * self.w if self.prev is None else x * self.w + self.prev
        self.prev = x.clone()
        return out


class Clears(nn.Module):
    def __init__(self, returns=True):
        super().__init__()
        self.returns = returns
        self.register_buffer("prev", torch.ones(3))

    @entrypoint(example_inputs=THREE)
    def step(self, x):
        return x if self.prev is None else x + self.prev

    @entrypoint(example_inputs=THREE)
    def clear(self, x):
        self.prev = None
        return x * 1 if self.returns else None


def test_a_buffer_that_starts_as_none_and_gets_a_tensor_is_refused(tmp_path):
    with pytest.raises(UnsupportedModelError, match=r"buffer 'prev', which starts as None.*initial tensor value"):
        transpile(LazyStart().eval(), None, tmp_path / "lazy")


@pytest.mark.parametrize("returns", [True, False])
def test_a_buffer_set_to_none_is_refused(tmp_path, returns):
    with pytest.raises(UnsupportedModelError, match=r"@entrypoint clear sets buffer 'prev' to None"):
        transpile(Clears(returns).eval(), None, tmp_path / "clears")


class DoesNothing(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 3)

    @entrypoint(example_inputs=THREE)
    def step(self, x):
        return self.lin(x)

    @entrypoint(example_inputs=THREE)
    def idle(self, x):
        return None


def test_an_entry_point_that_does_nothing_says_so_in_marker_terms(tmp_path):
    with pytest.raises(UnsupportedModelError, match=r"@entrypoint idle returns no tensor and changes no buffer"):
        transpile(DoesNothing().eval(), None, tmp_path / "idle")


# ── Changes outside the buffers ─────────────────────────────────────────────

class InputMemory(nn.Module):
    """Non-buffer state copied from the inputs, read by the next call."""

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.w = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        self.cache: dict = {}
        self.prev = None

    @entrypoint(example_inputs=THREE)
    def step(self, x):
        if self.kind == "dict":
            out = x * self.w + self.cache.get("last", torch.zeros(3))
            self.cache["last"] = x * 1.0
        else:
            out = x * self.w if self.prev is None else x * self.w + self.prev
            self.prev = x.clone()
        return out


@pytest.mark.parametrize("kind", ["dict", "lazy attribute"])
def test_state_copied_from_the_inputs_outside_buffers_is_refused(tmp_path, kind):
    # A second call with the same inputs left the value as it was, so this
    # was taken for a change on the first call only, with wrong numbers.
    with pytest.raises(UnsupportedModelError, match=r"changes self\.(cache|prev), a tensor attribute.*on every call"):
        transpile(InputMemory(kind).eval(), None, tmp_path / "memory")


class Arms(nn.Module):
    """``arm`` sets a Python value once, which ``look`` reads."""

    def __init__(self):
        super().__init__()
        self.scale = 1.0

    @entrypoint(example_inputs=THREE)
    def look(self, x):
        return x * self.scale

    @entrypoint(example_inputs=(torch.ones(1),))
    def arm(self, x):
        self.scale = 2.0
        return x * 1


def test_a_first_call_change_that_another_entry_point_reads_is_refused(tmp_path):
    with pytest.raises(UnsupportedModelError, match=r"@entrypoint arm changes self\.scale on its first call, and later calls of @entrypoint look give other results"):
        transpile(Arms().eval(), None, tmp_path / "arms")


class LateCache(nn.Module):
    """A value set on the first call that changes later calls of the method."""

    def __init__(self):
        super().__init__()
        self.cache = None

    @entrypoint(example_inputs=THREE)
    def step(self, x):
        out = x * (self.cache or 1.0)
        if self.cache is None:
            self.cache = 2.0
        return out


def test_a_first_call_change_that_changes_later_calls_is_refused(tmp_path):
    with pytest.raises(UnsupportedModelError, match=r"changes self\.cache on its first call, and its own later calls give other results"):
        transpile(LateCache().eval(), None, tmp_path / "late")


class OldSpectralNorm(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.inner = nn.utils.spectral_norm(nn.Linear(3, 3))

    @entrypoint(example_inputs=THREE)
    def step(self, x):
        return self.inner(x)


def test_old_style_spectral_norm_exports_in_an_entry_point(tmp_path):
    # It sets the plain attribute ``weight`` on its first forward, to what
    # every later call computes again.
    model = OldSpectralNorm().eval()
    with pytest.warns(UserWarning, match=r"changes self\.inner\.weight on its first call only"):
        result = transpile(model, None, tmp_path / "spectral")
    assert verify(model, None, result, calls=_calls("step", 3)).passed


class Counter(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("count", torch.zeros(()))

    def forward(self, x):
        self.count += 1
        return x * self.count


class TwoNames(nn.Module):
    """One submodule under two names, whose buffer is a state."""

    def __init__(self):
        super().__init__()
        shared = Counter()
        self.s1 = shared
        self.s2 = shared
        self.lin = nn.Linear(3, 3)

    @entrypoint(example_inputs=THREE)
    def tick(self, x):
        return self.s1(x) + self.s2(x)

    @entrypoint(example_inputs=THREE)
    def pure(self, x):
        return self.lin(x)


def test_a_submodule_under_two_names_doesnt_break_the_other_entry_points(tmp_path):
    model = TwoNames().eval()
    result = transpile(model, None, tmp_path / "two_names")
    pure = next(entry for entry in result.entrypoints if entry.name == "pure")
    assert pure.state_inputs == ()
    calls = [*_calls("tick", 2), *_calls("pure", 1), *_calls("tick", 1, seed=1)]
    assert verify(model, None, result, calls=calls).passed


class ListInPlace(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(3))
        self.mems = [torch.zeros(3)]

    def step(self, x):
        self.mems[0].copy_(x)
        return x * self.w + self.mems[0]


def test_an_in_place_update_doesnt_blame_a_parameter_with_the_same_values(tmp_path):
    program = ExportProgram(entrypoints=(EntryPoint(
        "step", method="step", arguments=(Input("x", torch.ones(3)),), returns=(Output("y"),),
    ),))
    with pytest.raises(UnsupportedOperationError) as raised:
        transpile(ListInPlace().eval(), program, tmp_path / "list")
    assert "self.w" not in str(raised.value)
    assert "updates a tensor of the model in place" in str(raised.value)


def test_a_marked_tensor_in_a_list_updated_in_place_is_named(tmp_path):
    class Marked(ListInPlace):
        step = entrypoint(example_inputs=THREE)(ListInPlace.step)

    with pytest.raises(UnsupportedModelError, match=r"changes self\.mems, a tensor attribute"):
        transpile(Marked().eval(), None, tmp_path / "marked_list")


# ── Each entry point traced from the model as given ─────────────────────────

class Flag(nn.Module):
    """An ExportProgram whose ``arm`` changes what ``look`` reads."""

    def __init__(self):
        super().__init__()
        self.scale = 1.0

    def arm(self, x):
        self.scale = 2.0
        return x * 1

    def look(self, x):
        return x * self.scale


FLAG_PROGRAM = ExportProgram(entrypoints=(
    EntryPoint("arm", method="arm", arguments=(Input("x", torch.ones(1)),), returns=(Output("y"),)),
    EntryPoint("look", method="look", arguments=(Input("x", torch.ones(3)),), returns=(Output("y"),)),
))


def test_an_entry_point_is_traced_without_the_changes_of_those_traced_before(tmp_path):
    result = transpile(Flag().eval(), FLAG_PROGRAM, tmp_path / "flag")
    (look,) = _first_call(result, "look", (torch.tensor([1.0, 2.0, 3.0]),))
    torch.testing.assert_close(look, torch.tensor([1.0, 2.0, 3.0]))


def test_verify_by_default_calls_the_entry_points_in_both_orders(tmp_path):
    model = Flag().eval()
    result = transpile(model, FLAG_PROGRAM, tmp_path / "flag")
    report = verify(model, FLAG_PROGRAM, result)
    assert [call.entrypoint for call in report.calls] == ["arm", "look", "look", "arm"]
    # In PyTorch, look after arm doubles; no sprite can, as scale isn't state.
    assert not report.passed
    assert report.calls[2].passed


# ── The length checks ───────────────────────────────────────────────────────

class CountRows(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self.register_buffer("h", torch.zeros(1, 2))

    @entrypoint(example_inputs=(torch.ones(2),), capacity={"h": 8})
    def push(self, x):
        self.h = torch.cat([self.h, x.reshape(1, 2)])
        return self.lin(x)

    @entrypoint(example_inputs=(torch.ones(2),))
    def rows(self, y):
        return torch.tensor([float(self.h.shape[0])]) + 0 * y.sum()


def test_gptq_runs_the_entry_point_length_checks(tmp_path):
    calibration = [ProgramCall("push", (torch.randn(2),)) for _ in range(4)]
    calibration.append(ProgramCall("rows", (torch.randn(2),)))
    with pytest.raises(UnsupportedModelError, match=r"entry point 'rows' with 2 rows in state 'h'"):
        transpile(
            CountRows().eval(), None, tmp_path / "gptq",
            quantization=QuantizationConfig(bits=8, method="gptq", min_calibration_rows=1),
            calibration=calibration,
        )


class LinearWithRows(nn.Module):
    """GPTQ with a length check it passes."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(4, 4)
        self.register_buffer("h", torch.zeros(0, 4))

    @entrypoint(example_inputs=(torch.ones(4),), capacity={"h": 6})
    def push(self, x):
        self.h = torch.cat([self.h, self.lin(x).reshape(1, 4)])
        return self.h.sum(0)


def test_a_gptq_export_that_follows_the_length_passes_its_checks(tmp_path):
    generator = torch.Generator().manual_seed(1)
    calibration = [ProgramCall("push", (torch.randn(4, generator=generator),)) for _ in range(5)]
    transpile(
        LinearWithRows().eval(), None, tmp_path / "gptq_ok",
        quantization=QuantizationConfig(bits=8, method="gptq", min_calibration_rows=1),
        calibration=calibration,
    )


class FlatAfterFour(nn.Module):
    """A length dependence that is constant up to 4 rows."""

    def __init__(self):
        super().__init__()
        self.register_buffer("h", torch.zeros(0, 2))

    @entrypoint(example_inputs=(torch.ones(2),), capacity={"h": 6})
    def push(self, x):
        self.h = torch.cat([self.h, x.reshape(1, 2)])
        return self.h.sum(0) / max(self.h.shape[0], 4)


def test_the_entry_point_growing_a_state_is_checked_near_its_capacity(tmp_path):
    with pytest.raises(UnsupportedModelError, match=r"entry point 'push' with \d+ rows in state 'h'"):
        transpile(FlatAfterFour().eval(), None, tmp_path / "flat")


BASE = 3e4


class Centered(nn.Module):
    """Small results from rows near 3e4, which float32 can't hold exactly."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "h", torch.tensor([[BASE + 0.25, BASE - 0.5], [BASE + 0.75, BASE + 0.125]]),
        )

    @entrypoint(example_inputs=(torch.ones(2),), capacity={"h": 600})
    def push(self, x):
        self.h = torch.cat([self.h, x.reshape(1, 2) + BASE])
        return x * 1

    @entrypoint(example_inputs=(torch.ones(2),))
    def deviation(self, y):
        return self.h[-1] - self.h.mean(0) + y


class CenteredForward(nn.Module):
    def forward(self, h):
        return h[-1] - h.mean(0)


def test_large_values_in_an_append_state_pass_the_length_check(tmp_path):
    model = Centered().eval()
    result = transpile(model, None, tmp_path / "centered")
    rng = random.Random(0)
    calls = [
        ProgramCall("push", (torch.tensor([rng.uniform(-1, 1), rng.uniform(-1, 1)]),))
        for _ in range(5)
    ]
    # PyTorch's own float32 result is off by about 1e-3 here.
    report = verify(
        model, None, result, calls=[*calls, ProgramCall("deviation", (torch.ones(2),))], atol=1e-2,
    )
    assert report.passed


def test_large_values_in_a_dynamic_input_pass_the_length_check(tmp_path):
    rows = torch.tensor([[BASE + 0.25, BASE - 0.5], [BASE + 0.75, BASE + 0.125]])
    transpile(CenteredForward().eval(), (rows,), tmp_path / "centered_plain", dynamic={0: (1, 600)})


class GeluSum(nn.Module):
    def __init__(self, uses_length=False):
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.uses_length = uses_length

    def forward(self, x):
        hidden = self.lin(x)
        out = nn.functional.gelu(hidden).sum(0) + self.norm(hidden).sum(0)
        if self.uses_length:
            out = out + torch.arange(x.shape[0], dtype=x.dtype).sum()
        return out


class GeluRows(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = GeluSum()
        self.register_buffer("h", torch.zeros(0, 4))

    @entrypoint(example_inputs=(torch.ones(4),), capacity={"h": 1024})
    def push(self, x):
        self.h = torch.cat([self.h, x.reshape(1, 4)])
        return x * 1

    @entrypoint(example_inputs=(torch.ones(4),))
    def read(self, y):
        return self.inner(self.h) + y


def test_fast_mode_approximations_growing_with_the_length_pass_the_check(tmp_path):
    example = (torch.randn(8, 4, generator=torch.Generator().manual_seed(0)),)
    transpile(GeluSum().eval(), example, tmp_path / "fast", dynamic={0: (1, 1024)}, optimization="fast")
    transpile(GeluRows().eval(), None, tmp_path / "fast_rows", optimization="fast")


def test_fast_mode_still_catches_a_length_dependence(tmp_path):
    example = (torch.randn(8, 4, generator=torch.Generator().manual_seed(0)),)
    with pytest.raises(UnsupportedModelError, match=r"items along the dynamic dimension"):
        transpile(
            GeluSum(uses_length=True).eval(), example, tmp_path / "fast_length",
            dynamic={0: (1, 64)}, optimization="fast",
        )


# ── Traced with one row ─────────────────────────────────────────────────────

class Delta(nn.Module):
    @entrypoint(example_inputs=(torch.randn(4, 2),), dynamic={"xs": (2, 8)})
    def delta(self, xs):
        return xs[-1] - xs[-2]


class SecondLast(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("h", torch.ones(2, 2))

    @entrypoint(example_inputs=(torch.ones(2),), capacity={"h": 6})
    def push(self, x):
        self.h = torch.cat([self.h, x.reshape(1, 2)])
        return x * 1

    @entrypoint(example_inputs=(torch.ones(2),))
    def read(self, y):
        return self.h[-2] + y


@pytest.mark.parametrize(("model", "what"), [
    (Delta, "input 'xs'"), (SecondLast, "append state 'h'"),
])
def test_a_method_that_cant_run_on_one_row_says_it_is_traced_with_one(tmp_path, model, what):
    with pytest.raises(UnsupportedModelError, match=rf"traced with one row of {what}.*IndexError"):
        transpile(model().eval(), None, tmp_path / "one_row")


# ── Settled while fixing the above ──────────────────────────────────────────

class Columns(nn.Module):
    """Two states that are the interleaved columns of one tensor."""

    def __init__(self):
        super().__init__()
        base = torch.zeros(4, 2)
        self.register_buffer("a", base[:, 0])
        self.register_buffer("b", base[:, 1])

    @entrypoint(example_inputs=(torch.ones(4),))
    def step(self, x):
        self.a.add_(x)
        self.b.mul_(0.5).add_(self.a)
        return self.a + self.b


def test_states_that_interleave_without_sharing_an_element_export(tmp_path):
    # Their byte spans overlap, but a write to one never reaches the other.
    model = Columns().eval()
    result = transpile(model, None, tmp_path / "columns")
    assert verify(model, None, result, calls=_calls("step", 4, width=4)).passed


class ComputedWeight(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(64, 32) * 0.1)
        self.register_buffer("mask", (torch.rand(64, 32) > 0.3).float())

    def a(self, value):
        return nn.functional.linear(value, self.w * self.mask)

    def z(self, value):
        return value * 2


@pytest.mark.parametrize("order", [("a", "z"), ("z", "a")])
def test_a_storage_warning_is_kept_when_the_last_entry_point_stores_no_weight(tmp_path, order):
    torch.manual_seed(0)
    program = ExportProgram(entrypoints=tuple(
        EntryPoint(
            name, method=name, arguments=(Input(f"x{name}", torch.randn(1, 32)),),
            returns=(Output(f"y{name}"),),
        )
        for name in order
    ))
    result = transpile(
        ComputedWeight().eval(), program, tmp_path / "computed",
        quantization=QuantizationConfig(bits=4, min_quantized_values=1024),
    )
    assert any(warning.startswith("weight computed from w") for warning in result.warnings)
