"""Experimental FX program, state, adapter, and analysis coverage."""

from __future__ import annotations

import json
import zipfile

import pytest
import torch
import torch.nn as nn

from cattorch import (
    BoundedDimension, CodegenConfig, ProgramResult, QuantizationConfig,
    transpile, verify,
)
from cattorch import (
    EntryPoint, ExportProgram, Input, ModuleAdapter, Output, ProgramCall, State,
    StateInput, StateUpdate,
)
from cattorch.experimental import ExpertFamily, SparseMoE, analyze
from cattorch.util.scratch.emulator import ScratchEmulator


def _emulator(result: ProgramResult) -> ScratchEmulator:
    with zipfile.ZipFile(result.path) as archive:
        return ScratchEmulator(json.loads(archive.read("sprite.json")))


def test_program_exports_multiple_named_outputs(tmp_path):
    class Model(nn.Module):
        def run(self, value):
            return value + 1, value * 2

    example = torch.tensor([1.0, 2.0])
    program = ExportProgram(entrypoints=(EntryPoint(
        "run", method="run",
        arguments=(Input("value", example),),
        returns=(Output("plus one"), Output("doubled")),
    ),))
    result = transpile(
        Model().eval(), program, tmp_path / "program",
        codegen=CodegenConfig(compact_internal_names=True),
    )
    assert isinstance(result, ProgramResult)
    assert result.entrypoints[0].procedure == "cattorch run"
    assert tuple(value.list_name for value in result.entrypoints[0].outputs) == (
        "cattorch run plus one", "cattorch run doubled",
    )
    emulator = _emulator(result)
    emulator.lists["cattorch run value"] = [3.0, 4.0]
    emulator.run_procedure("cattorch run")
    assert emulator.lists["cattorch run plus one"] == [4.0, 5.0]
    assert emulator.lists["cattorch run doubled"] == [6.0, 8.0]
    assert emulator.variables["cattorch status"] == "ok"


def test_program_replace_state_is_transactional_and_resettable(tmp_path):
    class Counter(nn.Module):
        def step(self, value, state):
            updated = state + value
            return updated, updated

    initial = torch.tensor([1.0, 2.0])
    program = ExportProgram(
        states=(State("counter", initial),),
        entrypoints=(EntryPoint(
            "step", method="step",
            arguments=(Input("delta", torch.ones(2)), StateInput("counter")),
            returns=(Output("value"), StateUpdate("counter")),
        ),),
    )
    result = transpile(Counter().eval(), program, tmp_path / "state")
    emulator = _emulator(result)
    emulator.lists["cattorch step delta"] = [3.0, 4.0]
    emulator.run_procedure("cattorch step")
    assert emulator.lists["cattorch step value"] == [4.0, 6.0]
    assert emulator.lists["cattorch state counter"] == [4.0, 6.0]
    emulator.lists["cattorch step delta"] = [1.0]  # invalid: no partial mutation
    emulator.run_procedure("cattorch step")
    assert emulator.lists["cattorch step value"] == [4.0, 6.0]
    assert emulator.lists["cattorch state counter"] == [4.0, 6.0]
    assert str(emulator.variables["cattorch status"]).startswith("invalid length")
    emulator.run_procedure("cattorch reset")
    assert emulator.lists["cattorch state counter"] == [1.0, 2.0]
    comparison = verify(
        Counter().eval(), program, result,
        calls=(
            ProgramCall("step", (torch.tensor([3.0, 4.0]),)),
        ),
    )
    assert comparison.passed, comparison


def test_scoped_public_adapter_uses_module_path_and_does_not_mutate_source(tmp_path):
    class UnsupportedNeg(nn.Module):
        def forward(self, value):
            return torch.square(value)  # adapter replaces this behavior

    class Replacement(nn.Module):
        def forward(self, value):
            return -value

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = UnsupportedNeg()

        def forward(self, value):
            return self.inner(value)

    paths = []
    adapter = ModuleAdapter(
        UnsupportedNeg,
        lambda _module, context: paths.append(context.module_path) or Replacement(),
    )
    example = torch.tensor([2.0])
    program = ExportProgram(entrypoints=(EntryPoint(
        "forward", arguments=(Input("value", example),), returns=(Output("value"),),
    ),))
    model = Model().eval()
    result = transpile(model, program, tmp_path / "adapter", adapters=(adapter,))
    emulator = _emulator(result)
    emulator.lists["cattorch forward value"] = [2.0]
    emulator.run_procedure("cattorch forward")
    assert emulator.lists["cattorch forward value"] == [-2.0]
    assert paths and paths[0].endswith("inner")
    assert model(torch.tensor([2.0])).item() == 4.0


def test_analysis_reports_operations_and_interface():
    model = nn.Sequential(nn.Linear(3, 2), nn.ReLU()).eval()
    program = ExportProgram(entrypoints=(EntryPoint(
        "forward", arguments=(Input("value", torch.ones(1, 3)),),
        returns=(Output("result"),),
    ),))
    report = analyze(model, program)
    operations = dict(report.entrypoints[0].operations)
    assert operations["aten.linear.default"] == 1
    assert operations["aten.relu.default"] == 1
    assert not report.entrypoints[0].unsupported_operations
    assert report.unique_parameters == 8
    assert report.estimated_json_bytes > report.estimated_payload_bytes


def test_analysis_reports_unsupported_operations_inside_expert_templates():
    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(1))

        def forward(self, value):
            return torch.sin(value * self.scale)

    model = SparseMoE(
        nn.Linear(2, 2),
        ExpertFamily.from_modules(
            [Expert(), Expert()], example_input=torch.ones(1, 2),
        ),
    ).eval()
    program = ExportProgram(entrypoints=(EntryPoint(
        "forward", arguments=(Input("value", torch.ones(1, 2)),),
        returns=(Output("value"),),
    ),))
    report = analyze(model, program)
    unsupported = report.entrypoints[0].unsupported_operations
    assert any("aten.sin.default" in operation for operation in unsupported)


def test_program_merges_multiple_entrypoints_and_shared_state(tmp_path):
    class Model(nn.Module):
        def add(self, value, state):
            return value + state

        def multiply(self, value, state):
            return value * state

    state = State("factor", torch.tensor([2.0]))
    program = ExportProgram(
        states=(state,),
        entrypoints=(
            EntryPoint(
                "add", method="add",
                arguments=(Input("value", torch.ones(1)), StateInput("factor")),
                returns=(Output("result"),),
            ),
            EntryPoint(
                "multiply", method="multiply",
                arguments=(Input("value", torch.ones(1)), StateInput("factor")),
                returns=(Output("result"),),
            ),
        ),
    )
    result = transpile(Model().eval(), program, tmp_path / "multiple")
    emulator = _emulator(result)
    emulator.lists["cattorch add value"] = [3.0]
    emulator.run_procedure("cattorch add")
    assert emulator.lists["cattorch add result"] == [5.0]
    emulator.lists["cattorch multiply value"] = [3.0]
    emulator.run_procedure("cattorch multiply")
    assert emulator.lists["cattorch multiply result"] == [6.0]
    assert {value.procedure for value in result.entrypoints} == {
        "cattorch add", "cattorch multiply",
    }


def test_append_state_capacity_failure_does_not_partially_commit(tmp_path):
    class Writer(nn.Module):
        def write(self, value):
            return value.reshape(1, 2)

    program = ExportProgram(
        states=(State("rows", torch.empty(0, 2), mode="append", capacity=2),),
        entrypoints=(EntryPoint(
            "write", method="write",
            arguments=(Input("value", torch.ones(2)),),
            returns=(StateUpdate("rows"),),
        ),),
    )
    result = transpile(Writer().eval(), program, tmp_path / "append")
    emulator = _emulator(result)
    emulator.lists["cattorch write value"] = [1.0, 2.0]
    emulator.run_procedure("cattorch write")
    emulator.lists["cattorch write value"] = [3.0, 4.0]
    emulator.run_procedure("cattorch write")
    assert emulator.lists["cattorch state rows"] == [1.0, 2.0, 3.0, 4.0]
    emulator.lists["cattorch write value"] = [5.0, 6.0]
    emulator.run_procedure("cattorch write")
    assert emulator.lists["cattorch state rows"] == [1.0, 2.0, 3.0, 4.0]
    assert str(emulator.variables["cattorch status"]).startswith("state capacity")


def test_append_state_is_a_bounded_runtime_extent_through_linear(tmp_path):
    class Reader(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 3)

        def read(self, rows):
            return torch.relu(self.linear(rows))

    torch.manual_seed(7)
    model = Reader().eval()
    program = ExportProgram(
        states=(State("rows", torch.empty(0, 2), mode="append", capacity=5),),
        entrypoints=(EntryPoint(
            "read", method="read", arguments=(StateInput("rows"),),
            returns=(Output("values"),),
        ),),
    )
    result = transpile(model, program, tmp_path / "dynamic-linear")
    spec = result.entrypoints[0].outputs[0]
    assert spec.shape == (BoundedDimension("rows", 0, 5), 3)
    assert spec.numel is None
    assert spec.max_numel == 15

    rows = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    emulator = _emulator(result)
    emulator.lists["cattorch state rows"] = rows.flatten().tolist()
    emulator.run_procedure("cattorch read")
    torch.testing.assert_close(
        torch.tensor(emulator.lists["cattorch read values"]),
        model.read(rows).flatten(),
        atol=2e-4,
        rtol=0,
    )


def test_runtime_extent_can_move_through_transpose(tmp_path):
    class Reader(nn.Module):
        def read(self, rows):
            return rows.transpose(0, 1)

    program = ExportProgram(
        states=(State("rows", torch.empty(0, 2, 3), mode="append", capacity=4),),
        entrypoints=(EntryPoint(
            "read", method="read", arguments=(StateInput("rows"),),
            returns=(Output("values"),),
        ),),
    )
    result = transpile(Reader().eval(), program, tmp_path / "dynamic-transpose")
    assert result.entrypoints[0].outputs[0].shape == (
        2, BoundedDimension("rows", 0, 4), 3,
    )
    rows = torch.arange(18.0).reshape(3, 2, 3)
    emulator = _emulator(result)
    emulator.lists["cattorch state rows"] = rows.flatten().tolist()
    emulator.run_procedure("cattorch read")
    assert emulator.lists["cattorch read values"] == rows.transpose(0, 1).flatten().tolist()


@pytest.mark.parametrize("optimization", ["exact", "fast"])
@pytest.mark.parametrize("moved_axis", [False, True])
def test_append_state_softmax_uses_runtime_rows(tmp_path, optimization, moved_axis):
    class Reader(nn.Module):
        def read(self, rows):
            value = rows.transpose(0, 1) if moved_axis else rows
            return torch.softmax(value, dim=-1)

    model = Reader().eval()
    program = ExportProgram(
        states=(State("rows", torch.empty(0, 2, 3), mode="append", capacity=4),),
        entrypoints=(EntryPoint(
            "read", method="read", arguments=(StateInput("rows"),),
            returns=(Output("values"),),
        ),),
    )
    artifact = transpile(model, program, tmp_path / "softmax", optimization=optimization)
    spec = artifact.entrypoints[0].outputs[0]
    runtime = BoundedDimension("rows", 0, 4)
    assert spec.shape == ((2, runtime, 3) if moved_axis else (runtime, 2, 3))
    emulator = _emulator(artifact)
    for count in (0, 1, 3, 4, 0):
        rows = torch.arange(count * 6, dtype=torch.float32).reshape(count, 2, 3) / 3
        if optimization == "exact":
            rows += 1000  # Stable softmax must handle large logits.
        emulator.lists["cattorch state rows"] = rows.flatten().tolist()
        emulator.run_procedure("cattorch read")
        assert emulator.variables["cattorch status"] == "ok"
        torch.testing.assert_close(
            torch.tensor(emulator.lists["cattorch read values"]),
            model.read(rows).flatten(), atol=2e-4, rtol=1e-5,
        )
        assert emulator.lists["cattorch state rows"] == rows.flatten().tolist()


def test_multi_entrypoint_quantization_is_owned_once(tmp_path):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 16, bias=False)

        def first(self, value):
            return self.linear(value)

        def second(self, value):
            return torch.relu(self.linear(value))

    model = Model().eval()
    example = torch.randn(8, 16)
    program = ExportProgram(entrypoints=(
        EntryPoint(
            "first", method="first", arguments=(Input("value", example),),
            returns=(Output("result"),),
        ),
        EntryPoint(
            "second", method="second", arguments=(Input("value", example),),
            returns=(Output("result"),),
        ),
    ))
    result = transpile(
        model,
        program,
        tmp_path / "program-gptq",
        quantization=QuantizationConfig(
            bits=8, method="gptq", min_quantized_values=1,
            min_calibration_rows=1,
        ),
        calibration_calls=(
            ProgramCall("first", (example,)),
            ProgramCall("second", (example,)),
        ),
    )
    assert result.quantization is not None
    assert result.quantization.calibration_batches == 2
    assert result.quantization.gptq_values == model.linear.weight.numel()


def test_shared_input_output_name_requires_matching_shape(tmp_path):
    from cattorch.errors import UnsupportedModelError

    program = ExportProgram(entrypoints=(EntryPoint(
        "forward", arguments=(Input("value", torch.ones(1, 3)),),
        returns=(Output("value"),),
    ),))
    model = nn.Linear(3, 2).eval()
    with pytest.raises(UnsupportedModelError, match="shares its list with an input"):
        transpile(model, program, tmp_path / "shared-name")


@torch.no_grad()
def test_symmetric_program_verify_matches_moe_bias_bank_storage(tmp_path):
    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Linear(4, 8)
            self.down = nn.Linear(8, 4)

        def forward(self, value):
            return self.down(torch.relu(self.up(value)))

    torch.manual_seed(17)
    model = SparseMoE(
        nn.Linear(4, 3),
        ExpertFamily.from_modules(
            [Expert(), Expert(), Expert()], example_input=torch.ones(1, 4),
        ),
        top_k=2,
    ).eval()
    example = torch.randn(2, 4)
    program = ExportProgram(entrypoints=(EntryPoint(
        "forward", arguments=(Input("value", example),), returns=(Output("result"),),
    ),))
    config = QuantizationConfig(bits=8, min_quantized_values=1)
    result = transpile(model, program, tmp_path / "moe-quantized", quantization=config)
    comparison = verify(model, program, result, quantization=config, atol=2e-5, rtol=0)
    assert comparison.passed, comparison


def test_analysis_counts_repeated_expert_errors_once():
    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(1))

        def forward(self, value):
            return torch.sin(value * self.scale)

    model = SparseMoE(
        nn.Linear(2, 2),
        ExpertFamily.from_modules(
            [Expert(), Expert()], example_input=torch.ones(1, 2),
        ),
    ).eval()
    entrypoints = tuple(
        EntryPoint(
            name, arguments=(Input("value", torch.ones(1, 2)),),
            returns=(Output("result"),),
        )
        for name in ("first", "second")
    )
    report = analyze(model, ExportProgram(entrypoints=entrypoints))
    unique = {
        operation for entry in report.entrypoints
        for operation in entry.unsupported_operations
    }
    assert report.warnings == (
        f"{len(unique)} captured operation contracts are unsupported",
    )


@torch.no_grad()
def test_symmetric_program_verify_handles_right_hand_matmul_weights(tmp_path):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(64, 48))

        def forward(self, value):
            return value @ self.weight

    torch.manual_seed(8)
    model = Model().eval()
    example = torch.randn(2, 64)
    program = ExportProgram(entrypoints=(EntryPoint(
        "forward", arguments=(Input("value", example),), returns=(Output("result"),),
    ),))
    config = QuantizationConfig(bits=4, min_quantized_values=1)
    result = transpile(model, program, tmp_path / "rhs-program", quantization=config)
    comparison = verify(model, program, result, quantization=config, atol=1e-4, rtol=1e-5)
    assert comparison.passed, comparison


class _Accumulator(nn.Module):
    def step(self, delta, total):
        updated = total + delta
        return updated * 2, updated


def _accumulator_program(initial: torch.Tensor) -> ExportProgram:
    return ExportProgram(
        states=(State("total", initial),),
        entrypoints=(EntryPoint(
            "step", method="step",
            arguments=(Input("delta", torch.ones_like(initial)), StateInput("total")),
            returns=(Output("value"), StateUpdate("total")),
        ),),
    )


def _stored_lists(result: ProgramResult) -> dict[str, list]:
    with zipfile.ZipFile(result.path) as archive:
        sprite = json.loads(archive.read("sprite.json"))
    return {name: values for name, values in sprite["lists"].values()}


def _floats(values) -> list[float]:
    return [float(value) for value in values]


def test_constant_initial_state_is_filled_instead_of_stored(tmp_path):
    initial = torch.zeros(3)
    program = _accumulator_program(initial)
    result = transpile(_Accumulator().eval(), program, tmp_path / "zeros")
    stored = _stored_lists(result)
    assert stored["cattorch state total"] == []
    assert not any(name.startswith("_cattorch initial state") for name in stored)

    emulator = _emulator(result)
    emulator.run_procedure("cattorch init")
    assert _floats(emulator.lists["cattorch state total"]) == [0.0, 0.0, 0.0]
    emulator.lists["cattorch step delta"] = [1.0, 2.0, 3.0]
    emulator.run_procedure("cattorch step")
    assert emulator.variables["cattorch status"] == "ok"
    assert _floats(emulator.lists["cattorch state total"]) == [1.0, 2.0, 3.0]
    emulator.run_procedure("cattorch init")  # init doesn't reset live state
    assert _floats(emulator.lists["cattorch state total"]) == [1.0, 2.0, 3.0]
    emulator.run_procedure("cattorch reset")
    assert emulator.variables["cattorch status"] == "ok"
    assert _floats(emulator.lists["cattorch state total"]) == [0.0, 0.0, 0.0]

    # A freshly loaded project works without init, as when the state was stored.
    calls = tuple(ProgramCall("step", (torch.tensor([1.0, 2.0, 3.0]),)) for _ in range(2))
    comparison = verify(_Accumulator().eval(), program, result, calls=calls)
    assert comparison.passed, comparison


def test_first_call_keeps_a_constant_state_written_before_it(tmp_path):
    result = transpile(_Accumulator().eval(), _accumulator_program(torch.zeros(2)), tmp_path / "write")
    emulator = _emulator(result)
    emulator.lists["cattorch state total"] = [5.0, 6.0]
    emulator.lists["cattorch step delta"] = [1.0, 1.0]
    emulator.run_procedure("cattorch step")
    assert _floats(emulator.lists["cattorch state total"]) == [6.0, 7.0]


def test_non_constant_initial_state_is_still_stored(tmp_path):
    initial = torch.tensor([1.0, 2.0, 3.0])
    result = transpile(_Accumulator().eval(), _accumulator_program(initial), tmp_path / "stored")
    stored = _stored_lists(result)
    assert stored["cattorch state total"] == [1.0, 2.0, 3.0]
    assert stored["_cattorch initial state total"] == [1.0, 2.0, 3.0]
    emulator = _emulator(result)
    emulator.lists["cattorch step delta"] = [1.0, 1.0, 1.0]
    emulator.run_procedure("cattorch step")
    assert emulator.lists["cattorch state total"] == [2.0, 3.0, 4.0]
    emulator.run_procedure("cattorch reset")
    assert emulator.lists["cattorch state total"] == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("initial", [
    torch.full((2, 2), 2.5),
    torch.full((3,), -1, dtype=torch.int64),
])
def test_constant_non_zero_initial_state(tmp_path, initial):
    program = _accumulator_program(initial)
    result = transpile(_Accumulator().eval(), program, tmp_path / "fill")
    assert _stored_lists(result)["cattorch state total"] == []
    emulator = _emulator(result)
    emulator.run_procedure("cattorch reset")
    expected = _floats(initial.flatten().tolist())
    assert _floats(emulator.lists["cattorch state total"]) == expected
    comparison = verify(
        _Accumulator().eval(), program, result,
        calls=(ProgramCall("step", (torch.ones_like(initial),)),),
    )
    assert comparison.passed, comparison


def test_mixed_sign_zero_initial_state_is_stored(tmp_path):
    initial = torch.tensor([0.0, -0.0])
    result = transpile(_Accumulator().eval(), _accumulator_program(initial), tmp_path / "signed")
    assert "_cattorch initial state total" in _stored_lists(result)


def test_sharded_constant_initial_state(tmp_path, small_list_limit):
    from cattorch.util.scratch.interface import read_logical_list, write_logical_list

    limit = small_list_limit(100)
    size = 2 * limit + limit // 2
    program = _accumulator_program(torch.zeros(size))
    result = transpile(_Accumulator().eval(), program, tmp_path / "sharded")
    assert "cattorch state total" in result.sharded_lists
    stored = _stored_lists(result)
    shards = ("cattorch state total", "cattorch state total shard 2", "cattorch state total shard 3")
    assert all(stored[name] == [] for name in shards)
    assert not any(name.startswith("_cattorch initial state") for name in stored)

    emulator = _emulator(result)
    emulator.run_procedure("cattorch init")
    assert [len(emulator.lists[name]) for name in shards] == [limit, limit, limit // 2]
    assert _floats(read_logical_list(emulator.lists, "cattorch state total")) == [0.0] * size
    delta = torch.arange(size, dtype=torch.float32)
    write_logical_list(emulator.lists, "cattorch step delta", delta.tolist())
    emulator.run_procedure("cattorch step")
    assert emulator.variables["cattorch status"] == "ok"
    assert _floats(read_logical_list(emulator.lists, "cattorch state total")) == delta.tolist()
    emulator.run_procedure("cattorch reset")
    assert [len(emulator.lists[name]) for name in shards] == [limit, limit, limit // 2]
    assert _floats(read_logical_list(emulator.lists, "cattorch state total")) == [0.0] * size

    comparison = verify(_Accumulator().eval(), program, result, calls=(ProgramCall("step", (delta,)),))
    assert comparison.passed, comparison


def test_sharded_input_read_by_a_sharded_weight(tmp_path, small_list_limit):
    # The interface lists are split after the weights, in a second pass over
    # the same blocks.
    small_list_limit(300)

    class Project(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(600, 8)

        def step(self, value):
            return self.linear(value)

    torch.manual_seed(0)
    model = Project().eval()
    program = ExportProgram(entrypoints=(EntryPoint(
        "step", method="step", arguments=(Input("value", torch.randn(1, 600)),),
        returns=(Output("result"),),
    ),))
    result = transpile(model, program, tmp_path / "wide")
    assert "cattorch step value" in result.sharded_lists
    calls = tuple(ProgramCall("step", (torch.randn(1, 600),)) for _ in range(2))
    assert verify(model, program, result, calls=calls).passed


@pytest.mark.parametrize("compact", [False, True])
def test_private_states_and_renamed_interface(tmp_path, compact):
    from cattorch.program_export import transpile_program

    program = ExportProgram(
        states=(State("total", torch.zeros(2)), State("offset", torch.tensor([1.0, 2.0]))),
        entrypoints=(EntryPoint(
            "step", method="step",
            arguments=(Input("delta", torch.ones(2)), StateInput("total")),
            returns=(Output("value"), StateUpdate("total")),
        ),),
    )
    result = transpile_program(
        _Accumulator().eval(), program, tmp_path / "private",
        codegen=CodegenConfig(compact_internal_names=compact),
        _private_states=True,
        _public_names={
            "cattorch step delta": "cattorch tokens",
            "cattorch step value": "cattorch logits",
            "cattorch step": "cattorch run step",
            "cattorch step complete": "cattorch step done",
        },
    )
    assert result.states == ()
    (entry,) = result.entrypoints
    assert entry.procedure == "cattorch run step"
    assert entry.completion_broadcast == "cattorch step done"
    assert [value.list_name for value in entry.inputs] == ["cattorch tokens"]
    assert [value.list_name for value in entry.outputs] == ["cattorch logits"]
    assert entry.state_inputs == entry.state_updates == ()
    assert result.procedures == (
        "cattorch init", "cattorch reset", "cattorch prepare for save", "cattorch run step",
    )
    stored = _stored_lists(result)
    assert not any("state" in name for name in stored if not name.startswith("_"))
    assert not any(name.startswith("cattorch step") for name in stored)
    if not compact:
        assert stored["_cattorch state total"] == []
        assert stored["_cattorch state offset"] == [1.0, 2.0]
    with zipfile.ZipFile(result.path) as archive:
        sprite = json.loads(archive.read("sprite.json"))
    assert "cattorch step done" in sprite["broadcasts"].values()

    emulator = _emulator(result)
    emulator.lists["cattorch tokens"] = [1.0, 2.0]
    emulator.run_procedure("cattorch run step")
    assert emulator.variables["cattorch status"] == "ok"
    assert _floats(emulator.lists["cattorch logits"]) == [2.0, 4.0]
    emulator.run_procedure("cattorch run step")
    assert _floats(emulator.lists["cattorch logits"]) == [4.0, 8.0]
    emulator.run_procedure("cattorch reset")
    emulator.run_procedure("cattorch run step")
    assert _floats(emulator.lists["cattorch logits"]) == [2.0, 4.0]


def test_public_names_reject_unknown_and_colliding_names(tmp_path):
    from cattorch.program_export import transpile_program

    program = _accumulator_program(torch.zeros(2))
    with pytest.raises(ValueError, match="no entry point name"):
        transpile_program(
            _Accumulator().eval(), program, tmp_path / "unknown",
            _public_names={"cattorch step missing": "cattorch tokens"},
        )
    with pytest.raises(ValueError, match="distinct, unused"):
        transpile_program(
            _Accumulator().eval(), program, tmp_path / "collide",
            _public_names={"cattorch step delta": "cattorch step value"},
        )


@pytest.mark.parametrize("compact", [False, True])
def test_edit_program_callback_adds_a_procedure(tmp_path, compact, small_list_limit):
    from cattorch.program_export import ProgramAssembly, transpile_program
    from cattorch.sprite import _add_warp_procedure
    from cattorch.util.scratch.dsl import Program, append, call, clear, item, repeat

    limit = small_list_limit(100)
    seen: list[ProgramAssembly] = []

    def edit(sprite, assembly):
        seen.append(assembly)
        procedure = assembly.names["cattorch step"]
        output = assembly.names["cattorch step value"]
        history = "cattorch history"
        _add_warp_procedure(sprite, "cattorch twice", Program(
            "twice", lists=(output, history),
            body=(
                clear(history),
                call(procedure),
                append(history, item(output, 1)),
                call(procedure),
                append(history, item(output, 1)),
                repeat(limit, (append(history, 0),)),
            ),
        ), x=0, y=0)
        assembly.capacities[history] = limit + 2
        assembly.public_data_names.add(history)
        assembly.public_procedures.add("cattorch twice")

    result = transpile_program(
        _Accumulator().eval(), _accumulator_program(torch.zeros(1)), tmp_path / "edit",
        codegen=CodegenConfig(compact_internal_names=compact),
        _edit_program=edit,
        _private_states=True,
    )
    (assembly,) = seen
    assert assembly.states == {"total": "_cattorch state total"}
    assert assembly.names["cattorch reset"] == "cattorch reset"
    assert "cattorch history" in result.sharded_lists
    emulator = _emulator(result)
    emulator.lists["cattorch step delta"] = [1.5]
    emulator.run_procedure("cattorch twice")
    assert emulator.variables["cattorch status"] == "ok"
    history = emulator.lists["cattorch history"] + emulator.lists["cattorch history shard 2"]
    assert _floats(history[:2]) == [3.0, 6.0]
    assert len(history) == limit + 2


def test_program_replay_reset_restores_initial_states():
    from cattorch.program_runtime import ProgramReplay

    program = _accumulator_program(torch.tensor([1.0, 2.0]))
    replay = ProgramReplay(_Accumulator().eval(), program)
    call_value = ProgramCall("step", (torch.ones(2),))
    replay.run(call_value)
    replay.run(call_value)
    torch.testing.assert_close(replay.states["total"], torch.tensor([3.0, 4.0]))
    replay.reset()
    torch.testing.assert_close(replay.states["total"], torch.tensor([1.0, 2.0]))
    (value, _state) = replay.run(call_value)
    torch.testing.assert_close(value, torch.tensor([4.0, 6.0]))
    assert program.states[0].initial.tolist() == [1.0, 2.0]
