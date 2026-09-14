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
from cattorch.experimental import (
    EntryPoint, ExpertFamily, ExportProgram, Input, ModuleAdapter, Output,
    ProgramCall, SparseMoE, State, StateInput, StateUpdate, analyze,
)
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
        Model(), program, tmp_path / "program",
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
    result = transpile(Counter(), program, tmp_path / "state")
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
        Counter(), program, result,
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
    model = Model()
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
    result = transpile(Model(), program, tmp_path / "multiple")
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
    result = transpile(Writer(), program, tmp_path / "append")
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
    result = transpile(Reader(), program, tmp_path / "dynamic-transpose")
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
