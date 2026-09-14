"""Regression coverage for program boundaries and shared export infrastructure."""

import json
import zipfile

import pytest
import torch
from torch import nn

from cattorch import (
    CodegenConfig,
    FastConfig,
    GenerationProgram,
    QuantizationConfig,
    StorageConfig,
    UnsupportedModelError,
    transpile,
    verify,
)
from cattorch.experimental import (
    EntryPoint,
    ExportProgram,
    Input,
    ModuleAdapter,
    Output,
    ProgramCall,
    State,
    StateInput,
    StateUpdate,
    analyze,
)
from cattorch.graph_transforms import fold_eval_batch_norms
from cattorch.operator_registry import default_registry
from cattorch.util.scratch.emulator import ScratchEmulator
from cattorch.util.scratch.interface import (
    read_logical_list,
    rename_list,
    write_logical_list,
)


def emulator(artifact):
    with zipfile.ZipFile(artifact.path) as archive:
        return ScratchEmulator(json.loads(archive.read("sprite.json")))


class SharedBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2, bias=False)
        self.bn = nn.BatchNorm1d(2)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(2))
            self.bn.running_mean.fill_(3)
            self.bn.running_var.fill_(4)

    def forward(self, x):
        return self.bn(self.fc(x))

    def raw(self, x):
        return self.fc(x)


def test_batchnorm_folding_respects_all_program_methods(tmp_path):
    model = SharedBatchNorm().eval()
    program = ExportProgram(
        entrypoints=tuple(
            EntryPoint(
                name, method=name, arguments=(Input("x", torch.ones(1, 2)),), returns=(Output("y"),)
            )
            for name in ("forward", "raw")
        )
    )
    artifact = transpile(model, program, tmp_path / "bn")
    result = verify(model, program, artifact)
    assert result.passed, result
    assert isinstance(model.bn, nn.BatchNorm1d)
    assert isinstance(fold_eval_batch_norms(model).bn, nn.Identity)
    assert isinstance(
        fold_eval_batch_norms(model, methods=("forward", "raw")).bn,
        nn.BatchNorm1d,
    )


@pytest.mark.parametrize("compact", [False, True])
def test_program_shards_inputs_and_outputs_after_binding_names(tmp_path, compact):
    class Model(nn.Module):
        def first(self, x):
            return x + 1

        def second(self, x):
            return -x

    x = torch.zeros(200_001)
    x[-1] = 7
    model = Model().eval()
    program = ExportProgram(
        entrypoints=tuple(
            EntryPoint(name, method=name, arguments=(Input("x", x),), returns=(Output("y"),))
            for name in ("first", "second")
        )
    )
    artifact = transpile(
        model,
        program,
        tmp_path / "shards",
        codegen=CodegenConfig(compact_internal_names=compact),
    )
    result = verify(model, program, artifact)
    assert result.passed, result
    for name in ("first", "second"):
        assert f"cattorch {name} x" in artifact.sharded_lists
        assert f"cattorch {name} y" in artifact.sharded_lists
    assert "input" not in artifact.sharded_lists


def test_program_shards_dynamic_alias_outputs_and_state(tmp_path):
    class Model(nn.Module):
        def forward(self, state):
            return state

    initial = torch.zeros(100_001, 2)
    initial[-1] = torch.tensor([3.0, 7.0])
    program = ExportProgram(
        states=(State("rows", initial, mode="append", capacity=100_001),),
        entrypoints=(
            EntryPoint("read", arguments=(StateInput("rows"),), returns=(Output("values"),)),
        ),
    )
    artifact = transpile(
        Model().eval(),
        program,
        tmp_path / "dynamic-shards",
        codegen=CodegenConfig(compact_internal_names=True),
    )
    result = verify(Model().eval(), program, artifact)
    assert result.passed, result
    assert "cattorch read values" in artifact.sharded_lists


@pytest.mark.parametrize("compression", [False, True])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("bits", [None, 4])
def test_entrypoint_weights_do_not_collide_across_repeated_calls(
    tmp_path, compression, compact, bits
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(2, 2, bias=False)
            self.b = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.a.weight.fill_(1)
                self.b.weight.fill_(2)

        def left(self, x):
            return self.a(x)

        def right(self, x):
            return self.b(x)

    model = Model().eval()
    x = torch.ones(1, 2)
    program = ExportProgram(
        entrypoints=tuple(
            EntryPoint(name, method=name, arguments=(Input("x", x),), returns=(Output("y"),))
            for name in ("left", "right")
        )
    )
    artifact = transpile(
        model,
        program,
        tmp_path / "weights",
        storage=StorageConfig(compression=compression),
        quantization=QuantizationConfig(
            bits=bits, min_quantized_values=1, scale_precision="float32"
        )
        if bits
        else None,
        codegen=CodegenConfig(compact_internal_names=compact),
    )
    result = verify(
        model,
        program,
        artifact,
        calls=tuple(ProgramCall(name, (x,)) for name in ("left", "right", "left", "right")),
    )
    assert result.passed, result


def test_append_checks_actual_length_and_keeps_commits_transactional(tmp_path):
    class Model(nn.Module):
        def forward(self, rows):
            return rows + 0, rows + 10

    program = ExportProgram(
        states=(
            State("source", torch.ones(1, 1), mode="append", capacity=4),
            State("destination", torch.zeros(1, 1), mode="append", capacity=4),
        ),
        entrypoints=(
            EntryPoint(
                "copy",
                arguments=(StateInput("source"),),
                returns=(StateUpdate("destination"), Output("value")),
            ),
        ),
    )
    em = emulator(transpile(Model().eval(), program, tmp_path / "append"))
    for rows, expected in (([], [0.0]), ([1.0], [0.0, 1.0]), ([2.0, 3.0], [0.0, 1.0, 2.0, 3.0])):
        em.lists["cattorch state source"] = rows
        em.run_procedure("cattorch copy")
        assert em.variables["cattorch status"] == "ok"
        assert em.lists["cattorch state destination"] == expected
    em.lists["cattorch state source"] = [9.0]
    em.run_procedure("cattorch copy")
    assert em.variables["cattorch status"] == "state capacity exceeded: destination"
    assert em.lists["cattorch state destination"] == [0.0, 1.0, 2.0, 3.0]
    assert em.lists["cattorch copy value"] == [12.0, 13.0]
    em.run_procedure("cattorch reset")
    assert em.variables["cattorch status"] == "ok"


def test_duplicate_state_updates_are_rejected():
    with pytest.raises(ValueError, match="duplicate state updates"):
        EntryPoint(
            "write",
            arguments=(Input("x", torch.ones(1)),),
            returns=(StateUpdate("s"), StateUpdate("s")),
        )


def test_verification_fails_rejected_call_and_resumes_from_unmodified_state(tmp_path):
    class Model(nn.Module):
        def forward(self, ignored, state):
            return state + 1

    model = Model().eval()
    program = ExportProgram(
        states=(State("s", torch.zeros(2)),),
        entrypoints=(
            EntryPoint(
                "step",
                arguments=(Input("x", torch.zeros(2)), StateInput("s")),
                returns=(StateUpdate("s"),),
            ),
        ),
    )
    artifact = transpile(model, program, tmp_path / "status")
    result = verify(
        model,
        program,
        artifact,
        calls=(
            ProgramCall("step", (torch.ones(1),)),
            ProgramCall("step", (torch.ones(2),)),
        ),
    )
    assert not result.passed
    assert not result.calls[0].passed
    assert result.calls[0].status == "invalid length: cattorch step x"
    assert result.calls[1].passed
    assert result.calls[1].status == "ok"


def test_gptq_replays_state_updates_before_later_calls(tmp_path, monkeypatch):
    import cattorch.program_export as export_module

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=False)

        def step(self, x, state):
            updated = state + x
            return self.linear(updated), updated

    initial = torch.zeros(1, 2)
    program = ExportProgram(
        states=(State("s", initial),),
        entrypoints=(
            EntryPoint(
                "step",
                method="step",
                arguments=(
                    Input("x", torch.ones(1, 2)),
                    StateInput("s"),
                ),
                returns=(Output("y"), StateUpdate("s")),
            ),
        ),
    )
    observed = []
    original = export_module.quantize_model

    def observe(model, *args, **kwargs):
        handle = model.model.linear.register_forward_pre_hook(
            lambda _module, args: observed.append(args[0].detach().clone())
        )
        try:
            return original(model, *args, **kwargs)
        finally:
            handle.remove()

    monkeypatch.setattr(export_module, "quantize_model", observe)
    model = Model().eval()
    weights = model.linear.weight.detach().clone()
    artifact = transpile(
        model,
        program,
        tmp_path / "calibration",
        quantization=QuantizationConfig(
            bits=4, method="gptq", min_quantized_values=1, min_calibration_rows=1
        ),
        calibration_calls=(ProgramCall("step", (torch.ones(1, 2),)),) * 2,
    )
    torch.testing.assert_close(observed[0], torch.ones(1, 2))
    torch.testing.assert_close(observed[1], torch.full((1, 2), 2.0))
    torch.testing.assert_close(initial, torch.zeros(1, 2))
    torch.testing.assert_close(model.linear.weight, weights)
    assert artifact.quantization.calibration_batches == 2


def test_gptq_replays_append_state_across_different_methods(tmp_path, monkeypatch):
    import cattorch.program_export as export_module

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=False)

        def write(self, x):
            return x

        def read(self, rows):
            return self.linear(rows)

    program = ExportProgram(
        states=(State("rows", torch.empty(0, 2), mode="append", capacity=3),),
        entrypoints=(
            EntryPoint(
                "write",
                method="write",
                arguments=(Input("x", torch.ones(1, 2)),),
                returns=(StateUpdate("rows"),),
            ),
            EntryPoint(
                "read", method="read", arguments=(StateInput("rows"),), returns=(Output("y"),)
            ),
        ),
    )
    observed = []
    original = export_module.quantize_model

    def observe(model, *args, **kwargs):
        handle = model.model.linear.register_forward_pre_hook(
            lambda _module, args: observed.append(args[0].detach().clone())
        )
        try:
            return original(model, *args, **kwargs)
        finally:
            handle.remove()

    monkeypatch.setattr(export_module, "quantize_model", observe)
    artifact = transpile(
        Model().eval(),
        program,
        tmp_path / "append-calibration",
        quantization=QuantizationConfig(
            bits=4, method="gptq", min_quantized_values=1, min_calibration_rows=1
        ),
        calibration_calls=(
            ProgramCall("write", (torch.ones(1, 2),)),
            ProgramCall("read", ()),
            ProgramCall("write", (torch.full((1, 2), 3.0),)),
            ProgramCall("read", ()),
        ),
    )
    torch.testing.assert_close(observed[0], torch.ones(1, 2))
    torch.testing.assert_close(observed[1], torch.tensor([[1.0, 1.0], [3.0, 3.0]]))
    assert artifact.quantization.calibration_rows == 3


def test_eager_state_replay_rejects_overflow_without_partial_updates():
    from cattorch.program_runtime import ProgramReplay

    class Model(nn.Module):
        def forward(self, x):
            return x, x

    program = ExportProgram(
        states=(
            State("replace", torch.zeros(1, 1)),
            State("append", torch.ones(1, 1), mode="append", capacity=1),
        ),
        entrypoints=(
            EntryPoint(
                "step",
                arguments=(Input("x", torch.ones(1, 1)),),
                returns=(StateUpdate("replace"), StateUpdate("append")),
            ),
        ),
    )
    replay = ProgramReplay(Model().eval(), program)
    with pytest.raises(ValueError, match="state capacity exceeded"):
        replay.run(ProgramCall("step", (torch.full((1, 1), 2.0),)))
    torch.testing.assert_close(replay.states["replace"], torch.zeros(1, 1))
    torch.testing.assert_close(replay.states["append"], torch.ones(1, 1))


def test_adapters_preserve_shared_module_references(tmp_path):
    class Block(nn.Module):
        def forward(self, x):
            return x + 1

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = Block()
            self.b = self.a
            self.nested = nn.Sequential(self.a)

        def forward(self, x):
            return self.a(x) + self.b(x) + self.nested(x)

    paths = []
    adapter = ModuleAdapter(
        Block, lambda module, context: paths.append(context.module_path) or nn.Identity()
    )
    registry = default_registry().clone()
    registry.register_adapter(adapter)
    original = Model().eval()
    adapted = registry.adapt_model(original)
    assert adapted.a is adapted.b is adapted.nested[0]
    assert paths == ["a"]
    assert isinstance(original.a, Block)
    assert original.a is original.b
    program = ExportProgram(
        entrypoints=(
            EntryPoint("run", arguments=(Input("x", torch.zeros(1)),), returns=(Output("y"),)),
        )
    )
    em = emulator(transpile(original, program, tmp_path / "adapters", adapters=(adapter,)))
    em.lists["cattorch run x"] = [0.0]
    em.run_procedure("cattorch run")
    assert em.lists["cattorch run y"] == [0.0]


@pytest.mark.parametrize("name", ["init", "reset", "prepare for save"])
def test_lifecycle_names_are_reserved(name):
    with pytest.raises(ValueError, match="reserved"):
        EntryPoint(name, arguments=(Input("x", torch.ones(1)),), returns=(Output("y"),))


def test_generated_interface_names_cannot_collide():
    with pytest.raises(ValueError, match="collision"):
        ExportProgram(
            entrypoints=(
                EntryPoint("a b", arguments=(Input("c", torch.ones(1)),), returns=(Output("x"),)),
                EntryPoint("a", arguments=(Input("b c", torch.ones(1)),), returns=(Output("y"),)),
            )
        )
    with pytest.raises(ValueError, match="collision"):
        ExportProgram(
            states=(State("s", torch.ones(1)),),
            entrypoints=(
                EntryPoint("state", arguments=(Input("s", torch.ones(1)),), returns=(Output("y"),)),
            ),
        )
    with pytest.raises(ValueError, match="reserved shard"):
        ExportProgram(
            entrypoints=(
                EntryPoint(
                    "run", arguments=(Input("x shard 2", torch.ones(1)),), returns=(Output("y"),)
                ),
            )
        )


@pytest.mark.parametrize("size,index", [(3, 3), (3, 5), (3, -4), (0, 0)])
def test_select_rejects_out_of_bounds_indices(tmp_path, size, index):
    class Model(nn.Module):
        def forward(self, x):
            return x.select(0, index)

    x = torch.arange(float(size))
    with pytest.raises(IndexError):
        Model()(x)
    with pytest.raises(UnsupportedModelError, match="out of range"):
        transpile(Model().eval(), x, tmp_path / "invalid")
    assert not (tmp_path / "invalid.sprite3").exists()


@pytest.mark.parametrize("index", [-3, -1, 0, 2])
def test_select_preserves_valid_indices(tmp_path, index):
    class Model(nn.Module):
        def forward(self, x):
            return x.select(0, index)

    model = Model().eval()
    x = torch.arange(3.0)
    assert verify(model, x, transpile(model, x, tmp_path / "valid")).passed


@pytest.mark.parametrize("kind", ["ordinary", "program", "generation"])
@pytest.mark.parametrize(
    "options,error",
    [
        ({"optimization": "typo"}, ValueError),
        ({"sig_figs": 0}, ValueError),
        ({"sig_figs": True}, ValueError),
        ({"fast_config": FastConfig()}, ValueError),
        ({"optimization": "fast", "fast_config": {}}, TypeError),
        ({"storage": {}}, TypeError),
        ({"codegen": {}}, TypeError),
        ({"name": ""}, ValueError),
        (
            {
                "storage": StorageConfig(precision="int4"),
                "quantization": QuantizationConfig(bits=4),
            },
            ValueError,
        ),
    ],
)
def test_export_options_are_validated_before_interface_dispatch(tmp_path, kind, options, error):
    example = torch.ones(1)
    interface = {
        "ordinary": example,
        "program": ExportProgram(
            entrypoints=(
                EntryPoint("run", arguments=(Input("x", example),), returns=(Output("y"),)),
            )
        ),
        "generation": GenerationProgram("forward", torch.tensor([[0]]), max_context=4),
    }[kind]
    with pytest.raises(error):
        transpile(nn.Identity().eval(), interface, tmp_path / "invalid", **options)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("invalid", ["non_tensor", "count", "state_shape", "dropout"])
def test_analysis_and_export_share_contract_validation(tmp_path, invalid):
    class Model(nn.Module):
        def forward(self, x):
            if invalid == "non_tensor":
                return x, 7
            if invalid == "count":
                return x, x
            if invalid == "dropout":
                return torch.nn.functional.dropout(x, p=0.5, training=True)
            return x

    program = ExportProgram(
        states=(State("s", torch.zeros(3)),) if invalid == "state_shape" else (),
        entrypoints=(
            EntryPoint(
                "run",
                arguments=(Input("x", torch.ones(2)),),
                returns=(StateUpdate("s") if invalid == "state_shape" else Output("y"),),
            ),
        ),
    )
    report = analyze(Model().eval(), program)
    assert report.entrypoints[0].unsupported_operations
    assert report.warnings
    with pytest.raises((ValueError, NotImplementedError)):
        transpile(Model().eval(), program, tmp_path / "invalid")
    assert not list(tmp_path.iterdir())


def test_analysis_reports_bounded_output_shape_and_public_name(tmp_path):
    program = ExportProgram(
        states=(State("rows", torch.empty(0, 2), mode="append", capacity=4),),
        entrypoints=(
            EntryPoint("run", arguments=(StateInput("rows"),), returns=(Output("values"),)),
        ),
    )
    model = nn.Identity().eval()
    report = analyze(model, program)
    artifact = transpile(model, program, tmp_path / "analysis")
    assert not report.entrypoints[0].unsupported_operations
    assert report.entrypoints[0].outputs == artifact.entrypoints[0].outputs


def test_shared_logical_list_helpers_clear_stale_shards_and_rename_references():
    lists = {"input": [1], "input shard 2": [2]}
    write_logical_list(lists, "input", [3])
    assert lists == {"input": [3], "input shard 2": []}
    assert read_logical_list(lists, "input") == [3]
    sprite = {
        "lists": {"a": ["input", []], "b": ["input shard 2", []]},
        "variables": {"v": ["cattorch input shard count", 2]},
        "blocks": {"read": {"fields": {"LIST": ["input shard 2", "b"]}}},
    }
    rename_list(sprite, "input", "cattorch run x")
    assert sprite["lists"]["b"][0] == "cattorch run x shard 2"
    assert sprite["blocks"]["read"]["fields"]["LIST"] == ["cattorch run x shard 2", "b"]
    assert sprite["variables"]["v"][0] == "cattorch run x shard count"


def test_benchmark_input_restore_and_output_read_are_shard_aware():
    from cattorch.benchmark import _add_input_restore_procedure, _set_inputs, _target_output

    sprite = {
        "lists": {"a": ["input", []], "b": ["input shard 2", []]},
        "variables": {},
        "blocks": {},
        "broadcasts": {},
    }
    x = torch.zeros(200_001)
    x[-1] = 7
    _set_inputs(sprite, (x,))
    assert len(sprite["lists"]["a"][1]) == 200_000
    assert sprite["lists"]["b"][1] == [7.0]
    _add_input_restore_procedure(sprite, (x,))
    assert len(sprite["blocks"]) < 100
    em = ScratchEmulator(sprite)
    em.lists["input"] = []
    em.lists["input shard 2"] = []
    em.run_procedure("cattorch restore benchmark input")
    assert read_logical_list(em.lists, "input") == x.tolist()
    assert _target_output({"lists": {"a": ["output", [1.0]], "b": ["output shard 2", [2.0]]}}) == [
        1.0,
        2.0,
    ]
