"""Exported sprites in the official Scratch VM, compared with PyTorch.

The rest of the suite runs sprites in ``ScratchEmulator``. These run them in
``@scratch/scratch-vm``, so a difference between the emulator and Scratch
shows up here. They need Node.js and ``npm ci`` in ``benchmarks/``, and skip
without them unless ``CATTORCH_REQUIRE_VM`` is set, as it is in CI.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cattorch import (
    CharTokenizer, CodegenConfig, GenerationProgram, QuantizationConfig, StorageConfig, build_project,
    entrypoint, transpile,
)
from cattorch.util.scratch.emulator import ProjectEmulator

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "benchmarks" / "run_project_vm.cjs"
MODULES = ROOT / "benchmarks" / "node_modules"

pytestmark = pytest.mark.vm


@pytest.fixture(scope="module", autouse=True)
def scratch_vm():
    if shutil.which("node") is None or not (MODULES / "@scratch" / "scratch-vm").is_dir():
        message = "needs Node.js and `npm ci` in benchmarks/"
        if os.environ.get("CATTORCH_REQUIRE_VM"):
            pytest.fail(message)
        pytest.skip(message)


def run_vm(project, tmp_path, *, answers=(), lists=None, calls=(), flag=True) -> dict:
    """Run a project in scratch-vm and return what it left on the stage.

    ``lists`` are set before the green flag. Each of ``calls`` is
    ``(message, lists)``: the lists are set, then the message is broadcast and
    waited for.
    """
    output = tmp_path / "vm.json"
    command = ["node", str(RUNNER), str(project.path), f"--out={output}"]
    command += [f"--answer={answer}" for answer in answers]
    command += [f"--list={name}={json.dumps(values)}" for name, values in (lists or {}).items()]
    for message, inputs in calls:
        command += [f"--list={name}={json.dumps(values)}" for name, values in inputs.items()]
        command.append(f"--broadcast={message}")
    if not flag:
        command.append("--no-flag")
    environment = {**os.environ, "NODE_PATH": str(MODULES)}
    completed = subprocess.run(
        command, capture_output=True, text=True, env=environment, timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(output.read_text())


def values(stage, name) -> torch.Tensor:
    return torch.tensor([float(value) for value in stage[name]])


class Convolutional(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(4, 4, 3, groups=2), nn.GELU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(4, 3)

    def forward(self, image):
        return self.head(self.features(image).flatten(1)).log_softmax(-1)


MODELS = {
    "mlp": (
        lambda: nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.LayerNorm(16), nn.Linear(16, 3)),
        lambda: torch.randn(2, 4),
    ),
    "cnn": (Convolutional, lambda: torch.randn(1, 1, 8, 8)),
    "transformer": (
        lambda: nn.TransformerEncoderLayer(8, 2, 16, dropout=0.0, batch_first=True),
        lambda: torch.randn(1, 5, 8),
    ),
}


@pytest.mark.parametrize("name", MODELS)
def test_models_match_pytorch(tmp_path, name):
    build, example = MODELS[name]
    torch.manual_seed(0)
    model, value = build().eval(), example()
    result = transpile(model, value, tmp_path / name)
    project = build_project(tmp_path / name, result, example_inputs=value)
    stage = run_vm(project, tmp_path)["stage"]
    torch.testing.assert_close(
        values(stage, f"{name} output"), model(value).detach().flatten(), atol=1e-4, rtol=1e-4,
    )


@pytest.mark.parametrize(("storage", "quantization"), [
    (StorageConfig(precision="float16"), None),
    (StorageConfig(), QuantizationConfig(bits=8, min_quantized_values=1)),
    (StorageConfig(), QuantizationConfig(bits=4, min_quantized_values=1)),
    (StorageConfig(compression=False), None),
], ids=["float16", "int8", "int4", "uncompressed"])
def test_stored_weights_decode_as_in_the_emulator(tmp_path, storage, quantization):
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 4)).eval()
    value = torch.randn(1, 64)
    result = transpile(
        model, value, tmp_path / "stored", storage=storage, quantization=quantization,
    )
    project = build_project(tmp_path / "stored", result, example_inputs=value)
    stage = run_vm(project, tmp_path)["stage"]
    emulator = ProjectEmulator(project.path)
    emulator.green_flag()
    expected = values(emulator.stage.lists, "stored output")
    torch.testing.assert_close(values(stage, "stored output"), expected, atol=1e-9, rtol=1e-9)


class Pool(nn.Module):
    """Attention from a learned query over a variable-length memory."""

    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(8, 2, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, 8))
        self.recurrent = nn.GRU(8, 8, batch_first=True)

    def forward(self, memory):
        pooled = self.attention(self.query, memory, memory, need_weights=False)[0]
        return pooled[:, 0] + self.recurrent(memory)[1][0] + memory.mean(1)


def test_variable_lengths_match_pytorch(tmp_path):
    torch.manual_seed(2)
    model = Pool().eval()
    result = transpile(model, torch.randn(1, 3, 8), tmp_path / "pool", dynamic={0: {1: (1, 12)}})
    project = build_project(tmp_path / "pool", result)
    for length in (1, 9):
        memory = torch.randn(1, length, 8)
        stage = run_vm(
            project, tmp_path, lists={"pool input": memory.flatten().tolist()},
        )["stage"]
        torch.testing.assert_close(
            values(stage, "pool output"), model(memory).detach().flatten(), atol=1e-4, rtol=1e-4,
        )


class Tally(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("seen", torch.zeros(0, 3))
        self.weights = nn.Parameter(torch.randn(3))

    @entrypoint(example=lambda: (torch.ones(3),), capacity={"seen": 6})
    def add(self, row):
        self.seen = torch.cat([self.seen, row.reshape(1, 3)])
        return (self.seen @ self.weights).softmax(0) @ self.seen


def test_program_state_carries_between_calls(tmp_path):
    torch.manual_seed(3)
    model = Tally().eval()
    result = transpile(model, None, tmp_path / "tally")
    project = build_project(tmp_path / "tally", result)
    rows = torch.randn(3, 3)
    stage = run_vm(project, tmp_path, flag=False, calls=[
        ("tally add", {"tally cattorch add row": row.tolist()}) for row in rows
    ])["stage"]
    expected = None
    for row in rows:
        expected = model.add(row)
    torch.testing.assert_close(
        values(stage, "tally cattorch add output"), expected.detach(), atol=1e-4, rtol=1e-4,
    )


class Readout(nn.Module):
    """A five-value output reduced from a product longer than a Scratch list."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[0.5], [1.0], [1.5], [2.0], [2.5]]))
        self.bias = nn.Parameter(torch.arange(5.0))

    def read(self, memory):
        return (memory.reshape(5, -1) * self.weight).sum(-1) + self.bias


def test_an_output_list_holding_a_longer_intermediate_is_sharded(tmp_path):
    # The output's list holds the 200,005-value product before the reduction.
    from cattorch import EntryPoint, ExportProgram, Output, State, StateInput

    program = ExportProgram(
        states=(State("memory", torch.ones(200_005)),),
        entrypoints=(EntryPoint(
            "read", method="read", arguments=(StateInput("memory"),), returns=(Output("value"),),
        ),),
    )
    model = Readout().eval()
    result = transpile(model, program, tmp_path / "readout")
    project = build_project(tmp_path / "readout", result)
    stage = run_vm(project, tmp_path, flag=False, calls=[("readout read", {})])["stage"]
    expected = model.read(torch.ones(200_005)).detach()
    torch.testing.assert_close(values(stage, "readout cattorch read value"), expected, atol=1e-6, rtol=1e-6)


def _layered_language():
    from test_project import Language

    class Layered(Language):
        """Two ``blocks.N`` layers, which layer sharing factors into one."""

        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(
                nn.TransformerEncoderLayer(8, 2, 16, dropout=0.0, batch_first=True)
                for _ in range(2)
            )
            del self.layer

        def forward(self, tokens):
            length = tokens.shape[1]
            x = self.tokens(tokens) + self.positions(torch.arange(length))
            mask = nn.Transformer.generate_square_subsequent_mask(length)
            for block in self.blocks:
                x = block(x, src_mask=mask, is_causal=True)
            return self.head(x)

    return Layered()


@pytest.mark.parametrize("sharing", ["off", "auto"])
def test_generation_with_its_tokenizer_matches_pytorch(tmp_path, sharing):
    from test_project import CHARACTERS, Language, char_tokenizer, greedy

    torch.manual_seed(2)
    model = (Language() if sharing == "off" else _layered_language()).eval()
    result = transpile(
        model, GenerationProgram(max_context=16), tmp_path / "lm",
        codegen=CodegenConfig(layer_sharing=sharing),
    )
    if sharing == "auto":
        with zipfile.ZipFile(result.path) as archive:
            assert b"_cattorch shared transformer layer" in archive.read("sprite.json")
    tokenizer = CharTokenizer(char_tokenizer()).save(tmp_path / "chars", name="chars")
    project = build_project(
        tmp_path / "demo", result, tokenizer=tokenizer, max_new_tokens=6, temperature=0,
    )
    said = [entry["text"] for entry in run_vm(project, tmp_path, answers=["the cat", ""])["said"]]
    text = "".join(CHARACTERS[index] for index in greedy(model, "the cat", 6))
    assert "the cat" + text in said


def test_a_benchmark_suite_runs(tmp_path):
    from cattorch import build_benchmark_suite

    path = build_benchmark_suite(
        [("first", nn.Linear(3, 2).eval(), torch.randn(2, 3))], tmp_path / "suite.sb3",
        iterations=2,
    )
    stage = run_vm(SimpleNamespace(path=path), tmp_path)["stage"]
    results = stage["cattorch benchmark results"]
    assert [entry.split(":")[0] for entry in results] == ["first exact", "first fast"]


def test_a_reset_while_a_prefill_is_paused_stops_the_prefill(tmp_path):
    # Scratch pauses the prefill once it has run for 500 ms, and runs the
    # second script, which resets: the prefill must not then report ok, and
    # decode must fail until the next prefill.
    from cattorch.project import _Builder
    from cattorch.util.scratch.dsl import (
        add, append, call, change_var, clear, eq, for_each, gt, item, length, repeat_until,
        set_var, var,
    )
    from test_autoregressive import NanoGPT
    from test_generation_automatic import INTERRUPTED, logits_of
    from test_generation_overlap import OVERTAKEN

    torch.manual_seed(4)
    model = NanoGPT(layers=4, width=96, heads=4, context=32).eval()
    result = transpile(model, GenerationProgram(max_context=32), tmp_path / "lm")
    builder = _Builder()
    sprite = builder.add_sprite(result.path, visible=False)
    first, second, logits = (
        builder.global_list(name) for name in ("first results", "second results", "logits")
    )
    done = builder.global_variable("first done", 0)
    flag = {"opcode": "event_whenflagclicked", "fields": {}}
    status, busy = "cattorch status", "_cattorch generation busy"
    tokens = "cattorch tokens"

    def run(procedure, values, results):
        return (
            clear(tokens), *(append(tokens, value) for value in values),
            call(procedure), append(results, var(status)),
        )

    builder.script(sprite, flag, (
        call("cattorch init"),
        *run("cattorch prefill", list(range(1, 25)), first),
        set_var(done, 1),
    ), lists=(tokens, first), variables=(status, done))
    builder.script(sprite, flag, (
        repeat_until(gt(add(var(busy), var(done)), 0), (change_var("waits", 1),)),
        # 0 if the prefill was still running.
        append(second, var(done)),
        call("cattorch reset"), append(second, var(status)),
        repeat_until(eq(var(done), 1), (change_var("waits", 1),)),
        *run("cattorch decode", [6], second),
        *run("cattorch prefill", [5, 3], second),
        *run("cattorch decode", [7], second),
        for_each("index", length("cattorch logits"), (
            append(logits, item("cattorch logits", var("index"))),
        )),
    ), lists=(tokens, second, logits, "cattorch logits"),
        variables=(status, busy, done, "waits", "index"), y=400)
    path = tmp_path / "overlap.sb3"
    builder.write(path)
    stage = run_vm(SimpleNamespace(path=path), tmp_path)["stage"]
    assert stage["first results"] == [OVERTAKEN]
    assert stage["second results"] == [0, "ok", INTERRUPTED, "ok", "ok"]
    torch.testing.assert_close(
        values(stage, "logits"), logits_of(model, [5, 3, 7]), atol=1e-4, rtol=0,
    )
