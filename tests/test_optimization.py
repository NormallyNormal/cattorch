import json
import zipfile

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import cattorch.graph as graph_module

from cattorch import (
    FastConfig,
    FastLayerConfig,
    GenerationConfig,
    StorageConfig,
    TranspileResult,
    transpile,
    UnsupportedModelError,
    UnsupportedOperationError,
    verify,
)
from cattorch.transpiler import _prepare_graph
from cattorch.fast import prepare_fast_matmul_weights, prepare_fast_model
from cattorch.benchmark import (
    EXACT_TIMINGS,
    FAST_TIMINGS,
    SUITE_RESULTS,
    analyze_benchmark,
    build_benchmark_suite,
    build_generation_benchmark_suite,
    build_paired_benchmark,
)
from cattorch.util.scratch.dsl import (
    Program,
    add,
    append,
    change_var,
    clear,
    costume_number,
    for_each,
    item,
    mul,
    random,
    repeat,
    round_,
    set_var,
    switch_costume,
    var,
)
from cattorch.util.scratch.emulator import ScratchEmulator
from cattorch.util.instruction.optimized import LinearInstruction


def test_export_scopes_older_torch_dynamo_recompile_limits(monkeypatch):
    config = torch._dynamo.config
    names = [
        name
        for name in (
            "cache_size_limit",
            "recompile_limit",
            "accumulated_cache_size_limit",
        )
        if hasattr(config, name)
    ]
    observed = {}

    def fake_export(model, inputs):
        observed.update({name: getattr(config, name) for name in names})
        return "exported"

    monkeypatch.setattr(graph_module, "export", fake_export)
    with config.patch(**{name: 1 for name in names}):
        assert graph_module._export_model(nn.Identity(), (torch.ones(1),)) == "exported"
        assert all(getattr(config, name) == 1 for name in names)

    assert all(value >= 1024 for value in observed.values())


class LinearActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)

    def forward(self, x):
        return torch.relu(self.linear(x) * 0.5)


class LinearGelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)

    def forward(self, x):
        return F.gelu(self.linear(x), approximate="tanh")


class GroupedLinearEpilogue(nn.Module):
    def __init__(self, inner=128, outputs=384):
        super().__init__()
        self.linear = nn.Linear(inner, outputs)

    def forward(self, value, residual):
        return torch.relu(self.linear(value) * 0.5 + residual)


class ArithmeticChain(nn.Module):
    def forward(self, x, y, residual):
        return (x + y) * 0.5 - residual


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(8, 16)
        self.value = nn.Linear(8, 16)

    def forward(self, x):
        # Deliberately compute the value branch after SiLU to exercise fused
        # lifetime management when the other operand appears later in FX order.
        return F.silu(self.gate(x)) * self.value(x)


class CausalMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mask", torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1))

    def forward(self, x):
        return x.masked_fill(self.mask, float("-inf"))


class MultiHeadCausalSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mask", torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1))

    def forward(self, scores):
        return F.softmax(scores.masked_fill(self.mask, float("-inf")), dim=-1)


class NonDivisibleAdaptivePool(nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (3, 2))


class CachedAttention(nn.Module):
    def __init__(self, width=8, heads=2, context=6):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.projection = nn.Linear(width, width, bias=False)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context, context, dtype=torch.bool), diagonal=1),
        )

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        k = k.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        v = v.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / self.head_width ** 0.5
        weights = F.softmax(
            scores.masked_fill(self.mask[:length, :length], float("-inf")), dim=-1,
        )
        joined = (weights @ v).transpose(1, 2).reshape(batch, length, width)
        return self.projection(joined)


class CachedMQAAttention(nn.Module):
    def __init__(self, width=8, heads=2, context=6):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, width + 2 * self.head_width, bias=False)
        self.projection = nn.Linear(width, width, bias=False)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context, context, dtype=torch.bool), diagonal=1),
        )

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).split(
            (width, self.head_width, self.head_width), dim=-1,
        )
        q = q.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        k = k.view(batch, length, 1, self.head_width).transpose(1, 2)
        v = v.view(batch, length, 1, self.head_width).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / self.head_width ** 0.5
        weights = F.softmax(
            scores.masked_fill(self.mask[:length, :length], float("-inf")), dim=-1,
        )
        joined = (weights @ v).transpose(1, 2).reshape(batch, length, width)
        return self.projection(joined)


class CachedBlock(nn.Module):
    def __init__(self, context=6):
        super().__init__()
        self.norm1 = nn.LayerNorm(8)
        self.attention = CachedAttention(context=context)
        self.norm2 = nn.LayerNorm(8)
        self.mlp = nn.Sequential(
            nn.Linear(8, 12), nn.GELU(approximate="tanh"), nn.Linear(12, 8),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class CachedLM(nn.Module):
    def __init__(self, context=6, layers=2):
        super().__init__()
        self.token = nn.Embedding(13, 8)
        self.position = nn.Embedding(context, 8)
        self.blocks = nn.Sequential(*(CachedBlock(context) for _ in range(layers)))
        self.norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, 13, bias=False)

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.token(tokens) + self.position(torch.arange(length))
        return self.head(self.norm(self.blocks(x)))


class CachedMQALM(CachedLM):
    def __init__(self, context=6, layers=2):
        super().__init__(context=context, layers=layers)
        for block in self.blocks:
            block.attention = CachedMQAAttention(context=context)


class LayerPositionCachedMQABlock(CachedBlock):
    def __init__(self, context=6):
        super().__init__(context=context)
        self.attention = CachedMQAAttention(context=context)
        self.position = nn.Embedding(context, 8)
        self.register_buffer("position_index", torch.zeros(1, dtype=torch.long))

    def forward(self, x):
        position = self.position(self.position_index).view(1, 1, 8)
        x = x + self.attention(self.norm1(x) + position)
        return x + self.mlp(self.norm2(x))


class LayerPositionCachedMQALM(CachedLM):
    def __init__(self, context=6, layers=2):
        super().__init__(context=context, layers=layers)
        self.blocks = nn.Sequential(*(
            LayerPositionCachedMQABlock(context=context) for _ in range(layers)
        ))


def _load_sprite(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("sprite.json"))


def _run(sprite, values):
    emulator = ScratchEmulator(sprite)
    emulator.lists["input"] = list(values)
    emulator.run()
    return [float(value) for value in emulator.lists["output"]]


def test_dsl_compiles_and_renders_readable_kernel():
    program = Program(
        "dot",
        variables=("i", "sum"),
        lists=("T1", "T2"),
        body=(
            clear("T2"),
            set_var("i", 1),
            set_var("sum", 0),
            repeat(3, (
                change_var("sum", mul(item("T1", var("i")), 2)),
                change_var("i", 1),
            )),
            append("T2", add(var("sum"), 1)),
        ),
    )
    sprite = program.compile()
    emulator = ScratchEmulator(sprite)
    emulator.lists["T1"] = [1, 2, 3]
    emulator.run()
    assert emulator.lists["T2"] == [13]
    assert "repeat 3" in program.pseudocode()
    assert "T1[i]" in program.pseudocode()


def test_dsl_round_reporter():
    program = Program(
        "round",
        lists=("output",),
        body=(append("output", round_(2.6)),),
    )
    sprite = program.compile()
    emulator = ScratchEmulator(sprite)
    emulator.run()
    assert emulator.lists["output"] == [3]
    assert "round(2.6)" in program.pseudocode()


def test_dsl_random_reporter():
    program = Program(
        "random",
        variables=("sample",),
        body=(set_var("sample", random(2, 2)),),
    )
    sprite = program.compile()
    emulator = ScratchEmulator(sprite)
    emulator.run()
    assert emulator.variables["sample"] == 2
    assert "random(2, 2)" in program.pseudocode()
    assert "operator_random" in {
        block["opcode"] for block in sprite["blocks"].values()
    }


def test_dsl_costume_lookup_blocks():
    program = Program(
        "costume lookup",
        variables=("index",),
        body=(
            switch_costume("A"),
            set_var("index", costume_number()),
        ),
    )
    sprite = program.compile()
    blocks = sprite["blocks"].values()
    opcodes = {block["opcode"] for block in blocks}
    assert "looks_switchcostumeto" in opcodes
    assert "looks_costumenumbername" in opcodes
    reporter = next(
        block for block in blocks
        if block["opcode"] == "looks_costumenumbername"
    )
    assert reporter["fields"]["NUMBER_NAME"] == ["number", None]
    switch = next(
        block for block in blocks
        if block["opcode"] == "looks_switchcostumeto"
    )
    shadow_id = switch["inputs"]["COSTUME"][1]
    shadow = sprite["blocks"][shadow_id]
    assert shadow["opcode"] == "looks_costume"
    assert shadow["fields"]["COSTUME"] == ["A", None]
    assert shadow["shadow"] is True
    assert "switch costume to 'A'" in program.pseudocode()
    assert "costume_number()" in program.pseudocode()


def test_dsl_dynamic_costume_lookup_uses_costume_menu_shadow():
    program = Program(
        "dynamic costume lookup",
        variables=("encoded",),
        body=(switch_costume(var("encoded")),),
    )
    sprite = program.compile()
    switch = next(
        block for block in sprite["blocks"].values()
        if block["opcode"] == "looks_switchcostumeto"
    )
    input_value = switch["inputs"]["COSTUME"]
    assert input_value[0] == 3
    assert input_value[1][0] == 12
    shadow = sprite["blocks"][input_value[2]]
    assert shadow["opcode"] == "looks_costume"
    assert shadow["fields"]["COSTUME"] == ["!", None]
    assert shadow["shadow"] is True


def test_dsl_for_each_uses_one_based_indices():
    program = Program(
        "for each",
        variables=("index", "sum"),
        lists=("output",),
        body=(
            set_var("sum", 0),
            for_each("index", 4, (change_var("sum", var("index")),)),
            append("output", var("sum")),
        ),
    )
    sprite = program.compile()
    emulator = ScratchEmulator(sprite)
    emulator.run()

    assert emulator.lists["output"] == [10]
    assert any(
        block["opcode"] == "control_for_each"
        for block in sprite["blocks"].values()
    )
    assert "for each index in 4:" in program.pseudocode()


def test_dsl_auto_lowers_safe_index_mutation_to_for_each():
    program = Program(
        "automatic for each",
        variables=("index",),
        lists=("source", "output"),
        list_values={"source": [2, 4, 6]},
        body=(
            set_var("index", 0),
            repeat(3, (
                change_var("index", 1),
                append("output", item("source", var("index"))),
            )),
        ),
    )
    sprite = program.compile()
    emulator = ScratchEmulator(sprite)
    emulator.run()

    assert emulator.lists["output"] == [2, 4, 6]
    opcodes = [block["opcode"] for block in sprite["blocks"].values()]
    assert "control_for_each" in opcodes
    assert "data_changevariableby" not in opcodes


@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_optimization_modes_preserve_results(tmp_path, mode):
    torch.manual_seed(4)
    model = LinearActivation()
    x = torch.randn(2, 3)
    expected = model(x).flatten().tolist()
    path = tmp_path / mode
    transpile(model, x, str(path), optimization=mode)
    actual = _run(_load_sprite(path.with_suffix(".sprite3")), x.flatten().tolist())
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), atol=1e-4, rtol=0)


def test_invalid_optimization_mode(tmp_path):
    with pytest.raises(ValueError, match="optimization must be"):
        transpile(LinearActivation(), torch.randn(1, 3), str(tmp_path / "bad"), optimization="none")


def test_fast_config_is_only_valid_in_fast_mode(tmp_path):
    with pytest.raises(ValueError, match="fast_config can only"):
        transpile(
            LinearActivation(), torch.randn(1, 3), str(tmp_path / "bad_config"),
            optimization="exact", fast_config=FastConfig(),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pruning": 1.0},
        {"pruning": True},
        {"rank": 2.5},
        {"rank": 0},
        {"rank_ratio": 0},
        {"rank": 2, "rank_ratio": 0.5},
    ],
)
def test_fast_layer_config_rejects_invalid_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        FastLayerConfig(**kwargs)


@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_multi_head_causal_softmax_broadcasts_mask_and_stays_finite(tmp_path, mode):
    torch.manual_seed(8)
    model = MultiHeadCausalSoftmax()
    scores = torch.randn(1, 2, 4, 4)
    path = tmp_path / f"causal_{mode}"
    transpile(model, scores, str(path), optimization=mode)
    actual = torch.tensor(
        _run(_load_sprite(path.with_suffix(".sprite3")), scores.flatten().tolist())
    ).reshape_as(scores)
    expected = model(scores)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)


def test_exact_lifecycle_is_idempotent_and_repeatable(tmp_path):
    torch.manual_seed(5)
    model = LinearActivation()
    first = torch.randn(2, 3)
    second = torch.randn(2, 3)
    path = tmp_path / "lifecycle"
    transpile(model, first, str(path), optimization="exact")
    sprite = _load_sprite(path.with_suffix(".sprite3"))
    opcodes = {block["opcode"] for block in sprite["blocks"].values()}
    assert {"procedures_definition", "procedures_prototype", "procedures_call"} <= opcodes

    emulator = ScratchEmulator(sprite)
    emulator.lists["input"] = first.flatten().tolist()
    emulator.run()
    torch.testing.assert_close(
        torch.tensor(emulator.lists["output"]), model(first).flatten(), atol=1e-4, rtol=0,
    )
    emulator.lists["input"] = second.flatten().tolist()
    emulator.run()
    torch.testing.assert_close(
        torch.tensor(emulator.lists["output"]), model(second).flatten(), atol=1e-4, rtol=0,
    )
    assert emulator.variables["cattorch initialized"] == 1


def test_exact_linear_uses_direct_weight_layout_and_fuses_epilogue(tmp_path):
    model = LinearActivation()
    x = torch.randn(2, 3)
    base = tmp_path / "exact"
    transpile(model, x, str(base), optimization="exact")
    sprite = _load_sprite(base.with_suffix(".sprite3"))
    exact_variables = {entry[0] for entry in sprite["variables"].values()}
    assert "weight_offset" in exact_variables
    assert "weight_index" not in exact_variables


def test_grouped_linear_matches_previous_kernel_with_rows_bias_residual_and_relu(
    tmp_path, monkeypatch,
):
    torch.manual_seed(49)
    model = GroupedLinearEpilogue().eval()
    value = torch.randn(2, 128)
    residual = torch.randn(2, 384)
    sprites = {}
    outputs = {}
    for name, threshold in (("previous", float("inf")), ("grouped", 49_152)):
        monkeypatch.setattr(LinearInstruction, "grouped_min_macs", threshold)
        base = tmp_path / name
        transpile(
            model, (value, residual), str(base), optimization="exact",
            storage=StorageConfig(compression=False),
        )
        sprite = _load_sprite(base.with_suffix(".sprite3"))
        sprites[name] = sprite
        emulator = ScratchEmulator(sprite)
        emulator.lists["input"] = value.flatten().tolist()
        emulator.lists["input_1"] = residual.flatten().tolist()
        emulator.run()
        outputs[name] = emulator.lists["output"]

    assert outputs["grouped"] == outputs["previous"]
    torch.testing.assert_close(
        torch.tensor(outputs["grouped"]), model(value, residual).flatten(),
        atol=2e-4, rtol=0,
    )
    previous_variables = {entry[0] for entry in sprites["previous"]["variables"].values()}
    grouped_variables = {entry[0] for entry in sprites["grouped"]["variables"].values()}
    assert "sum a" not in previous_variables
    assert {"sum a", "sum b", "sum c", "sum d", "input value"} <= grouped_variables
    grouped_mutations = {
        block.get("fields", {}).get("VARIABLE", [None])[0]
        for block in sprites["grouped"]["blocks"].values()
    }
    assert "weight" in grouped_mutations
    assert not {"weight a", "weight b", "weight c", "weight d"} & grouped_mutations

    weight = model.linear.weight.detach()
    expected_prefix = weight[:4, :2].T.flatten().tolist()
    stored = next(
        entry[1]
        for entry in sprites["grouped"]["lists"].values()
        if entry[0].startswith("W_linear_")
    )
    assert stored[:8] == expected_prefix


def test_large_grouped_linear_uses_row_aligned_physical_weights(tmp_path):
    torch.manual_seed(50)
    model = nn.Linear(128, 1564).eval()
    value = torch.randn(1, 128)
    path = tmp_path / "row_sharded_linear"
    result = transpile(
        model,
        value,
        str(path),
        optimization="exact",
        storage=StorageConfig(compression=False),
    )
    sprite = _load_sprite(path.with_suffix(".sprite3"))
    physical = [
        entry
        for entry in sprite["lists"].values()
        if entry[0].startswith("W_linear_")
    ]
    assert [len(entry[1]) for entry in physical] == [199_680, 512]
    assert not any(name.startswith("W_linear_") for name in result.sharded_lists)

    emulator = ScratchEmulator(sprite)
    emulator.lists["input"] = value.flatten().tolist()
    emulator.run()
    torch.testing.assert_close(
        torch.tensor(emulator.lists["output"]),
        model(value).flatten(),
        atol=1e-4,
        rtol=0,
    )


@pytest.mark.parametrize("inner,outputs", [(128, 128), (384, 129)])
def test_grouped_linear_keeps_fallback_for_small_or_non_multiple_shapes(
    tmp_path, inner, outputs,
):
    model = nn.Linear(inner, outputs).eval()
    value = torch.randn(1, inner)
    base = tmp_path / f"fallback_{inner}_{outputs}"
    transpile(
        model, value, str(base), optimization="exact",
        storage=StorageConfig(compression=False),
    )
    sprite = _load_sprite(base.with_suffix(".sprite3"))
    variables = {entry[0] for entry in sprite["variables"].values()}
    assert "sum a" not in variables
    assert "weight_offset" in variables


def test_exact_fuses_same_shape_arithmetic_chain(tmp_path):
    torch.manual_seed(27)
    model = ArithmeticChain()
    inputs = tuple(torch.randn(2, 3) for _ in range(3))
    graph = _prepare_graph(model, inputs, optimization="exact")
    assert len(graph.elementwise_fusions) == 1
    assert len(next(iter(graph.elementwise_fusions.values()))) == 3

    base = tmp_path / "arithmetic_exact"
    transpile(model, inputs, str(base), optimization="exact")
    sprite = _load_sprite(base.with_suffix(".sprite3"))

    emulator = ScratchEmulator(sprite)
    for index, value in enumerate(inputs):
        name = "input" if index == 0 else f"input_{index}"
        emulator.lists[name] = value.flatten().tolist()
    emulator.run()
    actual = [float(value) for value in emulator.lists["output"]]
    expected = model(*inputs).flatten()
    torch.testing.assert_close(
        torch.tensor(actual), expected, atol=1e-6, rtol=0,
    )


def test_paired_benchmark_project_and_analyzer(tmp_path):
    project_path = build_paired_benchmark(
        LinearActivation(), torch.randn(2, 3), tmp_path / "paired.sb3",
        warmups=1, repeats=3,
    )
    with zipfile.ZipFile(project_path) as archive:
        project = json.loads(archive.read("project.json"))
        assets = {name: archive.read(name) for name in archive.namelist() if name != "project.json"}
    assert [target["name"] for target in project["targets"]] == [
        "Stage", "cattorch exact", "cattorch fast",
    ]
    stage = project["targets"][0]
    for entry in stage["lists"].values():
        if entry[0] == EXACT_TIMINGS:
            entry[1] = [2.0, 2.2, 1.8]
        elif entry[0] == FAST_TIMINGS:
            entry[1] = [1.0, 1.1, 0.9]
    with zipfile.ZipFile(project_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("project.json", json.dumps(project))
        for name, data in assets.items():
            archive.writestr(name, data)

    report = analyze_benchmark(project_path)
    assert report["runs"] == 3
    assert report["speedup"] == pytest.approx(2.0)
    assert report["output_max_abs_error"] is None  # Project was built, not run.

    for entry in stage["lists"].values():
        if entry[0] == EXACT_TIMINGS:
            entry[1] = [0.033, 0.033]
        elif entry[0] == FAST_TIMINGS:
            entry[1] = [0, 0]
    exact_output = next(
        entry for entry in project["targets"][1]["lists"].values()
        if entry[0] == "output"
    )
    exact_output[1] = [1.0]
    with zipfile.ZipFile(project_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("project.json", json.dumps(project))
        for name, data in assets.items():
            archive.writestr(name, data)

    below_resolution = analyze_benchmark(project_path)
    assert below_resolution["speedup"] is None
    assert below_resolution["speedup_95_percent_interval"] is None
    assert below_resolution["output_length_match"] is False
    assert below_resolution["exact_output_values"] == 1
    assert below_resolution["fast_output_values"] == 0


def test_benchmark_suite_runs_init_before_timed_repeats_and_analyzes(tmp_path):
    suite_path = build_benchmark_suite(
        [
            ("first", LinearActivation(), torch.randn(2, 3)),
            ("second", LinearActivation(), torch.randn(2, 3)),
        ],
        tmp_path / "suite.sb3",
        iterations=3,
    )
    with zipfile.ZipFile(suite_path) as archive:
        project = json.loads(archive.read("project.json"))
        assets = {
            name: archive.read(name)
            for name in archive.namelist()
            if name != "project.json"
        }

    assert [target["name"] for target in project["targets"]] == [
        "Stage", "first exact", "first fast", "second exact", "second fast",
    ]
    stage = project["targets"][0]
    result_list = next(
        entry for entry in stage["lists"].values() if entry[0] == SUITE_RESULTS
    )
    result_id = next(
        identifier
        for identifier, entry in stage["lists"].items()
        if entry[0] == SUITE_RESULTS
    )
    monitor = next(item for item in project["monitors"] if item["id"] == result_id)
    assert monitor["visible"] is True

    hat_id = next(
        identifier
        for identifier, block in stage["blocks"].items()
        if block["opcode"] == "event_whenflagclicked"
    )
    opcodes = []
    block_id = stage["blocks"][hat_id]["next"]
    while block_id:
        block = stage["blocks"][block_id]
        opcodes.append(block["opcode"])
        block_id = block["next"]
    assert opcodes[:5] == [
        "data_deletealloflist",
        "event_broadcastandwait",  # init
        "sensing_resettimer",
        "control_repeat",          # forwards
        "data_addtolist",
    ]
    timed_repeats = [
        block for block in stage["blocks"].values()
        if block["opcode"] == "control_repeat"
    ]
    assert all(block["inputs"]["TIMES"][1][1] == 3 for block in timed_repeats)

    result_list[1] = [
        "first exact: 3.0", "first fast: 2.0",
        "second exact: 8.0", "second fast: 4.0",
    ]
    with zipfile.ZipFile(suite_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("project.json", json.dumps(project))
        for name, data in assets.items():
            archive.writestr(name, data)

    report = analyze_benchmark(suite_path)
    assert report["kind"] == "suite"
    assert report["iterations_per_result"] == 3
    assert [item["speedup"] for item in report["benchmarks"]] == [1.5, 2.0]


def test_fast_benchmark_suite_compares_exact_and_fast(tmp_path):
    case_config = FastConfig(
        activations=False,
        weights=FastLayerConfig(pruning=0.5),
    )
    suite_path = build_benchmark_suite(
        [("linear", nn.Linear(16, 12), torch.randn(2, 16), case_config)],
        tmp_path / "fast_suite.sb3",
        iterations=10,
    )
    with zipfile.ZipFile(suite_path) as archive:
        project = json.loads(archive.read("project.json"))
        assets = {
            name: archive.read(name)
            for name in archive.namelist()
            if name != "project.json"
        }
    assert [target["name"] for target in project["targets"]] == [
        "Stage", "linear exact", "linear fast",
    ]
    fast_target = project["targets"][2]
    assert "_linear_block_offsets" in {
        entry[0] for entry in fast_target["lists"].values()
    }
    stage = project["targets"][0]
    results = next(entry for entry in stage["lists"].values() if entry[0] == SUITE_RESULTS)
    results[1] = ["linear exact: 2.0", "linear fast: 1.0"]
    with zipfile.ZipFile(suite_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("project.json", json.dumps(project))
        for name, data in assets.items():
            archive.writestr(name, data)

    report = analyze_benchmark(suite_path)
    assert report["comparison"] == "exact_vs_fast"
    assert report["benchmarks"][0]["exact_seconds"] == 2.0
    assert report["benchmarks"][0]["fast_seconds"] == 1.0
    assert report["benchmarks"][0]["speedup"] == 2.0


def test_nonfinite_values_are_valid_json_and_keep_scratch_semantics(tmp_path):
    path = tmp_path / "causal"
    transpile(CausalMask(), torch.randn(3, 3), str(path))
    with zipfile.ZipFile(path.with_suffix(".sprite3")) as archive:
        raw = archive.read("sprite.json").decode()

    def reject_nonstandard_number(token):
        raise AssertionError(f"Non-standard JSON number: {token}")

    sprite = json.loads(raw, parse_constant=reject_nonstandard_number)
    assert '"-Infinity"' in raw
    actual = _run(sprite, list(range(9)))
    assert actual[1] == float("-inf")


def test_exact_adaptive_pool_handles_nonuniform_receptive_fields(tmp_path):
    torch.manual_seed(19)
    model = NonDivisibleAdaptivePool()
    x = torch.randn(1, 2, 5, 7)
    expected = model(x)
    path = tmp_path / "adaptive_nonuniform"
    transpile(model, x, str(path), optimization="exact")
    actual = torch.tensor(
        _run(_load_sprite(path.with_suffix(".sprite3")), x.flatten().tolist())
    ).reshape_as(expected)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (nn.GELU(approximate="tanh"), lambda x: x * torch.sigmoid(1.702 * x)),
    ],
    ids=("quick_gelu",),
)
def test_fast_activations_use_documented_approximations(tmp_path, model, expected):
    x = torch.linspace(-5, 5, 41)
    path = tmp_path / type(model).__name__
    transpile(model, x, str(path), optimization="fast")
    actual = torch.tensor(
        _run(_load_sprite(path.with_suffix(".sprite3")), x.tolist())
    )
    torch.testing.assert_close(actual, expected(x), atol=1e-5, rtol=0)


def test_fast_fused_linear_uses_quick_gelu_and_can_be_disabled(tmp_path):
    torch.manual_seed(21)
    model = LinearGelu().eval()
    x = torch.randn(2, 3)
    linear = model.linear(x)

    fast_path = tmp_path / "fused_quick_gelu"
    transpile(model, x, str(fast_path), optimization="fast")
    fast = torch.tensor(
        _run(_load_sprite(fast_path.with_suffix(".sprite3")), x.flatten().tolist())
    ).reshape(2, 4)
    torch.testing.assert_close(
        fast, linear * torch.sigmoid(1.702 * linear), atol=1e-4, rtol=0,
    )

    disabled_path = tmp_path / "fused_exact_gelu"
    transpile(
        model, x, str(disabled_path), optimization="fast",
        fast_config=FastConfig(activations=False),
    )
    disabled = torch.tensor(
        _run(_load_sprite(disabled_path.with_suffix(".sprite3")), x.flatten().tolist())
    ).reshape(2, 4)
    torch.testing.assert_close(disabled, model(x), atol=1e-4, rtol=0)


@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_swiglu_fusion_is_exact_in_both_modes(tmp_path, mode):
    torch.manual_seed(22)
    model = SwiGLU().eval()
    x = torch.randn(3, 8)
    path = tmp_path / f"swiglu_{mode}"
    transpile(model, x, str(path), optimization=mode)
    actual = torch.tensor(
        _run(_load_sprite(path.with_suffix(".sprite3")), x.flatten().tolist())
    ).reshape_as(model(x))
    torch.testing.assert_close(actual, model(x), atol=1e-4, rtol=0)
    graph = _prepare_graph(model, (x,), optimization=mode)
    assert len(graph.paired_swiglu_fusions) == 1
    sprite = _load_sprite(path.with_suffix(".sprite3"))
    variables = {entry[0] for entry in sprite["variables"].values()}
    assert {"gate sum", "value sum", "input value"} <= variables


def test_fast_layernorm_matches_one_pass_formula(tmp_path):
    torch.manual_seed(23)
    model = nn.LayerNorm(8)
    x = torch.randn(4, 8)
    mean = x.double().mean(dim=-1, keepdim=True)
    variance = x.double().square().mean(dim=-1, keepdim=True) - mean.square()
    expected = (x.double() - mean) / torch.sqrt(torch.clamp_min(variance, 0) + model.eps)
    path = tmp_path / "fast_layernorm"
    transpile(model, x, str(path), optimization="fast")
    actual = torch.tensor(
        _run(_load_sprite(path.with_suffix(".sprite3")), x.flatten().tolist()),
        dtype=torch.double,
    ).reshape_as(expected)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)


class NamedLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(16, 12)

    def forward(self, x):
        return self.projection(x)


class StaticMatmul(nn.Module):
    def __init__(self, inner=16, columns=12):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(inner, columns))

    def forward(self, x):
        return x @ self.weight


class TransposedStaticMatmul(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(12, 16))

    def forward(self, x):
        return x @ self.weight.T


class DynamicMatmul(nn.Module):
    def forward(self, left, right):
        return left @ right


def test_oversized_static_matmul_uses_row_aligned_output_groups(tmp_path):
    torch.manual_seed(32)
    model = StaticMatmul(inner=128, columns=1564).eval()
    value = torch.randn(1, 128)
    path = tmp_path / "row_sharded_matmul"
    result = transpile(
        model,
        value,
        str(path),
        storage=StorageConfig(compression=False),
    )
    sprite = _load_sprite(path.with_suffix(".sprite3"))
    weights = [
        entry
        for entry in sprite["lists"].values()
        if entry[0].startswith("W_matmul_")
    ]
    assert [len(entry[1]) for entry in weights] == [199_680, 512]
    assert not any(name.startswith("W_matmul_") for name in result.sharded_lists)

    emulator = ScratchEmulator(sprite)
    emulator.lists["input"] = value.flatten().tolist()
    emulator.run()
    torch.testing.assert_close(
        torch.tensor(emulator.lists["output"]),
        model(value).flatten(),
        atol=1e-4,
        rtol=0,
    )


@pytest.mark.parametrize(
    "policy",
    [
        FastLayerConfig(rank=4),
        FastLayerConfig(pruning=0.5),
        FastLayerConfig(rank=4, pruning=0.5),
    ],
    ids=("low_rank", "pruned", "low_rank_pruned"),
)
def test_fast_weight_transforms_match_the_transformed_pytorch_model(tmp_path, policy):
    torch.manual_seed(29)
    model = NamedLinear().eval()
    original_weight = model.projection.weight.detach().clone()
    x = torch.randn(3, 16)
    config = FastConfig(weights=policy, activations=False, layer_norm=False, softmax=False)
    transformed = prepare_fast_model(model, config)
    expected = transformed(x)

    path = tmp_path / "fast_weights"
    transpile(model, x, str(path), optimization="fast", fast_config=config)
    actual = torch.tensor(
        _run(_load_sprite(path.with_suffix(".sprite3")), x.flatten().tolist())
    ).reshape_as(expected)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)
    torch.testing.assert_close(model.projection.weight, original_weight)


def test_fast_module_override_replaces_global_weight_policy():
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    config = FastConfig(
        weights=FastLayerConfig(pruning=0.5),
        overrides={"1": FastLayerConfig()},
    )
    transformed = prepare_fast_model(model, config)
    assert torch.count_nonzero(transformed[0].weight) < transformed[0].weight.numel()
    assert torch.count_nonzero(transformed[1].weight) == transformed[1].weight.numel()


def test_fast_unknown_module_override_is_rejected():
    with pytest.raises(ValueError, match="missing"):
        prepare_fast_model(
            nn.Linear(8, 8),
            FastConfig(overrides={"missing": FastLayerConfig(pruning=0.5)}),
        )


def test_fast_pruned_linear_preserves_non_block_remainder(tmp_path):
    torch.manual_seed(31)
    model = nn.Linear(10, 8).eval()
    x = torch.randn(3, 10)
    config = FastConfig(weights=FastLayerConfig(pruning=0.5))
    expected = prepare_fast_model(model, config)(x)
    path = tmp_path / "pruned_remainder"
    transpile(model, x, str(path), optimization="fast", fast_config=config)
    actual = torch.tensor(
        _run(_load_sprite(path.with_suffix(".sprite3")), x.flatten().tolist())
    ).reshape_as(expected)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)


@pytest.mark.parametrize(
    "policy",
    [
        FastLayerConfig(pruning=0.5),
        FastLayerConfig(rank=4),
        FastLayerConfig(rank=4, pruning=0.5),
    ],
    ids=("pruned", "low_rank", "low_rank_pruned"),
)
def test_fast_static_matmul_matches_export_time_weight_transform(tmp_path, policy):
    torch.manual_seed(33)
    model = StaticMatmul().eval()
    original = model.weight.detach().clone()
    x = torch.randn(3, 16)
    factors = prepare_fast_matmul_weights(model.weight, policy)
    expected = x
    for factor in factors:
        expected = expected @ factor
    config = FastConfig(
        activations=False,
        layer_norm=False,
        softmax=False,
        weights=policy,
    )
    path = tmp_path / "static_matmul"
    transpile(model, x, str(path), optimization="fast", fast_config=config)
    actual = torch.tensor(
        _run(_load_sprite(path.with_suffix(".sprite3")), x.flatten().tolist())
    ).reshape_as(expected)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)
    torch.testing.assert_close(model.weight, original)


def test_fast_static_matmul_pruning_preserves_remainder(tmp_path):
    torch.manual_seed(35)
    model = StaticMatmul(inner=10, columns=8).eval()
    x = torch.randn(3, 10)
    policy = FastLayerConfig(pruning=0.5)
    expected = x @ prepare_fast_matmul_weights(model.weight, policy)[0]
    config = FastConfig(weights=policy)
    path = tmp_path / "static_matmul_remainder"
    transpile(model, x, str(path), optimization="fast", fast_config=config)
    actual = torch.tensor(
        _run(_load_sprite(path.with_suffix(".sprite3")), x.flatten().tolist())
    ).reshape_as(expected)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)


def test_fast_transposed_static_matmul_is_transformed(tmp_path):
    torch.manual_seed(36)
    model = TransposedStaticMatmul().eval()
    x = torch.randn(3, 16)
    policy = FastLayerConfig(rank=4)
    factors = prepare_fast_matmul_weights(model.weight.T, policy)
    expected = x @ factors[0] @ factors[1]
    path = tmp_path / "transposed_static_matmul"
    transpile(
        model, x, str(path), optimization="fast",
        fast_config=FastConfig(weights=policy),
    )
    actual = torch.tensor(
        _run(_load_sprite(path.with_suffix(".sprite3")), x.flatten().tolist())
    ).reshape_as(expected)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)


def test_fast_dynamic_matmul_remains_exact_with_weight_policy(tmp_path):
    torch.manual_seed(38)
    model = DynamicMatmul()
    left = torch.randn(2, 3, 8)
    right = torch.randn(2, 8, 5)
    path = tmp_path / "dynamic_matmul"
    transpile(
        model, (left, right), str(path), optimization="fast",
        fast_config=FastConfig(weights=FastLayerConfig(rank_ratio=0.25)),
    )
    sprite = _load_sprite(path.with_suffix(".sprite3"))
    # Multiple inputs occupy distinct Scratch lists; use the emulator directly.
    emulator = ScratchEmulator(sprite)
    emulator.lists["input"] = left.flatten().tolist()
    emulator.lists["input_1"] = right.flatten().tolist()
    emulator.run()
    actual = torch.tensor(emulator.lists["output"]).reshape(2, 3, 5)
    torch.testing.assert_close(actual, model(left, right), atol=1e-4, rtol=0)
    assert not any(
        entry[0].startswith("W_fast_") for entry in sprite["lists"].values()
    )


@pytest.mark.parametrize(
    ("model", "x", "policy"),
    [
        (nn.Conv1d(4, 8, 3, padding=1), torch.randn(1, 4, 7), FastLayerConfig(pruning=0.5)),
        (nn.Conv2d(4, 8, 3, padding=1), torch.randn(1, 4, 4, 5), FastLayerConfig(pruning=0.5)),
        (nn.Conv1d(8, 12, 3, padding=1), torch.randn(1, 8, 7), FastLayerConfig(rank=3)),
        (nn.Conv2d(8, 12, 3, padding=1), torch.randn(1, 8, 4, 5), FastLayerConfig(rank=3)),
    ],
    ids=("conv1d_pruned", "conv2d_pruned", "conv1d_low_rank", "conv2d_low_rank"),
)
def test_fast_convolution_transforms_match_transformed_model(tmp_path, model, x, policy):
    torch.manual_seed(37)
    model.eval()
    config = FastConfig(weights=policy)
    expected = prepare_fast_model(model, config)(x)
    path = tmp_path / "fast_conv"
    transpile(model, x, str(path), optimization="fast", fast_config=config)
    actual = torch.tensor(
        _run(_load_sprite(path.with_suffix(".sprite3")), x.flatten().tolist())
    ).reshape_as(expected)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)


def test_exact_causal_attention_is_recognized_as_one_structural_path():
    torch.manual_seed(40)
    model = CachedAttention(width=8, heads=2, context=16).eval()
    x = torch.randn(1, 16, 8)
    graph = _prepare_graph(model, (x,), optimization="exact")
    assert len(graph.causal_score_fusions) == 1
    assert len(graph.causal_softmaxes) == 1
    assert len(graph.causal_value_matmuls) == 1
    assert len(graph.qkv_linear_fusions) == 1


def test_embedding_add_fuses_token_and_position_lookups():
    model = CachedLM(context=4, layers=1).eval()
    tokens = torch.randint(0, 13, (1, 4))
    graph = _prepare_graph(model, (tokens,), optimization="exact")
    assert len(graph.embedding_add_fusions) == 1


def test_generation_prefill_decode_reset_and_overflow(tmp_path):
    torch.manual_seed(41)
    model = CachedLM(context=5, layers=2).eval()
    path = tmp_path / "cached_generation"
    transpile(
        model,
        torch.tensor([[1]]),
        str(path),
        generation=GenerationConfig(max_context=5),
    )
    sprite = _load_sprite(path.with_suffix(".sprite3"))
    emulator = ScratchEmulator(sprite)
    assert {
        "cattorch forward", "cattorch init", "cattorch reset cache",
        "cattorch prefill", "cattorch decode",
    } <= set(emulator._procedures)
    assert set(sprite["broadcasts"].values()) == {
        "cattorch init done", "cattorch prefill done",
    }
    assert emulator.variables["cattorch max context"] == 5
    assert emulator.variables["cattorch cache length"] == 0

    prompt = [1, 3, 5]
    emulator.lists["input"] = prompt
    emulator._exec_chain(emulator._procedures["cattorch prefill"])
    assert emulator.broadcasts == [
        "cattorch init done", "cattorch prefill done",
    ]
    expected = model(torch.tensor([prompt]))[0, -1]
    torch.testing.assert_close(
        torch.tensor(emulator.lists["output"]), expected, atol=1e-4, rtol=0,
    )
    assert emulator.variables["cattorch cache length"] == 3
    assert len(emulator.lists["cattorch K cache 1"]) == 3 * 8
    assert len(emulator.lists["cattorch K cache 2"]) == 3 * 8

    for token in (2, 4):
        prompt.append(token)
        emulator.lists["input"] = [token]
        emulator._exec_chain(emulator._procedures["cattorch decode"])
        expected = model(torch.tensor([prompt]))[0, -1]
        torch.testing.assert_close(
            torch.tensor(emulator.lists["output"]), expected, atol=1e-4, rtol=0,
        )
    assert emulator.variables["cattorch cache length"] == 5

    emulator.lists["input"] = [6]
    emulator._exec_chain(emulator._procedures["cattorch decode"])
    assert emulator.variables["cattorch status"] == "maximum context exceeded"
    assert emulator.lists["output"] == []
    assert emulator.variables["cattorch cache length"] == 5

    emulator._exec_chain(emulator._procedures["cattorch reset cache"])
    assert emulator.variables["cattorch status"] == "ok"
    assert emulator.variables["cattorch cache length"] == 0
    assert emulator.lists["cattorch K cache 1"] == []
    assert emulator.lists["cattorch V cache 2"] == []


def test_generation_rejects_malformed_runtime_inputs_without_mutating_cache(tmp_path):
    model = CachedLM(context=5, layers=1).eval()
    path = tmp_path / "cached_generation_guards"
    transpile(
        model,
        torch.tensor([[1]]),
        path,
        generation=GenerationConfig(max_context=5, top_k=3),
    )
    emulator = ScratchEmulator(_load_sprite(path.with_suffix(".sprite3")))

    for invalid in ([], [1, 2]):
        emulator.lists["input"] = invalid
        emulator.run_procedure("cattorch decode")
        assert emulator.variables["cattorch status"] == "decode requires exactly one token"
        assert emulator.variables["cattorch cache length"] == 0
        assert emulator.lists["output"] == []
        assert emulator.lists["cattorch top k ids"] == []

    emulator.lists["input"] = []
    emulator.run_procedure("cattorch prefill")
    assert emulator.variables["cattorch status"] == "prefill requires at least one token"
    assert emulator.variables["cattorch cache length"] == 0

    emulator.lists["input"] = [1, 2, 3, 4, 5, 6]
    emulator.run_procedure("cattorch prefill")
    assert emulator.variables["cattorch status"] == "prefill exceeds maximum context"
    assert emulator.variables["cattorch cache length"] == 0
    assert emulator.lists["output"] == []


def test_generation_mqa_caches_one_kv_head_and_matches_full_forward(tmp_path):
    torch.manual_seed(45)
    model = CachedMQALM(context=5, layers=2).eval()
    path = tmp_path / "cached_mqa_generation"
    transpile(
        model,
        torch.tensor([[1]]),
        str(path),
        generation=GenerationConfig(max_context=5),
    )
    emulator = ScratchEmulator(_load_sprite(path.with_suffix(".sprite3")))

    prompt = [1, 3, 5]
    emulator.lists["input"] = prompt
    emulator._exec_chain(emulator._procedures["cattorch prefill"])
    expected = model(torch.tensor([prompt]))[0, -1]
    torch.testing.assert_close(
        torch.tensor(emulator.lists["output"]), expected, atol=1e-4, rtol=0,
    )
    assert len(emulator.lists["cattorch K cache 1"]) == 3 * 4
    assert len(emulator.lists["cattorch V cache 2"]) == 3 * 4

    emulator.lists["input"] = [2]
    emulator._exec_chain(emulator._procedures["cattorch decode"])
    expected = model(torch.tensor([[*prompt, 2]]))[0, -1]
    torch.testing.assert_close(
        torch.tensor(emulator.lists["output"]), expected, atol=1e-4, rtol=0,
    )


def test_generation_single_pass_top_k_matches_torch_and_clears_state(tmp_path):
    torch.manual_seed(47)
    model = CachedLM(context=5, layers=1).eval()
    path = tmp_path / "cached_top_k"
    transpile(
        model,
        torch.tensor([[1]]),
        str(path),
        generation=GenerationConfig(max_context=5, top_k=5),
    )
    emulator = ScratchEmulator(_load_sprite(path.with_suffix(".sprite3")))
    assert "cattorch select top k" in emulator._procedures

    prompt = [1, 3, 5]
    emulator.lists["input"] = prompt
    emulator._exec_chain(emulator._procedures["cattorch prefill"])
    expected_logits = model(torch.tensor([prompt]))[0, -1]
    expected_values, expected_ids = torch.topk(expected_logits, 5)
    assert emulator.lists["cattorch top k ids"] == expected_ids.tolist()
    assert emulator.lists["cattorch top k values"] == [
        emulator.lists["output"][index] for index in expected_ids.tolist()
    ]
    torch.testing.assert_close(
        torch.tensor(emulator.lists["cattorch top k values"]),
        expected_values,
        atol=2e-4,
        rtol=0,
    )

    emulator.lists["input"] = [2]
    emulator._exec_chain(emulator._procedures["cattorch decode"])
    expected_logits = model(torch.tensor([[*prompt, 2]]))[0, -1]
    expected_values, expected_ids = torch.topk(expected_logits, 5)
    assert emulator.lists["cattorch top k ids"] == expected_ids.tolist()
    assert emulator.lists["cattorch top k values"] == [
        emulator.lists["output"][index] for index in expected_ids.tolist()
    ]
    torch.testing.assert_close(
        torch.tensor(emulator.lists["cattorch top k values"]),
        expected_values,
        atol=2e-4,
        rtol=0,
    )

    emulator.lists["input"] = [4]
    emulator._exec_chain(emulator._procedures["cattorch decode"])
    assert len(emulator.lists["cattorch top k values"]) == 5
    emulator.lists["input"] = [6]
    emulator._exec_chain(emulator._procedures["cattorch decode"])
    assert emulator.variables["cattorch status"] == "maximum context exceeded"
    assert emulator.lists["output"] == []
    assert emulator.lists["cattorch top k values"] == []
    assert emulator.lists["cattorch top k ids"] == []

    emulator._exec_chain(emulator._procedures["cattorch reset cache"])
    assert emulator.lists["cattorch top k values"] == []
    assert emulator.lists["cattorch top k ids"] == []


def test_generation_layer_positions_use_their_own_attention_cache():
    model = LayerPositionCachedMQALM(context=5, layers=2).eval()
    graph = _prepare_graph(
        model,
        (torch.tensor([[1]]),),
        optimization="exact",
        generation=GenerationConfig(max_context=5),
    )
    caches = [
        graph.generation.position_embeddings[node.name][0]
        for node in graph.nodes
        if node.name in graph.generation.position_embeddings
    ]
    assert caches == [
        "cattorch K cache 1",
        "cattorch K cache 1",
        "cattorch K cache 2",
    ]


def test_long_generation_fuses_attention_and_emits_qkv_caches_directly(tmp_path):
    torch.manual_seed(51)
    model = CachedMQALM(context=64, layers=1).eval()
    token = torch.tensor([[1]])
    graph = _prepare_graph(
        model,
        (token,),
        optimization="exact",
        generation=GenerationConfig(max_context=64),
    )
    assert len(graph.generation.qkv_caches) == 1
    assert len(graph.generation.prepopulated_scores) == 1
    assert len(graph.generation.prepopulated_values) == 1
    assert len(graph.generation.attention_fusions) == 1
    assert graph.generation.hidden_prefill_qkv in graph.generation.qkv_caches

    prompt = [1, 3, 5, 2]
    path = tmp_path / "long_cached_attention"
    transpile(
        model,
        token,
        str(path),
        generation=GenerationConfig(max_context=64),
        storage=StorageConfig(compression=False),
    )
    emulator = ScratchEmulator(_load_sprite(path.with_suffix(".sprite3")))
    emulator.lists["input"] = prompt
    emulator._exec_chain(emulator._procedures["cattorch prefill"])

    torch.testing.assert_close(
        torch.tensor(emulator.lists["output"]),
        model(torch.tensor([prompt]))[0, -1],
        atol=1e-4,
        rtol=0,
    )
    assert len(emulator.lists["cattorch K cache 1"]) == len(prompt) * 4
    assert len(emulator.lists["cattorch V cache 1"]) == len(prompt) * 4


def test_transformer_sized_cached_qkv_uses_interleaved_destination_weights(tmp_path):
    class WideCachedLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(11, 128)
            self.attention = CachedAttention(width=128, heads=4, context=5)
            self.head = nn.Linear(128, 11, bias=False)

        def forward(self, tokens):
            return self.head(self.attention(self.embedding(tokens)))

    torch.manual_seed(52)
    model = WideCachedLM().eval()
    token = torch.tensor([[2]])
    path = tmp_path / "grouped_cached_qkv"
    transpile(
        model,
        token,
        str(path),
        generation=GenerationConfig(max_context=5, hidden_prefill=False),
        storage=StorageConfig(compression=False),
    )
    sprite = _load_sprite(path.with_suffix(".sprite3"))
    qkv_weights = [
        entry
        for entry in sprite["lists"].values()
        if entry[0].startswith("W_qkv_")
    ]
    assert [len(entry[1]) for entry in qkv_weights] == [16_384] * 3
    mutations = {
        block.get("fields", {}).get("VARIABLE", [None])[0]
        for block in sprite["blocks"].values()
    }
    assert {"sum a", "sum b", "sum c", "sum d", "weight_index"} <= mutations

    emulator = ScratchEmulator(sprite)
    emulator.lists["input"] = [2]
    emulator._exec_chain(emulator._procedures["cattorch decode"])
    torch.testing.assert_close(
        torch.tensor(emulator.lists["output"]),
        model(token).flatten(),
        atol=1e-4,
        rtol=0,
    )


def test_hidden_prefill_skips_intermediate_output_heads(tmp_path):
    torch.manual_seed(46)
    model = CachedLM(context=5, layers=1).eval()
    prompt = [1, 3, 5, 2]
    emulators = {}
    for hidden_prefill in (False, True):
        path = tmp_path / f"prefill_{hidden_prefill}"
        transpile(
            model,
            torch.tensor([[1]]),
            str(path),
            generation=GenerationConfig(
                max_context=5, hidden_prefill=hidden_prefill,
            ),
        )
        emulator = ScratchEmulator(_load_sprite(path.with_suffix(".sprite3")))
        emulator.lists["input"] = prompt
        emulator._exec_chain(emulator._procedures["cattorch prefill"])
        emulators[hidden_prefill] = emulator

    expected = model(torch.tensor([prompt]))[0, -1]
    for emulator in emulators.values():
        torch.testing.assert_close(
            torch.tensor(emulator.lists["output"]), expected, atol=1e-4, rtol=0,
        )
        assert emulator.variables["cattorch cache length"] == len(prompt)
    assert (
        emulators[True].opcode_counts["data_changevariableby"]
        < emulators[False].opcode_counts["data_changevariableby"]
    )


def test_generation_config_validates_hidden_prefill():
    with pytest.raises(TypeError, match="hidden_prefill must be a boolean"):
        GenerationConfig(max_context=5, hidden_prefill=1)


@pytest.mark.parametrize("top_k", [True, 1.5, "5"])
def test_generation_config_validates_top_k_type(top_k):
    with pytest.raises(TypeError, match="top_k must be an integer or None"):
        GenerationConfig(max_context=5, top_k=top_k)


@pytest.mark.parametrize("top_k", [0, 17])
def test_generation_config_validates_top_k_range(top_k):
    with pytest.raises(ValueError, match="top_k must be between 1 and 16"):
        GenerationConfig(max_context=5, top_k=top_k)


def test_generation_top_k_cannot_exceed_output_size(tmp_path):
    model = CachedLM(context=4, layers=1).eval()
    with pytest.raises(ValueError, match=r"top_k \(14\) exceeds output size \(13\)"):
        transpile(
            model,
            torch.tensor([[1]]),
            str(tmp_path / "top_k_too_large"),
            generation=GenerationConfig(max_context=4, top_k=14),
        )


def test_generation_validates_input_shape_and_shards_oversized_cache(tmp_path):
    model = CachedLM(context=4, layers=1).eval()
    with pytest.raises(ValueError, match=r"shape \[1, 1\]"):
        transpile(
            model, torch.tensor([[1, 2]]), str(tmp_path / "bad_generation"),
            generation=GenerationConfig(4),
        )

    with pytest.raises(ValueError, match="position table capacity"):
        transpile(
            model, torch.tensor([[1]]), str(tmp_path / "position_overflow"),
            generation=GenerationConfig(5),
        )

    class PositionlessCachedLM(CachedLM):
        def forward(self, tokens):
            hidden = self.token(tokens)
            return self.head(self.norm(self.blocks(hidden)))

    model = PositionlessCachedLM(context=4, layers=1).eval()
    path = tmp_path / "oversized_cache"
    transpile(
        model, torch.tensor([[1]]), str(path),
        generation=GenerationConfig(25_001),
    )
    sprite = _load_sprite(path.with_suffix(".sprite3"))
    names = {entry[0] for entry in sprite["lists"].values()}
    assert "cattorch K cache 1 shard 2" in names
    assert "cattorch V cache 1 shard 2" in names
    variables = {entry[0]: entry[1] for entry in sprite["variables"].values()}
    assert variables["cattorch K cache 1 shard count"] == 2
    assert variables["cattorch K cache 1 logical length"] == 25_001 * 8


def test_eval_conv_batchnorm_is_folded_without_mutating_model(tmp_path):
    torch.manual_seed(42)
    model = nn.Sequential(nn.Conv1d(3, 4, 3, padding=1), nn.BatchNorm1d(4)).eval()
    original_weight = model[0].weight.detach().clone()
    x = torch.randn(1, 3, 5)
    path = tmp_path / "folded_batchnorm"
    transpile(model, x, str(path))
    actual = torch.tensor(
        _run(_load_sprite(path.with_suffix(".sprite3")), x.flatten().tolist())
    ).reshape_as(model(x))
    torch.testing.assert_close(actual, model(x), atol=1e-4, rtol=0)
    torch.testing.assert_close(model[0].weight, original_weight)
    assert isinstance(model[1], nn.BatchNorm1d)


def test_fast_whole_neuron_pruning_changes_hidden_shape_and_exports(tmp_path):
    torch.manual_seed(43)
    model = nn.Sequential(
        nn.Linear(6, 12), nn.GELU(approximate="tanh"), nn.Linear(12, 4),
    ).eval()
    config = FastConfig(
        activations=False, layer_norm=False, softmax=False, neuron_pruning=0.5,
    )
    transformed = prepare_fast_model(model, config)
    assert transformed[0].out_features == 6
    assert transformed[2].in_features == 6
    assert model[0].out_features == 12
    x = torch.randn(2, 6)
    path = tmp_path / "neuron_pruned"
    transpile(model, x, str(path), optimization="fast", fast_config=config)
    actual = torch.tensor(
        _run(_load_sprite(path.with_suffix(".sprite3")), x.flatten().tolist())
    ).reshape_as(transformed(x))
    torch.testing.assert_close(actual, transformed(x), atol=1e-4, rtol=0)


def test_generation_benchmark_suite_bundles_stateless_and_cached(tmp_path):
    model = CachedLM(context=4, layers=1).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    path = build_generation_benchmark_suite(
        [("cached_lm", model, tokens, tokens[:, :2])],
        tmp_path / "generation_suite.sb3",
        iterations=2,
    )
    with zipfile.ZipFile(path) as archive:
        project = json.loads(archive.read("project.json"))
    assert [target["name"] for target in project["targets"]] == [
        "Stage", "cached_lm stateless", "cached_lm cached",
    ]
    results = next(
        entry for entry in project["targets"][0]["lists"].values()
        if entry[0] == "cattorch benchmark results"
    )
    assert results[1] == []


def test_parameter_resolution_uses_export_signature_without_name_collisions(tmp_path):
    class Nested(nn.Module):
        def __init__(self):
            super().__init__()
            self.bar = nn.Linear(2, 2, bias=False)

    class CollidingNames(nn.Module):
        def __init__(self):
            super().__init__()
            self.foo_bar = nn.Linear(2, 2, bias=False)
            self.foo = Nested()
            self.foo_bar.weight.data.fill_(1)
            self.foo.bar.weight.data.fill_(2)

        def forward(self, value):
            return self.foo_bar(value) + self.foo.bar(value)

    model = CollidingNames().eval()
    value = torch.tensor([[1.0, 2.0]])
    path = tmp_path / "colliding_state_names"
    transpile(model, value, str(path), storage=StorageConfig(compression=False))
    sprite = _load_sprite(path.with_suffix(".sprite3"))
    input_names = {entry[0] for entry in sprite["lists"].values() if entry[0].startswith("input")}
    assert input_names == {"input"}
    torch.testing.assert_close(
        torch.tensor(_run(sprite, value.flatten().tolist())),
        model(value).flatten(),
        atol=1e-5,
        rtol=0,
    )


def test_batchnorm_folding_requires_real_dataflow_connection(tmp_path):
    class Branched(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 3)
            self.bn = nn.BatchNorm1d(3)

        def forward(self, value):
            return self.linear(value) + self.bn(value)

    torch.manual_seed(61)
    model = Branched().eval()
    model.bn.running_mean.copy_(torch.tensor([0.2, -0.3, 0.5]))
    model.bn.running_var.copy_(torch.tensor([0.5, 1.5, 2.0]))
    value = torch.randn(2, 3)
    path = tmp_path / "branched_batchnorm"
    transpile(model, value, str(path), storage=StorageConfig(compression=False))
    actual = torch.tensor(
        _run(_load_sprite(path.with_suffix(".sprite3")), value.flatten().tolist())
    ).reshape_as(model(value))
    torch.testing.assert_close(actual, model(value), atol=1e-4, rtol=0)


@pytest.mark.parametrize(
    "left_shape,right_shape",
    [((2, 3), (2, 1)), ((2, 1), (2, 3)), ((2, 3, 4), (1, 3, 1))],
)
def test_exact_elementwise_supports_general_pytorch_broadcasting(
    tmp_path, left_shape, right_shape,
):
    class Add(nn.Module):
        def forward(self, left, right):
            return left + right

    left = torch.randn(left_shape)
    right = torch.randn(right_shape)
    path = tmp_path / f"broadcast_{len(left_shape)}_{len(right_shape)}"
    transpile(Add(), (left, right), str(path), storage=StorageConfig(compression=False))
    emulator = ScratchEmulator(_load_sprite(path.with_suffix(".sprite3")))
    emulator.lists["input"] = left.flatten().tolist()
    emulator.lists["input_1"] = right.flatten().tolist()
    emulator.run()
    torch.testing.assert_close(
        torch.tensor(emulator.lists["output"]),
        (left + right).flatten(),
        atol=1e-6,
        rtol=0,
    )


def test_identity_model_materializes_a_distinct_output(tmp_path):
    class Identity(nn.Module):
        def forward(self, value):
            return value

    value = torch.randn(2, 3)
    path = tmp_path / "identity"
    transpile(Identity(), value, str(path))
    sprite = _load_sprite(path.with_suffix(".sprite3"))
    assert {entry[0] for entry in sprite["lists"].values()} >= {"input", "output"}
    torch.testing.assert_close(
        torch.tensor(_run(sprite, value.flatten().tolist())), value.flatten(),
    )


def test_multiple_model_outputs_are_rejected_explicitly(tmp_path):
    class Pair(nn.Module):
        def forward(self, value):
            return value + 1, value * 2

    with pytest.raises(UnsupportedModelError, match="exactly one tensor output"):
        transpile(Pair(), torch.ones(2), str(tmp_path / "pair"))


def test_training_dropout_is_rejected_instead_of_silently_removed(tmp_path):
    with pytest.raises(UnsupportedOperationError, match=r"model\.eval\(\)"):
        transpile(nn.Dropout(0.5).train(), torch.ones(4), str(tmp_path / "dropout"))


def test_semantic_runtime_dtype_cast_is_rejected(tmp_path):
    class ToInteger(nn.Module):
        def forward(self, value):
            return value.to(torch.int64)

    with pytest.raises(UnsupportedOperationError, match="dtype conversion"):
        transpile(ToInteger(), torch.tensor([1.2, -2.8]), str(tmp_path / "dtype"))


def test_evaluation_dropout_remains_a_safe_alias(tmp_path):
    value = torch.randn(4)
    path = tmp_path / "eval_dropout"
    transpile(nn.Dropout(0.5).eval(), value, str(path))
    torch.testing.assert_close(
        torch.tensor(_run(_load_sprite(path.with_suffix(".sprite3")), value.tolist())),
        value,
    )


def test_masked_fill_supports_non_suffix_broadcasting(tmp_path):
    class Mask(nn.Module):
        def forward(self, value, mask):
            return value.masked_fill(mask, -10.0)

    value = torch.randn(2, 3)
    mask = torch.tensor([[True], [False]])
    path = tmp_path / "broadcast_mask"
    transpile(
        Mask(), (value, mask), str(path),
        storage=StorageConfig(compression=False),
    )
    emulator = ScratchEmulator(_load_sprite(path.with_suffix(".sprite3")))
    emulator.lists["input"] = value.flatten().tolist()
    emulator.lists["input_1"] = mask.flatten().tolist()
    emulator.run()
    torch.testing.assert_close(
        torch.tensor(emulator.lists["output"]),
        value.masked_fill(mask, -10.0).flatten(),
    )


def test_neuron_pruning_requires_real_linear_activation_linear_dataflow():
    class BranchedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Linear(4, 8)
            self.activation = nn.GELU()
            self.down = nn.Linear(8, 3)

        def forward(self, value, unrelated_hidden):
            return self.activation(self.up(value)), self.down(unrelated_hidden)

    model = BranchedMLP().eval()
    transformed = prepare_fast_model(
        model,
        FastConfig(neuron_pruning=0.5),
    )
    assert transformed.up.out_features == 8
    assert transformed.down.in_features == 8


def test_transpile_returns_metadata_and_separates_path_from_sprite_name(tmp_path):
    model = nn.Linear(3, 2).eval()
    value = torch.randn(1, 3)
    result = transpile(
        model,
        value,
        tmp_path / "nested" / "model.sprite3",
        name="Friendly model",
    )
    assert isinstance(result, TranspileResult)
    assert result.path == tmp_path / "nested" / "model.sprite3"
    assert result.sprite_name == "Friendly model"
    assert result.archive_bytes == result.path.stat().st_size
    assert result.expanded_json_bytes > result.archive_bytes
    assert result.block_count > 0
    assert result.list_count > 0
    assert result.inputs[0].shape == (1, 3)
    assert result.output.shape == (1, 2)
    assert {"cattorch init", "cattorch forward", "cattorch prepare for save"} <= set(
        result.procedures
    )
    assert _load_sprite(result.path)["name"] == "Friendly model"


def test_transpile_defaults_sprite_name_to_path_stem_and_avoids_double_extension(tmp_path):
    result = transpile(nn.ReLU(), torch.ones(2), tmp_path / "folder" / "relu.sprite3")
    assert result.path.name == "relu.sprite3"
    assert result.sprite_name == "relu"


@pytest.mark.parametrize("sig_figs", [True, 0, -1, 1.5])
def test_transpile_validates_significant_figures(tmp_path, sig_figs):
    with pytest.raises(ValueError, match="sig_figs"):
        transpile(nn.ReLU(), torch.ones(2), tmp_path / "bad_sig_figs", sig_figs=sig_figs)


def test_verify_reports_exact_and_lossy_export_error(tmp_path):
    model = nn.Linear(3, 2, bias=False).eval()
    model.weight.data.copy_(torch.tensor([[0.1234567, -0.7654321, 0.3333333], [1.234567, 0.2, -0.4]]))
    value = torch.tensor([[0.7, -1.1, 2.3]])

    exact = transpile(model, value, tmp_path / "verify_exact")
    exact_report = verify(model, value, exact)
    assert exact_report.passed
    assert exact_report.expected_shape == (1, 2)
    assert exact_report.actual_values == 2

    lossy = transpile(
        model,
        value,
        tmp_path / "verify_lossy",
        storage=StorageConfig(precision="float16"),
    )
    lossy_report = verify(model, value, lossy, atol=0, rtol=0)
    assert not lossy_report.passed
    assert lossy_report.max_abs_error > 0
    assert lossy_report.worst_index in {0, 1}


def test_unsupported_operation_error_names_module_and_exported_node(tmp_path):
    class CosineBlock(nn.Module):
        def forward(self, value):
            return torch.cos(value)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = CosineBlock()

        def forward(self, value):
            return self.features(value)

    with pytest.raises(UnsupportedOperationError) as caught:
        transpile(Model(), torch.ones(2), tmp_path / "cosine")
    message = str(caught.value)
    assert "aten.cos.default" in message
    assert "module: features" in message
    assert "exported node: cos" in message
