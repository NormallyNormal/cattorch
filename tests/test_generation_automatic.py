"""Automatic cached generation end to end in the Scratch emulator."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from cattorch import (
    CodegenConfig, FastConfig, GenerationProgram, QuantizationConfig,
    UnsupportedModelError, transpile,
)
from cattorch.util.scratch.emulator import ScratchEmulator

sys.path.insert(0, str(Path(__file__).parent))
from test_autoregressive import NanoGPT, NanoMQA, RopeLM, tiny_decoder  # noqa: E402
from test_optimization import CachedLM  # noqa: E402

TOLERANCE = 1e-4


def load_sprite(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("sprite.json"))


def logits_of(model, prefix):
    with torch.no_grad():
        output = model(torch.tensor([prefix]))
    if not isinstance(output, torch.Tensor):
        output = output[0] if isinstance(output, tuple) else output.logits
    return output[0, -1].flatten()


def emulator_for(model, tmp_path, name="auto", *, codegen=None, **program):
    program.setdefault("max_context", 16)
    result = transpile(
        model, GenerationProgram(**program), tmp_path / name, codegen=codegen,
    )
    return ScratchEmulator(load_sprite(result.path)), result


def assert_logits(emulator, model, prefix):
    actual = torch.tensor(emulator.lists["cattorch logits"])
    torch.testing.assert_close(
        actual, logits_of(model, prefix), atol=TOLERANCE, rtol=0,
    )


def prefill_and_decode(emulator, model, prompt, more):
    emulator.run_procedure("cattorch init")
    emulator.lists["cattorch tokens"] = list(prompt)
    emulator.run_procedure("cattorch prefill")
    assert emulator.variables["cattorch status"] == "ok"
    prefix = list(prompt)
    assert_logits(emulator, model, prefix)
    for token in more:
        prefix.append(token)
        emulator.lists["cattorch tokens"] = [token]
        emulator.run_procedure("cattorch decode")
        assert emulator.variables["cattorch status"] == "ok"
        assert_logits(emulator, model, prefix)
    return prefix


def nano():
    torch.manual_seed(5)
    return NanoGPT(layers=2, width=16, heads=2, context=16).eval()


MODELS = {
    "nanogpt": nano,
    "mqa_broadcast": lambda: (torch.manual_seed(8), NanoMQA(layers=2, heads=4, context=16).eval())[1],
    "rope_gqa": lambda: (torch.manual_seed(6), RopeLM(layers=2).eval())[1],
    "cached_lm": lambda: (torch.manual_seed(7), CachedLM(context=16, layers=2).eval())[1],
    "gpt2": lambda: tiny_decoder("gpt2").eval(),
    "llama": lambda: tiny_decoder("llama").eval(),
    "qwen2": lambda: tiny_decoder("qwen2").eval(),
}


@pytest.fixture(scope="module", params=sorted(MODELS))
def exported(request, tmp_path_factory):
    model = MODELS[request.param]()
    emulator, result = emulator_for(
        model, tmp_path_factory.mktemp(request.param), max_context=16,
    )
    return request.param, model, emulator, result


def test_prefill_then_decode_matches_full_forward(exported):
    name, model, emulator, _ = exported
    prefill_and_decode(emulator, model, [1, 3, 5, 2, 7], [4, 6, 9, 11, 2])
    assert emulator.variables["cattorch cache length"] == 10
    assert len(emulator.lists["cattorch logits"]) == logits_of(model, [1]).numel()
    if name in {"llama", "qwen2"}:
        # 2 KV heads of width 32 / 4, not 4 query heads: caches hold KV heads.
        assert len(emulator.lists["_cattorch K cache 1"]) == 10 * 2 * 8
        assert len(emulator.lists["_cattorch V cache 2"]) == 10 * 2 * 8


def test_reset_restores_the_first_step(exported):
    _, model, emulator, _ = exported
    prefill_and_decode(emulator, model, [3, 1], [2])
    emulator.run_procedure("cattorch reset")
    assert emulator.variables["cattorch cache length"] == 0
    assert emulator.lists["_cattorch K cache 1"] == []
    prefill_and_decode(emulator, model, [2, 4, 6], [1])


def test_overflow_rolls_back_and_reset_recovers(tmp_path):
    model = nano()
    emulator, _ = emulator_for(model, tmp_path, max_context=5)
    prefix = prefill_and_decode(emulator, model, [1, 2, 3], [4, 5])
    assert len(prefix) == 5
    caches = {
        name: list(emulator.lists[name])
        for name in ("_cattorch K cache 1", "_cattorch V cache 2")
    }
    emulator.lists["cattorch tokens"] = [6]
    emulator.run_procedure("cattorch decode")
    assert emulator.variables["cattorch status"] == "maximum context exceeded"
    assert emulator.lists["cattorch logits"] == []
    assert emulator.variables["cattorch cache length"] == 5
    for name, values in caches.items():
        assert emulator.lists[name] == values
    emulator.lists["cattorch tokens"] = [1, 2, 3, 4, 5, 6]
    emulator.run_procedure("cattorch prefill")
    assert emulator.variables["cattorch status"] == "prefill exceeds maximum context"
    emulator.run_procedure("cattorch reset")
    prefill_and_decode(emulator, model, [4], [2])


def test_top_k_reports_the_largest_logits(tmp_path):
    model = nano()
    emulator, _ = emulator_for(model, tmp_path, max_context=8, top_k=3)
    emulator.run_procedure("cattorch init")
    emulator.lists["cattorch tokens"] = [2, 9, 4]
    emulator.run_procedure("cattorch prefill")
    expected = logits_of(model, [2, 9, 4]).topk(3)
    assert emulator.lists["cattorch top k ids"] == expected.indices.tolist()
    torch.testing.assert_close(
        torch.tensor(emulator.lists["cattorch top k values"]), expected.values,
        atol=TOLERANCE, rtol=0,
    )


@pytest.mark.parametrize("sharing", ["off", "auto"])
def test_hidden_prefill_skips_the_final_layers_attention_and_mlp(tmp_path, sharing):
    torch.manual_seed(13)
    model = CachedLM(context=16, layers=2).eval()
    counts = {}
    for hidden in (False, True):
        emulator, _ = emulator_for(
            model, tmp_path, f"{sharing}_{hidden}", max_context=16,
            hidden_prefill=hidden, codegen=CodegenConfig(layer_sharing=sharing),
        )
        shared = "_cattorch shared transformer layer" in emulator._procedures
        assert shared == (sharing == "auto")
        emulator.run_procedure("cattorch init")
        before = sum(emulator.opcode_counts.values())
        emulator.lists["cattorch tokens"] = [1, 2, 3, 4, 5, 6, 7, 8]
        emulator.run_procedure("cattorch prefill")
        counts[hidden] = sum(emulator.opcode_counts.values()) - before
        assert_logits(emulator, model, [1, 2, 3, 4, 5, 6, 7, 8])
        # The skipped tokens still filled the final layer's cache.
        emulator.lists["cattorch tokens"] = [9]
        emulator.run_procedure("cattorch decode")
        assert_logits(emulator, model, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert counts[True] < counts[False] * 0.8


@pytest.mark.parametrize("sharing", ["off", "auto"])
def test_decode_before_any_prefill_writes_logits(tmp_path, sharing):
    torch.manual_seed(13)
    model = CachedLM(context=8, layers=2).eval()
    emulator, _ = emulator_for(
        model, tmp_path, max_context=8, codegen=CodegenConfig(layer_sharing=sharing),
    )
    emulator.lists["cattorch tokens"] = [3]
    emulator.run_procedure("cattorch decode")
    assert emulator.variables["cattorch status"] == "ok"
    assert_logits(emulator, model, [3])


def test_layers_differing_in_a_constant_are_not_shared(tmp_path):
    class Block(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale
            self.qkv = nn.Linear(8, 24)
            self.mlp = nn.Linear(8, 8)

        def forward(self, x):
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            attended = F.scaled_dot_product_attention(
                q[:, None], k[:, None], v[:, None], is_causal=True,
            )[:, 0]
            x = x + self.scale * attended
            return x + self.mlp(x)

    class Scaled(nn.Module):
        def __init__(self):
            super().__init__()
            self.tokens = nn.Embedding(16, 8)
            self.blocks = nn.ModuleList(Block(1.0 / (index + 1)) for index in range(2))
            self.head = nn.Linear(8, 16)

        def forward(self, tokens):
            x = self.tokens(tokens)
            for block in self.blocks:
                x = block(x)
            return self.head(x)

    torch.manual_seed(0)
    model = Scaled().eval()
    emulator, _ = emulator_for(
        model, tmp_path, max_context=4, codegen=CodegenConfig(layer_sharing="auto"),
    )
    assert "_cattorch shared transformer layer" not in emulator._procedures
    prefill_and_decode(emulator, model, [1, 2, 3], [4])


def test_rotate_half_rope_uses_the_rotary_kernel(tmp_path, monkeypatch):
    import cattorch.graph

    fused = []
    original = cattorch.graph._fuse_rotate_half

    def counting(graph):
        original(graph)
        fused.extend(
            node for node in graph.nodes if str(node.target) == "cattorch.rotary_half.default"
        )

    monkeypatch.setattr(cattorch.graph, "_fuse_rotate_half", counting)
    torch.manual_seed(6)
    model = RopeLM(layers=1).eval()
    emulator, _ = emulator_for(model, tmp_path, max_context=8)
    assert len(fused) == 2  # queries and keys
    prefill_and_decode(emulator, model, [1, 2, 3], [4])


def test_caches_and_position_are_not_static_weights(exported):
    _, _, _, result = exported
    sprite = load_sprite(result.path)
    names = [entry[0] for entry in sprite["lists"].values()]
    assert "_cattorch K cache 1" in names
    assert not any(name.startswith("W_") and "cache" in name for name in names)
    assert not any("position" in name for name in names if name.startswith("W_"))


def test_fast_softmax_kernel_skips_the_maximum_pass(tmp_path):
    model = nano()
    only_softmax = FastConfig(activations=False, layer_norm=False, softmax=True)
    fast = transpile(
        model, GenerationProgram(max_context=8), tmp_path / "fast",
        optimization="fast", fast_config=only_softmax,
    )
    exact = transpile(model, GenerationProgram(max_context=8), tmp_path / "exact")
    names = lambda result: {e[0] for e in load_sprite(result.path)["variables"].values()}
    assert "maximum" in names(exact) and "maximum" not in names(fast)
    assert fast.block_count < exact.block_count
    emulator = ScratchEmulator(load_sprite(fast.path))
    prefill_and_decode(emulator, model, [1, 2, 3], [4, 5])


def test_layer_sharing_banks_the_caches(tmp_path):
    torch.manual_seed(8)
    model = CachedLM(context=8, layers=3).eval()
    shared = transpile(
        model, GenerationProgram(max_context=8), tmp_path / "shared",
        codegen=CodegenConfig(layer_sharing="auto", unrolling="compact"),
    )
    plain = transpile(
        model, GenerationProgram(max_context=8), tmp_path / "plain",
        codegen=CodegenConfig(layer_sharing="off", unrolling="compact"),
    )
    assert shared.block_count < plain.block_count * 0.8
    emulator = ScratchEmulator(load_sprite(shared.path))
    assert "_cattorch shared transformer layer" in emulator._procedures
    assert "_cattorch K cache" in emulator.lists
    prefill_and_decode(emulator, model, [1, 3, 5], [2, 4])


def test_layer_sharing_falls_back_for_models_without_block_layers(tmp_path):
    # Hugging Face layers live in `model.layers`, not `blocks.N`.
    hf = tiny_decoder("gpt2").eval()
    result = transpile(
        hf, GenerationProgram(max_context=8), tmp_path / "hf",
        codegen=CodegenConfig(layer_sharing="auto"),
    )
    emulator = ScratchEmulator(load_sprite(result.path))
    assert "_cattorch shared transformer layer" not in emulator._procedures
    prefill_and_decode(emulator, hf, [1, 2, 3], [4])


def test_unfactorable_layers_retry_without_layer_sharing(tmp_path, caplog):
    model = nano()
    # A different MLP width in the second layer defeats the shared procedure.
    model.blocks[1].mlp = torch.nn.Sequential(
        torch.nn.Linear(16, 24), torch.nn.GELU(), torch.nn.Linear(24, 16),
    )
    model.eval()
    with caplog.at_level("WARNING", logger="cattorch"):
        emulator, _ = emulator_for(
            model, tmp_path, max_context=8, codegen=CodegenConfig(layer_sharing="auto"),
        )
    assert "exporting without layer sharing" in caplog.text
    assert "_cattorch shared transformer layer" not in emulator._procedures
    # Per-layer caches again: the retry does not bank them.
    assert "_cattorch K cache 1" in emulator.lists and "_cattorch K cache" not in emulator.lists
    prefill_and_decode(emulator, model, [1, 2, 3], [4, 5])


def test_int8_storage_and_quantization_cover_buffer_weights(tmp_path):
    torch.manual_seed(9)
    model = CachedLM(context=8, layers=2).eval()
    quantized = transpile(
        model, GenerationProgram(max_context=8), tmp_path / "q8",
        quantization=QuantizationConfig(bits=8, min_quantized_values=16),
    )
    plain = transpile(model, GenerationProgram(max_context=8), tmp_path / "f32")
    assert quantized.quantization is not None and quantized.quantization.tensors
    assert quantized.archive_bytes < plain.archive_bytes
    sprite = load_sprite(quantized.path)
    storage = {
        entry[0]: entry[1] for entry in sprite["variables"].values()
        if entry[0].startswith("cattorch storage precision")
    }
    assert storage["cattorch storage precision"] == "int8"


def test_module_paths_survive_into_the_traced_step():
    from cattorch.autoregressive import GenerationStepModule, build_generation_step
    from cattorch.frontend import capture_model
    import copy
    model = nano()
    step = GenerationStepModule(build_generation_step(model, 8))
    captured = capture_model(copy.deepcopy(step), (torch.zeros(1, 1, dtype=torch.long),))
    paths = {
        entry[0]
        for node in captured.graph.nodes
        for entry in node.meta.get("nn_module_stack", {}).values()
    }
    assert {"blocks.0.mlp", "blocks.1.q"} <= paths


class WithMethod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = nano()

    def forward(self, tokens):
        raise RuntimeError("not the exported method")

    def decode(self, tokens):
        return self.inner(tokens)


def test_exports_a_named_method(tmp_path):
    model = WithMethod().eval()
    emulator, _ = emulator_for(model, tmp_path, max_context=8, method="decode")
    expected = model.inner
    prefill_and_decode(emulator, expected, [1, 2], [3])


class LongContextLM(torch.nn.Module):
    """One attention layer with learned positions; caches outgrow one list."""

    def __init__(self, context, width=64):
        super().__init__()
        self.wte = torch.nn.Embedding(32, width)
        self.wpe = torch.nn.Embedding(context, width)
        self.q = torch.nn.Linear(width, width)
        self.k = torch.nn.Linear(width, width)
        self.v = torch.nn.Linear(width, width)
        self.head = torch.nn.Linear(width, 32, bias=False)

    def forward(self, tokens):
        x = self.wte(tokens) + self.wpe(torch.arange(tokens.shape[1]))
        x = x + torch.nn.functional.scaled_dot_product_attention(
            self.q(x), self.k(x), self.v(x), is_causal=True,
        )
        return self.head(x)


def test_long_context_shards_the_caches_and_tables(tmp_path, small_list_limit):
    torch.manual_seed(11)
    small_list_limit(2_000)
    context = 48  # 48 rows x 64 = 3,072 items, above the lowered list limit
    model = LongContextLM(context).eval()
    result = transpile(model, GenerationProgram(max_context=context), tmp_path / "long")
    assert "_cattorch K cache 1" in result.sharded_lists
    assert "_cattorch V cache 1" in result.sharded_lists
    emulator = ScratchEmulator(load_sprite(result.path))
    prefill_and_decode(emulator, model, [3, 1, 4], [1])
    assert len(emulator.lists["_cattorch K cache 1"]) == 4 * 64


def test_generation_program_signature_and_validation():
    program = GenerationProgram(64)
    assert (program.method, program.max_context) == ("forward", 64)
    assert GenerationProgram(max_context=32, method="decode").method == "decode"
    with pytest.raises(TypeError):
        GenerationProgram()
    with pytest.raises(TypeError):
        GenerationProgram("forward", torch.tensor([[1]]), 64)
    with pytest.raises(TypeError, match="keyword-only"):
        GenerationProgram("forward")
    with pytest.raises(ValueError, match="positive"):
        GenerationProgram(max_context=0)


def test_non_causal_model_is_rejected_at_export(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(2)
    bert = transformers.BertForMaskedLM(transformers.BertConfig(
        vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=32,
    )).eval()
    with pytest.raises(UnsupportedModelError, match="isn't causal"):
        transpile(bert, GenerationProgram(max_context=8), tmp_path / "bert")
