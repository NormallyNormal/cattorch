"""Cached generation with sliding windows, ALiBi biases and attention sinks."""

from __future__ import annotations

import json
import math
import sys
import zipfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cattorch import CodegenConfig, GenerationProgram, transpile
from cattorch.util.scratch.emulator import ScratchEmulator

sys.path.insert(0, str(Path(__file__).parent))
from test_generation_automatic import load_sprite, nano, prefill_and_decode  # noqa: E402

CONTEXT, VOCAB, WIDTH, HEADS = 16, 24, 16, 2


class MaskedLM(nn.Module):
    """One attention layer whose mask or bias is a registered buffer."""

    def __init__(self, kind, window=3):
        super().__init__()
        self.kind = kind
        self.wte = nn.Embedding(VOCAB, WIDTH)
        self.wpe = nn.Embedding(CONTEXT, WIDTH)
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)
        self.head = nn.Linear(WIDTH, VOCAB)
        index = torch.arange(CONTEXT)
        allowed = index[None, :] <= index[:, None]
        if kind == "window":
            allowed &= index[:, None] - index[None, :] < window
        self.register_buffer("blocked", ~allowed)
        slopes = torch.tensor([0.5, 0.25])
        # ALiBi as BLOOM writes it: slope * key, which equals slope *
        # (key - query) up to a per-row constant.
        self.register_buffer("alibi", slopes[:, None, None] * index[None, None, :].float())
        self.sinks = nn.Parameter(torch.tensor([0.7, -0.4]))

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.wte(tokens)
        if self.kind != "alibi":
            x = x + self.wpe(torch.arange(length))
        q, k, v = (
            part.view(1, length, HEADS, -1).transpose(1, 2)
            for part in self.qkv(x).chunk(3, dim=-1)
        )
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        if self.kind == "alibi":
            scores = scores + self.alibi[:, :, :length]
        scores = scores.masked_fill(self.blocked[:length, :length], float("-inf"))
        if self.kind == "sinks":
            sinks = self.sinks.reshape(1, -1, 1, 1).expand(1, -1, length, 1)
            logits = torch.cat([scores, sinks], dim=-1)
            logits = logits - logits.max(dim=-1, keepdim=True).values
            weights = F.softmax(logits, dim=-1)[..., :-1]
        else:
            weights = F.softmax(scores, dim=-1)
        attended = (weights @ v).transpose(1, 2).reshape(1, length, -1)
        return self.head(x + attended)


@torch.no_grad()
@pytest.mark.parametrize("kind", ["window", "alibi", "sinks"])
def test_masks_biases_and_sinks_generate_with_a_cache(tmp_path, kind):
    torch.manual_seed(0)
    model = MaskedLM(kind).eval()
    result = transpile(model, GenerationProgram(max_context=12), tmp_path / kind)
    # Decoding past the window shows that old keys drop out.
    prefill_and_decode(ScratchEmulator(load_sprite(result.path)), model, [3, 1, 4, 1], [5, 9, 2])


@torch.no_grad()
def test_alibi_with_the_fast_kernel(tmp_path):
    torch.manual_seed(1)
    model = MaskedLM("alibi").eval()
    result = transpile(
        model, GenerationProgram(max_context=12), tmp_path / "fast", optimization="fast",
    )
    prefill_and_decode(ScratchEmulator(load_sprite(result.path)), model, [3, 1, 4], [1, 5])


@torch.no_grad()
def test_huggingface_sliding_window_model_generates(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(2)
    config = transformers.MistralConfig(
        vocab_size=64, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32,
        sliding_window=3,
    )
    model = transformers.AutoModelForCausalLM.from_config(config).eval()
    result = transpile(model, GenerationProgram(max_context=10), tmp_path / "mistral")
    with zipfile.ZipFile(result.path) as archive:
        emulator = ScratchEmulator(json.loads(archive.read("sprite.json")))
    prefill_and_decode(emulator, model, [3, 17, 5], [40, 9, 2])


def _project_output_guards(sprite) -> int:
    """``if cattorch project output = 1`` blocks."""
    blocks = sprite["blocks"]
    count = 0
    for block in blocks.values():
        if not isinstance(block, dict) or block.get("opcode") != "control_if":
            continue
        condition = blocks.get(block["inputs"].get("CONDITION", [None, None])[1])
        operand = (condition or {}).get("inputs", {}).get("OPERAND1", [None, None])[1]
        if isinstance(operand, list) and operand[1] == "cattorch project output":
            count += 1
    return count


@torch.no_grad()
def test_hidden_prefill_skips_the_final_query_projection(tmp_path, monkeypatch):
    import cattorch.transpiler as transpiler

    model = nano()
    options = dict(codegen=CodegenConfig(layer_sharing="off"))
    guarded = load_sprite(transpile(
        model, GenerationProgram(max_context=12), tmp_path / "hidden", **options,
    ).path)
    monkeypatch.setattr(transpiler, "_query_only_nodes", lambda graph, attention: set())
    unguarded = load_sprite(transpile(
        model, GenerationProgram(max_context=12), tmp_path / "plain", **options,
    ).path)
    assert _project_output_guards(guarded) > _project_output_guards(unguarded)

    def prefill_cost(sprite):
        emulator = ScratchEmulator(sprite)
        emulator.run_procedure("cattorch init")
        before = sum(emulator.opcode_counts.values())
        emulator.lists["cattorch tokens"] = [3, 1, 4, 1, 5]
        emulator.run_procedure("cattorch prefill")
        return sum(emulator.opcode_counts.values()) - before

    assert prefill_cost(guarded) < prefill_cost(unguarded)
    prefill_and_decode(ScratchEmulator(guarded), model, [3, 1, 4, 1, 5], [9])
