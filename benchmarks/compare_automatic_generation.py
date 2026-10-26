#!/usr/bin/env python3
"""Compare automatic and legacy cached generation on the TinyStories model.

The legacy path needs a model written for it (``CachedMQAExportModel``: one
token per call, a combined QKV linear, positions read from the cache length).
The automatic path takes an ordinary stateless ``forward(tokens) -> logits``.
This script builds a stateless twin with the same weights, exports both with
the production settings (int8 storage, compact code), and reports prefill and
decode opcode counts and sprite size measured in the Python emulator, and
whether the two give the same logits.

Run from the repository root::

    python benchmarks/compare_automatic_generation.py [--prompt-tokens 8]

The trained checkpoint is read from the sibling ``catgpt2`` checkout. If it is
missing, a randomly initialised model with the same dimensions is used and the
report says so.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import types
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CATGPT2 = ROOT.parent / "catgpt2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CATGPT2 / "src"))

try:  # The tokenizer import is the only use of sentencepiece we would hit.
    import sentencepiece  # noqa: F401
except ImportError:
    sys.modules["sentencepiece"] = types.ModuleType("sentencepiece")

from benchmarks.catgpt2_approx_experiment import (  # noqa: E402
    CachedMQAExportAttention, CachedMQAExportModel,
)
from cattorch import (  # noqa: E402
    CodegenConfig, GenerationProgram, StorageConfig, rotary_embedding, transpile,
)
from cattorch.util.scratch.emulator import ScratchEmulator  # noqa: E402

CHECKPOINT = (
    CATGPT2 / "runs" / "tinystories_1200k_4k_w112_d6_ctx256_short224_stage3" / "best.pt"
)


class StatelessMQAExportAttention(CachedMQAExportAttention):
    """The cached attention's weights behind a plain ``forward(x)`` over all rows."""

    def __init__(self, source, context):
        super().__init__(source, context)
        del self.position, self.causal_mask
        self.register_buffer("cos", self.rope_cos.weight.detach().clone()[None, None])
        self.register_buffer("sin", self.rope_sin.weight.detach().clone()[None, None])
        del self.rope_cos, self.rope_sin
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(context, context, dtype=torch.bool), diagonal=1),
        )

    def forward(self, value):
        batch, length, width = value.shape
        kv_width = self.kv_heads * self.head_dim
        query, key, projected_value = self.qkv(value).split(
            (width, kv_width, kv_width), dim=-1,
        )
        query = query.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        projected_value = projected_value.view(
            batch, length, self.kv_heads, self.head_dim,
        ).transpose(1, 2)
        cos, sin = self.cos[:, :, :length], self.sin[:, :, :length]
        query = rotary_embedding(query, cos, sin)
        key = rotary_embedding(key, cos, sin)
        if self.kv_heads not in (1, self.heads):
            groups = self.heads // self.kv_heads
            key = key.repeat_interleave(groups, dim=1)
            projected_value = projected_value.repeat_interleave(groups, dim=1)
        scores = (query @ key.transpose(-2, -1)) / self.head_dim ** 0.5
        scores = scores.masked_fill(self.causal_mask[:length, :length], float("-inf"))
        attended = F.softmax(scores, dim=-1) @ projected_value
        return self.o_proj(
            attended.transpose(1, 2).contiguous().view(batch, length, width)
        )


def load_source(checkpoint: Path):
    """The trained model, or a random one of the same size if there is no file."""
    from microchat.checkpoint import load_checkpoint
    if checkpoint.exists():
        source, _ = load_checkpoint(checkpoint, device="cpu")
        return source.eval(), f"trained weights ({checkpoint.name})"
    from microchat.model import MicroChat, ModelConfig
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=4096, block_size=256, n_layer=6, n_head=8, n_kv_head=8,
        n_embd=112, n_hidden=217, embedding_dim=112,
    )
    return MicroChat(config).eval(), "random weights at the same dimensions"


def load_sprite(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("sprite.json"))


def opcodes(emulator) -> int:
    return sum(emulator.opcode_counts.values())


def measure(path, prompt, decode):
    emulator = ScratchEmulator(load_sprite(path))
    emulator.run_procedure("cattorch init")
    start = opcodes(emulator)
    emulator.lists["cattorch tokens"] = list(prompt)
    emulator.run_procedure("cattorch prefill")
    assert emulator.variables["cattorch status"] == "ok"
    after_prefill = opcodes(emulator)
    logits = [torch.tensor(emulator.lists["cattorch logits"])]
    per_token = []
    for token in decode:
        before = opcodes(emulator)
        emulator.lists["cattorch tokens"] = [token]
        emulator.run_procedure("cattorch decode")
        assert emulator.variables["cattorch status"] == "ok"
        per_token.append(opcodes(emulator) - before)
        logits.append(torch.tensor(emulator.lists["cattorch logits"]))
    return after_prefill - start, per_token, logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=3)
    parser.add_argument(
        "--layer-sharing", choices=("auto", "off"), default="auto",
        help="'off' also exercises the final-layer hidden prefill skip",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    source, weights = load_source(args.checkpoint)
    context = args.context
    cached = CachedMQAExportModel(source, context).eval()
    stateless = CachedMQAExportModel(
        source, context, attention_type=StatelessMQAExportAttention,
    ).eval()

    storage = StorageConfig(precision="int8", group_size=64, scale_precision="float16")
    codegen = CodegenConfig(
        unrolling="compact", compact_internal_names=True, id_namespace="",
        layer_sharing=args.layer_sharing,
    )
    output = args.output_dir or Path(tempfile.mkdtemp(prefix="compare_generation_"))
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "legacy": transpile(
            cached,
            GenerationProgram("forward", torch.tensor([[1]]), context),
            output / "legacy", storage=storage, codegen=codegen,
        ),
        "automatic": transpile(
            stateless, GenerationProgram(max_context=context), output / "automatic",
            storage=storage, codegen=codegen,
        ),
    }

    generator = torch.Generator().manual_seed(1)
    vocabulary = source.config.vocab_size
    tokens = torch.randint(1, vocabulary, (args.prompt_tokens + args.decode_tokens,), generator=generator)
    prompt, decode = tokens[:args.prompt_tokens].tolist(), tokens[args.prompt_tokens:].tolist()
    with torch.no_grad():
        eager = [
            stateless(tokens[None, :length])[0, -1]
            for length in range(args.prompt_tokens, len(tokens) + 1)
        ]

    rows = {}
    for label, result in results.items():
        prefill, per_token, logits = measure(result.path, prompt, decode)
        rows[label] = dict(
            prefill=prefill, decode=per_token, logits=logits,
            blocks=result.block_count, bytes=result.archive_bytes,
        )

    legacy, auto = rows["legacy"], rows["automatic"]
    agree = max(float((a - b).abs().max()) for a, b in zip(auto["logits"], legacy["logits"]))
    drift = {
        label: max(float((a - b).abs().max()) for a, b in zip(row["logits"], eager))
        for label, row in rows.items()
    }
    print(f"weights: {weights}; context {context}; int8 storage, compact code, "
          f"layer sharing {args.layer_sharing}")
    print(f"prefill {args.prompt_tokens} tokens, then decode {args.decode_tokens}")
    print()
    print(f"{'':24}{'legacy':>12}{'automatic':>12}{'ratio':>9}")

    def line(name, a, b):
        print(f"{name:24}{a:>12,}{b:>12,}{b / a:>9.3f}")

    line("prefill opcodes", legacy["prefill"], auto["prefill"])
    for index, (a, b) in enumerate(zip(legacy["decode"], auto["decode"])):
        line(f"decode step {index + 1} opcodes", a, b)
    line("blocks", legacy["blocks"], auto["blocks"])
    line("archive bytes", legacy["bytes"], auto["bytes"])
    print()
    print(f"max |automatic - legacy| logits: {agree:.3g}")
    print(f"max |export - float eager| logits: legacy {drift['legacy']:.3g}, "
          f"automatic {drift['automatic']:.3g} (int8 rounding)")


if __name__ == "__main__":
    main()
