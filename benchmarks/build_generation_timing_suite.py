#!/usr/bin/env python3
"""Time legacy and automatic cached generation on the TinyStories model in Scratch.

The legacy recognizer stays until real Scratch shows the automatic path is as
fast (opcode counts already match in the emulator). This script exports the
same weights both ways with the production settings (int8 storage, compact
code, layer sharing) and puts the two sprites in one project. On the green
flag each sprite runs ``init``, a timed prefill, and timed decode steps, one
sprite after the other, and the stage list ``cattorch benchmark results`` gets
lines such as ``legacy prefill 32: 1.234``.

Run from the repository root::

    python benchmarks/build_generation_timing_suite.py [--kv-heads 1]

The trained checkpoint is read from the sibling ``catgpt2`` checkout. Without
it, a randomly initialised model of the same size is used; timings don't
depend on the weights.
"""

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
import types
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
CATGPT2 = ROOT.parent / "catgpt2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CATGPT2 / "src"))

try:
    import microchat.checkpoint  # noqa: F401
except ImportError:
    # catgpt2_approx_experiment imports this at module level; only the
    # trained checkpoint needs it.
    stub = types.ModuleType("microchat.checkpoint")
    stub.load_checkpoint = stub.load_checkpoint_tokenizer = None
    sys.modules.setdefault("microchat", types.ModuleType("microchat"))
    sys.modules["microchat.checkpoint"] = stub

from benchmarks.compare_automatic_generation import (  # noqa: E402
    CHECKPOINT, StatelessMQAExportAttention,
)
from benchmarks.catgpt2_approx_experiment import CachedMQAExportModel  # noqa: E402
from cattorch import CodegenConfig, GenerationProgram, StorageConfig, transpile  # noqa: E402
from cattorch.benchmark import (  # noqa: E402
    SUITE_RESULTS, _add_broadcast_entry, _load_sprite, _stage_target, _StageBlocks,
    _write_project,
)
from cattorch.transpiler import _add_warp_procedure, _merge_lists_by_name  # noqa: E402
from cattorch.util.scratch.dsl import Program, append, call, clear  # noqa: E402
from cattorch.util.scratch.emulator import ScratchEmulator  # noqa: E402

OUTPUT = ROOT / "benchmarks" / "artifacts"


class _SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, value):
        return self.down(nn.functional.silu(self.gate(value)) * self.up(value))


def _random_source(kv_heads: int, context: int) -> nn.Module:
    """A stand-in with the TinyStories checkpoint's sizes and attribute names."""
    width, heads, hidden, layers, vocabulary = 112, 8, 217, 6, 4096
    head_dim = width // heads
    angles = torch.outer(
        torch.arange(context, dtype=torch.float32),
        10000.0 ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim),
    ).repeat_interleave(2, dim=-1)

    def block():
        attn = nn.Module()
        attn.n_head, attn.n_kv_head, attn.head_dim = heads, kv_heads, head_dim
        attn.q_proj = nn.Linear(width, width, bias=False)
        attn.k_proj = nn.Linear(width, kv_heads * head_dim, bias=False)
        attn.v_proj = nn.Linear(width, kv_heads * head_dim, bias=False)
        attn.o_proj = nn.Linear(width, width, bias=False)
        attn.rope = types.SimpleNamespace(
            cos=angles.cos()[None, None], sin=angles.sin()[None, None],
        )
        result = nn.Module()
        result.attn_norm, result.attn = nn.RMSNorm(width), attn
        result.ffn_norm, result.ffn = nn.RMSNorm(width), _SwiGLU(width, hidden)
        return result

    torch.manual_seed(0)
    source = nn.Module()
    source.token_embedding = nn.Embedding(vocabulary, width)
    source.input_projection = nn.Identity()
    source.blocks = nn.ModuleList(block() for _ in range(layers))
    source.final_norm = nn.RMSNorm(width)
    source.lm_head = nn.Linear(width, vocabulary, bias=False)
    source.config = types.SimpleNamespace(vocab_size=vocabulary, block_size=context)
    return source.eval()


def load_source(checkpoint: Path, kv_heads: int, context: int):
    if checkpoint.exists() and sys.modules["microchat.checkpoint"].load_checkpoint:
        from microchat.checkpoint import load_checkpoint
        source, _ = load_checkpoint(checkpoint, device="cpu")
        return source.eval(), f"trained weights ({checkpoint.name})"
    return _random_source(kv_heads, context), (
        f"random weights at the checkpoint's size, {kv_heads} KV heads"
    )


def _timed_sprite(path: Path, name: str, decode_token: int, prompt: list[int]):
    sprite, assets = _load_sprite(path)
    for entry in sprite["lists"].values():
        if entry[0] == "cattorch tokens":
            entry[1] = list(prompt)
    _add_warp_procedure(
        sprite,
        "cattorch benchmark decode",
        Program(
            "timing_benchmark_decode",
            lists=("cattorch tokens",),
            body=(
                clear("cattorch tokens"),
                append("cattorch tokens", decode_token),
                call("cattorch decode"),
            ),
        ),
        x=900,
        y=0,
    )
    _merge_lists_by_name(sprite, {"cattorch tokens"})
    sprite["name"] = name
    sprite["visible"] = False
    broadcasts = {}
    for step, procedure in (
        ("init", "cattorch init"),
        ("prefill", "cattorch prefill"),
        ("decode", "cattorch benchmark decode"),
    ):
        message = f"{name} {step}"
        broadcasts[message] = f"timing_{name}_{step}"
        _add_broadcast_entry(sprite, message, broadcasts[message], procedure)
    _check_sprite(sprite, name)
    return sprite, assets, broadcasts


def _check_sprite(sprite: dict, name: str) -> None:
    """Run one prefill and one wrapped decode in the emulator; both must succeed."""
    emulator = ScratchEmulator(copy.deepcopy(sprite))
    for procedure in ("cattorch init", "cattorch prefill", "cattorch benchmark decode"):
        emulator.run_procedure(procedure)
        status = emulator.variables["cattorch status"]
        if procedure != "cattorch init" and status != "ok":
            raise RuntimeError(f"{name}: {procedure} left status {status!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--kv-heads", type=int, default=8,
                        help="KV heads of the random stand-in model")
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--layer-sharing", choices=("auto", "off"), default="auto")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    source, weights = load_source(args.checkpoint, args.kv_heads, args.context)
    context = args.context
    models = {
        "legacy": (
            CachedMQAExportModel(source, context).eval(),
            GenerationProgram("forward", torch.tensor([[1]]), context),
        ),
        "automatic": (
            CachedMQAExportModel(
                source, context, attention_type=StatelessMQAExportAttention,
            ).eval(),
            GenerationProgram(max_context=context),
        ),
    }
    storage = StorageConfig(precision="int8", group_size=64, scale_precision="float16")
    codegen = CodegenConfig(
        unrolling="compact", compact_internal_names=True, id_namespace="",
        layer_sharing=args.layer_sharing,
    )
    generator = torch.Generator().manual_seed(1)
    prompt = torch.randint(
        1, source.config.vocab_size, (args.prompt_tokens + 1,), generator=generator,
    ).tolist()
    prompt, decode_token = prompt[:-1], prompt[-1]

    sprites, assets, broadcasts = [], {}, {}
    result_id = "timing_results"
    builder = _StageBlocks({}, {SUITE_RESULTS: result_id})
    steps = [("clear", SUITE_RESULTS)]
    with tempfile.TemporaryDirectory() as directory:
        for label, (model, program) in models.items():
            result = transpile(
                model, program, Path(directory) / label, storage=storage, codegen=codegen,
            )
            print(f"{label}: {result.block_count:,} blocks, {result.archive_bytes:,} bytes")
            sprite, sprite_assets, sprite_broadcasts = _timed_sprite(
                result.path, label, decode_token, prompt,
            )
            sprite["layerOrder"] = len(sprites) + 1
            sprites.append(sprite)
            assets.update(sprite_assets)
            broadcasts.update(sprite_broadcasts)
            steps += [
                ("broadcast", f"{label} init"),
                ("reset_timer",),
                ("broadcast", f"{label} prefill"),
                ("record_result", SUITE_RESULTS, f"{label} prefill {len(prompt)}"),
                ("reset_timer",),
                ("repeat", args.decode_tokens, (("broadcast", f"{label} decode"),)),
                ("record_result", SUITE_RESULTS, f"{label} decode {args.decode_tokens}"),
            ]
    builder.broadcast_ids = broadcasts
    hat = builder._id()
    first, _ = builder._chain(steps, hat)
    builder.blocks[hat] = {
        "opcode": "event_whenflagclicked", "next": first, "parent": None,
        "inputs": {}, "fields": {}, "shadow": False, "topLevel": True, "x": 0, "y": 0,
    }
    stage, md5ext, stage_bytes = _stage_target(
        builder.blocks, broadcasts, {result_id: [SUITE_RESULTS, []]},
    )
    monitor = {
        "id": result_id, "mode": "list", "opcode": "data_listcontents",
        "params": {"LIST": SUITE_RESULTS}, "spriteName": None, "value": [],
        "width": 460, "height": 180, "x": 10, "y": 10, "visible": True,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output = args.output or OUTPUT / f"generation_timing_kv{args.kv_heads}.sb3"
    _write_project(
        output, stage, sprites, assets, (md5ext, stage_bytes),
        "Generation timing", monitors=[monitor],
    )
    print(f"{weights}; context {context}, prompt {len(prompt)}, "
          f"decode {args.decode_tokens}, layer sharing {args.layer_sharing}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
