"""Evaluate and VM-benchmark CatGPT2 low-rank/sparse Scratch exports."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from cattorch import rotary_embedding


ROOT = Path(__file__).resolve().parents[1]
CATGPT2 = ROOT.parent / "catgpt2"
CHECKPOINT = CATGPT2 / "artifacts" / "catgpt-generalized-998k.pt"
VALIDATION = CATGPT2 / "data" / "catgpt_v3_generalizer" / "val.jsonl"
OUTPUT = ROOT / "benchmarks" / "artifacts" / "catgpt2_approx"
sys.path.insert(0, str(CATGPT2 / "src"))

from microchat.checkpoint import load_checkpoint, load_checkpoint_tokenizer

from cattorch import (
    FastConfig,
    FastLayerConfig,
    GenerationProgram,
    StorageConfig,
    transpile,
)
from cattorch.fast import prepare_fast_model
from cattorch.benchmark import (
    SUITE_RESULTS,
    _StageBlocks,
    _add_broadcast_entry,
    _load_sprite,
    _set_inputs,
    _stage_target,
    _write_project,
)
from cattorch.transpiler import _add_warp_procedure, _merge_lists_by_name
from cattorch.util.scratch.dsl import Program, append, call, clear


def policies() -> dict[str, FastConfig]:
    exact_arithmetic = dict(activations=False, layer_norm=False, softmax=False)
    body_names = [
        f"blocks.{layer}.{name}"
        for layer in range(3)
        for name in (
            "attn.qkv", "attn.o_proj", "ffn.gate", "ffn.up", "ffn.down",
        )
    ]
    ffn_names = [
        f"blocks.{layer}.ffn.{name}"
        for layer in range(3)
        for name in ("gate", "up", "down")
    ]
    qkv_names = [f"blocks.{layer}.attn.qkv" for layer in range(3)]
    return {
        "exact": FastConfig(**exact_arithmetic),
        "head_rank112": FastConfig(
            **exact_arithmetic,
            overrides={"lm_head": FastLayerConfig(rank=112)},
        ),
        "head_rank96": FastConfig(
            **exact_arithmetic,
            overrides={"lm_head": FastLayerConfig(rank=96)},
        ),
        "head_rank80": FastConfig(
            **exact_arithmetic,
            overrides={"lm_head": FastLayerConfig(rank=80)},
        ),
        "head_rank64": FastConfig(
            **exact_arithmetic,
            overrides={"lm_head": FastLayerConfig(rank=64)},
        ),
        "rank75": FastConfig(
            **exact_arithmetic, weights=FastLayerConfig(rank_ratio=0.75),
        ),
        "rank50": FastConfig(
            **exact_arithmetic, weights=FastLayerConfig(rank_ratio=0.50),
        ),
        "rank25": FastConfig(
            **exact_arithmetic, weights=FastLayerConfig(rank_ratio=0.25),
        ),
        "sparse25": FastConfig(
            **exact_arithmetic, weights=FastLayerConfig(pruning=0.25),
        ),
        "sparse50": FastConfig(
            **exact_arithmetic, weights=FastLayerConfig(pruning=0.50),
        ),
        "sparse75": FastConfig(
            **exact_arithmetic, weights=FastLayerConfig(pruning=0.75),
        ),
        "body_sparse25": FastConfig(
            **exact_arithmetic,
            overrides={name: FastLayerConfig(pruning=0.25) for name in body_names},
        ),
        "body_sparse50": FastConfig(
            **exact_arithmetic,
            overrides={name: FastLayerConfig(pruning=0.50) for name in body_names},
        ),
        "ffn_sparse25": FastConfig(
            **exact_arithmetic,
            overrides={name: FastLayerConfig(pruning=0.25) for name in ffn_names},
        ),
        "qkv_sparse25": FastConfig(
            **exact_arithmetic,
            overrides={name: FastLayerConfig(pruning=0.25) for name in qkv_names},
        ),
        "ffn_rank80": FastConfig(
            **exact_arithmetic,
            overrides={name: FastLayerConfig(rank=80) for name in ffn_names},
        ),
        "ffn_rank64": FastConfig(
            **exact_arithmetic,
            overrides={name: FastLayerConfig(rank=64) for name in ffn_names},
        ),
        "qkv_rank80": FastConfig(
            **exact_arithmetic,
            overrides={name: FastLayerConfig(rank=80) for name in qkv_names},
        ),
        "qkv_rank64": FastConfig(
            **exact_arithmetic,
            overrides={name: FastLayerConfig(rank=64) for name in qkv_names},
        ),
    }


class ExportAttention(nn.Module):
    """Cattorch-facing, exactly equivalent expansion of trained MQA."""

    def __init__(self, source: nn.Module, context: int):
        super().__init__()
        self.heads = source.n_head
        self.head_dim = source.head_dim
        width = self.heads * self.head_dim
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        repeats = source.n_head // source.n_kv_head
        with torch.no_grad():
            self.qkv.weight.copy_(torch.cat((
                source.q_proj.weight,
                source.k_proj.weight.repeat(repeats, 1),
                source.v_proj.weight.repeat(repeats, 1),
            )))
        self.o_proj = copy.deepcopy(source.o_proj)
        self.register_buffer("cos", source.rope.cos[:, :, :context].clone())
        self.register_buffer("sin", source.rope.sin[:, :, :context].clone())
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(context, context, dtype=torch.bool), diagonal=1),
        )

    def forward(self, value):
        batch, length, width = value.shape
        query, key, projected_value = self.qkv(value).chunk(3, dim=-1)
        query = query.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        projected_value = projected_value.view(
            batch, length, self.heads, self.head_dim,
        ).transpose(1, 2)
        query = rotary_embedding(
            query, self.cos[:, :, :length], self.sin[:, :, :length],
        )
        key = rotary_embedding(
            key, self.cos[:, :, :length], self.sin[:, :, :length],
        )
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:length, :length], float("-inf"))
        attended = F.softmax(scores, dim=-1) @ projected_value
        return self.o_proj(
            attended.transpose(1, 2).contiguous().view(batch, length, width)
        )


class CachedMQAExportAttention(nn.Module):
    """Single-token MQA whose RoPE position is supplied by cache length in Scratch."""

    def __init__(self, source: nn.Module, context: int):
        super().__init__()
        self.heads = source.n_head
        self.kv_heads = source.n_kv_head
        self.head_dim = source.head_dim
        width = self.heads * self.head_dim
        kv_width = self.kv_heads * self.head_dim
        self.qkv = nn.Linear(width, width + 2 * kv_width, bias=False)
        with torch.no_grad():
            self.qkv.weight.copy_(torch.cat((
                source.q_proj.weight,
                source.k_proj.weight,
                source.v_proj.weight,
            )))
        self.o_proj = copy.deepcopy(source.o_proj)
        self.rope_cos = nn.Embedding.from_pretrained(
            source.rope.cos[0, 0, :context].clone(), freeze=True,
        )
        self.rope_sin = nn.Embedding.from_pretrained(
            source.rope.sin[0, 0, :context].clone(), freeze=True,
        )
        # The generation lowerer replaces this zero with the current cache length.
        self.register_buffer("position", torch.zeros(1, dtype=torch.long))
        self.register_buffer("causal_mask", torch.zeros(1, 1, dtype=torch.bool))

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
        cos = self.rope_cos(self.position).view(1, 1, 1, self.head_dim)
        sin = self.rope_sin(self.position).view(1, 1, 1, self.head_dim)
        query = rotary_embedding(query, cos, sin)
        key = rotary_embedding(key, cos, sin)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask, float("-inf"))
        attended = F.softmax(scores, dim=-1) @ projected_value
        return self.o_proj(
            attended.transpose(1, 2).contiguous().view(batch, length, width)
        )


class CachedExpandedExportAttention(CachedMQAExportAttention):
    """Cached control that physically duplicates the trained MQA K/V heads."""

    def __init__(self, source: nn.Module, context: int):
        super().__init__(source, context)
        self.kv_heads = self.heads
        width = self.heads * self.head_dim
        repeats = self.heads // source.n_kv_head
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        with torch.no_grad():
            self.qkv.weight.copy_(torch.cat((
                source.q_proj.weight,
                source.k_proj.weight.repeat(repeats, 1),
                source.v_proj.weight.repeat(repeats, 1),
            )))


class ExportBlock(nn.Module):
    def __init__(self, source: nn.Module, context: int):
        super().__init__()
        self.attn_norm = copy.deepcopy(source.attn_norm)
        self.attn = ExportAttention(source.attn, context)
        self.ffn_norm = copy.deepcopy(source.ffn_norm)
        self.ffn = copy.deepcopy(source.ffn)

    def forward(self, value):
        value = value + self.attn(self.attn_norm(value))
        return value + self.ffn(self.ffn_norm(value))


class CachedMQAExportBlock(ExportBlock):
    def __init__(
        self,
        source: nn.Module,
        context: int,
        attention_type=CachedMQAExportAttention,
    ):
        nn.Module.__init__(self)
        self.attn_norm = copy.deepcopy(source.attn_norm)
        self.attn = attention_type(source.attn, context)
        self.ffn_norm = copy.deepcopy(source.ffn_norm)
        self.ffn = copy.deepcopy(source.ffn)


class ExportModel(nn.Module):
    def __init__(self, source: nn.Module, context: int, *, last_only: bool):
        super().__init__()
        self.context = context
        self.last_only = last_only
        self.token_embedding = copy.deepcopy(source.token_embedding)
        self.input_projection = copy.deepcopy(source.input_projection)
        self.blocks = nn.ModuleList(ExportBlock(block, context) for block in source.blocks)
        self.final_norm = copy.deepcopy(source.final_norm)
        self.lm_head = copy.deepcopy(source.lm_head)
        # Preserve the trained input/output tie in the exact wrapper.
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens):
        value = self.input_projection(self.token_embedding(tokens))
        for block in self.blocks:
            value = block(value)
        value = self.final_norm(value)
        if self.last_only:
            value = value[:, self.context - 1:self.context]
        if isinstance(self.input_projection, nn.Linear):
            value = value @ self.input_projection.weight
        return self.lm_head(value)


class CachedMQAExportModel(nn.Module):
    def __init__(
        self,
        source: nn.Module,
        context: int,
        attention_type=CachedMQAExportAttention,
    ):
        super().__init__()
        self.token_embedding = copy.deepcopy(source.token_embedding)
        self.input_projection = copy.deepcopy(source.input_projection)
        self.blocks = nn.ModuleList(
            CachedMQAExportBlock(block, context, attention_type)
            for block in source.blocks
        )
        self.final_norm = copy.deepcopy(source.final_norm)
        self.lm_head = copy.deepcopy(source.lm_head)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens):
        value = self.input_projection(self.token_embedding(tokens))
        for block in self.blocks:
            value = block(value)
        value = self.final_norm(value)
        if isinstance(self.input_projection, nn.Linear):
            value = value @ self.input_projection.weight
        return self.lm_head(value)


def load_export_model(context: int, *, last_only: bool):
    source, payload = load_checkpoint(CHECKPOINT)
    source.eval()
    result = ExportModel(source, context, last_only=last_only).eval()
    return source, result, payload


def load_cached_mqa_export_model(context: int):
    source, payload = load_checkpoint(CHECKPOINT)
    source.eval()
    return source, CachedMQAExportModel(source, context).eval(), payload


def _validation_rows(limit: int) -> list[dict]:
    rows = [json.loads(line) for line in VALIDATION.read_text().splitlines() if line]
    selected = []
    counts = defaultdict(int)
    per_category = max(1, limit // len({row["category"] for row in rows}))
    for row in rows:
        if counts[row["category"]] >= per_category:
            continue
        selected.append(row)
        counts[row["category"]] += 1
        if len(selected) >= limit:
            break
    return selected


def _round_static_f16(model: nn.Module) -> nn.Module:
    result = copy.deepcopy(model)
    with torch.no_grad():
        for value in (*result.parameters(), *result.buffers()):
            if value.is_floating_point():
                value.copy_(value.half().float())
    return result.eval()


def evaluate(limit: int = 120, context: int = 128) -> dict:
    _, exact_model, payload = load_export_model(context, last_only=False)
    tokenizer = load_checkpoint_tokenizer(payload)
    rows = _validation_rows(limit)
    samples = []
    for row in rows:
        tokens, mask = tokenizer.format_sft(row["messages"])
        tokens = tokens[-(context + 1):]
        mask = mask[-(context + 1):]
        samples.append((tokens[:-1], tokens[1:], mask[1:]))

    variants = {"exact": exact_model}
    for name, policy in policies().items():
        if name != "exact":
            variants[name] = prepare_fast_model(exact_model, policy).eval()
    variants["exact_f16_storage"] = _round_static_f16(exact_model)
    variants["qkv_rank80_f16_storage"] = _round_static_f16(
        variants["qkv_rank80"]
    )

    stats = {
        name: {"loss_sum": 0.0, "correct": 0, "tokens": 0, "agree": 0}
        for name in variants
    }
    batch_size = 8
    for start in range(0, len(samples), batch_size):
        group = samples[start:start + batch_size]
        width = max(len(item[0]) for item in group)
        inputs = torch.zeros(len(group), width, dtype=torch.long)
        targets = torch.full((len(group), width), -100, dtype=torch.long)
        selected = torch.zeros(len(group), width, dtype=torch.bool)
        for row_index, (source, target, mask) in enumerate(group):
            inputs[row_index, :len(source)] = torch.tensor(source)
            targets[row_index, :len(target)] = torch.tensor(target)
            selected[row_index, :len(mask)] = torch.tensor(mask)
        selected &= targets != -100
        with torch.inference_mode():
            exact_logits = variants["exact"](inputs)
            exact_predictions = exact_logits.argmax(-1)
            for name, model in variants.items():
                logits = exact_logits if name == "exact" else model(inputs)
                losses = F.cross_entropy(
                    logits.flatten(0, 1), targets.flatten(), reduction="none", ignore_index=-100,
                ).view_as(targets)
                predictions = logits.argmax(-1)
                count = int(selected.sum())
                stats[name]["loss_sum"] += float(losses[selected].sum())
                stats[name]["correct"] += int((predictions[selected] == targets[selected]).sum())
                stats[name]["agree"] += int((predictions[selected] == exact_predictions[selected]).sum())
                stats[name]["tokens"] += count

    baseline_loss = stats["exact"]["loss_sum"] / stats["exact"]["tokens"]
    results = {}
    for name, values in stats.items():
        count = values["tokens"]
        loss = values["loss_sum"] / count
        results[name] = {
            "assistant_tokens": count,
            "cross_entropy": loss,
            "perplexity": math.exp(loss),
            "loss_increase": loss - baseline_loss,
            "target_token_accuracy": values["correct"] / count,
            "top1_agreement_with_exact": values["agree"] / count,
        }
    report = {
        "checkpoint": str(CHECKPOINT),
        "validation": str(VALIDATION),
        "examples": len(samples),
        "context": context,
        "results": results,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "accuracy.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _build_one_project(model, tokens, name: str, output: Path, iterations: int) -> Path:
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory) / name
        transpile(
            model,
            tokens,
            str(base),
            optimization="exact",
            storage=StorageConfig(precision="float16"),
        )
        sprite_path = base.with_suffix(".sprite3")
        standalone_size = sprite_path.stat().st_size
        sprite, assets = _load_sprite(sprite_path)
    _set_inputs(sprite, (tokens,))
    sprite["name"] = name
    sprite["visible"] = False
    sprite["layerOrder"] = 1
    broadcasts = {
        f"{name} init": f"{name}_init",
        f"{name} forward": f"{name}_forward",
    }
    _add_broadcast_entry(sprite, f"{name} init", broadcasts[f"{name} init"], "cattorch init")
    _add_broadcast_entry(sprite, f"{name} forward", broadcasts[f"{name} forward"], "cattorch forward")
    result_id = "cattorch_catgpt2_results"
    builder = _StageBlocks(broadcasts, {SUITE_RESULTS: result_id})
    hat = builder._id()
    first, _ = builder._chain((
        ("clear", SUITE_RESULTS),
        ("broadcast", f"{name} init"),
        ("reset_timer",),
        ("repeat", iterations, (("broadcast", f"{name} forward"),)),
        ("record_result", SUITE_RESULTS, name),
    ), hat)
    builder.blocks[hat] = {
        "opcode": "event_whenflagclicked", "next": first, "parent": None,
        "inputs": {}, "fields": {}, "shadow": False, "topLevel": True,
        "x": 0, "y": 0,
    }
    stage, md5ext, stage_bytes = _stage_target(
        builder.blocks,
        broadcasts,
        {result_id: [SUITE_RESULTS, []]},
        {
            "catgpt2_iterations": ["cattorch benchmark iterations", iterations],
            "catgpt2_sprite_bytes": ["cattorch standalone sprite bytes", standalone_size],
        },
    )
    monitor = {
        "id": result_id, "mode": "list", "opcode": "data_listcontents",
        "params": {"LIST": SUITE_RESULTS}, "spriteName": None, "value": [],
        "width": 460, "height": 180, "x": 10, "y": 10, "visible": True,
    }
    return _write_project(
        output, stage, [sprite], assets, (md5ext, stage_bytes),
        f"CatGPT2 {name} benchmark", monitors=[monitor],
    )


def _build_cached_project(
    model,
    prompt: torch.Tensor,
    name: str,
    output: Path,
    decode_iterations: int,
    *,
    hidden_prefill: bool = True,
) -> Path:
    """Time unpack-free prefill and valid cached decode calls in one project."""
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory) / name
        transpile(
            model,
            GenerationProgram(
                method="forward",
                example_token=prompt[:, :1],
                max_context=prompt.shape[1] + decode_iterations,
                hidden_prefill=hidden_prefill,
            ),
            str(base),
            optimization="exact",
            storage=StorageConfig(precision="float16"),
        )
        sprite_path = base.with_suffix(".sprite3")
        standalone_size = sprite_path.stat().st_size
        sprite, assets = _load_sprite(sprite_path)
    for entry in sprite.get("lists", {}).values():
        if entry[0] == "cattorch tokens":
            entry[1] = prompt.detach().flatten().tolist()
    decode_token = int(prompt[0, -1])
    _add_warp_procedure(
        sprite,
        "cattorch benchmark decode",
        Program(
            "catgpt2_benchmark_decode",
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
    _merge_lists_by_name(sprite, {"input"})
    sprite["name"] = name
    sprite["visible"] = False
    sprite["layerOrder"] = 1
    broadcasts = {
        f"{name} init": f"{name}_init",
        f"{name} prefill": f"{name}_prefill",
        f"{name} decode": f"{name}_decode",
    }
    _add_broadcast_entry(sprite, f"{name} init", broadcasts[f"{name} init"], "cattorch init")
    _add_broadcast_entry(
        sprite, f"{name} prefill", broadcasts[f"{name} prefill"], "cattorch prefill",
    )
    _add_broadcast_entry(
        sprite,
        f"{name} decode",
        broadcasts[f"{name} decode"],
        "cattorch benchmark decode",
    )
    result_id = "cattorch_catgpt2_cached_results"
    builder = _StageBlocks(broadcasts, {SUITE_RESULTS: result_id})
    hat = builder._id()
    first, _ = builder._chain((
        ("clear", SUITE_RESULTS),
        ("broadcast", f"{name} init"),
        ("reset_timer",),
        ("broadcast", f"{name} prefill"),
        ("record_result", SUITE_RESULTS, f"{name} prefill"),
        ("reset_timer",),
        (
            "repeat", decode_iterations,
            (("broadcast", f"{name} decode"),),
        ),
        ("record_result", SUITE_RESULTS, f"{name} decode{decode_iterations}"),
    ), hat)
    builder.blocks[hat] = {
        "opcode": "event_whenflagclicked", "next": first, "parent": None,
        "inputs": {}, "fields": {}, "shadow": False, "topLevel": True,
        "x": 0, "y": 0,
    }
    stage, md5ext, stage_bytes = _stage_target(
        builder.blocks,
        broadcasts,
        {result_id: [SUITE_RESULTS, []]},
        {
            "catgpt2_decode_iterations": [
                "cattorch benchmark iterations", decode_iterations,
            ],
            "catgpt2_sprite_bytes": [
                "cattorch standalone sprite bytes", standalone_size,
            ],
        },
    )
    monitor = {
        "id": result_id, "mode": "list", "opcode": "data_listcontents",
        "params": {"LIST": SUITE_RESULTS}, "spriteName": None, "value": [],
        "width": 460, "height": 180, "x": 10, "y": 10, "visible": True,
    }
    return _write_project(
        output, stage, [sprite], assets, (md5ext, stage_bytes),
        f"CatGPT2 cached {name} benchmark", monitors=[monitor],
    )


def build_cached(contexts=(1, 8, 32, 128), decode_iterations: int = 3) -> list[Path]:
    source, _, payload = load_cached_mqa_export_model(max(contexts) + decode_iterations)
    tokenizer = load_checkpoint_tokenizer(payload)
    row = _validation_rows(1)[0]
    prompt_tokens = tokenizer.format_messages(
        row["messages"][:-1], add_generation_prompt=True,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for context in contexts:
        model = CachedMQAExportModel(source, context + decode_iterations).eval()
        values = ([tokenizer.BOS] * context + prompt_tokens)[-context:]
        prompt = torch.tensor([values])
        name = f"mqa_cached_ctx{context}"
        paths.append(_build_cached_project(
            model,
            prompt,
            name,
            OUTPUT / f"{name}.sb3",
            decode_iterations,
        ))
    control_context = 32
    control_model = CachedMQAExportModel(
        source,
        control_context + decode_iterations,
        CachedExpandedExportAttention,
    ).eval()
    control_values = (
        [tokenizer.BOS] * control_context + prompt_tokens
    )[-control_context:]
    control_name = f"expanded_cached_ctx{control_context}"
    paths.append(_build_cached_project(
        control_model,
        torch.tensor([control_values]),
        control_name,
        OUTPUT / f"{control_name}.sb3",
        decode_iterations,
    ))
    baseline_name = f"mqa_fullhead_prefill_ctx{control_context}"
    paths.append(_build_cached_project(
        CachedMQAExportModel(
            source, control_context + decode_iterations,
        ).eval(),
        torch.tensor([control_values]),
        baseline_name,
        OUTPUT / f"{baseline_name}.sb3",
        decode_iterations,
        hidden_prefill=False,
    ))
    return paths


def build(contexts=(1, 8, 32, 128), approximation_context=32) -> list[Path]:
    _, _, payload = load_export_model(max(*contexts, approximation_context), last_only=True)
    tokenizer = load_checkpoint_tokenizer(payload)
    rows = _validation_rows(1)
    prompt = tokenizer.format_messages(rows[0]["messages"][:-1], add_generation_prompt=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for context in contexts:
        _, model, _ = load_export_model(context, last_only=True)
        values = ([tokenizer.BOS] * context + prompt)[-context:]
        tokens = torch.tensor([values])
        paths.append(_build_one_project(
            model, tokens, f"exact_ctx{context}", OUTPUT / f"exact_ctx{context}.sb3", 1,
        ))
    _, approximation_base, _ = load_export_model(approximation_context, last_only=True)
    values = ([tokenizer.BOS] * approximation_context + prompt)[-approximation_context:]
    tokens = torch.tensor([values])
    timed_variants = (
        "head_rank112", "head_rank96", "head_rank80", "head_rank64",
        "rank50", "sparse25", "body_sparse25", "body_sparse50",
        "qkv_rank80", "qkv_rank64",
    )
    all_policies = policies()
    for name in timed_variants:
        policy = all_policies[name]
        model = prepare_fast_model(approximation_base, policy).eval()
        paths.append(_build_one_project(
            model, tokens, f"{name}_ctx{approximation_context}",
            OUTPUT / f"{name}_ctx{approximation_context}.sb3", 1,
        ))
    repeated = {
        "exact": approximation_base,
        "qkv_rank80": prepare_fast_model(
            approximation_base, all_policies["qkv_rank80"],
        ).eval(),
    }
    for name, model in repeated.items():
        benchmark_name = f"{name}_ctx{approximation_context}_repeat3"
        paths.append(_build_one_project(
            model, tokens, benchmark_name, OUTPUT / f"{benchmark_name}.sb3", 3,
        ))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build-cached", action="store_true")
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()
    if not args.evaluate and not args.build and not args.build_cached:
        args.evaluate = args.build = True
    if args.evaluate:
        print(json.dumps(evaluate(args.limit), indent=2))
    if args.build:
        for path in build():
            print(path)
    if args.build_cached:
        for path in build_cached():
            print(path)


if __name__ == "__main__":
    main()
