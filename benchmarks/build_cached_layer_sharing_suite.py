#!/usr/bin/env python3
"""Build a vanilla-Scratch whole-network cached layer-sharing benchmark."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CATGPT2 = ROOT.parent / "catgpt2"
CHECKPOINT = (
    CATGPT2
    / "runs"
    / "tinystories_6m_500k_v9888_e16_pilot"
    / "best.pt"
)
OUTPUT = ROOT / "benchmarks" / "artifacts" / "cached_layer_sharing_suite.sb3"
DECODE_ITERATIONS = 2

for package_dir in (CATGPT2 / ".venv" / "lib").glob("python*/site-packages"):
    sys.path.append(str(package_dir))
sys.path.insert(0, str(CATGPT2 / "src"))

from microchat.checkpoint import load_checkpoint  # noqa: E402
from microchat.model import Top1SwiGLUMoE  # noqa: E402

from cattorch import (  # noqa: E402
    CodegenConfig,
    GenerationProgram,
    QuantizationConfig,
    transpile,
)
from cattorch.benchmark import (  # noqa: E402
    SUITE_RESULTS,
    _StageBlocks,
    _add_broadcast_entry,
    _load_sprite,
    _stage_target,
    _write_project,
)
from cattorch.experimental import stacked_swiglu_moe_adapter  # noqa: E402
from cattorch.transpiler import _add_warp_procedure, _merge_lists_by_name  # noqa: E402
from cattorch.util.scratch.dsl import Program, append, call, clear  # noqa: E402


def _compile_variant(model, directory: Path, label: str, sharing: str, namespace: str):
    result = transpile(
        model,
        GenerationProgram(
            method="next_token_logits",
            example_token=torch.tensor([[1]]),
            max_context=1 + DECODE_ITERATIONS,
            top_k=20,
        ),
        directory / label,
        quantization=QuantizationConfig(
            bits=4,
            group_size=256,
            min_quantized_values=1,
        ),
        codegen=CodegenConfig(
            id_namespace=namespace,
            unrolling="compact",
            compact_internal_names=True,
            compact_schema=True,
            layer_sharing=sharing,
        ),
        adapters=(stacked_swiglu_moe_adapter(Top1SwiGLUMoE),),
    )
    sprite, assets = _load_sprite(result.path)
    for entry in sprite.get("lists", {}).values():
        if entry[0] == "cattorch tokens":
            entry[1] = [1]
    _add_warp_procedure(
        sprite,
        "cattorch sharing benchmark decode",
        Program(
            "cached_layer_sharing_decode",
            lists=("cattorch tokens",),
            body=(
                clear("cattorch tokens"),
                append("cattorch tokens", 1),
                call("cattorch decode"),
            ),
        ),
        x=900,
        y=0,
    )
    _merge_lists_by_name(sprite, {"cattorch tokens"})
    sprite["name"] = label
    sprite["visible"] = False
    return sprite, assets, result


def main() -> None:
    model, _ = load_checkpoint(CHECKPOINT, device="cpu")
    model.eval()
    variants = (
        ("unshared q4", "off", "u00"),
        ("banked shared q4", "auto", "s00"),
    )
    broadcasts = {}
    sprites = []
    assets = {}
    controllers = []
    sizes = {}

    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        for index, (label, sharing, namespace) in enumerate(variants):
            sprite, sprite_assets, result = _compile_variant(
                model, temporary, label, sharing, namespace,
            )
            sprite["layerOrder"] = index + 1
            sprites.append(sprite)
            assets.update(sprite_assets)
            sizes[label] = result.expanded_json_bytes
            messages = {}
            for action, procedure in (
                ("init", "cattorch init"),
                ("prefill", "cattorch prefill"),
                ("decode", "cattorch sharing benchmark decode"),
            ):
                message = f"cattorch sharing {index} {action}"
                identifier = f"cattorch_sharing_{index}_{action}"
                broadcasts[message] = identifier
                _add_broadcast_entry(sprite, message, identifier, procedure)
                messages[action] = message
            controllers.append((label, messages))

    result_id = "cattorch_cached_layer_sharing_results"
    builder = _StageBlocks(broadcasts, {SUITE_RESULTS: result_id})
    specs = [("clear", SUITE_RESULTS)]
    for label, messages in controllers:
        specs.extend((
            ("reset_timer",),
            ("broadcast", messages["init"]),
            ("record_result", SUITE_RESULTS, f"{label} init"),
            ("reset_timer",),
            ("broadcast", messages["prefill"]),
            ("record_result", SUITE_RESULTS, f"{label} prefill"),
            ("reset_timer",),
            (
                "repeat",
                DECODE_ITERATIONS,
                (("broadcast", messages["decode"]),),
            ),
            ("record_result", SUITE_RESULTS, f"{label} decode{DECODE_ITERATIONS}"),
        ))
    hat = builder._id()
    first, _ = builder._chain(tuple(specs), hat)
    builder.blocks[hat] = {
        "opcode": "event_whenflagclicked",
        "next": first,
        "parent": None,
        "inputs": {},
        "fields": {},
        "shadow": False,
        "topLevel": True,
        "x": 0,
        "y": 0,
    }
    stage, stage_md5ext, stage_bytes = _stage_target(
        builder.blocks,
        broadcasts,
        {result_id: [SUITE_RESULTS, []]},
        {
            "cached_sharing_iterations": [
                "cattorch benchmark iterations",
                DECODE_ITERATIONS,
            ],
            **{
                f"cached_sharing_size_{index}": [f"{label} JSON bytes", sizes[label]]
                for index, (label, _, _) in enumerate(variants)
            },
        },
    )
    monitor = {
        "id": result_id,
        "mode": "list",
        "opcode": "data_listcontents",
        "params": {"LIST": SUITE_RESULTS},
        "spriteName": None,
        "value": [],
        "width": 520,
        "height": 180,
        "x": 10,
        "y": 10,
        "visible": True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    path = _write_project(
        OUTPUT,
        stage,
        sprites,
        assets,
        (stage_md5ext, stage_bytes),
        "Cached layer-sharing suite",
        monitors=[monitor],
    )
    print(f"Cached layer-sharing suite written: {path}")
    for label, size in sizes.items():
        print(f"{label}: {size} expanded JSON bytes")


if __name__ == "__main__":
    main()
