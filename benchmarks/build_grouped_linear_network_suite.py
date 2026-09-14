"""Build an isolated whole-network before/after suite for grouped linear."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CATGPT2 = ROOT.parent / "catgpt2"
# CatGPT2 owns its tokenizer dependency. Keep cattorch's CPU torch first on
# sys.path, then make the sibling environment available for missing packages.
for site_packages in (CATGPT2 / ".venv" / "lib").glob("python*/site-packages"):
    sys.path.append(str(site_packages))
sys.path.insert(0, str(Path(__file__).parent))

from catgpt2_approx_experiment import CachedMQAExportModel, load_cached_mqa_export_model
from cattorch import GenerationProgram, StorageConfig, transpile
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
from cattorch.util.instruction.optimized import LinearInstruction
from cattorch.util.scratch.dsl import Program, append, call, clear


OUTPUT = Path(__file__).parent / "artifacts" / "grouped_linear_network_suite.sb3"
CONTEXT = 32
DECODE_ITERATIONS = 3


def _compile_variant(source, name: str, threshold: float):
    model = CachedMQAExportModel(
        source, CONTEXT + DECODE_ITERATIONS,
    ).eval()
    prompt = torch.zeros((1, CONTEXT), dtype=torch.long)
    previous_threshold = LinearInstruction.grouped_min_macs
    try:
        LinearInstruction.grouped_min_macs = threshold
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / name
            transpile(
                model,
                GenerationProgram(
                    method="forward",
                    example_token=prompt[:, :1],
                    max_context=CONTEXT + DECODE_ITERATIONS,
                    hidden_prefill=True,
                ),
                str(base),
                optimization="exact",
                storage=StorageConfig(precision="float16"),
            )
            sprite_path = base.with_suffix(".sprite3")
            standalone_size = sprite_path.stat().st_size
            sprite, assets = _load_sprite(sprite_path)
    finally:
        LinearInstruction.grouped_min_macs = previous_threshold

    for entry in sprite.get("lists", {}).values():
        if entry[0] == "cattorch tokens":
            entry[1] = prompt.detach().flatten().tolist()
    _add_warp_procedure(
        sprite,
        "cattorch grouped benchmark decode",
        Program(
            "grouped_network_benchmark_decode",
            lists=("cattorch tokens",),
            body=(
                clear("cattorch tokens"),
                append("cattorch tokens", 0),
                call("cattorch decode"),
            ),
        ),
        x=900,
        y=0,
    )
    _merge_lists_by_name(sprite, {"cattorch tokens"})
    sprite["name"] = name
    sprite["visible"] = False
    return sprite, assets, standalone_size


def main():
    source, _, _ = load_cached_mqa_export_model(
        CONTEXT + DECODE_ITERATIONS,
    )
    source.eval()
    variants = (
        ("catgpt2 previous exact", float("inf")),
        ("catgpt2 grouped exact", 49_152),
    )
    sprites = []
    assets = {}
    standalone_sizes = {}
    broadcasts = {}
    controller = []

    for index, (name, threshold) in enumerate(variants):
        sprite, sprite_assets, standalone_size = _compile_variant(
            source, name, threshold,
        )
        sprite["layerOrder"] = index + 1
        sprites.append(sprite)
        assets.update(sprite_assets)
        standalone_sizes[name] = standalone_size

        messages = {}
        for action, procedure in (
            ("init", "cattorch init"),
            ("prefill", "cattorch prefill"),
            ("decode", "cattorch grouped benchmark decode"),
        ):
            message = f"{name} {action}"
            broadcast_id = f"grouped_network_{index}_{action}"
            broadcasts[message] = broadcast_id
            _add_broadcast_entry(sprite, message, broadcast_id, procedure)
            messages[action] = message
        controller.append((name, messages))

    result_id = "cattorch_grouped_network_results"
    builder = _StageBlocks(broadcasts, {SUITE_RESULTS: result_id})
    specs = [("clear", SUITE_RESULTS)]
    for name, messages in controller:
        specs.extend((
            ("broadcast", messages["init"]),
            ("reset_timer",),
            ("broadcast", messages["prefill"]),
            ("record_result", SUITE_RESULTS, f"{name} prefill{CONTEXT}"),
            ("reset_timer",),
            (
                "repeat", DECODE_ITERATIONS,
                (("broadcast", messages["decode"]),),
            ),
            (
                "record_result", SUITE_RESULTS,
                f"{name} decode{DECODE_ITERATIONS}",
            ),
        ))
    hat = builder._id()
    first, _ = builder._chain(tuple(specs), hat)
    builder.blocks[hat] = {
        "opcode": "event_whenflagclicked", "next": first, "parent": None,
        "inputs": {}, "fields": {}, "shadow": False, "topLevel": True,
        "x": 0, "y": 0,
    }
    stage, stage_md5ext, stage_bytes = _stage_target(
        builder.blocks,
        broadcasts,
        {result_id: [SUITE_RESULTS, []]},
        {
            "grouped_network_context": ["cattorch benchmark context", CONTEXT],
            "grouped_network_decodes": [
                "cattorch benchmark iterations", DECODE_ITERATIONS,
            ],
            **{
                f"grouped_network_size_{index}": [f"{name} sprite bytes", size]
                for index, (name, size) in enumerate(standalone_sizes.items())
            },
        },
    )
    monitor = {
        "id": result_id, "mode": "list", "opcode": "data_listcontents",
        "params": {"LIST": SUITE_RESULTS}, "spriteName": None, "value": [],
        "width": 520, "height": 220, "x": 10, "y": 10, "visible": True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    path = _write_project(
        OUTPUT,
        stage,
        sprites,
        assets,
        (stage_md5ext, stage_bytes),
        "Grouped-linear whole-network suite",
        monitors=[monitor],
    )
    print(f"Grouped-linear whole-network suite written: {path}")


if __name__ == "__main__":
    main()
