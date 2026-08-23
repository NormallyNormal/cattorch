"""Build one vanilla-Scratch suite for JSON compaction and shared layers."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

from cattorch import CodegenConfig, StorageConfig, transpile
from cattorch.benchmark import (
    SUITE_RESULTS,
    _StageBlocks,
    _add_broadcast_entry,
    _load_sprite,
    _set_inputs,
    _stage_target,
    _write_project,
)


ROOT = Path(__file__).resolve().parents[1]
CATGPT2 = ROOT.parent / "catgpt2"
CHECKPOINT = CATGPT2 / "artifacts" / "tinystories-500k.pt"
OUTPUT = ROOT / "benchmarks" / "artifacts" / "json_size_suite.sb3"
SIZE_RESULTS = "cattorch JSON size results"


def _load_model():
    sys.path.insert(0, str(CATGPT2 / "src"))
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from catgpt2_approx_experiment import ExportModel, load_checkpoint

    source, _ = load_checkpoint(CHECKPOINT)
    return ExportModel(source.eval(), 1, last_only=True).eval()


def build_json_size_suite(
    output_path: str | Path = OUTPUT,
    *,
    selected_variants: tuple[str, ...] | None = None,
) -> Path:
    model = _load_model()
    inputs = (torch.tensor([[1]]),)
    variants = {
        "baseline": CodegenConfig(
            id_namespace="b00", unrolling="compact", layer_sharing="off",
        ),
        "shared": CodegenConfig(
            id_namespace="s00", unrolling="compact", layer_sharing="auto",
        ),
        "compact": CodegenConfig(
            id_namespace="c00", unrolling="compact", layer_sharing="auto",
            compact_internal_names=True, compact_schema=True,
        ),
    }
    if selected_variants is not None:
        variants = {name: variants[name] for name in selected_variants}
    broadcasts = {}
    sprites = []
    assets = {}
    sizes = []
    controller = {"name": "tinystories layer sharing"}

    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        for index, (variant, codegen) in enumerate(variants.items()):
            result = transpile(
                model,
                inputs,
                temporary / variant,
                storage=StorageConfig(
                    precision="int8", scale_precision="float16",
                ),
                codegen=codegen,
            )
            sprite, sprite_assets = _load_sprite(result.path)
            _set_inputs(sprite, inputs)
            sprite["name"] = f"tinystories {variant}"
            sprite["visible"] = False
            sprite["layerOrder"] = index + 1
            messages = {}
            for procedure in ("init", "forward"):
                message = f"cattorch JSON size {variant} {procedure}"
                identifier = f"cattorch_json_size_{variant}_{procedure}"
                broadcasts[message] = identifier
                _add_broadcast_entry(
                    sprite, message, identifier, f"cattorch {procedure}",
                )
                messages[procedure] = message
            controller[variant] = messages
            sprites.append(sprite)
            assets.update(sprite_assets)
            sizes.append(
                f"{variant}: {result.expanded_json_bytes} JSON bytes, "
                f"{result.block_count} blocks"
            )

    results_id = "cattorch_json_size_benchmark_results"
    sizes_id = "cattorch_json_size_results"
    stage_blocks = _StageBlocks(
        broadcasts, {SUITE_RESULTS: results_id, SIZE_RESULTS: sizes_id},
    ).build_suite([controller], 3, 0, tuple(variants))
    stage, md5ext, stage_bytes = _stage_target(
        stage_blocks,
        broadcasts,
        {
            results_id: [SUITE_RESULTS, []],
            sizes_id: [SIZE_RESULTS, sizes],
        },
        {},
    )
    monitors = [
        {
            "id": identifier, "mode": "list", "opcode": "data_listcontents",
            "params": {"LIST": name}, "spriteName": None, "value": values,
            "width": 520, "height": 220, "x": 10, "y": 10 + position * 230,
            "visible": True,
        }
        for position, (identifier, name, values) in enumerate((
            (results_id, SUITE_RESULTS, []),
            (sizes_id, SIZE_RESULTS, sizes),
        ))
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return _write_project(
        output_path,
        stage,
        sprites,
        assets,
        (md5ext, stage_bytes),
        "JSON-size and shared-layer benchmark suite",
        monitors=monitors,
    )


if __name__ == "__main__":
    print(build_json_size_suite())
