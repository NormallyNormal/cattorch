"""Build an upload-safe whole-transformer before/after grouped-linear suite."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from build_suite import TinyGPT
from cattorch import StorageConfig, transpile
from cattorch.benchmark import (
    SUITE_RESULTS,
    _stage_blocks,
    _timed,
    _add_broadcast_entry,
    _load_sprite,
    _set_inputs,
    _stage_target,
    _write_project,
)
from cattorch.util.instruction.optimized import LinearInstruction
from cattorch.util.scratch.dsl import append, broadcast_and_wait, clear, repeat, reset_timer  # noqa: E402


OUTPUT = Path(__file__).parent / "artifacts" / "grouped_linear_whole_suite.sb3"
ITERATIONS = 5


def _compile_variant(model, tokens, name: str, threshold: float):
    previous_threshold = LinearInstruction.grouped_min_macs
    try:
        LinearInstruction.grouped_min_macs = threshold
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
    finally:
        LinearInstruction.grouped_min_macs = previous_threshold
    _set_inputs(sprite, (tokens,))
    sprite["name"] = name
    sprite["visible"] = False
    return sprite, assets, standalone_size


def main():
    torch.manual_seed(53)
    model = TinyGPT(
        width=128,
        heads=2,
        hidden=384,
        context=4,
        vocab=257,
    ).eval()
    tokens = torch.tensor([[3, 17, 91, 4]])
    variants = (
        ("transformer previous exact", float("inf")),
        ("transformer grouped exact", 49_152),
    )
    sprites = []
    assets = {}
    sizes = {}
    broadcasts = {}
    controller = []

    for index, (name, threshold) in enumerate(variants):
        sprite, sprite_assets, standalone_size = _compile_variant(
            model, tokens, name, threshold,
        )
        sprite["layerOrder"] = index + 1
        sprites.append(sprite)
        assets.update(sprite_assets)
        sizes[name] = standalone_size
        messages = {}
        for action in ("init", "forward"):
            message = f"{name} {action}"
            broadcast_id = f"grouped_whole_{index}_{action}"
            broadcasts[message] = broadcast_id
            _add_broadcast_entry(
                sprite, message, broadcast_id, f"cattorch {action}",
            )
            messages[action] = message
        controller.append((name, messages))

    result_id = "cattorch_grouped_whole_results"
    statements = [clear(SUITE_RESULTS)]
    for name, messages in controller:
        statements += [
            broadcast_and_wait(messages["init"]),
            broadcast_and_wait(messages["forward"]),
            reset_timer(),
            repeat(ITERATIONS, (broadcast_and_wait(messages["forward"]),)),
            append(SUITE_RESULTS, _timed(f"{name} forward{ITERATIONS}")),
        ]
    stage_blocks = _stage_blocks(statements, broadcasts, {SUITE_RESULTS: result_id})
    stage, stage_md5ext, stage_bytes = _stage_target(
        stage_blocks,
        broadcasts,
        {result_id: [SUITE_RESULTS, []]},
        {
            "grouped_whole_iterations": [
                "cattorch benchmark iterations", ITERATIONS,
            ],
            **{
                f"grouped_whole_size_{index}": [f"{name} sprite bytes", size]
                for index, (name, size) in enumerate(sizes.items())
            },
        },
    )
    monitor = {
        "id": result_id, "mode": "list", "opcode": "data_listcontents",
        "params": {"LIST": SUITE_RESULTS}, "spriteName": None, "value": [],
        "width": 520, "height": 180, "x": 10, "y": 10, "visible": True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    path = _write_project(
        OUTPUT,
        stage,
        sprites,
        assets,
        (stage_md5ext, stage_bytes),
        "Grouped-linear upload-safe whole-network suite",
        monitors=[monitor],
    )
    print(f"Grouped-linear whole-transformer suite written: {path}")


if __name__ == "__main__":
    main()
