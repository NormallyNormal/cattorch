"""Build a vanilla-Scratch timing/size comparison for generation top-k."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

from cattorch.benchmark import (
    SUITE_RESULTS,
    _StageBlocks,
    _add_broadcast_entry,
    _load_sprite,
    _stage_target,
    _write_project,
)
from cattorch.codegen import CodegenConfig
from cattorch.sprite import (
    _TOP_K_PROCEDURE,
    _add_warp_procedure,
    _build_compact_top_k_program,
    _build_top_k_program,
)
from cattorch.util.scratch.finalize_scratch import finalize_sprite


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "artifacts" / "top_k_size_suite.sb3"
SIZE_RESULTS = "cattorch top k size results"


def build_top_k_size_suite(
    output_path: str | Path = OUTPUT,
    *,
    top_k: int = 16,
    iterations: int = 20,
) -> Path:
    logits = [math.sin(index * 0.37) * 5 + math.cos(index * 0.11) for index in range(1024)]
    programs = {
        "unrolled": _build_top_k_program(top_k),
        "compact": _build_compact_top_k_program(top_k),
    }
    broadcasts: dict[str, str] = {}
    sprites = []
    assets = {}
    sizes = []
    controller = {"name": f"top-k {top_k}"}

    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        for position, (name, program) in enumerate(programs.items()):
            sprite = {
                "blocks": {}, "variables": {}, "lists": {},
                "broadcasts": {}, "comments": {},
            }
            sprite = _add_warp_procedure(
                sprite, _TOP_K_PROCEDURE, program, x=0, y=0,
            )
            for entry in sprite["lists"].values():
                if entry[0] == "output":
                    entry[1] = logits
            finalized = finalize_sprite(
                sprite,
                temporary / f"top_k_{name}.sprite3",
                sprite_name=f"top-k {name}",
                codegen=CodegenConfig(id_namespace={
                    "unrolled": "unr", "compact": "cmp",
                }[name]),
            )
            target, target_assets = _load_sprite(finalized.path)
            target["name"] = f"top-k {name}"
            target["isStage"] = False
            target["visible"] = False
            target["layerOrder"] = position + 1
            message = f"cattorch top k {name}"
            broadcast_id = f"cattorch_top_k_{name}"
            broadcasts[message] = broadcast_id
            _add_broadcast_entry(
                target, message, broadcast_id, _TOP_K_PROCEDURE,
            )
            controller[name] = {"init": message, "forward": message}
            sprites.append(target)
            assets.update(target_assets)
            sizes.append(
                f"{name}: {finalized.expanded_json_bytes} JSON bytes, "
                f"{len(target['blocks'])} blocks"
            )

    results_id = "cattorch_top_k_benchmark_results"
    sizes_id = "cattorch_top_k_size_results"
    stage_blocks = _StageBlocks(
        broadcasts, {SUITE_RESULTS: results_id, SIZE_RESULTS: sizes_id},
    ).build_suite([controller], iterations, 0, tuple(programs))
    stage, md5ext, stage_bytes = _stage_target(
        stage_blocks,
        broadcasts,
        {
            results_id: [SUITE_RESULTS, []],
            sizes_id: [SIZE_RESULTS, sizes],
        },
        {},
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return _write_project(
        output_path,
        stage,
        sprites,
        assets,
        (md5ext, stage_bytes),
        "Compact generation top-k benchmark suite",
    )


if __name__ == "__main__":
    print(build_top_k_size_suite())
