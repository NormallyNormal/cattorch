"""Compare cattorch storage formats on the current TinyStories checkpoint.

The Python pass evaluates many validation windows cheaply.  The VM pass builds
four self-contained projects and checks representative logits in the official
vanilla Scratch VM, including the real unpacking blocks used by an export.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from cattorch import StorageConfig, transpile
from cattorch.benchmark import (
    SUITE_RESULTS,
    _StageBlocks,
    _add_broadcast_entry,
    _load_sprite,
    _set_inputs,
    _stage_target,
    _write_project,
)
from cattorch.storage import rounded_codebook_values, rounded_values
from cattorch.transpiler import _add_warp_procedure, _merge_lists_by_name
from cattorch.util.scratch.dsl import (
    Program,
    append,
    call,
    clear,
    for_each,
    item,
    length,
    var,
)

from catgpt2_approx_experiment import ExportModel, load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CATGPT2 = ROOT.parent / "catgpt2"
CHECKPOINT = CATGPT2 / "artifacts" / "tinystories-500k.pt"
VALIDATION = CATGPT2 / "data" / "processed" / "tinystories_valid_1k_scratch.bin"
OUTPUT = ROOT / "benchmarks" / "artifacts" / "tinystories_quant"
VM_RUNNER = ROOT / "benchmarks" / "run_scratch_vm.cjs"
LOGIT_PREFIX = "cattorch quant logits "

FORMATS = {
    "f32": StorageConfig(precision="float32"),
    "f16": StorageConfig(precision="float16"),
    "int8": StorageConfig(precision="int8", group_size=64),
    "int8_f16_scales": StorageConfig(
        precision="int8", group_size=64, scale_precision="float16",
    ),
    "int6": StorageConfig(precision="int6", group_size=64),
    "int4": StorageConfig(precision="int4", group_size=64),
}


def _checkpoint_hash() -> str:
    digest = hashlib.sha256()
    with CHECKPOINT.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_windows(count: int, context: int) -> tuple[torch.Tensor, torch.Tensor]:
    raw = VALIDATION.read_bytes()
    token_count = len(raw) // 2
    tokens = struct.unpack(f"<{token_count}H", raw)
    # Spread fixed windows through the beginning of the held-out corpus.  The
    # offsets are deterministic so checkpoint-to-checkpoint reports compare
    # the same examples.
    stride = 257
    rows = []
    targets = []
    for index in range(count):
        start = index * stride
        rows.append(tokens[start:start + context])
        targets.append(tokens[start + context])
    return torch.tensor(rows, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def _storage_rounded_model(
    model: torch.nn.Module,
    config: StorageConfig,
) -> torch.nn.Module:
    result = copy.deepcopy(model).eval()
    seen: set[int] = set()
    with torch.no_grad():
        for value in (*result.parameters(), *result.buffers()):
            if id(value) in seen or not value.is_floating_point():
                continue
            seen.add(id(value))
            precision = config.precision
            if precision in {"int8", "int6", "int4"}:
                if value.ndim < 2:
                    precision = "float16"
                elif value.numel() < config.min_quantized_values:
                    precision = "float16"
            rounded = rounded_values(
                value.flatten(), precision, config.group_size,
                config.scale_precision,
            )
            value.copy_(torch.tensor(rounded, dtype=value.dtype).reshape_as(value))
    return result


def _codebook_rounded_model(model: torch.nn.Module) -> torch.nn.Module:
    """Build the quality-gated, non-public learned-codebook experiment."""
    result = copy.deepcopy(model).eval()
    seen: set[int] = set()
    with torch.no_grad():
        for value in (*result.parameters(), *result.buffers()):
            if id(value) in seen or not value.is_floating_point():
                continue
            seen.add(id(value))
            if value.ndim < 2 or value.numel() < 16_384:
                rounded = rounded_values(value.flatten(), "float16")
            else:
                rounded = rounded_codebook_values(value.flatten())
            value.copy_(torch.tensor(rounded, dtype=value.dtype).reshape_as(value))
    return result


def _logit_metrics(
    logits: torch.Tensor,
    baseline: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float | int]:
    delta = logits - baseline
    return {
        "cross_entropy": float(F.cross_entropy(logits, targets)),
        "perplexity": math.exp(float(F.cross_entropy(logits, targets))),
        "top1_agreement_with_f32": float(
            (logits.argmax(-1) == baseline.argmax(-1)).float().mean()
        ),
        "mean_abs_logit_error": float(delta.abs().mean()),
        "max_abs_logit_error": float(delta.abs().max()),
        "logit_rmse": float(delta.square().mean().sqrt()),
        "mean_cosine_similarity": float(
            F.cosine_similarity(logits, baseline, dim=-1).mean()
        ),
        "windows": logits.shape[0],
    }


def evaluate_python(*, windows: int = 256, context: int = 16) -> dict:
    source, _ = load_checkpoint(CHECKPOINT)
    model = ExportModel(source.eval(), context, last_only=True).eval()
    inputs, targets = _validation_windows(windows, context)
    results = {}
    with torch.inference_mode():
        baseline = model(inputs)[:, -1, :]
        for name, config in FORMATS.items():
            variant = model if name == "f32" else _storage_rounded_model(model, config)
            logits = baseline if name == "f32" else variant(inputs)[:, -1, :]
            results[name] = _logit_metrics(logits, baseline, targets)
        codebook = _codebook_rounded_model(model)
        results["codebook6_experimental"] = _logit_metrics(
            codebook(inputs)[:, -1, :], baseline, targets,
        )
    return {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": _checkpoint_hash(),
        "validation": str(VALIDATION),
        "context": context,
        "results": results,
    }


def _add_case_procedure(sprite: dict, tokens: torch.Tensor, index: int) -> str:
    result_name = f"{LOGIT_PREFIX}{index:02d}"
    loop_name = f"quant output index {index:02d}"
    procedure = f"cattorch quant case {index:02d}"
    program = Program(
        f"tinystories_quant_case_{index:02d}",
        variables=(loop_name,),
        lists=("input", "output", result_name),
        body=(
            clear("input"),
            *(append("input", int(token)) for token in tokens),
            call("cattorch forward"),
            clear(result_name),
            for_each(loop_name, length("output"), (
                append(result_name, item("output", var(loop_name))),
            )),
        ),
    )
    _add_warp_procedure(sprite, procedure, program, x=900, y=index * 80)
    _merge_lists_by_name(sprite, {"input", "output", result_name})
    return procedure


def _build_project(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    name: str,
    storage: StorageConfig,
) -> tuple[Path, dict]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        export = transpile(
            model,
            inputs[:1],
            Path(directory) / name,
            optimization="exact",
            storage=storage,
        )
        sprite, assets = _load_sprite(export.path)
    _set_inputs(sprite, (inputs[:1],))
    sprite["name"] = name
    sprite["visible"] = False
    sprite["layerOrder"] = 1

    procedures = [
        _add_case_procedure(sprite, tokens, index)
        for index, tokens in enumerate(inputs)
    ]
    broadcasts = {f"{name} init": f"{name}_init"}
    _add_broadcast_entry(
        sprite, f"{name} init", broadcasts[f"{name} init"], "cattorch init",
    )
    for index, procedure in enumerate(procedures):
        message = f"{name} case {index:02d}"
        broadcasts[message] = f"{name}_case_{index:02d}"
        _add_broadcast_entry(sprite, message, broadcasts[message], procedure)

    result_id = "cattorch_tinystories_quant_results"
    builder = _StageBlocks(broadcasts, {SUITE_RESULTS: result_id})
    commands = [("clear", SUITE_RESULTS), ("broadcast", f"{name} init")]
    for index in range(len(procedures)):
        commands.extend((
            ("reset_timer",),
            ("broadcast", f"{name} case {index:02d}"),
            ("record_result", SUITE_RESULTS, f"{name} case {index:02d}"),
        ))
    hat = builder._id()
    first, _ = builder._chain(tuple(commands), hat)
    builder.blocks[hat] = {
        "opcode": "event_whenflagclicked", "next": first, "parent": None,
        "inputs": {}, "fields": {}, "shadow": False, "topLevel": True,
        "x": 0, "y": 0,
    }
    stage, md5ext, stage_bytes = _stage_target(
        builder.blocks,
        broadcasts,
        {result_id: [SUITE_RESULTS, []]},
        {},
    )
    path = _write_project(
        OUTPUT / f"{name}.sb3",
        stage,
        [sprite],
        assets,
        (md5ext, stage_bytes),
        f"TinyStories quantization comparison: {name}",
    )
    metadata = {
        "path": str(path),
        "archive_bytes": path.stat().st_size,
        "base_sprite_archive_bytes": export.archive_bytes,
        "base_sprite_expanded_json_bytes": export.expanded_json_bytes,
        "block_count": export.block_count,
    }
    return path, metadata


def build_vm(*, windows: int = 8, context: int = 16) -> dict:
    source, _ = load_checkpoint(CHECKPOINT)
    model = ExportModel(source.eval(), context, last_only=True).eval()
    inputs, targets = _validation_windows(windows, context)
    projects = {}
    for name, storage in FORMATS.items():
        _, projects[name] = _build_project(model, inputs, name, storage)
    manifest = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": _checkpoint_hash(),
        "validation": str(VALIDATION),
        "context": context,
        "targets": targets.tolist(),
        "projects": projects,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _run_vm_project(path: Path) -> dict:
    completed = subprocess.run(
        [
            "node",
            str(VM_RUNNER),
            str(path),
            f"--list-values-prefix={LOGIT_PREFIX}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    # The official VM logger writes asset warnings to stdout before the runner
    # report. Locate our document rather than assuming stdout is JSON-only.
    marker = completed.stdout.find('{\n  "project"')
    if marker < 0:
        raise RuntimeError(f"Scratch VM produced no JSON report:\n{completed.stdout}")
    return json.loads(completed.stdout[marker:])


def compare_vm() -> dict:
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    targets = torch.tensor(manifest["targets"], dtype=torch.long)
    reports = {}
    outputs = {}
    for name in FORMATS:
        report = _run_vm_project(Path(manifest["projects"][name]["path"]))
        reports[name] = {
            "wall_seconds": report["wall_seconds"],
            "scratch_timings": report["results"],
        }
        ordered = sorted(report["list_values"], key=lambda entry: entry["name"])
        outputs[name] = torch.tensor([entry["values"] for entry in ordered])
    baseline = outputs["f32"]
    for name, logits in outputs.items():
        reports[name].update(_logit_metrics(logits, baseline, targets))
    result = {
        **manifest,
        "vm": "@scratch/scratch-vm 15.0.1",
        "results": reports,
    }
    (OUTPUT / "vm_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", action="store_true")
    parser.add_argument("--build-vm", action="store_true")
    parser.add_argument("--run-vm", action="store_true")
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--python-windows", type=int, default=256)
    parser.add_argument("--vm-windows", type=int, default=8)
    args = parser.parse_args()
    if not (args.python or args.build_vm or args.run_vm):
        args.python = args.build_vm = args.run_vm = True
    if args.python:
        report = evaluate_python(windows=args.python_windows, context=args.context)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "python_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    if args.build_vm:
        print(json.dumps(build_vm(windows=args.vm_windows, context=args.context), indent=2))
    if args.run_vm:
        print(json.dumps(compare_vm(), indent=2))


if __name__ == "__main__":
    main()
