#!/usr/bin/env python3
"""Build the production TinyStories processor, tokenizer, and sampler sprite."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CATGPT2 = ROOT.parent / "catgpt2"
sys.path.insert(0, str(CATGPT2 / "src"))

from microchat.checkpoint import load_checkpoint, load_checkpoint_tokenizer

from benchmarks.add_tinystories_sampler import (
    EOS_ID,
    FREQUENCY_PENALTY,
    TEMPERATURE,
    TOP_K,
    add_sampler,
)
from benchmarks.catgpt2_approx_experiment import CachedMQAExportModel
from benchmarks.combine_sprites import combine_sprites
from cattorch import (
    CodegenConfig,
    GenerationProgram,
    StorageConfig,
    transpile,
    transpile_tokenizer,
)
from cattorch.util.scratch.finalize_scratch import finalize_sprite


DEFAULT_CHECKPOINT = (
    CATGPT2
    / "runs"
    / "tinystories_1200k_4k_w112_d6_ctx256_short224_stage3"
    / "best.pt"
)
DEFAULT_OUTPUT = ROOT / "benchmarks" / "artifacts"
STEM = "tinystories-1200k-ctx256"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sprite(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("sprite.json"))


def build(checkpoint: Path, output_dir: Path) -> dict:
    source, payload = load_checkpoint(checkpoint, device="cpu")
    source.eval()
    context = source.config.block_size
    if context != 256:
        raise ValueError(f"expected a 256-token checkpoint, got {context}")

    output_dir.mkdir(parents=True, exist_ok=True)
    processor_path = output_dir / f"{STEM}-int8.sprite3"
    tokenizer_path = output_dir / f"{STEM}-tokenizer.sprite3"
    complete_path = output_dir / f"{STEM}-int8-complete.sprite3"
    manifest_path = output_dir / f"{STEM}-int8-complete.manifest.json"

    codegen = CodegenConfig(
        unrolling="compact",
        compact_internal_names=True,
        id_namespace="",
    )
    storage = StorageConfig(
        precision="int8",
        group_size=64,
        scale_precision="float16",
    )

    # Generation must use the cached wrapper. It presents all RoPE positions
    # as dynamic embeddings; a one-token stateless wrapper only contains
    # position zero and silently produces incorrect cached generations.
    processor_model = CachedMQAExportModel(source, context).eval()
    processor = transpile(
        processor_model,
        GenerationProgram(
            method="forward",
            example_token=torch.tensor([[1]]),
            max_context=context,
        ),
        processor_path,
        name=f"{STEM} int8",
        optimization="exact",
        storage=storage,
        codegen=codegen,
    )
    tokenizer = transpile_tokenizer(
        load_checkpoint_tokenizer(payload),
        tokenizer_path,
        name=f"{STEM} tokenizer",
        scratch_casefold=True,
        codegen=codegen,
    )

    with tempfile.TemporaryDirectory() as directory:
        combined_path = Path(directory) / "combined.sprite3"
        combine_sprites(
            processor.path,
            tokenizer.path,
            combined_path,
            name=f"{STEM} combined",
        )
        combined = _read_sprite(combined_path)
    complete = finalize_sprite(
        add_sampler(combined),
        complete_path,
        sprite_name="tinystories-1200k complete",
        # Sampler and EDIT HERE procedure names are a public project interface.
        codegen=CodegenConfig(unrolling="compact", id_namespace=""),
    )
    complete_sprite = _read_sprite(complete.path)

    config = source.config
    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "model": {
            "parameters": sum(parameter.numel() for parameter in source.parameters()),
            "vocabulary": config.vocab_size,
            "context": context,
            "layers": config.n_layer,
            "heads": config.n_head,
            "kv_heads": config.n_kv_head,
            "width": config.n_embd,
            "hidden_width": config.n_hidden,
        },
        "export": {
            "optimization": "exact",
            "generation_wrapper": "cached dynamic RoPE",
            "weight_precision": "int8",
            "scale_precision": "float16",
            "compression": "costume-base92-huffman-int4",
            "unrolling": "compact",
            "compact_internal_names": {
                "processor_tokenizer_and_combined_internals": True,
                "final_assembly": False,
                "reason": (
                    "preserve the public TinyStories sampler and EDIT HERE "
                    "procedure names"
                ),
            },
            "id_namespace": "",
        },
        "sampling": {
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "frequency_penalty": FREQUENCY_PENALTY,
            "eos_id": EOS_ID,
            "stop": "EOS or total context length",
        },
        "broadcasts": [
            "cattorch init complete", "cattorch reset complete",
            "cattorch prefill complete", "cattorch decode complete",
        ],
        "artifacts": {
            "processor": {
                "path": processor.path.name,
                "archive_bytes": processor.archive_bytes,
                "expanded_json_bytes": processor.expanded_json_bytes,
                "sha256": _sha256(processor.path),
            },
            "tokenizer": {
                "path": tokenizer.path.name,
                "archive_bytes": tokenizer.archive_bytes,
                "expanded_json_bytes": tokenizer.expanded_json_bytes,
                "sha256": _sha256(tokenizer.path),
            },
            "complete": {
                "path": complete.path.name,
                "archive_bytes": complete.archive_bytes,
                "expanded_json_bytes": complete.expanded_json_bytes,
                "blocks": len(complete_sprite["blocks"]),
                "lists": len(complete_sprite["lists"]),
                "variables": len(complete_sprite["variables"]),
                "sha256": _sha256(complete.path),
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.checkpoint, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
