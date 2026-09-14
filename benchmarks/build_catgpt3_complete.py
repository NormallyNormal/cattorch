#!/usr/bin/env python3
"""Build a complete CatGPT3 MoE processor, tokenizer, and sampler sprite."""

from __future__ import annotations

import argparse
import hashlib
import json
import site
import sys
import tempfile
import zipfile
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
CATGPT2 = ROOT.parent / "catgpt2"
sys.path.insert(0, str(ROOT))
for package_dir in (CATGPT2 / ".venv" / "lib").glob("python*/site-packages"):
    site.addsitedir(str(package_dir))
sys.path.insert(0, str(CATGPT2 / "src"))

from microchat.checkpoint import load_checkpoint, load_checkpoint_tokenizer  # noqa: E402
from microchat.model import RotaryEmbedding, Top1SwiGLUMoE  # noqa: E402

from benchmarks.add_tinystories_sampler import (  # noqa: E402
    EOS_ID,
    FREQUENCY_PENALTY,
    TEMPERATURE,
    TOP_K,
    add_sampler,
)
from benchmarks.combine_sprites import combine_sprites  # noqa: E402
from cattorch import (  # noqa: E402
    CodegenConfig,
    GenerationProgram,
    QuantizationConfig,
    rotary_embedding,
    transpile,
    transpile_tokenizer,
)
from cattorch.experimental import ModuleAdapter, stacked_swiglu_moe_adapter  # noqa: E402
from cattorch.util.scratch.finalize_scratch import finalize_sprite  # noqa: E402


DEFAULT_CHECKPOINT = (
    CATGPT2 / "runs" / "tinystories_6m_rsft_hq_r05" / "step_000500.pt"
)
DEFAULT_OUTPUT = CATGPT2 / "artifacts"
DEFAULT_STEM = "catgpt3-6m-rsft-hq-r05-step500"


class _ExportRotaryEmbedding(nn.Module):
    """Preserve dynamic RoPE tables behind cattorch's fused semantic op."""

    def __init__(self, source: RotaryEmbedding) -> None:
        super().__init__()
        self.register_buffer("cos", source.cos.detach().clone(), persistent=False)
        self.register_buffer("sin", source.sin.detach().clone(), persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        length = value.shape[-2]
        return rotary_embedding(
            value,
            self.cos[:, :, :length].to(dtype=value.dtype),
            self.sin[:, :, :length].to(dtype=value.dtype),
        )


def _rotary_adapter() -> ModuleAdapter:
    return ModuleAdapter(
        RotaryEmbedding,
        lambda module, _context: _ExportRotaryEmbedding(module),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sprite(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("sprite.json"))


def build(checkpoint: Path, output_dir: Path, stem: str = DEFAULT_STEM) -> dict:
    source, payload = load_checkpoint(checkpoint, device="cpu")
    source.eval()
    config = source.config
    if config.block_size != 256:
        raise ValueError(f"expected a 256-token checkpoint, got {config.block_size}")
    if not any(isinstance(module, Top1SwiGLUMoE) for module in source.modules()):
        raise ValueError("CatGPT3 export requires Top1SwiGLUMoE expert blocks")

    output_dir.mkdir(parents=True, exist_ok=True)
    processor_path = output_dir / f"{stem}-q4.sprite3"
    tokenizer_path = output_dir / f"{stem}-tokenizer.sprite3"
    complete_path = output_dir / f"{stem}-q4-complete.sprite3"
    manifest_path = output_dir / f"{stem}-q4-complete.manifest.json"

    compact_codegen = CodegenConfig(
        unrolling="compact",
        compact_internal_names=True,
        compact_schema=True,
        id_namespace="",
        layer_sharing="auto",
    )
    quantization = QuantizationConfig(
        bits=4,
        method="symmetric",
        group_size=256,
        scale_precision="float16",
        min_quantized_values=1,
    )
    adapters = (
        stacked_swiglu_moe_adapter(Top1SwiGLUMoE),
        _rotary_adapter(),
    )
    processor = transpile(
        source,
        GenerationProgram(
            method="next_token_logits",
            example_token=torch.tensor([[1]]),
            max_context=config.block_size,
        ),
        processor_path,
        name=f"{stem} q4",
        optimization="exact",
        quantization=quantization,
        codegen=compact_codegen,
        adapters=adapters,
    )
    tokenizer = transpile_tokenizer(
        load_checkpoint_tokenizer(payload),
        tokenizer_path,
        name=f"{stem} tokenizer",
        scratch_casefold=True,
        codegen=compact_codegen,
    )

    with tempfile.TemporaryDirectory() as directory:
        combined_path = Path(directory) / "combined.sprite3"
        combine_sprites(
            processor.path,
            tokenizer.path,
            combined_path,
            name=f"{stem} combined",
        )
        combined = _read_sprite(combined_path)
    complete = finalize_sprite(
        add_sampler(combined),
        complete_path,
        sprite_name=f"{stem} complete",
        # Preserve sampler, lifecycle, and EDIT HERE public procedure names.
        codegen=CodegenConfig(unrolling="compact", id_namespace=""),
    )
    complete_sprite = _read_sprite(complete.path)

    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "model": {
            "parameters": sum(parameter.numel() for parameter in source.parameters()),
            "active_parameters": payload.get("metadata", {}).get("active_parameters"),
            "vocabulary": config.vocab_size,
            "context": config.block_size,
            "layers": config.n_layer,
            "heads": config.n_head,
            "kv_heads": config.n_kv_head,
            "width": config.n_embd,
            "hidden_width": config.n_hidden,
            "experts": config.n_expert,
            "top_k_experts": config.moe_top_k,
            "router_gate": config.moe_router_gate,
        },
        "export": {
            "optimization": "exact",
            "generation_method": "next_token_logits",
            "generation_wrapper": "cached MQA with fused semantic RoPE",
            "quantization": "symmetric q4",
            "group_size": 256,
            "scale_precision": "float16",
            "compression": "costume-base92 with canonical Huffman q4",
            "unrolling": "compact",
            "compact_schema": True,
            "compact_internal_names": True,
            "layer_sharing": "auto",
        },
        "sampling": {
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "frequency_penalty": FREQUENCY_PENALTY,
            "eos_id": EOS_ID,
            "stop": "EOS or total context length",
        },
        "broadcasts": [
            "cattorch init complete",
            "cattorch reset complete",
            "cattorch prefill complete",
            "cattorch decode complete",
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
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    args = parser.parse_args()
    print(json.dumps(build(args.checkpoint, args.output_dir, args.stem), indent=2))


if __name__ == "__main__":
    main()
