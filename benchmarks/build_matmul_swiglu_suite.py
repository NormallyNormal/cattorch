"""Generate focused static-matmul and exact SwiGLU-fusion benchmarks."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from cattorch import FastConfig, FastLayerConfig, build_benchmark_suite


OUTPUT = Path(__file__).parent / "artifacts" / "matmul_swiglu_suite.sb3"


class StaticMatmul(nn.Module):
    def __init__(self, inner, columns):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(inner, columns))

    def forward(self, x):
        return x @ self.weight


class FusedSwiGLU(nn.Module):
    def forward(self, gate, value):
        return F.silu(gate) * value


class MaterializedSwiGLU(nn.Module):
    def forward(self, gate, value):
        # clone is a flat-data alias in cattorch, but prevents graph fusion so
        # the benchmark materializes SiLU before the multiplication traversal.
        return F.silu(gate).clone() * value


def _weight_only(policy):
    return FastConfig(
        activations=False,
        layer_norm=False,
        softmax=False,
        weights=policy,
    )


def build_cases():
    torch.manual_seed(47)
    matrix_input = torch.randn(32, 32)
    gate = torch.randn(128, 128)
    value = torch.randn(128, 128)
    return [
        (
            "static_matmul_prune50",
            StaticMatmul(32, 96).eval(),
            matrix_input,
            _weight_only(FastLayerConfig(pruning=0.5)),
        ),
        (
            "static_matmul_rank25",
            StaticMatmul(32, 96).eval(),
            matrix_input,
            _weight_only(FastLayerConfig(rank_ratio=0.25)),
        ),
        ("swiglu_fused", FusedSwiGLU(), (gate, value)),
        ("swiglu_materialized", MaterializedSwiGLU(), (gate, value)),
    ]


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    path = build_benchmark_suite(
        build_cases(),
        OUTPUT,
        iterations=100,
    )
    print(f"Static-matmul/SwiGLU suite written: {path}")


if __name__ == "__main__":
    main()
