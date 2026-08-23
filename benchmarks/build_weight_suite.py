"""Generate one Scratch suite for structured pruning and low-rank weights."""

from pathlib import Path

import torch
import torch.nn as nn

from cattorch import FastConfig, FastLayerConfig, build_benchmark_suite
from build_suite import TinyGPT


OUTPUT = Path(__file__).parent / "artifacts" / "fast_weight_suite.sb3"


def _weight_only(policy: FastLayerConfig) -> FastConfig:
    return FastConfig(
        activations=False,
        layer_norm=False,
        softmax=False,
        weights=policy,
    )


def build_cases():
    torch.manual_seed(42)
    models = [
        ("linear", nn.Linear(32, 96), torch.randn(32, 32)),
        ("conv1d", nn.Conv1d(4, 8, 3, padding=1), torch.randn(1, 4, 64)),
        ("conv2d", nn.Conv2d(4, 8, 3, padding=1), torch.randn(1, 4, 16, 16)),
        ("quick_gpt", TinyGPT(16, 2, 32, 8), torch.randint(0, 29, (1, 8))),
        ("catgpt1", TinyGPT(32, 2, 64, 32), torch.randint(0, 29, (1, 32))),
    ]
    pruning = _weight_only(FastLayerConfig(pruning=0.5))
    low_rank = _weight_only(FastLayerConfig(rank_ratio=0.25))
    cases = []
    for name, model, example_input in models:
        model.eval()
        cases.extend((
            (f"{name}_prune50", model, example_input, pruning),
            (f"{name}_rank25", model, example_input, low_rank),
        ))
    return cases


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    path = build_benchmark_suite(
        build_cases(),
        OUTPUT,
        iterations=100,
    )
    print(f"Fast weight suite written: {path}")


if __name__ == "__main__":
    main()
