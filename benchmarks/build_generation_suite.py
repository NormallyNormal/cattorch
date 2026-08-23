"""Generate one sequential stateless-forward versus KV-cache Scratch suite."""

from pathlib import Path

import torch

from cattorch import build_generation_benchmark_suite
from build_suite import TinyGPT


OUTPUT = Path(__file__).parent / "artifacts" / "generation_suite.sb3"


def main():
    torch.manual_seed(44)
    context = 32
    prompt_length = 16
    cases = []
    for name, width, heads, hidden in (
        ("quick_gpt_generation", 16, 2, 32),
        ("catgpt1_generation", 32, 2, 64),
    ):
        model = TinyGPT(width, heads, hidden, context).eval()
        full = torch.randint(0, 29, (1, context))
        cases.append((name, model, full, full[:, :prompt_length]))
    path = build_generation_benchmark_suite(cases, OUTPUT, iterations=16)
    print(f"Generation suite written: {path}")


if __name__ == "__main__":
    main()
