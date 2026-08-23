"""Generate one sequential storage-size and startup-cost Scratch suite."""

from pathlib import Path

import torch

from cattorch import build_storage_benchmark_suite
from build_suite import ParameterMatMul, TinyGPT


OUTPUT = Path(__file__).parent / "artifacts" / "storage_suite.sb3"


def main():
    torch.manual_seed(47)
    cases = [
        ("matmul_256x256", ParameterMatMul(256, 256), torch.randn(1, 256)),
        ("quick_gpt", TinyGPT(16, 2, 32, 8).eval(), torch.randint(0, 29, (1, 8))),
        ("catgpt1", TinyGPT(32, 2, 64, 32).eval(), torch.randint(0, 29, (1, 32))),
    ]
    path = build_storage_benchmark_suite(cases, OUTPUT, iterations=10)
    print(f"Storage suite written: {path}")


if __name__ == "__main__":
    main()
