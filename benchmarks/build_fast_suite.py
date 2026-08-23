"""Generate one exact/fast Scratch project for arithmetic approximations."""

from pathlib import Path

import torch

from cattorch import build_benchmark_suite
from build_kernel_suite import build_cases as build_kernel_cases
from build_suite import build_cases as build_model_cases


OUTPUT = Path(__file__).parent / "artifacts" / "fast_suite.sb3"


def main():
    torch.manual_seed(123)
    cases = [*build_kernel_cases(), *build_model_cases()]
    path = build_benchmark_suite(
        cases,
        OUTPUT,
        iterations=100,
    )
    print(f"Exact/fast suite written: {path}")


if __name__ == "__main__":
    main()
