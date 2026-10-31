"""Benchmark projects run: each green flag fills the timing lists."""

from __future__ import annotations

import torch
import torch.nn as nn

from cattorch import build_benchmark_suite, build_paired_benchmark, build_storage_benchmark_suite
from cattorch.util.scratch.emulator import ProjectEmulator


def _stage_list(emulator, name):
    return emulator.stage.lists[name]


def test_a_paired_benchmark_records_each_repeat(tmp_path):
    path = build_paired_benchmark(
        nn.Linear(3, 2).eval(), torch.randn(2, 3), tmp_path / "paired.sb3", warmups=1, repeats=3,
    )
    emulator = ProjectEmulator(path)
    emulator.green_flag()
    for name in ("cattorch exact timings", "cattorch fast timings"):
        timings = _stage_list(emulator, name)
        assert len(timings) == 3 and all(float(value) >= 0 for value in timings)


def test_a_suite_records_each_case_and_variant(tmp_path):
    path = build_benchmark_suite(
        [("first", nn.Linear(3, 2).eval(), torch.randn(2, 3))], tmp_path / "suite.sb3",
        iterations=2,
    )
    emulator = ProjectEmulator(path)
    emulator.green_flag()
    results = _stage_list(emulator, "cattorch benchmark results")
    assert [entry.split(":")[0] for entry in results] == ["first exact", "first fast"]
    assert all(float(entry.split(": ")[1]) >= 0 for entry in results)


def test_a_storage_suite_records_init_and_forward(tmp_path):
    path = build_storage_benchmark_suite(
        [("tiny", nn.Linear(3, 2).eval(), torch.randn(1, 3))], tmp_path / "storage",
        iterations=1,
    )
    emulator = ProjectEmulator(path)
    emulator.green_flag()
    labels = [entry.split(":")[0] for entry in _stage_list(emulator, "cattorch benchmark results")]
    assert "tiny plain-f32 init" in labels and "tiny plain-f32 forward" in labels
