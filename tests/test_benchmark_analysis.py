"""Benchmark projects saved before a run finished should fail with a clear message."""

from __future__ import annotations

import json
import zipfile

import pytest

from cattorch.benchmark import (
    STORAGE_SIZES, STORAGE_VARIANTS, SUITE_RESULTS, analyze_benchmark,
)


def _project(path, stage_lists, targets):
    project = {"targets": [
        {
            "isStage": True, "name": "Stage", "variables": {},
            "lists": {
                f"list{index}": [name, values]
                for index, (name, values) in enumerate(stage_lists.items())
            },
        },
        *targets,
    ]}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("project.json", json.dumps(project))
    return path


def test_partly_run_storage_suite_reports_missing_result(tmp_path):
    path = _project(
        tmp_path / "storage.sb3",
        {
            SUITE_RESULTS: ["case plain-f32 init: 0.5"],
            STORAGE_SIZES: [f"case {variant}: 100 bytes" for variant in STORAGE_VARIANTS],
        },
        [
            {"isStage": False, "name": f"case {variant}", "lists": {}}
            for variant in STORAGE_VARIANTS
        ],
    )
    with pytest.raises(ValueError, match="has not finished"):
        analyze_benchmark(path)


def test_non_numeric_benchmark_output_is_reported(tmp_path):
    path = _project(
        tmp_path / "suite.sb3",
        {SUITE_RESULTS: ["case exact: 1.0", "case fast: 0.5"]},
        [
            {"isStage": False, "name": "case exact",
             "lists": {"out": ["output", ["1", "oops"]]}},
            {"isStage": False, "name": "case fast", "lists": {"out": ["output", ["1", "2"]]}},
        ],
    )
    with pytest.raises(ValueError, match="non-numeric output"):
        analyze_benchmark(path)
