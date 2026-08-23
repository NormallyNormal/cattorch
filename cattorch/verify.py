"""Numerically verify generated Scratch sprites against PyTorch."""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import torch

from cattorch.errors import UnsupportedModelError
from cattorch.results import TranspileResult, VerifyResult
from cattorch.util.scratch.emulator import ScratchEmulator
from cattorch.util.scratch.sharding import SCRATCH_LIST_LIMIT


def _normalise_inputs(example_inputs):
    return (example_inputs,) if isinstance(example_inputs, torch.Tensor) else tuple(example_inputs)


def _set_logical_list(emulator: ScratchEmulator, name: str, values: list) -> None:
    for position, start in enumerate(range(0, max(1, len(values)), SCRATCH_LIST_LIMIT)):
        shard_name = name if position == 0 else f"{name} shard {position + 1}"
        if shard_name not in emulator.lists:
            raise UnsupportedModelError(
                f"Generated sprite is missing expected input list {shard_name!r}"
            )
        emulator.lists[shard_name] = values[start:start + SCRATCH_LIST_LIMIT]


def _get_logical_list(emulator: ScratchEmulator, name: str) -> list:
    values = []
    position = 0
    while True:
        shard_name = name if position == 0 else f"{name} shard {position + 1}"
        if shard_name not in emulator.lists:
            break
        values.extend(emulator.lists[shard_name])
        position += 1
    return values


def verify(
    model: torch.nn.Module,
    example_inputs: torch.Tensor | tuple[torch.Tensor, ...],
    sprite: str | Path | TranspileResult,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-5,
) -> VerifyResult:
    """Run one generated forward in the emulator and compare it with PyTorch.

    This checks numerical lowering and lossy export settings; it does not
    estimate browser Scratch performance.
    """
    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative")
    inputs = _normalise_inputs(example_inputs)
    if not inputs or any(not isinstance(value, torch.Tensor) for value in inputs):
        raise TypeError("example_inputs must be a tensor or tuple of tensors")

    path = sprite.path if isinstance(sprite, TranspileResult) else Path(sprite)
    with zipfile.ZipFile(path) as archive:
        project = json.loads(archive.read("sprite.json"))
    emulator = ScratchEmulator(project)
    for index, value in enumerate(inputs):
        name = "input" if index == 0 else f"input_{index}"
        _set_logical_list(emulator, name, value.detach().cpu().flatten().tolist())
    emulator.run_procedure("cattorch forward")

    with torch.no_grad():
        expected = model(*inputs)
    if not isinstance(expected, torch.Tensor):
        raise UnsupportedModelError("verify() requires a model with one tensor output")
    expected_values = expected.detach().cpu().to(torch.float64).flatten()
    raw_actual = _get_logical_list(emulator, "output")
    try:
        actual_values = torch.tensor([float(value) for value in raw_actual], dtype=torch.float64)
    except (TypeError, ValueError) as error:
        raise UnsupportedModelError("Generated output contains a non-numeric Scratch value") from error

    expected_count = expected_values.numel()
    actual_count = actual_values.numel()
    if expected_count != actual_count:
        return VerifyResult(
            passed=False,
            values_compared=min(expected_count, actual_count),
            expected_shape=tuple(expected.shape),
            actual_values=actual_count,
            max_abs_error=math.inf,
            max_rel_error=math.inf,
            mean_abs_error=math.inf,
            worst_index=None,
        )

    close = torch.isclose(actual_values, expected_values, atol=atol, rtol=rtol, equal_nan=True)
    difference = torch.abs(actual_values - expected_values)
    same_nonfinite = (
        (torch.isnan(actual_values) & torch.isnan(expected_values))
        | (actual_values == expected_values)
    )
    difference = torch.where(same_nonfinite, 0.0, difference)
    difference = torch.nan_to_num(difference, nan=math.inf, posinf=math.inf, neginf=math.inf)
    denominator = torch.abs(expected_values)
    relative = torch.where(
        denominator == 0,
        torch.where(difference == 0, 0.0, math.inf),
        difference / denominator,
    )
    worst_index = int(torch.argmax(difference).item()) if expected_count else None
    return VerifyResult(
        passed=bool(torch.all(close).item()),
        values_compared=expected_count,
        expected_shape=tuple(expected.shape),
        actual_values=actual_count,
        max_abs_error=float(torch.max(difference).item()) if expected_count else 0.0,
        max_rel_error=float(torch.max(relative).item()) if expected_count else 0.0,
        mean_abs_error=float(torch.mean(difference).item()) if expected_count else 0.0,
        worst_index=worst_index,
    )
