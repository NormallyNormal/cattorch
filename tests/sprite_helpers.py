"""Load exported sprites and run them in the emulator, for tests."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile

import torch
from torch import nn

from cattorch.transpiler import transpile
from cattorch.util.scratch.emulator import ScratchEmulator


def load_sprite(source) -> dict:
    """The sprite JSON of a ``.sprite3`` file, given its path or a transpile result."""
    with zipfile.ZipFile(getattr(source, "path", source)) as archive:
        return json.loads(archive.read("sprite.json"))


def load_emulator(source) -> ScratchEmulator:
    """An emulator for a ``.sprite3`` file, given its path or a transpile result."""
    return ScratchEmulator(load_sprite(source))


def run_sprite(model: nn.Module, x: torch.Tensor) -> tuple[list[float], list[float]]:
    """Transpile a model, run through emulator, return (expected, actual)."""
    model = model.eval()
    with torch.no_grad():
        expected = model(x).flatten().tolist()

    with tempfile.TemporaryDirectory(prefix="cattorch-sprite-") as directory:
        result = transpile(model, x, os.path.join(directory, "output"))
        emulator = load_emulator(result)
    emulator.lists["input"] = [float(v) for v in x.flatten().tolist()]
    emulator.run()
    actual = [float(v) for v in emulator.lists.get("output", [])]
    return expected, actual


def assert_outputs_close(expected: list[float], actual: list[float], tolerance=1e-4):
    """Each output within ``tolerance`` of the expected value, or equal to it."""
    assert len(expected) == len(actual), (
        f"Length mismatch: expected {len(expected)}, got {len(actual)}"
    )
    for i, (e, a) in enumerate(zip(expected, actual)):
        if e == a:
            continue
        assert abs(e - a) < tolerance, (
            f"Index {i}: expected {e:.6f}, got {a:.6f}"
        )
