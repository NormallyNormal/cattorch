"""The scripts in examples/ run end to end, with short training."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cattorch.util.scratch.emulator import ProjectEmulator

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load(name):
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["mlp", "cnn"])
def test_classifier_examples(tmp_path, name):
    example = load(name)
    project = example.main(tmp_path, steps=150)
    emulator = ProjectEmulator(project.path)
    emulator.green_flag()
    (prediction,) = (name for name in project.variables if name.endswith("prediction"))
    (output,) = (name for name in project.lists if name.endswith("output"))
    scores = emulator.stage.lists[output]
    label = example.LABELS[max(range(len(scores)), key=scores.__getitem__)]
    assert emulator.stage.variables[prediction] == label
    if name == "mlp":
        # The example point (0.5, -0.25) is to the right and below.
        assert label == "bottom right"


def test_text_generator_example(tmp_path):
    example = load("text_generator")
    project = example.main(tmp_path, steps=20, max_new_tokens=3)
    emulator = ProjectEmulator(project.path, answers=["twinkle", ""])
    emulator.green_flag()
    said = emulator.sprites["Demo"].said
    continued = [text for text in said if text.startswith("twinkle") and text != "twinkle..."]
    assert len(continued[-1]) == len("twinkle") + 3
