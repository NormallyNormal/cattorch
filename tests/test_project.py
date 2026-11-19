"""Runnable projects from build_project, run whole in the project emulator."""

from __future__ import annotations

from cattorch import interface
import collections
import math

import pytest
import torch
from torch import nn

from cattorch import (
    CharTokenizer, GenerationProgram, build_project, entrypoint, transpile,
)
from cattorch.project import _sample
from cattorch.util.scratch.dsl import Program
from cattorch.util.scratch.emulator import ProjectEmulator
from shared_models import CHARACTERS, Language, char_tokenizer, greedy
from sprite_helpers import SpriteEmulator


def test_non_finite_example_values_are_written_as_strict_json(tmp_path):
    # Bare Infinity and NaN aren't JSON, and the Scratch VM refuses them.
    import json
    import zipfile

    value = torch.tensor([1.5, math.inf, -math.inf, math.nan])
    result = transpile(nn.Identity().eval(), value, tmp_path / "same")
    project = build_project(result, tmp_path / "same", example_inputs=value)
    with zipfile.ZipFile(project.path) as archive:
        stored = json.loads(archive.read("project.json"), parse_constant=pytest.fail)
    stage = stored["targets"][0]
    assert stage["lists"] and all(
        entry[1] == [1.5, "Infinity", "-Infinity", "NaN"] for entry in stage["lists"].values()
        if entry[0] == "same input"
    )
    emulator = ProjectEmulator(project.path)
    emulator.green_flag()
    # NaN isn't a number the sprite takes; it writes no output.
    assert emulator.stage.variables["same status"] == interface.invalid_item("same input")
    assert emulator.stage.lists["same output"] == []
    emulator.stage.lists["same input"] = [1.5, math.inf, -math.inf, 0]
    emulator.green_flag()
    output = [float(value) for value in emulator.stage.lists["same output"]]
    assert output == [1.5, math.inf, -math.inf, 0]


def test_a_classifier_runs_on_the_green_flag_and_space(tmp_path):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 6), nn.Tanh(), nn.Linear(6, 3)).eval()
    example = torch.randn(1, 3)
    result = transpile(model, example, tmp_path / "net")
    project = build_project(
        result, tmp_path / "net", example_inputs=example, labels=["a", "b", "c"],
    )
    assert project.path.suffix == ".sb3"
    assert project.sprites == ("Demo", "net")
    assert project.broadcasts == (
        "net forward", "net forward complete", "net init", "net init complete",
        "net prepare for save", "net prepare for save complete", "cattorch prepare for save",
        "net predict", "net predict complete",
    )
    assert set(project.lists) == {"net input", "net output", "net labels"}
    assert set(project.variables) == {"net status", "net prediction"}
    emulator = ProjectEmulator(project.path)
    emulator.green_flag()
    expected = model(example)[0]
    assert emulator.stage.lists["net output"] == pytest.approx(expected.tolist(), abs=1e-5)
    assert emulator.stage.variables["net prediction"] == "abc"[int(expected.argmax())]

    other = torch.randn(1, 3)
    emulator.stage.lists["net input"] = other.flatten().tolist()
    emulator.press("space")
    assert emulator.stage.lists["net output"] == pytest.approx(model(other)[0].tolist(), abs=1e-5)


def test_inputs_start_as_zeros_and_labels_must_match(tmp_path):
    model = nn.Linear(2, 2).eval()
    result = transpile(model, torch.ones(1, 2), tmp_path / "linear")
    project = build_project(result, tmp_path / "linear")
    emulator = ProjectEmulator(project.path)
    assert emulator.stage.lists["linear input"] == [0, 0]
    with pytest.raises(ValueError, match="labels has 3 entries"):
        build_project(result, tmp_path / "bad", labels=["a", "b", "c"])
    batch = transpile(model, torch.ones(3, 2), tmp_path / "batch", dynamic={0: (1, 4)})
    with pytest.raises(ValueError, match="variable-length dimension"):
        build_project(batch, tmp_path / "batch", labels=["a", "b"])


class Counter(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.register_buffer("total", torch.zeros(1, 2))

    @entrypoint(example_inputs=lambda: (torch.ones(1, 2),))
    def add(self, value):
        self.total += self.linear(value)
        return self.total.clone()


def test_entry_points_run_by_their_messages(tmp_path):
    torch.manual_seed(1)
    model = Counter().eval()
    result = transpile(model, None, tmp_path / "counter")
    project = build_project(result, tmp_path / "counter")
    assert set(project.broadcasts) == {
        "counter add", "counter add complete", "counter init", "counter init complete",
        "counter reset", "counter reset complete", "counter prepare for save",
        "counter prepare for save complete", "cattorch prepare for save",
    }
    emulator = ProjectEmulator(project.path)
    value = torch.tensor([[0.5, -1.0]])
    name = result.entrypoints[0].inputs[0].list_name
    assert name == "counter add value"
    emulator.stage.lists[name] = value.flatten().tolist()
    output = result.entrypoints[0].outputs[0].list_name
    for _ in range(2):
        emulator.broadcast("counter add")
    with torch.no_grad():
        expected = 2 * model.linear(value)
    assert emulator.stage.variables["counter status"] == interface.OK
    assert emulator.sprites["counter"].broadcasts[-1] == "counter add complete"
    assert emulator.stage.lists[output] == pytest.approx(
        expected.flatten().tolist(), abs=1e-5,
    )


@pytest.fixture(scope="module")
def generator(tmp_path_factory):
    torch.manual_seed(2)
    model = Language().eval()
    directory = tmp_path_factory.mktemp("generator")
    result = transpile(model, GenerationProgram(max_context=16), directory / "lm")
    tokenizer = CharTokenizer(char_tokenizer()).save(directory / "chars", name="chars")
    return model, result, tokenizer, directory


def test_a_text_generator_continues_each_prompt(generator):
    model, result, tokenizer, directory = generator
    project = build_project(
        result, directory / "demo", tokenizer=tokenizer, max_new_tokens=5, temperature=0,
    )
    emulator = ProjectEmulator(project.path, answers=["hi", "the cat", ""])
    emulator.green_flag()
    said = emulator.sprites["Demo"].said
    for prompt in ("hi", "the cat"):
        text = "".join(CHARACTERS[index] for index in greedy(model, prompt, 5))
        assert prompt + text in said
    assert len(emulator.questions) == 3
    # Asking replaces the speech bubble, so each next question shows the
    # finished continuation.
    for prompt, question in zip(("hi", "the cat"), emulator.questions[1:]):
        text = "".join(CHARACTERS[index] for index in greedy(model, prompt, 5))
        assert question.startswith(prompt + text + "\n\n")
    # The prefill decoded the weights; nothing else ran init.
    model_sprite = emulator.sprites[result.sprite_name]
    assert "lm init complete" not in model_sprite.broadcasts
    assert model_sprite.variables["cattorch initialized"] == 1


def test_generation_stops_at_the_end_token_and_the_context(generator):
    model, result, tokenizer, directory = generator
    first = greedy(model, "ab", 1)[0]
    project = build_project(
        result, directory / "eos", tokenizer=tokenizer, temperature=0, eos_token_id=first,
    )
    emulator = ProjectEmulator(project.path, answers=["ab", ""])
    emulator.green_flag()
    assert emulator.sprites["Demo"].said[-1] == "ab..."

    # 40 new tokens don't fit in a context of 16: generation stops when full.
    project = build_project(result, directory / "full", tokenizer=tokenizer, temperature=0)
    emulator = ProjectEmulator(project.path, answers=["abc", ""])
    emulator.green_flag()
    assert emulator.stage.variables["lm status"] == interface.CONTEXT_FULL
    # 16 cached tokens, plus the one their logits predict.
    assert len(emulator.sprites["Demo"].said[-1]) == 17


def test_sampling_follows_the_softmax(tmp_path):
    from cattorch.project import _BEST, _INDEX, _PICK, _TOTAL

    program = Program(
        "sample", lists=("logits",), variables=(_INDEX, _BEST, _TOTAL, _PICK, "heat"),
        variable_values={"heat": 0.5}, list_values={"logits": [0.0, 0.5, 1.0]},
        body=_sample("logits", "heat"),
    )
    emulator = SpriteEmulator(program.compile())
    torch.manual_seed(3)
    counts = collections.Counter()
    for _ in range(3000):
        emulator.run()
        counts[int(emulator.variables[_BEST])] += 1
    weights = [math.exp(value / 0.5) for value in (0.0, 0.5, 1.0)]
    for position, weight in enumerate(weights, 1):
        assert counts[position] / 3000 == pytest.approx(weight / sum(weights), abs=0.03)


def test_generators_need_their_tokenizer(generator):
    _, result, tokenizer, directory = generator
    with pytest.raises(ValueError, match="needs its tokenizer"):
        build_project(result, directory / "none")
    with pytest.raises(TypeError, match="result of cattorch.transpile"):
        build_project(tokenizer, directory / "wrong")


def test_names_a_project_cant_use(tmp_path):
    model, example = nn.Linear(2, 2).eval(), torch.randn(1, 2)
    # Its prepare for save would be the message every sprite answers.
    with pytest.raises(ValueError, match="every exported sprite answers"):
        transpile(model, example, tmp_path / "m", name="Cattorch")
    # The project's own sprite gives way: the model's name is its interface.
    result = transpile(model, example, tmp_path / "m", name="Demo")
    project = build_project(result, tmp_path / "project")
    assert project.sprites == ("Demo2", "Demo")
    assert set(project.lists) == {"Demo input", "Demo output"}
    result = transpile(model, example, tmp_path / "m", name="Stage")
    with pytest.raises(ValueError, match="named 'Stage'"):
        build_project(result, tmp_path / "project")


def test_two_sprites_of_one_name_or_global_are_refused(generator, tmp_path):
    from shared_models import char_tokenizer

    _, result, _, _ = generator
    words = CharTokenizer(char_tokenizer()).save(tmp_path / "words", name="lm")
    with pytest.raises(ValueError, match="named 'lm'"):
        build_project(result, tmp_path / "same", tokenizer=words)


@pytest.mark.parametrize(("options", "error", "expected"), [
    ({"max_new_tokens": 2.5}, TypeError, "max_new_tokens must be an integer"),
    ({"max_new_tokens": True}, TypeError, "max_new_tokens must be an integer"),
    ({"max_new_tokens": 0}, ValueError, "max_new_tokens must be positive"),
    ({"temperature": -1}, ValueError, "temperature must be a finite number of 0 or more"),
    ({"temperature": math.nan}, ValueError, "temperature must be a finite"),
    ({"temperature": math.inf}, ValueError, "temperature must be a finite"),
    ({"temperature": "0.5"}, TypeError, "temperature must be a number"),
    ({"eos_token_id": len(CHARACTERS)}, ValueError, "not a token ID of the model"),
    ({"eos_token_id": -1}, ValueError, "not a token ID of the model"),
    ({"eos_token_id": 1.0}, TypeError, "eos_token_id must be an integer"),
    ({"prompt": ""}, ValueError, "prompt must be a non-empty string"),
    ({"labels": ["a"]}, ValueError, "labels= applies only to a TranspileResult"),
    ({"example_inputs": torch.zeros(1, 2)}, ValueError, "example_inputs= applies only"),
])
def test_generation_project_options_are_checked(generator, options, error, expected):
    _, result, tokenizer, directory = generator
    with pytest.raises(error, match=expected):
        build_project(result, directory / "bad", tokenizer=tokenizer, **options)
    assert not (directory / "bad.sb3").exists()


def test_generation_project_without_its_tokenizer_mismatch_does_not_warn(generator):
    import warnings

    _, result, tokenizer, directory = generator
    assert result.vocabulary_size == tokenizer.token_count
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        project = build_project(result, directory / "matched", tokenizer=tokenizer)
    assert project.warnings == ()


def test_options_of_another_kind_of_project_are_rejected(tmp_path):
    torch.manual_seed(0)
    model = nn.Linear(4, 2).eval()
    example = torch.randn(1, 4)
    result = transpile(model, example, tmp_path / "net")
    for options, expected in (
        ({"temperature": 0.5}, "temperature= applies only to a text generator"),
        ({"max_new_tokens": 3}, "max_new_tokens= applies only"),
        ({"eos_token_id": 1}, "eos_token_id= applies only"),
        ({"prompt": "Say"}, "prompt= applies only"),
    ):
        with pytest.raises(ValueError, match=expected):
            build_project(result, tmp_path / "bad", **options)

    class Counter(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("total", torch.zeros(2))

        @entrypoint(example_inputs=(torch.ones(2),))
        def add(self, value):
            self.total = self.total + value
            return self.total

    program = transpile(Counter().eval(), None, tmp_path / "counter")
    for options, expected in (
        ({"labels": ["a", "b"]}, "labels= applies only to a TranspileResult"),
        ({"example_inputs": (torch.ones(2),)}, "example_inputs= applies only"),
        ({"temperature": 0}, "temperature= applies only"),
    ):
        with pytest.raises(ValueError, match=expected):
            build_project(program, tmp_path / "bad", **options)
    assert not list(tmp_path.glob("bad*"))


def test_example_inputs_must_fit_the_exported_inputs(tmp_path):
    model = nn.Linear(4, 2).eval()
    result = transpile(model, torch.randn(1, 4), tmp_path / "net")
    for example, error, expected in (
        (torch.randn(2, 4), ValueError, r"shape \(2, 4\), but 'net input' takes float32 of shape \(1, 4\)"),
        (torch.zeros(1, 4, dtype=torch.int64), ValueError, "int64 tensor"),
        ((torch.randn(1, 4), torch.randn(1, 4)), ValueError, "has 2 tensors, but the export has 1"),
        (([1.0, 2.0, 3.0, 4.0],), TypeError, "example input 0 is a list"),
        ({"x": torch.randn(1, 4)}, TypeError, "example_inputs must be a tensor or a tuple"),
    ):
        with pytest.raises(error, match=expected):
            build_project(result, tmp_path / "bad", example_inputs=example)
    with pytest.raises(TypeError, match="labels must be a sequence"):
        build_project(result, tmp_path / "bad", labels="ab")
    assert build_project(result, tmp_path / "ok", example_inputs=torch.randn(1, 4)).path.exists()
