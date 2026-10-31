"""Runnable projects from build_project, run whole in the project emulator."""

from __future__ import annotations

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
from cattorch.util.scratch.emulator import ProjectEmulator, ScratchEmulator


def test_a_classifier_runs_on_the_green_flag_and_space(tmp_path):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 6), nn.Tanh(), nn.Linear(6, 3)).eval()
    example = torch.randn(1, 3)
    result = transpile(model, example, tmp_path / "net")
    project = build_project(
        tmp_path / "net", result, example_inputs=example, labels=["a", "b", "c"],
    )
    assert project.path.suffix == ".sb3"
    assert project.messages == ("net run", "net predict")
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
    project = build_project(tmp_path / "linear", result)
    emulator = ProjectEmulator(project.path)
    assert emulator.stage.lists["linear input"] == [0, 0]
    with pytest.raises(ValueError, match="labels has 3 entries"):
        build_project(tmp_path / "bad", result, labels=["a", "b", "c"])
    batch = transpile(model, torch.ones(3, 2), tmp_path / "batch", dynamic={0: (1, 4)})
    with pytest.raises(ValueError, match="variable-length dimension"):
        build_project(tmp_path / "batch", batch, labels=["a", "b"])


class Counter(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.register_buffer("total", torch.zeros(1, 2))

    @entrypoint(example=lambda: (torch.ones(1, 2),))
    def add(self, value):
        self.total += self.linear(value)
        return self.total.clone()


def test_entry_points_are_bridged(tmp_path):
    torch.manual_seed(1)
    model = Counter().eval()
    result = transpile(model, None, tmp_path / "counter")
    project = build_project(tmp_path / "counter", result)
    assert project.messages == ("counter add",)
    emulator = ProjectEmulator(project.path)
    value = torch.tensor([[0.5, -1.0]])
    name = result.entrypoints[0].inputs[0].list_name
    emulator.stage.lists[f"counter {name}"] = value.flatten().tolist()
    output = result.entrypoints[0].outputs[0].list_name
    for _ in range(2):
        emulator.broadcast("counter add")
    with torch.no_grad():
        expected = 2 * model.linear(value)
    assert emulator.stage.lists[f"counter {output}"] == pytest.approx(
        expected.flatten().tolist(), abs=1e-5,
    )


CHARACTERS = "?abcdefghijklmnopqrstuvwxyz ."


def char_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel(
        vocab={character: index for index, character in enumerate(CHARACTERS)}, unk_token="?",
    ))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="?")


class Language(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = nn.Embedding(len(CHARACTERS), 8)
        self.positions = nn.Embedding(16, 8)
        self.layer = nn.TransformerEncoderLayer(8, 2, 16, dropout=0.0, batch_first=True)
        self.head = nn.Linear(8, len(CHARACTERS))

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.tokens(tokens) + self.positions(torch.arange(length))
        mask = nn.Transformer.generate_square_subsequent_mask(length)
        return self.head(self.layer(x, src_mask=mask, is_causal=True))


@pytest.fixture(scope="module")
def generator(tmp_path_factory):
    torch.manual_seed(2)
    model = Language().eval()
    directory = tmp_path_factory.mktemp("generator")
    result = transpile(model, GenerationProgram(max_context=16), directory / "lm")
    tokenizer = CharTokenizer(char_tokenizer()).save(directory / "chars", name="chars")
    return model, result, tokenizer, directory


def greedy(model, prompt, count):
    ids = [CHARACTERS.index(character) for character in prompt]
    with torch.no_grad():
        for _ in range(count):
            ids.append(int(model(torch.tensor([ids]))[0, -1].argmax()))
    return ids[len(prompt):]


def test_a_text_generator_continues_each_prompt(generator):
    model, result, tokenizer, directory = generator
    project = build_project(
        directory / "demo", result, tokenizer=tokenizer, max_new_tokens=5, temperature=0,
    )
    emulator = ProjectEmulator(project.path, answers=["hi", "the cat", ""])
    emulator.green_flag()
    said = emulator.sprites["Demo"].said
    for prompt in ("hi", "the cat"):
        text = "".join(CHARACTERS[index] for index in greedy(model, prompt, 5))
        assert prompt + text in said
    assert len(emulator.questions) == 3
    # The weights are decoded once, not before every token.
    model_sprite = next(name for name in emulator.sprites if name not in {"Demo", tokenizer.sprite_name})
    assert emulator.sprites[model_sprite].broadcasts.count("cattorch init complete") == 1


def test_generation_stops_at_the_end_token_and_the_context(generator):
    model, result, tokenizer, directory = generator
    first = greedy(model, "ab", 1)[0]
    project = build_project(
        directory / "eos", result, tokenizer=tokenizer, temperature=0, eos_token_id=first,
    )
    emulator = ProjectEmulator(project.path, answers=["ab", ""])
    emulator.green_flag()
    assert emulator.sprites["Demo"].said[-1] == "ab..."

    # 40 new tokens don't fit in a context of 16: generation stops when full.
    project = build_project(directory / "full", result, tokenizer=tokenizer, temperature=0)
    emulator = ProjectEmulator(project.path, answers=["abc", ""])
    emulator.green_flag()
    assert emulator.stage.variables["lm status"] == "maximum context exceeded"
    # 16 cached tokens, plus the one their logits predict.
    assert len(emulator.sprites["Demo"].said[-1]) == 17


def test_sampling_follows_the_softmax(tmp_path):
    from cattorch.project import _BEST

    program = Program(
        "sample", lists=("logits",), variables=("_cattorch bridge index", _BEST,
                                                "_cattorch bridge total",
                                                "_cattorch bridge pick", "heat"),
        variable_values={"heat": 0.5}, list_values={"logits": [0.0, 0.5, 1.0]},
        body=_sample("logits", "heat"),
    )
    emulator = ScratchEmulator(program.compile())
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
        build_project(directory / "none", result)
    with pytest.raises(TypeError, match="result of cattorch.transpile"):
        build_project(directory / "wrong", tokenizer)


@pytest.mark.parametrize("name", ["cattorch", "Demo", "Stage"])
def test_reserved_sprite_names_are_renamed(tmp_path, name):
    result = transpile(nn.Linear(2, 2).eval(), torch.randn(1, 2), tmp_path / "m", name=name)
    project = build_project(tmp_path / "project", result)
    assert name not in project.sprites
    assert len(set(project.sprites)) == len(project.sprites)
    assert not any(entry.startswith("cattorch ") for entry in project.lists)
