"""The versioned Scratch interface: docs/scratch-interface.md against real sprites.

The contract page is the source of truth. These tests read its tables and
check, for a sprite of every kind, that each name it documents is in the
sprite and that every other name in the sprite is internal (starts with
``_``), and that ``cattorch.interface``, where the code takes the names and
status values from, agrees with it. Within interface version 1 no public
name or status value may be renamed or removed: if a check fails because one
disappeared, that breaks Scratch projects, so restore it. A new name is
added to the page.

The status values, and the public names below, are written out only here;
other tests take them from ``cattorch.interface``.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pytest
import torch
from torch import nn

from cattorch import (
    BPETokenizer, CharTokenizer, EntryPoint, ExportProgram, GenerationProgram, Input, Output,
    RecurrentGenerationProgram, SentencePieceBPETokenizer, State, StateInput, StateUpdate,
    build_project, entrypoint, interface, transpile,
)
from cattorch.util.scratch.archive import PROJECT_JSON, read_archive
from cattorch.util.scratch.emulator import ScratchEmulator

PAGE = Path(__file__).parents[1] / "docs" / "scratch-interface.md"

# The frozen status values, by situation, and what the code writes for each.
STATUSES = {
    "ok": interface.OK,
    "wrong number of items: LIST": interface.wrong_number_of_items("LIST"),
    "too few items: LIST": interface.too_few_items("LIST"),
    "too many items: LIST": interface.too_many_items("LIST"),
    "invalid item: LIST": interface.invalid_item("LIST"),
    "state full: LIST": interface.state_full("LIST"),
    "context full": interface.CONTEXT_FULL,
    "another call ran during this one": interface.OVERTAKEN,
    "an earlier call was stopped or is still running; reset first": interface.INTERRUPTED,
}
# Names whose meaning other tests rely on, written out once.
NAMES = {
    "cattorch status": interface.STATUS_VARIABLE,
    "cattorch initialized": interface.INITIALIZED_VARIABLE,
    "cattorch interface version": interface.VERSION_VARIABLE,
    "cattorch token ids": interface.TOKEN_IDS,
    "cattorch logits": interface.LOGITS,
    "cattorch text": interface.TEXT,
    "cattorch init": interface.INIT,
    "cattorch forward": interface.FORWARD,
    "cattorch prepare for save": interface.PREPARE_FOR_SAVE,
    "cattorch reset": interface.RESET,
    "cattorch prefill": interface.PREFILL,
    "cattorch decode": interface.DECODE,
    "cattorch tokenize": interface.TOKENIZE,
    "cattorch detokenize": interface.DETOKENIZE,
    "cattorch forward complete": interface.completion(interface.FORWARD),
    "cattorch bridge points run": interface.bridge_procedure("points run"),
    "story token ids": interface.project_name("story", interface.TOKEN_IDS),
}
KINDS = {"block": "procedures", "list": "data", "variable": "data", "broadcast": "broadcasts"}


def sections() -> dict[str, list[tuple[list[str], str]]]:
    """Each ``##`` section's table rows, as the names of the first column and the second."""
    found: dict[str, list[tuple[list[str], str]]] = {}
    title = ""
    for line in PAGE.read_text().splitlines():
        if line.startswith("## "):
            title = line[3:].strip()
            found[title] = []
        elif line.startswith("|") and not line.startswith(("|---", "| Name", "| Status")):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            found[title].append((re.findall(r"`([^`]+)`", cells[0]), cells[1]))
    return found


def documented(title: str) -> dict[str, set[str]]:
    names: dict[str, set[str]] = {"procedures": set(), "data": set(), "broadcasts": set()}
    for row, kind in sections()[title]:
        names[KINDS[kind]].update(row)
    return names


def load(path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("sprite.json"))


def sprite_names(sprite: dict) -> dict[str, set[str]]:
    return {
        "procedures": {
            block["mutation"]["proccode"] for block in sprite["blocks"].values()
            if isinstance(block, dict) and block.get("opcode") == "procedures_prototype"
        },
        "data": {
            entry[0] for section in ("variables", "lists") for entry in sprite[section].values()
        },
        "broadcasts": set(sprite.get("broadcasts", {}).values()),
    }


def expand(names: dict[str, set[str]], **values) -> dict[str, set[str]]:
    """``names`` with each placeholder (ENTRY, STATE, ...) replaced by every value given."""
    expanded = {}
    for kind, patterns in names.items():
        result = set()
        for pattern in patterns:
            options = [pattern]
            for placeholder, choices in values.items():
                if placeholder in pattern:
                    options = [
                        option.replace(placeholder, choice)
                        for option in options for choice in choices
                    ]
            result.update(options)
        expanded[kind] = result
    return expanded


def check(sprite: dict, expected: dict[str, set[str]]) -> None:
    """The sprite has every expected name, and every other one is internal."""
    actual = sprite_names(sprite)
    for kind in expected:
        public = {name for name in actual[kind] if not name.startswith(interface.INTERNAL_PREFIX)}
        if kind == "data":
            public = {name for name in public if " shard " not in name}
        assert public == expected[kind], kind
    for section in ("variables", "lists"):
        # One slot per name, or a user's block could pick the wrong copy.
        listed = [entry[0] for entry in sprite[section].values()]
        assert len(listed) == len(set(listed)), section
    variables = {entry[0]: entry[1] for entry in sprite["variables"].values()}
    assert variables["cattorch interface version"] == 1
    assert variables["cattorch status"] == ""
    if "cattorch initialized" in variables:
        assert variables["cattorch initialized"] == 0


def every(storage: bool = True) -> dict[str, set[str]]:
    names = documented("Every sprite")
    if not storage:
        names["data"] = {name for name in names["data"] if name == "cattorch interface version"}
    return names


def union(*parts: dict[str, set[str]]) -> dict[str, set[str]]:
    return {kind: set().union(*(part[kind] for part in parts)) for kind in parts[0]}


def model_names(result) -> dict[str, set[str]]:
    names = documented("Model sprites")
    names["data"] = {name for name in names["data"] if not re.fullmatch(r"(input|output)(_1)?", name)}
    names["data"] |= {spec.list_name for spec in (*result.inputs, *result.outputs)}
    return union(names, every())


def test_the_status_table_is_the_frozen_one():
    rows = {row[0] for row, _ in sections()["Status values"]}
    assert rows == set(STATUSES)
    for documented_value, value in STATUSES.items():
        assert value == documented_value


def test_the_fixed_names_are_the_documented_ones():
    names = union(*(documented(title) for title in (
        "Model sprites", "Generation sprites", "Program sprites", "Tokenizer sprites",
        "Every sprite",
    )))
    fixed = {kind: {name for name in found if not re.search(r"[A-Z]", name)} for kind, found in names.items()}
    assert fixed["procedures"] == interface.PUBLIC_PROCEDURES
    assert fixed["data"] - {"input", "input_1", "output", "output_1"} == interface.PUBLIC_DATA
    assert fixed["broadcasts"] == interface.PUBLIC_BROADCASTS
    assert set(interface.STORAGE_VARIABLES) <= fixed["data"]
    for documented_name, name in NAMES.items():
        assert name == documented_name


class Sums(nn.Module):
    def forward(self, value, index):
        return value.sum(), torch.nn.functional.embedding(index, value)


@pytest.mark.parametrize("kind", ["fixed", "variable length", "per item"])
def test_a_model_sprite(tmp_path, kind):
    if kind == "fixed":
        model, example, options = Sums().eval(), (torch.randn(3, 2), torch.tensor([[1, 2]])), {}
    elif kind == "variable length":
        model, example, options = nn.Linear(3, 2).eval(), torch.randn(2, 3), {"dynamic": {0: (1, 4)}}
    else:
        model = nn.TransformerEncoderLayer(4, 1, 8, dropout=0.0, batch_first=True).eval()
        example, options = torch.randn(2, 3, 4), {"dynamic": {0: (1, 4)}}
    result = transpile(model, example, tmp_path / "model", **options)
    check(load(result.path), model_names(result))


@pytest.mark.parametrize("recurrent", [False, True])
def test_a_generation_sprite(tmp_path, recurrent):
    from test_project import Language
    from test_recurrent_generation import GRULM

    if recurrent:
        model = GRULM().eval()
        program = RecurrentGenerationProgram(model.states(), top_k=3)
    else:
        model, program = Language().eval(), GenerationProgram(max_context=16, top_k=3)
    result = transpile(model, program, tmp_path / "lm")
    check(load(result.path), union(documented("Generation sprites"), every()))


class Memory(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("seen", torch.zeros(1, 3))

    @entrypoint(example_inputs=lambda: (torch.ones(3),), capacity={"seen": 8})
    def add(self, row):
        self.seen = torch.cat([self.seen, row.reshape(1, 3)])
        return self.seen.sum(0)

    @entrypoint(example_inputs=lambda: (torch.ones(2, 3),), dynamic={"rows": (1, 4)})
    def latest(self, rows):
        return rows @ self.seen[-1]

    def step(self, value, total):
        total = total + value
        return total * 2, total


def program_names(result) -> dict[str, set[str]]:
    names = documented("Program sprites")
    entries = result.entrypoints
    expanded = expand(
        {
            kind: {name for name in found if "INPUT" not in name and "OUTPUT" not in name}
            for kind, found in names.items()
        },
        ENTRY=[entry.name for entry in entries],
        STATE=[state.name for state in result.states],
    )
    expanded["data"] |= {
        spec.list_name for entry in entries for spec in (*entry.inputs, *entry.outputs)
    }
    return union(expanded, every())


@pytest.mark.parametrize("kind", ["markers", "explicit"])
def test_a_program_sprite(tmp_path, kind):
    if kind == "markers":
        result = transpile(Memory().eval(), None, tmp_path / "program")
    else:
        program = ExportProgram(
            states=(State("total", torch.zeros(2)),),
            entrypoints=(EntryPoint(
                "step", method="step",
                arguments=(Input("value", torch.ones(2)), StateInput("total")),
                returns=(Output("doubled"), StateUpdate("total")),
            ),),
        )
        result = transpile(Memory().eval(), program, tmp_path / "program")
    check(load(result.path), program_names(result))


class _SentencePiece:
    pieces = ("<unk>", "a", "b", "ab", "▁")

    def get_piece_size(self):
        return len(self.pieces)

    def id_to_piece(self, token_id):
        return self.pieces[token_id]

    def is_byte(self, token_id):
        return False

    def is_control(self, token_id):
        return False

    def get_score(self, token_id):
        return -float(token_id)

    def unk_id(self):
        return 0


@pytest.mark.parametrize("kind", ["characters", "bpe", "sentencepiece"])
def test_a_tokenizer_sprite(tmp_path, kind):
    from test_project import char_tokenizer

    if kind == "characters":
        result = CharTokenizer(char_tokenizer()).save(tmp_path / "chars")
    elif kind == "bpe":
        from tokenizers import Tokenizer, models
        from transformers import PreTrainedTokenizerFast

        vocab = {"a": 0, "b": 1, "ab": 2, "?": 3}
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[("a", "b")], unk_token="?"))
        result = BPETokenizer(
            PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="?"),
        ).save(tmp_path / "bpe")
    else:
        result = SentencePieceBPETokenizer(_SentencePiece()).save(tmp_path / "sentencepiece")
    check(load(result.path), union(documented("Tokenizer sprites"), every(storage=False)))


def project_names(project) -> dict[str, set[str]]:
    """The public names a project adds: globals, messages and bridge blocks."""
    data, _ = read_archive(project.path, PROJECT_JSON)
    stage, *sprites = data["targets"]
    names = {
        "data": {entry[0] for section in ("variables", "lists") for entry in stage[section].values()},
        # The sprites' own broadcasts are unified on the stage too.
        "broadcasts": {
            message for message in stage["broadcasts"].values()
            if not message.startswith(interface.PUBLIC_PREFIX)
        },
        "procedures": set(),
    }
    for sprite in sprites:
        names["procedures"] |= {
            block["mutation"]["proccode"] for block in sprite["blocks"].values()
            if block.get("opcode") == "procedures_prototype"
            and block["mutation"]["proccode"].startswith(interface.bridge_procedure(""))
        }
        if sprite["name"] == "Demo":
            demo = sprite_names(sprite)
            assert all(
                name.startswith(interface.INTERNAL_PREFIX)
                for kind in ("data", "procedures") for name in demo[kind]
            )
    return names


def test_project_names(tmp_path):
    from test_project import Language, char_tokenizer

    torch.manual_seed(0)
    plain = transpile(nn.Linear(2, 3).eval(), torch.randn(1, 2), tmp_path / "points")
    program = transpile(Memory().eval(), None, tmp_path / "memory")
    lm = transpile(Language().eval(), GenerationProgram(max_context=16), tmp_path / "story")
    words = CharTokenizer(char_tokenizer()).save(tmp_path / "words")
    projects = {
        "points": build_project(plain, tmp_path / "a", labels=["x", "y", "z"]),
        "memory": build_project(program, tmp_path / "b"),
        "story": build_project(lm, tmp_path / "c", tokenizer=words),
    }
    patterns = documented("Projects")
    used: set[str] = set()
    for name, project in projects.items():
        values = {
            "NAME": [name, "words"], "ENTRY": ["add", "latest"],
            "INPUT": ["row", "rows"], "OUTPUT": ["output"],
        }
        for kind, found in project_names(project).items():
            for actual in found:
                matches = [
                    pattern for pattern in patterns[kind]
                    if actual in expand({kind: {pattern}}, **values)[kind]
                    or (pattern == "NAME input" and re.fullmatch(rf"{name} input(_\d+)?", actual))
                    or (pattern == "NAME output" and re.fullmatch(rf"{name} output(_\d+)?", actual))
                ]
                assert matches, (name, kind, actual)
                used.update(matches)
        # The project's result names what it adds.
        found = project_names(project)
        assert set(project.broadcasts) <= found["broadcasts"]
        assert set(project.lists) | set(project.variables) == found["data"]
    assert used == set().union(*patterns.values())


def test_a_split_output_keeps_its_public_names(tmp_path):
    model = nn.Linear(1, 200_005, bias=False).eval()
    result = transpile(model, torch.ones(1, 1), tmp_path / "wide")
    names = sprite_names(load(result.path))["data"]
    assert "output" in result.sharded_lists
    assert {"output", "output shard 2", *interface.shard_variables("output")} <= names


def test_a_reduced_output_adds_no_variable_named_like_a_list(tmp_path):
    # The reduction kernel named a working variable "output", which sat
    # beside the public list.
    example = torch.randn(3, 2)
    result = transpile(Sums().eval(), (example, torch.tensor([[1, 2]])), tmp_path / "sums")
    sprite = load(result.path)
    lists = {entry[0] for entry in sprite["lists"].values()}
    assert not lists & {entry[0] for entry in sprite["variables"].values()}
    emulator = ScratchEmulator(sprite)
    emulator.lists["input"] = example.flatten().tolist()
    emulator.lists["input_1"] = [1, 2]
    emulator.run_procedure("cattorch forward")
    torch.testing.assert_close(
        torch.tensor([float(value) for value in emulator.lists["output"]]),
        example.sum().reshape(1),
    )
