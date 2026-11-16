"""The versioned Scratch interface: docs/scratch-interface.md against real sprites.

The contract page is the source of truth. These tests read its tables and
check, for a sprite of every kind, that each name it documents is in the
sprite and that every other name in the sprite is internal (starts with
``_``), and that ``cattorch.interface``, where the code takes the names and
status values from, agrees with it. Within interface version 1 no public
name or status value may be renamed or removed: if a check fails because one
disappeared, that breaks Scratch projects, so restore it. A new name is
added to the page.

The public names below, and the status values other tests don't check,
are written out only here; the status values they do check are written out
once in ``scratch_status``. Other tests take the names from
``cattorch.interface``.
"""

from __future__ import annotations

import re
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
from sprite_helpers import load_sprite
from scratch_status import INTERRUPTED, OVERTAKEN

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
    OVERTAKEN: interface.OVERTAKEN,
    INTERRUPTED: interface.INTERRUPTED,
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
    check(load_sprite(result.path), model_names(result))


@pytest.mark.parametrize("recurrent", [False, True])
def test_a_generation_sprite(tmp_path, recurrent):
    from shared_models import GRULM, Language

    if recurrent:
        model = GRULM().eval()
        program = RecurrentGenerationProgram(model.states(), top_k=3)
    else:
        model, program = Language().eval(), GenerationProgram(max_context=16, top_k=3)
    result = transpile(model, program, tmp_path / "lm")
    check(load_sprite(result.path), union(documented("Generation sprites"), every()))


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
    check(load_sprite(result.path), program_names(result))


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
    from shared_models import char_tokenizer

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
    check(load_sprite(result.path), union(documented("Tokenizer sprites"), every(storage=False)))


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
    from shared_models import Language, char_tokenizer

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
    names = sprite_names(load_sprite(result.path))["data"]
    assert "output" in result.sharded_lists
    assert {"output", "output shard 2", *interface.shard_variables("output")} <= names


def test_sharded_lists_names_only_public_lists(tmp_path):
    # The weight is split in four, but it is internal.
    model = nn.Linear(1000, 250, bias=False).eval()
    result = transpile(model, torch.randn(1, 1000), tmp_path / "weights")
    names = sprite_names(load_sprite(result.path))["data"]
    assert any(name.startswith("_W_") and name.endswith(" shard 2") for name in names)
    assert result.sharded_lists == ()


def test_a_reduced_output_adds_no_variable_named_like_a_list(tmp_path):
    # The reduction kernel named a working variable "output", which sat
    # beside the public list.
    example = torch.randn(3, 2)
    result = transpile(Sums().eval(), (example, torch.tensor([[1, 2]])), tmp_path / "sums")
    sprite = load_sprite(result.path)
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


def test_benchmark_project_names(tmp_path):
    """Each builder's stage names are the documented public ones."""
    from cattorch import (
        benchmark, build_benchmark_suite, build_paired_benchmark, build_storage_benchmark_suite,
    )

    names = documented("Benchmark projects")["data"]
    assert names == {
        benchmark.EXACT_TIMINGS, benchmark.FAST_TIMINGS, benchmark.SUITE_RESULTS,
        benchmark.STORAGE_SIZES, benchmark.REPEATS, benchmark.WARMUPS, benchmark.COMPARISON,
    }
    model, example = nn.Linear(3, 2).eval(), torch.randn(1, 3)
    projects = {
        "paired": build_paired_benchmark(model, example, tmp_path / "paired", repeats=1),
        "suite": build_benchmark_suite([("one", model, example)], tmp_path / "suite", repeats=1),
        "storage": build_storage_benchmark_suite(
            [("one", model, example)], tmp_path / "storage", repeats=1,
        ),
    }
    settings = {benchmark.REPEATS, benchmark.WARMUPS}
    expected = {
        "paired": {benchmark.EXACT_TIMINGS, benchmark.FAST_TIMINGS, *settings, benchmark.COMPARISON},
        "suite": {benchmark.SUITE_RESULTS, *settings, benchmark.COMPARISON},
        "storage": {benchmark.SUITE_RESULTS, benchmark.STORAGE_SIZES, *settings},
    }
    for kind, path in projects.items():
        stage = next(
            target for target in read_archive(path, PROJECT_JSON)[0]["targets"] if target["isStage"]
        )
        public = {
            entry[0] for section in ("variables", "lists") for entry in stage[section].values()
            if not entry[0].startswith(interface.INTERNAL_PREFIX)
        }
        assert public == expected[kind], kind


# ── Index inputs (docs/supported-models.md#index-inputs) ────────────────────

class Indexes(nn.Module):
    """One operation that takes integer input ``ids`` as indices."""

    def __init__(self, operation):
        super().__init__()
        self.table = nn.Parameter(torch.arange(10.0).reshape(5, 2))
        self.operation = operation

    def forward(self, value, ids):
        return self.operation(self.table, value, ids)


F = torch.nn.functional
# The operation, the ids, and the valid range of the first id: (lowest, highest).
INDEXING = {
    "embedding": (lambda t, x, i: F.embedding(i, t), [1, 3], (0, 4)),
    "embedding padding_idx": (lambda t, x, i: F.embedding(i, t, padding_idx=0), [1, 3], (0, 4)),
    "embedding_bag": (lambda t, x, i: F.embedding_bag(i.view(1, 2), t), [1, 3], (0, 4)),
    "index_select": (lambda t, x, i: t.index_select(0, i), [1, 3], (0, 4)),
    "gather": (lambda t, x, i: x.gather(1, i.view(1, 2)), [1, 2], (0, 2)),
    "take_along_dim": (lambda t, x, i: torch.take_along_dim(t, i.view(1, 2), 0), [1, 3], (0, 4)),
    "one_hot": (lambda t, x, i: F.one_hot(i, 4).float(), [1, 3], (0, 3)),
    "scatter": (
        lambda t, x, i: torch.zeros(1, 3).scatter(1, i.view(1, 2), x[:, :2]), [0, 2], (0, 2),
    ),
    "index_add": (lambda t, x, i: t.index_add(0, i, x.view(-1)[:2].expand(2, 2)), [1, 3], (0, 4)),
    "index_copy": (lambda t, x, i: t.index_copy(0, i, x.view(-1)[:2].expand(2, 2)), [1, 3], (0, 4)),
    "getitem": (lambda t, x, i: t[i], [1, 3], (-5, 4)),
    "getitem columns": (lambda t, x, i: t[:, i], [1, 0], (-2, 1)),
    "take": (lambda t, x, i: torch.take(t, i), [1, 3], (-10, 9)),
    "index_put": (lambda t, x, i: t.index_put((i,), x[0, :2]), [1, 3], (-5, 4)),
    "put": (lambda t, x, i: t.put(i, x[0, :2]), [1, 3], (-10, 9)),
    "index_fill": (lambda t, x, i: t.index_fill(0, i, -1.0), [1, 3], (-5, 4)),
    # Used twice: the intersection.
    "two uses": (lambda t, x, i: t[i] + F.embedding(i, t[:3]), [1, 2], (0, 2)),
    # Through a cast and a view.
    "cast and view": (lambda t, x, i: t[i.long().view(2, 1)], [1, 3], (-5, 4)),
}


def forward_status(sprite: dict, value: torch.Tensor, ids: list) -> tuple[str, list]:
    emulator = ScratchEmulator(sprite)
    emulator.lists["input"] = value.flatten().tolist()
    emulator.lists["input_1"] = list(ids)
    emulator.run_procedure(interface.FORWARD)
    return emulator.variables[interface.STATUS_VARIABLE], emulator.lists["output"]


def assert_index_range(sprite, model, value, ids, lowest, highest):
    """Every first id from ``lowest`` to ``highest`` gives PyTorch's output, and
    every other one, which PyTorch refuses, fails naming the input."""
    for first in range(lowest - 2, highest + 3):
        status, output = forward_status(sprite, value, [first, *ids[1:]])
        if lowest <= first <= highest:
            expected = model(value, torch.tensor([first, *ids[1:]]))
            assert status == interface.OK, first
            torch.testing.assert_close(
                torch.tensor([float(item) for item in output]), expected.detach().flatten(),
            )
        else:
            assert status == interface.invalid_item("input_1"), first


@pytest.mark.parametrize("name", sorted(INDEXING))
def test_an_index_input_takes_pytorchs_indices(tmp_path, name):
    operation, ids, (lowest, highest) = INDEXING[name]
    model = Indexes(operation).eval()
    value = torch.randn(1, 3)
    dtype = torch.int32 if name == "cast and view" else torch.long
    result = transpile(model, (value, torch.tensor(ids, dtype=dtype)), tmp_path / "ids")
    sprite = load_sprite(result.path)
    assert_index_range(sprite, model, value, ids, lowest, highest)
    for invalid in (0.5, "", "x", "Infinity"):
        status, _ = forward_status(sprite, value, [invalid, *ids[1:]])
        assert status == interface.invalid_item("input_1"), invalid


COMPUTED = {
    "shifted": (lambda t, x, i: t[i + 1], [1, 3], (-6, 3)),
    "cross_entropy": (lambda t, x, i: F.cross_entropy(x, i).view(1), [1, 2], (0, 2)),
    # Bags that don't start at 0 leave indices in no bag.
    "embedding_bag offsets": (
        lambda t, x, i: F.embedding_bag(torch.tensor([1, 2, 3]), t, i), [0, 2], (0, 0),
    ),
}


@pytest.mark.parametrize("name", sorted(COMPUTED))
def test_an_index_computed_from_an_input_is_checked_as_the_block_runs(tmp_path, name):
    operation, ids, (lowest, highest) = COMPUTED[name]
    model = Indexes(operation).eval()
    value = torch.randn(2, 3)
    result = transpile(model, (value, torch.tensor(ids)), tmp_path / "computed")
    sprite = load_sprite(result.path)
    assert any(
        isinstance(block, dict) and block.get("opcode") == "data_setvariableto"
        and interface.invalid_item("input_1") in str(block.get("inputs"))
        for block in sprite["blocks"].values()
    )
    for first in (lowest - 1, highest + 1):
        with pytest.raises((IndexError, RuntimeError)):
            model(value, torch.tensor([first, *ids[1:]]))
    assert_index_range(sprite, model, value, ids, lowest, highest)


def test_a_program_entry_points_index_inputs(tmp_path):
    class Tables(nn.Module):
        def __init__(self):
            super().__init__()
            self.table = nn.Parameter(torch.arange(10.0).reshape(5, 2))

        def look(self, ids):
            return self.table[ids]

        def shifted(self, ids, total):
            rows = self.table[ids + 1]
            return rows, total + rows.sum().reshape(1)

    program = ExportProgram(
        states=(State("total", torch.zeros(1)),),
        entrypoints=(
            EntryPoint("look", method="look", arguments=(Input("ids", torch.tensor([1, 2])),),
                       returns=(Output("rows"),)),
            EntryPoint("shift", method="shifted", arguments=(
                Input("ids", torch.tensor([1, 2])), StateInput("total"),
            ), returns=(Output("rows"), StateUpdate("total"))),
        ),
    )
    result = transpile(Tables().eval(), program, tmp_path / "program")
    emulator = ScratchEmulator(load_sprite(result.path))
    for entry, valid, invalid in (("look", (-5, 4), (-6, 5)), ("shift", (-6, 3), (-7, 4))):
        for first in (*valid, *invalid):
            emulator.lists[f"cattorch {entry} ids"] = [first, 1]
            state = list(emulator.lists["cattorch state total"])
            emulator.run_procedure(f"cattorch {entry}")
            status = emulator.variables[interface.STATUS_VARIABLE]
            if first in valid:
                assert status == interface.OK, (entry, first)
            else:
                assert status == interface.invalid_item(f"cattorch {entry} ids"), (entry, first)
                assert emulator.lists["cattorch state total"] == state


def test_a_generator_refuses_an_index_computed_from_the_token_ids(tmp_path):
    from shared_models import VOCAB, NanoGPT

    from cattorch import UnsupportedModelError

    class Halves(NanoGPT):
        def __init__(self):
            super().__init__(layers=1)
            self.halves = nn.Parameter(torch.randn(VOCAB // 2, 16))

        def forward(self, tokens):
            positions = torch.arange(0, tokens.shape[1])
            x = self.wte(tokens) + self.halves[tokens // 2] + self.wpe(positions)
            return self.head(self.ln_f(self.blocks[0](x)))

    with pytest.raises(UnsupportedModelError, match="index its model computes from the token IDs"):
        transpile(Halves().eval(), GenerationProgram(max_context=8), tmp_path / "lm")


# ── Item rules (docs/scratch-interface.md#how-every-block-behaves) ──────────

class Affine(nn.Module):
    def forward(self, value, count):
        return value * 2, count * 2 + 1


# Each item, and whether a floating-point and an integer list take it.
ITEMS = {
    1: (True, True), -3: (True, True), "1e3": (True, True), 0.5: (True, False),
    "Infinity": (True, False), "-Infinity": (True, False), "NaN": (False, False),
    "": (False, False), "x": (False, False),
    2**53: (True, True), -(2**53): (True, True),
    2**53 + 2: (True, False), -(2**53) - 2: (True, False),
}


@pytest.mark.parametrize("program", [False, True])
def test_every_kind_checks_items_by_the_documented_rules(tmp_path, program):
    if program:
        example = ExportProgram(entrypoints=(EntryPoint("go", arguments=(
            Input("value", torch.ones(2)), Input("count", torch.ones(2, dtype=torch.long)),
        ), returns=(Output("a"), Output("b"))),))
        result = transpile(Affine().eval(), example, tmp_path / "items")
        lists, block = ("cattorch go value", "cattorch go count"), "cattorch go"
    else:
        result = transpile(
            Affine().eval(), (torch.ones(2), torch.ones(2, dtype=torch.long)), tmp_path / "items",
        )
        lists, block = ("input", "input_1"), interface.FORWARD
    sprite = load_sprite(result.path)
    for value, accepted in ITEMS.items():
        for position, name in enumerate(lists):
            emulator = ScratchEmulator(sprite)
            for other in lists:
                emulator.lists[other] = [1, 1]
            emulator.lists[name] = [value, 1]
            emulator.run_procedure(block)
            status = emulator.variables[interface.STATUS_VARIABLE]
            expected = interface.OK if accepted[position] else interface.invalid_item(name)
            assert status == expected, (name, value)


def test_an_append_state_holds_at_least_its_initial_rows(tmp_path):
    result = transpile(Memory().eval(), None, tmp_path / "memory")
    sprite = load_sprite(result.path)
    state = "cattorch state seen"
    for rows, status in (
        ([], interface.too_few_items(state)), ([1, 2, 3], interface.OK),
        ([1, 2, 3] * 8, interface.OK), ([1, 2, 3] * 9, interface.too_many_items(state)),
        ([1, 2], interface.too_few_items(state)), ([1, 2, 3, 4], interface.wrong_number_of_items(state)),
    ):
        emulator = ScratchEmulator(sprite)
        emulator.run_procedure(interface.RESET)
        emulator.lists[state] = list(rows)
        emulator.lists["cattorch latest rows"] = [1.0, 2.0, 3.0]
        emulator.run_procedure("cattorch latest")
        assert emulator.variables[interface.STATUS_VARIABLE] == status, rows


# ── Long lists (docs/scratch-interface.md#long-lists) ───────────────────────

def fill_split(emulator, name, values, sizes):
    """Lay ``values`` out over ``name``'s shards, ``sizes`` items in each."""
    start = 0
    for position, size in enumerate(sizes):
        emulator.lists[interface.shard_name(name, position)] = list(values[start:start + size])
        start += size


# A list of 11 split at 8: valid only as 8 + 3.
SPLITS = {(8, 3): True, (7, 4): False, (0, 11): False, (11, 0): False, (8, 2): False}


def test_a_split_input_checks_each_shard(tmp_path, small_list_limit):
    small_list_limit(8)

    class Double(nn.Module):
        def forward(self, value):
            return value * 2

    model = Double().eval()
    result = transpile(model, torch.randn(11, 1), tmp_path / "split")
    assert "input" in result.sharded_lists
    sprite = load_sprite(result.path)
    values = [float(value) for value in range(11)]
    for sizes, valid in SPLITS.items():
        emulator = ScratchEmulator(sprite)
        fill_split(emulator, "input", values, sizes)
        emulator.run_procedure(interface.FORWARD)
        status = emulator.variables[interface.STATUS_VARIABLE]
        assert status == (interface.OK if valid else interface.wrong_number_of_items("input")), sizes
    emulator = ScratchEmulator(sprite)
    fill_split(emulator, "input", [*values[:9], "x", values[10]], (8, 3))
    emulator.run_procedure(interface.FORWARD)
    assert emulator.variables[interface.STATUS_VARIABLE] == interface.invalid_item("input")


def test_a_programs_split_lists_check_each_shard(tmp_path, small_list_limit):
    small_list_limit(8)

    class Total(nn.Module):
        def step(self, value, ids, total):
            total = total + value + ids.float()
            return total * 2, total

    program = ExportProgram(
        states=(State("total", torch.zeros(11)),),
        entrypoints=(EntryPoint("step", method="step", arguments=(
            Input("value", torch.ones(11)), Input("ids", torch.ones(11, dtype=torch.long)),
            StateInput("total"),
        ), returns=(Output("doubled"), StateUpdate("total"))),),
    )
    result = transpile(Total().eval(), program, tmp_path / "program")
    sprite = load_sprite(result.path)
    lists = ("cattorch step value", "cattorch step ids", "cattorch state total")
    assert set(lists) <= set(result.sharded_lists)

    def run(sizes=(8, 3), at=lists[0], items=None):
        emulator = ScratchEmulator(sprite)
        emulator.run_procedure(interface.RESET)
        for name in lists:
            values = [0 if name == lists[2] else 1] * 11
            fill_split(emulator, name, values if name != at or items is None else items,
                       sizes if name == at else (8, 3))
        emulator.run_procedure("cattorch step")
        return emulator.variables[interface.STATUS_VARIABLE]

    for name in lists:
        for sizes, valid in SPLITS.items():
            expected = interface.OK if valid else interface.wrong_number_of_items(name)
            assert run(sizes, name) == expected, (name, sizes)
    # Items in the second shard are checked as numbers, not added to the first's.
    assert run(at=lists[0], items=[1] * 9 + ["abc", 1]) == interface.invalid_item(lists[0])
    assert run(at=lists[1], items=[1] * 9 + [1.5, 1]) == interface.invalid_item(lists[1])
