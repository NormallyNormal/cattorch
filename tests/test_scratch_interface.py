"""The versioned Scratch interface (docs/scratch-interface.md).

Within interface version 1 no public name may be renamed or removed. If one
of these snapshots fails because a name disappeared, that breaks Scratch
projects: restore the name. A new public name may be added to the snapshot
and the docs.
"""

from __future__ import annotations

import json
import zipfile

import pytest
import torch
from torch import nn

from cattorch import (
    CharTokenizer, GenerationProgram, entrypoint, interface, transpile,
)

PUBLIC = interface.PUBLIC_PREFIX

STORAGE = {
    "cattorch storage compression", "cattorch storage precision",
    "cattorch storage scale precision", "cattorch quantization method",
    "cattorch quantization bits",
}
MODEL = {
    "procedures": {"cattorch init", "cattorch forward", "cattorch prepare for save"},
    "data": {
        "cattorch interface version", "cattorch status", "cattorch initialized",
        "input", "output", *STORAGE,
    },
    "broadcasts": {"cattorch init complete", "cattorch forward complete"},
}
GENERATION = {
    "procedures": {"cattorch init", "cattorch reset", "cattorch prefill", "cattorch decode"},
    "data": {
        "cattorch interface version", "cattorch status", "cattorch initialized",
        "cattorch tokens", "cattorch logits", "cattorch cache length", "cattorch max context",
        "cattorch top k values", "cattorch top k ids", *STORAGE,
    },
    "broadcasts": {
        "cattorch init complete", "cattorch reset complete", "cattorch prefill complete",
        "cattorch decode complete",
    },
}
PROGRAM = {
    "procedures": {
        "cattorch init", "cattorch reset", "cattorch prepare for save", "cattorch add",
        "cattorch latest",
    },
    "data": {
        "cattorch interface version", "cattorch status", "cattorch add row",
        "cattorch add output", "cattorch latest rows", "cattorch latest output",
        "cattorch state seen", *STORAGE,
    },
    "broadcasts": {
        "cattorch init complete", "cattorch add complete", "cattorch latest complete",
    },
}
TOKENIZER = {
    "procedures": {"cattorch tokenize", "cattorch detokenize"},
    "data": {"cattorch interface version", "cattorch text", "cattorch token ids"},
    "broadcasts": set(),
}


def interface_names(path) -> tuple[dict[str, set[str]], dict]:
    """Public names of a sprite, and its variables by name."""
    with zipfile.ZipFile(path) as archive:
        sprite = json.loads(archive.read("sprite.json"))
    procedures = {
        block["mutation"]["proccode"] for block in sprite["blocks"].values()
        if isinstance(block, dict) and block.get("opcode") == "procedures_prototype"
    }
    data = {entry[0] for section in ("variables", "lists") for entry in sprite[section].values()}
    names = {
        "procedures": {name for name in procedures if name.startswith(PUBLIC)},
        "data": {
            name.split(" shard ", 1)[0] for name in data
            if name.startswith(PUBLIC) or interface.is_public_data(name)
        },
        "broadcasts": set(sprite.get("broadcasts", {}).values()),
    }
    variables = {entry[0]: entry[1] for entry in sprite["variables"].values()}
    return names, variables


def check(path, expected):
    names, variables = interface_names(path)
    assert variables[interface.VERSION_VARIABLE] == interface.INTERFACE_VERSION
    with zipfile.ZipFile(path) as archive:
        sprite = json.loads(archive.read("sprite.json"))
    for section in ("variables", "lists"):
        # One slot per name, or a user's block could pick the wrong copy.
        listed = [entry[0] for entry in sprite[section].values()]
        assert len(listed) == len(set(listed)), section
    for kind, public in expected.items():
        assert names[kind] == public, kind


def test_a_model_sprite(tmp_path):
    result = transpile(nn.Linear(3, 2).eval(), torch.randn(1, 3), tmp_path / "model")
    check(result.path, MODEL)


def test_a_model_run_once_per_item(tmp_path):
    layer = nn.TransformerEncoderLayer(8, 2, 16, dropout=0.0, batch_first=True).eval()
    result = transpile(layer, torch.randn(2, 3, 8), tmp_path / "items", dynamic={0: (1, 4)})
    assert "runs the model once per item" in result.warnings[-1]
    check(result.path, MODEL)


def test_a_generation_sprite(tmp_path):
    from test_project import Language

    result = transpile(
        Language().eval(), GenerationProgram(max_context=16, top_k=3), tmp_path / "lm",
    )
    check(result.path, GENERATION)


class Memory(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("seen", torch.zeros(1, 3))

    @entrypoint(example=lambda: (torch.ones(3),), capacity={"seen": 8})
    def add(self, row):
        self.seen = torch.cat([self.seen, row.reshape(1, 3)])
        return self.seen.sum(0)

    @entrypoint(example=lambda: (torch.ones(2, 3),), dynamic={"rows": (1, 4)})
    def latest(self, rows):
        return rows @ self.seen[-1]


def test_a_program_sprite(tmp_path):
    result = transpile(Memory().eval(), None, tmp_path / "program")
    check(result.path, PROGRAM)


@pytest.mark.parametrize("kind", ["characters", "bpe"])
def test_a_tokenizer_sprite(tmp_path, kind):
    from test_project import char_tokenizer

    if kind == "characters":
        result = CharTokenizer(char_tokenizer()).save(tmp_path / "chars")
    else:
        from tokenizers import Tokenizer, models
        from transformers import PreTrainedTokenizerFast

        from cattorch import BPETokenizer

        vocab = {"a": 0, "b": 1, "ab": 2, "?": 3}
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[("a", "b")], unk_token="?"))
        result = BPETokenizer(
            PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="?"),
        ).save(tmp_path / "bpe")
    check(result.path, TOKENIZER)


def test_a_split_output_keeps_its_public_names(tmp_path):
    model = nn.Linear(1, 200_005, bias=False).eval()
    result = transpile(model, torch.ones(1, 1), tmp_path / "wide")
    names, variables = interface_names(result.path)
    assert "output" in result.sharded_lists
    assert {"cattorch output shard count", "cattorch output logical length"} <= set(variables)
