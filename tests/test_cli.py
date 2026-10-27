"""The ``cattorch`` command line."""

from __future__ import annotations

import json
import zipfile

import pytest
import torch
from torch import nn

from cattorch.cli import main

MODELS = '''
import torch
from torch import nn

import cattorch


@cattorch.example(lambda: (torch.randn(2, 3),))
class Marked(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, value):
        return self.linear(value).relu()


def unmarked():
    torch.manual_seed(0)
    return nn.Sequential(nn.Embedding(10, 4), nn.Flatten(), nn.Linear(12, 2))


class Language(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.tokens = nn.Embedding(12, 8)
        self.layer = nn.TransformerEncoderLayer(8, 2, 16, dropout=0.0, batch_first=True)
        self.head = nn.Linear(8, 12)

    def forward(self, tokens):
        mask = nn.Transformer.generate_square_subsequent_mask(tokens.shape[1])
        return self.head(self.layer(self.tokens(tokens), src_mask=mask, is_causal=True))
'''


@pytest.fixture
def models(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models.py").write_text(MODELS)
    return tmp_path


def run(capsys, *argv):
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, json.loads(captured.out) if code == 0 else captured.err


def test_export_reads_the_example_marker(models, capsys):
    code, report = run(capsys, "export", "models.py:Marked", "--project")
    assert code == 0 and report["verified"]
    assert report["sprite"].endswith("marked.sprite3")
    with zipfile.ZipFile(report["project"]) as archive:
        assert "project.json" in archive.namelist()


def test_export_with_shapes_from_a_module_path(models, capsys):
    code, report = run(
        capsys, "export", "models:unmarked", "--tokens", "1,3", "--vocab", "10",
        "-o", "out/net", "--precision", "float16",
    )
    assert code == 0 and report["verified"]
    assert report["sprite"] == "out/net.sprite3"


def test_export_a_saved_module_with_a_dynamic_batch(models, capsys):
    torch.save(nn.Sequential(nn.Linear(4, 2), nn.Sigmoid()), models / "saved.pt")
    code, report = run(capsys, "export", "saved.pt", "--shape", "3,4", "--dynamic", "0=1:8")
    assert code == 0 and report["verified"]


def test_errors_are_reported_without_a_traceback(models, capsys):
    code, message = run(capsys, "export", "models.py:unmarked")
    assert code == 1 and "--shape" in message
    code, message = run(capsys, "export", "models.py:missing", "--shape", "1,2")
    assert code == 1 and "no attribute 'missing'" in message
    with pytest.raises(SystemExit):
        main(["export", "models.py:Marked", "--dynamic", "zero"])


def test_generate_with_a_tokenizer_writes_a_project(models, capsys):
    from tokenizers import Tokenizer, models as tokenizer_models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    characters = "?abcdefghij "
    backend = Tokenizer(tokenizer_models.WordLevel(
        vocab={character: index for index, character in enumerate(characters)}, unk_token="?",
    ))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="?").save_pretrained(
        models / "chars",
    )
    code, report = run(
        capsys, "generate", "models.py:Language", "--tokenizer", "chars",
        "--max-context", "8", "--project",
    )
    assert code == 0
    assert report["tokenizer"].endswith("language_tokenizer.sprite3")
    assert report["project"].endswith("language.sb3")

    code, message = run(capsys, "generate", "models.py:Language", "--project")
    assert code == 1 and "needs a tokenizer" in message
