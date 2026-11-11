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


@cattorch.example(lambda: (torch.randn(3, 3),))
class Determinant(nn.Module):
    def forward(self, value):
        return torch.linalg.det(value)


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


def test_fast_exports_verify_and_dotted_names_keep_their_project(models, capsys):
    code, report = run(
        capsys, "export", "models.py:Marked", "--fast", "-o", "out/net.v2", "--project",
    )
    assert code == 0 and report["verified"]
    assert report["sprite"] == "out/net.v2.sprite3"
    assert report["project"] == "out/net.v2.sb3"


def test_export_a_saved_module_with_a_dynamic_batch(models, capsys):
    torch.save(nn.Sequential(nn.Linear(4, 2), nn.Sigmoid()), models / "saved.pt")
    code, report = run(capsys, "export", "saved.pt", "--shape", "3,4", "--dynamic", "0=1:8")
    assert code == 0 and report["verified"]


def test_errors_are_reported_without_a_traceback(models, capsys):
    code, message = run(capsys, "export", "models.py:unmarked")
    assert code == 1 and "--shape" in message
    code, message = run(capsys, "export", "models.py:missing", "--shape", "1,2")
    assert code == 1 and "no attribute 'missing'" in message
    code, message = run(capsys, "export", "models.py:Determinant")
    assert code == 1 and "Unsupported operation" in message
    with pytest.raises(SystemExit):
        main(["export", "models.py:Marked", "--dynamic", "zero"])


CHARACTERS = "?abcdefghij "


def save_characters(directory):
    """A Hugging Face tokenizer with one token per character of ``CHARACTERS``."""
    from tokenizers import Tokenizer, models as tokenizer_models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(tokenizer_models.WordLevel(
        vocab={character: index for index, character in enumerate(CHARACTERS)}, unk_token="?",
    ))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="?").save_pretrained(directory)


def test_generate_with_a_tokenizer_writes_a_project(models, capsys):
    save_characters(models / "chars")
    code, report = run(
        capsys, "generate", "models.py:Language", "--tokenizer", "chars",
        "--max-context", "8", "--project",
    )
    assert code == 0
    assert report["tokenizer"].endswith("language_tokenizer.sprite3")
    assert report["project"].endswith("language.sb3")

    code, message = run(capsys, "generate", "models.py:Language", "--project")
    assert code == 1 and "needs a tokenizer" in message


@pytest.mark.parametrize("output", ["build/demo", "build/demo.sprite3", "build/demo.sb3"])
def test_every_output_is_named_after_the_output_option(models, capsys, output):
    save_characters(models / "chars")
    code, report = run(
        capsys, "generate", "models.py:Language", "--tokenizer", "chars",
        "--max-context", "4", "--project", "-o", output,
    )
    assert code == 0
    assert (report["sprite"], report["tokenizer"], report["project"]) == (
        "build/demo.sprite3", "build/demo_tokenizer.sprite3", "build/demo.sb3",
    )
    assert sorted(path.name for path in (models / "build").iterdir()) == [
        "demo.sb3", "demo.sprite3", "demo_tokenizer.sprite3",
    ]


def max_context(sprite_path):
    with zipfile.ZipFile(sprite_path) as archive:
        sprite = json.loads(archive.read("sprite.json"))
    (value,) = (
        entry[1] for entry in sprite["variables"].values() if entry[0] == "cattorch max context"
    )
    return value


def test_generate_detects_a_recurrent_hugging_face_model(models, capsys):
    transformers = pytest.importorskip("transformers")
    from cattorch.util.scratch.emulator import ProjectEmulator

    save_characters(models / "chars")
    torch.manual_seed(0)
    model = transformers.Mamba2ForCausalLM(transformers.Mamba2Config(
        vocab_size=len(CHARACTERS), hidden_size=16, num_hidden_layers=1, expand=2, num_heads=4,
        head_dim=8, state_size=4, n_groups=1, conv_kernel=3, chunk_size=4,
    )).eval()
    model.save_pretrained(models / "ssm")
    code, report = run(
        capsys, "generate", "--hf", "ssm", "--tokenizer", "chars", "--project",
        "--temperature", "0", "--max-new-tokens", "6",
    )
    assert code == 0 and report["max_context"] is None
    assert max_context(report["sprite"]) == "Infinity"

    # The demo runs past any context limit until its token budget is spent.
    emulator = ProjectEmulator(report["project"], answers=["badge", ""])
    emulator.green_flag()
    ids = [CHARACTERS.index(character) for character in "badge"]
    with torch.no_grad():
        for _ in range(6):
            ids.append(int(model(torch.tensor([ids])).logits[0, -1].argmax()))
    text = "".join(CHARACTERS[index] for index in ids)
    assert text in emulator.sprites["Demo"].said

    code, report = run(capsys, "generate", "--hf", "ssm", "--max-context", "5", "-o", "bounded")
    assert code == 0 and report["max_context"] == 5
    assert max_context(report["sprite"]) == 5
