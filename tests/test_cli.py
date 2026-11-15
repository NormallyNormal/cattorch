"""The ``cattorch`` command line."""

from __future__ import annotations

import json
import zipfile

import pytest
import torch
from torch import nn

from cattorch.cli import main
from sprite_helpers import load_sprite

MODELS = '''
import torch
from torch import nn

import cattorch


@cattorch.example_inputs(lambda: (torch.randn(2, 3),))
class Marked(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, value):
        return self.linear(value).relu()


def unmarked():
    torch.manual_seed(0)
    return nn.Sequential(nn.Embedding(10, 4), nn.Flatten(), nn.Linear(12, 2))


@cattorch.example_inputs(lambda: (torch.randn(3, 3),))
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
    assert code == 0 and report["verify"]["passed"]
    assert report["path"].endswith("marked.sprite3")
    with zipfile.ZipFile(report["project"]["path"]) as archive:
        assert "project.json" in archive.namelist()


def test_export_with_shapes_from_a_module_path(models, capsys):
    code, report = run(
        capsys, "export", "models:unmarked", "--tokens", "1,3", "--vocab", "10",
        "-o", "out/net", "--precision", "float16", "--name", "Net",
    )
    assert code == 0 and report["verify"]["passed"]
    assert report["path"] == "out/net.sprite3" and report["sprite_name"] == "Net"


def test_quantized_exports_are_verified_against_the_stored_weights(models, capsys):
    code, report = run(capsys, "export", "models.py:Marked", "--quantize", "8", "--sig-figs", "4")
    assert code == 0 and report["verify"]["passed"]
    assert report["verify"]["max_abs_error"] < 1e-4
    torch.save([torch.randn(2, 3) for _ in range(4)], models / "calibration.pt")
    code, report = run(
        capsys, "export", "models.py:Marked", "--quantize", "4", "--quantize-method", "gptq",
        "--calibration", "calibration.pt", "-o", "gptq",
    )
    assert code == 0 and report["verify"]["passed"]


def test_fast_exports_verify_and_dotted_names_keep_their_project(models, capsys):
    code, report = run(
        capsys, "export", "models.py:Marked", "--optimization", "fast", "-o", "out/net.v2",
        "--project",
    )
    assert code == 0 and report["verify"]["passed"]
    assert report["path"] == "out/net.v2.sprite3"
    assert report["project"]["path"] == "out/net.v2.sb3"


def test_export_a_saved_module_with_a_dynamic_batch(models, capsys):
    torch.save(nn.Sequential(nn.Linear(4, 2), nn.Sigmoid()), models / "saved.pt")
    code, report = run(capsys, "export", "saved.pt", "--shape", "3,4", "--dynamic", "0=1:8")
    assert code == 0 and report["verify"]["passed"]


def test_exit_statuses(models, capsys, monkeypatch):
    from cattorch import cli

    # 2: invalid arguments, from argparse or from cattorch's own checks.
    assert main([]) == cli.EXIT_USAGE == 2
    assert main(["export", "models.py:Marked", "--dynamic", "zero"]) == 2
    code, message = run(capsys, "export", "models.py:unmarked")
    assert code == 2 and "--shape" in message
    code, message = run(capsys, "export", "models.py:missing", "--shape", "1,2")
    assert code == 2 and "no attribute 'missing'" in message
    code, message = run(capsys, "generate", "models.py:Language", "--top-k", "100")
    assert code == 2 and "top_k" in message
    code, message = run(capsys, "export", "models.py:Marked", "--quantize-method", "gptq")
    assert code == 2 and "needs --quantize" in message
    # 1: the export failed.
    code, message = run(capsys, "export", "models.py:Determinant")
    assert code == cli.EXIT_EXPORT_FAILED == 1 and "unsupported operation" in message
    # 3: the sprite didn't match PyTorch.
    monkeypatch.setattr(cli, "_summary", lambda result: {})
    import sys

    verify_module = sys.modules["cattorch.verify"]
    original = verify_module.verify

    def failing(*args, **kwargs):
        import dataclasses

        return dataclasses.replace(original(*args, **kwargs), passed=False)

    monkeypatch.setattr(verify_module, "verify", failing)
    code = main(["export", "models.py:Marked"])
    assert code == cli.EXIT_VERIFY_FAILED == 3
    assert json.loads(capsys.readouterr().out)["verify"]["passed"] is False


def test_every_option_has_help_and_shows_its_default(capsys):
    import argparse

    from cattorch.cli import _parser

    commands = next(
        action for action in _parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    for command, parser in commands.choices.items():
        for action in parser._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            assert action.help, (command, action.dest)
            if action.option_strings and not isinstance(action, argparse._StoreTrueAction):
                text = parser._get_formatter()._expand_help(action)
                assert "default" in text, (command, action.option_strings, text)


def test_version_and_help(capsys):
    from importlib.metadata import version

    import cattorch

    assert cattorch.__version__ == version("cattorch")
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"cattorch {cattorch.__version__}"
    for command, expected in (
        ("generate", (
            "--top-k K", "(default: 40)", "(default: 0.8)", "(default: float32)",
            "--eos-token-id ID", "--quantize BITS",
        )),
        ("analyze-benchmark", ("saved from Scratch",)),
    ):
        assert main([command, "--help"]) == 0
        text = " ".join(capsys.readouterr().out.split())
        assert all(part in text for part in expected), text


def test_user_mistakes_are_reported_without_a_traceback(models, capsys, tmp_path_factory):
    import subprocess
    import sys

    for argv in (["--shape", "1,x"], ["--tokens", "1,2", "--vocab", "0"]):
        assert main(["export", "models.py:Marked", *argv]) == 2
    code, message = run(
        capsys, "export", "models.py:Marked", "--shape", "3,3",
        "--dynamic", "0=1:4", "--dynamic", "0.0=1:8",
    )
    assert code == 2 and "dimension 0 twice" in message
    # A wrongly shaped input: the model's own error, with the shapes it got.
    code, message = run(capsys, "export", "models.py:Marked", "--shape", "2,5")
    assert code == 1 and message.startswith("cattorch: error: ")
    assert "mat1 and mat2 shapes cannot be multiplied" in message
    assert "the model raised RuntimeError when called with inputs of shape (2, 5)" in message
    assert "control flow" not in message
    code, message = run(capsys, "export", "no_such_module:Net", "--shape", "1,2")
    assert code == 2 and "no module named 'no_such_module'" in message
    for path in ("missing.sb3", "models.py"):
        code, message = run(capsys, "analyze-benchmark", path)
        assert code == 2 and path in message

    # A model's class must be importable to unpickle it. Its module beside
    # the .pt file is found from any directory.
    saved = tmp_path_factory.mktemp("saved")
    (saved / "saved_models.py").write_text(MODELS)
    script = (
        "import sys, torch; sys.path.insert(0, sys.argv[1]); import saved_models; "
        "torch.save(saved_models.Marked(), sys.argv[1] + '/beside.pt')"
    )
    subprocess.run([sys.executable, "-c", script, str(saved)], check=True)
    code, report = run(capsys, "export", str(saved / "beside.pt"), "--shape", "2,3")
    assert code == 0 and report["verify"]["passed"]
    # A class defined in the script that saved the model can't be found.
    script = (
        "import sys, torch\nclass Net(torch.nn.Linear): pass\n"
        "torch.save(Net(2, 2), sys.argv[1] + '/main.pt')\n"
    )
    subprocess.run([sys.executable, "-c", script, str(saved)], check=True)
    code, message = run(capsys, "export", str(saved / "main.pt"), "--shape", "1,2")
    assert code == 2 and "must be importable" in message and "script that saved" in message


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
    assert report["tokenizer"]["path"].endswith("language_tokenizer.sprite3")
    assert report["tokenizer"]["tokenizer_type"] == "char"
    assert report["project"]["path"].endswith("language.sb3")
    assert (report["max_context"], report["vocabulary_size"], report["top_k"]) == (8, 12, None)

    code, message = run(capsys, "generate", "models.py:Language", "--project")
    assert code == 2 and "needs a tokenizer" in message


@pytest.mark.parametrize("output", ["build/demo", "build/demo.sprite3", "build/demo.sb3"])
def test_every_output_is_named_after_the_output_option(models, capsys, output):
    save_characters(models / "chars")
    code, report = run(
        capsys, "generate", "models.py:Language", "--tokenizer", "chars",
        "--max-context", "4", "--project", "-o", output,
    )
    assert code == 0
    assert (report["path"], report["tokenizer"]["path"], report["project"]["path"]) == (
        "build/demo.sprite3", "build/demo_tokenizer.sprite3", "build/demo.sb3",
    )
    assert sorted(path.name for path in (models / "build").iterdir()) == [
        "demo.sb3", "demo.sprite3", "demo_tokenizer.sprite3",
    ]


def max_context(sprite_path):
    sprite = load_sprite(sprite_path)
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
    assert max_context(report["path"]) == "Infinity"

    # The demo runs past any context limit until its token budget is spent.
    emulator = ProjectEmulator(report["project"]["path"], answers=["badge", ""])
    emulator.green_flag()
    ids = [CHARACTERS.index(character) for character in "badge"]
    with torch.no_grad():
        for _ in range(6):
            ids.append(int(model(torch.tensor([ids])).logits[0, -1].argmax()))
    text = "".join(CHARACTERS[index] for index in ids)
    assert text in emulator.sprites["Demo"].said

    code, report = run(capsys, "generate", "--hf", "ssm", "--max-context", "5", "-o", "bounded")
    assert code == 0 and report["max_context"] == 5
    assert max_context(report["path"]) == 5


MORE_MODELS = '''
import torch
from torch import nn

import cattorch


@cattorch.example_inputs(lambda: (torch.randn(1, 3),))
class Two(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, value):
        return self.linear(value), value * 2


class Counter(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("total", torch.zeros(3))

    @cattorch.entrypoint(example_inputs=(torch.ones(3),))
    def add(self, value):
        self.total = self.total + value
        return self.total * 1.5
'''


def test_verify_summary_has_the_largest_error_for_every_kind_of_export(models, capsys):
    (models / "more.py").write_text(MORE_MODELS)
    for spec in ("more.py:Two", "more.py:Counter"):
        code, report = run(capsys, "export", spec)
        assert code == 0, report
        assert report["verify"]["passed"] is True
        assert 0 <= report["verify"]["max_abs_error"] < 1e-4


def test_max_abs_error_of_program_and_multi_output_results():
    from cattorch.cli import _max_abs_error
    from cattorch.results import (
        MultiOutputVerifyResult, ProgramCallVerifyResult, ProgramVerifyResult, VerifyResult,
    )

    def compared(error):
        return VerifyResult(True, 1, (1,), 1, error, error, error, 0)

    assert _max_abs_error(MultiOutputVerifyResult(True, (compared(0.5), compared(0.25)))) == 0.5
    calls = (
        ProgramCallVerifyResult("a", True, (compared(0.1),), {"s": compared(0.3)}),
        ProgramCallVerifyResult("b", True, (compared(0.2),), {}),
    )
    assert _max_abs_error(ProgramVerifyResult(True, calls)) == 0.3
    wrong_status = ProgramCallVerifyResult("c", False, (), {}, status="ok", expected_status="x")
    assert _max_abs_error(ProgramVerifyResult(False, (*calls, wrong_status))) == float("inf")


def test_options_that_cannot_apply_together_are_rejected(models, capsys):
    save_characters(models / "chars")
    for argv, expected in (
        (["generate", "models.py:Language", "--hf", "chars"], "not both"),
        (["export", "models.py:Marked", "--labels", "a,b"], "--labels needs --project"),
        (["export", "models.py:Marked", "--vocab", "5"], "--vocab needs --tokens"),
        (["generate", "models.py:Language", "--temperature", "0"], "needs --project"),
        (["generate", "models.py:Language", "--max-new-tokens", "3"], "needs --project"),
        (["generate", "models.py:Language", "--eos-token-id", "3"], "needs --project"),
    ):
        code, message = run(capsys, *argv)
        assert code == 2 and expected in message, (argv, message)
    assert not list(models.glob("*.sprite3"))


def test_hugging_face_names_that_cannot_be_loaded_exit_with_usage(models, capsys, monkeypatch):
    for argv in (
        ["generate", "--hf", "./no_such_directory"],
        ["generate", "models.py:Language", "--tokenizer", "./no_such_directory"],
    ):
        code, message = run(capsys, *argv)
        assert code == 2 and "can't load" in message and "no_such_directory" in message
    # Without transformers, both flags say how to install it.
    import sys

    monkeypatch.setitem(sys.modules, "transformers", None)
    for argv, flag in (
        (["generate", "--hf", "chars"], "--hf"),
        (["generate", "models.py:Language", "--tokenizer", "chars"], "--tokenizer"),
    ):
        code, message = run(capsys, *argv)
        assert code == 2 and f"{flag} needs transformers" in message, message


def test_an_output_option_naming_a_directory_is_rejected(models, capsys):
    (models / "build").mkdir()
    for output in ("build/", "build", ""):
        code, message = run(capsys, "export", "models.py:Marked", "-o", output)
        assert code == 2 and "-o" in message, (output, message)
    assert not list(models.rglob("*.sprite3"))


def test_an_entry_point_failing_on_its_example_exits_without_a_traceback(models, capsys):
    (models / "entry.py").write_text(
        "import torch\nfrom torch import nn\nimport cattorch\n\n\n"
        "class EntryBad(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.register_buffer('total', torch.zeros(3))\n\n"
        "    @cattorch.entrypoint(example_inputs=(torch.zeros(4),))\n"
        "    def add(self, value):\n"
        "        self.total = self.total + value\n"
        "        return self.total\n"
    )
    code, message = run(capsys, "export", "entry.py:EntryBad")
    assert code == 1 and "@entrypoint add raised RuntimeError" in message
    assert "Traceback" not in message
