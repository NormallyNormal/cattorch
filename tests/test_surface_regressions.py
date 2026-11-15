"""Regressions from a bug hunt of the user-facing surface: warnings, arguments, paths."""

import ast
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import cattorch
from cattorch import CattorchWarning, entrypoint, transpile
from cattorch.experimental import Kernel
from cattorch.scratch import clear

PACKAGE = Path(cattorch.__file__).parent


class _MixedOutputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x), None, "label"


class _EntryNonTensor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    @entrypoint(example_inputs=(torch.randn(1, 4),))
    def run(self, x):
        return self.fc(x), None


@pytest.mark.parametrize("case", ["outputs", "entrypoint", "kernel"])
def test_export_warnings_are_cattorch_warnings_in_the_result(tmp_path, case):
    if case == "outputs":
        def export():
            return transpile(_MixedOutputs().eval(), torch.randn(1, 4), tmp_path / "m")
        expected = "ignoring non-tensor model outputs"
    elif case == "entrypoint":
        def export():
            return transpile(_EntryNonTensor().eval(), None, tmp_path / "m")
        expected = "@entrypoint run: ignoring non-tensor results"
    else:
        unused = Kernel(
            torch.ops.aten.mish.default, lambda ctx: (clear(ctx.output.list),), name="mish",
        )

        def export():
            return transpile(
                nn.Linear(4, 3).eval(), torch.randn(1, 4), tmp_path / "m", kernels=[unused],
            )
        expected = "was not used"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = export()
    matching = [w for w in caught if expected in str(w.message)]
    assert matching
    assert all(issubclass(w.category, CattorchWarning) for w in matching)
    assert [m for m in result.warnings if expected in m]


# ``warnings.warn`` calls allowed to bypass ``cattorch.errors.warn``: the helper
# itself, and the deprecation of the graduated ``cattorch.experimental`` names
# (raised on import, not by an export).
_RAW_WARN_ALLOWED = {
    ("errors.py", "warn"),
    ("errors.py", "with_warnings"),
    ("experimental.py", "__getattr__"),
}


class _RawWarnFinder(ast.NodeVisitor):
    def __init__(self):
        self.owners = ["<module>"]
        self.found = []

    def visit_FunctionDef(self, node):
        self.owners.append(node.name)
        self.generic_visit(node)
        self.owners.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        target = node.func
        if (
            isinstance(target, ast.Attribute) and target.attr == "warn"
            and isinstance(target.value, ast.Name) and target.value.id == "warnings"
        ):
            self.found.append((node.lineno, self.owners[-1]))
        self.generic_visit(node)


def test_every_export_warning_goes_through_the_warning_helper():
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        finder = _RawWarnFinder()
        finder.visit(ast.parse(path.read_text(encoding="utf-8")))
        offenders += [
            f"{path.relative_to(PACKAGE)}:{line} in {owner}"
            for line, owner in finder.found
            if (path.name, owner) not in _RAW_WARN_ALLOWED
        ]
    assert not offenders, (
        "use cattorch.errors.warn so the warning is a CattorchWarning recorded in "
        f"the export's result: {offenders}"
    )


def _char_tokenizer():
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({"?": 0, "a": 1, "b": 2}, unk_token="?"))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    return PreTrainedTokenizerFast(tokenizer_object=backend)


@pytest.mark.parametrize("bad", ["", ".", "..", "build/", "sub/.", "sub/.."])
def test_output_path_that_names_no_file_raises(tmp_path, monkeypatch, bad):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="output_path .* doesn't name a file"):
        transpile(nn.Linear(4, 3).eval(), torch.randn(1, 4), bad)
    assert list(tmp_path.iterdir()) == []


def test_output_path_naming_a_directory_raises(tmp_path):
    (tmp_path / "build").mkdir()
    (tmp_path / "taken.sprite3").mkdir()
    model, x = nn.Linear(4, 3).eval(), torch.randn(1, 4)
    for bad in (tmp_path / "build", tmp_path / "taken"):
        with pytest.raises(ValueError, match="is a directory"):
            transpile(model, x, bad)
    with pytest.raises(TypeError, match="output_path must be a str or path"):
        transpile(model, x, 3)


def test_every_writer_shares_the_output_path_rule(tmp_path):
    from cattorch import CharTokenizer, build_paired_benchmark, build_project
    from cattorch.util.paths import output_file

    model, x = nn.Linear(4, 3).eval(), torch.randn(1, 4)
    result = transpile(model, x, tmp_path / "net.v2")
    assert result.path == tmp_path / "net.v2.sprite3"
    assert transpile(model, x, tmp_path / "upper.SPRITE3").path.name == "upper.SPRITE3"
    assert build_project(result, tmp_path / "net.v2").path == tmp_path / "net.v2.sb3"
    # Benchmarks used to replace the last suffix instead of appending.
    assert build_paired_benchmark(model, x, tmp_path / "bench.v2", repeats=1) == (
        tmp_path / "bench.v2.sb3"
    )
    for writer in (
        lambda path: build_project(result, path),
        lambda path: build_paired_benchmark(model, x, path, repeats=1),
        lambda path: CharTokenizer(_char_tokenizer()).save(path),
    ):
        with pytest.raises(ValueError, match="doesn't name a file"):
            writer(f"{tmp_path}/")
    assert output_file("x", ".sb3", parameter="-o") == Path("x.sb3")
