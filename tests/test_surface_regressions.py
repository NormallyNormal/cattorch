"""Regressions from a bug hunt of the user-facing surface: warnings, arguments, paths."""

import ast
import dataclasses
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


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


@pytest.mark.parametrize("model", [None, lambda x: x, "model"])
def test_transpile_rejects_a_model_that_is_not_a_module(tmp_path, model):
    with pytest.raises(TypeError, match="model must be a torch.nn.Module"):
        transpile(model, torch.randn(1, 4), tmp_path / "m")


@pytest.mark.parametrize(("inputs", "expected"), [
    ({"x": torch.randn(1, 4)}, "can't be a dict.*example_kwargs"),
    (1.0, "must be a tensor or a tuple of tensors.*not float"),
    ("x", "not str"),
])
def test_transpile_rejects_example_inputs_of_the_wrong_kind(tmp_path, inputs, expected):
    with pytest.raises(TypeError, match=expected):
        transpile(_Net().eval(), inputs, tmp_path / "m")


class _EntryWrongExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("total", torch.zeros(3))

    @entrypoint(example_inputs=(torch.zeros(4),))
    def add(self, value):
        self.total = self.total + value
        return self.total


def test_an_entry_point_failing_on_its_example_is_a_model_error(tmp_path):
    with pytest.raises(cattorch.UnsupportedModelError, match=(
        r"@entrypoint add raised RuntimeError when called with inputs of shape \(4,\)"
    )):
        transpile(_EntryWrongExample().eval(), None, tmp_path / "m")


class _Broadcasting(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x, y):
        return self.fc(x) * y


def test_a_model_failing_within_its_dynamic_bounds_is_a_model_error(tmp_path):
    with pytest.raises(cattorch.UnsupportedModelError, match="with 2 items along the dynamic"):
        transpile(
            _Broadcasting().eval(), (torch.randn(5, 4), torch.randn(5, 3)), tmp_path / "m",
            dynamic={0: (1, 8)},
        )


def test_verify_checks_its_tolerances_and_reads_the_sprite_as_transpile_named_it(
    tmp_path, monkeypatch,
):
    from cattorch import build_project, verify

    monkeypatch.chdir(tmp_path)
    model, x = _Net().eval(), torch.randn(1, 4)
    result = transpile(model, x, "model")
    assert verify(model, x, "model").passed
    assert verify(model, x, tmp_path / "model.sprite3").passed
    for value, error in ((float("nan"), ValueError), (-1, ValueError), ("1", TypeError),
                         (True, TypeError)):
        with pytest.raises(error, match="atol"):
            verify(model, x, result, atol=value)
        with pytest.raises(error, match="rtol"):
            verify(model, x, result, rtol=value)
    project = build_project(result, "project")
    with pytest.raises(ValueError, match="is a Scratch project \\(\\.sb3\\), not a Scratch sprite"):
        verify(model, x, project.path)
    with pytest.raises(ValueError, match="no such file"):
        verify(model, x, "missing")
    (tmp_path / "notes.txt").write_text("hello")
    with pytest.raises(ValueError, match="is not a Scratch sprite"):
        verify(model, x, "notes.txt")
    with pytest.raises(TypeError, match="not a ProjectResult"):
        verify(model, x, project)
    with pytest.raises(TypeError, match="model must be a torch.nn.Module"):
        verify(None, x, result)
    # A model that fails on the inputs is the model's error, as in transpile.
    with pytest.raises(cattorch.UnsupportedModelError, match="the model raised RuntimeError"):
        verify(model, torch.randn(1, 5), result)


def test_analyze_benchmark_explains_files_that_are_not_benchmark_projects(tmp_path):
    from cattorch import analyze_benchmark, build_project

    result = transpile(_Net().eval(), torch.randn(1, 4), tmp_path / "model")
    project = build_project(result, tmp_path / "project")
    (tmp_path / "notes.txt").write_text("hello")
    for path, expected in (
        (result.path, "is a Scratch sprite \\(\\.sprite3\\), not a Scratch project"),
        (tmp_path / "notes.txt", "is not a Scratch project"),
        (tmp_path / "missing.sb3", "no such file"),
        (project.path, "is not a cattorch benchmark project"),
    ):
        with pytest.raises(ValueError, match=expected):
            analyze_benchmark(path)


def test_benchmark_builders_check_example_inputs_before_exporting(tmp_path):
    from cattorch import (
        GenerationProgram, build_benchmark_suite, build_paired_benchmark,
        build_storage_benchmark_suite,
    )

    model = _Net().eval()
    for inputs in (None, GenerationProgram(8), {"x": torch.randn(1, 4)}, ()):
        with pytest.raises(TypeError, match="example_inputs (must|can.t) be"):
            build_paired_benchmark(model, inputs, tmp_path / "b")
    with pytest.raises(TypeError, match="model must be a torch.nn.Module"):
        build_paired_benchmark(None, torch.randn(1, 4), tmp_path / "b")
    for builder in (build_benchmark_suite, build_storage_benchmark_suite):
        with pytest.raises(TypeError, match="case 'net': example_inputs must be"):
            builder([("net", model, None)], tmp_path / "b")
    assert not list(tmp_path.iterdir())


@dataclasses.dataclass
class _Pair:
    logits: torch.Tensor
    hidden: torch.Tensor


@dataclasses.dataclass
class _Note:
    text: str


class _DataclassOutput(nn.Module):
    def __init__(self, holder):
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.holder = holder

    def forward(self, x):
        y = self.fc(x)
        return (y, _Note("meta")) if self.holder is _Note else self.holder(y, y * 2)


def test_tensors_in_an_unsupported_container_are_named(tmp_path):
    with pytest.raises(cattorch.UnsupportedModelError, match=r"inside output \(_Pair\)"):
        transpile(_DataclassOutput(_Pair).eval(), torch.randn(1, 4), tmp_path / "m")
    # A container without tensors is dropped with the usual warning.
    with pytest.warns(CattorchWarning, match=r"output\[1\] \(_Note\)"):
        result = transpile(_DataclassOutput(_Note).eval(), torch.randn(1, 4), tmp_path / "n")
    assert len(result.outputs) == 1


class _EntryDataclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    @entrypoint(example_inputs=(torch.randn(1, 4),))
    def run(self, x):
        y = self.fc(x)
        return _Pair(y, y)


def test_an_entry_point_returning_tensors_in_a_dataclass_is_rejected(tmp_path):
    with pytest.raises(cattorch.UnsupportedModelError, match=r"@entrypoint run's result \(_Pair\)"):
        transpile(_EntryDataclass().eval(), None, tmp_path / "m")


@pytest.mark.parametrize(("options", "error", "expected"), [
    ({"capacity": [4]}, TypeError, "capacity must be a dict"),
    ({"capacity": {"history": 0}}, ValueError, "positive integer"),
    ({"capacity": {"history": True}}, ValueError, "positive integer"),
    ({"dynamic": [(1, 4)]}, TypeError, "dynamic must be a dict"),
    ({"input_names": "rows"}, TypeError, "input_names must be a tuple of strings"),
    ({"output_names": ("a", 1)}, TypeError, "output_names must be a tuple of strings"),
    ({"name": ""}, ValueError, "name must be a non-empty string"),
    ({"name": "two\nlines"}, ValueError, "control characters"),
])
def test_entrypoint_options_are_checked_when_declared(options, error, expected):
    with pytest.raises(error, match=expected):
        entrypoint(**options)


def test_interface_names_reject_control_characters():
    from cattorch import EntryPoint, Output

    with pytest.raises(ValueError, match="control characters"):
        EntryPoint("a\tb", returns=(Output("y"),))
    with pytest.raises(ValueError, match="control characters"):
        Output("y\nz")


def _sentencepiece_processor():
    import io

    spm = pytest.importorskip("sentencepiece")
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(["the cat sat"] * 10), model_writer=model, vocab_size=20,
        add_dummy_prefix=False, remove_extra_whitespaces=False,
        normalization_rule_name="identity", minloglevel=3, hard_vocab_limit=False,
        model_type="bpe",
    )
    return spm.SentencePieceProcessor(model_proto=model.getvalue())


def test_tokenizer_classes_name_the_kind_they_take():
    from cattorch import BPETokenizer, CharTokenizer

    processor = _sentencepiece_processor()
    for exporter in (CharTokenizer, BPETokenizer):
        with pytest.raises(TypeError) as raised:
            exporter(processor)
        message = str(raised.value)
        assert "needs a Hugging Face tokenizer" in message
        assert "or a sentencepiece" not in message
        assert "use SentencePieceBPETokenizer" in message or (
            "with SentencePieceBPETokenizer" in message
        )


@pytest.mark.parametrize("codegen", [{}, [], 0, "compact"])
def test_tokenizer_exports_check_codegen_as_transpile_does(tmp_path, codegen):
    from cattorch import CharTokenizer, transpile_tokenizer

    tokenizer = _char_tokenizer()
    with pytest.raises(TypeError, match="codegen must be a CodegenConfig"):
        transpile(nn.Linear(2, 2).eval(), torch.zeros(1, 2), tmp_path / "m", codegen=codegen)
    with pytest.raises(TypeError, match="codegen must be a CodegenConfig"):
        CharTokenizer(tokenizer).save(tmp_path / "a", codegen=codegen)
    with pytest.raises(TypeError, match="codegen must be a CodegenConfig"):
        transpile_tokenizer(tokenizer, tmp_path / "b", codegen=codegen)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("option", [{"layer_sharing": True}, {"unrolling": "speed"}])
def test_codegen_options_for_models_are_rejected_for_tokenizers(tmp_path, option):
    from cattorch import CharTokenizer, CodegenConfig, transpile_tokenizer

    field = next(iter(option))
    with pytest.raises(ValueError, match=f"CodegenConfig.{field} applies to model exports"):
        CharTokenizer(_char_tokenizer()).save(tmp_path / "a", codegen=CodegenConfig(**option))
    with pytest.raises(ValueError, match=f"CodegenConfig.{field} applies to model exports"):
        transpile_tokenizer(_char_tokenizer(), tmp_path / "b", codegen=CodegenConfig(**option))
    result = CharTokenizer(_char_tokenizer()).save(
        tmp_path / "c", codegen=CodegenConfig(compact_internal_names=True, id_namespace="abc"),
    )
    assert result.path.exists()


@pytest.mark.parametrize("value", ["no", 1, 0, None])
def test_scratch_casefold_must_be_a_bool(tmp_path, value):
    from cattorch import SentencePieceBPETokenizer, transpile_tokenizer

    processor = _sentencepiece_processor()
    with pytest.raises(TypeError, match="scratch_casefold must be True or False"):
        SentencePieceBPETokenizer(processor, scratch_casefold=value)
    with pytest.raises(TypeError, match="scratch_casefold must be True or False"):
        transpile_tokenizer(processor, tmp_path / "t", scratch_casefold=value)
    with pytest.raises(TypeError, match="scratch_casefold must be True or False"):
        transpile_tokenizer(_char_tokenizer(), tmp_path / "t", scratch_casefold=value)


def test_other_flags_reject_values_that_are_only_truthy():
    from cattorch import moe

    with pytest.raises(TypeError, match="normalize must be True or False"):
        moe(router="router", experts="experts", top_k=1, normalize="yes")
    with pytest.raises(TypeError, match="approximate must be True or False"):
        Kernel(torch.ops.aten.mish.default, lambda ctx: (), approximate=1)
