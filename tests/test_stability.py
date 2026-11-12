"""The stable Python API and command line (docs/stability.md).

A name in ``STABLE`` may not be removed within 1.x, and ``SIGNATURES`` and
``CLI`` pin how each is called: every parameter of every stable callable
(its name, kind and default), every field of every stable result, config and
program class, and every option of every command. If a test here fails,
the change breaks callers: restore the old form, or keep it working with a
``DeprecationWarning``. An addition (a new name, a parameter with a default,
a field with a default, an option) is allowed: add it to the snapshot.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import inspect
import warnings

import pytest

import cattorch
import cattorch.experimental

STABLE = {
    "transpile", "transpile_tokenizer", "verify", "build_project",
    "CharTokenizer", "BPETokenizer", "SentencePieceBPETokenizer",
    "FastConfig", "FastLayerConfig", "StorageConfig", "QuantizationConfig",
    "QuantizationOverride", "CodegenConfig",
    "GenerationProgram", "RecurrentGenerationProgram", "BoundedDimension", "rotary_embedding",
    "ExportProgram", "EntryPoint", "Input", "Output", "State", "StateInput", "StateUpdate",
    "ProgramCall", "ModuleAdapter", "AdapterContext", "CustomMoEWarning", "CattorchWarning",
    "adapter", "entrypoint", "example_inputs", "export_forward", "generate", "marked_program",
    "MarkedProgram", "moe",
    "CattorchError", "UnsupportedModelError", "UnsupportedOperationError", "INTERFACE_VERSION",
    "ArtifactResult", "EntryPointResult", "GenerationResult", "ProgramCallVerifyResult",
    "ProgramResult", "ProgramVerifyResult", "ProjectResult", "StateResult", "TensorSpec",
    "TokenizerResult", "TranspileResult", "VerifyResult", "MultiOutputVerifyResult",
    "QuantizationReport", "TensorQuantizationReport",
    "build_paired_benchmark", "build_benchmark_suite", "analyze_benchmark",
    "build_generation_benchmark_suite", "build_storage_benchmark_suite",
    "BenchmarkAnalysis", "BenchmarkCase", "StorageBenchmarkCase", "StorageBenchmarkVariant",
}
EXPERIMENTAL = {
    "Kernel", "KernelContext", "TensorInfo", "check_kernel", "build_kernel_benchmark",
    "scratch", "op", "Operation", "ExpertFamily", "SparseMoE", "StackedSwiGLUMoE",
    "routed_experts", "select_routes", "RoutingScores",
    "analyze", "AnalysisReport", "EntryPointAnalysis",
}


def _default(value) -> str:
    if value is inspect.Parameter.empty or value is dataclasses.MISSING:
        return "required"
    if isinstance(value, enum.Enum):
        return str(value)
    return repr(value)


def _parameters(function) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (name, parameter.kind.name, _default(parameter.default))
        for name, parameter in inspect.signature(function).parameters.items()
    )


def _fields(cls) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            field.name,
            _default(field.default) if field.default_factory is dataclasses.MISSING
            else f"factory {field.default_factory.__name__}",
        )
        for field in dataclasses.fields(cls)
    )


def signatures() -> dict:
    """How every stable name is called, or what fields it has."""
    snapshot = {}
    for name in sorted(STABLE):
        value = getattr(cattorch, name)
        if dataclasses.is_dataclass(value):
            snapshot[name] = ("fields", _fields(value))
        elif inspect.isclass(value) and issubclass(value, tuple) and hasattr(value, "_fields"):
            snapshot[name] = ("fields", tuple((field, "required") for field in value._fields))
        elif inspect.isclass(value) and issubclass(value, BaseException):
            snapshot[name] = ("exception", tuple(base.__name__ for base in value.__mro__[1:-2]))
        elif inspect.isclass(value):
            methods = {"__init__": _parameters(value.__init__)}
            if hasattr(value, "save"):
                methods["save"] = _parameters(value.save)
            snapshot[name] = ("class", tuple(sorted(methods.items())))
        elif callable(value):
            snapshot[name] = ("function", _parameters(value))
        else:
            snapshot[name] = ("value", repr(value))
    return snapshot


def cli_options() -> dict:
    """Every option of every ``cattorch`` command: its flags, default and choices."""
    from cattorch.cli import _parser

    snapshot = {}
    parser = _parser()
    commands = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    for command, subparser in sorted(commands.choices.items()):
        snapshot[command] = tuple(
            (
                tuple(action.option_strings) or action.dest,
                repr(action.default),
                tuple(action.choices) if action.choices else (),
            )
            for action in subparser._actions
            if not isinstance(action, argparse._HelpAction)
        )
    return snapshot


SIGNATURES: dict = {}  # Filled in below.

CLI: dict = {}  # Filled in below.


def test_every_stable_name_is_importable_from_cattorch():
    assert STABLE <= set(cattorch.__all__)
    for name in STABLE:
        assert getattr(cattorch, name) is not None


def test_experimental_names_stay_out_of_the_stable_namespace():
    assert set(cattorch.experimental.__all__) == EXPERIMENTAL
    assert not EXPERIMENTAL & set(cattorch.__all__)
    assert "warnings" not in vars(cattorch.experimental)


@pytest.mark.parametrize("name", sorted(STABLE))
def test_stable_signatures_and_fields_are_pinned(name):
    assert signatures()[name] == SIGNATURES[name]


@pytest.mark.parametrize("command", sorted(CLI))
def test_command_line_options_are_pinned(command):
    assert cli_options()[command] == CLI[command]


@pytest.mark.parametrize("name", ["ExportProgram", "ProgramCall", "entrypoint", "ModuleAdapter"])
def test_graduated_names_still_import_from_experimental(name):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = getattr(cattorch.experimental, name)
    assert value is getattr(cattorch, name)
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)


def test_interface_version_is_public():
    from cattorch.interface import INTERFACE_VERSION

    assert cattorch.INTERFACE_VERSION == INTERFACE_VERSION == 1
