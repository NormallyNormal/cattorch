"""The stable Python API (docs/stability.md).

A name in ``STABLE`` may not be removed within 1.x. If this fails because a
name is missing, restore it, or keep it as a deprecated alias. A new stable
name may be added here.
"""

from __future__ import annotations

import warnings

import pytest

import cattorch
import cattorch.experimental

STABLE = {
    "transpile", "transpile_tokenizer", "verify", "build_project",
    "CharTokenizer", "BPETokenizer", "SentencePieceBPETokenizer",
    "FastConfig", "FastLayerConfig", "StorageConfig", "QuantizationConfig", "CodegenConfig",
    "GenerationProgram", "BoundedDimension", "rotary_embedding",
    "ExportProgram", "EntryPoint", "Input", "Output", "State", "StateInput", "StateUpdate",
    "ProgramCall", "ModuleAdapter", "AdapterContext", "CustomMoEWarning",
    "adapter", "entrypoint", "example", "export_forward", "generate", "marked_program", "moe",
    "CattorchError", "UnsupportedModelError", "UnsupportedOperationError",
    "ArtifactResult", "EntryPointResult", "ProgramCallVerifyResult", "ProgramResult",
    "ProgramVerifyResult", "ProjectResult", "StateResult", "TensorSpec", "TokenizerResult",
    "TranspileResult", "VerifyResult", "MultiOutputVerifyResult",
    "QuantizationReport", "TensorQuantizationReport",
    "build_paired_benchmark", "build_benchmark_suite", "analyze_benchmark",
    "build_generation_benchmark_suite", "build_storage_benchmark_suite",
}
EXPERIMENTAL = {
    "Kernel", "KernelContext", "TensorInfo", "check_kernel", "build_kernel_benchmark",
    "scratch", "op", "Operation", "ExpertFamily", "SparseMoE", "StackedSwiGLUMoE",
    "routed_experts", "select_routes", "RoutingScores", "stacked_swiglu_moe_adapter",
    "analyze", "AnalysisReport", "EntryPointAnalysis",
}


def test_every_stable_name_is_importable_from_cattorch():
    assert STABLE <= set(cattorch.__all__)
    for name in STABLE:
        assert getattr(cattorch, name) is not None


def test_experimental_names_stay_out_of_the_stable_namespace():
    assert set(cattorch.experimental.__all__) == EXPERIMENTAL
    assert not EXPERIMENTAL & set(cattorch.__all__)


@pytest.mark.parametrize("name", ["ExportProgram", "ProgramCall", "entrypoint", "ModuleAdapter"])
def test_graduated_names_still_import_from_experimental(name):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = getattr(cattorch.experimental, name)
    assert value is getattr(cattorch, name)
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
