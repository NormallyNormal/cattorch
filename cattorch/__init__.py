from cattorch.transpiler import transpile
from cattorch.tokenizer import (
    BPETokenizer,
    CharTokenizer,
    SentencePieceBPETokenizer,
    transpile_tokenizer,
)
from cattorch.fast import FastConfig, FastLayerConfig
from cattorch.storage import StorageConfig
from cattorch.quantization import (
    QuantizationConfig,
    QuantizationReport,
    TensorQuantizationReport,
)
from cattorch.codegen import CodegenConfig
from cattorch.ops import rotary_embedding
from cattorch.adapters import AdapterContext, ModuleAdapter
from cattorch.moe_detection import CustomMoEWarning
from cattorch.markers import (
    adapter, entrypoint, example, export_forward, generate, marked_program, moe,
)
from cattorch.program import (
    EntryPoint, ExportProgram, GenerationProgram, Input, Output, ProgramCall, State,
    StateInput, StateUpdate,
)
from cattorch.errors import CattorchError, UnsupportedModelError, UnsupportedOperationError
from cattorch.results import (
    ArtifactResult, BoundedDimension, EntryPointResult, ProgramCallVerifyResult, ProgramResult,
    ProgramVerifyResult, ProjectResult, StateResult, TensorSpec, TokenizerResult,
    TranspileResult, VerifyResult, MultiOutputVerifyResult,
)
from cattorch.verify import verify
from cattorch.project import build_project


_BENCHMARK_EXPORTS = {
    "analyze_benchmark", "build_benchmark_suite", "build_generation_benchmark_suite",
    "build_paired_benchmark", "build_storage_benchmark_suite",
}


def __getattr__(name):
    if name in _BENCHMARK_EXPORTS:
        from cattorch import benchmark

        value = getattr(benchmark, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "transpile", "transpile_tokenizer", "CharTokenizer", "BPETokenizer",
    "SentencePieceBPETokenizer",
    "FastConfig", "FastLayerConfig", "GenerationProgram", "BoundedDimension",
    "StorageConfig", "QuantizationConfig", "QuantizationReport",
    "TensorQuantizationReport",
    "CodegenConfig", "rotary_embedding",
    "adapter", "entrypoint", "example", "export_forward", "generate", "marked_program", "moe",
    "EntryPoint", "ExportProgram", "Input", "Output", "ProgramCall", "State", "StateInput",
    "StateUpdate", "AdapterContext", "ModuleAdapter", "CustomMoEWarning",
    "CattorchError", "UnsupportedModelError", "UnsupportedOperationError",
    "ArtifactResult", "EntryPointResult", "ProgramCallVerifyResult",
    "ProgramResult", "ProgramVerifyResult", "StateResult",
    "TensorSpec", "TokenizerResult", "TranspileResult",
    "VerifyResult", "MultiOutputVerifyResult", "verify", "build_project", "ProjectResult",
    "build_paired_benchmark", "build_benchmark_suite", "analyze_benchmark",
    "build_generation_benchmark_suite",
    "build_storage_benchmark_suite",
]
