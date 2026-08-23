from cattorch.transpiler import GenerationConfig, transpile
from cattorch.tokenizer import (
    BPETokenizer,
    CharTokenizer,
    SentencePieceBPETokenizer,
    transpile_tokenizer,
)
from cattorch.fast import FastConfig, FastLayerConfig
from cattorch.storage import StorageConfig
from cattorch.codegen import CodegenConfig
from cattorch.errors import CattorchError, UnsupportedModelError, UnsupportedOperationError
from cattorch.results import ArtifactResult, TensorSpec, TokenizerResult, TranspileResult, VerifyResult
from cattorch.verify import verify


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
    "FastConfig", "FastLayerConfig", "GenerationConfig",
    "StorageConfig",
    "CodegenConfig",
    "CattorchError", "UnsupportedModelError", "UnsupportedOperationError",
    "ArtifactResult", "TensorSpec", "TokenizerResult", "TranspileResult",
    "VerifyResult", "verify",
    "build_paired_benchmark", "build_benchmark_suite", "analyze_benchmark",
    "build_generation_benchmark_suite",
    "build_storage_benchmark_suite",
]
