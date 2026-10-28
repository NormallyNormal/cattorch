"""APIs outside cattorch's 1.x compatibility promise.

These may change in any release: the kernel interface and its Scratch DSL,
``op`` for declaring operations, the explicit mixture-of-experts classes, and
``analyze``. See ``docs/stability.md``.

Programs, markers and adapters were experimental before 1.0 and are now
stable top-level names; importing them from here still works but warns.
"""

import warnings

from cattorch import scratch
from cattorch.analysis import AnalysisReport, EntryPointAnalysis, analyze
from cattorch.benchmark import build_kernel_benchmark
from cattorch.custom_ops import Operation, op
from cattorch.kernel_api import Kernel, KernelContext, TensorInfo
from cattorch.kernel_testing import check_kernel
from cattorch.experts import (
    ExpertFamily, RoutingScores, SparseMoE, StackedSwiGLUMoE, routed_experts,
    select_routes, stacked_swiglu_moe_adapter,
)

_GRADUATED = frozenset({
    "adapter", "entrypoint", "example", "export_forward", "generate", "marked_program", "moe",
    "AdapterContext", "ModuleAdapter", "CustomMoEWarning", "EntryPoint", "ExportProgram",
    "GenerationProgram", "Input", "Output", "ProgramCall", "State", "StateInput", "StateUpdate",
})


def __getattr__(name):
    if name in _GRADUATED:
        import cattorch

        warnings.warn(
            f"cattorch.experimental.{name} is now stable; import it from cattorch instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(cattorch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnalysisReport", "EntryPointAnalysis", "ExpertFamily", "Kernel", "KernelContext",
    "Operation", "RoutingScores", "SparseMoE", "StackedSwiGLUMoE", "TensorInfo", "analyze",
    "build_kernel_benchmark", "check_kernel", "op", "routed_experts", "scratch",
    "select_routes", "stacked_swiglu_moe_adapter",
]
