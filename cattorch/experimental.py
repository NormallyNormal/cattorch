"""Public experimental APIs whose compatibility may evolve during 0.5.x."""

from cattorch import scratch
from cattorch.adapters import AdapterContext, ModuleAdapter
from cattorch.analysis import AnalysisReport, EntryPointAnalysis, analyze
from cattorch.benchmark import build_kernel_benchmark
from cattorch.custom_ops import Operation, op
from cattorch.kernel_api import Kernel, KernelContext, TensorInfo
from cattorch.kernel_testing import check_kernel
from cattorch.moe import (
    ExpertFamily, RoutingScores, SparseMoE, StackedSwiGLUMoE, routed_experts,
    select_routes, stacked_swiglu_moe_adapter,
)
from cattorch.moe_detection import CustomMoEWarning
from cattorch.program import (
    EntryPoint,
    ExportProgram,
    GenerationProgram,
    Input,
    Output,
    ProgramCall,
    State,
    StateInput,
    StateUpdate,
)

__all__ = [
    "AdapterContext", "AnalysisReport", "CustomMoEWarning", "EntryPoint", "EntryPointAnalysis",
    "ExpertFamily", "ExportProgram", "GenerationProgram", "Input", "Kernel", "KernelContext",
    "ModuleAdapter", "Operation", "Output", "ProgramCall", "RoutingScores", "SparseMoE", "StackedSwiGLUMoE",
    "State", "StateInput", "StateUpdate", "TensorInfo", "analyze", "build_kernel_benchmark",
    "check_kernel", "op", "routed_experts", "scratch", "select_routes", "stacked_swiglu_moe_adapter",
]
