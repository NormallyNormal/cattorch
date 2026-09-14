"""Public experimental APIs whose compatibility may evolve during 0.5.x."""

from cattorch.adapters import AdapterContext, ModuleAdapter
from cattorch.analysis import AnalysisReport, EntryPointAnalysis, analyze
from cattorch.moe import (
    ExpertFamily, RoutingScores, SparseMoE, StackedSwiGLUMoE, select_routes,
    stacked_swiglu_moe_adapter,
)
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
    "AdapterContext", "AnalysisReport", "EntryPoint", "EntryPointAnalysis",
    "ExpertFamily", "ExportProgram", "GenerationProgram", "Input", "ModuleAdapter", "Output",
    "ProgramCall", "RoutingScores", "SparseMoE", "StackedSwiGLUMoE", "State", "StateInput",
    "StateUpdate", "analyze", "select_routes",
    "stacked_swiglu_moe_adapter",
]
