"""Compatibility facade for cattorch's domain-specific production kernels."""

from cattorch.util.instruction.optimized_linear import (
    CausalScoreInstruction,
    CausalSoftmaxInstruction,
    CausalValueMatMulInstruction,
    FastCausalSoftmaxInstruction,
    FastLowRankMatMulInstruction,
    FastPrunedMatMulInstruction,
    LinearInstruction,
    OptimizedMatMulInstruction,
    QKVLinearInstruction,
)

__all__ = [
    "CausalScoreInstruction",
    "CausalSoftmaxInstruction",
    "CausalValueMatMulInstruction",
    "FastCausalSoftmaxInstruction",
    "FastLowRankMatMulInstruction",
    "FastPrunedMatMulInstruction",
    "LinearInstruction",
    "OptimizedMatMulInstruction",
    "QKVLinearInstruction",
]
