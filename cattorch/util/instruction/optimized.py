"""Compatibility facade for cattorch's domain-specific production kernels."""

from cattorch.util.instruction.optimized_convolution import (
    OptimizedConvolutionInstruction,
    OptimizedPoolingInstruction,
)
from cattorch.util.instruction.optimized_elementwise import (
    FastElementwiseInstruction,
    FusedArithmeticInstruction,
    OptimizedElementwiseInstruction,
    OptimizedSwiGLUInstruction,
    PairedLinearSwiGLUInstruction,
)
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
    "FastElementwiseInstruction",
    "FastLowRankMatMulInstruction",
    "FastPrunedMatMulInstruction",
    "FusedArithmeticInstruction",
    "LinearInstruction",
    "OptimizedConvolutionInstruction",
    "OptimizedElementwiseInstruction",
    "OptimizedMatMulInstruction",
    "OptimizedPoolingInstruction",
    "OptimizedSwiGLUInstruction",
    "PairedLinearSwiGLUInstruction",
    "QKVLinearInstruction",
]
