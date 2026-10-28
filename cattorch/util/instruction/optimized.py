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
from cattorch.util.instruction.optimized_normalization import (
    FastLayerNormInstruction,
    FastSoftmaxInstruction,
    OptimizedBatchNormInstruction,
    OptimizedLayerNormInstruction,
    OptimizedLogSoftmaxInstruction,
    OptimizedMeanInstruction,
    OptimizedRMSNormInstruction,
    OptimizedSoftmaxInstruction,
)
from cattorch.util.instruction.optimized_tensor import (
    EmbeddingAddInstruction,
    OptimizedCatInstruction,
    OptimizedEmbeddingInstruction,
    OptimizedGetItemInstruction,
    OptimizedMaskedFillInstruction,
    OptimizedTransposeInstruction,
    StridedSliceInstruction,
)

__all__ = [
    "CausalScoreInstruction",
    "CausalSoftmaxInstruction",
    "CausalValueMatMulInstruction",
    "EmbeddingAddInstruction",
    "FastCausalSoftmaxInstruction",
    "FastElementwiseInstruction",
    "FastLayerNormInstruction",
    "FastLowRankMatMulInstruction",
    "FastPrunedMatMulInstruction",
    "FastSoftmaxInstruction",
    "FusedArithmeticInstruction",
    "LinearInstruction",
    "OptimizedBatchNormInstruction",
    "OptimizedCatInstruction",
    "OptimizedConvolutionInstruction",
    "OptimizedElementwiseInstruction",
    "OptimizedEmbeddingInstruction",
    "OptimizedGetItemInstruction",
    "OptimizedLayerNormInstruction",
    "OptimizedLogSoftmaxInstruction",
    "OptimizedMaskedFillInstruction",
    "OptimizedMatMulInstruction",
    "OptimizedMeanInstruction",
    "OptimizedPoolingInstruction",
    "OptimizedRMSNormInstruction",
    "OptimizedSoftmaxInstruction",
    "OptimizedSwiGLUInstruction",
    "PairedLinearSwiGLUInstruction",
    "OptimizedTransposeInstruction",
    "StridedSliceInstruction",
    "QKVLinearInstruction",
]
