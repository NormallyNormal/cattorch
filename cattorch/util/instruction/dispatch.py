"""Central production-kernel registry for ordinary exported ATen nodes."""

from __future__ import annotations

from cattorch.util.instruction.instruction import Instruction
from cattorch.util.instruction.optimized import (
    FastElementwiseInstruction,
    FastLayerNormInstruction,
    FastSoftmaxInstruction,
    LinearInstruction,
    OptimizedBatchNormInstruction,
    OptimizedConvolutionInstruction,
    OptimizedElementwiseInstruction,
    OptimizedEmbeddingInstruction,
    OptimizedLayerNormInstruction,
    OptimizedMaskedFillInstruction,
    OptimizedMatMulInstruction,
    OptimizedMeanInstruction,
    OptimizedPoolingInstruction,
    OptimizedRMSNormInstruction,
    OptimizedSoftmaxInstruction,
    OptimizedTransposeInstruction,
)


ELEMENTWISE_OPS = {
    "aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor", "aten.div.Tensor",
    "aten.neg.default", "aten.relu.default", "aten.sigmoid.default",
    "aten.tanh.default", "aten.gelu.default", "aten.silu.default",
    "aten.leaky_relu.default", "aten.elu.default", "aten.pow.Tensor_Scalar",
    "aten.rsqrt.default",
}

MATMUL_OPS = {
    "aten.matmul.default", "aten.mm.default", "aten.bmm.default",
}

_EXACT_KERNELS: dict[str, type[Instruction]] = {
    "aten.linear.default": LinearInstruction,
    "aten.softmax.int": OptimizedSoftmaxInstruction,
    "aten.mean.dim": OptimizedMeanInstruction,
    "aten.layer_norm.default": OptimizedLayerNormInstruction,
    "aten.rms_norm.default": OptimizedRMSNormInstruction,
    "aten.batch_norm.default": OptimizedBatchNormInstruction,
    "aten.embedding.default": OptimizedEmbeddingInstruction,
    "aten.numpy_T.default": OptimizedTransposeInstruction,
    "aten.transpose.int": OptimizedTransposeInstruction,
    "aten.permute.default": OptimizedTransposeInstruction,
    "aten.masked_fill.Scalar": OptimizedMaskedFillInstruction,
    "aten.conv1d.default": OptimizedConvolutionInstruction,
    "aten.conv2d.default": OptimizedConvolutionInstruction,
    "aten.max_pool1d.default": OptimizedPoolingInstruction,
    "aten.max_pool2d.default": OptimizedPoolingInstruction,
    "aten.avg_pool1d.default": OptimizedPoolingInstruction,
    "aten.avg_pool2d.default": OptimizedPoolingInstruction,
    "aten.adaptive_avg_pool2d.default": OptimizedPoolingInstruction,
    **{operation: OptimizedElementwiseInstruction for operation in ELEMENTWISE_OPS},
    **{operation: OptimizedMatMulInstruction for operation in MATMUL_OPS},
}


def production_kernel(
    aten_op: str,
    *,
    fast_activations: bool = False,
    fast_layer_norm: bool = False,
    fast_softmax: bool = False,
) -> type[Instruction] | None:
    """Select the ordinary production kernel for an ATen operation.

    Graph-pattern specializations such as cached attention are handled before
    this lookup. This registry owns the context-free exact/fast selection.
    """
    if aten_op == "aten.gelu.default" and fast_activations:
        return FastElementwiseInstruction
    if aten_op == "aten.layer_norm.default" and fast_layer_norm:
        return FastLayerNormInstruction
    if aten_op == "aten.softmax.int" and fast_softmax:
        return FastSoftmaxInstruction
    return _EXACT_KERNELS.get(aten_op)


__all__ = ["ELEMENTWISE_OPS", "MATMUL_OPS", "production_kernel"]
