"""Compatibility facade for the internal operation registry."""

from __future__ import annotations

from cattorch.operator_registry import ELEMENTWISE_OPS, MATMUL_OPS, default_registry
from cattorch.util.instruction.instruction import Instruction


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
    from cattorch.fast import FastConfig

    return default_registry().kernel(
        aten_op,
        fast=fast_activations or fast_layer_norm or fast_softmax,
        config=FastConfig(
            activations=fast_activations,
            layer_norm=fast_layer_norm,
            softmax=fast_softmax,
        ),
    )


__all__ = ["ELEMENTWISE_OPS", "MATMUL_OPS", "production_kernel"]
