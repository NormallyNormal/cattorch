"""Semantic tensor operations with dedicated Scratch lowerings."""

from __future__ import annotations

import torch


@torch.library.custom_op("cattorch::rotary_embedding", mutates_args=())
def _rotary_embedding(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Eager reference for pairwise rotary-position application."""
    pairs = value.reshape(*value.shape[:-1], -1, 2)
    even, odd = pairs.unbind(-1)
    rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
    return value * cosine + rotated * sine


@_rotary_embedding.register_fake
def _rotary_embedding_fake(value, cosine, sine):
    return torch.empty_like(value)


def rotary_embedding(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Apply pairwise RoPE without materializing a rotation matrix.

    The last dimension of ``value`` is treated as adjacent (even, odd) pairs
    and must be even. ``cosine`` and ``sine`` must broadcast to ``value``'s
    shape without enlarging it. cattorch exports this as a single Scratch
    loop.
    """
    if value.ndim < 1 or value.shape[-1] % 2:
        raise ValueError("rotary_embedding requires an even final dimension")
    try:
        output_shape = torch.broadcast_shapes(
            tuple(value.shape), tuple(cosine.shape), tuple(sine.shape),
        )
    except RuntimeError as error:
        raise ValueError("rotary cosine and sine must broadcast with value") from error
    if tuple(output_shape) != tuple(value.shape):
        raise ValueError("rotary cosine and sine may not expand the value shape")
    return _rotary_embedding(value, cosine, sine)


__all__ = ["rotary_embedding"]
