"""Semantic tensor operations with dedicated Scratch lowerings."""

from __future__ import annotations

import torch

from cattorch.custom_ops import op


# Declared like user operations; their kernels are in ``builtin_kernels``.
@op(name="cattorch::rotary_embedding", fake=lambda value, cosine, sine: torch.empty_like(value))
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


def _topk_shape(value, k, dim):
    shape = list(value.shape)
    shape[dim] = k
    return shape


@op(
    name="cattorch::topk_values",
    fake=lambda value, k, dim, largest: value.new_empty(_topk_shape(value, k, dim)),
)
def _topk_values(
    value: torch.Tensor, k: int, dim: int, largest: bool,
) -> torch.Tensor:
    """Sorted top-k values; the capture frontend splits ``torch.topk`` into this."""
    return torch.topk(value, k, dim, largest, True).values


@op(
    name="cattorch::topk_indices",
    fake=lambda value, k, dim, largest: value.new_empty(
        _topk_shape(value, k, dim), dtype=torch.int64,
    ),
)
def _topk_indices(
    value: torch.Tensor, k: int, dim: int, largest: bool,
) -> torch.Tensor:
    """Zero-based positions matching ``cattorch::topk_values``.

    Ties keep the lower position, which is also what the Scratch kernel does.
    """
    order = torch.sort(value, dim=dim, descending=largest, stable=True).indices
    return order.narrow(dim, 0, k)


# ── Private ops for the automatic generation step module ────────────────────
# These are not exported: ``cattorch.autoregressive.build_step_module`` inserts
# them, and the generation lowering gives them dedicated Scratch kernels.

def _step_index(position: torch.Tensor, limit: int, what: str) -> int:
    index = int(position)
    if not 0 <= index < limit:
        raise RuntimeError(
            f"the generation step module is full: position {index} is outside "
            f"the {limit} rows of {what}; call reset() or raise max_context"
        )
    return index


@torch.library.custom_op("cattorch::sequence_row", mutates_args=())
def _sequence_row(table: torch.Tensor, position: torch.Tensor, axis: int) -> torch.Tensor:
    """Row ``position`` of ``table`` along ``axis``, keeping the axis with size 1."""
    index = _step_index(position, table.shape[axis], "a position table")
    return table.narrow(axis, index, 1).clone()


@_sequence_row.register_fake
def _sequence_row_fake(table, position, axis):
    shape = list(table.shape)
    shape[axis] = 1
    return table.new_empty(shape)


@torch.library.custom_op("cattorch::cached_attention", mutates_args=("k_cache", "v_cache"))
def _cached_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    position: torch.Tensor,
    scale: float,
    kv_groups: int,
    window: int = 0,
    slopes: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """One causal attention step over a key/value cache.

    ``q`` is ``[B, H, 1, D]`` and ``k``/``v`` are ``[B, H_kv, 1, D]`` with
    ``H = H_kv * kv_groups``. The caches are ``[B, H_kv, max_context, D]``. The
    new key and value are written to row ``position``, then the query attends
    to rows ``0..position``, or the last ``window`` of them when ``window`` is
    positive. ``slopes`` (``[H]``) adds ``slope * (key - position)`` to each
    score, as ALiBi does. ``sinks`` (``[H]``) adds one extra logit per head to
    the softmax that no value is attached to, as GPT-OSS does. Rank-3 tensors
    (``[B, 1, D]`` with caches ``[B, max_context, D]``) are attention without a
    head axis.
    """
    headless = q.ndim == 3
    if headless:
        q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
    cache_k = k_cache.unsqueeze(1) if headless else k_cache
    cache_v = v_cache.unsqueeze(1) if headless else v_cache
    index = _step_index(position, cache_k.shape[-2], "the key/value cache")
    cache_k.narrow(-2, index, 1).copy_(k)
    cache_v.narrow(-2, index, 1).copy_(v)
    first = max(0, index + 1 - window) if window > 0 else 0
    keys = cache_k.narrow(-2, first, index + 1 - first)
    values = cache_v.narrow(-2, first, index + 1 - first)
    if kv_groups > 1:
        keys = keys.repeat_interleave(kv_groups, dim=-3)
        values = values.repeat_interleave(kv_groups, dim=-3)
    scores = torch.matmul(q, keys.transpose(-2, -1)) * scale
    if slopes is not None:
        distance = torch.arange(first - index, 1, dtype=scores.dtype)
        scores = scores + slopes.to(scores.dtype).view(1, -1, 1, 1) * distance
    if sinks is not None:
        sink = sinks.to(scores.dtype).view(1, -1, 1, 1).expand(*scores.shape[:-1], 1)
        probabilities = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[..., :-1]
    else:
        probabilities = torch.softmax(scores, dim=-1)
    out = torch.matmul(probabilities, values)
    return out.squeeze(1) if headless else out


@_cached_attention.register_fake
def _cached_attention_fake(
    q, k, v, k_cache, v_cache, position, scale, kv_groups, window=0, slopes=None, sinks=None,
):
    return q.new_empty((*q.shape[:-1], v.shape[-1]))


@torch.library.custom_op("cattorch::advance_position", mutates_args=("position",))
def _advance_position(position: torch.Tensor) -> None:
    """Move the step module on by one token."""
    position.add_(1)


@_advance_position.register_fake
def _advance_position_fake(position):
    return None


__all__ = ["rotary_embedding"]
