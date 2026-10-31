"""Built-in kernels written against the private context interface.

These double as the proof that ``cattorch.kernel_api`` is sufficient: rotary
embedding uses a custom operator and optional index tables, and top-k uses
static tables, private scratch lists and a layout-dependent epilogue.
"""

from __future__ import annotations

import math
from functools import partial

import torch

from cattorch.kernel_api import Kernel, KernelContext
from cattorch.util.instruction.generic import _reduction_layout
from cattorch.util.instruction.optimized_common import (
    _broadcast_index,
    _static_unrolled_repeat,
    _unrolled_offset_dot_statements,
)
from cattorch.scratch import (
    add,
    append,
    change_var,
    clear,
    div,
    eq,
    gt,
    if_,
    item,
    length,
    lt,
    mathop,
    mul,
    repeat,
    replace,
    set_var,
    static_unrolled_repeat,
    sub,
    var,
)


def _validate_rotary_embedding(ctx: KernelContext) -> str | None:
    value_shape = ctx.inputs["value"].shape
    if not value_shape or value_shape[-1] % 2:
        return "Rotary embedding requires an even final dimension"
    try:
        output_shape = torch.broadcast_shapes(
            value_shape, ctx.inputs["cosine"].shape, ctx.inputs["sine"].shape,
        )
    except RuntimeError:
        return "Rotary cosine and sine must broadcast with the input"
    if tuple(output_shape) != tuple(value_shape):
        return "Rotary cosine and sine may not expand the input shape"
    return None


def _emit_rotary_embedding(ctx: KernelContext):
    """Fused exact pairwise rotary embedding application."""
    value, cosine, sine = ctx.inputs["value"], ctx.inputs["cosine"], ctx.inputs["sine"]
    output_shape = value.shape
    if not output_shape or output_shape[-1] % 2:
        raise ValueError("rotary embedding requires an even final dimension")
    index = ctx.temp_var("index")
    out = ctx.output.list
    even = var(index)
    odd = add(var(index), 1)
    cos_even, cos_even_map = _broadcast_index(
        cosine.shape, output_shape, "_rope cos map", even,
    )
    cos_odd, cos_odd_map = _broadcast_index(
        cosine.shape, output_shape, "_rope cos map", odd,
    )
    sin_even, sin_even_map = _broadcast_index(
        sine.shape, output_shape, "_rope sin map", even,
    )
    sin_odd, sin_odd_map = _broadcast_index(
        sine.shape, output_shape, "_rope sin map", odd,
    )
    if cos_even_map is not None or cos_odd_map is not None:
        ctx.static_list("_rope cos map", cos_even_map or cos_odd_map)
    if sin_even_map is not None or sin_odd_map is not None:
        ctx.static_list("_rope sin map", sin_even_map or sin_odd_map)
    pair = (
        append(out, add(
            mul(item(value.list, even), item(cosine.list, cos_even)),
            mul(mul(-1, item(value.list, odd)), item(sine.list, sin_even)),
        )),
        append(out, add(
            mul(item(value.list, odd), item(cosine.list, cos_odd)),
            mul(item(value.list, even), item(sine.list, sin_odd)),
        )),
        change_var(index, 2),
    )
    return (
        clear(out),
        set_var(index, 1),
        static_unrolled_repeat(
            math.prod(output_shape) // 2, pair, max_factor=4, priority=3,
        ),
    )


def _emit_rotary_half(ctx: KernelContext):
    """Fused exact Hugging Face RoPE, pairing position ``i`` with ``i + half``.

    Each row writes its first half, then its second half, so the output is
    appended in order.
    """
    value, cosine, sine = ctx.inputs["value"], ctx.inputs["cosine"], ctx.inputs["sine"]
    output_shape = value.shape
    if not output_shape or output_shape[-1] % 2:
        raise ValueError("rotary embedding requires an even final dimension")
    half = output_shape[-1] // 2
    index = ctx.temp_var("index")
    out = ctx.output.list
    first = var(index)
    second = add(var(index), half)
    indices = {}
    for name, tensor in (("cos", cosine), ("sin", sine)):
        map_name = f"_rope {name} map"
        low, table = _broadcast_index(tensor.shape, output_shape, map_name, first)
        high, _ = _broadcast_index(tensor.shape, output_shape, map_name, second)
        if table is not None:
            ctx.static_list(map_name, table)
        indices[name] = (low, high)
    (cos_low, cos_high), (sin_low, sin_high) = indices["cos"], indices["sin"]
    low_half = _static_unrolled_repeat(half, (
        append(out, sub(
            mul(item(value.list, first), item(cosine.list, cos_low)),
            mul(item(value.list, second), item(sine.list, sin_low)),
        )),
        change_var(index, 1),
    ))
    high_half = _static_unrolled_repeat(half, (
        append(out, add(
            mul(item(value.list, second), item(cosine.list, cos_high)),
            mul(item(value.list, first), item(sine.list, sin_high)),
        )),
        change_var(index, 1),
    ))
    row = (*low_half, change_var(index, -half), *high_half, change_var(index, half))
    rows = math.prod(output_shape) // output_shape[-1]
    return (
        clear(out),
        set_var(index, 1),
        *(row if rows == 1 else (repeat(rows, row),)),
    )


def _emit_topk(ctx: KernelContext, *, values: bool):
    """Select the ``k`` largest or smallest entries along one dimension.

    Each group is scanned ``k`` times, marking chosen positions in a scratch
    list, so the cost is ``k`` passes over the dimension. That suits the small
    ``k`` of MoE routers and sampling. Ties keep the lower position. The
    ``cattorch.topk_values`` target writes values and ``cattorch.topk_indices``
    writes zero-based positions.
    """
    source = ctx.inputs["value"]
    shape = source.shape
    k = int(ctx.scalars["k"])
    dim = int(ctx.scalars["dim"]) % len(shape)
    compare = gt if ctx.scalars["largest"] else lt
    extent = shape[dim]
    stride = math.prod(shape[dim + 1:])
    groups = math.prod(shape) // extent
    _, _, bases, _ = _reduction_layout(shape, (dim,))
    for name in ("group", "base", "index", "position", "best", "acc"):
        ctx.temp_var(name)
    bases_list = ctx.static_list("topk_bases", [base + 1 for base in bases])
    taken = ctx.temp_list("topk_taken", extent)
    order_list = scratch = None
    order = []
    if stride != 1:
        # Results are group-major; permute them into PyTorch's layout.
        outer = groups // stride
        for block in range(outer):
            for rank in range(k):
                for offset in range(stride):
                    group = block * stride + offset
                    order.append(group * k + rank + 1)
        order_list = ctx.static_list("topk_order", order)
        scratch = ctx.temp_list("topk_scratch", len(order))
    out = ctx.output.list

    candidate = item(source.list, var("index"))
    scan = (
        set_var("best", 0),
        set_var("index", var("base")),
        set_var("position", 1),
        repeat(extent, (
            if_(eq(item(taken, var("position")), 0), (
                if_(
                    # The first untaken position always becomes the best.
                    eq(var("best"), 0),
                    (set_var("best", var("position")), set_var("acc", candidate)),
                ),
                if_(compare(candidate, var("acc")), (
                    set_var("best", var("position")), set_var("acc", candidate),
                )),
            )),
            change_var("index", stride),
            change_var("position", 1),
        )),
        replace(taken, var("best"), 1),
        append(out, var("acc") if values else sub(var("best"), 1)),
    )
    body = (
        clear(taken),
        repeat(extent, (append(taken, 0),)),
        set_var("base", item(bases_list, var("group"))),
        repeat(k, scan),
        change_var("group", 1),
    )
    program_body = [clear(out), set_var("group", 1), repeat(groups, body)]
    if stride != 1:
        program_body += [
            clear(scratch),
            set_var("index", 1),
            repeat(len(order), (
                append(scratch, item(out, item(order_list, var("index")))),
                change_var("index", 1),
            )),
            clear(out),
            set_var("index", 1),
            repeat(len(order), (
                append(out, item(scratch, var("index"))),
                change_var("index", 1),
            )),
        ]
    return tuple(program_body)


# ── Automatic generation ────────────────────────────────────────────────────
# ``cattorch::sequence_row`` and ``cattorch::cached_attention`` come from
# ``cattorch.autoregressive``. The compiler binds their cache inputs to the
# ``cattorch K/V cache n`` lists and drops their ``position`` input: both
# kernels read the position from ``cattorch cache length``, which the
# generation wrapper keeps equal to the number of cached tokens.

_POSITION = "cattorch cache length"


def _attention_geometry(shapes):
    q, k, v, cache_k, cache_v = shapes
    headless = len(q) == 3
    heads = 1 if headless else q[-3]
    kv_heads = 1 if headless else k[-3]
    return headless, heads, kv_heads, q[-1], v[-1], cache_k[-2]


def _validate_cached_attention(ctx: KernelContext) -> str | None:
    names = ("q", "k", "v", "k_cache", "v_cache")
    if any(name not in ctx.inputs for name in names):
        return "cached attention needs q, k, v and both caches"
    q, k, v, cache_k, cache_v = (ctx.inputs[name].shape for name in names)
    if len(q) not in (3, 4) or not (len(q) == len(k) == len(v) == len(cache_k) == len(cache_v)):
        return "cached attention takes [B, H, 1, D] (or [B, 1, D]) tensors"
    if q[0] != 1 or q[-2] != 1 or k[-2] != 1 or v[-2] != 1:
        return "cached attention processes one token of one sequence"
    headless, heads, kv_heads, width, value_width, capacity = _attention_geometry(
        (q, k, v, cache_k, cache_v),
    )
    groups = int(ctx.scalars["kv_groups"])
    if k[-1] != width or cache_k[-1] != width or cache_v[-1] != value_width:
        return "cached attention key and value widths do not match the caches"
    if groups < 1 or heads != kv_heads * groups:
        return f"{heads} query heads are not {groups} x {kv_heads} key/value heads"
    if cache_v[-2] != capacity:
        return "the key and value caches differ in length"
    return None


def _emit_cached_attention(ctx: KernelContext, *, fast: bool):
    """One causal attention step over the generation K/V cache lists.

    Appends the new key and value rows to their caches, then for each query
    head scores it against every cached key (``cache_tokens`` of them),
    softmaxes and applies the cached values. Grouped-query heads share the key
    and value head ``head // kv_groups``. The exact kernel subtracts the row
    maximum; the fast kernel (``softmax`` flag) skips it.

    A positive ``window`` attends only to the last ``window`` cached tokens.
    ``slopes`` adds ``slope * (key - query)`` per head (ALiBi), and ``sinks``
    adds a per-head logit to the softmax denominator only (GPT-OSS).
    """
    q, k, v = ctx.inputs["q"], ctx.inputs["k"], ctx.inputs["v"]
    k_cache, v_cache = ctx.inputs["k_cache"], ctx.inputs["v_cache"]
    _, heads, kv_heads, width, value_width, capacity = _attention_geometry(
        tuple(info.shape for info in (q, k, v, k_cache, v_cache)),
    )
    groups = int(ctx.scalars["kv_groups"])
    scale = float(ctx.scalars["scale"])
    window = int(ctx.scalars.get("window") or 0)
    slopes, sinks = ctx.inputs.get("slopes"), ctx.inputs.get("sinks")
    key_row = kv_heads * width
    value_row = kv_heads * value_width
    stride = int(ctx.options.get("cache_stride", 0))
    key_stride = stride or key_row
    value_stride = stride or value_row
    base_name = ctx.options.get("cache_base_variable")
    base = 0
    if base_name:
        base = var(ctx.temp_var(str(base_name)))
    position = ctx.temp_var(_POSITION)
    for name in (
        "copy_index", "cache_tokens", "query_start", "left_index", "key_offset",
        "key_head", "value_head", "heads_on_key", "feature", "value_index",
        "score_index", "sum", "denominator", "score", *(() if fast else ("maximum",)),
        *(("attend_tokens", "first_token") if window else ()),
        *(("head_number",) if slopes is not None or sinks is not None else ()),
        *(("key_distance",) if slopes is not None else ()),
    ):
        ctx.temp_var(name)
    scores = ctx.temp_list("attention scores", capacity)
    out = ctx.output.list

    def append_row(source: str, cache: str, count: int):
        return (
            set_var("copy_index", 1),
            *_static_unrolled_repeat(count, (
                append(cache, item(source, var("copy_index"))),
                change_var("copy_index", 1),
            )),
        )

    dot = _unrolled_offset_dot_statements(
        width, left_list=q.list, right_list=k_cache.list, right_offset="key_offset",
    )
    logit = mul(var("sum"), scale)
    if slopes is not None:
        logit = add(logit, mul(item(slopes.list, var("head_number")), var("key_distance")))
    sink = None if sinks is None else item(sinks.list, var("head_number"))
    # Without a window every cached token is attended, from the first.
    tokens = var("attend_tokens") if window else var("cache_tokens")

    def from_first(start, stride):
        return add(start, mul(var("first_token"), stride)) if window else start

    if fast:
        score_step = (
            set_var("score", mathop("e ^", logit)),
            append(scores, var("score")),
            change_var("denominator", var("score")),
        )
    else:
        score_step = (
            set_var("score", logit),
            append(scores, var("score")),
            if_(gt(var("score"), var("maximum")), (set_var("maximum", var("score")),)),
        )
    if slopes is not None:
        score_step += (change_var("key_distance", 1),)
    sink_term = () if sink is None else (
        change_var("denominator", mathop("e ^", sink if fast else sub(sink, var("maximum")))),
    )
    normalize = sink_term if fast else (
        set_var("denominator", 0),
        set_var("score_index", 1),
        repeat(tokens, (
            set_var("score", mathop("e ^", sub(item(scores, var("score_index")), var("maximum")))),
            replace(scores, var("score_index"), var("score")),
            change_var("denominator", var("score")),
            change_var("score_index", 1),
        )),
        *sink_term,
    )
    head = (
        clear(scores),
        # A sink takes part in the maximum, so the shifted exponents stay small.
        *(() if fast else (set_var("maximum", -1e300 if sink is None else sink),)),
        set_var("denominator", 0),
        set_var("key_offset", from_first(
            sub(var("key_head"), var("query_start")), key_stride,
        )),
        *(() if slopes is None else (
            set_var("key_distance", sub(
                var("first_token") if window else 0, var(position),
            )),
        )),
        repeat(tokens, (
            set_var("sum", 0),
            set_var("left_index", var("query_start")),
            *dot,
            *score_step,
            change_var("key_offset", key_stride),
        )),
        *normalize,
        set_var("feature", 0),
        repeat(value_width, (
            set_var("sum", 0),
            set_var("score_index", 1),
            set_var("value_index", from_first(
                add(var("value_head"), var("feature")), value_stride,
            )),
            repeat(tokens, (
                change_var("sum", mul(
                    item(scores, var("score_index")),
                    item(v_cache.list, var("value_index")),
                )),
                change_var("score_index", 1),
                change_var("value_index", value_stride),
            )),
            append(out, div(var("sum"), var("denominator"))),
            change_var("feature", 1),
        )),
        change_var("query_start", width),
        *(() if slopes is None and sinks is None else (change_var("head_number", 1),)),
        change_var("heads_on_key", 1),
        if_(gt(var("heads_on_key"), groups - 1), (
            set_var("heads_on_key", 0),
            change_var("key_head", width),
            change_var("value_head", value_width),
        )),
    )
    attend = (
        clear(out),
        set_var("cache_tokens", add(var(position), 1)),
        *(() if not window else (
            set_var("first_token", 0),
            if_(gt(var("cache_tokens"), window), (
                set_var("first_token", sub(var("cache_tokens"), window)),
            )),
            set_var("attend_tokens", sub(var("cache_tokens"), var("first_token"))),
        )),
        *(() if slopes is None and sinks is None else (set_var("head_number", 1),)),
        set_var("query_start", 1),
        set_var("key_head", add(base, 1)),
        set_var("value_head", add(base, 1)),
        set_var("heads_on_key", 0),
        repeat(heads, head),
    )
    if ctx.options.get("hidden_prefill_last"):
        # The final layer's attention output only feeds the logits, which a
        # hidden prefill token never reads: keep the cache append, skip the rest.
        attend = (if_(eq(var(ctx.temp_var("cattorch project output")), 1), attend),)
    return (
        *append_row(k.list, k_cache.list, key_row),
        *append_row(v.list, v_cache.list, value_row),
        *attend,
    )


def _validate_sequence_row(ctx: KernelContext) -> str | None:
    shape = ctx.inputs["table"].shape
    axis = int(ctx.scalars["axis"])
    if not shape or not -len(shape) <= axis < len(shape):
        return "sequence_row axis is outside the table"
    return None


def _emit_sequence_row(ctx: KernelContext):
    """Copy row ``cattorch cache length`` of a position table along one axis."""
    table = ctx.inputs["table"]
    shape = table.shape
    axis = int(ctx.scalars["axis"]) % len(shape)
    rows = shape[axis]
    inner = math.prod(shape[axis + 1:])
    outer = math.prod(shape[:axis])
    position = ctx.temp_var(_POSITION)
    index = ctx.temp_var("row_index")
    out = ctx.output.list
    copy = _static_unrolled_repeat(inner, (
        append(out, item(table.list, var(index))),
        change_var(index, 1),
    ))
    if outer == 1:
        loop = copy
    else:
        gap = (rows - 1) * inner
        loop = (repeat(outer, (*copy, *((change_var(index, gap),) if gap else ()))),)
    return (
        clear(out),
        set_var(index, add(mul(var(position), inner), 1)),
        *loop,
    )


def _index_bases(index_shape, target_shape, dim: int) -> list[int]:
    """One-based flat positions in ``target_shape`` of each index entry, with
    the ``dim`` coordinate set to zero."""
    strides = [math.prod(target_shape[axis + 1:]) for axis in range(len(target_shape))]
    bases = []
    for position in range(math.prod(index_shape)):
        offset, remainder = 0, position
        for axis in reversed(range(len(index_shape))):
            coordinate = remainder % index_shape[axis]
            remainder //= index_shape[axis]
            if axis != dim:
                offset += coordinate * strides[axis]
        bases.append(offset + 1)
    return bases


def _validate_runtime_index(ctx: KernelContext) -> str | None:
    source, index = ctx.inputs["self"], ctx.inputs["index"]
    if len(index.shape) != len(source.shape):
        return "index must have as many dimensions as the input"
    dim = int(ctx.scalars["dim"]) % max(len(source.shape), 1)
    if any(
        index.shape[axis] > source.shape[axis]
        for axis in range(len(source.shape)) if axis != dim
    ):
        return "index may not be larger than the input outside dim"
    src = ctx.inputs.get("src")
    if src is not None and any(
        index.shape[axis] > src.shape[axis] for axis in range(len(src.shape))
    ):
        return "index may not be larger than src"
    return None


def _validate_scatter(ctx: KernelContext) -> str | None:
    if ctx.scalars.get("reduce") not in (None, "none"):
        return "this scatter takes no reduction"
    return _validate_runtime_index(ctx)


# The identity each reduction starts a target from without ``include_self``.
_SCATTER_REDUCTIONS = {
    "sum": 0, "add": 0, "mean": 0, "prod": 1, "multiply": 1,
    "amax": -math.inf, "amin": math.inf,
}


def _validate_scatter_reduce(ctx: KernelContext) -> str | None:
    reduce = ctx.scalars.get("reduce", "sum")
    if reduce not in _SCATTER_REDUCTIONS:
        return f"scatter reduction {reduce!r} is not supported"
    return _validate_runtime_index(ctx)


def _emit_gather(ctx: KernelContext):
    """``gather`` whose indices are computed at run time: one read per output."""
    source, index = ctx.inputs["self"], ctx.inputs["index"]
    dim = int(ctx.scalars["dim"]) % len(source.shape)
    stride = math.prod(source.shape[dim + 1:])
    bases = ctx.static_list("gather_bases", _index_bases(index.shape, source.shape, dim))
    position = ctx.temp_var("index")
    out = ctx.output.list
    return (
        clear(out),
        set_var(position, 1),
        repeat(index.numel, (
            append(out, item(source.list, add(
                item(bases, var(position)), mul(item(index.list, var(position)), stride),
            ))),
            change_var(position, 1),
        )),
    )


def _emit_scatter_reduce(ctx: KernelContext, *, scalar: bool, default: str | None = None):
    """``scatter_add`` and ``scatter_reduce``: combine each entry into its target.

    Targets start from the input, or from the reduction's identity without
    ``include_self``. A mean also counts the values reaching each position.
    """
    source, index = ctx.inputs["self"], ctx.inputs["index"]
    dim = int(ctx.scalars["dim"]) % len(source.shape)
    stride = math.prod(source.shape[dim + 1:])
    reduce = default or ctx.scalars.get("reduce") or "sum"
    include_self = bool(ctx.scalars.get("include_self", True))
    targets = ctx.static_list("scatter_targets", _index_bases(index.shape, source.shape, dim))
    position = ctx.temp_var("index")
    target = ctx.temp_var("target")
    if scalar:
        written = ctx.scalars["value"]
    else:
        src = ctx.inputs["src"]
        sources = ctx.static_list(
            "scatter_sources", _index_bases(index.shape, src.shape, len(src.shape)),
        )
        written = item(src.list, item(sources, var(position)))
    out = ctx.output.list
    current = item(out, var(target))
    if reduce in {"amax", "amin"}:
        better = gt if reduce == "amax" else lt
        combine = (if_(better(written, current), (replace(out, var(target), written),)),)
    elif reduce in {"prod", "multiply"}:
        combine = (replace(out, var(target), mul(current, written)),)
    else:
        combine = (replace(out, var(target), add(current, written)),)

    def each_entry(*statements):
        return (
            set_var(position, 1),
            repeat(index.numel, (
                set_var(target, add(
                    item(targets, var(position)), mul(item(index.list, var(position)), stride),
                )),
                *statements,
                change_var(position, 1),
            )),
        )

    body = [
        clear(out),
        set_var(position, 1),
        repeat(source.numel, (
            append(out, item(source.list, var(position))),
            change_var(position, 1),
        )),
    ]
    if not include_self:
        body += each_entry(replace(out, var(target), _SCATTER_REDUCTIONS[reduce]))
    if reduce == "mean":
        counts = ctx.temp_list("scatter_counts", source.numel)
        body += [
            clear(counts),
            repeat(source.numel, (append(counts, 1 if include_self else 0),)),
        ]
        body += each_entry(
            *combine,
            replace(counts, var(target), add(item(counts, var(target)), 1)),
        )
        body += [
            set_var(position, 1),
            repeat(source.numel, (
                if_(gt(item(counts, var(position)), 0), (replace(
                    out, var(position),
                    div(item(out, var(position)), item(counts, var(position))),
                ),)),
                change_var(position, 1),
            )),
        ]
    else:
        body += each_entry(*combine)
    return tuple(body)


def _emit_scatter(ctx: KernelContext, *, scalar: bool):
    """``scatter`` whose indices are computed at run time: copy, then write."""
    source, index = ctx.inputs["self"], ctx.inputs["index"]
    dim = int(ctx.scalars["dim"]) % len(source.shape)
    stride = math.prod(source.shape[dim + 1:])
    targets = ctx.static_list("scatter_targets", _index_bases(index.shape, source.shape, dim))
    position = ctx.temp_var("index")
    if scalar:
        written = ctx.scalars["value"]
    else:
        src = ctx.inputs["src"]
        # Each index entry reads ``src`` at its own coordinates; no axis is
        # replaced, so pass a ``dim`` past the last axis.
        sources = ctx.static_list(
            "scatter_sources", _index_bases(index.shape, src.shape, len(src.shape)),
        )
        written = item(src.list, item(sources, var(position)))
    out = ctx.output.list
    return (
        clear(out),
        set_var(position, 1),
        repeat(source.numel, (
            append(out, item(source.list, var(position))),
            change_var(position, 1),
        )),
        set_var(position, 1),
        repeat(index.numel, (
            replace(out, add(
                item(targets, var(position)), mul(item(index.list, var(position)), stride),
            ), written),
            change_var(position, 1),
        )),
    )


_GATES = {"gru": 3, "lstm": 4, "rnn_tanh": 1, "rnn_relu": 1}


def _sigmoid(value):
    return div(1, add(1, mathop("e ^", mul(-1, value))))


def _tanh(value):
    # Matches the elementwise tanh kernel.
    return sub(div(2, add(1, mathop("e ^", mul(-2, value)))), 1)


def _emit_recurrent_scan(ctx: KernelContext):
    """Step a GRU, LSTM or RNN layer through time in one Scratch loop.

    Each step takes the hidden-to-hidden product into a scratch list, applies
    the gates unit by unit while updating a private copy of the state, then
    appends the state to the output. The cost is one ``G * H * P`` product per
    step and batch item, and the generated program does not grow with the
    sequence length.
    """
    gates, state_in, weight = ctx.inputs["gates"], ctx.inputs["state"], ctx.inputs["weight"]
    bias, projection = ctx.inputs.get("bias"), ctx.inputs.get("projection")
    kind = str(ctx.scalars["kind"])
    steps, batch, gate_width = gates.shape
    width = state_in.shape[-1]
    hidden = gate_width // _GATES[kind]
    previous = weight.shape[1]
    for name in ("scan_gate", "scan_state", "scan_weight", "scan_unit", "scan_index", "scan_acc",
                 "scan_cell"):
        ctx.temp_var(name)
    state = ctx.temp_list("scan_state", batch * width)
    recurrent = ctx.temp_list("scan_recurrent", gate_width)
    raw = ctx.temp_list("scan_raw", hidden) if projection is not None else None
    out = ctx.output.list

    def gate(number):
        offset = add(var("scan_unit"), number * hidden)
        return add(item(gates.list, add(var("scan_gate"), offset)), item(recurrent, offset))

    def state_at(offset):
        return add(var("scan_state"), offset)

    product = (
        set_var("scan_acc", 0 if bias is None else item(bias.list, var("scan_unit"))),
        set_var("scan_index", 1),
        static_unrolled_repeat(previous, (
            change_var("scan_acc", mul(
                item(weight.list, add(var("scan_weight"), var("scan_index"))),
                item(state, state_at(var("scan_index"))),
            )),
            change_var("scan_index", 1),
        ), max_factor=4, priority=3),
        append(recurrent, var("scan_acc")),
        change_var("scan_weight", previous),
        change_var("scan_unit", 1),
    )
    unit = var("scan_unit")
    if kind == "gru":
        # ``scan_cell`` holds the candidate state n, ``scan_acc`` the update gate z.
        update = (
            set_var("scan_acc", _sigmoid(gate(1))),
            set_var("scan_cell", _tanh(add(
                item(gates.list, add(var("scan_gate"), add(unit, 2 * hidden))),
                mul(_sigmoid(gate(0)), item(recurrent, add(unit, 2 * hidden))),
            ))),
            replace(state, state_at(unit), add(var("scan_cell"), mul(
                var("scan_acc"), sub(item(state, state_at(unit)), var("scan_cell")),
            ))),
        )
    elif kind == "lstm":
        cell = state_at(add(unit, previous))
        hidden_value = mul(_sigmoid(gate(3)), _tanh(var("scan_cell")))
        update = (
            set_var("scan_cell", add(
                mul(_sigmoid(gate(1)), item(state, cell)),
                mul(_sigmoid(gate(0)), _tanh(gate(2))),
            )),
            replace(state, cell, var("scan_cell")),
            replace(raw, unit, hidden_value) if raw is not None
            else replace(state, state_at(unit), hidden_value),
        )
    elif kind == "rnn_tanh":
        update = (replace(state, state_at(unit), _tanh(gate(0))),)
    else:
        update = (
            set_var("scan_acc", gate(0)),
            replace(state, state_at(unit), mul(var("scan_acc"), gt(var("scan_acc"), 0))),
        )
    project = ()
    if raw is not None:
        # The projected hidden state replaces the first ``previous`` entries.
        project = (
            set_var("scan_weight", 0),
            set_var("scan_unit", 1),
            repeat(previous, (
                set_var("scan_acc", 0),
                set_var("scan_index", 1),
                static_unrolled_repeat(hidden, (
                    change_var("scan_acc", mul(
                        item(projection.list, add(var("scan_weight"), var("scan_index"))),
                        item(raw, var("scan_index")),
                    )),
                    change_var("scan_index", 1),
                ), max_factor=4, priority=3),
                replace(state, state_at(unit), var("scan_acc")),
                change_var("scan_weight", hidden),
                change_var("scan_unit", 1),
            )),
        )
    reverse = bool(ctx.scalars["reverse"])
    if reverse:
        # Steps run from the last to the first but are written in time order.
        write = (replace(out, add(var("scan_out"), var("scan_index")),
                         item(state, state_at(var("scan_index")))),)
    else:
        write = (append(out, item(state, state_at(var("scan_index")))),)
    per_item = (
        clear(recurrent),
        set_var("scan_weight", 0),
        set_var("scan_unit", 1),
        repeat(gate_width, product),
        set_var("scan_unit", 1),
        repeat(hidden, (*update, change_var("scan_unit", 1))),
        *project,
        set_var("scan_index", 1),
        repeat(width, (*write, change_var("scan_index", 1))),
        change_var("scan_state", width),
        change_var("scan_gate", gate_width),
        change_var("scan_out", width),
    )
    if raw is not None:
        per_item = (clear(raw), repeat(hidden, (append(raw, 0),)), *per_item)
    ctx.temp_var("scan_out")
    count = (
        div(length(gates.list), batch * gate_width) if gates.dynamic_axis is not None else steps
    )
    step = (set_var("scan_state", 0), repeat(batch, per_item))
    if reverse:
        start = (
            set_var("scan_gate", mul(sub(count, 1), batch * gate_width)),
            set_var("scan_out", mul(sub(count, 1), batch * width)),
            repeat(mul(count, batch * width), (append(out, 0),)),
        )
        step += (
            change_var("scan_gate", -2 * batch * gate_width),
            change_var("scan_out", -2 * batch * width),
        )
    else:
        start = (set_var("scan_gate", 0), set_var("scan_out", 0))
    return (
        clear(out),
        clear(state),
        set_var("scan_index", 1),
        repeat(batch * width, (
            append(state, item(state_in.list, var("scan_index"))),
            change_var("scan_index", 1),
        )),
        *start,
        repeat(count, step),
    )


def _emit_recurrent_final(ctx: KernelContext):
    """Copy the first or last step of ``recurrent_scan``'s output."""
    steps = ctx.inputs["steps"]
    size = math.prod(steps.shape[1:])
    position = ctx.temp_var("index")
    if ctx.scalars["reverse"]:
        first = 1
    else:
        first = sub(length(steps.list), size - 1) if steps.dynamic_axis is not None else (
            steps.numel - size + 1
        )
    out = ctx.output.list
    return (
        clear(out),
        set_var(position, first),
        repeat(size, (append(out, item(steps.list, var(position))), change_var(position, 1))),
    )


def _runtime_layout(tensor, dim: int):
    """``(outer, extent, inner)`` item counts before, along and after ``dim``.

    The count that holds a runtime axis is an expression the list's length
    gives; the others are integers.
    """
    shape = tensor.shape
    counts = [math.prod(shape[:dim]), shape[dim], math.prod(shape[dim + 1:])]
    axis = tensor.dynamic_axis
    if axis is not None:
        part = 0 if axis < dim else 1 if axis == dim else 2
        others = math.prod(count for index, count in enumerate(counts) if index != part)
        counts[part] = length(tensor.list) if others == 1 else div(length(tensor.list), others)
    return tuple(counts)


def _emit_runtime_length(ctx: KernelContext):
    """The number of items along a runtime axis."""
    value = ctx.inputs["value"]
    _, extent, _ = _runtime_layout(value, int(ctx.scalars["dim"]) % len(value.shape))
    out = ctx.output.list
    return (clear(out), append(out, extent))


def _emit_runtime_reduce(ctx: KernelContext):
    """Reduce along a runtime axis, one output per position around it."""
    value = ctx.inputs["value"]
    dim = int(ctx.scalars["dim"]) % len(value.shape)
    kind = ctx.scalars["kind"]
    outer, extent, inner = _runtime_layout(value, dim)
    count, base, column, position, total = (
        ctx.temp_var(f"reduce {name}") for name in ("count", "base", "column", "position", "total")
    )
    current = item(value.list, var(position))
    if kind in {"amax", "amin"}:
        start = (set_var(total, current), change_var(position, inner))
        steps = sub(var(count), 1)
        better = gt(current, var(total)) if kind == "amax" else lt(current, var(total))
        combine = if_(better, (set_var(total, current),))
    elif kind == "prod":
        start, steps = (set_var(total, 1),), var(count)
        combine = set_var(total, mul(var(total), current))
    else:
        start, steps = (set_var(total, 0),), var(count)
        combine = change_var(total, current)
    result = div(var(total), var(count)) if kind == "mean" else var(total)
    out = ctx.output.list
    return (
        clear(out),
        set_var(count, extent),
        set_var(base, 0),
        repeat(outer, (
            set_var(column, 1),
            repeat(inner, (
                set_var(position, add(var(base), var(column))),
                *start,
                repeat(steps, (combine, change_var(position, inner))),
                append(out, result),
                change_var(column, 1),
            )),
            change_var(base, mul(var(count), inner)),
        )),
    )


def _emit_runtime_softmax(ctx: KernelContext):
    """Softmax along a runtime axis: maximum, sum of exponentials, then scale."""
    value = ctx.inputs["value"]
    dim = int(ctx.scalars["dim"]) % len(value.shape)
    outer, extent, inner = _runtime_layout(value, dim)
    count, base, column, start, position, peak, total = (
        ctx.temp_var(f"softmax {name}")
        for name in ("count", "base", "column", "start", "position", "peak", "total")
    )
    current = item(value.list, var(position))
    shifted = sub(current, var(peak))
    if ctx.scalars["log"]:
        result = sub(shifted, mathop("ln", var(total)))
    else:
        result = div(mathop("e ^", shifted), var(total))
    out = ctx.output.list
    # Positions along the axis are ``inner`` apart, so with ``inner > 1`` the
    # output is filled first and written in place.
    write = append(out, result) if inner == 1 else replace(out, var(position), result)
    prefill = () if inner == 1 else (repeat(length(value.list), (append(out, 0),)),)
    return (
        clear(out),
        *prefill,
        set_var(count, extent),
        set_var(base, 0),
        repeat(outer, (
            set_var(column, 1),
            repeat(inner, (
                set_var(start, add(var(base), var(column))),
                set_var(position, var(start)),
                set_var(peak, current),
                repeat(sub(var(count), 1), (
                    change_var(position, inner),
                    if_(gt(current, var(peak)), (set_var(peak, current),)),
                )),
                set_var(total, 0),
                set_var(position, var(start)),
                repeat(var(count), (
                    change_var(total, mathop("e ^", shifted)),
                    change_var(position, inner),
                )),
                set_var(position, var(start)),
                repeat(var(count), (write, change_var(position, inner))),
                change_var(column, 1),
            )),
            change_var(base, mul(var(count), inner)),
        )),
    )


def _matrix(tensor, side: str):
    """``(batch, rows, columns, runtime)`` of a matmul operand.

    A vector is a row on the left and a column on the right. ``runtime`` is
    ``rows`` or ``columns`` when that dimension holds the runtime extent.
    """
    shape = tensor.shape
    if len(shape) == 1:
        shape = (1, shape[0]) if side == "left" else (shape[0], 1)
        axis = None if tensor.dynamic_axis is None else (1 if side == "left" else 0)
    else:
        axis = None if tensor.dynamic_axis is None else tensor.dynamic_axis - (len(shape) - 2)
    batch = math.prod(shape[:-2])
    runtime = {0: "rows", 1: "columns"}.get(axis) if axis is not None else None
    if axis is not None and runtime is None:
        raise ValueError("a runtime matmul extent must be a matrix dimension")
    return batch, shape[-2], shape[-1], runtime


def _emit_runtime_matmul(ctx: KernelContext):
    """``left @ right`` whose contraction, or right's columns, have a runtime extent."""
    left, right = ctx.inputs["left"], ctx.inputs["right"]
    left_batch, rows, inner, left_runtime = _matrix(left, "left")
    right_batch, _, columns, right_runtime = _matrix(right, "right")
    batch = max(left_batch, right_batch)
    if {left_batch, right_batch} - {1, batch}:
        raise ValueError("runtime matmul operands need the same batch dimensions")
    size, width, left_base, right_base, row, column, left_position, right_position, total = (
        ctx.temp_var(f"matmul {name}") for name in (
            "inner", "columns", "left base", "right base", "row", "column",
            "left position", "right position", "total",
        )
    )
    contraction = inner
    if left_runtime == "columns":
        contraction = div(length(left.list), left_batch * rows) if left_batch * rows != 1 else (
            length(left.list)
        )
    elif left_runtime is not None:
        raise ValueError("the left operand's rows can't have a runtime extent here")
    if right_runtime == "columns":
        columns = div(length(right.list), mul(right_batch, var(size)) if right_batch != 1 else var(
            size,
        ))
    out = ctx.output.list
    return (
        clear(out),
        set_var(size, contraction),
        set_var(width, columns),
        set_var(left_base, 0),
        set_var(right_base, 0),
        repeat(batch, (
            set_var(row, 0),
            repeat(rows, (
                set_var(column, 1),
                repeat(var(width), (
                    set_var(total, 0),
                    set_var(left_position, add(add(var(left_base), var(row)), 1)),
                    set_var(right_position, add(var(right_base), var(column))),
                    repeat(var(size), (
                        change_var(total, mul(
                            item(left.list, var(left_position)),
                            item(right.list, var(right_position)),
                        )),
                        change_var(left_position, 1),
                        change_var(right_position, var(width)),
                    )),
                    append(out, var(total)),
                    change_var(column, 1),
                )),
                change_var(row, var(size)),
            )),
            *((change_var(left_base, mul(rows, var(size))),) if left_batch > 1 else ()),
            *((change_var(right_base, mul(var(size), var(width))),) if right_batch > 1 else ()),
        )),
    )


def _emit_runtime_cat(ctx: KernelContext):
    """Join two tensors along an axis where either has a runtime extent."""
    first, second = ctx.inputs["first"], ctx.inputs["second"]
    dim = int(ctx.scalars["dim"]) % len(first.shape)
    outer, first_extent, inner = _runtime_layout(first, dim)
    _, second_extent, _ = _runtime_layout(second, dim)
    first_width, second_width, first_position, second_position = (
        ctx.temp_var(f"cat {name}")
        for name in ("first width", "second width", "first position", "second position")
    )
    out = ctx.output.list
    return (
        clear(out),
        set_var(first_width, mul(first_extent, inner)),
        set_var(second_width, mul(second_extent, inner)),
        set_var(first_position, 1),
        set_var(second_position, 1),
        repeat(outer, (
            repeat(var(first_width), (
                append(out, item(first.list, var(first_position))),
                change_var(first_position, 1),
            )),
            repeat(var(second_width), (
                append(out, item(second.list, var(second_position))),
                change_var(second_position, 1),
            )),
        )),
    )


_END = 2**63 - 1


def _emit_runtime_slice(ctx: KernelContext):
    """Copy ``start:end`` along a runtime axis, counting negatives from its end."""
    value = ctx.inputs["value"]
    dim = int(ctx.scalars["dim"]) % len(value.shape)
    outer, extent, inner = _runtime_layout(value, dim)
    count, first, last, base, position = (
        ctx.temp_var(f"slice {name}") for name in ("count", "first", "last", "base", "position")
    )

    def bound(name, offset):
        if offset >= _END:
            return (set_var(name, var(count)),)
        place = add(var(count), offset) if offset < 0 else offset
        return (
            set_var(name, place),
            if_(lt(var(name), 0), (set_var(name, 0),)),
            if_(gt(var(name), var(count)), (set_var(name, var(count)),)),
        )

    out = ctx.output.list
    return (
        clear(out),
        set_var(count, extent),
        *bound(first, int(ctx.scalars["start"])),
        *bound(last, int(ctx.scalars["end"])),
        if_(lt(var(last), var(first)), (set_var(last, var(first)),)),
        set_var(base, 0),
        repeat(outer, (
            set_var(position, add(add(var(base), mul(var(first), inner)), 1)),
            repeat(mul(sub(var(last), var(first)), inner), (
                append(out, item(value.list, var(position))),
                change_var(position, 1),
            )),
            change_var(base, mul(var(count), inner)),
        )),
    )


BUILTIN_KERNELS = (
    Kernel(
        "cattorch.runtime_length.default", _emit_runtime_length, name="runtime_length",
    ),
    Kernel(
        "cattorch.runtime_reduce.default", _emit_runtime_reduce, name="runtime_reduce",
    ),
    Kernel(
        "cattorch.runtime_softmax.default", _emit_runtime_softmax, name="runtime_softmax",
    ),
    Kernel(
        "cattorch.runtime_matmul.default", _emit_runtime_matmul, name="runtime_matmul",
    ),
    Kernel("cattorch.runtime_cat.default", _emit_runtime_cat, name="runtime_cat"),
    Kernel("cattorch.runtime_slice.default", _emit_runtime_slice, name="runtime_slice"),
    Kernel(
        "cattorch.recurrent_scan.default", _emit_recurrent_scan, name="recurrent_scan",
    ),
    Kernel(
        "cattorch.recurrent_final.default", _emit_recurrent_final, name="recurrent_final",
    ),
    Kernel(
        "aten.gather.default", _emit_gather, name="gather_runtime",
        validator=_validate_runtime_index,
    ),
    Kernel(
        "aten.scatter.src", partial(_emit_scatter, scalar=False), name="scatter_runtime",
        validator=_validate_scatter,
    ),
    Kernel(
        "aten.scatter.value", partial(_emit_scatter, scalar=True), name="scatter_runtime",
        validator=_validate_scatter,
    ),
    Kernel(
        "aten.scatter_add.default", partial(_emit_scatter_reduce, scalar=False, default="sum"),
        name="scatter_reduce", validator=_validate_runtime_index,
    ),
    Kernel(
        "aten.scatter_reduce.two", partial(_emit_scatter_reduce, scalar=False),
        name="scatter_reduce", validator=_validate_scatter_reduce,
    ),
    Kernel(
        "aten.scatter.reduce", partial(_emit_scatter_reduce, scalar=False),
        name="scatter_reduce", validator=_validate_scatter_reduce,
    ),
    Kernel(
        "aten.scatter.value_reduce", partial(_emit_scatter_reduce, scalar=True),
        name="scatter_reduce", validator=_validate_scatter_reduce,
    ),
    Kernel(
        "cattorch.rotary_embedding.default",
        _emit_rotary_embedding,
        name="rotary_embedding_fused",
        validator=_validate_rotary_embedding,
    ),
    Kernel(
        "cattorch.rotary_half.default",
        _emit_rotary_half,
        name="rotary_half_fused",
        validator=_validate_rotary_embedding,
    ),
    Kernel(
        "cattorch.topk_values.default",
        partial(_emit_topk, values=True),
        name="topk_generic",
    ),
    Kernel(
        "cattorch.topk_indices.default",
        partial(_emit_topk, values=False),
        name="topk_generic",
    ),
    Kernel(
        "cattorch.sequence_row.default",
        _emit_sequence_row,
        name="sequence_row",
        validator=_validate_sequence_row,
    ),
    Kernel(
        "cattorch.cached_attention.default",
        partial(_emit_cached_attention, fast=False),
        name="cached_attention",
        validator=_validate_cached_attention,
    ),
    Kernel(
        "cattorch.cached_attention.default",
        partial(_emit_cached_attention, fast=True),
        name="cached_attention_fast",
        validator=_validate_cached_attention,
        fast_flag="softmax",
    ),
)
