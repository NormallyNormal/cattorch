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


BUILTIN_KERNELS = (
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
