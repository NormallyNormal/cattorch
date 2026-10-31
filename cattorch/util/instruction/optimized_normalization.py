"""Softmax, reduction, and normalization production kernels."""

import math

import torch

from cattorch.kernel_api import Kernel, KernelContext
from cattorch.util.scratch.dsl import (
    add,
    append,
    change_var,
    clear,
    div,
    gt,
    if_,
    if_else,
    item,
    length,
    mathop,
    mul,
    repeat,
    replace,
    set_var,
    sub,
    var,
)
from cattorch.util.instruction.optimized_common import _static_unrolled_repeat


class _SoftmaxLayout:
    """Where each softmax group starts, along ``dim`` of ``source``.

    Contiguous groups advance a running index in Scratch, including bounded
    runtime rows. Only fixed strided reductions need a table of starts.
    """

    def __init__(self, ctx: KernelContext):
        source = ctx.inputs["self"]
        shape = source.shape
        dim = ctx.scalars["dim"] % len(shape)
        self.source = source
        self.dim_size = shape[dim]
        self.stride = math.prod(shape[dim + 1:]) if dim + 1 < len(shape) else 1
        dynamic = source.dynamic_axis is not None
        self.total = length(source.list) if dynamic else math.prod(shape)
        self.groups = (
            div(length(source.list), self.dim_size) if dynamic else self.total // self.dim_size
        )
        block = self.dim_size * self.stride
        self.starts = [
            base + offset + 1
            for base in range(0, math.prod(shape), block)
            for offset in range(self.stride)
        ] if self.stride != 1 else []
        # The group counter is the program's first variable.
        self.group = ctx.temp_var("group_start" if self.stride == 1 else "group")

    def group_loop(self, ctx: KernelContext, out: str, body):
        """``body(start)`` once per group, after the output is prepared."""
        prefix = (clear(out),)
        if self.stride != 1:
            prefix += (repeat(self.total, (append(out, 0),)),)
        group = self.group
        if self.stride == 1:
            start, setup, finish = var(group), (set_var(group, 1),), (change_var(group, self.dim_size),)
        else:
            starts = ctx.static_list("_softmax_starts", self.starts)
            start = item(starts, var(group))
            setup, finish = (set_var(group, 1),), (change_var(group, 1),)
        return (*prefix, *setup, repeat(self.groups, (*body(start), *finish)))

    def write(self, out: str, index: str, value):
        return append(out, value) if self.stride == 1 else replace(out, var(index), value)


def _emit_softmax(ctx: KernelContext, *, log: bool = False):
    """Stable softmax, or log-softmax ``x - max - ln(sum(exp(x - max)))``."""
    layout = _SoftmaxLayout(ctx)
    out = ctx.output.list
    index, maximum, total = ctx.temp_var("index"), ctx.temp_var("max"), ctx.temp_var("sum")
    current = item(layout.source.list, var(index))
    stride, size = layout.stride, layout.dim_size
    maximum_scan = () if size <= 1 else (
        change_var(index, stride),
        *_static_unrolled_repeat(size - 1, (
            if_(gt(current, var(maximum)), (set_var(maximum, current),)),
            change_var(index, stride),
        )),
    )
    if log:
        # ``sum`` becomes the per-group log normalizer.
        after_sum = (set_var(total, add(var(maximum), mathop("ln", var(total)))),)
        normalized = sub(current, var(total))
    else:
        after_sum = ()
        normalized = div(mathop("e ^", sub(current, var(maximum))), var(total))

    def body(start):
        return (
            set_var(index, start),
            set_var(maximum, current),
            *maximum_scan,
            set_var(index, start),
            set_var(total, 0),
            *_static_unrolled_repeat(size, (
                change_var(total, mathop("e ^", sub(current, var(maximum)))),
                change_var(index, stride),
            )),
            *after_sum,
            set_var(index, start),
            *_static_unrolled_repeat(size, (
                layout.write(out, index, normalized),
                change_var(index, stride),
            )),
        )
    return layout.group_loop(ctx, out, body)


def _emit_fast_softmax(ctx: KernelContext):
    """Two-pass softmax that trades overflow protection for less Scratch work."""
    layout = _SoftmaxLayout(ctx)
    out = ctx.output.list
    index, total, exponential = (
        ctx.temp_var("index"), ctx.temp_var("sum"), ctx.temp_var("exponential")
    )
    current = item(layout.source.list, var(index))
    stride, size = layout.stride, layout.dim_size

    def body(start):
        return (
            set_var(index, start),
            set_var(total, 0),
            *_static_unrolled_repeat(size, (
                set_var(exponential, mathop("e ^", current)),
                change_var(total, var(exponential)),
                layout.write(out, index, var(exponential)),
                change_var(index, stride),
            )),
            set_var(index, start),
            *_static_unrolled_repeat(size, (
                replace(out, var(index), div(item(out, var(index)), var(total))),
                change_var(index, stride),
            )),
        )
    return layout.group_loop(ctx, out, body)


def _softmax_kernels(emit, name, targets):
    return tuple(Kernel(target, emit, name=name) for target in targets)


SOFTMAX_KERNELS = _softmax_kernels(_emit_softmax, "softmax_optimized", ("aten.softmax.int",))
LOG_SOFTMAX_KERNELS = tuple(
    Kernel(target, lambda ctx: _emit_softmax(ctx, log=True), name="softmax_optimized")
    for target in ("aten.log_softmax.int", "aten._log_softmax.default")
)
FAST_SOFTMAX_KERNEL = Kernel("aten.softmax.int", _emit_fast_softmax, name="softmax_fast")


def _emit_mean(ctx: KernelContext):
    """Mean over consecutive dimensions with a one-based running start index."""
    source = ctx.inputs["self"]
    shape = source.shape
    dims = ctx.scalars["dim"]
    dims = sorted(dim % len(shape) for dim in ((dims,) if isinstance(dims, int) else dims))
    first = dims[0]
    reduce_size = math.prod(shape[dim] for dim in dims)
    trailing = first + len(dims)
    inner = math.prod(shape[trailing:]) if trailing < len(shape) else 1
    outer = (
        div(length(source.list), reduce_size * inner) if source.dynamic_axis is not None
        else (math.prod(shape[:first]) if first else 1)
    )
    start, read, total = ctx.temp_var("start"), ctx.temp_var("read"), ctx.temp_var("sum")
    out = ctx.output.list
    return (
        clear(out),
        set_var(start, 1),
        repeat(outer, (
            repeat(inner, (
                set_var(read, var(start)),
                set_var(total, 0),
                *_static_unrolled_repeat(reduce_size, (
                    change_var(total, item(source.list, var(read))),
                    change_var(read, inner),
                )),
                append(out, div(var(total), reduce_size)),
                change_var(start, 1),
            )),
            change_var(start, (reduce_size - 1) * inner),
        )),
    )


MEAN_KERNEL = Kernel("aten.mean.dim", _emit_mean, name="mean_optimized")


class _Normalized:
    """The rows of a layer or RMS norm, and its optional weight and bias."""

    def __init__(self, ctx: KernelContext):
        self.source = ctx.inputs["input"]
        self.size = math.prod(ctx.scalars["normalized_shape"])
        self.groups = (
            div(length(self.source.list), self.size) if self.source.dynamic_axis is not None
            else self.source.numel // self.size
        )
        # Trailing absent arguments, such as the bias of LayerNorm(bias=False),
        # are left out of the call; an explicit None takes no list either.
        self.weight = ctx.inputs.get("weight")
        self.bias = ctx.inputs.get("bias")


def _emit_layer_norm(ctx: KernelContext):
    norm = _Normalized(ctx)
    eps = ctx.scalars["eps"]
    index, feature, mean, variance, inv_std = (
        ctx.temp_var(name) for name in ("index", "feature", "mean", "variance", "inv_std")
    )
    current = item(norm.source.list, var(index))
    weight = item(norm.weight.list, var(feature)) if norm.weight else 1
    bias = item(norm.bias.list, var(feature)) if norm.bias else 0
    out = ctx.output.list
    size = norm.size
    return (
        clear(out),
        set_var(index, 1),
        repeat(norm.groups, (
            set_var(mean, 0),
            *_static_unrolled_repeat(size, (change_var(mean, current), change_var(index, 1))),
            set_var(mean, div(var(mean), size)),
            change_var(index, -size),
            set_var(variance, 0),
            *_static_unrolled_repeat(size, (
                change_var(variance, mul(sub(current, var(mean)), sub(current, var(mean)))),
                change_var(index, 1),
            )),
            set_var(inv_std, div(1, mathop("sqrt", add(div(var(variance), size), eps)))),
            change_var(index, -size),
            set_var(feature, 1),
            *_static_unrolled_repeat(size, (
                append(out, add(mul(mul(sub(current, var(mean)), var(inv_std)), weight), bias)),
                change_var(index, 1),
                change_var(feature, 1),
            )),
        )),
    )


def _emit_fast_layer_norm(ctx: KernelContext):
    """One-pass statistics using E[x^2] - E[x]^2."""
    norm = _Normalized(ctx)
    eps = ctx.scalars["eps"]
    index, feature, mean, mean_square, variance, inv_std = (
        ctx.temp_var(name)
        for name in ("index", "feature", "mean", "mean_square", "variance", "inv_std")
    )
    current = item(norm.source.list, var(index))
    weight = item(norm.weight.list, var(feature)) if norm.weight else 1
    bias = item(norm.bias.list, var(feature)) if norm.bias else 0
    out = ctx.output.list
    size = norm.size
    return (
        clear(out),
        set_var(index, 1),
        repeat(norm.groups, (
            set_var(mean, 0),
            set_var(mean_square, 0),
            *_static_unrolled_repeat(size, (
                change_var(mean, current),
                change_var(mean_square, mul(current, current)),
                change_var(index, 1),
            )),
            set_var(mean, div(var(mean), size)),
            set_var(variance, sub(div(var(mean_square), size), mul(var(mean), var(mean)))),
            if_else(
                gt(var(variance), 0),
                (set_var(inv_std, div(1, mathop("sqrt", add(var(variance), eps)))),),
                (set_var(inv_std, div(1, mathop("sqrt", eps))),),
            ),
            change_var(index, -size),
            set_var(feature, 1),
            *_static_unrolled_repeat(size, (
                append(out, add(mul(mul(sub(current, var(mean)), var(inv_std)), weight), bias)),
                change_var(index, 1),
                change_var(feature, 1),
            )),
        )),
    )


LAYER_NORM_KERNEL = Kernel("aten.layer_norm.default", _emit_layer_norm, name="layernorm_optimized")
FAST_LAYER_NORM_KERNEL = Kernel(
    "aten.layer_norm.default", _emit_fast_layer_norm, name="layernorm_fast",
)


def _emit_rms_norm(ctx: KernelContext):
    norm = _Normalized(ctx)
    # ``F.rms_norm(x, shape)`` passes neither a weight nor eps.
    eps = ctx.scalars.get("eps")
    eps = torch.finfo(torch.float32).eps if eps is None else eps
    index, feature, total, rms = (ctx.temp_var(name) for name in ("index", "feature", "sum", "rms"))
    current = item(norm.source.list, var(index))
    weight = item(norm.weight.list, var(feature)) if norm.weight else 1
    out = ctx.output.list
    size = norm.size
    return (
        clear(out),
        set_var(index, 1),
        repeat(norm.groups, (
            set_var(total, 0),
            *_static_unrolled_repeat(size, (
                change_var(total, mul(current, current)),
                change_var(index, 1),
            )),
            set_var(rms, mathop("sqrt", add(div(var(total), size), eps))),
            change_var(index, -size),
            set_var(feature, 1),
            *_static_unrolled_repeat(size, (
                append(out, mul(div(current, var(rms)), weight)),
                change_var(index, 1),
                change_var(feature, 1),
            )),
        )),
    )


RMS_NORM_KERNEL = Kernel("aten.rms_norm.default", _emit_rms_norm, name="rmsnorm_optimized")


def _emit_batch_norm(ctx: KernelContext):
    """Normalize each channel with its running statistics.

    The square root is taken in Scratch, once per channel. Rewriting the
    stored variance instead would change it for its other readers, such as a
    second call of the same BatchNorm.
    """
    source = ctx.inputs["input"]
    weight, bias = ctx.inputs.get("weight"), ctx.inputs.get("bias")
    mean_list, variance_list = ctx.inputs["running_mean"].list, ctx.inputs["running_var"].list
    shape = source.shape
    batches = (
        div(length(source.list), math.prod(shape[1:])) if source.dynamic_axis is not None
        else shape[0]
    )
    channel, index, mean, std, scale, offset = (
        ctx.temp_var(name) for name in ("channel", "index", "mean", "std", "scale", "offset")
    )
    out = ctx.output.list
    result = add(mul(div(sub(item(source.list, var(index)), var(mean)), var(std)), var(scale)), var(offset))
    return (
        clear(out),
        set_var(index, 1),
        repeat(batches, (
            set_var(channel, 1),
            repeat(shape[1], (
                set_var(mean, item(mean_list, var(channel))),
                set_var(std, mathop("sqrt", add(item(variance_list, var(channel)), ctx.scalars["eps"]))),
                set_var(scale, item(weight.list, var(channel)) if weight else 1),
                set_var(offset, item(bias.list, var(channel)) if bias else 0),
                *_static_unrolled_repeat(math.prod(shape[2:]), (
                    append(out, result),
                    change_var(index, 1),
                )),
                change_var(channel, 1),
            )),
        )),
    )


BATCH_NORM_KERNEL = Kernel("aten.batch_norm.default", _emit_batch_norm, name="batchnorm_optimized")


__all__ = [
    "BATCH_NORM_KERNEL", "FAST_LAYER_NORM_KERNEL", "FAST_SOFTMAX_KERNEL", "LAYER_NORM_KERNEL",
    "LOG_SOFTMAX_KERNELS", "MEAN_KERNEL", "RMS_NORM_KERNEL", "SOFTMAX_KERNELS",
]
