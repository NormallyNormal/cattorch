"""Softmax, reduction, and normalization production kernels."""

import math

import torch

from cattorch.util.instruction.instruction import Instruction
from cattorch.util.scratch.dsl import (
    Program,
    add,
    append,
    change_var,
    clear,
    div,
    eq,
    gt,
    if_,
    if_else,
    item,
    length,
    lt,
    mathop,
    mod,
    mul,
    repeat,
    replace,
    set_var,
    sub,
    var,
)


from cattorch.util.instruction.optimized_common import (
    _broadcast_index,
    _static_unrolled_repeat,
    _unrolled_copy_statements,
    _unrolled_dot_statements,
    _unrolled_offset_dot_statements,
)

class OptimizedSoftmaxInstruction(Instruction):
    """Stable softmax specialized for contiguous output groups."""

    def prepare(self):
        shape = self.args[0].shape
        dim = self.args[1].value
        if dim < 0:
            dim += len(shape)
        self.dim_size = shape[dim]
        self.stride = math.prod(shape[dim + 1:]) if dim + 1 < len(shape) else 1
        self.total = length("T1") if self.args[0].dynamic else math.prod(shape)
        self.num_groups = (
            div(length("T1"), self.dim_size)
            if self.args[0].dynamic else self.total // self.dim_size
        )
        block_size = self.dim_size * self.stride
        # Contiguous groups advance a running index in Scratch, including
        # bounded runtime rows. Only fixed strided reductions need a table.
        self.group_starts = (
            [
                base + offset + 1
                for base in range(0, math.prod(shape), block_size)
                for offset in range(self.stride)
            ]
            if self.stride != 1 else []
        )

    def finalize(self):
        current = item("T1", var("index"))
        maximum_scan = ()
        if self.dim_size > 1:
            maximum_scan = (
                change_var("index", self.stride),
                *_static_unrolled_repeat(self.dim_size - 1, (
                    if_(gt(current, var("max")), (set_var("max", current),)),
                    change_var("index", self.stride),
                )),
            )

        prefix = (clear("T2"),)
        if self.stride != 1:
            prefix += (repeat(self.total, (append("T2", 0),)),)

        if self.stride == 1:
            start = var("group_start")
            group_setup = (set_var("group_start", 1),)
            group_finish = (change_var("group_start", self.dim_size),)
            variables = ("group_start", "index", "max", "sum")
            lists = ("T1", "T2")
            list_values = {}
        else:
            start = item("_softmax_starts", var("group"))
            group_setup = (set_var("group", 1),)
            group_finish = (change_var("group", 1),)
            variables = ("group", "index", "max", "sum")
            lists = ("T1", "T2", "_softmax_starts")
            list_values = {"_softmax_starts": self.group_starts}

        body = prefix + group_setup + (
            repeat(self.num_groups, (
                set_var("index", start),
                set_var("max", current),
                *maximum_scan,
                set_var("index", start),
                set_var("sum", 0),
                *_static_unrolled_repeat(self.dim_size, (
                    change_var("sum", mathop("e ^", sub(current, var("max")))),
                    change_var("index", self.stride),
                )),
                set_var("index", start),
                *_static_unrolled_repeat(self.dim_size, (
                    (
                        append(
                            "T2",
                            div(mathop("e ^", sub(current, var("max"))), var("sum")),
                        )
                        if self.stride == 1
                        else replace(
                            "T2", var("index"),
                            div(mathop("e ^", sub(current, var("max"))), var("sum")),
                        )
                    ),
                    change_var("index", self.stride),
                )),
                *group_finish,
            )),
        )
        return Program(
            "softmax_optimized",
            variables=variables,
            lists=lists,
            list_values=list_values,
            body=body,
        ).compile()


class FastSoftmaxInstruction(OptimizedSoftmaxInstruction):
    """Two-pass softmax that trades overflow protection for less Scratch work."""

    def finalize(self):
        current = item("T1", var("index"))
        stored = item("T2", var("index"))
        prefix = (clear("T2"),)
        if self.stride != 1:
            prefix += (repeat(self.total, (append("T2", 0),)),)

        if self.stride == 1:
            start = var("group_start")
            group_setup = (set_var("group_start", 1),)
            group_finish = (change_var("group_start", self.dim_size),)
            variables = ("group_start", "index", "sum", "exponential")
            lists = ("T1", "T2")
            list_values = {}
        else:
            start = item("_softmax_starts", var("group"))
            group_setup = (set_var("group", 1),)
            group_finish = (change_var("group", 1),)
            variables = ("group", "index", "sum", "exponential")
            lists = ("T1", "T2", "_softmax_starts")
            list_values = {"_softmax_starts": self.group_starts}

        store_exponential = (
            append("T2", var("exponential"))
            if self.stride == 1
            else replace("T2", var("index"), var("exponential"))
        )
        body = prefix + group_setup + (
            repeat(self.num_groups, (
                set_var("index", start),
                set_var("sum", 0),
                *_static_unrolled_repeat(self.dim_size, (
                    set_var("exponential", mathop("e ^", current)),
                    change_var("sum", var("exponential")),
                    store_exponential,
                    change_var("index", self.stride),
                )),
                set_var("index", start),
                *_static_unrolled_repeat(self.dim_size, (
                    replace("T2", var("index"), div(stored, var("sum"))),
                    change_var("index", self.stride),
                )),
                *group_finish,
            )),
        )
        return Program(
            "softmax_fast",
            variables=variables,
            lists=lists,
            list_values=list_values,
            body=body,
        ).compile()


class OptimizedMeanInstruction(Instruction):
    """Mean reduction with a one-based running start index."""

    def prepare(self):
        shape = self.args[0].shape
        dims = self.args[1].value
        ndim = len(shape)
        dims = sorted(dim if dim >= 0 else ndim + dim for dim in dims)
        first = dims[0]
        self.reduce_size = math.prod(shape[dim] for dim in dims)
        trailing = first + len(dims)
        self.inner = math.prod(shape[trailing:]) if trailing < ndim else 1
        self.outer = (
            div(length("T1"), self.reduce_size * self.inner)
            if self.args[0].dynamic
            else (math.prod(shape[:first]) if first else 1)
        )
        self.block_advance = (self.reduce_size - 1) * self.inner

    def finalize(self):
        return Program(
            "mean_optimized",
            variables=("start", "read", "sum"),
            lists=("T1", "T2"),
            body=(
                clear("T2"),
                set_var("start", 1),
                repeat(self.outer, (
                    repeat(self.inner, (
                        set_var("read", var("start")),
                        set_var("sum", 0),
                        *_static_unrolled_repeat(self.reduce_size, (
                            change_var("sum", item("T1", var("read"))),
                            change_var("read", self.inner),
                        )),
                        append("T2", div(var("sum"), self.reduce_size)),
                        change_var("start", 1),
                    )),
                    change_var("start", self.block_advance),
                )),
            ),
        ).compile()


class OptimizedLayerNormInstruction(Instruction):
    def prepare(self):
        self.norm_size = math.prod(self.args[1].value)
        self.groups = (
            div(length("T1"), self.norm_size)
            if self.args[0].dynamic
            else math.prod(self.args[0].shape) // self.norm_size
        )
        self.eps = self.args[4].value if len(self.args) > 4 else 1e-5
        self.has_weight = self.args[2].value is not None or bool(self.args[2].shape)
        self.has_bias = self.args[3].value is not None or bool(self.args[3].shape)

    def finalize(self):
        weight = item("T2", var("feature")) if self.has_weight else 1
        bias = item("T3", var("feature")) if self.has_bias else 0
        normalized = add(
            mul(
                mul(sub(item("T1", var("index")), var("mean")), var("inv_std")),
                weight,
            ),
            bias,
        )
        return Program(
            "layernorm_optimized",
            variables=(
                "index", "feature", "mean", "variance",
                "inv_std",
            ),
            lists=("T1", "T2", "T3", "T4"),
            body=(
                clear("T4"),
                set_var("index", 1),
                repeat(self.groups, (
                    set_var("mean", 0),
                    *_static_unrolled_repeat(self.norm_size, (
                        change_var("mean", item("T1", var("index"))),
                        change_var("index", 1),
                    )),
                    set_var("mean", div(var("mean"), self.norm_size)),
                    change_var("index", -self.norm_size),
                    set_var("variance", 0),
                    *_static_unrolled_repeat(self.norm_size, (
                        change_var(
                            "variance",
                            mul(
                                sub(item("T1", var("index")), var("mean")),
                                sub(item("T1", var("index")), var("mean")),
                            ),
                        ),
                        change_var("index", 1),
                    )),
                    set_var(
                        "inv_std",
                        div(
                            1,
                            mathop(
                                "sqrt",
                                add(div(var("variance"), self.norm_size), self.eps),
                            ),
                        ),
                    ),
                    change_var("index", -self.norm_size),
                    set_var("feature", 1),
                    *_static_unrolled_repeat(self.norm_size, (
                        append("T4", normalized),
                        change_var("index", 1),
                        change_var("feature", 1),
                    )),
                )),
            ),
        ).compile()


class FastLayerNormInstruction(OptimizedLayerNormInstruction):
    """One-pass statistics using E[x^2] - E[x]^2."""

    def finalize(self):
        weight = item("T2", var("feature")) if self.has_weight else 1
        bias = item("T3", var("feature")) if self.has_bias else 0
        current = item("T1", var("index"))
        normalized = add(
            mul(mul(sub(current, var("mean")), var("inv_std")), weight),
            bias,
        )
        return Program(
            "layernorm_fast",
            variables=("index", "feature", "mean", "mean_square", "variance", "inv_std"),
            lists=("T1", "T2", "T3", "T4"),
            body=(
                clear("T4"),
                set_var("index", 1),
                repeat(self.groups, (
                    set_var("mean", 0),
                    set_var("mean_square", 0),
                    *_static_unrolled_repeat(self.norm_size, (
                        change_var("mean", current),
                        change_var("mean_square", mul(current, current)),
                        change_var("index", 1),
                    )),
                    set_var("mean", div(var("mean"), self.norm_size)),
                    set_var(
                        "variance",
                        sub(div(var("mean_square"), self.norm_size), mul(var("mean"), var("mean"))),
                    ),
                    if_else(
                        gt(var("variance"), 0),
                        (set_var(
                            "inv_std",
                            div(1, mathop("sqrt", add(var("variance"), self.eps))),
                        ),),
                        (set_var("inv_std", div(1, mathop("sqrt", self.eps))),),
                    ),
                    change_var("index", -self.norm_size),
                    set_var("feature", 1),
                    *_static_unrolled_repeat(self.norm_size, (
                        append("T4", normalized),
                        change_var("index", 1),
                        change_var("feature", 1),
                    )),
                )),
            ),
        ).compile()


class OptimizedRMSNormInstruction(Instruction):
    def prepare(self):
        self.norm_size = math.prod(self.args[1].value)
        self.groups = (
            div(length("T1"), self.norm_size)
            if self.args[0].dynamic
            else math.prod(self.args[0].shape) // self.norm_size
        )
        self.eps = (
            self.args[3].value
            if len(self.args) > 3
            else torch.finfo(torch.float32).eps
        )
        self.has_weight = self.args[2].value is not None or bool(self.args[2].shape)

    def finalize(self):
        current = item("T1", var("index"))
        weight = item("T2", var("feature")) if self.has_weight else 1
        return Program(
            "rmsnorm_optimized",
            variables=("index", "feature", "sum", "rms"),
            lists=("T1", "T2", "T3"),
            body=(
                clear("T3"),
                set_var("index", 1),
                repeat(self.groups, (
                    set_var("sum", 0),
                    *_static_unrolled_repeat(self.norm_size, (
                        change_var("sum", mul(current, current)),
                        change_var("index", 1),
                    )),
                    set_var(
                        "rms",
                        mathop(
                            "sqrt",
                            add(div(var("sum"), self.norm_size), self.eps),
                        ),
                    ),
                    change_var("index", -self.norm_size),
                    set_var("feature", 1),
                    *_static_unrolled_repeat(self.norm_size, (
                        append("T3", mul(div(current, var("rms")), weight)),
                        change_var("index", 1),
                        change_var("feature", 1),
                    )),
                )),
            ),
        ).compile()


class OptimizedBatchNormInstruction(Instruction):
    def prepare(self):
        shape = self.args[0].shape
        self.batches = (
            div(length("T1"), math.prod(shape[1:]))
            if self.args[0].dynamic else shape[0]
        )
        self.channels = shape[1]
        self.spatial = math.prod(shape[2:])
        self.has_weight = self.args[1].value is not None or bool(self.args[1].shape)
        self.has_bias = self.args[2].value is not None or bool(self.args[2].shape)

    def transform_weights(self, static_lists):
        eps = self.args[7].value
        var_key = self.args[4].name
        raw_key = var_key[2:] if var_key.startswith("W_") else var_key
        if raw_key in static_lists:
            static_lists[raw_key] = torch.sqrt(static_lists[raw_key] + eps)

    def finalize(self):
        result = add(
            mul(
                div(
                    sub(item("T1", var("index")), var("mean")),
                    var("std"),
                ),
                var("scale"),
            ),
            var("offset"),
        )
        return Program(
            "batchnorm_optimized",
            variables=(
                "channel", "index",
                "mean", "std", "scale", "offset",
            ),
            lists=("T1", "T2", "T3", "T4", "T5", "T6"),
            body=(
                clear("T6"),
                set_var("index", 1),
                repeat(self.batches, (
                    set_var("channel", 1),
                    repeat(self.channels, (
                        set_var("mean", item("T4", var("channel"))),
                        set_var("std", item("T5", var("channel"))),
                        set_var(
                            "scale",
                            item("T2", var("channel")) if self.has_weight else 1,
                        ),
                        set_var(
                            "offset",
                            item("T3", var("channel")) if self.has_bias else 0,
                        ),
                        *_static_unrolled_repeat(self.spatial, (
                            append("T6", result),
                            change_var("index", 1),
                        )),
                        change_var("channel", 1),
                    )),
                )),
            ),
        ).compile()
