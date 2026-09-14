"""Elementwise and fused-activation production kernels."""

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
    for_each,
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

class OptimizedElementwiseInstruction(Instruction):
    """One-based, shape-specialized elementwise kernels.

    The old lowering repeatedly converted a zero-based loop index to Scratch's
    one-based list index. Equal-shape binary tensors also pay for a modulo that
    is only needed by the limited broadcasting fallback. This lowering avoids
    both costs and caches list values only when an expression reuses them.
    """

    _BINARY = {
        "aten.add.Tensor": add,
        "aten.sub.Tensor": sub,
        "aten.mul.Tensor": mul,
        "aten.div.Tensor": div,
    }

    def prepare(self):
        self.op = str(self.torch_name)
        if self.op in self._BINARY and self.args[1].value is None:
            self.output_shape = torch.broadcast_shapes(
                self.args[0].shape, self.args[1].shape,
            )
        else:
            self.output_shape = self.args[0].shape
        self.size = length("T1") if self.args[0].dynamic else math.prod(self.output_shape)
        self.broadcast_lists = {}

    def _binary_expr(self):
        operation = self._BINARY[self.op]
        if self.args[1].value is not None:
            left = item("T1", var("index"))
            return operation(left, self.args[1].value), ("T1", "T2")

        left_index, left_map = _broadcast_index(
            self.args[0].shape, self.output_shape, "_left_broadcast_map",
        )
        right_index, right_map = _broadcast_index(
            self.args[1].shape, self.output_shape, "_right_broadcast_map",
        )
        if left_map is not None:
            self.broadcast_lists["_left_broadcast_map"] = left_map
        if right_map is not None:
            self.broadcast_lists["_right_broadcast_map"] = right_map
        lists = ("T1", "T2", *self.broadcast_lists, "T3")
        return operation(
            item("T1", left_index), item("T2", right_index),
        ), lists

    def _unary(self):
        direct = item("T1", var("index"))
        cached = var("value")
        setup = ()
        output = None

        if self.op == "aten.neg.default":
            result = mul(-1, direct)
        elif self.op == "aten.sigmoid.default":
            result = div(1, add(1, mathop("e ^", mul(-1, direct))))
        elif self.op == "aten.rsqrt.default":
            result = div(1, mathop("sqrt", direct))
        elif self.op == "aten.tanh.default":
            result = sub(
                div(2, add(1, mathop("e ^", mul(-2, direct)))), 1,
            )
        elif self.op == "aten.relu.default":
            output = if_else(
                gt(direct, 0),
                (append("T2", direct),),
                (append("T2", 0),),
            )
            result = None
        elif self.op == "aten.silu.default":
            setup = (set_var("value", direct),)
            result = mul(cached, div(1, add(1, mathop("e ^", mul(-1, cached)))))
        elif self.op == "aten.gelu.default":
            setup = (
                set_var("value", direct),
                set_var(
                    "inner",
                    mul(
                        0.7978845608028654,
                        add(cached, mul(0.044715, mul(mul(cached, cached), cached))),
                    ),
                ),
            )
            tanh_inner = sub(
                div(2, add(1, mathop("e ^", mul(-2, var("inner"))))), 1,
            )
            result = mul(mul(0.5, cached), add(1, tanh_inner))
        elif self.op == "aten.leaky_relu.default":
            slope = self.args[1].value if len(self.args) > 1 else 0.01
            output = if_else(
                gt(direct, 0),
                (append("T2", direct),),
                (append("T2", mul(direct, slope)),),
            )
            result = None
        elif self.op == "aten.elu.default":
            alpha = self.args[1].value if len(self.args) > 1 else 1.0
            output = if_else(
                gt(direct, 0),
                (append("T2", direct),),
                (append("T2", mul(alpha, add(mathop("e ^", direct), -1))),),
            )
            result = None
        elif self.op == "aten.pow.Tensor_Scalar":
            exponent = self.args[1].value
            if exponent == 2:
                result = mul(direct, direct)
            else:
                result = mathop("e ^", mul(exponent, mathop("ln", direct)))
        else:
            raise NotImplementedError(self.op)

        if output is None:
            output = append("T2", result)
        return setup, output

    def finalize(self):
        if self.op in self._BINARY:
            expression, lists = self._binary_expr()
            output_name = lists[-1]
            variables = ("index",)
            iteration = (append(output_name, expression), change_var("index", 1))
        else:
            lists = ("T1", "T2")
            output_name = "T2"
            setup, output = self._unary()
            variables = ("index", "value", "inner")
            iteration = (*setup, output, change_var("index", 1))

        if self.args[0].dynamic:
            traversal = (
                set_var("index", 1),
                repeat(self.size, iteration),
            )
        elif self.op == "aten.gelu.default":
            # GELU's large reporter tree regressed when unrolled, but the
            # palette-hidden official loop still removes one index mutation.
            traversal = (for_each("index", self.size, iteration[:-1]),)
        else:
            lightweight = self.op in self._BINARY or self.op in {
                "aten.neg.default", "aten.relu.default", "aten.leaky_relu.default",
            }
            traversal = (
                set_var("index", 1),
                *_static_unrolled_repeat(
                    self.size,
                    iteration,
                    factor=8 if lightweight else 4,
                ),
            )

        return Program(
            "elementwise_optimized",
            variables=variables,
            lists=lists,
            list_values=self.broadcast_lists,
            body=(
                clear(output_name),
                *traversal,
            ),
        ).compile()


class FusedArithmeticInstruction(Instruction):
    """Evaluate a same-shape arithmetic expression in one list traversal."""

    def __init__(self, torch_name, output, *args, expression, size):
        self.expression = expression
        self.size = size
        super().__init__(torch_name, output, *args)

    def prepare(self):
        pass

    def finalize(self):
        destination = f"T{len(self.args) + 1}"
        lists = tuple(f"T{index}" for index in range(1, len(self.args) + 2))
        traversal = (
            (repeat(self.size, (
                append(destination, self.expression),
                change_var("index", 1),
            )),)
            if self.args[0].dynamic
            else _static_unrolled_repeat(self.size, (
                append(destination, self.expression),
                change_var("index", 1),
            ), factor=8)
        )
        return Program(
            "arithmetic_expression_fused",
            variables=("index",),
            lists=lists,
            body=(
                clear(destination),
                set_var("index", 1),
                *traversal,
            ),
        ).compile()


class FastElementwiseInstruction(OptimizedElementwiseInstruction):
    """Approximate nonlinearities that avoid expensive transcendental work."""

    def _unary(self):
        direct = item("T1", var("index"))
        if self.op == "aten.gelu.default":
            result = mul(
                direct,
                div(1, add(1, mathop("e ^", mul(-1.702, direct)))),
            )
            return (), append("T2", result)
        return super()._unary()


class OptimizedSwiGLUInstruction(Instruction):
    """Fuse ``silu(gate) * value`` into one exact elementwise traversal."""

    def prepare(self):
        self.size = length("T1") if self.args[0].dynamic else math.prod(self.args[0].shape)

    def finalize(self):
        gate = var("gate")
        result = mul(
            mul(gate, div(1, add(1, mathop("e ^", mul(-1, gate))))),
            item("T2", var("index")),
        )
        iteration = (
            set_var("gate", item("T1", var("index"))),
            append("T3", result),
            change_var("index", 1),
        )
        return Program(
            "swiglu_optimized",
            variables=("index", "gate"),
            lists=("T1", "T2", "T3"),
            body=(
                clear("T3"),
                set_var("index", 1),
                *(
                    (repeat(self.size, iteration),)
                    if self.args[0].dynamic
                    else _static_unrolled_repeat(self.size, iteration, factor=4)
                ),
            ),
        ).compile()


class PairedLinearSwiGLUInstruction(Instruction):
    """Fuse two same-input Linear projections and their SwiGLU epilogue."""

    def __init__(self, torch_name, output, *args, shard_rows=()):
        self.shard_rows = tuple(shard_rows)
        super().__init__(torch_name, output, *args)

    def prepare(self):
        input_shape = self.args[0].shape
        gate_shape = self.args[1].shape
        value_shape = self.args[3].shape
        if gate_shape != value_shape or len(gate_shape) != 2:
            raise ValueError("paired SwiGLU weights must have the same matrix shape")
        self.rows = (
            div(length("T1"), input_shape[-1])
            if self.args[0].dynamic
            else (math.prod(input_shape[:-1]) if len(input_shape) > 1 else 1)
        )
        self.inner = input_shape[-1]
        self.hidden = gate_shape[0]
        if not self.shard_rows:
            self.shard_rows = (self.hidden,)
        if sum(self.shard_rows) != self.hidden:
            raise ValueError("paired SwiGLU shards must cover every hidden row")
        shard_count = len(self.shard_rows)
        self.weight_lists = tuple(
            (f"T{2 + 2 * index}", f"T{3 + 2 * index}")
            for index in range(shard_count)
        )
        self.gate_bias_list = f"T{2 + 2 * shard_count}"
        self.value_bias_list = f"T{3 + 2 * shard_count}"
        self.output_list = f"T{4 + 2 * shard_count}"
        self.gate_bias = self.args[2].value is not None or bool(self.args[2].shape)
        self.value_bias = self.args[4].value is not None or bool(self.args[4].shape)

    def finalize(self):
        gate_initial = (
            item(self.gate_bias_list, var("output index"))
            if self.gate_bias else 0
        )
        value_initial = (
            item(self.value_bias_list, var("output index"))
            if self.value_bias else 0
        )
        gate = var("gate sum")
        swiglu = mul(
            mul(gate, div(1, add(1, mathop("e ^", mul(-1, gate))))),
            var("value sum"),
        )
        projections = []
        for (gate_weight, value_weight), shard_rows in zip(
            self.weight_lists, self.shard_rows,
        ):
            step = (
                set_var("input value", item("T1", var("input index"))),
                change_var(
                    "gate sum",
                    mul(
                        var("input value"),
                        item(
                            gate_weight,
                            add(var("gate offset"), var("input index")),
                        ),
                    ),
                ),
                change_var(
                    "value sum",
                    mul(
                        var("input value"),
                        item(
                            value_weight,
                            add(var("value offset"), var("input index")),
                        ),
                    ),
                ),
                change_var("input index", 1),
            )
            projections.extend((
                set_var("gate offset", sub(1, var("row start"))),
                set_var("value offset", sub(1, var("row start"))),
                repeat(shard_rows, (
                    set_var("gate sum", gate_initial),
                    set_var("value sum", value_initial),
                    set_var("input index", var("row start")),
                    *_static_unrolled_repeat(self.inner, step, factor=4),
                    append(self.output_list, swiglu),
                    change_var("gate offset", self.inner),
                    change_var("value offset", self.inner),
                    change_var("output index", 1),
                )),
            ))

        return Program(
            "swiglu_paired_linear",
            variables=(
                "row start", "input index", "gate offset", "value offset",
                "output index", "input value", "gate sum", "value sum",
            ),
            lists=tuple(
                f"T{index}" for index in range(1, 2 * len(self.shard_rows) + 5)
            ),
            body=(
                clear(self.output_list),
                set_var("row start", 1),
                repeat(self.rows, (
                    set_var("output index", 1),
                    *projections,
                    change_var("row start", self.inner),
                )),
            ),
        ).compile()
