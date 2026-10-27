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

# Numerical Recipes ``erfcc`` coefficients: fractional error below 1.2e-7
# everywhere, so the negative GELU tail keeps its relative accuracy.
_ERFC_COEFFICIENTS = (
    -1.26551223, 1.00002368, 0.37409196, 0.09678418, -0.18628806,
    0.27886807, -1.13520398, 1.48851587, -0.82215223, 0.17087277,
)


def _erfc_statements(value, target: str):
    """Set variable ``target`` to ``erfc(|value|)``; ``value`` is evaluated twice."""
    magnitude = mathop("abs", value)
    t = var(target)
    polynomial = _ERFC_COEFFICIENTS[-1]
    for coefficient in reversed(_ERFC_COEFFICIENTS[:-1]):
        polynomial = add(coefficient, mul(t, polynomial))
    return (
        set_var(target, div(1, add(1, mul(0.5, magnitude)))),
        set_var(target, mul(t, mathop(
            "e ^", sub(polynomial, mul(magnitude, magnitude)),
        ))),
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

    # Binary operations whose result is chosen by a condition. Each maps to
    # ``(condition(left, right), value_if_true, value_if_false)`` where the
    # values are "left", "right", or a constant.
    _SELECT = {
        "aten.eq.Tensor": (eq, 1, 0), "aten.eq.Scalar": (eq, 1, 0),
        "aten.ne.Tensor": (eq, 0, 1), "aten.ne.Scalar": (eq, 0, 1),
        "aten.gt.Tensor": (gt, 1, 0), "aten.gt.Scalar": (gt, 1, 0),
        "aten.lt.Tensor": (lt, 1, 0), "aten.lt.Scalar": (lt, 1, 0),
        "aten.ge.Tensor": (lt, 0, 1), "aten.ge.Scalar": (lt, 0, 1),
        "aten.le.Tensor": (gt, 0, 1), "aten.le.Scalar": (gt, 0, 1),
        "aten.maximum.default": (gt, "left", "right"),
        "aten.minimum.default": (lt, "left", "right"),
    }
    _WHERE = frozenset({
        "aten.where.self", "aten.where.ScalarSelf", "aten.where.ScalarOther",
        "aten.where.Scalar",
    })

    def prepare(self):
        self.op = str(self.torch_name)
        # Tensor list slots are numbered over every non-constant argument,
        # including explicit ``_none`` placeholders, followed by the output.
        self.slots = {}
        for position, argument in enumerate(self.args):
            if not str(argument.name).startswith("C_"):
                self.slots[position] = f"T{len(self.slots) + 1}"
        self.output_list = f"T{len(self.slots) + 1}"
        tensor_shapes = [
            self.args[position].shape for position in self.slots
            if self.args[position].name != "_none"
        ]
        if self._is_nary():
            self.output_shape = torch.broadcast_shapes(*tensor_shapes)
        else:
            self.output_shape = self.args[0].shape
        self.size = length("T1") if self.args[0].dynamic else math.prod(self.output_shape)
        self.broadcast_lists = {}

    def _is_nary(self):
        return (
            self.op in self._WHERE
            or ((self.op in self._BINARY or self.op in self._SELECT)
                and self.args[1].value is None)
        )

    def _operand(self, position, map_name):
        """Expression for argument ``position`` at the current output index."""
        argument = self.args[position]
        if position not in self.slots:
            return argument.value
        index, values = _broadcast_index(argument.shape, self.output_shape, map_name)
        if values is not None:
            self.broadcast_lists[map_name] = values
        return item(self.slots[position], index)

    def _select(self):
        condition, if_true, if_false = self._SELECT[self.op]
        left = self._operand(0, "_left_broadcast_map")
        right = self._operand(1, "_right_broadcast_map")
        choices = {"left": left, "right": right}
        return (), if_else(
            condition(left, right),
            (append(self.output_list, choices.get(if_true, if_true)),),
            (append(self.output_list, choices.get(if_false, if_false)),),
        )

    def _where(self):
        condition = self._operand(0, "_condition_broadcast_map")
        left = self._operand(1, "_left_broadcast_map")
        right = self._operand(2, "_right_broadcast_map")
        return (), if_else(
            eq(condition, 0),
            (append(self.output_list, right),),
            (append(self.output_list, left),),
        )

    def _binary_expr(self):
        operation = self._BINARY[self.op]
        if self.args[1].value is not None:
            left = item("T1", var("index"))
            return operation(left, self.args[1].value), ("T1", "T2")
        if self.args[0].value is not None:
            # A constant left operand, as in ``1 - x``.
            right = self._operand(1, "_right_broadcast_map")
            lists = (*self.slots.values(), *self.broadcast_lists, self.output_list)
            return operation(self.args[0].value, right), lists

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
        elif self.op == "aten.gelu.default" and self._exact_gelu():
            # 0.5 x (1 + erf(x / sqrt(2))), using erfc for accuracy in the
            # negative tail: 1 + erf(z) is 2 - erfc(|z|) or erfc(|z|).
            setup = (
                set_var("value", direct),
                *_erfc_statements(mul(cached, 0.7071067811865476), "inner"),
            )
            output = if_else(
                lt(cached, 0),
                (append(self.output_list, mul(mul(0.5, cached), var("inner"))),),
                (append(
                    self.output_list,
                    mul(cached, sub(1, mul(0.5, var("inner")))),
                ),),
            )
            result = None
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
            # scale * x, or scale * alpha * (exp(input_scale * x) - 1); SELU
            # and CELU use the scale arguments.
            alpha, scale, input_scale = (
                self.args[index].value if len(self.args) > index else 1.0
                for index in (1, 2, 3)
            )
            exponent = direct if input_scale == 1 else mul(input_scale, direct)
            output = if_else(
                gt(direct, 0),
                (append("T2", direct if scale == 1 else mul(scale, direct)),),
                (append("T2", mul(alpha * scale, add(mathop("e ^", exponent), -1))),),
            )
            result = None
        elif self.op == "aten.pow.Tensor_Scalar":
            exponent = self.args[1].value
            if exponent == 0:
                result = 1
            elif exponent == 0.5:
                result = mathop("sqrt", direct)
            elif exponent == -0.5:
                result = div(1, mathop("sqrt", direct))
            elif float(exponent).is_integer() and 1 <= abs(exponent) <= 4:
                setup = (set_var("value", direct),)
                result = cached
                for _ in range(int(abs(exponent)) - 1):
                    result = mul(result, cached)
                if exponent < 0:
                    result = div(1, result)
            elif float(exponent).is_integer():
                # |x| ^ n, negated for a negative base and an odd exponent.
                setup = (set_var("value", direct),)
                magnitude = mathop("e ^", mul(exponent, mathop("ln", mathop("abs", cached))))
                if int(exponent) % 2:
                    output = if_else(
                        lt(cached, 0),
                        (append(self.output_list, mul(-1, magnitude)),),
                        (append(self.output_list, magnitude),),
                    )
                    result = None
                else:
                    result = magnitude
            else:
                # A negative base gives NaN through ln, as in PyTorch.
                result = mathop("e ^", mul(exponent, mathop("ln", direct)))
        elif self.op in self._MATHOP:
            result = mathop(self._MATHOP[self.op], direct)
        elif self.op in {"aten.sin.default", "aten.cos.default", "aten.tan.default"}:
            # Scratch's trigonometry takes degrees.
            name = self.op.split(".")[1]
            result = mathop(name, mul(direct, 57.29577951308232))
        elif self.op in {"aten.asin.default", "aten.acos.default", "aten.atan.default"}:
            # ... and returns them.
            name = self.op.split(".")[1]
            result = mul(mathop(name, direct), 0.017453292519943295)
        elif self.op == "aten.exp2.default":
            result = mathop("e ^", mul(direct, 0.6931471805599453))
        elif self.op == "aten.log2.default":
            result = mul(mathop("ln", direct), 1.4426950408889634)
        elif self.op == "aten.erfc.default":
            setup = (set_var("value", direct), *_erfc_statements(cached, "inner"))
            output = if_else(
                lt(cached, 0),
                (append(self.output_list, sub(2, var("inner"))),),
                (append(self.output_list, var("inner")),),
            )
            result = None
        elif self.op == "aten.reciprocal.default":
            result = div(1, direct)
        elif self.op == "aten.erf.default":
            setup = (set_var("value", direct), *_erfc_statements(cached, "inner"))
            output = if_else(
                lt(cached, 0),
                (append(self.output_list, sub(var("inner"), 1)),),
                (append(self.output_list, sub(1, var("inner"))),),
            )
            result = None
        elif self.op in {"aten.logical_not.default", "aten.bitwise_not.default"}:
            output = if_else(
                eq(direct, 0),
                (append(self.output_list, 1),),
                (append(self.output_list, 0),),
            )
            result = None
        elif self.op in self._CLAMP:
            low, high = self._clamp_bounds()
            setup = (set_var("value", direct),)
            if low is not None:
                setup += (if_(lt(cached, low), (set_var("value", low),)),)
            if high is not None:
                setup += (if_(gt(cached, high), (set_var("value", high),)),)
            result = cached
        elif self.op == "aten.to.dtype":
            output = self._cast(direct)
            result = None
        else:
            raise NotImplementedError(self.op)

        if output is None:
            output = append(self.output_list, result)
        return setup, output

    _MATHOP = {
        "aten.exp.default": "e ^",
        "aten.log.default": "ln",
        "aten.log10.default": "log",
        "aten.sqrt.default": "sqrt",
        "aten.abs.default": "abs",
        "aten.floor.default": "floor",
        "aten.ceil.default": "ceiling",
    }
    _CLAMP = frozenset({
        "aten.clamp.default", "aten.clamp_min.default", "aten.clamp_max.default",
        "aten.hardtanh.default", "aten.relu6.default",
    })

    def _clamp_bounds(self):
        values = [argument.value for argument in self.args[1:]]
        if self.op == "aten.clamp_min.default":
            return values[0], None
        if self.op == "aten.clamp_max.default":
            return None, values[0]
        if self.op == "aten.relu6.default":
            return 0, 6
        if self.op == "aten.hardtanh.default":
            values = values + [-1.0, 1.0][len(values):]
        values = values + [None, None][len(values):]
        return values[0], values[1]

    def _cast(self, direct):
        # Only value-changing conversions reach the kernel; graph analysis
        # aliases conversions to floating point and same-dtype conversions.
        target = self.args[1].value
        if target == torch.bool:
            return if_else(
                eq(direct, 0),
                (append(self.output_list, 0),),
                (append(self.output_list, 1),),
            )
        # Integer conversion truncates toward zero like PyTorch.
        return if_else(
            lt(direct, 0),
            (append(self.output_list, mathop("ceiling", direct)),),
            (append(self.output_list, mathop("floor", direct)),),
        )

    def _exact_gelu(self):
        approximate = self.args[1].value if len(self.args) > 1 else "none"
        return approximate != "tanh"

    def finalize(self):
        if self.op in self._BINARY:
            expression, lists = self._binary_expr()
            output_name = lists[-1]
            variables = ("index",)
            iteration = (append(output_name, expression), change_var("index", 1))
        else:
            if self.op in self._SELECT:
                setup, output = self._select()
            elif self.op in self._WHERE:
                setup, output = self._where()
            else:
                setup, output = self._unary()
            output_name = self.output_list
            lists = (*self.slots.values(), *self.broadcast_lists, output_name)
            variables = ("index", "value", "inner")
            iteration = (*setup, output, change_var("index", 1))

        if self.args[0].dynamic:
            traversal = (
                set_var("index", 1),
                repeat(self.size, iteration),
            )
        elif self.op in {"aten.gelu.default", "aten.erf.default"}:
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
