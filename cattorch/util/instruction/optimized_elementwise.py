"""Elementwise and fused-activation production kernels."""

import math

import torch

from cattorch.kernel_api import Kernel, KernelContext, TensorInfo
from cattorch.util.scratch.dsl import (
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
    mul,
    for_each,
    repeat,
    set_var,
    sub,
    var,
)


from cattorch.util.instruction.optimized_common import (
    _broadcast_index,
    _static_unrolled_repeat,
    argument_at,
    quick_gelu_of,
    sigmoid_of,
    tanh_gelu_inner,
    tanh_of,
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


def _is_tensor(value) -> bool:
    return isinstance(value, TensorInfo)


_BINARY = {
    "aten.add.Tensor": add,
    "aten.sub.Tensor": sub,
    "aten.mul.Tensor": mul,
    "aten.div.Tensor": div,
}

# Binary operations whose result is chosen by a condition. Each maps to
# ``(condition(left, right), value_if_true, value_if_false)`` where the values
# are "left", "right", or a constant.
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


class _Elementwise:
    """One-based, shape-specialized elementwise lowering.

    Equal-shape binary tensors index both lists directly, without the modulo
    that broadcasting needs, and list values are cached in a variable only
    when an expression reuses them.
    """

    def __init__(self, ctx: KernelContext):
        self.ctx = ctx
        self.op = ctx.target
        self.args = [argument_at(ctx, position) for position in range(len(ctx.argument_names))]
        tensors = [argument for argument in self.args if _is_tensor(argument)]
        if self._is_nary():
            self.output_shape = torch.broadcast_shapes(*(tensor.shape for tensor in tensors))
        else:
            self.output_shape = self.args[0].shape
        self.size = math.prod(self.output_shape)
        dynamic = [tensor for tensor in tensors if tensor.dynamic_axis is not None]
        if dynamic:
            # The output repeats this many values for each one of the runtime
            # input, when broadcasting adds dimensions to it.
            per_value = self.size // max(math.prod(dynamic[0].shape), 1)
            self.size = length(dynamic[0].list)
            if per_value != 1:
                self.size = mul(self.size, per_value)

    def _is_nary(self):
        return self.op in _WHERE or (
            (self.op in _BINARY or self.op in _SELECT) and _is_tensor(self.args[1])
        )

    def operand(self, position, map_name):
        """Expression for argument ``position`` at the current output index."""
        argument = self.args[position]
        if not _is_tensor(argument):
            return argument
        if argument.dynamic_axis is not None and tuple(argument.shape) == tuple(self.output_shape):
            # One item may trace as one value; the list grows with the length.
            return item(argument.list, var("index"))
        index, values = _broadcast_index(argument.shape, self.output_shape, map_name)
        if values is not None:
            self.ctx.static_list(map_name, values)
        return item(argument.list, index)

    def emit(self, *, fast: bool = False):
        ctx, op = self.ctx, self.op
        if op in _BINARY:
            ctx.temp_var("index")
            expression = self._binary()
            out = ctx.output.list
            iteration = (append(out, expression), change_var("index", 1))
        else:
            for name in ("index", "value", "inner"):
                ctx.temp_var(name)
            if op in _SELECT:
                setup, output = self._select()
            elif op in _WHERE:
                setup, output = self._where()
            elif fast and op == "aten.gelu.default":
                setup, output = (), append(
                    ctx.output.list, quick_gelu_of(item(self.args[0].list, var("index"))),
                )
            else:
                setup, output = self._unary()
            out = ctx.output.list
            iteration = (*setup, output, change_var("index", 1))

        if not isinstance(self.size, int):
            traversal = (set_var("index", 1), repeat(self.size, iteration))
        elif op in {"aten.gelu.default", "aten.erf.default"}:
            # GELU's large reporter tree regressed when unrolled, but the
            # palette-hidden official loop still removes one index mutation.
            traversal = (for_each("index", self.size, iteration[:-1]),)
        else:
            lightweight = op in _BINARY or op in {
                "aten.neg.default", "aten.relu.default", "aten.leaky_relu.default",
            }
            traversal = (
                set_var("index", 1),
                *_static_unrolled_repeat(self.size, iteration, factor=8 if lightweight else 4),
            )
        return (clear(out), *traversal)

    def _select(self):
        condition, if_true, if_false = _SELECT[self.op]
        left = self.operand(0, "_left_broadcast_map")
        right = self.operand(1, "_right_broadcast_map")
        choices = {"left": left, "right": right}
        out = self.ctx.output.list
        return (), if_else(
            condition(left, right),
            (append(out, choices.get(if_true, if_true)),),
            (append(out, choices.get(if_false, if_false)),),
        )

    def _where(self):
        condition = self.operand(0, "_condition_broadcast_map")
        left = self.operand(1, "_left_broadcast_map")
        right = self.operand(2, "_right_broadcast_map")
        out = self.ctx.output.list
        return (), if_else(eq(condition, 0), (append(out, right),), (append(out, left),))

    def _binary(self):
        operation = _BINARY[self.op]
        left, right = self.args[0], self.args[1]
        if not _is_tensor(right):
            return operation(item(left.list, var("index")), right)
        if not _is_tensor(left):
            # A constant left operand, as in ``1 - x``.
            return operation(left, self.operand(1, "_right_broadcast_map"))
        return operation(
            self.operand(0, "_left_broadcast_map"), self.operand(1, "_right_broadcast_map"),
        )

    def _unary(self):
        op = self.op
        out = self.ctx.output.list
        direct = item(self.args[0].list, var("index"))
        cached = var("value")
        setup = ()
        output = None

        def choose(condition, if_true, if_false):
            return if_else(condition, (append(out, if_true),), (append(out, if_false),))

        if op == "aten.neg.default":
            result = mul(-1, direct)
        elif op == "aten.sigmoid.default":
            result = sigmoid_of(direct)
        elif op == "aten.rsqrt.default":
            result = div(1, mathop("sqrt", direct))
        elif op == "aten.tanh.default":
            result = tanh_of(direct)
        elif op == "aten.relu.default":
            output = choose(gt(direct, 0), direct, 0)
        elif op == "aten.silu.default":
            setup = (set_var("value", direct),)
            result = mul(cached, sigmoid_of(cached))
        elif op == "aten.gelu.default" and argument_at(self.ctx, 1, "none") != "tanh":
            # 0.5 x (1 + erf(x / sqrt(2))), using erfc for accuracy in the
            # negative tail: 1 + erf(z) is 2 - erfc(|z|) or erfc(|z|).
            setup = (
                set_var("value", direct),
                *_erfc_statements(mul(cached, 0.7071067811865476), "inner"),
            )
            output = choose(
                lt(cached, 0),
                mul(mul(0.5, cached), var("inner")),
                mul(cached, sub(1, mul(0.5, var("inner")))),
            )
        elif op == "aten.gelu.default":
            setup = (
                set_var("value", direct),
                set_var("inner", tanh_gelu_inner(cached)),
            )
            result = mul(mul(0.5, cached), add(1, tanh_of(var("inner"))))
        elif op == "aten.leaky_relu.default":
            output = choose(gt(direct, 0), direct, mul(direct, argument_at(self.ctx, 1, 0.01)))
        elif op == "aten.elu.default":
            # scale * x, or scale * alpha * (exp(input_scale * x) - 1); SELU
            # and CELU use the scale arguments.
            alpha, scale, input_scale = (
                argument_at(self.ctx, position, 1.0) for position in (1, 2, 3)
            )
            exponent = direct if input_scale == 1 else mul(input_scale, direct)
            output = choose(
                gt(direct, 0),
                direct if scale == 1 else mul(scale, direct),
                mul(alpha * scale, add(mathop("e ^", exponent), -1)),
            )
        elif op == "aten.pow.Tensor_Scalar":
            setup, result, output = self._power(direct, cached, choose)
        elif op in _MATHOP:
            result = mathop(_MATHOP[op], direct)
        elif op in {"aten.sin.default", "aten.cos.default", "aten.tan.default"}:
            # Scratch's trigonometry takes degrees.
            result = mathop(op.split(".")[1], mul(direct, 57.29577951308232))
        elif op in {"aten.asin.default", "aten.acos.default", "aten.atan.default"}:
            # ... and returns them.
            result = mul(mathop(op.split(".")[1], direct), 0.017453292519943295)
        elif op == "aten.exp2.default":
            result = mathop("e ^", mul(direct, 0.6931471805599453))
        elif op == "aten.log2.default":
            result = mul(mathop("ln", direct), 1.4426950408889634)
        elif op == "aten.erfc.default":
            setup = (set_var("value", direct), *_erfc_statements(cached, "inner"))
            output = choose(lt(cached, 0), sub(2, var("inner")), var("inner"))
        elif op == "aten.reciprocal.default":
            result = div(1, direct)
        elif op == "aten.erf.default":
            setup = (set_var("value", direct), *_erfc_statements(cached, "inner"))
            output = choose(lt(cached, 0), sub(var("inner"), 1), sub(1, var("inner")))
        elif op in {"aten.logical_not.default", "aten.bitwise_not.default"}:
            output = choose(eq(direct, 0), 1, 0)
        elif op in _CLAMP:
            low, high = self._clamp_bounds()
            setup = (set_var("value", direct),)
            if low is not None:
                setup += (if_(lt(cached, low), (set_var("value", low),)),)
            if high is not None:
                setup += (if_(gt(cached, high), (set_var("value", high),)),)
            result = cached
        elif op == "aten.to.dtype":
            # Only value-changing conversions reach the kernel; graph analysis
            # aliases conversions to floating point and same-dtype ones.
            if self.args[1] == torch.bool:
                output = choose(eq(direct, 0), 0, 1)
            else:
                # Integer conversion truncates toward zero like PyTorch.
                output = choose(lt(direct, 0), mathop("ceiling", direct), mathop("floor", direct))
        else:
            raise NotImplementedError(op)

        if output is None:
            output = append(out, result)
        return setup, output

    def _power(self, direct, cached, choose):
        exponent = self.args[1]
        if exponent == 0:
            return (), 1, None
        if exponent == 0.5:
            return (), mathop("sqrt", direct), None
        if exponent == -0.5:
            return (), div(1, mathop("sqrt", direct)), None
        if float(exponent).is_integer() and 1 <= abs(exponent) <= 4:
            result = cached
            for _ in range(int(abs(exponent)) - 1):
                result = mul(result, cached)
            if exponent < 0:
                result = div(1, result)
            return (set_var("value", direct),), result, None
        if float(exponent).is_integer() and abs(exponent) <= 64:
            # Square and multiply: exact for integer results, unlike e^(n ln x).
            magnitude = int(abs(exponent))
            setup = [set_var("value", direct), set_var("inner", cached)]
            for bit in bin(magnitude)[3:]:
                setup.append(set_var("inner", mul(var("inner"), var("inner"))))
                if bit == "1":
                    setup.append(set_var("inner", mul(var("inner"), cached)))
            result = var("inner") if exponent > 0 else div(1, var("inner"))
            return tuple(setup), result, None
        if float(exponent).is_integer():
            # |x| ^ n, negated for a negative base and an odd exponent.
            setup = (set_var("value", direct),)
            magnitude = mathop("e ^", mul(exponent, mathop("ln", mathop("abs", cached))))
            if int(exponent) % 2:
                return setup, None, choose(lt(cached, 0), mul(-1, magnitude), magnitude)
            return setup, magnitude, None
        # A negative base gives NaN through ln, as in PyTorch.
        return (), mathop("e ^", mul(exponent, mathop("ln", direct))), None

    def _clamp_bounds(self):
        values = self.args[1:]
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


def _emit_elementwise(ctx: KernelContext):
    return _Elementwise(ctx).emit()


def _emit_fast_elementwise(ctx: KernelContext):
    """Approximate nonlinearities that avoid expensive transcendental work."""
    return _Elementwise(ctx).emit(fast=True)


def elementwise_kernel(target: str) -> Kernel:
    """The exact elementwise kernel for ``target``."""
    return Kernel(target, _emit_elementwise, name="elementwise_optimized")


FAST_GELU_KERNEL = Kernel("aten.gelu.default", _emit_fast_elementwise, name="elementwise_optimized")


def _arithmetic_value(ctx: KernelContext, values, operand):
    kind, value = operand
    if kind == "input":
        return item(ctx.inputs[ctx.argument_names[value]].list, var("index"))
    if kind == "step":
        return values[value]
    return value


def _emit_arithmetic_expression(ctx: KernelContext):
    """Evaluate a same-shape arithmetic chain in one list traversal.

    ``ctx.options["steps"]`` lists ``(target, operands)`` in order; each
    operand is ``("input", position)``, ``("step", index)`` of an earlier
    step, or ``("constant", value)``. The last step is the result, of
    ``ctx.options["size"]`` values.
    """
    index = ctx.temp_var("index")
    values = []
    for target, operands in ctx.options["steps"]:
        resolved = [_arithmetic_value(ctx, values, operand) for operand in operands]
        values.append(
            mul(-1, resolved[0]) if target == "aten.neg.default"
            else _BINARY[target](*resolved)
        )
    out = ctx.output.list
    first = ctx.inputs[ctx.argument_names[0]]
    step = (append(out, values[-1]), change_var(index, 1))
    traversal = (
        (repeat(length(first.list), step),) if first.dynamic_axis is not None
        else _static_unrolled_repeat(ctx.options["size"], step, factor=8)
    )
    return (clear(out), set_var(index, 1), *traversal)


ARITHMETIC_EXPRESSION_KERNEL = Kernel(
    "cattorch.arithmetic_expression", _emit_arithmetic_expression,
    name="arithmetic_expression_fused",
)


def _emit_swiglu(ctx: KernelContext):
    """Fuse ``silu(gate) * value`` into one exact elementwise traversal."""
    gate_input, value = ctx.inputs["gate"], ctx.inputs["value"]
    index, gate = ctx.temp_var("index"), ctx.temp_var("gate")
    out = ctx.output.list
    iteration = (
        set_var(gate, item(gate_input.list, var(index))),
        append(out, mul(mul(var(gate), sigmoid_of(var(gate))), item(value.list, var(index)))),
        change_var(index, 1),
    )
    dynamic = gate_input.dynamic_axis is not None
    return (
        clear(out),
        set_var(index, 1),
        *(
            (repeat(length(gate_input.list), iteration),) if dynamic
            else _static_unrolled_repeat(gate_input.numel, iteration, factor=4)
        ),
    )


SWIGLU_KERNEL = Kernel(
    "cattorch.swiglu", _emit_swiglu, name="swiglu_optimized", arguments=("gate", "value"),
)


def _emit_paired_linear_swiglu(ctx: KernelContext):
    """Fuse two same-input Linear projections and their SwiGLU epilogue.

    The arguments are the input, then each shard's gate and value weights,
    then the gate and value biases. ``ctx.options["shard_rows"]`` gives each
    shard's rows; without it there is one shard.
    """
    names = ctx.argument_names
    source = ctx.inputs[names[0]]
    pairs = [
        (ctx.inputs[names[position]], ctx.inputs[names[position + 1]])
        for position in range(1, len(names) - 2, 2)
    ]
    gate_bias, value_bias = ctx.inputs.get(names[-2]), ctx.inputs.get(names[-1])
    for gate_weight, value_weight in pairs:
        if gate_weight.shape != value_weight.shape or len(gate_weight.shape) != 2:
            raise ValueError("paired SwiGLU weights must have the same matrix shape")
    shard_rows = tuple(ctx.options.get("shard_rows") or (pairs[0][0].shape[0],))
    if len(shard_rows) != len(pairs):
        raise ValueError("paired SwiGLU shards must cover every hidden row")
    input_shape = source.shape
    inner = input_shape[-1]
    rows = (
        div(length(source.list), inner) if source.dynamic_axis is not None
        else (math.prod(input_shape[:-1]) if len(input_shape) > 1 else 1)
    )
    (row_start, input_index, gate_offset, value_offset, output_index,
     input_value, gate_sum, value_sum) = (
        ctx.temp_var(name) for name in (
            "row start", "input index", "gate offset", "value offset",
            "output index", "input value", "gate sum", "value sum",
        )
    )
    out = ctx.output.list
    gate_initial = item(gate_bias.list, var(output_index)) if gate_bias else 0
    value_initial = item(value_bias.list, var(output_index)) if value_bias else 0
    gate = var(gate_sum)
    swiglu = mul(mul(gate, sigmoid_of(gate)), var(value_sum))
    projections = []
    for (gate_weight, value_weight), count in zip(pairs, shard_rows):
        step = (
            set_var(input_value, item(source.list, var(input_index))),
            change_var(gate_sum, mul(
                var(input_value), item(gate_weight.list, add(var(gate_offset), var(input_index))),
            )),
            change_var(value_sum, mul(
                var(input_value), item(value_weight.list, add(var(value_offset), var(input_index))),
            )),
            change_var(input_index, 1),
        )
        projections.extend((
            set_var(gate_offset, sub(1, var(row_start))),
            set_var(value_offset, sub(1, var(row_start))),
            repeat(count, (
                set_var(gate_sum, gate_initial),
                set_var(value_sum, value_initial),
                set_var(input_index, var(row_start)),
                *_static_unrolled_repeat(inner, step, factor=4),
                append(out, swiglu),
                change_var(gate_offset, inner),
                change_var(value_offset, inner),
                change_var(output_index, 1),
            )),
        ))
    return (
        clear(out),
        set_var(row_start, 1),
        repeat(rows, (
            set_var(output_index, 1),
            *projections,
            change_var(row_start, inner),
        )),
    )


PAIRED_LINEAR_SWIGLU_KERNEL = Kernel(
    "cattorch.paired_linear_swiglu", _emit_paired_linear_swiglu, name="swiglu_paired_linear",
)


__all__ = [
    "ARITHMETIC_EXPRESSION_KERNEL", "FAST_GELU_KERNEL", "PAIRED_LINEAR_SWIGLU_KERNEL",
    "SWIGLU_KERNEL", "elementwise_kernel",
]
