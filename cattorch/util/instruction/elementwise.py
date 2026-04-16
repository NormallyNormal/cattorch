"""
Elementwise and scalar instructions.

Each class just specifies the aten op, template name, and constant mapping.
The shared TemplateInstruction base handles loading the template and applying
the constants.
"""

import math

from cattorch.util.instruction.instruction import TemplateInstruction


# ── Activations ──────────────────────────────────────────────────────────────


class ReLUInstruction(TemplateInstruction):
    aten_op = "aten.relu.default"
    template_name = "relu"


class SigmoidInstruction(TemplateInstruction):
    aten_op = "aten.sigmoid.default"
    template_name = "sigmoid"


class TanhInstruction(TemplateInstruction):
    aten_op = "aten.tanh.default"
    template_name = "tanh"


class GELUInstruction(TemplateInstruction):
    aten_op = "aten.gelu.default"
    template_name = "gelu"


class SiLUInstruction(TemplateInstruction):
    aten_op = "aten.silu.default"
    template_name = "silu"


class LeakyReLUInstruction(TemplateInstruction):
    aten_op = "aten.leaky_relu.default"
    template_name = "leaky_relu"

    def get_constants(self):
        slope = self.args[1].value if len(self.args) > 1 else 0.01
        return {101: math.prod(self.args[0].shape), 102: slope}


class ELUInstruction(TemplateInstruction):
    aten_op = "aten.elu.default"
    template_name = "elu"

    def get_constants(self):
        alpha = self.args[1].value if len(self.args) > 1 else 1.0
        return {101: math.prod(self.args[0].shape), 102: alpha}


# ── Scalar ops ───────────────────────────────────────────────────────────────


class MultiplyInstruction(TemplateInstruction):
    aten_op = "aten.mul.Tensor"

    @property
    def template_name(self):
        if self.args[1].value is not None:
            return "scalar_multiply"
        return "tensor_multiply"

    def get_constants(self):
        if self.args[1].value is not None:
            return {101: math.prod(self.args[0].shape), 102: self.args[1].value}
        return {
            101: math.prod(self.args[0].shape),
            102: math.prod(self.args[1].shape),
        }


class NegateInstruction(TemplateInstruction):
    aten_op = "aten.neg.default"
    template_name = "tensor_negate"


class ScalarDivideInstruction(TemplateInstruction):
    aten_op = "aten.div.Tensor"
    template_name = "scalar_divide"

    def get_constants(self):
        return {101: math.prod(self.args[0].shape), 102: self.args[1].value}


class ScalarPowInstruction(TemplateInstruction):
    aten_op = "aten.pow.Tensor_Scalar"
    template_name = "scalar_pow"

    def get_constants(self):
        exponent = self.args[1].value
        return {101: exponent, 102: math.prod(self.args[0].shape)}


class RSqrtInstruction(TemplateInstruction):
    aten_op = "aten.rsqrt.default"
    template_name = "rsqrt"


# ── Tensor elementwise ───────────────────────────────────────────────────────


class TensorAddInstruction(TemplateInstruction):
    aten_op = "aten.add.Tensor"

    @property
    def template_name(self):
        if self.args[1].value is not None:
            return "scalar_add"
        return "tensor_add"

    def get_constants(self):
        if self.args[1].value is not None:
            return {101: math.prod(self.args[0].shape), 102: self.args[1].value}
        return {
            101: math.prod(self.args[0].shape),
            102: math.prod(self.args[1].shape),
        }


class TensorSubtractInstruction(TemplateInstruction):
    aten_op = "aten.sub.Tensor"

    @property
    def template_name(self):
        if self.args[1].value is not None:
            return "scalar_add"
        return "tensor_subtract"

    def get_constants(self):
        if self.args[1].value is not None:
            return {101: math.prod(self.args[0].shape), 102: -self.args[1].value}
        return {
            101: math.prod(self.args[0].shape),
            102: math.prod(self.args[1].shape),
        }
