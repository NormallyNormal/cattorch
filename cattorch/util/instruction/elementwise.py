"""
Simple elementwise and scalar instructions.

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


class ScalarDivideInstruction(TemplateInstruction):
    aten_op = "aten.div.Tensor"
    template_name = "scalar_divide"

    def get_constants(self):
        return {101: math.prod(self.args[0].shape), 102: self.args[1].value}


# ── Tensor elementwise ───────────────────────────────────────────────────────


class TensorAddInstruction(TemplateInstruction):
    aten_op = "aten.add.Tensor"
    template_name = "tensor_add"

    def get_constants(self):
        return {
            101: math.prod(self.args[0].shape),
            102: math.prod(self.args[1].shape),
        }


# ── Layer norm ───────────────────────────────────────────────────────────────


class LayerNormInstruction(TemplateInstruction):
    aten_op = "aten.layer_norm.default"
    template_name = "layernorm"

    def prepare(self):
        normalized_shape = self.args[1].value
        self.norm_size = math.prod(normalized_shape)
        self.num_groups = math.prod(self.args[0].shape) // self.norm_size

    def get_constants(self):
        return {101: self.norm_size, 102: self.num_groups}


class RMSNormInstruction(TemplateInstruction):
    aten_op = "aten.rms_norm.default"
    template_name = "rms_norm"

    def prepare(self):
        normalized_shape = self.args[1].value
        self.norm_size = math.prod(normalized_shape)
        self.num_groups = math.prod(self.args[0].shape) // self.norm_size

    def get_constants(self):
        return {101: self.norm_size, 102: self.num_groups}
