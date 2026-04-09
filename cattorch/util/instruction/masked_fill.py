import math

from cattorch.util.instruction.instruction import TemplateInstruction


class MaskedFillInstruction(TemplateInstruction):
    aten_op = "aten.masked_fill.Scalar"
    template_name = "masked_fill"

    def get_constants(self):
        return {
            101: math.prod(self.args[0].shape),
            102: self.args[2].value,
        }
