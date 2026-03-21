import json
import math

from cattorch.templates.template import TEMPLATE_DIR
from cattorch.util.instruction.instruction import Instruction, ScratchInstruction
from cattorch.util.scratch.constant_replacer import ConstantReplacer


class LeakyReLUInstruction(Instruction):
    aten_op = "aten.leaky_relu.default"

    def prepare(self):
        self.scratch_instruction = ScratchInstruction.NONE
        self.total = math.prod(self.args[0].shape)
        # negative_slope is passed as second arg; default 0.01
        if len(self.args) > 1:
            self.negative_slope = self.args[1].value
        else:
            self.negative_slope = 0.01

    def finalize(self):
        constant_replacer = ConstantReplacer({
            101: self.total,
            102: self.negative_slope,
        })

        template_path = TEMPLATE_DIR / "leaky_relu" / "template.json"
        with open(template_path, "r") as jsonfile:
            data = json.load(jsonfile)

        data = constant_replacer.apply(data)
        return data
