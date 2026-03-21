import json
import math

from cattorch.templates.template import TEMPLATE_DIR
from cattorch.util.instruction.instruction import Instruction, ScratchInstruction
from cattorch.util.scratch.constant_replacer import ConstantReplacer


class ELUInstruction(Instruction):
    aten_op = "aten.elu.default"

    def prepare(self):
        self.scratch_instruction = ScratchInstruction.NONE
        self.total = math.prod(self.args[0].shape)
        # alpha is passed as second arg; default 1.0
        if len(self.args) > 1:
            self.alpha = self.args[1].value
        else:
            self.alpha = 1.0

    def finalize(self):
        constant_replacer = ConstantReplacer({
            101: self.total,
            102: self.alpha,
        })

        template_path = TEMPLATE_DIR / "elu" / "template.json"
        with open(template_path, "r") as jsonfile:
            data = json.load(jsonfile)

        data = constant_replacer.apply(data)
        return data
