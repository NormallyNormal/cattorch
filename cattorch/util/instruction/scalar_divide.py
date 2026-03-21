import json
import math

from cattorch.templates.template import TEMPLATE_DIR
from cattorch.util.instruction.instruction import Instruction, ScratchInstruction
from cattorch.util.scratch.constant_replacer import ConstantReplacer


class ScalarDivideInstruction(Instruction):
    aten_op = "aten.div.Tensor"

    def prepare(self):
        self.scratch_instruction = ScratchInstruction.NONE
        self.total = math.prod(self.args[0].shape)
        self.scalar = self.args[1].value

    def finalize(self):
        constant_replacer = ConstantReplacer({
            101: self.total,
            102: self.scalar,
        })

        template_path = TEMPLATE_DIR / "scalar_divide" / "template.json"
        with open(template_path, "r") as jsonfile:
            data = json.load(jsonfile)

        data = constant_replacer.apply(data)
        return data
