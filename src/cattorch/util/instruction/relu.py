import json
import math

from src.cattorch.templates.template import TEMPLATE_DIR
from src.cattorch.util.instruction.instruction import Instruction, ScratchInstruction
from src.cattorch.util.scratch.constant_replacer import ConstantReplacer


class ReLUInstruction(Instruction):
    def prepare(self):
        assert len(self.args) == 1
        self.scratch_instruction = ScratchInstruction.RELU

    def finalize(self):
        constant_replacer = ConstantReplacer({
            101: math.prod(self.args[0].shape),
        })

        template_path = TEMPLATE_DIR / "relu" / "template.json"
        with open(template_path, "r") as jsonfile:
            data = json.load(jsonfile)

        data = constant_replacer.apply(data)
        return data