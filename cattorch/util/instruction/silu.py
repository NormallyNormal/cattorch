import json
import math

from cattorch.templates.template import TEMPLATE_DIR
from cattorch.util.instruction.instruction import Instruction, ScratchInstruction
from cattorch.util.scratch.constant_replacer import ConstantReplacer


class SiLUInstruction(Instruction):
    aten_op = "aten.silu.default"

    def prepare(self):
        assert len(self.args) == 1
        self.scratch_instruction = ScratchInstruction.NONE

    def finalize(self):
        constant_replacer = ConstantReplacer({
            101: math.prod(self.args[0].shape),
        })

        template_path = TEMPLATE_DIR / "silu" / "template.json"
        with open(template_path, "r") as jsonfile:
            data = json.load(jsonfile)

        data = constant_replacer.apply(data)
        return data
