import json
import math

from src.cattorch.templates.template import TEMPLATE_DIR
from src.cattorch.util.instruction.instruction import Instruction, ScratchInstruction
from src.cattorch.util.scratch.constant_replacer import ConstantReplacer


class TensorAddInstruction(Instruction):
    def prepare(self):
        assert len(self.args) == 2
        assert self.args[0].shape[-1] == self.args[1].shape[-1]
        self.scratch_instruction = ScratchInstruction.TENSOR_ADD

    def finalize(self):
        constant_replacer = ConstantReplacer({
            101: math.prod(self.args[0].shape),
            102: math.prod(self.args[1].shape),
        })

        template_path = TEMPLATE_DIR / "tensor_add" / "template.json"
        with open(template_path, "r") as jsonfile:
            data = json.load(jsonfile)

        data = constant_replacer.apply(data)
        return data