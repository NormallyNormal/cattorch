import json
import math

from cattorch.templates.template import TEMPLATE_DIR
from cattorch.util.instruction.instruction import Instruction, ScratchInstruction
from cattorch.util.scratch.constant_replacer import ConstantReplacer


class TensorAddInstruction(Instruction):
    aten_op = "aten.add.Tensor"
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