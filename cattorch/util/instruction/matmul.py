import json
import math

from cattorch.templates.template import TEMPLATE_DIR
from cattorch.util.instruction.instruction import Instruction, ScratchInstruction
from cattorch.util.scratch.constant_replacer import ConstantReplacer

class MatMulInstruction(Instruction):
    aten_op = "aten.matmul.default"
    def prepare(self):
        assert len(self.args) == 2
        a, b = self.args[0], self.args[1]
        if len(b.shape) == 1:
            assert a.shape[-1] == b.shape[0]
        else:
            assert a.shape[-1] == b.shape[-2]
        self.scratch_instruction = ScratchInstruction.MATMUL

    def finalize(self):
        if len(self.args[1].shape) <= 2:
            return self.finalize_basic_matmul()
        else:
            return self.finalize_tensor_matmul()

    def finalize_basic_matmul(self):
        if len(self.args[0].shape) == 1:
            # [N] @ [N, M] -> [M]
            constant_replacer = ConstantReplacer({
                101: 1,
                102: self.args[1].shape[-1],
                103: self.args[0].shape[-1],
            })
        elif len(self.args[1].shape) == 1:
            # [..., K, N] @ [N] -> [K]
            constant_replacer = ConstantReplacer({
                101: math.prod(self.args[0].shape[:-1]),
                102: 1,
                103: self.args[0].shape[-1],
            })
        else:
            # [..., K, N] @ [N, M] -> [K, M]
            constant_replacer = ConstantReplacer({
                101: math.prod(self.args[0].shape[:-1]),
                102: self.args[1].shape[-1],
                103: self.args[0].shape[-1],
            })

        template_path = TEMPLATE_DIR / "matmul" / "template.json"
        with open(template_path, "r") as jsonfile:
            data = json.load(jsonfile)

        data = constant_replacer.apply(data)
        return data


    def finalize_tensor_matmul(self):
        pass