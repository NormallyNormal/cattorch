import json
import math

from cattorch.templates.template import TEMPLATE_DIR
from cattorch.util.instruction.instruction import Instruction
from cattorch.util.scratch.constant_replacer import ConstantReplacer


class MatMulInstruction(Instruction):
    aten_op = ["aten.matmul.default", "aten.mm.default", "aten.bmm.default"]

    def prepare(self):
        assert len(self.args) == 2
        a, b = self.args[0], self.args[1]
        if len(b.shape) == 1:
            assert a.shape[-1] == b.shape[0]
        else:
            assert a.shape[-1] == b.shape[-2]

    def finalize(self):
        if len(self.args[1].shape) <= 2:
            return self._finalize_basic()
        else:
            return self._finalize_tensor()

    def _finalize_basic(self):
        if len(self.args[0].shape) == 1:
            # [N] @ [N, M] -> [M]
            constants = {
                101: 1,
                102: self.args[1].shape[-1],
                103: self.args[0].shape[-1],
                104: 1,
            }
        elif len(self.args[1].shape) == 1:
            # [..., K, N] @ [N] -> [K]
            constants = {
                101: math.prod(self.args[0].shape[:-1]),
                102: 1,
                103: self.args[0].shape[-1],
                104: 1,
            }
        else:
            # [..., K, N] @ [N, M] -> [..., K, M]
            constants = {
                101: math.prod(self.args[0].shape[:-1]),
                102: self.args[1].shape[-1],
                103: self.args[0].shape[-1],
                104: 1,
            }

        template_path = TEMPLATE_DIR / "matmul" / "template_simple.json"
        with open(template_path) as f:
            data = json.load(f)
        return ConstantReplacer(constants).apply(data)

    def _finalize_tensor(self):
        # [..., K, N] @ [..., N, M] -> [..., K, M]
        constants = {
            101: self.args[0].shape[-2],
            102: self.args[1].shape[-1],
            103: self.args[0].shape[-1],
            104: math.prod(self.args[1].shape[:-2]),
        }

        template_path = TEMPLATE_DIR / "matmul" / "template_tensor.json"
        with open(template_path) as f:
            data = json.load(f)
        return ConstantReplacer(constants).apply(data)
