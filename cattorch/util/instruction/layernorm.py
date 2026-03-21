import json
import math

from cattorch.templates.template import TEMPLATE_DIR
from cattorch.util.instruction.instruction import Instruction, ScratchInstruction
from cattorch.util.scratch.constant_replacer import ConstantReplacer


class LayerNormInstruction(Instruction):
    aten_op = "aten.layer_norm.default"

    def prepare(self):
        self.scratch_instruction = ScratchInstruction.NONE
        input_shape = self.args[0].shape
        normalized_shape = self.args[1].value  # e.g. [4] or [3, 4]

        self.norm_size = math.prod(normalized_shape)
        total = math.prod(input_shape)
        self.num_groups = total // self.norm_size

    def finalize(self):
        constant_replacer = ConstantReplacer({
            101: self.norm_size,
            102: self.num_groups,
        })

        template_path = TEMPLATE_DIR / "layernorm" / "template.json"
        with open(template_path, "r") as jsonfile:
            data = json.load(jsonfile)

        data = constant_replacer.apply(data)
        return data
