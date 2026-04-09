import math

from cattorch.util.instruction.instruction import TemplateInstruction


class CatInstruction(TemplateInstruction):
    template_name = "cat"

    def prepare(self):
        pass

    def get_constants(self):
        # args[0] = first input tensor
        # args[1] = second input tensor
        # args[2] = num_outer (value)
        # args[3] = chunk_1 (value)
        # args[4] = chunk_2 (value)
        return {
            101: self.args[2].value,
            102: self.args[3].value,
            103: self.args[4].value,
        }
