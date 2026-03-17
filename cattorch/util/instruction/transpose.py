import torch

from cattorch.util.instruction.instruction import Instruction, ScratchInstruction


class TransposeInstruction(Instruction):
    def prepare(self):
        assert len(self.args) == 3
        self.scratch_instruction = ScratchInstruction.TRANSPOSE
        meta_a = torch.empty(self.args[0].shape, device='meta')
        self.constants = list(torch.transpose(meta_a, self.args[1].name, self.args[2].name).shape)
        return