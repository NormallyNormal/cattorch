import math

from cattorch.util.instruction.instruction import TemplateInstruction


class GetItemInstruction(TemplateInstruction):
    template_name = "getitem"

    def prepare(self):
        # args[0] = source tensor
        # args[1] = chunk_size (value)
        # args[2] = num_rows (value)
        # args[3] = skip (value: row_stride - chunk_size)
        # args[4] = offset (value: chunk_index * chunk_size)
        pass

    def get_constants(self):
        return {
            101: self.args[1].value,
            102: self.args[2].value,
            103: self.args[3].value,
            104: self.args[4].value,
        }
