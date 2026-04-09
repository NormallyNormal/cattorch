import math

from cattorch.util.instruction.instruction import TemplateInstruction


class EmbeddingInstruction(TemplateInstruction):
    aten_op = "aten.embedding.default"
    template_name = "embedding"

    def get_constants(self):
        # args[0] = weight [vocab_size, embed_dim]
        # args[1] = indices [seq_len]
        return {
            101: math.prod(self.args[1].shape),
            102: self.args[0].shape[1],
        }
