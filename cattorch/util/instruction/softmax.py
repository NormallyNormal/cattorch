import json
import math

from cattorch.templates.template import TEMPLATE_DIR
from cattorch.util.instruction.instruction import Instruction, ScratchInstruction
from cattorch.util.scratch.constant_replacer import ConstantReplacer


def _compute_softmax_indices(shape, dim):
    """Compute the 0-based flat starting index for each softmax group.

    For shape [2,2,2] with dim=1: stride=2, block_size=4.
    Groups start at [0, 1, 4, 5] — one per unique combination of
    non-softmax-dim indices.
    """
    total = math.prod(shape)
    dim_size = shape[dim]
    stride = math.prod(shape[dim + 1:]) if dim + 1 < len(shape) else 1
    block_size = dim_size * stride

    indices = []
    for base in range(0, total, block_size):
        for offset in range(stride):
            indices.append(base + offset)
    return indices


class SoftmaxInstruction(Instruction):
    aten_op = "aten.softmax.int"

    def prepare(self):
        self.scratch_instruction = ScratchInstruction.NONE
        input_shape = self.args[0].shape
        dim = self.args[1].value

        # Normalize negative dim
        if dim < 0:
            dim = len(input_shape) + dim

        self.dim_size = input_shape[dim]
        self.stride = math.prod(input_shape[dim + 1:]) if dim + 1 < len(input_shape) else 1
        self.num_groups = math.prod(input_shape) // self.dim_size
        self.softmax_indices = _compute_softmax_indices(input_shape, dim)

    def finalize(self):
        constant_replacer = ConstantReplacer({
            101: self.dim_size,
            102: self.num_groups,
            103: self.stride,
        })

        template_path = TEMPLATE_DIR / "softmax" / "template.json"
        with open(template_path, "r") as jsonfile:
            data = json.load(jsonfile)

        data = constant_replacer.apply(data)

        # Inject softmax group start indices
        for entry in data["lists"].values():
            if entry[0] == "_softmax_indices":
                entry[1] = self.softmax_indices
                break

        return data
