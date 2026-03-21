import json
import math

import torch

from cattorch.templates.template import TEMPLATE_DIR
from cattorch.util.instruction.instruction import Instruction, ScratchInstruction
from cattorch.util.scratch.constant_replacer import ConstantReplacer


def _compute_index_map(input_shape: torch.Size, permutation: list[int]) -> list[int]:
    """Compute a flat index map for a transpose/permute operation.

    Returns a list where index_map[i] is the 1-based output position
    that input element i should be written to.
    """
    total = math.prod(input_shape)
    indices = torch.arange(total).reshape(input_shape)
    output_flat = indices.permute(permutation).flatten()

    # output_flat[out_pos] = in_pos
    # We need index_map[in_pos] = out_pos + 1 (1-based for Scratch)
    index_map = [0] * total
    for out_pos in range(total):
        in_pos = output_flat[out_pos].item()
        index_map[in_pos] = out_pos + 1

    return index_map


class TransposeInstruction(Instruction):
    aten_op = [
        "aten.numpy_T.default",
        "aten.transpose.int",
        "aten.permute.default",
    ]

    def prepare(self):
        self.scratch_instruction = ScratchInstruction.TRANSPOSE
        input_shape = self.args[0].shape
        ndim = len(input_shape)
        aten_op = str(self.torch_name)

        if aten_op == "aten.numpy_T.default":
            self.permutation = list(reversed(range(ndim)))
        elif aten_op == "aten.transpose.int":
            dim0 = self.args[1].value
            dim1 = self.args[2].value
            self.permutation = list(range(ndim))
            self.permutation[dim0], self.permutation[dim1] = (
                self.permutation[dim1],
                self.permutation[dim0],
            )
        elif aten_op == "aten.permute.default":
            self.permutation = list(self.args[1].value)

        self.index_map = _compute_index_map(input_shape, self.permutation)

    def finalize(self):
        total = math.prod(self.args[0].shape)

        constant_replacer = ConstantReplacer({
            101: total,
        })

        template_path = TEMPLATE_DIR / "transpose" / "template.json"
        with open(template_path, "r") as jsonfile:
            data = json.load(jsonfile)

        data = constant_replacer.apply(data)

        # Inject the precomputed index map into the _index_map list
        for entry in data["lists"].values():
            if entry[0] == "_index_map":
                entry[1] = self.index_map
                break

        return data
