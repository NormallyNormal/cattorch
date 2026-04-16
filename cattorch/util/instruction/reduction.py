"""
Reduction instructions.
"""

import math

from cattorch.util.instruction.instruction import TemplateInstruction


class MeanDimInstruction(TemplateInstruction):
    aten_op = "aten.mean.dim"
    template_name = "mean"

    def prepare(self):
        input_shape = self.args[0].shape
        dims = self.args[1].value
        ndim = len(input_shape)
        dims = sorted(d if d >= 0 else ndim + d for d in dims)
        # For a single reduction dim d:
        #   num_inner = prod(shape[d+1:])  (stride between reduced elements)
        #   num_outer = prod(shape[:d])
        #   reduce_size = shape[d]
        # For multiple contiguous trailing dims, same logic generalises.
        d_first = dims[0]
        self.reduce_size = math.prod(input_shape[d] for d in dims)
        self.num_inner = math.prod(input_shape[d_first + len(dims):]) if d_first + len(dims) < ndim else 1
        self.num_outer = math.prod(input_shape[:d_first]) if d_first > 0 else 1
        self.skip = (self.reduce_size - 1) * self.num_inner

    def get_constants(self):
        return {
            101: self.reduce_size,
            102: self.num_inner,
            103: self.num_outer,
            104: self.skip,
        }
