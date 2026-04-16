"""
Convolution instruction.

Handles both Conv1d and Conv2d.
Conv1d is treated as Conv2d with H_in=1, kH=1.
"""

from cattorch.util.instruction.instruction import TemplateInstruction


class ConvolutionInstruction(TemplateInstruction):
    aten_op = ["aten.conv1d.default", "aten.conv2d.default"]
    template_name = "conv2d"

    def prepare(self):
        input_shape = self.args[0].shape
        weight_shape = self.args[1].shape
        stride = self.args[3].value
        padding = self.args[4].value
        # dilation is not passed by torch.export for conv1d/conv2d, defaults to 1

        if len(input_shape) == 3:
            # Conv1d: (N, C_in, L) -> treat as (N, C_in, 1, L)
            self.N = input_shape[0]
            self.C_in = input_shape[1]
            self.H_in = 1
            self.W_in = input_shape[2]
            self.C_out = weight_shape[0]
            self.kH = 1
            self.kW = weight_shape[2]
            self.sH = 1
            self.sW = stride[0]
            self.pH = 0
            self.pW = padding[0]
        else:
            # Conv2d: (N, C_in, H_in, W_in)
            self.N = input_shape[0]
            self.C_in = input_shape[1]
            self.H_in = input_shape[2]
            self.W_in = input_shape[3]
            self.C_out = weight_shape[0]
            self.kH = weight_shape[2]
            self.kW = weight_shape[3]
            self.sH = stride[0]
            self.sW = stride[1]
            self.pH = padding[0]
            self.pW = padding[1]

        self.dH = 1
        self.dW = 1

        self.H_out = (self.H_in + 2 * self.pH - self.dH * (self.kH - 1) - 1) // self.sH + 1
        self.W_out = (self.W_in + 2 * self.pW - self.dW * (self.kW - 1) - 1) // self.sW + 1

        # bias is None when not present, otherwise it's a tensor arg with a shape
        self.has_bias = self.args[2].value is not None or len(self.args[2].shape) > 0

    def get_constants(self):
        return {
            101: self.N,
            102: self.C_out,
            103: self.H_out,
            104: self.W_out,
            105: self.C_in,
            106: self.kH,
            107: self.kW,
            108: self.sH,
            109: self.sW,
            110: self.pH,
            111: self.pW,
            112: self.dH,
            113: self.dW,
            114: self.H_in,
            115: self.W_in,
            116: 1 if self.has_bias else 0,
        }

    def get_lists(self):
        if not self.has_bias:
            return {"T3": [0]}
        return {}
