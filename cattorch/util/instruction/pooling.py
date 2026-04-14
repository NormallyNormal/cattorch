"""
Pooling instructions.

Handles MaxPool1d/2d, AvgPool1d/2d, and AdaptiveAvgPool2d.
1d variants are treated as 2d with H_in=1, kH=1.
"""

from cattorch.util.instruction.instruction import TemplateInstruction


class MaxPool2dInstruction(TemplateInstruction):
    aten_op = ["aten.max_pool2d.default", "aten.max_pool1d.default"]
    template_name = "max_pool2d"

    def prepare(self):
        input_shape = self.args[0].shape

        if len(input_shape) == 3:
            # Pool1d: (N, C, L)
            self.N = input_shape[0]
            self.C = input_shape[1]
            self.H_in = 1
            self.W_in = input_shape[2]
            kernel_size = self.args[1].value
            stride = self.args[2].value
            padding = self.args[3].value if len(self.args) > 3 else [0]
            self.kH = 1
            self.kW = kernel_size[0]
            self.sH = 1
            self.sW = stride[0]
            self.pH = 0
            self.pW = padding[0]
        else:
            # Pool2d: (N, C, H_in, W_in)
            self.N = input_shape[0]
            self.C = input_shape[1]
            self.H_in = input_shape[2]
            self.W_in = input_shape[3]
            kernel_size = self.args[1].value
            stride = self.args[2].value
            padding = self.args[3].value if len(self.args) > 3 else [0, 0]
            self.kH = kernel_size[0]
            self.kW = kernel_size[1]
            self.sH = stride[0]
            self.sW = stride[1]
            self.pH = padding[0]
            self.pW = padding[1]

        self.H_out = (self.H_in + 2 * self.pH - self.kH) // self.sH + 1
        self.W_out = (self.W_in + 2 * self.pW - self.kW) // self.sW + 1

    def get_constants(self):
        return {
            101: self.N,
            102: self.C,
            103: self.H_out,
            104: self.W_out,
            105: self.kH,
            106: self.kW,
            107: self.sH,
            108: self.sW,
            109: self.pH,
            110: self.pW,
            111: self.H_in,
            112: self.W_in,
        }


class AvgPool2dInstruction(TemplateInstruction):
    aten_op = ["aten.avg_pool2d.default", "aten.avg_pool1d.default"]
    template_name = "avg_pool2d"

    def prepare(self):
        input_shape = self.args[0].shape

        if len(input_shape) == 3:
            self.N = input_shape[0]
            self.C = input_shape[1]
            self.H_in = 1
            self.W_in = input_shape[2]
            kernel_size = self.args[1].value
            stride = self.args[2].value
            padding = self.args[3].value if len(self.args) > 3 else [0]
            self.kH = 1
            self.kW = kernel_size[0]
            self.sH = 1
            self.sW = stride[0]
            self.pH = 0
            self.pW = padding[0]
        else:
            self.N = input_shape[0]
            self.C = input_shape[1]
            self.H_in = input_shape[2]
            self.W_in = input_shape[3]
            kernel_size = self.args[1].value
            stride = self.args[2].value
            padding = self.args[3].value if len(self.args) > 3 else [0, 0]
            self.kH = kernel_size[0]
            self.kW = kernel_size[1]
            self.sH = stride[0]
            self.sW = stride[1]
            self.pH = padding[0]
            self.pW = padding[1]

        self.H_out = (self.H_in + 2 * self.pH - self.kH) // self.sH + 1
        self.W_out = (self.W_in + 2 * self.pW - self.kW) // self.sW + 1

    def get_constants(self):
        return {
            101: self.N,
            102: self.C,
            103: self.H_out,
            104: self.W_out,
            105: self.kH,
            106: self.kW,
            107: self.sH,
            108: self.sW,
            109: self.pH,
            110: self.pW,
            111: self.H_in,
            112: self.W_in,
            113: self.kH * self.kW,
        }


class AdaptiveAvgPool2dInstruction(TemplateInstruction):
    aten_op = "aten.adaptive_avg_pool2d.default"
    template_name = "adaptive_avg_pool2d"

    def prepare(self):
        input_shape = self.args[0].shape
        output_size = self.args[1].value

        self.N = input_shape[0]
        self.C = input_shape[1]
        self.H_in = input_shape[2]
        self.W_in = input_shape[3]
        self.H_out = output_size[0]
        self.W_out = output_size[1]
        self.kH = self.H_in // self.H_out
        self.kW = self.W_in // self.W_out

    def get_constants(self):
        return {
            101: self.N,
            102: self.C,
            103: self.H_out,
            104: self.W_out,
            105: self.kH,
            106: self.kW,
            107: self.H_in,
            108: self.W_in,
            109: self.kH * self.kW,
        }
