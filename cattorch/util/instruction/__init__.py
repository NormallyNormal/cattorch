from cattorch.util.instruction.instruction import Instruction
from cattorch.util.instruction.elementwise import (
    ELUInstruction,
    GELUInstruction,
    LayerNormInstruction,
    LeakyReLUInstruction,
    ReLUInstruction,
    ScalarDivideInstruction,
    ScalarMultiplyInstruction,
    SigmoidInstruction,
    SiLUInstruction,
    TanhInstruction,
    TensorAddInstruction,
)
from cattorch.util.instruction.matmul import MatMulInstruction
from cattorch.util.instruction.softmax import SoftmaxInstruction
from cattorch.util.instruction.transpose import TransposeInstruction
