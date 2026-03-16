from abc import abstractmethod, ABC
from enum import Enum
from pathlib import Path

import torch

from src.cattorch.util.argument import Argument

class ScratchInstruction(Enum):
    NONE = 0
    MATMUL = 1
    TRANSPOSE = 2
    RELU = 3
    TENSOR_ADD = 4

class Instruction(ABC):
    def __init__(self, torch_name : str, output: Argument, *args : Argument):
        self.torch_name = torch_name
        self.output = output
        self.args = args
        self.constants = []
        self.scratch_instruction = ScratchInstruction.NONE
        self.prepare()

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def finalize(self):
        pass