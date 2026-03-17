from abc import abstractmethod, ABC
from enum import Enum

from cattorch.util.argument import Argument


class ScratchInstruction(Enum):
    NONE = 0
    MATMUL = 1
    TRANSPOSE = 2
    RELU = 3
    TENSOR_ADD = 4


class Instruction(ABC):
    _registry: dict[str, type["Instruction"]] = {}
    aten_op: str  # subclasses must set this

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "aten_op"):
            Instruction._registry[cls.aten_op] = cls

    def __init__(self, torch_name: str, output: Argument, *args: Argument):
        self.torch_name = torch_name
        self.output = output
        self.args = args
        self.constants = []
        self.scratch_instruction = ScratchInstruction.NONE
        self.prepare()

    @classmethod
    def create(cls, aten_op: str, torch_name: str, output: Argument, *args: Argument) -> "Instruction":
        if aten_op not in cls._registry:
            raise NotImplementedError(f"Unsupported operation: {aten_op}")
        return cls._registry[aten_op](torch_name, output, *args)

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def finalize(self):
        pass