import json
import math
from abc import abstractmethod, ABC

from cattorch.templates.template import TEMPLATE_DIR
from cattorch.util.argument import Argument
from cattorch.util.scratch.constant_replacer import ConstantReplacer


class Instruction(ABC):
    _registry: dict[str, type["Instruction"]] = {}
    aten_op: str | list[str]  # subclasses must set this

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "aten_op"):
            ops = cls.aten_op if isinstance(cls.aten_op, list) else [cls.aten_op]
            for op in ops:
                Instruction._registry[op] = cls

    def __init__(self, torch_name: str, output: Argument, *args: Argument):
        self.torch_name = torch_name
        self.output = output
        self.args = args
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


class TemplateInstruction(Instruction):
    """Base class for instructions that load a template and apply constants.

    Subclasses set `template_name` and optionally override `get_constants()`.
    This covers all simple elementwise ops, scalar ops, and most others.
    """
    template_name: str

    def prepare(self):
        pass

    def get_constants(self) -> dict:
        return {101: math.prod(self.args[0].shape)}

    def finalize(self):
        template_path = TEMPLATE_DIR / self.template_name / "template.json"
        with open(template_path) as f:
            data = json.load(f)
        return ConstantReplacer(self.get_constants()).apply(data)
