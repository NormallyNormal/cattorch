from abc import ABC, abstractmethod

from cattorch.util.argument import Argument


class Instruction(ABC):
    def __init__(self, torch_name: str, output: Argument, *args: Argument):
        self.torch_name = torch_name
        self.output = output
        self.args = args
        self.prepare()

    @abstractmethod
    def prepare(self):
        pass

    def transform_weights(self, static_lists: dict) -> None:
        """Mutate static weight tensors before they are stored in the sprite.

        Called by the transpiler after weight resolution.  The default is a
        no-op; subclasses can override to precompute derived values (e.g.
        BatchNorm precomputes ``sqrt(running_var + eps)``).
        """

    @abstractmethod
    def finalize(self):
        pass
