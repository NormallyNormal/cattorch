import torch
from torch import Size


class Argument:
    def __init__(
        self,
        name: str,
        shape: Size,
        value=None,
        *,
        dynamic_axis: int | None = None,
        dtype: torch.dtype | None = None,
        static: bool = False,
    ):
        self.name = name
        self.shape = shape
        self.value = value
        self.dynamic_axis = dynamic_axis
        self.dtype = dtype
        self.static = static

    @property
    def dynamic(self) -> bool:
        return self.dynamic_axis is not None
