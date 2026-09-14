from torch import Size


class Argument:
    def __init__(
        self, name: str, shape: Size, value=None, *, dynamic_axis: int | None = None,
    ):
        self.name = name
        self.shape = shape
        self.value = value
        self.dynamic_axis = dynamic_axis

    @property
    def dynamic(self) -> bool:
        return self.dynamic_axis is not None
