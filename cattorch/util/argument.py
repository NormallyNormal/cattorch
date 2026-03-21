from torch import Size


class Argument:
    def __init__(self, name: str, shape: Size, value=None):
        self.name = name
        self.shape = shape
        self.value = value