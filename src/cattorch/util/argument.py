from torch import Size

class Argument():
    def __init__(self, name: str, shape: Size):
        self.name = name
        self.shape = shape