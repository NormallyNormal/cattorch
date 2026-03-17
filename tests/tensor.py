import torch
import torch.nn as nn

from cattorch import transpile


class TensorMul(nn.Module):
    def __init__(self):
        super().__init__()  # <-- this was missing
        self.a = nn.Parameter(torch.randn(1, 2, 3, 4, 5))
        self.b = nn.Parameter(torch.randn(1, 2, 3, 5, 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a @ self.b


# Test it
torch.manual_seed(42)
module = TensorMul()
x = torch.randn(1, 2, 3, 4, 5)
out = module(x)
print(out.flatten()[23:])
print(len(out.flatten()))
print(out.shape)  # torch.Size([1, 2, 3, 4, 6])
transpile(module, x.shape, "tensor")