import torch
import torch.nn as nn
import torch.nn.functional as F

from cattorch.transpiler import transpile


class Softmax1D(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=0)


class Softmax2D(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=-1)


class Softmax2DDim0(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=0)


class SoftmaxBatch(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=-1)


if __name__ == "__main__":
    print("=== Test 1: 1D softmax ===")
    m1 = Softmax1D()
    x1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    print("output:", F.softmax(x1, dim=0).tolist())
    transpile(m1, torch.Size([4]), "softmax_1d")

    print("\n=== Test 2: 2D softmax dim=-1 ===")
    m2 = Softmax2D()
    x2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print("output:", F.softmax(x2, dim=-1).tolist())
    transpile(m2, torch.Size([2, 3]), "softmax_2d")

    print("\n=== Test 3: 2D softmax dim=0 ===")
    m3 = Softmax2DDim0()
    transpile(m3, torch.Size([2, 3]), "softmax_2d_dim0")

    print("\n=== Test 4: numerical stability ===")
    x4 = torch.tensor([1000.0, 1001.0, 1002.0])
    print("output:", F.softmax(x4, dim=0).tolist())
    transpile(Softmax1D(), torch.Size([3]), "softmax_stability")

    print("\n=== Test 5: batch softmax [3, 2] ===")
    m5 = SoftmaxBatch()
    x5 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    print("output:", F.softmax(x5, dim=-1).tolist())
    transpile(m5, torch.Size([3, 2]), "softmax_batch")