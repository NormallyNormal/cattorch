import torch
import torch.nn as nn

from cattorch.transpiler import transpile


class SimpleMatMulModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.W2 = nn.Parameter(torch.randn(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, input_dim]
        # [batch, input_dim] @ [input_dim, hidden_dim] -> [batch, hidden_dim]
        hidden = x @ self.W1
        print(hidden)

        # [batch, hidden_dim] @ [hidden_dim, output_dim] -> [batch, output_dim]
        out = hidden @ self.W2

        return out


if __name__ == "__main__":
    torch.manual_seed(42)
    model = SimpleMatMulModel(input_dim=4, hidden_dim=8, output_dim=2)
    x = torch.tensor([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]], dtype=torch.float32)
    print(model.forward(x))
    transpile(model, torch.Size([3, 4]), "simple_matmul")