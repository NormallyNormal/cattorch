import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cattorch.transpiler import transpile


class SimpleNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.W2 = nn.Parameter(torch.randn(hidden_dim, output_dim))
        self.b2 = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, input_dim] @ [input_dim, hidden_dim] + [hidden_dim] -> [batch, hidden_dim]
        hidden = x @ self.W1 + self.b1
        hidden = F.relu(hidden)

        # [batch, hidden_dim] @ [hidden_dim, output_dim] + [output_dim] -> [batch, output_dim]
        out = hidden @ self.W2 + self.b2

        return out


if __name__ == "__main__":
    torch.manual_seed(42)
    model = SimpleNN(input_dim=4, hidden_dim=8, output_dim=2)
    x = torch.tensor([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]], dtype=torch.float32)
    out = model(x)
    print("input shape: ", x.shape)
    print("output shape:", out.shape)
    print("output:", out)
    transpile(model, torch.Size([3, 4]), "simple_neural_net")