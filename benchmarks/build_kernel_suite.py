"""Generate broad exact/fast coverage for every cattorch Scratch kernel."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from cattorch import build_benchmark_suite


OUTPUT = Path(__file__).parent / "artifacts" / "kernel_suite.sb3"


class FunctionCase(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, *inputs):
        return self.function(*inputs)


class ParameterMatMul(nn.Module):
    def __init__(self, inner, columns):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(inner, columns))

    def forward(self, x):
        return x @ self.weight


class EmbeddingCase(nn.Module):
    def __init__(self, vocab, width):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)

    def forward(self, indices):
        return self.embedding(indices)


class CausalMaskCase(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1),
        )

    def forward(self, scores):
        return scores.masked_fill(self.mask, float("-inf"))


def build_cases():
    vector = torch.randn(32, 32)
    positive = vector.abs() + 0.1
    other = torch.randn(32, 32)
    image = torch.randn(1, 4, 16, 16)
    signal = torch.randn(1, 4, 64)

    cases = [
        ("scalar_add", FunctionCase(lambda x: x + 0.25), vector),
        ("scalar_mul", FunctionCase(lambda x: x * 0.25), vector),
        ("scalar_div", FunctionCase(lambda x: x / 3.0), vector),
        ("pow_square", FunctionCase(lambda x: torch.pow(x, 2)), vector),
        ("pow_zero", FunctionCase(lambda x: torch.pow(x, 0)), vector),
        ("tensor_add", FunctionCase(torch.add), (vector, other)),
        ("tensor_sub", FunctionCase(torch.sub), (vector, other)),
        ("tensor_mul", FunctionCase(torch.mul), (vector, other)),
        ("tensor_div", FunctionCase(torch.div), (vector, positive)),
        (
            "arithmetic_chain",
            FunctionCase(lambda x, y, residual: (x + y) * 0.5 - residual),
            (vector, other, torch.randn(32, 32)),
        ),
        ("negate", FunctionCase(torch.neg), vector),
        ("relu", FunctionCase(torch.relu), vector),
        ("sigmoid", FunctionCase(torch.sigmoid), vector),
        ("tanh", FunctionCase(torch.tanh), vector),
        ("gelu_tanh", FunctionCase(lambda x: F.gelu(x, approximate="tanh")), vector),
        ("silu", FunctionCase(F.silu), vector),
        ("leaky_relu", FunctionCase(lambda x: F.leaky_relu(x, 0.2)), vector),
        ("elu", FunctionCase(F.elu), vector),
        ("rsqrt", FunctionCase(torch.rsqrt), positive),
        ("mean_last", FunctionCase(lambda x: x.mean(dim=-1)), vector),
        ("softmax_last", FunctionCase(lambda x: F.softmax(x, dim=-1)), vector),
        ("transpose_3d", FunctionCase(lambda x: x.transpose(0, 2)), torch.randn(4, 16, 16)),
        ("slice_columns", FunctionCase(lambda x: x[:, 4:28]), vector),
        ("cat_columns", FunctionCase(lambda x, y: torch.cat((x, y), dim=1)), (vector, other)),
        ("embedding", EmbeddingCase(128, 32), torch.randint(0, 128, (256,))),
        ("masked_fill", CausalMaskCase(16), torch.randn(2, 16, 16)),
        ("matmul", ParameterMatMul(32, 64), vector),
        (
            "batched_matmul",
            FunctionCase(torch.matmul),
            (torch.randn(2, 32, 16), torch.randn(2, 16, 32)),
        ),
        ("layernorm", nn.LayerNorm(32), vector),
        ("rmsnorm", nn.RMSNorm(32), vector),
        ("batchnorm1d", nn.BatchNorm1d(4).eval(), signal),
        ("batchnorm2d", nn.BatchNorm2d(4).eval(), image),
        ("conv1d", nn.Conv1d(4, 8, 3, padding=1).eval(), signal),
        ("conv2d", nn.Conv2d(4, 8, 3, padding=1).eval(), image),
        ("maxpool1d", nn.MaxPool1d(2, 2), signal),
        ("maxpool2d", nn.MaxPool2d(2, 2), image),
        ("avgpool1d", nn.AvgPool1d(2, 2), signal),
        ("avgpool2d", nn.AvgPool2d(2, 2), image),
        ("adaptive_avgpool2d", nn.AdaptiveAvgPool2d((4, 4)), image),
    ]
    for _, model, _ in cases:
        model.eval()
    return cases


def main():
    torch.manual_seed(123)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    path = build_benchmark_suite(build_cases(), OUTPUT, iterations=100)
    print(f"Kernel suite written: {path}")


if __name__ == "__main__":
    main()
