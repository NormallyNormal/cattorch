"""Generate the stock-Scratch performance suite into benchmarks/artifacts/."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from cattorch import build_benchmark_suite


OUTPUT = Path(__file__).parent / "artifacts"


class ParameterMatMul(nn.Module):
    def __init__(self, inner, columns):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(inner, columns))

    def forward(self, x):
        return x @ self.weight


class BatchedMatMul(nn.Module):
    def forward(self, left, right):
        return left @ right


class SoftmaxCase(nn.Module):
    def forward(self, x):
        return F.softmax(x, dim=-1)


class NormCase(nn.Module):
    def __init__(self, width, rms=False):
        super().__init__()
        self.norm = nn.RMSNorm(width) if rms else nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x)


class TinyGPTBlock(nn.Module):
    def __init__(self, width, heads, hidden, context):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.scale = self.head_width ** 0.5
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.projection = nn.Linear(width, width, bias=False)
        self.norm2 = nn.LayerNorm(width)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        mask = torch.triu(torch.ones(context, context, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        k = k.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        v = v.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / self.scale
        scores = scores.masked_fill(self.mask[:length, :length], float("-inf"))
        attention = F.softmax(scores, dim=-1)
        joined = (attention @ v).transpose(1, 2).contiguous().view(batch, length, width)
        x = x + self.projection(joined)
        return x + self.down(F.gelu(self.up(self.norm2(x)), approximate="tanh"))


class TinyGPT(nn.Module):
    def __init__(self, width, heads, hidden, context, vocab=29):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, width)
        self.position_embedding = nn.Embedding(context, width)
        self.register_buffer("positions", torch.arange(context))
        self.block = TinyGPTBlock(width, heads, hidden, context)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.token_embedding(tokens) + self.position_embedding(self.positions[:length])
        return self.head(self.norm(self.block(x)))


def build_cases():
    torch.manual_seed(42)
    cases = [
        ("matmul_32x32_32x96", ParameterMatMul(32, 96), torch.randn(32, 32)),
        ("matmul_32x32_32x64", ParameterMatMul(32, 64), torch.randn(32, 32)),
        (
            "batched_matmul_2x32x16_2x16x32",
            BatchedMatMul(),
            (torch.randn(2, 32, 16), torch.randn(2, 16, 32)),
        ),
        ("softmax_2x32x32", SoftmaxCase(), torch.randn(2, 32, 32)),
        ("layernorm_32x32", NormCase(32), torch.randn(32, 32)),
        ("rmsnorm_32x32", NormCase(32, rms=True), torch.randn(32, 32)),
        ("quick_gpt", TinyGPT(16, 2, 32, 8), torch.randint(0, 29, (1, 8))),
        ("catgpt1", TinyGPT(32, 2, 64, 32), torch.randint(0, 29, (1, 32))),
    ]
    for _, model, _ in cases:
        model.eval()
    return cases


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = build_benchmark_suite(
        build_cases(),
        OUTPUT / "cattorch_suite.sb3",
        iterations=100,
    )
    print(f"Suite written: {path}")


if __name__ == "__main__":
    main()
