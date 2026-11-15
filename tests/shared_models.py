"""Small models that several test modules export and check against.

Toy decoders for cached generation, recurrent language models with their
states, and a character-level language model with its tokenizer.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cattorch import State

CONTEXT = 16
VOCAB = 40


# ── Toy decoders ─────────────────────────────────────────────────────────────

class NanoBlock(nn.Module):
    """Separate q/k/v projections and a registered causal mask buffer."""

    def __init__(self, width, heads, context, mix=None):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(width)
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.proj = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.register_buffer("bias", torch.tril(torch.ones(context, context))[None, None])
        self.mix = mix

    def forward(self, x):
        batch, length, width = x.shape
        h = self.ln1(x)
        split = lambda t: t.view(batch, length, self.heads, -1).transpose(1, 2)
        q, k, v = split(self.q(h)), split(self.k(h)), split(self.v(h))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.shape[-1])
        scores = scores.masked_fill(self.bias[:, :, :length, :length] == 0, float("-inf"))
        out = F.softmax(scores, dim=-1) @ v
        x = x + self.proj(out.transpose(1, 2).reshape(batch, length, width))
        if self.mix is not None:
            x = self.mix(x)
        return x + self.mlp(self.ln2(x))


class NanoGPT(nn.Module):
    def __init__(self, layers=2, width=16, heads=2, context=CONTEXT, mix=None):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, width)
        self.wpe = nn.Embedding(context, width)
        self.blocks = nn.ModuleList(NanoBlock(width, heads, context, mix) for _ in range(layers))
        self.ln_f = nn.LayerNorm(width)
        self.head = nn.Linear(width, VOCAB, bias=False)

    def forward(self, tokens):
        positions = torch.arange(0, tokens.shape[1])
        x = self.wte(tokens) + self.wpe(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


class MQABlock(NanoBlock):
    """Broadcast multi-query attention: one k/v head against all q heads, no repeat_kv."""

    def __init__(self, width, heads, context, mix=None):
        super().__init__(width, heads, context, mix)
        self.k = nn.Linear(width, width // heads)
        self.v = nn.Linear(width, width // heads)

    def forward(self, x):
        batch, length, width = x.shape
        h = self.ln1(x)
        q = self.q(h).view(batch, length, self.heads, -1).transpose(1, 2)
        k = self.k(h).view(batch, length, 1, -1).transpose(1, 2)
        v = self.v(h).view(batch, length, 1, -1).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.shape[-1])
        scores = scores.masked_fill(self.bias[:, :, :length, :length] == 0, float("-inf"))
        out = F.softmax(scores, dim=-1) @ v
        x = x + self.proj(out.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.ln2(x))


class NanoMQA(NanoGPT):
    def __init__(self, layers=2, width=16, heads=2, context=CONTEXT):
        super().__init__(layers, width, heads, context)
        self.blocks = nn.ModuleList(MQABlock(width, heads, context) for _ in range(layers))


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RopeBlock(nn.Module):
    """Grouped-query attention with Hugging Face style ``repeat_kv`` and RoPE."""

    def __init__(self, width, heads, kv_heads, **attention):
        super().__init__()
        self.heads, self.kv_heads = heads, kv_heads
        self.dim = width // heads
        self.q = nn.Linear(width, heads * self.dim, bias=False)
        self.k = nn.Linear(width, kv_heads * self.dim, bias=False)
        self.v = nn.Linear(width, kv_heads * self.dim, bias=False)
        self.o = nn.Linear(heads * self.dim, width, bias=False)
        self.attention = attention

    def repeat(self, x):
        batch, kv, length, dim = x.shape
        groups = self.heads // self.kv_heads
        x = x[:, :, None].expand(batch, kv, groups, length, dim)
        return x.reshape(batch, kv * groups, length, dim)

    def forward(self, x, cos, sin):
        batch, length, _ = x.shape
        q = self.q(x).view(batch, length, self.heads, self.dim).transpose(1, 2)
        k = self.k(x).view(batch, length, self.kv_heads, self.dim).transpose(1, 2)
        v = self.v(x).view(batch, length, self.kv_heads, self.dim).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        out = F.scaled_dot_product_attention(q, self.repeat(k), self.repeat(v), **self.attention)
        return self.o(out.transpose(1, 2).reshape(batch, length, -1))


class RopeLM(nn.Module):
    def __init__(self, layers=2, width=16, heads=4, kv_heads=2, **attention):
        super().__init__()
        attention = attention or {"is_causal": True}
        self.dim = width // heads
        self.embed = nn.Embedding(VOCAB, width)
        self.layers = nn.ModuleList(
            RopeBlock(width, heads, kv_heads, **attention) for _ in range(layers)
        )
        self.head = nn.Linear(width, VOCAB, bias=False)
        self.register_buffer("inv_freq", 1.0 / 100 ** (torch.arange(0, self.dim, 2) / self.dim))

    def forward(self, tokens):
        angles = torch.arange(tokens.shape[1])[:, None] * self.inv_freq[None]
        angles = torch.cat((angles, angles), dim=-1)
        cos, sin = angles.cos()[None, None], angles.sin()[None, None]
        x = self.embed(tokens)
        for layer in self.layers:
            x = x + layer(x, cos, sin)
        return self.head(x)


def nano():
    torch.manual_seed(5)
    return NanoGPT(layers=2, width=16, heads=2, context=16).eval()


# ── Hugging Face decoders ────────────────────────────────────────────────────

def tiny_decoder(kind, layers=2, heads=4, kv_heads=2):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(3)
    if kind == "gpt2":
        return transformers.GPT2LMHeadModel(transformers.GPT2Config(
            n_embd=32, n_layer=layers, n_head=heads, vocab_size=64, n_positions=64,
        ))
    config = dict(
        hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=kv_heads, vocab_size=64,
        max_position_embeddings=64,
    )
    if kind == "llama":
        return transformers.LlamaForCausalLM(transformers.LlamaConfig(**config))
    return transformers.Qwen2ForCausalLM(transformers.Qwen2Config(**config))


# ── Cached attention language models ─────────────────────────────────────────

class CachedAttention(nn.Module):
    def __init__(self, width=8, heads=2, context=6):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.projection = nn.Linear(width, width, bias=False)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context, context, dtype=torch.bool), diagonal=1),
        )

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        k = k.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        v = v.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / self.head_width ** 0.5
        weights = F.softmax(
            scores.masked_fill(self.mask[:length, :length], float("-inf")), dim=-1,
        )
        joined = (weights @ v).transpose(1, 2).reshape(batch, length, width)
        return self.projection(joined)


class CachedMQAAttention(nn.Module):
    def __init__(self, width=8, heads=2, context=6):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, width + 2 * self.head_width, bias=False)
        self.projection = nn.Linear(width, width, bias=False)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context, context, dtype=torch.bool), diagonal=1),
        )

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).split(
            (width, self.head_width, self.head_width), dim=-1,
        )
        q = q.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        k = k.view(batch, length, 1, self.head_width).transpose(1, 2)
        v = v.view(batch, length, 1, self.head_width).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / self.head_width ** 0.5
        weights = F.softmax(
            scores.masked_fill(self.mask[:length, :length], float("-inf")), dim=-1,
        )
        joined = (weights @ v).transpose(1, 2).reshape(batch, length, width)
        return self.projection(joined)


class CachedBlock(nn.Module):
    def __init__(self, context=6):
        super().__init__()
        self.norm1 = nn.LayerNorm(8)
        self.attention = CachedAttention(context=context)
        self.norm2 = nn.LayerNorm(8)
        self.mlp = nn.Sequential(
            nn.Linear(8, 12), nn.GELU(approximate="tanh"), nn.Linear(12, 8),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class CachedLM(nn.Module):
    def __init__(self, context=6, layers=2):
        super().__init__()
        self.token = nn.Embedding(13, 8)
        self.position = nn.Embedding(context, 8)
        self.blocks = nn.Sequential(*(CachedBlock(context) for _ in range(layers)))
        self.norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, 13, bias=False)

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.token(tokens) + self.position(torch.arange(length))
        return self.head(self.norm(self.blocks(x)))


class CachedMQALM(CachedLM):
    def __init__(self, context=6, layers=2):
        super().__init__(context=context, layers=layers)
        for block in self.blocks:
            block.attention = CachedMQAAttention(context=context)


# ── Recurrent language models ────────────────────────────────────────────────

class GRULM(nn.Module):
    """A GRU language model that starts from a learned, non-constant state."""

    def __init__(self, vocab=16, width=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.gru = nn.GRU(width, width, batch_first=True)
        self.head = nn.Linear(width, vocab)
        self.register_buffer("h0", torch.randn(1, 1, width))

    def states(self):
        return (State("hidden", self.h0.clone()),)

    def forward(self, tokens):
        output, _ = self.gru(self.embed(tokens), self.h0)
        return self.head(output)

    def step(self, token, hidden):
        output, hidden = self.gru(self.embed(token), hidden)
        return self.head(output), hidden


class LSTMLM(nn.Module):
    """Two states; the cell starts at one non-zero value everywhere."""

    def __init__(self, vocab=16, width=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.lstm = nn.LSTM(width, width, batch_first=True)
        self.head = nn.Linear(width, vocab)
        self.width = width

    def states(self):
        return (
            State("hidden", torch.zeros(1, 1, self.width)),
            State("cell", torch.full((1, 1, self.width), 0.25)),
        )

    def forward(self, tokens):
        hidden, cell = (state.initial for state in self.states())
        output, _ = self.lstm(self.embed(tokens), (hidden, cell))
        return self.head(output)

    def step(self, token, hidden, cell):
        output, (hidden, cell) = self.lstm(self.embed(token), (hidden, cell))
        return self.head(output)[:, 0], hidden, cell


def rms(value, weight, eps=1e-5):
    return weight * value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps)


class SSMBlock(nn.Module):
    """A Mamba-2 mixer: gated depthwise causal convolution and a selective scan."""

    def __init__(self, width, heads, head_dim, state, kernel):
        super().__init__()
        self.inner, self.heads, self.head_dim = heads * head_dim, heads, head_dim
        self.state, self.kernel = state, kernel
        self.conv_dim = self.inner + 2 * state
        self.norm = nn.Parameter(1 + 0.1 * torch.randn(width))
        self.in_proj = nn.Linear(width, self.inner + self.conv_dim + heads, bias=False)
        self.conv_weight = nn.Parameter(0.5 * torch.randn(self.conv_dim, kernel))
        self.conv_bias = nn.Parameter(0.1 * torch.randn(self.conv_dim))
        self.dt_bias = nn.Parameter(0.1 * torch.randn(heads))
        self.A_log = nn.Parameter(torch.log(torch.linspace(1, 4, heads)))
        self.D = nn.Parameter(torch.ones(heads))
        self.gate_norm = nn.Parameter(1 + 0.1 * torch.randn(self.inner))
        self.out_proj = nn.Linear(self.inner, width, bias=False)

    def project(self, h):
        return self.in_proj(rms(h, self.norm)).split(
            [self.inner, self.conv_dim, self.heads], dim=-1,
        )

    def scan(self, ssm, xbc, dt):
        x, B, C = F.silu(xbc).split([self.inner, self.state, self.state], dim=-1)
        x = x.reshape(self.heads, self.head_dim)
        dt = F.softplus(dt.reshape(self.heads) + self.dt_bias)
        decay = torch.exp(-dt * torch.exp(self.A_log))
        ssm = ssm * decay[:, None, None] + (dt[:, None] * x)[:, :, None] * B[None, None, :]
        y = (ssm * C[None, None, :]).sum(-1) + self.D[:, None] * x
        return ssm, y.reshape(1, self.inner)

    def finish(self, h, y, gate):
        return h + self.out_proj(rms(y * F.silu(gate), self.gate_norm))

    def step(self, h, conv, ssm):
        gate, xbc, dt = self.project(h)
        window = torch.cat([conv, xbc.reshape(self.conv_dim, 1)], dim=-1)
        mixed = (window * self.conv_weight).sum(-1) + self.conv_bias
        ssm, y = self.scan(ssm, mixed, dt)
        return self.finish(h, y, gate), window[:, 1:], ssm

    def forward(self, h):
        gate, xbc, dt = self.project(h)
        length = h.shape[0]
        mixed = F.conv1d(
            xbc.T[None], self.conv_weight[:, None], self.conv_bias,
            padding=self.kernel - 1, groups=self.conv_dim,
        )[0, :, :length].T
        ssm = torch.zeros(self.heads, self.head_dim, self.state)
        rows = []
        for position in range(length):
            ssm, y = self.scan(ssm, mixed[position], dt[position])
            rows.append(y)
        return self.finish(h, torch.cat(rows), gate)


class SSMLM(nn.Module):
    def __init__(self, vocab=24, width=8, layers=2, heads=2, head_dim=4, state=4, kernel=3):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList(
            SSMBlock(width, heads, head_dim, state, kernel) for _ in range(layers)
        )
        self.norm = nn.Parameter(1 + 0.1 * torch.randn(width))
        self.head = nn.Linear(width, vocab, bias=False)

    def states(self):
        block = self.blocks[0]
        return (
            State("conv", torch.zeros(len(self.blocks), block.conv_dim, block.kernel - 1)),
            State("ssm", torch.zeros(len(self.blocks), block.heads, block.head_dim, block.state)),
        )

    def forward(self, tokens):
        h = self.embed(tokens)[0]
        for block in self.blocks:
            h = block(h)
        return self.head(rms(h, self.norm))[None]

    def step(self, token, conv, ssm):
        h = self.embed(token).reshape(1, -1)
        convs, ssms = [], []
        for index, block in enumerate(self.blocks):
            h, new_conv, new_ssm = block.step(h, conv[index], ssm[index])
            convs.append(new_conv)
            ssms.append(new_ssm)
        return self.head(rms(h, self.norm)), torch.stack(convs), torch.stack(ssms)


RECURRENT_MODELS = {"gru": GRULM, "lstm": LSTMLM, "ssm": SSMLM}


def recurrent_model(name):
    torch.manual_seed({"gru": 1, "lstm": 2, "ssm": 3}[name])
    return RECURRENT_MODELS[name]().eval()


def reference_rows(model, tokens):
    with torch.no_grad():
        return model(torch.tensor([tokens])).reshape(len(tokens), -1)


# ── A character-level language model ─────────────────────────────────────────

CHARACTERS = "?abcdefghijklmnopqrstuvwxyz ."


def char_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel(
        vocab={character: index for index, character in enumerate(CHARACTERS)}, unk_token="?",
    ))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="?")


class Language(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = nn.Embedding(len(CHARACTERS), 8)
        self.positions = nn.Embedding(16, 8)
        self.layer = nn.TransformerEncoderLayer(8, 2, 16, dropout=0.0, batch_first=True)
        self.head = nn.Linear(8, len(CHARACTERS))

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.tokens(tokens) + self.positions(torch.arange(length))
        mask = nn.Transformer.generate_square_subsequent_mask(length)
        return self.head(self.layer(x, src_mask=mask, is_causal=True))


def greedy(model, prompt, count):
    ids = [CHARACTERS.index(character) for character in prompt]
    with torch.no_grad():
        for _ in range(count):
            ids.append(int(model(torch.tensor([ids]))[0, -1].argmax()))
    return ids[len(prompt):]
