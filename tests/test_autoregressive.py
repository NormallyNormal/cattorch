"""Sequence analysis for automatic cached generation (``cattorch.autoregressive``)."""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cattorch import UnsupportedModelError, UnsupportedOperationError
from cattorch.autoregressive import (
    analyze_sequence, build_generation_step, build_step_module,
    check_step_equivalence, prepare_sequence_model,
)
from cattorch.frontend import capture_model

CONTEXT = 16
VOCAB = 40


def analyze(model, context=CONTEXT):
    return analyze_sequence(prepare_sequence_model(model.eval()), context)


# ── Toy models ───────────────────────────────────────────────────────────────

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


# ── Plans ────────────────────────────────────────────────────────────────────

def test_nanogpt_plan():
    plan = analyze(NanoGPT(layers=3, heads=2))
    assert plan.max_context == CONTEXT
    assert plan.n_layers == len(plan.attention) == 3
    for index, block in enumerate(plan.attention):
        assert block.index == index
        assert block.module_path == f"blocks.{index}"
        assert (block.n_heads, block.n_kv_heads, block.kv_groups) == (2, 2, 1)
        assert (block.head_dim, block.value_dim) == (8, 8)
        assert block.head_axis == 1
        assert block.mask_kind == "masked_fill"
        assert block.scale == pytest.approx(1 / math.sqrt(8))
        assert {block.scores, block.softmax, block.output} <= block.internal
        # Separate projections: each of q, k and v is its own sequence tensor.
        assert len({block.query, block.key, block.value}) == 3
        assert block.key_source == block.key and block.value_source == block.value
    # One learned position table: the wpe rows for arange(L), added to the tokens.
    (table,) = plan.tables
    assert (table.shape, table.axis, table.kind) == ((CONTEXT, 16), 0, "additive")
    assert table.value.shape == (CONTEXT, 16)
    # The L-graph literals that depend on the length were found.
    assert plan.literal_diffs
    assert all(d.long != d.short for d in plan.literal_diffs)
    assert plan.outputs[0][1].axis == 1
    assert plan.nodes[plan.token_input].shape == (1, CONTEXT)


@pytest.mark.parametrize("context", [CONTEXT, 9])
def test_context_may_equal_or_undercut_the_mask_buffer(context):
    # At context == buffer size the mask slice is a no-op and leaves no node.
    plan = analyze(NanoGPT(layers=2), context)
    assert len(plan.attention) == 2
    assert plan.tables[0].shape == (context, 16)


def test_literal_diffs_extrapolate_to_one_row():
    plan = analyze(NanoGPT(layers=1))
    views = [
        d for d in plan.literal_diffs
        if plan.nodes[d.node].target == "aten.view.default" and d.location == ("args", 1, 1)
    ]
    assert views  # view(h, [batch, length, heads, -1]): the length differs
    for diff in views:
        assert (diff.long, diff.short) == (CONTEXT, CONTEXT - 1)
        assert diff.value_at(CONTEXT, CONTEXT) == CONTEXT
        assert diff.value_at(CONTEXT - 1, CONTEXT) == CONTEXT - 1
        assert diff.value_at(1, CONTEXT) == 1
    assert plan.literal_diffs_for(views[0].node) == (views[0],)
    # Only literals that really differ are recorded.
    assert all(d.long != d.short for d in plan.literal_diffs)


def test_rope_plan_with_grouped_query_attention():
    plan = analyze(RopeLM(layers=2, heads=4, kv_heads=1), 12)
    assert len(plan.attention) == 2
    for block in plan.attention:
        assert (block.n_heads, block.n_kv_heads, block.kv_groups) == (4, 1, 4)
        assert block.key_source != block.key and block.value_source != block.value
        assert block.mask_kind == "masked_fill"
    cos_sin = [t for t in plan.tables if t.kind == "multiplicative"]
    assert cos_sin and len(cos_sin) == len(plan.tables)
    for table in plan.tables:
        assert table.shape == (1, 1, 12, 4) and table.axis == 2
        assert table.value.shape == (1, 1, 12, 4)
    # cos and sin both appear, with different values.
    assert len({tuple(t.value.flatten().tolist()) for t in plan.tables}) == 2


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


@pytest.mark.parametrize("kind", ["gpt2", "llama", "qwen2"])
def test_huggingface_decoder_plan(kind):
    plan = analyze(tiny_decoder(kind, layers=2), 24)
    assert len(plan.attention) == 2
    paths = [block.module_path for block in plan.attention]
    assert paths == [
        f"transformer.h.{i}.attn" if kind == "gpt2" else f"model.layers.{i}.self_attn"
        for i in range(2)
    ]
    for block in plan.attention:
        assert block.mask_kind == "masked_fill"  # SDPA bool mask -> masked_fill(-inf)
        assert block.head_dim == 8
        if kind == "gpt2":
            assert (block.n_heads, block.n_kv_heads, block.kv_groups) == (4, 4, 1)
            assert block.key_source == block.key
        else:
            assert (block.n_heads, block.n_kv_heads, block.kv_groups) == (4, 2, 2)
            assert block.key_source != block.key
            assert block.value_source != block.value
    if kind == "gpt2":
        (table,) = plan.tables
        assert (table.shape, table.axis, table.kind) == ((1, 24, 32), 1, "additive")
    else:
        # RoPE cos and sin: one (1, 1, L, head_dim) pair per layer.
        assert len(plan.tables) == 4
        assert {t.kind for t in plan.tables} == {"multiplicative"}
        assert {t.shape for t in plan.tables} == {(1, 1, 24, 8)}
        assert {t.axis for t in plan.tables} == {2}


def test_huggingface_single_kv_head():
    plan = analyze(tiny_decoder("llama", layers=1, heads=4, kv_heads=1), 16)
    (block,) = plan.attention
    assert (block.n_heads, block.n_kv_heads, block.kv_groups) == (4, 1, 4)


# ── Rejections ───────────────────────────────────────────────────────────────

class CumSum(nn.Module):
    def forward(self, x):
        return x + x.cumsum(dim=1)


class SequenceMean(nn.Module):
    def forward(self, x):
        return x - x.mean(dim=1, keepdim=True)


class Roll(nn.Module):
    def forward(self, x):
        return x + torch.roll(x, 1, dims=1)


@pytest.mark.parametrize("mixer", [CumSum(), SequenceMean(), Roll()], ids=["cumsum", "mean", "roll"])
def test_token_mixing_outside_attention_is_rejected(mixer):
    with pytest.raises(UnsupportedOperationError) as raised:
        analyze(NanoGPT(layers=1, mix=mixer))
    message = str(raised.value)
    assert "over the sequence" in message
    assert "blocks.0.mix" in message
    assert "mixing tokens only in causal attention" in message


def test_bert_is_rejected_as_non_causal():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(2)
    model = transformers.BertForMaskedLM(transformers.BertConfig(
        vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=32,
    ))
    with pytest.raises(UnsupportedModelError, match="isn't causal"):
        analyze(model, 16)


def test_attention_without_a_mask_is_rejected():
    with pytest.raises(UnsupportedModelError, match="isn't causal.*layers.0.*no mask"):
        analyze(RopeLM(layers=1, is_causal=False))


def test_sliding_window_mask_is_rejected():
    class Windowed(RopeLM):
        def forward(self, tokens):
            return super().forward(tokens)

    model = Windowed(layers=1, is_causal=False)
    for layer in model.layers:
        layer.forward = _windowed_forward(layer)
    with pytest.raises(UnsupportedModelError, match="isn't causal.*cannot attend to token 0"):
        analyze(model, 12)


def _windowed_forward(layer, window=3):
    def forward(x, cos, sin):
        batch, length, _ = x.shape
        q = layer.q(x).view(batch, length, layer.heads, layer.dim).transpose(1, 2)
        k = layer.k(x).view(batch, length, layer.kv_heads, layer.dim).transpose(1, 2)
        v = layer.v(x).view(batch, length, layer.kv_heads, layer.dim).transpose(1, 2)
        rows = torch.arange(length)[:, None]
        columns = torch.arange(length)[None]
        allowed = (columns <= rows) & (columns > rows - window)
        out = F.scaled_dot_product_attention(
            q, layer.repeat(k), layer.repeat(v), attn_mask=allowed,
        )
        return layer.o(out.transpose(1, 2).reshape(batch, length, -1))
    return forward


def test_length_dependent_branch_is_rejected():
    class Branching(NanoGPT):
        def forward(self, tokens):
            if tokens.shape[1] >= CONTEXT:
                return super().forward(tokens) * 2
            return super().forward(tokens)

    with pytest.raises(UnsupportedModelError, match="changes with the sequence length") as raised:
        analyze(Branching())
    assert "length-dependent Python" in str(raised.value)


def test_length_dependent_table_is_rejected():
    class Normalised(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = NanoGPT(layers=1)

        def forward(self, tokens):
            # Position features scaled by the length: row i changes with L.
            scale = (torch.arange(tokens.shape[1]) / tokens.shape[1])[None, :, None]
            return self.inner(tokens) + scale

    with pytest.raises(UnsupportedOperationError, match="rows must not depend on the sequence length"):
        analyze(Normalised())


def test_output_without_sequence_axis_is_rejected():
    class LastRow(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = NanoGPT(layers=1)

        def forward(self, tokens):
            return self.inner(tokens).sum(dim=1)

    with pytest.raises(UnsupportedModelError, match="no sequence axis"):
        analyze(LastRow())


def test_max_context_must_leave_room_for_the_second_trace():
    with pytest.raises(UnsupportedModelError, match="max_context >= 3"):
        analyze(NanoGPT(layers=1), 2)


# ── Step module ──────────────────────────────────────────────────────────────

def step_for(model, context):
    """Adapted model, plan and the unchecked step module."""
    adapted = prepare_sequence_model(model.eval())
    plan = analyze_sequence(adapted, context)
    return adapted, plan, build_step_module(adapted, plan)


def step_models():
    return {
        "nanogpt": (lambda: NanoGPT(layers=2), 16),
        "rope_gqa": (lambda: RopeLM(layers=2, heads=4, kv_heads=2), 12),
        "gpt2": (lambda: tiny_decoder("gpt2"), 24),
        "llama": (lambda: tiny_decoder("llama"), 24),
        "qwen2": (lambda: tiny_decoder("qwen2"), 24),
        "mqa_broadcast": (lambda: NanoMQA(layers=2, heads=4), 16),
        "llama_1kv": (lambda: tiny_decoder("llama", heads=4, kv_heads=1), 24),
    }


@pytest.mark.parametrize("name", list(step_models()))
def test_step_matches_the_stateless_model(name):
    make, context = step_models()[name]
    model = make().eval()
    step = build_generation_step(model, context)  # runs the equivalence check
    adapted = prepare_sequence_model(model)
    torch.manual_seed(5)
    tokens = torch.randint(0, VOCAB, (1, context))
    with torch.no_grad():
        expected = adapted(tokens)
        expected = getattr(expected, "logits", expected)
        rows = torch.cat([step(tokens[:, i:i + 1])[0] for i in range(context)], dim=1)
    torch.testing.assert_close(rows, expected, rtol=1e-4, atol=1e-5)
    assert int(step.position) == context
    assert step.max_context == context


def test_broadcast_multi_query_attention_plan_and_cache():
    model = NanoMQA(layers=2, heads=4).eval()
    plan = analyze(model)
    for block in plan.attention:
        assert (block.n_heads, block.n_kv_heads, block.kv_groups) == (4, 1, 4)
    step = build_generation_step(model, CONTEXT)  # runs the equivalence check
    for _, k_cache, v_cache in step.cache_buffers():
        assert k_cache.shape == v_cache.shape == (1, 1, CONTEXT, 4)


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_cache_has_the_kv_heads(kv_heads):
    model = tiny_decoder("llama", heads=4, kv_heads=kv_heads).eval()
    step = build_generation_step(model, 16)
    caches = step.cache_buffers()
    assert len(caches) == 2
    for _, k_cache, v_cache in caches:
        assert k_cache.shape == v_cache.shape == (1, kv_heads, 16, 8)
    assert step.position.dtype == torch.long and step.position.ndim == 0


def test_shared_tables_are_deduplicated():
    step = build_generation_step(tiny_decoder("llama").eval(), 16)
    tables = [name for name, _ in step.named_buffers() if name.startswith("table_")]
    assert len(tables) == 2  # cos and sin, shared by both layers


def test_reset_restores_the_first_step():
    step = build_generation_step(NanoGPT(layers=2).eval(), CONTEXT)
    token = torch.tensor([[7]])
    first = step(token)[0].clone()
    step(torch.tensor([[3]]))
    assert int(step.position) == 2
    step.reset()
    assert int(step.position) == 0
    assert not any(bool(k.any() or v.any()) for _, k, v in step.cache_buffers())
    assert torch.equal(step(token)[0], first)


def test_stepping_past_max_context_raises():
    step = build_generation_step(NanoGPT(layers=1).eval(), 8)
    for _ in range(8):
        step(torch.tensor([[1]]))
    with pytest.raises(RuntimeError, match="step module is full.*position 8.*reset"):
        step(torch.tensor([[1]]))
    step.reset()
    step(torch.tensor([[1]]))  # usable again


def test_broken_table_trips_the_equivalence_check():
    adapted, plan, step = step_for(NanoGPT(layers=2), CONTEXT)
    check_step_equivalence(adapted, step, plan)
    with torch.no_grad():
        step.table_0[5, 0] += 5.0
    with pytest.raises(UnsupportedModelError, match="disagrees with the model at output row 5") as raised:
        check_step_equivalence(adapted, step, plan)
    assert "max error" in str(raised.value)


def test_broken_attention_scale_trips_the_equivalence_check():
    adapted, plan, step = step_for(tiny_decoder("llama"), 16)
    for node in step.graph.nodes:
        if "cached_attention" in str(node.target):
            node.args = (*node.args[:6], node.args[6] * 3, node.args[7])
    step.recompile()
    with pytest.raises(UnsupportedModelError, match="disagrees with the model at output row"):
        check_step_equivalence(adapted, step, plan)


def test_broken_cache_op_trips_the_equivalence_check(monkeypatch):
    adapted, plan, step = step_for(NanoGPT(layers=1), CONTEXT)
    original = torch.ops.cattorch.cached_attention.default

    def forgetful(q, k, v, k_cache, v_cache, position, scale, groups):
        # Attends over the new row only, as if the cache were never read.
        return original(q, k, v, torch.zeros_like(k_cache), torch.zeros_like(v_cache),
                        position, scale, groups)
    for node in step.graph.nodes:
        if node.target is original:
            node.target = forgetful
    step.recompile()
    with pytest.raises(UnsupportedModelError, match="disagrees with the model"):
        check_step_equivalence(adapted, step, plan)


@pytest.mark.parametrize("name", ["nanogpt", "llama"])
def test_step_module_traces_with_custom_ops_and_no_mask(name):
    make, context = step_models()[name]
    step = build_generation_step(make().eval(), context)
    traceable = copy.deepcopy(step)
    traced = capture_model(traceable, (torch.zeros(1, 1, dtype=torch.long),))
    # Cache and position buffers reach the trace as the module's own tensors
    # (by identity), though tracing itself ran the step once.
    state = [t for t in traced.state_inputs.values()]
    for _, k_cache, v_cache in traceable.cache_buffers():
        assert any(t is k_cache for t in state) and any(t is v_cache for t in state)
    assert any(t is traceable.position for t in state)
    targets = [str(n.target) for n in traced.graph.nodes if n.op == "call_function"]
    assert targets.count("cattorch.cached_attention.default") == 2
    assert targets.count("cattorch.advance_position.default") == 1
    assert "cattorch.sequence_row.default" in targets
    assert not any(t.startswith("aten.softmax") or "masked_fill" in t for t in targets)
    for node in traced.graph.nodes:
        value = node.meta.get("val")
        if node.op == "call_function" and isinstance(value, torch.Tensor):
            assert sum(size == context for size in value.shape) < 2, (node.name, value.shape)
