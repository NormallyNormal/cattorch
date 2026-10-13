"""Models written in ordinary PyTorch, without cattorch-specific rewrites.

Each case exports a module as users typically write it and checks the sprite
against eager PyTorch in the emulator. ``KNOWN_UNSUPPORTED`` records what still
fails; those cases are strict xfails, so fixing one requires moving it to the
supported list and the matrix can only improve.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cattorch import CattorchError, UnsupportedModelError, StorageConfig, transpile, verify
from cattorch.graph import _prepare_graph


class Lambda(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, *args):
        return self.function(*args)


def _export_and_verify(tmp_path, model, inputs, *, atol=1e-4, rtol=1e-5):
    model = model.eval()
    result = transpile(
        model, inputs, tmp_path / "model", storage=StorageConfig(compression=False),
    )
    report = verify(model, inputs, result, atol=atol, rtol=rtol)
    assert report.passed, report
    return result


VECTOR = torch.linspace(-3.0, 3.0, 16).reshape(2, 8)
POSITIVE = torch.linspace(0.25, 4.0, 16).reshape(2, 8)

POINTWISE = {
    "exp": (torch.exp, VECTOR),
    "log": (torch.log, POSITIVE),
    "log10": (torch.log10, POSITIVE),
    "sqrt": (torch.sqrt, POSITIVE),
    "abs": (torch.abs, VECTOR),
    "floor": (torch.floor, VECTOR),
    "ceil": (torch.ceil, VECTOR),
    "reciprocal": (torch.reciprocal, POSITIVE),
    "erf": (torch.erf, VECTOR),
    "exact gelu": (F.gelu, VECTOR),
    "clamp": (lambda x: x.clamp(-1.0, 0.5), VECTOR),
    "clamp min": (lambda x: x.clamp(min=-1.0), VECTOR),
    "clamp max": (lambda x: x.clamp(max=0.5), VECTOR),
    "hardtanh": (F.hardtanh, VECTOR),
    "relu6": (F.relu6, VECTOR),
    "pow 3": (lambda x: x ** 3, VECTOR),
    "pow -2": (lambda x: x ** -2, POSITIVE),
    "pow 0.5": (lambda x: x ** 0.5, POSITIVE),
    "greater than": (lambda x: (x > 0.5).float(), VECTOR),
    "less or equal": (lambda x: (x <= 0.0).float() * x, VECTOR),
    "not equal": (lambda x: (x.floor() != 1).float(), VECTOR),
    "logical not": (lambda x: (~(x > 0)).float(), VECTOR),
    "where": (lambda x: torch.where(x > 0, x, 0.1 * x), VECTOR),
    "where scalar": (lambda x: torch.where(x > 0, x, 0.0), VECTOR),
    "maximum": (lambda x: torch.maximum(x, x * -0.5), VECTOR),
    "minimum": (lambda x: torch.minimum(x, x * 2), VECTOR),
    "log softmax": (lambda x: F.log_softmax(x, dim=-1), VECTOR),
    "log softmax first dim": (lambda x: F.log_softmax(x, dim=0), VECTOR),
    "truncate to integer": (lambda x: x.long(), VECTOR),
    "float to bool": (lambda x: x.floor().bool(), VECTOR),
}


@pytest.mark.parametrize("name", sorted(POINTWISE))
def test_pointwise_operation(tmp_path, name):
    function, value = POINTWISE[name]
    _export_and_verify(tmp_path, Lambda(function), (value,), atol=1e-6)


def test_integer_input_can_be_cast_to_float(tmp_path):
    tokens = torch.tensor([[1, 2, 3, 7]])
    _export_and_verify(tmp_path, Lambda(lambda x: x.float() * 0.5 + 1), (tokens,))


def test_exact_gelu_after_linear_is_not_approximated(tmp_path):
    """The fused linear epilogue only has the tanh form, so it must not fuse."""
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4))
    value = torch.randn(3, 8) * 3
    # The tanh approximation differs from exact GELU by up to about 5e-4.
    _export_and_verify(tmp_path, model, (value,), atol=2e-6)


def test_static_buffer_comparisons_fold(tmp_path):
    """nanoGPT builds its causal mask with ``bias == 0`` on a buffer."""
    torch.manual_seed(5)
    length, width, heads = 5, 8, 2

    class CausalSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = nn.Linear(width, 3 * width)
            self.c_proj = nn.Linear(width, width)
            self.register_buffer(
                "bias", torch.tril(torch.ones(length, length)).view(1, 1, length, length),
            )

        def forward(self, x):
            batch, steps, channels = x.size()
            q, k, v = self.c_attn(x).split(width, dim=2)
            q = q.view(batch, steps, heads, channels // heads).transpose(1, 2)
            k = k.view(batch, steps, heads, channels // heads).transpose(1, 2)
            v = v.view(batch, steps, heads, channels // heads).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :steps, :steps] == 0, float("-inf"))
            y = F.softmax(att, dim=-1) @ v
            y = y.transpose(1, 2).contiguous().view(batch, steps, channels)
            return self.c_proj(y)

    _export_and_verify(tmp_path, CausalSelfAttention(), (torch.randn(1, length, width),))


def test_rope_tables_computed_in_forward_fold(tmp_path):
    """Llama-style code builds cos/sin tables from arange and outer each call."""
    torch.manual_seed(6)
    width = 8

    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    class Rotary(nn.Module):
        def __init__(self):
            super().__init__()
            inverse = 1.0 / (10000 ** (torch.arange(0, width, 2).float() / width))
            self.register_buffer("inverse", inverse)

        def forward(self, x):
            positions = torch.arange(x.shape[-2], device=x.device).float()
            frequencies = torch.outer(positions, self.inverse)
            embedding = torch.cat((frequencies, frequencies), dim=-1)
            return x * embedding.cos() + rotate_half(x) * embedding.sin()

    _export_and_verify(tmp_path, Rotary(), (torch.randn(1, 2, 5, width),))


def test_torch_multihead_attention_with_weights(tmp_path):
    torch.manual_seed(7)
    value = torch.randn(1, 4, 8)
    model = nn.MultiheadAttention(8, 2, batch_first=True)
    _export_and_verify(tmp_path, model, (value, value, value))


def test_torch_transformer_encoder_classifier(tmp_path):
    torch.manual_seed(8)

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(20, 8)
            layer = nn.TransformerEncoderLayer(8, 2, 16, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
            self.head = nn.Linear(8, 3)

        def forward(self, tokens):
            return self.head(self.encoder(self.embedding(tokens)).mean(1))

    _export_and_verify(tmp_path, Classifier(), (torch.tensor([[1, 5, 2, 9, 3]]),))


def test_mnist_style_cnn_with_log_softmax(tmp_path):
    torch.manual_seed(9)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 2, 3)
            self.fc = nn.Linear(2 * 4 * 4, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv(x)), 2)
            return F.log_softmax(self.fc(torch.flatten(x, 1)), dim=1)

    _export_and_verify(tmp_path, Net(), (torch.randn(1, 1, 10, 10),))


def test_list_stays_live_while_read_through_a_view(tmp_path):
    """A view's later reads keep its source list from being reused."""

    def forward(scores, values):
        probabilities = torch.softmax(scores, -1)
        mixed = (probabilities @ values).transpose(0, 1) * 2 + 1
        return mixed, probabilities.view(1, 2, 4, 4).mean(1)

    torch.manual_seed(10)
    inputs = (torch.randn(2, 4, 4), torch.randn(2, 4, 4))
    _export_and_verify(tmp_path, Lambda(forward), inputs)


def test_output_list_is_not_reused_by_later_operations(tmp_path):
    def forward(scores, values):
        probabilities = torch.softmax(scores, -1)
        mixed = (probabilities @ values) * 3
        return probabilities, mixed + 1, mixed.transpose(0, 1) * 2

    torch.manual_seed(12)
    inputs = (torch.randn(2, 4, 4), torch.randn(2, 4, 4))
    _export_and_verify(tmp_path, Lambda(forward), inputs)


def test_huggingface_bert_classifier(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(11)
    config = transformers.BertConfig(
        vocab_size=32, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, max_position_embeddings=8,
    )
    model = transformers.BertForSequenceClassification(config).eval()

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, tokens):
            return self.model(input_ids=tokens).logits

    _export_and_verify(tmp_path, Wrapper(), (torch.tensor([[2, 7, 4, 3]]),))


class StaticBranch(nn.Module):
    def forward(self, x):
        # Depends only on a shape, so the branch is fixed while tracing.
        if (torch.arange(x.shape[-1]) == 0).any():
            return x * 2.0
        return x + 1.0


class NoGradInside(nn.Module):
    def forward(self, x):
        with torch.no_grad():
            scale = x.new_ones(1) * 3.0
        return x * scale


class CatEmpty(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("past", torch.zeros(2, 0))

    def forward(self, x):
        return torch.cat([self.past, x, torch.empty(2, 0)], dim=1) * 2.0


def test_static_item_branch_exports(tmp_path):
    _export_and_verify(tmp_path, StaticBranch(), (VECTOR,))


def test_input_dependent_item_still_rejected(tmp_path):
    for function in (lambda x: x * x.sum().item(), lambda x: x * 2 if x.sum() > 0 else x):
        with pytest.raises(UnsupportedModelError):
            transpile(Lambda(function).eval(), (VECTOR,), tmp_path / "model")


def test_no_grad_inside_forward_exports(tmp_path):
    _export_and_verify(tmp_path, NoGradInside(), (VECTOR,))


def test_cat_with_empty_tensor_exports(tmp_path):
    _export_and_verify(tmp_path, CatEmpty(), (VECTOR,))


def _tiny_decoder(kind):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(5)
    common = dict(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=64,
        max_position_embeddings=16,
    )
    if kind == "gpt2":
        return transformers.GPT2LMHeadModel(transformers.GPT2Config(
            n_embd=32, n_layer=1, n_head=2, vocab_size=64, n_positions=16,
        ))
    if kind == "llama":
        return transformers.LlamaForCausalLM(transformers.LlamaConfig(**common))
    return transformers.Qwen2ForCausalLM(transformers.Qwen2Config(**common))


class LogitsOnly(nn.Module):
    def __init__(self, model, use_cache):
        super().__init__()
        self.model = model
        self.use_cache = use_cache

    def forward(self, tokens):
        return self.model(tokens, use_cache=self.use_cache).logits


@pytest.mark.parametrize("kind", ["gpt2", "llama", "qwen2"])
def test_huggingface_decoder_logits(tmp_path, kind):
    wrapper = LogitsOnly(_tiny_decoder(kind), use_cache=False)
    _export_and_verify(tmp_path, wrapper, (torch.randint(0, 64, (1, 8)),))


def test_huggingface_gpt2_with_cache_dropped(tmp_path):
    wrapper = LogitsOnly(_tiny_decoder("gpt2"), use_cache=True)
    _export_and_verify(tmp_path, wrapper, (torch.randint(0, 64, (1, 8)),))


@pytest.mark.parametrize("kind", ["gpt2", "llama"])
@pytest.mark.parametrize("cache", ["off", "left on"])
def test_unwrapped_huggingface_decoder(tmp_path, kind, cache):
    model = _tiny_decoder(kind).eval()
    ids = torch.randint(0, 64, (1, 8))
    kwargs = {"use_cache": False} if cache == "off" else None
    if kwargs is None:
        with pytest.warns(UserWarning, match="past_key_values"):
            result = transpile(model, (ids,), tmp_path / "model", example_kwargs=kwargs)
    else:
        result = transpile(model, (ids,), tmp_path / "model", example_kwargs=kwargs)
    assert verify(model, (ids,), result, example_kwargs=kwargs).passed


def test_huggingface_model_in_training_mode_warns_and_stays_training(tmp_path):
    model = _tiny_decoder("gpt2")
    assert model.training
    ids = torch.randint(0, 64, (1, 8))
    kwargs = {"use_cache": False}
    with pytest.warns(UserWarning, match="training mode"):
        result = transpile(model, (ids,), tmp_path / "model", example_kwargs=kwargs)
    with pytest.warns(UserWarning, match="training mode"):
        assert verify(model, (ids,), result, example_kwargs=kwargs).passed
    assert model.training


class KeywordMask(nn.Module):
    def forward(self, x, *, mask, scale=1.0, label=None):
        return x * mask * scale


def test_tensor_keyword_becomes_next_input(tmp_path):
    x, mask = torch.randn(2, 4), torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    kwargs = {"mask": mask, "scale": 2.0, "label": "ignored"}
    model = KeywordMask().eval()
    result = transpile(model, (x,), tmp_path / "model", example_kwargs=kwargs)
    assert [spec.list_name for spec in result.inputs] == ["input", "input_1"]
    assert tuple(result.inputs[1].shape) == (1, 4)
    assert verify(model, (x,), result, example_kwargs=kwargs).passed


def test_tensor_keyword_with_export_frontend(tmp_path):
    x, mask = torch.randn(2, 4), torch.ones(1, 4)
    kwargs = {"mask": mask}
    model = KeywordMask().eval()
    result = transpile(
        model, (x,), tmp_path / "model", example_kwargs=kwargs, frontend="export",
    )
    assert verify(model, (x,), result, example_kwargs=kwargs).passed


class DictOutputs(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, x):
        if self.kind == "dict":
            return {"logits": x * 2.0, "cache": None, "count": 3, "extra": (x + 1.0, None)}
        transformers = pytest.importorskip("transformers")
        return transformers.modeling_outputs.CausalLMOutputWithPast(
            logits=x * 2.0, past_key_values=transformers.DynamicCache(),
            hidden_states=(x + 1.0,),
        )


@pytest.mark.parametrize("kind", ["dict", "model output"])
def test_non_tensor_output_leaves_are_dropped_with_a_warning(tmp_path, kind):
    model = DictOutputs(kind).eval()
    with pytest.warns(UserWarning, match="ignoring non-tensor model outputs") as record:
        result = transpile(model, (VECTOR,), tmp_path / "model")
    assert len(result.outputs) == 2
    text = str(record[0].message)
    assert ("output['cache'] (NoneType)" in text) if kind == "dict" else (
        "output['past_key_values'] (DynamicCache)" in text
    )
    assert verify(model, (VECTOR,), result).passed


def test_output_without_tensors_is_rejected(tmp_path):
    with pytest.raises(UnsupportedModelError, match="at least one tensor output"):
        transpile(Lambda(lambda x: {"a": None, "b": 1}).eval(), (VECTOR,), tmp_path / "model")


class WithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return self.dropout(self.linear(x))


def test_training_mode_model_exports_in_evaluation_mode(tmp_path):
    torch.manual_seed(3)
    model = WithDropout().train()
    with pytest.warns(UserWarning) as record:
        result = transpile(model, (VECTOR,), tmp_path / "model")
    assert [str(w.message).count("evaluation mode") for w in record] == [1]
    assert model.training and model.dropout.training
    with pytest.warns(UserWarning, match="training mode"):
        assert verify(model, (VECTOR,), result).passed
    assert model.training


def test_example_kwargs_rejections(tmp_path):
    from cattorch import QuantizationConfig
    from cattorch.experimental import (
        EntryPoint, ExportProgram, GenerationProgram, Input, Output,
    )
    model = KeywordMask().eval()
    x, mask = torch.randn(2, 4), torch.ones(1, 4)
    program = ExportProgram(entrypoints=(
        EntryPoint("run", arguments=(Input("x", x),), returns=(Output("y"),)),
    ))
    with pytest.raises(ValueError, match="ExportProgram or GenerationProgram"):
        transpile(model, program, tmp_path / "a", example_kwargs={"mask": mask})
    generation = GenerationProgram(
        "forward", torch.zeros(1, 1, dtype=torch.long), max_context=4,
    )
    with pytest.raises(ValueError, match="ExportProgram or GenerationProgram"):
        transpile(model, generation, tmp_path / "b", example_kwargs={"mask": mask})
    with pytest.raises(TypeError, match="dict with string keys"):
        transpile(model, (x,), tmp_path / "c", example_kwargs=[("mask", mask)])
    with pytest.raises(TypeError, match="dict with string keys"):
        transpile(model, (x,), tmp_path / "c", example_kwargs={1: mask})
    with pytest.raises(ValueError, match="GPTQ"):
        transpile(
            WithDropout().eval(), (VECTOR,), tmp_path / "d",
            quantization=QuantizationConfig(bits=4, method="gptq"),
            calibration_inputs=[VECTOR],
            example_kwargs={"mask": mask},
        )
    result = transpile(model, (x,), tmp_path / "e", example_kwargs={"mask": mask})
    with pytest.raises(ValueError, match="ExportProgram"):
        verify(model, program, result, example_kwargs={"mask": mask})


IMAGE = torch.randn(1, 4, 6, 6)

# Operations with no dedicated kernel that export through PyTorch's
# decompositions and the generic reduction, index-map, and top-k kernels.
DECOMPOSED = {
    "sum": (Lambda(lambda x: x.sum(-1)), (VECTOR,)),
    "sum keepdim over first dim": (Lambda(lambda x: x.sum(0, keepdim=True)), (VECTOR,)),
    "mean of all elements": (Lambda(lambda x: x.mean()), (VECTOR,)),
    "amax": (Lambda(lambda x: x.amax(-1)), (VECTOR,)),
    "amin over first dim": (Lambda(lambda x: x.amin(0)), (VECTOR,)),
    "max with indices": (Lambda(lambda x: torch.max(x, -1).values), (VECTOR,)),
    "argmax": (Lambda(lambda x: x.argmax(-1)), (VECTOR,)),
    "argmin over first dim": (Lambda(lambda x: x.argmin(0)), (VECTOR,)),
    "prod": (Lambda(lambda x: (x * 0.5).prod(-1)), (VECTOR,)),
    "any": (Lambda(lambda x: (x > 2).any(-1).float()), (VECTOR,)),
    "var": (Lambda(lambda x: x.var(-1)), (VECTOR,)),
    "std": (Lambda(lambda x: x.std(0)), (VECTOR,)),
    "cumsum": (Lambda(lambda x: x.cumsum(-1)), (VECTOR,)),
    "expand": (Lambda(lambda x: x[:, :1].expand(2, 8) + x), (VECTOR,)),
    "pad": (Lambda(lambda x: F.pad(x, (1, 2), value=-1.5)), (VECTOR,)),
    "reflection pad": (Lambda(lambda x: F.pad(x, (1, 1, 1, 1), mode="reflect")), (IMAGE,)),
    "flip": (Lambda(lambda x: x.flip(-1)), (VECTOR,)),
    "repeat": (Lambda(lambda x: x.repeat(2, 1)), (VECTOR,)),
    "roll": (Lambda(lambda x: x.roll(3, -1)), (VECTOR,)),
    "tril": (Lambda(lambda x: x.reshape(4, 4).tril()), (VECTOR,)),
    "sin": (Lambda(torch.sin), (VECTOR,)),
    "cos": (Lambda(torch.cos), (VECTOR,)),
    "softplus": (nn.Softplus(), (VECTOR,)),
    "mish": (nn.Mish(), (VECTOR,)),
    "hardswish": (nn.Hardswish(), (VECTOR,)),
    "normalize": (Lambda(lambda x: F.normalize(x, dim=-1)), (VECTOR,)),
    "gather": (
        Lambda(lambda x: x.gather(-1, torch.tensor([[2], [5]]))), (VECTOR,),
    ),
    "index select": (
        Lambda(lambda x: x.index_select(-1, torch.tensor([7, 0, 3]))), (VECTOR,),
    ),
    "topk values": (Lambda(lambda x: x.topk(3, -1).values), (VECTOR,)),
    "topk indices": (Lambda(lambda x: x.topk(3, -1).indices), (VECTOR,)),
    "topk smallest over first dim": (
        Lambda(lambda x: x.topk(1, 0, largest=False).values), (VECTOR,),
    ),
    "group norm": (nn.GroupNorm(2, 4), (IMAGE,)),
    "instance norm": (nn.InstanceNorm2d(4), (IMAGE,)),
    "nearest upsampling": (nn.Upsample(scale_factor=2), (IMAGE,)),
    "adaptive max pooling": (nn.AdaptiveMaxPool2d(1), (IMAGE,)),
    "masked attention": (
        Lambda(lambda q: F.scaled_dot_product_attention(
            q, q, q, attn_mask=torch.tril(torch.ones(4, 4, dtype=torch.bool)),
        )),
        (torch.randn(1, 2, 4, 8),),
    ),
    "lstm": (nn.LSTM(4, 4, batch_first=True), (torch.randn(1, 3, 4),)),
    "gru": (nn.GRU(4, 4, batch_first=True), (torch.randn(1, 3, 4),)),
}


@pytest.mark.parametrize("name", sorted(DECOMPOSED))
def test_decomposed_operation(tmp_path, name):
    torch.manual_seed(13)
    model, inputs = DECOMPOSED[name]
    _export_and_verify(tmp_path, model, inputs, atol=1e-5)


KNOWN_UNSUPPORTED = {
    "depthwise convolution": (nn.Conv2d(4, 4, 3, groups=4), (IMAGE,)),
    "dilated convolution": (nn.Conv2d(4, 4, 3, dilation=2), (IMAGE,)),
    "transposed convolution": (nn.ConvTranspose2d(4, 4, 2, stride=2), (IMAGE,)),
    "3d convolution": (nn.Conv3d(4, 4, 3), (torch.randn(1, 4, 4, 4, 4),)),
    "log gamma": (Lambda(torch.lgamma), (POSITIVE,)),
}


@pytest.mark.parametrize("name", sorted(KNOWN_UNSUPPORTED))
@pytest.mark.xfail(raises=CattorchError, strict=True)
def test_known_unsupported(name):
    model, inputs = KNOWN_UNSUPPORTED[name]
    _prepare_graph(model.eval(), inputs)
