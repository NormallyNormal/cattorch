"""Regressions found by adversarial bug hunts against the public API."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from cattorch import FastConfig, FastLayerConfig, transpile, verify
from cattorch.experimental import ExpertFamily, SparseMoE


def _check(model, inputs, path, **options):
    model = model.eval()
    inputs = inputs if isinstance(inputs, tuple) else (inputs,)
    result = transpile(model, inputs, path, **options)
    report = verify(model, inputs, result)
    assert report.passed, report
    return result


class CombinedQKV(nn.Module):
    """nanoGPT-style attention: one projection split into Q, K and V."""

    def __init__(self, extra=None, sequence=16, width=16, heads=4):
        super().__init__()
        self.extra, self.heads, self.width = extra, heads, width
        self.c_attn = nn.Linear(width, 3 * width)
        self.register_buffer("mask", torch.triu(torch.ones(sequence, sequence, dtype=torch.bool), 1))

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.c_attn(x).split(width, dim=2)
        q, k, v = (
            value.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
            for value in (q, k, v)
        )
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        scores = scores.masked_fill(self.mask, float("-inf"))
        y = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, length, width)
        if self.extra == "kv":
            return y, k, v
        if self.extra == "scaled_k":
            return y, k * 2
        return y


@pytest.mark.parametrize(("extra", "batch"), [(None, 2), ("kv", 1), ("scaled_k", 1)])
def test_combined_qkv_attention_with_batches_or_other_readers(tmp_path, extra, batch):
    torch.manual_seed(0)
    _check(CombinedQKV(extra), torch.randn(batch, 16, 16), tmp_path / "attention")


class Tied(nn.Module):
    def __init__(self, vocabulary=256, width=192, index=False):
        super().__init__()
        self.index = index
        self.embedding = nn.Embedding(vocabulary, width)
        self.head = nn.Linear(width, vocabulary, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, tokens):
        logits = self.head(self.embedding(tokens))
        return (logits, self.embedding.weight[tokens]) if self.index else logits


def test_tied_table_read_by_a_third_operation(tmp_path):
    torch.manual_seed(0)
    _check(Tied(index=True), torch.tensor([[7, 3]]), tmp_path / "tied")


def test_tied_table_with_fast_weights_that_change_nothing(tmp_path):
    torch.manual_seed(0)
    fast = FastConfig(weights=FastLayerConfig(rank=100_000))
    _check(Tied(), torch.tensor([[7, 3]]), tmp_path / "tied", optimization="fast", fast_config=fast)


def test_fast_pruning_of_a_matmul_leaves_other_readers_alone(tmp_path):
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.randn(32, 32))

        def forward(self, tokens, x):
            return x @ self.weight, F.embedding(tokens, self.weight)

    torch.manual_seed(0)
    model = Shared().eval()
    inputs = (torch.tensor([1, 5]), torch.randn(2, 32))
    fast = FastConfig(weights=FastLayerConfig(pruning=0.5))
    result = transpile(model, inputs, tmp_path / "shared", optimization="fast", fast_config=fast)
    report = verify(model, inputs, result)
    assert report.outputs[1].max_abs_error == 0


def test_residual_add_that_broadcasts_the_linear_output(tmp_path):
    class Broadcast(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 8)

        def forward(self, x, residual):
            return self.linear(x) + residual

    torch.manual_seed(0)
    _check(Broadcast(), (torch.randn(1, 16), torch.randn(4, 8)), tmp_path / "residual")


class ParameterExpert(nn.Module):
    """Operations on the expert's own parameters, and a computed weight."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.gate = nn.Parameter(torch.randn(4))
        self.weight = nn.Parameter(torch.randn(4, 4))

    def forward(self, x):
        gated = self.linear(x) * torch.sigmoid(self.gate) + self.gate[:4].clamp(-0.5, 0.5) ** 2
        return gated + F.linear(x, self.weight.t())


class Fixed(nn.Module):
    """Routes every token to one expert."""

    def __init__(self, expert):
        super().__init__()
        self.expert = expert

    def forward(self, x):
        scores = torch.zeros(3)
        scores[self.expert] = 1.0
        return x.new_tensor(scores.tolist()).expand(*x.shape[:-1], 3)


@pytest.mark.parametrize("expert", [1, 2])
def test_expert_operations_on_parameters_read_the_chosen_expert(tmp_path, expert):
    torch.manual_seed(0)
    family = ExpertFamily.from_modules(
        [ParameterExpert() for _ in range(3)], example_input=torch.ones(1, 4),
    )
    _check(SparseMoE(Fixed(expert), family, top_k=1), torch.randn(5, 4), tmp_path / "moe")


def test_boolean_add_is_a_logical_or(tmp_path):
    class Either(nn.Module):
        def forward(self, x):
            return ((x > 0) + (x > 1)).float(), ((x > 0) + True).float()

    _check(Either(), torch.tensor([[-1.0, 0.5, 2.0]]), tmp_path / "or")


@pytest.mark.parametrize("exponent", [5, -7, 13])
def test_integer_powers_are_exact(tmp_path, exponent):
    class Power(nn.Module):
        def forward(self, x):
            return x ** exponent

    model = Power().eval()
    value = torch.tensor([[3.0, -2.0, 6.0, 1.5]])
    result = transpile(model, value, tmp_path / "power")
    report = verify(model, value, result, atol=0, rtol=1e-6)
    assert report.passed, report


@pytest.mark.parametrize(("weight", "bias"), [((5,), None), ((3, 5), (1,)), ((3, 5), ())])
def test_functional_linear_with_a_vector_weight_or_broadcast_bias(tmp_path, weight, bias):
    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(weight))
            self.bias = None if bias is None else nn.Parameter(torch.randn(bias))

        def forward(self, x):
            return F.linear(x, self.weight, self.bias)

    torch.manual_seed(0)
    _check(Linear(), torch.randn(2, 5), tmp_path / "linear")


# ── Kernels passed to an export ──────────────────────────────────────────────

from cattorch.experimental import Kernel  # noqa: E402
from cattorch.scratch import append, clear, repeat  # noqa: E402


def _sevens(target, calls, **options):
    """A kernel that fills its output with 7s, so using it makes verify fail."""
    def emit(ctx):
        calls.append(ctx)
        return (clear(ctx.output.list), repeat(ctx.output.numel, (append(ctx.output.list, 7),)))
    return Kernel(target, emit, **options)


class Lambda(nn.Module):
    def __init__(self, function, **modules):
        super().__init__()
        self.function = function
        self.parts = nn.ModuleDict(modules)

    def forward(self, *values):
        return self.function(self.parts, *values)


def _causal(parts, value):
    scores = parts["q"](value) @ parts["k"](value).transpose(-1, -2) / math.sqrt(8)
    length = value.shape[-2]
    scores = scores.masked_fill(torch.triu(torch.ones(length, length, dtype=torch.bool), 1), float("-inf"))
    return torch.softmax(scores, -1) @ parts["v"](value)


class Rotary(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cos", torch.randn(3, 4))
        self.register_buffer("sin", torch.randn(3, 4))

    def forward(self, value):
        return value * self.cos + torch.cat((-value[..., 2:], value[..., :2]), -1) * self.sin


_X = torch.randn(2, 4)
_FUSED = {
    "linear epilogue": (
        lambda: nn.Sequential(nn.Linear(4, 4), nn.ReLU()), (_X,), torch.ops.aten.linear.default),
    "activation epilogue": (
        lambda: nn.Sequential(nn.Linear(4, 4), nn.ReLU()), (_X,), torch.ops.aten.relu.default),
    "residual": (
        lambda: Lambda(lambda parts, x: x + parts["linear"](x), linear=nn.Linear(4, 4)),
        (_X,), torch.ops.aten.linear.default),
    "arithmetic chain": (
        lambda: Lambda(lambda parts, x: torch.relu(x * 2 + 1)), (_X,), torch.ops.aten.mul.Tensor),
    "silu times": (
        lambda: Lambda(lambda parts, a, b: F.silu(a) * b), (_X, _X.clone()), torch.ops.aten.silu.default),
    "embedding add": (
        lambda: Lambda(
            lambda parts, t: parts["token"](t) + parts["position"](torch.arange(t.shape[-1])),
            token=nn.Embedding(10, 4), position=nn.Embedding(6, 4),
        ),
        (torch.tensor([[1, 2, 3]]),), torch.ops.aten.embedding.default),
    "causal softmax": (
        lambda: Lambda(_causal, q=nn.Linear(8, 8), k=nn.Linear(8, 8), v=nn.Linear(8, 8)),
        (torch.randn(1, 4, 8),), torch.ops.aten.softmax.int),
    "causal matmul": (
        lambda: Lambda(_causal, q=nn.Linear(8, 8), k=nn.Linear(8, 8), v=nn.Linear(8, 8)),
        (torch.randn(1, 4, 8),), torch.ops.aten.matmul.default),
    "rotary": (Rotary, (torch.randn(3, 4),), torch.ops.aten.mul.Tensor),
}


@pytest.mark.parametrize("case", sorted(_FUSED))
def test_a_fusion_never_skips_a_passed_kernel(tmp_path, case):
    make, inputs, target = _FUSED[case]
    model = make().eval()
    calls = []
    result = transpile(model, inputs, tmp_path / "fused", kernels=[_sevens(target, calls)])
    assert calls
    assert not verify(model, inputs, result).passed


@pytest.mark.parametrize("target", ["aten.gelu.default", "aten.layer_norm.default"])
def test_an_exact_kernel_replaces_the_approximate_one_in_fast_mode(tmp_path, target):
    model = nn.Sequential(nn.LayerNorm(4), nn.GELU()).eval()
    calls = []
    transpile(model, _X, tmp_path / "fast", optimization="fast", kernels=[_sevens(target, calls)])
    assert calls


def _plain_linear(ctx):
    """``x @ weight.T``, reading the weight in PyTorch's row order."""
    from cattorch.scratch import add, change_var, item, mul, set_var, var

    source, weight = ctx.inputs["input"], ctx.inputs["weight"]
    columns, inner = weight.shape
    row_start, weight_index, index, total = (
        ctx.temp_var(name) for name in ("row start", "weight index", "index", "total")
    )
    out = ctx.output.list
    return (
        clear(out),
        set_var(row_start, 0),
        repeat(source.numel // inner, (
            set_var(weight_index, 1),
            repeat(columns, (
                set_var(total, 0),
                set_var(index, 1),
                repeat(inner, (
                    change_var(total, mul(
                        item(source.list, add(var(row_start), var(index))),
                        item(weight.list, var(weight_index)),
                    )),
                    change_var(index, 1),
                    change_var(weight_index, 1),
                )),
                append(out, var(total)),
            )),
            change_var(row_start, inner),
        )),
    )


@pytest.mark.parametrize("target", ["embedding", "linear"])
def test_a_correct_kernel_on_a_tied_table(tmp_path, target):
    from cattorch.operator_registry import default_registry

    if target == "embedding":
        # The built-in kernel, passed as the user's own.
        builtin = default_registry().kernel(
            "aten.embedding.default", fast=False, config=FastConfig(),
        ).kernel
        kernel = Kernel(torch.ops.aten.embedding.default, builtin.emit)
    else:
        kernel = Kernel(torch.ops.aten.linear.default, _plain_linear)
    torch.manual_seed(0)
    model = Tied().eval()
    tokens = torch.tensor([[7, 3]])
    result = transpile(model, tokens, tmp_path / "tied", kernels=[kernel])
    assert verify(model, tokens, result).passed


def test_a_kernel_on_a_variable_length_input(tmp_path):
    from cattorch import UnsupportedModelError

    softmax = Lambda(lambda parts, x: torch.softmax(x, -1)).eval()
    calls = []
    builtin = Kernel(torch.ops.aten.softmax.int, _softmax_rows(calls))
    transpile(softmax, torch.randn(3, 4), tmp_path / "softmax", kernels=[builtin],
              dynamic={0: (1, 6)})
    # It runs once per item, so it sees one item.
    assert calls and all(ctx.inputs["self"].shape[0] == 1 for ctx in calls)

    total = Lambda(lambda parts, x: x.sum(0)).eval()
    with pytest.raises(UnsupportedModelError, match="can't be split into items"):
        transpile(total, torch.randn(3, 4), tmp_path / "sum", dynamic={0: (1, 6)},
                  kernels=[_sevens(torch.ops.aten.sum.dim_IntList, [])])


def _softmax_rows(calls):
    from cattorch.util.instruction.optimized_normalization import SOFTMAX_KERNELS

    def emit(ctx):
        calls.append(ctx)
        return SOFTMAX_KERNELS[0].emit(ctx)
    return emit


def test_a_validator_sees_what_emit_sees(tmp_path):
    seen = []

    def validator(ctx):
        seen.append((ctx.inputs["self"].dynamic_axis, ctx.inputs["self"].static))

    from cattorch.util.instruction.optimized_elementwise import elementwise_kernel

    model = Lambda(lambda parts, x: torch.relu(x)).eval()
    relu = elementwise_kernel("aten.relu.default").emit
    transpile(model, torch.randn(3, 4), tmp_path / "relu", dynamic={0: (1, 6)},
              kernels=[Kernel(torch.ops.aten.relu.default, relu, validator=validator)])
    assert seen and all(axis == 0 for axis, _ in seen)


def test_kernel_targets_must_name_one_tensor_operation():
    with pytest.raises(ValueError, match="aten.add.Tensor"):
        Kernel(torch.ops.aten.add, lambda ctx: ())
    with pytest.raises(TypeError, match="operation"):
        Kernel(torch.relu, lambda ctx: ())


def test_an_unused_kernel_warns(tmp_path):
    with pytest.warns(UserWarning, match="aten.gelu.default was not used"):
        transpile(nn.ReLU().eval(), _X, tmp_path / "relu",
                  kernels=[_sevens(torch.ops.aten.gelu.default, [])])


# ── Crashes and misleading errors ────────────────────────────────────────────

_DYNAMIC = {
    "add itself": lambda parts, a: a + a,
    "compare a column": lambda parts, a: (a[:, :1] > 0).float(),
    "silu times": lambda parts, a: F.silu(a) * a,
}


@pytest.mark.parametrize("case", sorted(_DYNAMIC))
def test_variable_length_elementwise_with_one_value_per_item(tmp_path, case):
    model = Lambda(_DYNAMIC[case]).eval()
    value = torch.tensor([[1.0, 2.0], [-1.0, 3.0], [7.0, -2.0]]) if case == "compare a column" \
        else torch.tensor([1.0, -1.0, 7.0])
    result = transpile(model, value, tmp_path / "dynamic", dynamic={0: (1, 8)})
    assert verify(model, value, result).passed


def test_variable_length_silu_times_a_parameter(tmp_path):
    class Scaled(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(1, 4))

        def forward(self, x):
            return F.silu(x) * self.weight

    model, value = Scaled().eval(), torch.randn(3, 4)
    result = transpile(model, value, tmp_path / "scaled", dynamic={0: (1, 8)})
    assert verify(model, value, result).passed


@pytest.mark.parametrize("pool", [
    lambda: nn.MaxPool1d(1),
    lambda: nn.MaxPool2d(1, stride=2),
    lambda: nn.MaxPool1d(2, 2, padding=1, ceil_mode=True),
])
def test_max_pooling_with_one_value_per_window(tmp_path, pool):
    model = pool().eval()
    value = torch.randn(1, 2, 2) if isinstance(model, nn.MaxPool1d) else torch.randn(1, 2, 4, 4)
    _check(model, value, tmp_path / "pool")


@pytest.mark.parametrize("option", [{"stride": (2,)}, {"padding": (1,)}])
def test_conv2d_with_a_one_value_tuple(tmp_path, option):
    torch.manual_seed(0)
    _check(nn.Conv2d(2, 3, 3, **option), torch.randn(1, 2, 6, 6), tmp_path / "conv")


@pytest.mark.parametrize("function", [
    lambda x: torch.softmax(x.sum(), 0),
    lambda x: torch.log_softmax(x.sum(), -1),
    lambda x: x.mean(dim=()),
    lambda x: x.mean(dim=None, keepdim=True),
])
def test_softmax_of_a_scalar_and_mean_over_every_dimension(tmp_path, function):
    _check(Lambda(lambda parts, x: function(x)), torch.randn(2, 3), tmp_path / "reduce")


def test_a_model_that_cannot_run_on_one_item(tmp_path):
    from cattorch import UnsupportedModelError

    model = Lambda(lambda parts, x: F.max_pool1d(x, 2)).eval()
    with pytest.raises(UnsupportedModelError, match="can't run on one"):
        transpile(model, torch.randn(1, 1, 4), tmp_path / "pool", dynamic={0: {2: (2, 6)}})


def test_fast_low_rank_matmul_with_a_variable_length_input(tmp_path):
    class Product(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.randn(32, 4) @ torch.randn(4, 32))

        def forward(self, x):
            return x @ self.weight

    torch.manual_seed(0)
    model, value = Product().eval(), torch.randn(3, 32)
    fast = FastConfig(weights=FastLayerConfig(rank=4))
    result = transpile(model, value, tmp_path / "low_rank", optimization="fast",
                       fast_config=fast, dynamic={0: (1, 8)})
    report = verify(model, value, result, atol=1e-3)
    assert report.passed, report


def test_stored_matmul_wider_than_a_list(tmp_path, small_list_limit):
    small_list_limit(2_000)

    class Product(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(2_100, 3))

        def forward(self, x):
            return x @ self.weight

    torch.manual_seed(0)
    model = Product().eval()
    value = torch.randn(1, 2_100)
    result = transpile(model, value, tmp_path / "wide")
    assert verify(model, value, result, atol=1e-3).passed


class GeluExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Parameter(torch.randn(4, 2))
        self.second = nn.Parameter(torch.randn(2, 4))

    def forward(self, x):
        # A low-rank layer, an activation, and a power of zero.
        return F.gelu((x @ self.first) @ self.second) + x ** 0


@pytest.mark.parametrize("expert", [0, 2])
def test_low_rank_experts(tmp_path, expert):
    torch.manual_seed(0)
    family = ExpertFamily.from_modules([GeluExpert() for _ in range(3)], example_input=torch.ones(1, 4))
    _check(SparseMoE(Fixed(expert), family, top_k=1), torch.randn(3, 4), tmp_path / "moe")


def test_fast_mode_reports_approximations_inside_experts(tmp_path):
    torch.manual_seed(0)
    family = ExpertFamily.from_modules([GeluExpert() for _ in range(3)], example_input=torch.ones(1, 4))
    model = SparseMoE(Fixed(1), family, top_k=1).eval()
    result = transpile(model, torch.randn(3, 4), tmp_path / "moe", optimization="fast")
    assert ("aten.gelu.default", 1) in result.approximate_operations


def test_nan_to_num_replaces_nan_and_infinities(tmp_path):
    model = Lambda(lambda parts, x: (
        torch.nan_to_num(torch.log(x)), torch.nan_to_num(x / 0, posinf=5.0, neginf=-5.0),
    )).eval()
    _check(model, torch.tensor([[-1.0, 0.5, 2.0, -3.0]]), tmp_path / "nan")
