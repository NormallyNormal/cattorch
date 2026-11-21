"""Operations declared with ``cattorch.experimental.op``."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from cattorch import transpile, verify
from cattorch.experimental import Operation, check_kernel, op
from cattorch.scratch import (
    add, append, change_var, clear, div, item, mathop, mul, repeat, set_var, var,
)


@op
def squash(x: Tensor, alpha: float = 1.0) -> Tensor:
    """``x / (1 + alpha |x|)``, a cheap tanh-like curve."""
    return x / (1 + alpha * x.abs())


@squash.kernel(name="squash")
def _emit_squash(ctx):
    value, out = ctx.inputs["x"], ctx.output.list
    index = ctx.temp_var("squash_index")
    current = item(value.list, var(index))
    return (
        clear(out),
        set_var(index, 1),
        repeat(value.numel, (
            append(out, div(current, add(1, mul(ctx.scalars["alpha"], mathop("abs", current))))),
            change_var(index, 1),
        )),
    )


@op(name="cattorch_test::blend")
def blend(x: Tensor, y: Tensor, weights: list[float]) -> Tensor:
    return torch.tanh(x) * weights[0] + y * weights[1]


@op
def rough_sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x)


@rough_sigmoid.kernel(approximate=True)
def _emit_rough(ctx):
    value, out = ctx.inputs["x"], ctx.output.list
    index = ctx.temp_var("rough_index")
    return (
        clear(out),
        set_var(index, 1),
        repeat(value.numel, (
            append(out, add(0.5, mul(0.25, item(value.list, var(index))))),
            change_var(index, 1),
        )),
    )


@op
def passthrough(x: Tensor) -> Tensor:
    return x


class Uses(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x, y):
        return (
            squash(self.linear(x), 0.5) + blend(x, y, [2.0, -1.0])
            + rough_sigmoid(x) + passthrough(y)
        )


def test_declared_operations_run_eagerly():
    assert isinstance(squash, Operation)
    assert squash.name.startswith("cattorch_user::")
    assert squash.__doc__.startswith("``x / (1")
    assert torch.allclose(squash(torch.tensor([3.0]), 1.0), torch.tensor([0.75]))
    value = torch.ones(2)
    assert passthrough(value) is not value


def test_an_attached_kernel_is_used_and_others_trace_their_eager_function(tmp_path):
    torch.manual_seed(0)
    model, inputs = Uses().eval(), (torch.randn(2, 3), torch.randn(2, 3))
    result = transpile(model, inputs, tmp_path / "exact")
    assert verify(model, inputs, result).passed
    fallbacks = dict(result.fallback_operations)
    assert "cattorch_test.blend.default" in fallbacks
    assert str(squash.operator) not in fallbacks
    assert result.approximate_operations == ()


def test_an_approximate_kernel_applies_in_fast_mode(tmp_path):
    torch.manual_seed(0)
    model, inputs = Uses().eval(), (torch.randn(2, 3), torch.randn(2, 3))
    result = transpile(model, inputs, tmp_path / "fast", optimization="fast")
    assert result.approximate_operations == ((str(rough_sigmoid.operator), 1),)


def test_check_kernel_takes_a_declared_operation():
    reports = check_kernel(squash.kernels[0], [(torch.randn(4), 2.0), (torch.randn(2, 3),)])
    assert all(report.passed for report in reports)


def test_declaring_needs_annotations():
    with pytest.raises(TypeError, match="annotate"):
        @op
        def unannotated(x):
            return x
