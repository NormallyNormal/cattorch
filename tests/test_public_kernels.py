"""User kernels through the public, experimental ``kernels=`` argument."""

from __future__ import annotations

import json
import zipfile

import pytest
import torch
import torch.nn as nn

from cattorch import GenerationProgram, UnsupportedOperationError, transpile, verify
from cattorch import EntryPoint, ExportProgram, Input, Output
from cattorch.experimental import Kernel, analyze, build_kernel_benchmark, check_kernel
from cattorch.scratch import (
    add, append, change_var, clear, div, item, mathop, mul, render, repeat, set_var, sub, var,
)


def _pointwise(function):
    """A kernel body applying ``function`` to each element of ``self``."""
    def emit(ctx):
        value, out = ctx.inputs["self"], ctx.output.list
        index = ctx.temp_var("user_index")
        return (
            clear(out),
            set_var(index, 1),
            repeat(value.numel, (
                append(out, function(item(value.list, var(index)))),
                change_var(index, 1),
            )),
        )
    return emit


def _mish(x):
    softplus = mathop("ln", add(1, mathop("e ^", x)))
    return mul(x, sub(div(2, add(1, mathop("e ^", mul(-2, softplus)))), 1))


MISH = Kernel(torch.ops.aten.mish, _pointwise(_mish), name="user_mish")


def _sprite_text(result) -> str:
    with zipfile.ZipFile(result.path) as archive:
        return archive.read("sprite.json").decode()


def test_a_kernel_replaces_a_decomposition(tmp_path):
    model = nn.Sequential(nn.Linear(4, 4), nn.Mish()).eval()
    value = torch.randn(3, 4)
    plain = transpile(model, value, tmp_path / "plain")
    assert plain.fallback_operations == (("aten.mish.default", 1),)
    result = transpile(model, value, tmp_path / "kernel", kernels=[MISH])
    assert result.fallback_operations == ()
    assert result.block_count < plain.block_count
    assert verify(model, value, result).passed


def test_a_kernel_replaces_a_builtin_kernel_for_one_export(tmp_path):
    calls = []

    def emit(ctx):
        calls.append(ctx.target)
        return _pointwise(lambda x: div(add(x, mathop("abs", x)), 2))(ctx)

    relu = Kernel(torch.ops.aten.relu.default, emit)
    model, value = nn.ReLU().eval(), torch.randn(2, 5)
    assert verify(model, value, transpile(model, value, tmp_path / "a", kernels=[relu])).passed
    assert calls == ["aten.relu.default"]
    transpile(model, value, tmp_path / "b")
    assert calls == ["aten.relu.default"]


def test_approximate_kernels_apply_only_in_fast_mode(tmp_path):
    rough = Kernel(torch.ops.aten.mish, _pointwise(lambda x: mul(x, 0.6)), approximate=True)
    model, value = nn.Mish().eval(), torch.randn(2, 4)
    exact = transpile(model, value, tmp_path / "exact", kernels=[rough])
    assert exact.fallback_operations == (("aten.mish.default", 1),)
    assert exact.approximate_operations == ()
    fast = transpile(model, value, tmp_path / "fast", kernels=[rough], optimization="fast")
    assert fast.approximate_operations == (("aten.mish.default", 1),)
    report = verify(model, value, fast)
    assert not report.passed
    assert report.approximate_operations == (("aten.mish.default", 1),)


def test_a_validator_rejection_names_the_reason(tmp_path):
    picky = Kernel(
        torch.ops.aten.mish, _pointwise(_mish),
        validator=lambda ctx: "needs a 1D input" if len(ctx.inputs["self"].shape) != 1 else None,
    )
    with pytest.raises(UnsupportedOperationError, match="needs a 1D input"):
        transpile(nn.Mish().eval(), torch.randn(2, 3), tmp_path / "picky", kernels=[picky])


def test_kernels_must_be_kernels(tmp_path):
    with pytest.raises(TypeError, match="Kernel"):
        transpile(nn.Mish().eval(), torch.randn(3), tmp_path / "bad", kernels=[_mish])


def test_kernels_apply_to_generation_and_programs(tmp_path):
    from test_generation_automatic import nano, prefill_and_decode, load_sprite
    from cattorch.util.scratch.emulator import ScratchEmulator

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = nano()

        def forward(self, tokens):
            return nn.functional.mish(self.inner(tokens))

    model = Block().eval()
    result = transpile(model, GenerationProgram(max_context=8), tmp_path / "gen", kernels=[MISH])
    assert result.fallback_operations == ()
    with torch.no_grad():
        prefill_and_decode(ScratchEmulator(load_sprite(result.path)), model, [1, 2], [3])

    program = ExportProgram(entrypoints=(EntryPoint(
        "squash", arguments=(Input("value", torch.randn(4)),), returns=(Output("value"),),
    ),))
    squash = nn.Mish().eval()
    assert analyze(squash, program, kernels=[MISH]).entrypoints[0].fallback_operations == ()
    transpile(squash, program, tmp_path / "program", kernels=[MISH])


def test_check_kernel_passes_a_correct_kernel_and_names_a_wrong_one():
    reports = check_kernel(MISH, [(torch.randn(5),), (torch.randn(2, 3) * 4,)])
    assert all(report.passed for report in reports)
    wrong = Kernel(torch.ops.aten.mish, _pointwise(lambda x: mul(x, 2)))
    with pytest.raises(AssertionError, match="sample 0"):
        check_kernel(wrong, [(torch.randn(5),)])


def test_check_kernel_takes_static_weights():
    seen = []

    def emit(ctx):
        value, other = ctx.inputs["self"], ctx.inputs["other"]
        seen.append((value.static, other.static))
        out, index = ctx.output.list, ctx.temp_var("user_index")
        return (
            clear(out),
            set_var(index, 1),
            repeat(value.numel, (
                append(out, mul(item(value.list, var(index)), item(other.list, var(index)))),
                change_var(index, 1),
            )),
        )

    multiply = Kernel(torch.ops.aten.mul.Tensor, emit)
    sample = (torch.randn(4), torch.randn(4))
    assert check_kernel(multiply, [sample], static=("other",))[0].passed
    assert seen[-1] == (False, True)
    assert check_kernel(multiply, [sample])[0].passed
    assert seen[-1] == (False, False)


def test_render_shows_the_kernel_body():
    body = _pointwise(_mish)(type("Context", (), {
        "inputs": {"self": type("Info", (), {"numel": 3, "list": "T1"})()},
        "output": type("Info", (), {"list": "T2"})(),
        "temp_var": staticmethod(lambda name: name),
    })())
    text = render(body)
    assert "repeat 3:" in text and "T2.append(" in text


def test_build_kernel_benchmark_compares_builtin_and_kernel(tmp_path):
    model = nn.Mish().eval()
    path = build_kernel_benchmark(MISH, [("mish", model, (torch.randn(8),))], tmp_path / "bench")
    with zipfile.ZipFile(path) as archive:
        project = json.loads(archive.read("project.json"))
    names = {target["name"] for target in project["targets"]}
    assert {"mish builtin", "mish kernel"} <= names
