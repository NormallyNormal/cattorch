"""Tests for the private context-style kernel interface."""

from __future__ import annotations

import json
import zipfile

import pytest
import torch
import torch.nn as nn

from cattorch import FastConfig, StorageConfig, UnsupportedOperationError, verify
from cattorch.kernel_api import Kernel, KernelContext, TensorInfo
from cattorch.operator_registry import default_registry
from cattorch.transpiler import _transpile
from cattorch.util.argument import Argument
from cattorch.util.scratch.dsl import (
    add,
    append,
    change_var,
    clear,
    item,
    mul,
    repeat,
    set_var,
    var,
)


@torch.library.custom_op("cattorch_kernel_test::scale_shift", mutates_args=())
def _scale_shift(
    value: torch.Tensor, weight: torch.Tensor, shift: float = 1.0,
) -> torch.Tensor:
    return value * weight + shift


@_scale_shift.register_fake
def _scale_shift_fake(value, weight, shift=1.0):
    return torch.empty_like(value)


TARGET = "cattorch_kernel_test.scale_shift.default"


class ScaleShift(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.5, 2.0, 6).reshape(2, 3))

    def forward(self, value):
        return torch.ops.cattorch_kernel_test.scale_shift(value, self.weight, 2.5)


class DefaultedShift(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.5, 2.0, 6).reshape(2, 3))

    def forward(self, value):
        return torch.ops.cattorch_kernel_test.scale_shift(value, self.weight)


def _elementwise_body(ctx: KernelContext, scale: float = 1.0, extra: float = 0.0):
    """out = value * weight * scale + shift + extra, one pass over the list."""
    value, weight = ctx.inputs["value"], ctx.inputs["weight"]
    position = ctx.temp_var("kernel_test_index")
    out = ctx.output.list
    return (
        clear(out),
        set_var(position, 1),
        repeat(value.numel, (
            append(out, add(
                add(
                    mul(
                        mul(item(value.list, var(position)),
                            item(weight.list, var(position))),
                        scale,
                    ),
                    ctx.scalars["shift"],
                ),
                extra,
            )),
            change_var(position, 1),
        )),
    )


def _registry(*kernels):
    registry = default_registry().clone()
    for kernel in kernels:
        registry.register_kernel(kernel)
    return registry


def _export(tmp_path, model, registry, *, name="model", **options):
    model = model.eval()
    value = torch.linspace(-1.0, 1.0, 6).reshape(2, 3)
    result = _transpile(
        model, value, tmp_path / name, storage=StorageConfig(compression=False),
        _registry=registry, **options,
    )
    return model, value, result


def _sprite(result):
    with zipfile.ZipFile(result.path) as archive:
        return json.loads(archive.read("sprite.json"))


def test_tensor_info_derived_properties():
    info = TensorInfo("x", "T1", (2, 3, 4), torch.float32, False)
    assert info.numel == 24
    assert info.strides == (12, 4, 1)
    assert TensorInfo("s", "T2", ()).numel == 1
    assert TensorInfo("s", "T2", ()).strides == ()


def test_context_names_shapes_dtypes_static_flags_and_scalars():
    operator = torch.ops.cattorch_kernel_test.scale_shift.default
    context = KernelContext.from_arguments(
        TARGET,
        Argument("T3", torch.Size([2, 3]), dtype=torch.float32),
        (
            Argument("T1", torch.Size([2, 3]), dtype=torch.float32),
            Argument("W_weight", torch.Size([2, 3]), dtype=torch.float32, static=True),
            Argument("C_2.5", torch.Size([]), value=2.5),
        ),
        operator,
    )
    assert list(context.inputs) == ["value", "weight"]
    value, weight = context.inputs["value"], context.inputs["weight"]
    assert (value.list, value.shape, value.dtype, value.static) == (
        "T1", (2, 3), torch.float32, False,
    )
    assert (weight.list, weight.static) == ("T2", True)
    assert context.scalars == {"shift": 2.5}
    assert context.output.list == "T3"
    assert context.output.shape == (2, 3)


def test_context_fills_schema_defaults_for_omitted_arguments():
    operator = torch.ops.cattorch_kernel_test.scale_shift.default
    context = KernelContext.from_arguments(
        TARGET,
        Argument("T3", torch.Size([2, 3])),
        (Argument("T1", torch.Size([2, 3])), Argument("T2", torch.Size([2, 3]))),
        operator,
    )
    assert context.scalars == {"shift": 1.0}


def test_context_allocation_order_and_private_names():
    context = KernelContext.from_arguments(
        TARGET,
        Argument("T2", torch.Size([4])),
        (Argument("T1", torch.Size([4])),),
        torch.ops.aten.relu.default,
    )
    scratch = context.temp_list("scratch", 10)
    table = context.static_list("table", [3, 1, 2])
    assert (scratch, table) == ("_scratch", "_table")
    assert context.static_list("_already private", [1]) == "_already private"
    assert context.temp_var("counter") == "counter"
    assert context.lists[:3] == ("T1", "_scratch", "_table")
    assert context.lists[-1] == "T2"
    # Reading the output places it where it is first requested.
    context.output
    assert context.lists.index("T2") == 4
    assert context.temporary_sizes == {"_scratch": 10}
    assert context.list_values["_table"] == [3, 1, 2]
    with pytest.raises(ValueError, match="already declared"):
        context.temp_list("scratch", 1)


def test_kernel_rejects_unknown_fast_flag():
    with pytest.raises(ValueError, match="unknown FastConfig flag"):
        Kernel(TARGET, _elementwise_body, fast_flag="not_a_flag")


def test_user_kernel_for_new_custom_op_matches_eager(tmp_path):
    seen = {}

    def emit(ctx):
        seen["context"] = ctx
        return _elementwise_body(ctx)

    registry = _registry(Kernel(TARGET, emit, name="scale_shift_test"))
    model, value, result = _export(tmp_path, ScaleShift(), registry)
    assert verify(model, value, result, atol=1e-5, rtol=0).passed

    context = seen["context"]
    assert context.inputs["value"].dtype == torch.float32
    assert context.inputs["value"].static is False
    assert context.inputs["weight"].static is True
    assert context.inputs["weight"].shape == (2, 3)
    assert context.scalars == {"shift": 2.5}
    assert context.output.dtype == torch.float32
    assert context.output.shape == (2, 3)


def test_omitted_scalar_uses_schema_default_in_export(tmp_path):
    registry = _registry(Kernel(TARGET, _elementwise_body))
    model, value, result = _export(tmp_path, DefaultedShift(), registry)
    assert verify(model, value, result, atol=1e-5, rtol=0).passed


def test_op_without_kernel_still_fails_closed(tmp_path):
    with pytest.raises(UnsupportedOperationError, match="scale_shift"):
        _export(tmp_path, ScaleShift(), default_registry().clone())


def test_replacing_a_builtin_kernel_is_scoped_and_keeps_the_observer():
    original = default_registry().operation("aten.linear.default")
    assert original.observer is not None
    replaced = _registry(Kernel("aten.linear.default", lambda ctx: ()))
    rule = replaced.operation("aten.linear.default")
    assert rule.exact_kernel is not original.exact_kernel
    assert rule.observer is original.observer
    assert rule.fast_kernel is original.fast_kernel
    assert default_registry().operation("aten.linear.default") is original


def test_replacing_relu_changes_only_the_cloned_registry(tmp_path):
    class Relu(nn.Module):
        def forward(self, value):
            return torch.relu(value)

    def emit(ctx):
        # Deliberately wrong: relu(x) + 1, to prove this kernel is the one used.
        source = ctx.inputs["self"]
        position = ctx.temp_var("kernel_test_index")
        out = ctx.output.list
        return (
            clear(out),
            set_var(position, 1),
            repeat(source.numel, (
                append(out, add(item(source.list, var(position)), 1)),
                change_var(position, 1),
            )),
        )

    value = torch.linspace(-1.0, 1.0, 6).reshape(2, 3)
    model = Relu().eval()
    registry = _registry(Kernel("aten.relu.default", emit))
    custom = _transpile(
        model, value, tmp_path / "custom", storage=StorageConfig(compression=False),
        _registry=registry,
    )
    assert not verify(model, value, custom).passed

    builtin = _transpile(
        model, value, tmp_path / "builtin", storage=StorageConfig(compression=False),
        _registry=default_registry().clone(),
    )
    assert verify(model, value, builtin).passed


def test_fast_flag_kernel_is_used_only_in_fast_mode(tmp_path):
    used = []

    def exact(ctx):
        used.append("exact")
        return _elementwise_body(ctx)

    def fast(ctx):
        used.append("fast")
        return _elementwise_body(ctx)

    registry = _registry(
        Kernel(TARGET, exact),
        Kernel(TARGET, fast, fast_flag="activations"),
    )
    rule = registry.operation(TARGET)
    assert rule.fast_flag == "activations"
    assert rule.exact_kernel is not None and rule.fast_kernel is not None

    _export(tmp_path, ScaleShift(), registry, name="exact")
    assert used == ["exact"]

    used.clear()
    _export(tmp_path, ScaleShift(), registry, name="fast", optimization="fast")
    assert used == ["fast"]

    used.clear()
    _export(
        tmp_path, ScaleShift(), registry, name="flag-off", optimization="fast",
        fast_config=FastConfig(activations=False),
    )
    assert used == ["exact"]


def test_fast_only_kernel_is_unsupported_in_exact_mode(tmp_path):
    registry = _registry(Kernel(TARGET, _elementwise_body, fast_flag="activations"))
    with pytest.raises(UnsupportedOperationError, match="scale_shift"):
        _export(tmp_path, ScaleShift(), registry, name="exact")
    model, value, result = _export(
        tmp_path, ScaleShift(), registry, name="fast", optimization="fast",
    )
    assert verify(model, value, result, atol=1e-5, rtol=0).passed


def test_validator_rejection_surfaces_in_unsupported_operation_error(tmp_path):
    seen = {}

    def validator(ctx):
        seen["shift"] = ctx.scalars["shift"]
        seen["static"] = ctx.inputs["weight"].static
        return "shift must be zero for this draft kernel"

    registry = _registry(Kernel(TARGET, _elementwise_body, validator=validator))
    with pytest.raises(UnsupportedOperationError) as error:
        _export(tmp_path, ScaleShift(), registry)
    assert "shift must be zero for this draft kernel" in str(error.value)
    assert error.value.detail == "shift must be zero for this draft kernel"
    assert seen == {"shift": 2.5, "static": True}


def test_validator_returning_none_accepts(tmp_path):
    registry = _registry(Kernel(TARGET, _elementwise_body, validator=lambda ctx: None))
    model, value, result = _export(tmp_path, ScaleShift(), registry)
    assert verify(model, value, result, atol=1e-5, rtol=0).passed


def test_large_temp_list_is_sharded(tmp_path):
    capacity = 250_000

    def emit(ctx):
        value, weight = ctx.inputs["value"], ctx.inputs["weight"]
        buffer = ctx.temp_list("kernel_test_buffer", capacity)
        position = ctx.temp_var("kernel_test_index")
        out = ctx.output.list
        return (
            clear(buffer),
            set_var(position, 1),
            repeat(value.numel, (
                append(buffer, add(
                    mul(item(value.list, var(position)),
                        item(weight.list, var(position))),
                    ctx.scalars["shift"],
                )),
                change_var(position, 1),
            )),
            clear(out),
            set_var(position, 1),
            repeat(value.numel, (
                append(out, item(buffer, var(position))),
                change_var(position, 1),
            )),
        )

    registry = _registry(Kernel(TARGET, emit))
    model, value, result = _export(tmp_path, ScaleShift(), registry)
    assert verify(model, value, result, atol=1e-5, rtol=0).passed
    names = [entry[0] for entry in _sprite(result)["lists"].values()]
    shards = [name for name in names if name.startswith("_kernel_test_buffer")]
    assert len(shards) == 2, shards


def test_register_kernel_creates_a_rule_and_does_not_leak_into_the_default():
    registry = _registry(Kernel(TARGET, _elementwise_body))
    assert registry.operation(TARGET).exact_kernel is not None
    assert default_registry().operation(TARGET) is None


def test_builtin_rotary_and_topk_use_context_kernels():
    for target in (
        "cattorch.rotary_embedding.default",
        "cattorch.topk_values.default",
        "cattorch.topk_indices.default",
    ):
        kernel = default_registry().operation(target).exact_kernel
        assert hasattr(kernel, "kernel") and isinstance(kernel.kernel, Kernel)
