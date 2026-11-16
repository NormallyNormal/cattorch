"""Differential coverage for the FX/export capture transition."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time

import pytest
import torch
import torch.nn as nn

import cattorch.frontend as frontend_module
from cattorch import UnsupportedModelError, UnsupportedOperationError, transpile, verify
from cattorch.frontend import capture_model
from cattorch.operator_registry import OperationRule, default_registry
from cattorch.transpiler import _transpile
from cattorch.util.instruction.optimized_elementwise import elementwise_kernel


class StatefulModel(nn.Module):
    def __init__(self):
        super().__init__()
        weight = nn.Parameter(torch.randn(4, 4))
        self.weight = weight
        self.tied_weight = weight
        self.register_buffer("bias", torch.randn(4))
        self.constant = torch.arange(4, dtype=torch.float32)

    def forward(self, value, residual):
        return torch.relu(value @ self.weight + self.bias + self.constant + residual)


def test_parameters_buffers_constants_and_multiple_inputs_are_lowered(tmp_path):
    torch.manual_seed(123)
    model = StatefulModel().eval()
    inputs = (torch.randn(2, 4), torch.randn(2, 4))
    result = transpile(model, inputs, tmp_path / "model")

    comparison = verify(model, inputs, result, atol=2e-4)
    assert comparison.passed, comparison
    assert tuple(item.list_name for item in result.inputs) == ("input", "input_1")
    assert result.output.shape == (2, 4)


class NestedLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(nn.Linear(3, 2))

    def forward(self, value):
        return self.stack(value)


def test_fx_capture_preserves_nested_module_context():
    captured = capture_model(NestedLinear().eval(), (torch.randn(1, 3),))
    linear = next(node for node in captured.graph.nodes if str(node.target) == "aten.linear.default")
    paths = [entry[0] for entry in linear.meta["nn_module_stack"].values()]
    assert "stack.0" in paths


def test_public_capture_serializes_process_global_tracing_state(monkeypatch):
    guard = Lock()
    active = 0
    maximum = 0

    def fake_capture(model, example_inputs, example_kwargs=None):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        with guard:
            active -= 1
        return model, example_inputs

    monkeypatch.setattr(frontend_module, "_capture_fx", fake_capture)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(
            lambda index: capture_model(index, (index,)),
            range(8),
        ))

    assert results == [(index, (index,)) for index in range(8)]
    assert maximum == 1


def test_multiple_tensor_outputs_are_preserved(tmp_path):
    class Multiple(nn.Module):
        def forward(self, value):
            return value, -value

    value = torch.ones(2)
    result = transpile(
        Multiple().eval(), value, tmp_path / "multiple",
    )
    assert tuple(output.list_name for output in result.outputs) == ("output", "output_1")
    assert all(output.shape == (2,) for output in result.outputs)


def test_fx_rejects_data_dependent_python_control_flow(tmp_path):
    class DataDependent(nn.Module):
        def forward(self, value):
            if value.sum().item() > 0:
                return value + 1
            return value - 1

    with pytest.raises(UnsupportedModelError, match="PyTorch couldn't trace this model"):
        transpile(DataDependent().eval(), torch.ones(2), tmp_path / "data-dependent")


def test_public_transpile_captures_an_export_owned_copy(tmp_path):
    class StatefulForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            return -value

    model = StatefulForward().eval()
    transpile(model, torch.ones(2), tmp_path / "owned")
    assert model.calls == 0


@torch.library.custom_op("cattorch_test::neg", mutates_args=())
def _custom_neg(value: torch.Tensor) -> torch.Tensor:
    return -value


@_custom_neg.register_fake
def _custom_neg_fake(value):
    return torch.empty_like(value)


class AdaptedNeg(nn.Module):
    def forward(self, value):
        return -value


class OpaqueNeg(nn.Module):
    def forward(self, value):
        return _custom_neg(value)


def test_private_registry_supports_module_adapters_and_semantic_ops(tmp_path):
    registry = default_registry().clone()
    registry.register_module_adapter(AdaptedNeg, lambda _module: OpaqueNeg())
    registry.register_operation(OperationRule(
        target="cattorch_test.neg.default",
        exact_kernel=elementwise_kernel("aten.neg.default").instruction_class(),
        lowering_target=torch.ops.aten.neg.default,
    ))
    model = AdaptedNeg().eval()
    result = _transpile(
        model,
        torch.tensor([1.0, -2.0]),
        tmp_path / "registered",
        _registry=registry,
    )
    assert verify(model, torch.tensor([1.0, -2.0]), result).passed


def test_unknown_semantic_op_still_fails_closed(tmp_path):
    with pytest.raises(UnsupportedOperationError, match="cattorch_test.neg.default"):
        transpile(OpaqueNeg().eval(), torch.ones(2), tmp_path / "unknown")


def test_runtime_full_slice_followed_by_view_resolves_complete_alias_chain(tmp_path):
    class FinalTokenProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(5, 4))

        def forward(self, value):
            selected = value[:, 0:1]
            return torch.nn.functional.linear(selected.view(1, 4), self.weight)

    model = FinalTokenProjection().eval()
    value = torch.randn(1, 1, 4)
    result = transpile(model, value, tmp_path / "alias-chain")
    assert verify(model, value, result).passed


def test_fx_decomposes_causal_sdpa_and_mqa_repeat_interleave(tmp_path):
    class Attention(nn.Module):
        def forward(self, query, key, value):
            key = key.repeat_interleave(2, dim=1)
            value = value.repeat_interleave(2, dim=1)
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=True,
            )

    inputs = (
        torch.randn(1, 2, 3, 2),
        torch.randn(1, 1, 3, 2),
        torch.randn(1, 1, 3, 2),
    )
    model = Attention().eval()
    result = transpile(model, inputs, tmp_path / "sdpa")
    assert verify(model, inputs, result, atol=2e-4).passed


def test_positive_strided_slice_lowers_without_an_index_map(tmp_path):
    class EvenFeatures(nn.Module):
        def forward(self, value):
            return value[:, ::2]

    model = EvenFeatures().eval()
    value = torch.randn(3, 7)
    result = transpile(model, value, tmp_path / "strided")
    assert verify(model, value, result).passed


class RouterUpdate(nn.Module):
    """Mixtral's router: normalize the top-k values in place."""

    def forward(self, value):
        weights, ids = torch.topk(torch.softmax(value, dim=-1), 2, dim=-1)
        weights /= weights.sum(dim=-1, keepdim=True)
        return weights, ids


class ReadBeforeUpdate(nn.Module):
    def forward(self, value):
        result = value.exp()
        before = result * 3
        result.mul_(0.5)
        result.add_(torch.ones(1))
        return before + result


class CastUpdate(nn.Module):
    def forward(self, value):
        result = torch.zeros(value.shape, dtype=torch.float64)
        result.copy_(value * 2)
        return result.float() + 1


class SliceFill(nn.Module):
    """DeepSeek-V3's key assembly: fill both halves of a fresh tensor."""

    def forward(self, value):
        result = value.new_empty(3, 7)
        result[..., :4].copy_(value * 2)
        result[..., 4:].copy_(value[..., :3].exp())
        return result


class SelectAndNestedAssign(nn.Module):
    def forward(self, value):
        result = torch.zeros(3, 4, 5)
        result[1] = value
        result[:, 1:][..., -2:] = value[0, :2] + 1
        result[0, :, ::1] = result[1] * 3
        return result


@pytest.mark.parametrize("model", [
    RouterUpdate(), ReadBeforeUpdate(), CastUpdate(), SliceFill(),
])
def test_in_place_updates_of_intermediates_become_dataflow(tmp_path, model):
    captured = capture_model(model.eval(), (torch.randn(3, 4),))
    targets = [str(node.target) for node in captured.graph.nodes]
    assert "aten.copy_.default" not in targets
    value = torch.randn(3, 4)
    result = transpile(model, value, tmp_path / "in-place")
    assert verify(model, value, result).passed


def test_slice_and_select_assignment_become_dataflow(tmp_path):
    model = SelectAndNestedAssign().eval()
    value = torch.randn(4, 5)
    captured = capture_model(model, (value,))
    assert "aten.copy_.default" not in [str(node.target) for node in captured.graph.nodes]
    result = transpile(model, value, tmp_path / "assign")
    assert verify(model, value, result).passed


class UpdatesItsInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        x += 1
        x.clamp_(-1.5, 1.5)
        x[0] = 0
        return self.linear(x)


def test_in_place_updates_of_an_input_become_dataflow(tmp_path):
    torch.manual_seed(0)
    model = UpdatesItsInput().eval()
    result = transpile(model, torch.randn(3, 4), tmp_path / "input")
    # Each update runs its operation's own kernel, not a decomposition.
    assert result.fallback_operations == ()
    value = torch.randn(3, 4)
    assert verify(model, value.clone(), result).passed


class InPlaceOperations(nn.Module):
    def forward(self, x):
        y = x * 2
        y.clamp_(-1, 1)
        y[1].fill_(0.25)
        y.masked_fill_(y > 0.9, -3.0)
        y.mul_(x).sigmoid_()
        x.fill_(2.0)
        x.lt_(y)
        return y + x


def test_in_place_operations_run_their_out_of_place_kernels(tmp_path):
    torch.manual_seed(1)
    model = InPlaceOperations().eval()
    result = transpile(model, torch.randn(3, 4), tmp_path / "operations")
    assert result.fallback_operations == ()
    value = torch.randn(3, 4)
    assert verify(model, value.clone(), result).passed


@pytest.mark.parametrize("update", [
    lambda x: x.add_(0.5), lambda x: x.clamp_(0.5), lambda x: x.mul_(torch.full((4,), 0.5)),
], ids=["add", "clamp", "mul"])
def test_in_place_operations_keep_pytorchs_casting_rule(tmp_path, update):
    # PyTorch can't write a float result into an integer tensor in place;
    # the copy back mustn't cast it silently.
    class Updates(nn.Module):
        def forward(self, x):
            update(x)
            return x

    with pytest.raises(UnsupportedModelError, match="Float can.t be cast to the desired output type"):
        transpile(Updates().eval(), torch.randint(0, 5, (3, 4)), tmp_path / "integer")


IN_PLACE_FORMS = {
    # In place, as written, and out of place: each pair must export alike.
    "remainder": (lambda y: y.remainder_(0.75), lambda y: y.remainder(0.75)),
    "%=": (lambda y: y.__imod__(0.75), lambda y: y % 0.75),
    "//=": (lambda y: y.__ifloordiv__(0.5), lambda y: y // 0.5),
    "fmod": (lambda y: y.fmod_(0.5), lambda y: y.fmod(0.5)),
    "trunc": (lambda y: y.trunc_(), lambda y: y.trunc()),
    "sinh": (lambda y: y.sinh_(), lambda y: y.sinh()),
    "atanh": (lambda y: y.atanh_(), lambda y: y.atanh()),
    "hypot": (lambda y: y.hypot_(y + 1), lambda y: y.hypot(y + 1)),
    "atan2": (lambda y: y.atan2_(y + 1), lambda y: y.atan2(y + 1)),
    "pow with a scalar exponent": (lambda y: y.pow_(2.0), lambda y: y.pow(2.0)),
    "ldexp": (lambda y: y.ldexp_(torch.ones_like(y)), lambda y: y.ldexp(torch.ones_like(y))),
    "logical_or": (lambda y: y.logical_or_(y - 0.5), lambda y: y.logical_or(y - 0.5)),
    "conj_physical": (lambda y: y.conj_physical_(), lambda y: y.conj_physical()),
}


@pytest.mark.parametrize("name", sorted(IN_PLACE_FORMS))
def test_in_place_and_out_of_place_forms_export_alike(tmp_path, name):
    in_place, out_of_place = IN_PLACE_FORMS[name]

    class Form(nn.Module):
        def __init__(self, update, written):
            super().__init__()
            self.update, self.written = update, written

        def forward(self, x):
            y = x * 0.9
            if self.written:
                self.update(y)
                return y * 1
            return self.update(y) * 1

    value = torch.rand(3, 4) * 0.8 + 0.05
    for written, update in ((True, in_place), (False, out_of_place)):
        model = Form(update, written).eval()
        result = transpile(model, value, tmp_path / f"form{written}")
        assert verify(model, value, result).passed


def test_an_integer_pointer_wraps_in_place(tmp_path):
    from cattorch import ProgramCall, entrypoint

    class Ring(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("pointer", torch.tensor(0))

        @entrypoint(example_inputs=())
        def advance(self):
            self.pointer += 3
            self.pointer %= 4
            return self.pointer * 1

    model = Ring().eval()
    result = transpile(model, None, tmp_path / "ring")
    assert verify(model, None, result, calls=[ProgramCall("advance", ())] * 6).passed


@pytest.mark.parametrize("update", [lambda x: x.remainder_(0.5), lambda x: x.__imod__(1.5)])
def test_in_place_remainder_keeps_pytorchs_casting_rule(tmp_path, update):
    class Updates(nn.Module):
        def forward(self, x):
            update(x)
            return x

    with pytest.raises(UnsupportedModelError, match="Float can.t be cast to the desired output type"):
        transpile(Updates().eval(), torch.randint(0, 5, (3, 4)), tmp_path / "integer")


@pytest.mark.parametrize("update", [lambda y: y.t_(), lambda y: y.unsqueeze_(0), lambda y: y.squeeze_()])
def test_in_place_shape_changes_say_what_to_write(tmp_path, update):
    class Reshape(nn.Module):
        def forward(self, x):
            y = x * 1
            update(y)
            return y * 1

    with pytest.raises(UnsupportedOperationError, match=r"changes the tensor's shape in place.*write x = x\."):
        transpile(Reshape().eval(), torch.randn(3, 4), tmp_path / "reshape")


def test_a_buffer_updated_in_forward_points_to_entry_points(tmp_path):
    class Counter(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("count", torch.zeros(1))

        def forward(self, x):
            self.count += 1
            return x + self.count

    with pytest.raises(UnsupportedOperationError, match=r"copy_.*cattorch\.entrypoint"):
        transpile(Counter().eval(), torch.randn(3), tmp_path / "counter")


def test_strided_slice_assignment_stays_unsupported(tmp_path):
    class EveryOther(nn.Module):
        def forward(self, value):
            result = torch.zeros(3, 8)
            result[:, ::2] = value
            return result

    with pytest.raises(UnsupportedOperationError, match="copy_"):
        transpile(EveryOther().eval(), torch.randn(3, 4), tmp_path / "strided-assign")


def _write_view_read_view(x):
    y = x * 1
    v = y.view(12)
    v[3:7] = 0
    return v * 2


def _write_row_read_row(x):
    y = x * 1
    row = y[1]
    row[2:4] = -1
    return row + 0


def _write_view_return_view(x):
    z = (x * 1)[1:]
    z[0, :2] = -5
    return z


def _write_input_row(x):
    r = x[0]
    r[1] = 7
    return r * 1, x * 1


def _write_input_view(x):
    v = x.view(2, 6)
    r = v[1]
    r[0:4] = -1
    return v * 1, r * 1


def _read_sibling_view(x):
    y = x * 1
    a = y[0]
    b = y[:2]
    a[1] = 7
    return b * 1


def _read_transposed_sibling(x):
    y = x * 1
    a = y[0]
    t = y.t()
    a[1] = 7
    return t


def _write_through_two_views(x):
    y = x * 1
    a = y[0]
    b = y[2]
    a[0] = 1
    b[1] = 2
    return y * 1, a + b


def _write_nested_views(x):
    y = x * 1
    a = y[1:]
    b = a[:, 1:]
    c = b[0]
    c[0] = 3
    return a * 1, b * 1, c * 1


def _view_of_view_after_write(x):
    y = x * 1
    v = y.view(12)
    v[3:7] = 0
    return v.view(4, 3) * 1


def _read_view_taken_before_update(x):
    y = x * 2
    row = y[:, 0]
    y += 1
    return row * 1


def _in_place_on_row(x):
    y = x * 1
    row = y[1]
    row.mul_(2)
    return row * 1, y


@pytest.mark.parametrize("function", [
    _write_view_read_view, _write_row_read_row, _write_view_return_view, _write_input_row,
    _write_input_view, _read_sibling_view, _read_transposed_sibling, _write_through_two_views,
    _write_nested_views, _view_of_view_after_write, _read_view_taken_before_update,
    _in_place_on_row,
])
def test_views_read_after_an_update_through_a_view_see_the_update(tmp_path, function):
    class Model(nn.Module):
        def forward(self, x):
            return function(x)

    model = Model().eval()
    value = torch.randn(3, 4)
    captured = capture_model(model, (value.clone(),))
    assert "aten.copy_.default" not in [str(node.target) for node in captured.graph.nodes]
    result = transpile(model, value.clone(), tmp_path / function.__name__)
    assert verify(model, value.clone(), result).passed


def test_in_place_update_seen_through_a_strided_view_stays_unsupported(tmp_path):
    class ViewThenUpdate(nn.Module):
        def forward(self, value):
            result = value * 2
            row = result.as_strided((3,), (4,))
            result += 1
            return row * 1

    with pytest.raises(UnsupportedOperationError, match="copy_"):
        transpile(ViewThenUpdate().eval(), torch.randn(3, 4), tmp_path / "aliased")


class _ListScaled(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scales", torch.tensor([2.0, 3.0]))

    def forward(self, x):
        # The buffer's values are fixed, so reading them in Python is fine.
        return x * self.scales.tolist()[1] / max(x.tolist())


class _NumpyScaled(nn.Module):
    def forward(self, x):
        return x * float(x.numpy().sum())


@pytest.mark.parametrize("model", [_ListScaled(), _NumpyScaled()])
def test_python_conversions_of_inputs_are_rejected(model):
    with pytest.raises(UnsupportedModelError, match="on a value computed from its inputs"):
        capture_model(model, (torch.tensor([1.0, 4.0]),))
    assert torch.Tensor.tolist.__name__ == "tolist"  # restored afterwards


def test_python_conversions_of_constants_still_fold():
    class Constant(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scales", torch.tensor([2.0, 3.0]))

        def forward(self, x):
            return x * self.scales.tolist()[1]

    captured = capture_model(Constant(), (torch.ones(2),))
    assert torch.equal(captured.graph_module(torch.ones(2))[0], torch.full((2,), 3.0))


def _default_mismatches(operator, handler) -> list[str]:
    """Schema arguments with a default that ``handler`` can't be called without."""
    import inspect

    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return []
    parameters = list(signature.parameters.values())
    kinds = {parameter.kind for parameter in parameters}
    positional = [
        parameter for parameter in parameters
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    problems = []
    schema_positional = 0
    for argument in operator._schema.arguments:
        if argument.kwarg_only:
            parameter = signature.parameters.get(argument.name)
            if parameter is None:
                if inspect.Parameter.VAR_KEYWORD not in kinds:
                    problems.append(f"keyword {argument.name!r} isn't accepted")
            elif argument.has_default_value() and parameter.default is parameter.empty:
                problems.append(f"keyword {argument.name!r} has no default")
            continue
        if schema_positional < len(positional):
            parameter = positional[schema_positional]
            if argument.has_default_value() and parameter.default is parameter.empty:
                problems.append(f"argument {argument.name!r} has no default")
        elif inspect.Parameter.VAR_POSITIONAL not in kinds:
            problems.append(f"argument {argument.name!r} isn't accepted")
        schema_positional += 1
    return problems


def test_decompositions_accept_every_schema_default_omitted():
    # The tracer leaves out arguments at their schema default (torch.cdist's
    # p=2), so a decomposition must not require one.
    from cattorch.frontend import _primitive_decompositions, own_decompositions

    table = dict(own_decompositions())
    for name, decomposition in _primitive_decompositions().items():
        table[getattr(torch.ops.prims, name).default] = decomposition
    problems = {
        str(operator): found for operator, handler in table.items()
        if (found := _default_mismatches(operator, handler))
    }
    assert not problems


def test_cdist_exports_with_its_default_p(tmp_path):
    class Distances(nn.Module):
        def __init__(self):
            super().__init__()
            self.points = nn.Parameter(torch.randn(5, 4))

        def forward(self, value):
            return torch.cdist(value, self.points)

    model = Distances().eval()
    value = torch.randn(3, 4)
    result = transpile(model, value, tmp_path / "cdist")
    assert verify(model, value, result).passed


class TwoInputsAndBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("offset", torch.tensor([1.0, -2.0, 3.0]))

    def forward(self, value, residual, *, scale):
        return 2 * value + residual * scale + self.offset


def test_one_tensor_passed_as_several_arguments_traces_one_input_each(tmp_path):
    # The tracer knows tensors by identity: the same object as two inputs,
    # or a buffer of the model as an input, must still give each argument
    # its own input list.
    model = TwoInputsAndBuffer().eval()
    shared = model.offset
    result = transpile(
        model, (shared, shared), tmp_path / "model",
        example_kwargs={"scale": shared},
    )
    distinct = (torch.tensor([0.5, 1.0, 2.0]), torch.tensor([-1.0, 4.0, 0.0]))
    check = verify(
        model, distinct, result, example_kwargs={"scale": torch.tensor([3.0, 0.0, 1.0])},
    )
    assert check.passed, check


class _IndexWrittenFrom(nn.Module):
    """An index input that the model also writes through a view of it."""

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.register_buffer("table", torch.arange(18.0).reshape(6, 3))

    def forward(self, ids, other):
        row = ids[0]
        if self.kind == "other":
            row[0] = other[0, 0]
        else:
            row[1] = 4
        return self.table[ids.view(-1)]


@pytest.mark.parametrize("kind", ["other", "constant"])
def test_an_index_input_written_through_a_view_checks_every_value_it_uses(tmp_path, kind):
    from cattorch import interface
    from cattorch.util.scratch.emulator import ScratchEmulator
    from sprite_helpers import load_sprite

    model = _IndexWrittenFrom(kind).eval()
    example = (torch.tensor([[1, 2, 3]]), torch.tensor([[5, 0, 0]]))
    result = transpile(model, tuple(value.clone() for value in example), tmp_path / kind)
    assert verify(model, tuple(value.clone() for value in example), result).passed
    emulator = ScratchEmulator(load_sprite(result.path))

    def status(ids, other):
        emulator.lists["input"], emulator.lists["input_1"] = ids, other
        emulator.run_procedure(interface.FORWARD)
        return emulator.variables[interface.STATUS_VARIABLE]

    assert status([1, 2, 3], [5, 0, 0]) == interface.OK
    # The input's own items are checked before the call.
    assert status([1, 9, 3], [5, 0, 0]) == interface.invalid_item("input")
    # A value written into it from another input is checked as it runs.
    expected = interface.invalid_item("input_1") if kind == "other" else interface.OK
    assert status([1, 2, 3], [9, 0, 0]) == expected
