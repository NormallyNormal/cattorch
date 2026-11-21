"""Bounded variable-length inputs for plain ``transpile``."""

from __future__ import annotations

import copy
import math

import torch

from cattorch.errors import UnsupportedModelError
from cattorch.interface import FORWARD, input_name, output_name


def normalize_dynamic(dynamic, flat_inputs, kwarg_names) -> dict[int, tuple[str, int, int, int]]:
    """``{flat input index: (dimension name, minimum, maximum, axis)}``.

    ``dynamic`` maps an input (its position, or the name of a tensor keyword)
    to ``(minimum, maximum)`` for its first dimension, or to
    ``{dimension: (minimum, maximum)}`` for another one.
    """
    if not isinstance(dynamic, dict) or not dynamic:
        raise TypeError("dynamic must be a non-empty dict, such as {0: (1, 64)}")
    positional = len(flat_inputs) - len(kwarg_names)
    result = {}
    for key, bounds in dynamic.items():
        if isinstance(key, str):
            if key not in kwarg_names:
                raise ValueError(f"dynamic names {key!r}, which isn't a tensor keyword input")
            index = positional + kwarg_names.index(key)
        elif isinstance(key, int) and not isinstance(key, bool) and 0 <= key < len(flat_inputs):
            index = key
        else:
            raise ValueError(f"dynamic key {key!r} is not an input position or keyword name")
        value = flat_inputs[index]
        if isinstance(bounds, dict):
            if len(bounds) != 1:
                raise ValueError("each input can have one dynamic dimension")
            ((axis, bounds),) = bounds.items()
        else:
            axis = 0
        if isinstance(axis, bool) or not isinstance(axis, int) or not 0 <= axis < value.ndim:
            raise ValueError(f"input {key!r} has no dimension {axis!r}")
        if math.prod(value.shape[:axis]) != 1:
            raise ValueError(
                f"input {key!r}'s dynamic dimension {axis} must come after dimensions of size 1"
            )
        if (
            not isinstance(bounds, tuple) or len(bounds) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in bounds)
            or not 1 <= bounds[0] <= bounds[1]
        ):
            raise ValueError(f"dynamic bounds for {key!r} must be (minimum, maximum) with 1 <= minimum <= maximum")
        name = input_name(index)
        result[index] = (name, bounds[0], bounds[1], axis)
    if len(result) > 1:
        raise UnsupportedModelError(
            "only one input can have a dynamic dimension; several independent "
            "lengths aren't supported yet"
        )
    return result


def at_length(value: torch.Tensor, axis: int, length: int) -> torch.Tensor:
    """``value`` with ``length`` items along ``axis``, repeating its rows."""
    rows = value.shape[axis]
    if rows == 0:
        raise ValueError("a dynamic example input needs at least one row")
    index = torch.arange(length) % rows
    return value.index_select(axis, index)


def varied_at_length(value: torch.Tensor, axis: int, length: int, seed: int = 0) -> torch.Tensor:
    """``value`` with ``length`` items along ``axis`` that differ from each other.

    Repeated rows would hide an operation that mixes items, such as a sort.
    Floats are drawn from the example's own values, and integers from the
    range it uses, so inputs stay in the range the model expects, and token
    IDs stay valid.
    """
    generator = torch.Generator().manual_seed(seed + length)
    repeated = at_length(value, axis, length)
    if value.dtype == torch.bool:
        return torch.rand(repeated.shape, generator=generator) > 0.5
    if value.is_floating_point():
        pool = value.flatten().unique()
        if pool.numel() == 1:
            # One value: vary it by its own size.
            scale = abs(float(pool[0])) or 1.0
            noise = torch.randn(repeated.shape, generator=generator, dtype=value.dtype)
            return pool[0] + 0.1 * scale * noise
        picks = torch.randint(0, pool.numel(), repeated.shape, generator=generator)
        return pool[picks]
    low, high = int(value.min()), int(value.max())
    if low == high:
        low = 0 if high > 0 else low
    return torch.randint(low, high + 1, repeated.shape, generator=generator, dtype=value.dtype)


_LONGEST_CHECKED = 1024


def reproducible_export(*, optimization, quantization, fast_config=None, registry=None) -> bool:
    """Whether a length check can reproduce the export's numbers exactly.

    Storage precision, ``sig_figs`` and symmetric quantization change only
    the stored values, which the reference model can hold too. Fast mode's
    built-in approximations are reproduced by running the reference under
    ``fast_reference`` (see there); its weight transforms, approximate
    kernels from ``register_kernel``, and GPTQ's calibration over inputs it
    has consumed can't be.
    """
    if quantization is not None and quantization.method != "symmetric":
        return False
    if optimization == "exact":
        return True
    from cattorch.fast import FastConfig
    from cattorch.operator_registry import default_registry

    config = fast_config or FastConfig()
    # The built-in approximate kernels are among those ``_FastReference``
    # reproduces; only ones passed to the export aren't.
    passed = set() if registry is None else (
        registry._approximate_targets - default_registry()._approximate_targets
    )
    return not config.has_weight_transforms and not passed


class _FastReference:
    """Run PyTorch as fast mode's built-in kernels compute.

    Fast mode replaces GELU with QuickGELU, ``x * sigmoid(1.702 x)``. Its
    one-pass LayerNorm variance and its softmax without the maximum are the
    same numbers in exact arithmetic, which a sprite's doubles come close
    to, so only GELU changes here.
    """

    def __init__(self, fast_config):
        self.active = fast_config.activations
        self.mode = None

    def __enter__(self):
        if self.active:
            from torch.utils._python_dispatch import TorchDispatchMode

            class QuickGelu(TorchDispatchMode):
                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    if func is torch.ops.aten.gelu.default:
                        value = args[0]
                        return value * torch.sigmoid(1.702 * value)
                    return func(*args, **(kwargs or {}))

            self.mode = QuickGelu()
            self.mode.__enter__()
        return self

    def __exit__(self, *exc):
        if self.mode is not None:
            self.mode.__exit__(*exc)
            self.mode = None
        return False


def fast_reference(optimization, fast_config):
    """A context in which PyTorch computes as the export's kernels do."""
    import contextlib

    from cattorch.fast import FastConfig

    if optimization != "fast":
        return contextlib.nullcontext()
    return _FastReference(fast_config or FastConfig())


# Tolerances for a sprite compared with a model holding its stored values,
# in ``verify``'s form (``atol + rtol * |expected|`` for each value), a
# little looser than its defaults because a long sum accumulates rounding.
# The absolute tolerance grows with the largest value compared: an output
# near zero among others near 1e5 differs by float32 noise of their size.
# The sprite computes in doubles, so the float32 reference's own error,
# which grows with the values it went through rather than those it
# returns (``h[-1] - h.mean(0)`` over rows near 3e4), is measured against a
# float64 run of the reference and allowed for too.
_EXACT_TOLERANCE = {"atol": 1e-3, "rtol": 1e-4}
_EXACT_MAGNITUDE_TOLERANCE = 1e-5


def _magnitude(leaves) -> float:
    return max(
        (
            float(leaf.detach().abs().nan_to_num(posinf=0.0, neginf=0.0).max())
            for leaf in leaves
            if isinstance(leaf, torch.Tensor) and leaf.numel() and leaf.is_floating_point()
        ),
        default=0.0,
    )


def _float32_error(leaves, precise_leaves) -> float:
    """The largest difference between float32 results and their float64 run."""
    if precise_leaves is None or len(leaves) != len(precise_leaves):
        return 0.0
    found = 0.0
    for leaf, precise in zip(leaves, precise_leaves):
        if (
            not isinstance(leaf, torch.Tensor) or not isinstance(precise, torch.Tensor)
            or not leaf.is_floating_point() or leaf.shape != precise.shape or not leaf.numel()
        ):
            continue
        difference = (leaf.double() - precise.double()).abs()
        difference = difference[torch.isfinite(difference)]
        if difference.numel():
            found = max(found, float(difference.max()))
    return found


def tolerance_for(leaves, precise_leaves=None) -> dict[str, float]:
    """``_EXACT_TOLERANCE`` for comparing a sprite with ``leaves``.

    ``atol`` is at least ``_EXACT_MAGNITUDE_TOLERANCE`` of their largest
    magnitude, plus twice the error of the float32 ``leaves`` measured
    against ``precise_leaves``, the same results computed in float64.
    """
    return {
        **_EXACT_TOLERANCE,
        "atol": max(_EXACT_TOLERANCE["atol"], _EXACT_MAGNITUDE_TOLERANCE * _magnitude(leaves))
        + 2 * _float32_error(leaves, precise_leaves),
    }


def _float64(value):
    """``value`` with its floating tensors in float64, at any depth."""
    from torch.utils import _pytree as pytree

    return pytree.tree_map(
        lambda leaf: leaf.double()
        if isinstance(leaf, torch.Tensor) and leaf.is_floating_point() else leaf,
        value,
    )


def float64_copy(model):
    """A float64 copy of ``model``, or ``None`` if it can't be made."""
    try:
        return copy.deepcopy(model).double()
    except Exception:  # noqa: BLE001 - the float32 tolerance applies then
        return None


def _exact_tolerance(reference, positional, keywords) -> dict[str, float]:
    """``tolerance_for`` the reference's output on these inputs."""
    from cattorch.frontend import tensor_output_leaves
    from cattorch.verify import _reading_stored_uses

    def run(model, values, named):
        with torch.no_grad(), _reading_stored_uses(model):
            return tensor_output_leaves(model(*values, **named), warn=False)

    try:
        leaves = run(copy.deepcopy(reference), positional, keywords)
    except Exception:
        # A model that fails at this length is reported by verify, which
        # runs it next and says so in the user's terms.
        return dict(_EXACT_TOLERANCE)
    precise = None
    double = float64_copy(reference)
    if double is not None:
        try:
            precise = run(double, _float64(tuple(positional)), _float64(dict(keywords)))
        except Exception:  # noqa: BLE001 - not every model runs in float64
            precise = None
    return tolerance_for(leaves, precise)


def _stored_values_model(
    model, positional, example_kwargs, *, storage, quantization, sig_figs, registry,
):
    """A copy of ``model`` holding the values the sprite stores.

    It is prepared as the export prepares its copy (adapters, folded batch
    norms, quantized weights), then rounded as storage rounds it, so its
    output differs from the sprite's only by float32 arithmetic.
    """
    from cattorch.graph_transforms import fold_eval_batch_norms
    from cattorch.quantization import quantize_model
    from cattorch.storage import StorageConfig
    from cattorch.verify import round_as_stored

    owned = copy.deepcopy(model).eval()
    lossy = (
        quantization is not None or sig_figs is not None
        or getattr(storage, "precision", "float32") != "float32"
    )
    if not lossy:
        return owned
    owned = fold_eval_batch_norms(registry.adapt_model(owned, _copy=False))
    report = None
    if quantization is not None:
        report = quantize_model(
            owned, quantization, None, orientation_inputs=[tuple(positional)],
            call_kwargs=example_kwargs, _registry=registry,
        )
    round_as_stored(
        owned, storage or StorageConfig(), quantization=quantization, sig_figs=sig_figs,
        report=report,
    )
    return owned


_USES_LENGTH = (
    "One possible cause is that the model uses the length itself, for example "
    "torch.arange or a position table sliced to it, which dynamic= can't follow "
    "(a GenerationProgram handles variable-length token sequences); if it "
    "doesn't, please report the model"
)


def _beyond_baseline(found: float, baseline: float, length: int, base_length: int) -> bool:
    """Whether an export the check can't reproduce differs too much at ``length``.

    Its error at ``base_length`` items is the baseline. An approximation's
    error in each item adds up over a sum of them, so the baseline grows in
    proportion to the length.
    """
    return found > 10 * baseline * max(1.0, length / max(base_length, 1)) + 1e-4


def check_lengths(
    model, flat_inputs, kwarg_names, spec, result, verify, example_kwargs, *,
    optimization="exact", storage=None, quantization=None, sig_figs=None,
    adapters=(), kernels=(), fast_config=None,
) -> None:
    """Compare the sprite with PyTorch at lengths other than the traced one.

    The trace saw one row, so a model that builds something from the length
    itself (``torch.arange(n)``, a position table, a reshape over it) is
    exported for one row only. Comparing at other lengths finds that. The
    sprite is compared with a copy of the model holding the values the
    sprite stores, run as fast mode's kernels compute (see
    ``reproducible_export``), and must match it at every length. An export
    the check can't reproduce is compared with the model, taking the error
    at the shortest length as the baseline (see ``_beyond_baseline``), so
    weight transforms, approximate kernels and GPTQ don't trip the check.
    """
    from cattorch.operator_registry import default_registry

    ((index, (_, minimum, maximum, axis)),) = spec.items()
    lengths = _lengths(minimum, flat_inputs[index].shape[axis], maximum)
    positional = len(flat_inputs) - len(kwarg_names)
    registry = default_registry().for_export(adapters=adapters, kernels=kernels)
    exact = reproducible_export(
        optimization=optimization, quantization=quantization, fast_config=fast_config,
        registry=registry,
    )
    reference = model
    if exact:
        reference = _stored_values_model(
            model, flat_inputs[:positional], example_kwargs, storage=storage,
            quantization=quantization, sig_figs=sig_figs, registry=registry,
        )

    def report(length):
        values = list(flat_inputs)
        values[index] = varied_at_length(values[index], axis, length)
        keywords = dict(example_kwargs or {})
        keywords.update(zip(kwarg_names, values[positional:]))
        with fast_reference(optimization if exact else "exact", fast_config):
            tolerance = (
                _exact_tolerance(reference, tuple(values[:positional]), keywords)
                if exact else {"atol": math.inf, "rtol": math.inf}
            )
            # verify runs the model; a fresh copy each time keeps a forward
            # that changes module state away from the caller's model and
            # later runs.
            try:
                return verify(
                    copy.deepcopy(reference), tuple(values[:positional]), result,
                    example_kwargs=keywords or None, **tolerance,
                )
            except UnsupportedModelError as error:
                raise UnsupportedModelError(
                    f"the export is checked at several lengths within the dynamic bounds, "
                    f"and with {length} items along the dynamic dimension: {error}"
                ) from error

    def error(outcome):
        """The largest difference, and whether it is within tolerance."""
        reports = getattr(outcome, "outputs", None) or (outcome,)
        if any(item.values_compared != item.actual_values for item in reports):
            return math.inf, False
        return max(item.max_abs_error for item in reports), outcome.passed

    baseline, matches = error(report(lengths[0]))
    if not matches:
        raise UnsupportedModelError(
            f"with {lengths[0]} items along the dynamic dimension the sprite doesn't "
            f"match the model (largest difference {baseline:.3g})"
        )
    for length in lengths[1:]:
        found, matches = error(report(length))
        if not matches or (not exact and _beyond_baseline(found, baseline, length, lengths[0])):
            cause = (
                _CHANGES_OPERATIONS
                if _operations_change(model, flat_inputs, kwarg_names, spec, example_kwargs)
                else _USES_LENGTH
            )
            raise UnsupportedModelError(
                f"with {length} items along the dynamic dimension the export gives "
                f"different numbers from the model (largest difference {found:.3g}), "
                f"though it matches with {lengths[0]}. {cause}"
            )


_CHANGES_OPERATIONS = (
    "The model runs other operations with two items than with the one the "
    "export traces, as when a slice past the first item, such as x[:, 1:], or "
    "the rest of a tensor after a write to its first row, holds nothing with "
    "one item; dynamic= can't follow such an operation"
)


def _operations_change(model, flat_inputs, kwarg_names, spec, example_kwargs) -> bool:
    """Whether the model's traced operations differ between one and two items."""
    from cattorch.frontend import capture_model
    from cattorch.graph import _dynamic_placeholders, _normalize_graph

    ((index, (_, _, _, axis)),) = spec.items()
    positional = len(flat_inputs) - len(kwarg_names)

    def operations(length):
        values = list(flat_inputs)
        values[index] = varied_at_length(values[index], axis, length)
        keywords = dict(example_kwargs or {})
        keywords.update(zip(kwarg_names, values[positional:]))
        captured = capture_model(
            copy.deepcopy(model), tuple(values[:positional]), example_kwargs=keywords or None,
        )
        _normalize_graph(captured.graph, varying=_dynamic_placeholders(captured, spec))
        return [(node.op, str(node.target)) for node in captured.graph.nodes]

    try:
        return operations(1) != operations(2)
    except Exception:  # noqa: BLE001 - without both traces, the cause stays open
        return False


def _lengths(minimum, original, maximum) -> list[int]:
    """Lengths to check, the first one as the baseline.

    The export traces one item and literals at two, so three items or more
    are needed to catch a wrong extrapolation. The maximum catches capacity
    mistakes and behaviour that starts past some length, up to
    ``_LONGEST_CHECKED`` items to keep the emulator's run time bounded.
    """
    longest = min(maximum, max(_LONGEST_CHECKED, original))
    return sorted({
        length for length in (minimum, 1, 2, 3, original, longest)
        if max(minimum, 1) <= length <= maximum
    })


def _state_at_length(state, length: int) -> torch.Tensor:
    """An append state's value with ``length`` rows that differ from each other."""
    initial = state.initial
    if initial.shape[0] == 0:
        initial = initial.new_zeros((1, *initial.shape[1:]))
    return varied_at_length(initial, 0, length, seed=7)


def _entry_reference(
    model, program, registry, *, optimization, fast_config, quantization, storage, sig_figs,
    gptq_reference, gptq_report,
):
    """``(reference model, whether the check reproduces the export)``."""
    from cattorch.storage import StorageConfig
    from cattorch.verify import _program_reference, round_as_stored

    exact = reproducible_export(
        optimization=optimization, quantization=quantization, fast_config=fast_config,
        registry=registry,
    )
    if gptq_reference is not None:
        # The export's own copy, holding the weights GPTQ chose, rounded as
        # storage keeps them, as symmetric quantization's reference is.
        reference = copy.deepcopy(gptq_reference)
        round_as_stored(
            reference, storage or StorageConfig(), quantization=quantization,
            sig_figs=sig_figs, report=gptq_report,
        )
        return reference, reproducible_export(
            optimization=optimization, quantization=None, fast_config=fast_config,
            registry=registry,
        )
    if quantization is not None and quantization.method != "symmetric":
        # Without the export's weights, the float model as storage keeps it.
        quantization = None
    return _program_reference(
        model, program, registry, quantization=quantization, storage=storage, sig_figs=sig_figs,
    ), exact


def check_entry_lengths(
    model, program, result, *, adapters=(), optimization="exact", storage=None,
    quantization=None, sig_figs=None, fast_config=None, kernels=(),
    gptq_reference=None, gptq_report=None,
) -> None:
    """Compare a program's entry points with PyTorch at lengths other than the traced one.

    Every variable-length input of an entry point, a ``dynamic=`` input or
    an append state it reads, is checked at several lengths on its own, the
    others at their examples and initial values, and then all together. An
    append state that the entry point grows is also checked as long as it
    can be before that call fills it. Each check calls the entry point once
    in the emulator, with its state lists set to the values being checked,
    and compares its outputs and updated states with the method's, run on a
    fresh copy of a model holding the values the sprite stores. A GPTQ
    export gives its own quantized copy of the model as ``gptq_reference``,
    with the ``QuantizationReport`` its quantization gave as ``gptq_report``.
    """
    from cattorch.experts import _expert_family_context
    from cattorch.interface import OK, STATUS_VARIABLE
    from cattorch.operator_registry import default_registry
    from cattorch.program import Input, Output, ProgramCall, StateInput, StateUpdate
    from cattorch.program_runtime import program_replay
    from cattorch.util.scratch.archive import read_json
    from cattorch.util.scratch.emulator import ScratchEmulator
    from cattorch.verify import (
        _compare_values, _get_logical_list, _reading_stored_uses, _set_logical_list,
    )

    registry = default_registry().for_export(adapters=adapters, kernels=kernels)
    reference, exact = _entry_reference(
        model, program, registry, optimization=optimization, fast_config=fast_config,
        quantization=quantization, storage=storage, sig_figs=sig_figs,
        gptq_reference=gptq_reference, gptq_report=gptq_report,
    )
    double = float64_copy(reference) if exact else None
    computing = fast_reference(optimization if exact else "exact", fast_config)
    emulator = ScratchEmulator(read_json(result.path))
    states = {state.name: state for state in program.states}

    def replayed(model, entry, inputs, overrides, *, in_float64=False):
        """``(results, states after)`` of one call on a fresh copy of ``model``."""
        replay = program_replay(copy.deepcopy(model), program)
        starting = {
            state.name: state.initial.detach().clone() for state in program.states
        }
        starting.update(overrides)
        if in_float64:
            # The inputs are checked against their contract as given, then
            # computed with in float64.
            checked = replay.checked_inputs
            replay.checked_inputs = (  # type: ignore[method-assign]
                lambda entry, values: _float64(checked(entry, values))
            )
            starting = _float64(starting)
        replay.states = starting
        with (
            _expert_family_context(registry.expert_families),
            _reading_stored_uses(replay.model, registry),
            computing,
        ):
            leaves = replay.run(ProgramCall(entry.name, tuple(inputs)))
        return leaves, dict(replay.states)

    def precise(entry, inputs, overrides):
        """The same call in float64, or ``None``."""
        if double is None:
            return None
        try:
            return replayed(double, entry, inputs, overrides, in_float64=True)
        except Exception:  # noqa: BLE001 - not every model runs in float64
            return None

    def appended(entry, inputs) -> dict[str, int]:
        """The rows the entry point adds to each append state, from its initial value."""
        try:
            _, after = replayed(reference, entry, inputs, {})
        except ValueError:
            return {}
        return {
            name: int(after[name].shape[0] - state.initial.shape[0])
            for name, state in states.items() if state.mode == "append"
        }

    def error(entry, inputs, overrides):
        """The largest difference of one call, whether it matches, and whether it ran."""
        replay_states = {
            state.name: state.initial.detach().clone() for state in program.states
        }
        replay_states.update(overrides)
        for name, value in replay_states.items():
            _set_logical_list(emulator, f"cattorch state {name}", value.flatten().tolist())
        bindings = [binding for binding in entry.arguments if isinstance(binding, Input)]
        for binding, value in zip(bindings, inputs):
            _set_logical_list(emulator, f"cattorch {entry.name} {binding.name}", value.flatten().tolist())
        emulator.call(f"cattorch {entry.name}")
        status = str(emulator.variables.get(emulator.public(STATUS_VARIABLE, is_list=False), ""))
        try:
            leaves, after = replayed(reference, entry, inputs, overrides)
        except ValueError as refused:
            if "capacity exceeded" not in str(refused):
                raise
            # A call past an append state's capacity must be refused, and
            # then has nothing to compare.
            return (0.0, True, False) if status != OK else (math.inf, False, True)
        if status != OK:
            return math.inf, False, True
        exactly = precise(entry, inputs, overrides)
        expected = [
            (leaf, None if exactly is None else exactly[0][position],
             f"cattorch {entry.name} {binding.name}")
            for position, (binding, leaf) in enumerate(zip(entry.returns, leaves))
            if isinstance(binding, Output)
        ]
        expected += [
            (after[name], None if exactly is None else exactly[1][name], f"cattorch state {name}")
            for name in states
        ]
        reports = [
            _compare_values(
                leaf, _get_logical_list(emulator, list_name),
                **(
                    tolerance_for([leaf], None if accurate is None else [accurate])
                    if exact else {"atol": math.inf, "rtol": math.inf}
                ),
            )
            for leaf, accurate, list_name in expected
        ]
        if any(report.values_compared != report.actual_values for report in reports):
            return math.inf, False, True
        found = max((report.max_abs_error for report in reports), default=0.0)
        return found, all(report.passed for report in reports), True

    for entry in program.entrypoints:
        inputs = [binding for binding in entry.arguments if isinstance(binding, Input)]
        grows = {
            binding.name for binding in entry.returns
            if isinstance(binding, StateUpdate) and states[binding.name].mode == "append"
        }
        examples_given = [binding.example for binding in inputs]
        added = appended(entry, examples_given) if grows else {}
        # ``(what it is, its key, lengths to check)``: an input's position,
        # or an append state's name.
        varying: list[tuple[str, int | str, list[int]]] = []
        for position, binding in enumerate(inputs):
            if binding.dynamic is not None:
                ((axis, (minimum, maximum)),) = binding.dynamic.items()
                varying.append((
                    f"input {binding.name!r}", position,
                    _lengths(minimum, binding.example.shape[axis], maximum),
                ))
        for binding in entry.arguments:
            state = states.get(binding.name) if isinstance(binding, StateInput) else None
            if state is not None and state.mode == "append":
                rows = int(state.initial.shape[0])
                capacity = int(state.capacity or rows)
                lengths = set(_lengths(rows, rows, capacity))
                if state.name in grows:
                    # At the capacity the call that grows the state is
                    # refused and compares nothing, so also check the
                    # longest it can be before the call, and halfway there.
                    fullest = capacity - added.get(state.name, 0)
                    lengths.update(
                        length for length in (fullest, (rows + fullest) // 2)
                        if rows <= length <= capacity
                    )
                varying.append((f"state {state.name!r}", state.name, sorted(lengths)))
        if not varying:
            continue

        def setting(settings):
            """The inputs and states for each ``(key, length)`` in ``settings``."""
            values = list(examples_given)
            overrides = {}
            for key, length in settings:
                if isinstance(key, str):
                    overrides[key] = _state_at_length(states[key], length)
                else:
                    ((axis, _),) = inputs[key].dynamic.items()
                    values[key] = varied_at_length(inputs[key].example, axis, length)
            return values, overrides

        def run(settings):
            """One call with each ``(key, length)`` in ``settings`` at that length."""
            values, overrides = setting(settings)
            return error(entry, values, overrides)

        def fitted(settings):
            """``settings`` with each state the call grows short enough to grow."""
            values, _ = setting([(key, length) for key, length in settings if not isinstance(key, str)])
            growth = appended(entry, values)
            return [
                (key, max(
                    int(states[key].initial.shape[0]),
                    min(length, int(states[key].capacity) - growth.get(key, 0)),
                ))
                if isinstance(key, str) and key in grows else (key, length)
                for key, length in settings
            ]

        def described(parts):
            return " and ".join(
                f"{length} {'row' if length == 1 else 'rows'} in {what}" for what, length in parts
            )

        examples = (
            "for example torch.arange over it, a table sliced to it, or a slice of "
            "one tensor to another's length"
        )
        baseline, matches, _ = run([])
        if not matches:
            names = " or ".join(what for what, _, _ in varying)
            raise UnsupportedModelError(
                f"entry point {entry.name!r} with its example inputs and the states' "
                "initial values fails or doesn't match the method in the sprite (largest "
                f"difference {baseline:.3g}). The method may compute something from the "
                f"length of {names}, which the export traces with one row, that the "
                f"sprite can't follow, {examples}"
            )
        base_length = max(
            [int(states[key].initial.shape[0]) if isinstance(key, str) else int(
                inputs[key].example.shape[next(iter(inputs[key].dynamic))]
            ) for _, key, _ in varying] + [1]
        )
        checks = [
            [(what, key, length)] for what, key, lengths in varying for length in lengths
        ]
        if len(varying) > 1:
            # All together, at a short and at the longest checked length.
            checks += [
                [(what, key, pick(lengths)) for what, key, lengths in varying]
                for pick in (lambda lengths: min(lengths, key=lambda value: abs(value - 3)), max)
            ]
            if grows:
                # And at the longest, with the states it grows short enough
                # for the call to be compared.
                longest = [(key, max(lengths)) for _, key, lengths in varying]
                whats = [what for what, _, _ in varying]
                checks.append([
                    (what, key, length) for what, (key, length) in zip(whats, fitted(longest))
                ])
        for check in checks:
            found, matches, _ = run([(key, length) for _, key, length in check])
            length = max(length for _, _, length in check)
            if not matches or (not exact and _beyond_baseline(found, baseline, length, base_length)):
                raise UnsupportedModelError(
                    f"entry point {entry.name!r} with {described((what, length) for what, _, length in check)} "
                    f"gives different numbers in the sprite from the method (largest "
                    f"difference {found:.3g}), though it matches with its example inputs and "
                    "the states' initial values. One possible cause is that the method "
                    f"computes something from that length that the sprite can't follow, "
                    f"{examples}; if it doesn't, please report the model"
                )


def unit_output_axes(model, flat_inputs, kwarg_names, spec, example_kwargs) -> list[int]:
    """Each output's axis for the dynamic input's units, checked eagerly.

    The model must treat units independently: its output for two units is its
    outputs for each unit joined along an axis with only size-1 dimensions
    before it.
    """
    from cattorch.frontend import tensor_output_leaves

    ((index, (_, _, _, axis)),) = spec.items()
    positional = len(flat_inputs) - len(kwarg_names)
    generator = torch.Generator().manual_seed(0)
    source = flat_inputs[index]
    shape = list(source.shape)
    shape[axis] = 2
    if source.is_floating_point():
        pair = torch.randn(shape, generator=generator, dtype=source.dtype)
    else:
        pair = at_length(source, axis, 2)
        first, second = pair.narrow(axis, 0, 1), pair.narrow(axis, 1, 1)
        if torch.equal(first, second):
            # One example item: make the second unit differ, with values the
            # example already uses, so mixing across units shows.
            second = first.flatten().roll(1).reshape(first.shape)
            if torch.equal(first, second) and source.dtype != torch.bool and int(source.max()) > 0:
                second = torch.randint(
                    0, int(source.max()) + 1, tuple(first.shape), generator=generator,
                    dtype=source.dtype,
                )
            pair = torch.cat([first, second], axis)

    def run(value):
        values = list(flat_inputs)
        values[index] = value
        keywords = dict(example_kwargs or {})
        keywords.update(zip(kwarg_names, values[positional:]))
        # Each run on a fresh copy: a forward that changes module state
        # must not reach the caller's model, nor the next run.
        with torch.no_grad():
            return tensor_output_leaves(
                copy.deepcopy(model)(*values[:positional], **keywords), warn=False,
            )

    both = run(pair)
    first, second = run(pair.narrow(axis, 0, 1)), run(pair.narrow(axis, 1, 1))
    axes = []
    for whole, one, other in zip(both, first, second):
        found = next(
            (
                dim for dim in range(one.ndim)
                if math.prod(one.shape[:dim]) == 1 and one.shape[dim] * 2 == whole.shape[dim]
                and torch.allclose(whole, torch.cat([one, other], dim), equal_nan=True, atol=1e-6)
            ),
            None,
        )
        if found is None:
            raise UnsupportedModelError(
                "the model mixes values across the dynamic dimension, so its "
                "length can't vary: the output for two items isn't the outputs "
                "for each item joined together"
            )
        axes.append(found)
    return axes


def run_per_unit(sprite: dict, spec, input_shapes, output_count: int) -> dict:
    """Make ``cattorch forward`` run the one-unit forward once per unit.

    ``sprite`` is a fixed export for one unit, before its interface is
    finished. Its forward becomes the internal ``_cattorch forward unit``,
    reading the dynamic input and writing the outputs through internal
    lists, and a new ``cattorch forward`` copies each unit in, calls it, and
    appends each unit's outputs. The export checks the input's length, as
    ``_unit_inputs`` gives it, around the new forward. The sprite is changed
    in place and returned.
    """
    from cattorch.procedures import _add_warp_procedure, _procedure_mutation
    from cattorch.util.scratch.dsl import (
        Program, add, append, call, change_var, clear, div, for_each, item, length, mul,
        set_var, var,
    )
    from cattorch.util.scratch.interface import rename_lists, rename_procedure

    ((index, _),) = spec.items()
    names = {entry[0] for entry in sprite.get("lists", {}).values()}
    if any(" shard " in value for value in names if value.startswith(("input", "output"))):
        raise UnsupportedModelError("a dynamic export's inputs and outputs must fit in one list each")

    unit, position = "_cattorch unit", "_cattorch unit item"
    public_input = input_name(index)
    private_input = f"_cattorch unit {public_input}"
    rename_procedure(sprite, FORWARD, "_cattorch forward unit")
    input_size = math.prod(input_shapes[index])
    outputs = []
    for position_index in range(output_count):
        public = output_name(position_index)
        outputs.append((public, f"_cattorch unit {public}"))
    rename_lists(sprite, {public_input: private_input, **dict(outputs)})
    body = (
        *(clear(public) for public, _ in outputs),
        set_var(unit, 0),
        for_each("_cattorch units", div(length(public_input), input_size), (
            clear(private_input),
            for_each(position, input_size, (
                append(private_input, item(public_input, add(mul(var(unit), input_size), var(position)))),
            )),
            call("_cattorch forward unit"),
            *(
                for_each(position, length(private), (append(public, item(private, var(position))),))
                for public, private in outputs
            ),
            change_var(unit, 1),
        )),
    )
    program = Program(
        "forward_by_unit",
        variables=(unit, position, "_cattorch units"),
        lists=(public_input, private_input, *(entry for pair in outputs for entry in pair)),
        body=body,
    )
    sprite = _add_warp_procedure(sprite, FORWARD, program, x=640, y=40)
    # The loose forward block on the canvas runs the public block.
    for block in sprite["blocks"].values():
        if (
            block.get("opcode") == "procedures_call" and block.get("topLevel")
            and block["mutation"].get("proccode") == "_cattorch forward unit"
        ):
            block["mutation"] = _procedure_mutation(FORWARD)
    return sprite


def per_unit_result(result, spec, output_axes, input_shapes, output_shapes):
    """``result`` of ``run_per_unit``'s export, with the dynamic dimension bounded."""
    import dataclasses

    from cattorch.results import BoundedDimension, TensorSpec

    ((index, (name, minimum, maximum, axis)),) = spec.items()

    def bounded(spec_value, shape, dim):
        public_shape = list(shape)
        # Each item can give several rows, as reshape(-1, 2) of four features does.
        rows = shape[dim]
        public_shape[dim] = BoundedDimension(name, minimum * rows, maximum * rows)
        return TensorSpec(
            list_name=spec_value.list_name, shape=tuple(public_shape), dtype=spec_value.dtype,
            numel=None, max_numel=math.prod(shape) * maximum,
        )

    inputs = tuple(
        bounded(value, input_shapes[position_index], axis) if position_index == index else value
        for position_index, value in enumerate(result.inputs)
    )
    specs = [
        bounded(value, output_shapes[position_index], output_axes[position_index])
        for position_index, value in enumerate(result.outputs)
    ]
    return dataclasses.replace(
        result,
        inputs=inputs,
        outputs=tuple(specs),
        warnings=result.warnings + (
            "the dynamic dimension runs the model once per item, because its "
            "operations can't take a variable length directly",
        ),
    )
