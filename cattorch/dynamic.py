"""Bounded variable-length inputs for plain ``transpile``."""

from __future__ import annotations

import math

import torch

from cattorch.errors import UnsupportedModelError
from cattorch.interface import STATUS_VARIABLE, input_name, output_name


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


def exact_export(*, optimization, storage, quantization, sig_figs) -> bool:
    """Whether an export's numbers should match PyTorch to float32 rounding."""
    precision = getattr(storage, "precision", "float32")
    return (
        optimization == "exact" and quantization is None and precision == "float32"
        and sig_figs is None
    )


# Tolerances for an export that should match PyTorch exactly, a little looser
# than ``verify``'s because a long sum accumulates float32 rounding.
_EXACT_TOLERANCE = {"atol": 1e-3, "rtol": 1e-4}


def check_lengths(
    model, flat_inputs, kwarg_names, spec, result, verify, example_kwargs, *, exact=False,
) -> None:
    """Compare the sprite with PyTorch at lengths other than the traced one.

    The trace saw one row, so a model that builds something from the length
    itself (``torch.arange(n)``, a position table, a reshape over it) is
    exported for one row only. Comparing at other lengths finds that. An
    ``exact`` export must match at every length. Otherwise the error at the
    shortest length is the baseline, so approximate kernels and quantization
    don't trip the check.
    """
    ((index, (_, minimum, maximum, axis)),) = spec.items()
    lengths = _lengths(minimum, flat_inputs[index].shape[axis], maximum)
    positional = len(flat_inputs) - len(kwarg_names)

    def report(length):
        values = list(flat_inputs)
        values[index] = varied_at_length(values[index], axis, length)
        keywords = dict(example_kwargs or {})
        keywords.update(zip(kwarg_names, values[positional:]))
        tolerance = _EXACT_TOLERANCE if exact else {"atol": math.inf, "rtol": math.inf}
        return verify(
            model, tuple(values[:positional]), result,
            example_kwargs=keywords or None, **tolerance,
        )

    def error(outcome):
        """The largest difference, and whether an exact export is within tolerance."""
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
        if not matches or found > 10 * baseline + 1e-4:
            raise UnsupportedModelError(
                f"with {length} items along the dynamic dimension the sprite doesn't "
                f"match the model (largest difference {found:.3g}), though it does with "
                f"{lengths[0]}. The model probably uses the length itself, for example "
                "torch.arange or a position table sliced to it, which dynamic= can't "
                "follow; a GenerationProgram handles variable-length token sequences"
            )


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


def check_entry_lengths(
    model, program, result, verify, *, adapters=(), quantization=None, exact=False,
) -> None:
    """``check_lengths`` for each variable-length input of a program's entry points.

    Each check calls the entry point once, from the initial state, with that
    input at another length and the other inputs at their examples.
    """
    from cattorch.program import Input, ProgramCall

    if quantization is not None and quantization.method != "symmetric":
        return  # verify can't reproduce other quantization methods
    for entry in program.entrypoints:
        inputs = [binding for binding in entry.arguments if isinstance(binding, Input)]
        for position, binding in enumerate(inputs):
            if binding.dynamic is None:
                continue
            minimum, maximum = binding.dynamic

            def error(length):
                values = [value.example for value in inputs]
                values[position] = varied_at_length(binding.example, 0, length)
                tolerance = _EXACT_TOLERANCE if exact else {"atol": math.inf, "rtol": math.inf}
                outcome = verify(
                    model, program, result, calls=(ProgramCall(entry.name, tuple(values)),),
                    adapters=adapters, quantization=quantization, **tolerance,
                )
                (call,) = outcome.calls
                reports = [*call.outputs, *(report for _, report in call.states)]
                if call.status != "ok" or any(
                    report.values_compared != report.actual_values for report in reports
                ):
                    return math.inf, False
                found = max((report.max_abs_error for report in reports), default=0.0)
                return found, outcome.passed

            lengths = _lengths(minimum, binding.example.shape[0], maximum)
            baseline, matches = error(lengths[0])
            where = f"input {binding.name!r} of entry point {entry.name!r}"
            if not matches:
                raise UnsupportedModelError(
                    f"with {lengths[0]} items in {where} the sprite fails or doesn't "
                    f"match the model (largest difference {baseline:.3g})"
                )
            for length in lengths[1:]:
                found, matches = error(length)
                if not matches or found > 10 * baseline + 1e-4:
                    raise UnsupportedModelError(
                        f"with {length} items in {where} the sprite doesn't match the model "
                        f"(largest difference {found:.3g}), though it does with {lengths[0]}. "
                        "The method probably uses the length itself, for example "
                        "torch.arange or a table sliced to it, which a dynamic input can't follow"
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
        with torch.no_grad():
            return tensor_output_leaves(model(*values[:positional], **keywords), warn=False)

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


def loop_over_units(result, spec, output_axes, input_shapes, output_shapes):
    """Make ``cattorch forward`` run the one-unit forward once per unit.

    ``result`` is a fixed export for one unit. Its forward becomes the
    internal ``_cattorch forward unit``, reading the dynamic input and writing
    the outputs through internal lists, and a new ``cattorch forward`` copies
    each unit in, calls it, and appends each unit's outputs.
    """
    import dataclasses
    import json
    import zipfile

    from cattorch.results import BoundedDimension, TensorSpec
    from cattorch.sprite import (
        _add_warp_procedure, _remove_completion_broadcast,
    )
    from cattorch.util.scratch.dsl import (
        Program, add, append, broadcast_and_wait, call, change_var, clear, div, eq, for_each,
        gt, if_, item, length, lt, mod, mul, set_var, var,
    )
    from cattorch.util.scratch.interface import rename_list, rename_procedure

    ((index, (name, minimum, maximum, axis)),) = spec.items()
    with zipfile.ZipFile(result.path) as archive:
        members = {entry: archive.read(entry) for entry in archive.namelist()}
    sprite = json.loads(members["sprite.json"])
    names = {entry[0] for entry in sprite.get("lists", {}).values()}
    if any(" shard " in value for value in names if value.startswith(("input", "output"))):
        raise UnsupportedModelError("a dynamic export's inputs and outputs must fit in one list each")

    unit, position = "_cattorch unit", "_cattorch unit item"
    public_input = input_name(index)
    private_input = f"_cattorch unit {public_input}"
    rename_procedure(sprite, "cattorch forward", "_cattorch forward unit")
    # The new forward reports completion once, after every unit.
    _remove_completion_broadcast(sprite, "cattorch forward complete")
    rename_list(sprite, public_input, private_input)
    input_size = math.prod(input_shapes[index])
    outputs = []
    for position_index, shape in enumerate(output_shapes):
        public = output_name(position_index)
        private = f"_cattorch unit {public}"
        rename_list(sprite, public, private)
        outputs.append((public, private, math.prod(shape)))
    run_units = (
        set_var(unit, 0),
        for_each("_cattorch units", div(length(public_input), input_size), (
            clear(private_input),
            for_each(position, input_size, (
                append(private_input, item(public_input, add(mul(var(unit), input_size), var(position)))),
            )),
            call("_cattorch forward unit"),
            *(
                for_each(position, length(private), (append(public, item(private, var(position))),))
                for public, private, _ in outputs
            ),
            change_var(unit, 1),
        )),
    )
    # Whole items within the bounds, as an exported forward checks them.
    status = STATUS_VARIABLE
    body = [
        *(clear(public) for public, _, _ in outputs),
        set_var(status, "ok"),
        if_(gt(mod(length(public_input), input_size), 0), (
            set_var(status, f"invalid dynamic length: {public_input}"),
        )),
        if_(lt(length(public_input), input_size * max(minimum, 1)), (
            set_var(status, f"too few items: {public_input}"),
        )),
        if_(gt(length(public_input), input_size * maximum), (
            set_var(status, f"dynamic capacity exceeded: {public_input}"),
        )),
        if_(eq(var(status), "ok"), run_units),
        broadcast_and_wait("cattorch forward complete"),
    ]
    program = Program(
        "forward_by_unit",
        variables=(unit, position, "_cattorch units", STATUS_VARIABLE),
        lists=(public_input, private_input, *(entry for pair in outputs for entry in pair[:2])),
        body=tuple(body),
    )
    sprite = _add_warp_procedure(sprite, "cattorch forward", program, x=640, y=40)
    data = json.dumps(sprite, separators=(",", ":"))
    members["sprite.json"] = data.encode()
    with zipfile.ZipFile(result.path, "w", zipfile.ZIP_DEFLATED) as archive:
        for entry, content in members.items():
            archive.writestr(entry, content)

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
        output=specs[0],
        additional_outputs=tuple(specs[1:]),
        archive_bytes=result.path.stat().st_size,
        expanded_json_bytes=len(data.encode()),
        block_count=len(sprite["blocks"]),
        warnings=result.warnings + (
            "the dynamic dimension runs the model once per item, because its "
            "operations can't take a variable length directly",
        ),
    )
