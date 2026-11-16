"""Numerically verify generated Scratch sprites against PyTorch."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence, cast

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cattorch.errors import UnsupportedModelError
from cattorch.adapters import ModuleAdapter
from cattorch.export_options import validate_export_options, validate_model_and_inputs
from cattorch.frontend import flat_example_inputs, model_failure, tensor_output_leaves
from cattorch.huggingface import prepare_huggingface_models
from cattorch.operator_registry import default_registry, eval_mode_copy_warning
from cattorch.quantization import (
    QuantizationConfig, expert_bank_effective_ndims, quantize_model,
)
from cattorch.storage import StorageConfig, quantized_storage
from cattorch.program import (
    ExportProgram, GenerationProgram, Input, Output, ProgramCall, RecurrentGenerationProgram, State,
)
from cattorch.program_runtime import program_replay
from cattorch.graph_transforms import fold_eval_batch_norms
from cattorch.results import (
    GenerationResult, ProgramCallVerifyResult, ProgramResult, ProgramVerifyResult,
    TranspileResult, VerifyResult, MultiOutputVerifyResult,
)
from cattorch.util.paths import output_file
from cattorch.util.scratch.archive import SPRITE_JSON, read_json, read_user_archive
from cattorch.util.scratch.emulator import ScratchEmulator
from cattorch.util.scratch.interface import read_logical_list, write_logical_list
from cattorch.interface import FORWARD, OK, STATUS_VARIABLE, input_name, output_name


def _normalise_inputs(example_inputs):
    return (example_inputs,) if isinstance(example_inputs, torch.Tensor) else tuple(example_inputs)


def _set_logical_list(emulator: ScratchEmulator, name: str, values: list) -> None:
    write_logical_list(emulator.lists, name, values)


def _get_logical_list(emulator: ScratchEmulator, name: str) -> list:
    return read_logical_list(emulator.lists, name)


def _compare_values(
    expected: torch.Tensor,
    raw_actual: list,
    *,
    atol: float,
    rtol: float,
) -> VerifyResult:
    expected_values = expected.detach().cpu().to(torch.float64).flatten()
    try:
        actual_values = torch.tensor([float(value) for value in raw_actual], dtype=torch.float64)
    except (TypeError, ValueError) as error:
        raise UnsupportedModelError("generated output contains a non-numeric Scratch value") from error
    expected_count = expected_values.numel()
    actual_count = actual_values.numel()
    if expected_count != actual_count:
        return VerifyResult(
            passed=False,
            values_compared=min(expected_count, actual_count),
            expected_shape=tuple(expected.shape),
            actual_values=actual_count,
            max_abs_error=math.inf,
            max_rel_error=math.inf,
            mean_abs_error=math.inf,
            worst_index=None,
        )
    close = torch.isclose(actual_values, expected_values, atol=atol, rtol=rtol, equal_nan=True)
    difference = torch.abs(actual_values - expected_values)
    same_nonfinite = (
        (torch.isnan(actual_values) & torch.isnan(expected_values))
        | (actual_values == expected_values)
    )
    difference = torch.where(same_nonfinite, 0.0, difference)
    difference = torch.nan_to_num(difference, nan=math.inf, posinf=math.inf, neginf=math.inf)
    denominator = torch.abs(expected_values)
    relative = torch.where(
        denominator == 0,
        torch.where(difference == 0, 0.0, math.inf),
        difference / denominator,
    )
    worst_index = int(torch.argmax(difference).item()) if expected_count else None
    return VerifyResult(
        passed=bool(torch.all(close).item()),
        values_compared=expected_count,
        expected_shape=tuple(expected.shape),
        actual_values=actual_count,
        max_abs_error=float(torch.max(difference).item()) if expected_count else 0.0,
        max_rel_error=float(torch.max(relative).item()) if expected_count else 0.0,
        mean_abs_error=float(torch.mean(difference).item()) if expected_count else 0.0,
        worst_index=worst_index,
    )


def round_as_stored(
    model: torch.nn.Module,
    storage: StorageConfig,
    *,
    quantization: QuantizationConfig | None = None,
    sig_figs: int | None = None,
    report=None,
) -> None:
    """Round ``model``'s floating parameters and buffers as a sprite stores them.

    ``storage``, ``quantization`` and ``sig_figs`` are the export's options;
    a quantized model has already had its weights quantized. Values are cut
    to ``sig_figs`` first, then kept or rounded to float16 as storage
    decides: integer storage keeps biases, small tensors and anything it
    would not reproduce exactly in float16. A comparison with the result
    then isolates Scratch lowering from the independently lossy storage.
    Values computed in ``forward`` rather than held by the model, such as a
    table built there, aren't rounded. ``report``, the ``QuantizationReport``
    of the model's quantization, gives the integer formats only it knows of,
    such as an output head's.

    Storage stores each matrix as each use reads it, and decides on that
    list: one use may read a weight in integers and another a float16 copy
    of it, as ``F.linear(x, w)`` and ``h @ w`` do. Each parameter is
    rounded as storage keeps it as a whole, and every matrix use run within
    ``_reading_stored_uses(model)`` reads its own list's values instead,
    rounded by the same decision storage makes for that list. Weights the
    model computes from parameters as quantization stored them are read
    the same way.
    """
    if quantization is not None:
        storage = quantized_storage(storage, quantization, report)
    effective_ndims = expert_bank_effective_ndims(model)
    # The values before rounding, of the matrices uses may read otherwise.
    originals: dict[tuple, torch.Tensor] = {}
    tensors = _stored_tensors(model)
    for key, value in tensors.items():
        if value.is_floating_point() and value.ndim >= 2:
            originals[key] = value.detach().clone()
    # What the uses quantization observed read, by tensor name.
    names = {tensor.untyped_storage().data_ptr(): key for key, tensor in tensors.items()}
    reads = tuple(
        (names[pointer], *rest) for pointer, *rest in (report._reads if report is not None else ())
        if pointer in names
    )
    uses = _StoredUses(storage, sig_figs, originals, reads)
    with torch.no_grad():
        for value in (*model.parameters(), *model.buffers()):
            if not value.is_floating_point():
                continue
            ndim = effective_ndims.get(value.untyped_storage().data_ptr())
            # Storage decides on the tensor as the model holds it; weights
            # used on the right of a matmul, and bilinear weights, are
            # stored transposed, so check that layout too.
            layout = value
            name = names.get(value.untyped_storage().data_ptr())
            precision, decided = uses.precision(value, layout, ndim, name)
            if precision == "float16" and value.ndim >= 2:
                layout = value.transpose(-1, -2).contiguous()
                precision, decided = uses.precision(value, layout, ndim, name)
                if precision == "float16":
                    layout = decided = value
            if quantization is not None and value.numel() < quantization.min_quantized_values:
                precision, decided = "float16", layout
            if sig_figs is None and precision not in {"float16"}:
                continue
            stored = torch.tensor(
                _stored_values(decided, precision, storage, sig_figs), dtype=value.dtype,
            ).reshape(layout.shape)
            value.copy_(stored if layout is value else stored.transpose(-1, -2))
    if originals and storage.precision != "float32":
        model._cattorch_stored_uses = uses
        if isinstance(model, torch.fx.GraphModule):
            # A copy of a ``GraphModule`` keeps only what its graph reads,
            # and the attributes its ``meta`` names; ``dynamic=`` checks a
            # copy of the reference at each length.
            from torch.fx.graph_module import _USER_PRESERVED_ATTRIBUTES_KEY

            model.meta.setdefault(_USER_PRESERVED_ATTRIBUTES_KEY, {})[
                "_cattorch_stored_uses"
            ] = uses


def _stored_values(layout: torch.Tensor, precision: str, storage, sig_figs) -> list:
    """The values storage holds of ``layout`` at ``precision``."""
    from cattorch.static_storage import _static_storage_group_size
    from cattorch.storage import rounded_values

    flat = layout.detach().flatten().tolist()
    if sig_figs is not None:
        flat = [float(f"%.{sig_figs}g" % item) for item in flat]
    if precision != "float32":
        # Integer storage quantizes the cut values again.
        flat = rounded_values(
            flat, cast(Any, precision), _static_storage_group_size(layout.contiguous(), storage),
            storage.scale_precision,
        )
    return flat


def _stored_tensors(model: torch.nn.Module) -> dict[tuple, torch.Tensor]:
    """The tensors a sprite stores matrices of, by a name that survives a
    copy of the model: its parameters, and the weights it computes from
    them as quantization stored them (see ``quantization._ComputedWeightHooks``)."""
    from cattorch.quantization import _ComputedWeightHooks

    found: dict[tuple, torch.Tensor] = {
        ("parameter", name): parameter for name, parameter in model.named_parameters()
    }
    for path, module in model.named_modules():
        for hook in module._forward_pre_hooks.values():
            owner = getattr(hook, "__self__", None)
            if isinstance(owner, _ComputedWeightHooks):
                for recipe, tensor in owner.table.items():
                    found[("computed", path, recipe)] = tensor
    return found


class _StoredUses:
    """What each matrix use of a model rounded by ``round_as_stored`` reads.

    ``originals`` are the matrices' values before rounding, by
    ``_stored_tensors`` name, and ``reads`` what the uses quantization
    observed read of them (``QuantizationReport._reads``, by that name).
    """

    def __init__(self, storage, sig_figs, originals: dict[tuple, torch.Tensor], reads=()):
        self.storage = storage
        self.sig_figs = sig_figs
        self.originals = originals
        self.reads: dict[tuple, list[tuple]] = {}
        for name, *read in reads:
            self.reads.setdefault(name, []).append(tuple(read))

    def precision(
        self, tensor: torch.Tensor, layout: torch.Tensor, ndim: int | None = None,
        name: tuple | None = None,
    ) -> tuple[str, torch.Tensor]:
        """The precision storage keeps the list of ``tensor``'s values
        ``layout`` holds in, and the list it decides that on.

        Storage decides each list on its own shape, and shares lists that
        hold the same values: the one it stores in the fewest bits stands in
        for the others (see ``static_storage._deduplicate_static_tensors``).
        So the list is decided as the cheapest of the lists the observed
        uses read of ``tensor`` (``name``, its ``_stored_tensors`` name)
        that hold exactly ``layout``'s values. ``min_quantized_values``
        applies to ``tensor`` as a whole, as it does in export.
        """
        from cattorch.quantization import _storage_numel
        from cattorch.static_storage import _static_storage_cost, _static_storage_precision

        storage, source = self.storage, _storage_numel(tensor)
        candidates = [layout]
        original = self.originals.get(name) if name is not None else None
        if original is not None and original.is_contiguous() and tensor.is_contiguous():
            flat = original.as_strided((source,), (1,), 0)
            values = layout.detach().reshape(-1)
            for offset, shape, stride, transposed in self.reads.get(cast(tuple, name), ()):
                view = flat.as_strided(shape, stride, offset)
                stored = (view.T if transposed else view).contiguous()
                if stored.ndim == layout.ndim and torch.equal(stored.reshape(-1), values):
                    candidates.append(stored)
        decided = candidates[0] if len(candidates) == 1 else min(
            candidates,
            key=lambda item: _static_storage_cost(item, storage, ndim, (), source),
        )
        return _static_storage_precision(decided, storage, ndim, (), source), decided


class _StoredUseMode(TorchDispatchMode):
    """Run every matrix product on the values its stored list holds."""

    def __init__(self, model: torch.nn.Module, uses: _StoredUses, registry):
        super().__init__()
        self.registry = registry
        self.uses = uses
        tensors = _stored_tensors(model)
        # Storage -> (current tensor, values before rounding, its name).
        self.by_storage = {
            tensor.untyped_storage().data_ptr(): (tensor, uses.originals[key], key)
            for key, tensor in tensors.items() if key in uses.originals
            and uses.originals[key].shape == tensor.shape
        }
        # (storage, view, orientation) -> the storage's values for that use,
        # or None when they are its values as rounded.
        self.substitutes: dict[tuple, torch.Tensor | None] = {}

    def _substitute(self, weight: torch.Tensor, orientation: str) -> torch.Tensor | None:
        from cattorch.quantization import _canonical_spec, _storage_numel

        pointer = weight.untyped_storage().data_ptr()
        found = self.by_storage.get(pointer)
        if found is None:
            return None
        key = (pointer, _canonical_spec(weight), orientation)
        if key in self.substitutes:
            return self.substitutes[key]
        tensor, original, name = found
        flat_original = original.as_strided(
            (_storage_numel(tensor),), (1,), 0,
        ) if original.is_contiguous() and tensor.is_contiguous() else None
        result = None
        if flat_original is not None:
            view = flat_original.as_strided(weight.shape, weight.stride(), weight.storage_offset())
            # The list storage keeps for this use: its matrix as the use
            # reads it, output-major.
            transposed = orientation == "rhs_matmul" and view.ndim == 2
            layout = (view.T if transposed else view).contiguous()
            precision, decided = self.uses.precision(tensor, layout, name=name)
            stored = torch.tensor(
                _stored_values(decided, precision, self.uses.storage, self.uses.sig_figs),
                dtype=tensor.dtype,
            ).reshape(layout.shape)
            if transposed:
                stored = stored.T
            current = tensor.detach().as_strided(
                weight.shape, weight.stride(), weight.storage_offset(),
            )
            if not torch.equal(stored, current):
                substitute = tensor.detach().as_strided(
                    (_storage_numel(tensor),), (1,), 0,
                ).clone()
                substitute.as_strided(
                    weight.shape, weight.stride(), weight.storage_offset(),
                ).copy_(stored)
                result = substitute
        self.substitutes[key] = result
        return result

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        from torch.utils import _pytree as pytree

        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        with torch._C._DisableTorchDispatch():
            observations = self.registry.observe(str(func), func, args, kwargs, result)
            replacements: dict[int, torch.Tensor] = {}
            for _, weight, orientation in observations:
                if isinstance(weight, torch.Tensor) and weight.is_floating_point():
                    substitute = self._substitute(weight, orientation)
                    if substitute is not None:
                        replacements[weight.untyped_storage().data_ptr()] = substitute
        if not replacements:
            return result

        def replaced(item):
            if isinstance(item, torch.Tensor):
                substitute = replacements.get(item.untyped_storage().data_ptr())
                if substitute is not None:
                    return substitute.as_strided(item.shape, item.stride(), item.storage_offset())
            return item

        return func(*pytree.tree_map(replaced, args), **pytree.tree_map(replaced, kwargs))


@contextlib.contextmanager
def _reading_stored_uses(model: torch.nn.Module, registry=None):
    """Within it, ``model``'s matrix uses read what their stored lists hold
    (see ``round_as_stored``)."""
    uses = getattr(model, "_cattorch_stored_uses", None)
    if uses is None:
        yield
        return
    with _StoredUseMode(model, uses, registry or default_registry()):
        yield


def _check_reproducible(quantization: QuantizationConfig | None) -> None:
    if quantization is not None and quantization.method != "symmetric":
        raise ValueError(
            "verify reproduces symmetric quantization only: GPTQ depends on the "
            "export's calibration data. Verify a GPTQ export against the float model "
            "instead, without quantization= and with a looser tolerance such as atol=0.05"
        )


def _lossy(quantization, storage, sig_figs) -> bool:
    return (
        quantization is not None or sig_figs is not None
        or (storage is not None and storage.precision != "float32")
    )


def _program_reference(
    model: torch.nn.Module,
    program: ExportProgram,
    registry,
    *,
    quantization: QuantizationConfig | None,
    storage: StorageConfig | None,
    sig_figs: int | None,
    logits: str | None = None,
) -> torch.nn.Module:
    """A copy of ``model`` prepared as a program export prepares its own.

    Adapters, folded batch norms, quantized weights, and the rounding of
    storage and ``sig_figs``, so it holds the values the sprite stores.
    ``logits`` names the output holding a generation export's logits, as
    ``transpile_program`` takes it.
    """
    owned = registry.adapt_model(
        model, state_names=frozenset(state.name for state in program.states),
    )
    if not _lossy(quantization, storage, sig_figs):
        return owned
    # Stored values are those of the folded model.
    owned = fold_eval_batch_norms(
        owned, methods=tuple(entry.method for entry in program.entrypoints),
    )
    if quantization is not None:
        from cattorch.program_export import _ProgramCalibrationModule, _program_output
        from cattorch.program_runtime import default_program_calls

        observed_calls = default_program_calls(program)
        report = quantize_model(
            _ProgramCalibrationModule(owned, program, observed_calls),
            quantization,
            None,
            orientation_inputs=[torch.zeros(1) for _ in observed_calls],
            _registry=registry,
            _names_from=owned,
            _logits_of=(
                None if logits is None else _program_output(program, observed_calls, logits)
            ),
        )
    else:
        report = None
    round_as_stored(
        owned, storage or StorageConfig(), quantization=quantization, sig_figs=sig_figs,
        report=report,
    )
    return owned


def _verify_program(
    model: torch.nn.Module,
    program: ExportProgram,
    sprite: Sprite,
    *,
    calls: tuple[ProgramCall, ...] | None,
    adapters: tuple[ModuleAdapter, ...],
    quantization: QuantizationConfig | None,
    storage: StorageConfig | None,
    sig_figs: int | None,
    atol: float,
    rtol: float,
) -> ProgramVerifyResult:
    project = _sprite_json(sprite, ProgramResult, "an ExportProgram's")
    registry = default_registry().for_export(adapters=adapters)
    owned = _program_reference(
        model, program, registry, quantization=quantization, storage=storage, sig_figs=sig_figs,
    )
    if calls is not None:
        passes = [(owned, calls)]
    else:
        from cattorch.program_runtime import default_program_calls
        calls = default_program_calls(program)
        passes = [(owned, calls)]
        if len(calls) > 1:
            # Then again in reverse order, from a fresh sprite and model, so
            # an entry point is called both before and after each other one:
            # one order alone can't show one's sprite depending on another
            # having run.
            passes.append((copy.deepcopy(owned), tuple(reversed(calls))))
    call_results = []
    for position, (reference, sequence) in enumerate(passes):
        call_results.extend(_program_calls(
            ScratchEmulator(copy.deepcopy(project)), reference, program, sequence, registry,
            atol=atol, rtol=rtol, tentative=position > 0,
        ))
    return ProgramVerifyResult(
        passed=all(value.passed for value in call_results),
        calls=tuple(call_results),
    )


def _program_calls(
    emulator: ScratchEmulator, owned: torch.nn.Module, program: ExportProgram,
    calls, registry, *, atol: float, rtol: float, tentative: bool = False,
) -> list[ProgramCallVerifyResult]:
    """Run ``calls`` in the emulator and on ``owned``, comparing each.

    ``tentative`` calls, which the user didn't ask for, stop where the model
    itself can't take the next one, as a read of a state still empty can't.
    """
    from cattorch.experts import _expert_family_context

    # For a marked model, the user's own methods on live buffers.
    replay = program_replay(owned, program)
    call_results = []

    def run(invocation):
        with (
            _expert_family_context(registry.expert_families),
            _reading_stored_uses(owned, registry),
        ):
            return replay.run(invocation)

    for invocation in calls:
        if not isinstance(invocation, ProgramCall):
            raise TypeError(f"calls must contain ProgramCall values, not {type(invocation).__name__}")
        entry = replay.entry(invocation)
        ran = None
        if tentative:
            try:
                ran = run(invocation)
            except Exception:  # noqa: BLE001 - the model can't take this order
                break
        for binding, value in zip(
            (binding for binding in entry.arguments if isinstance(binding, Input)),
            invocation.inputs,
        ):
            _set_logical_list(
                emulator, f"cattorch {entry.name} {binding.name}",
                value.detach().cpu().flatten().tolist(),
            )
        emulator.run_procedure(f"cattorch {entry.name}")
        status = str(emulator.variables.get(STATUS_VARIABLE, "missing runtime status"))
        output_results = []
        if status == OK:
            leaves = ran if ran is not None else run(invocation)
            for binding, expected in zip(entry.returns, leaves):
                if isinstance(binding, Output):
                    output_results.append(_compare_values(
                        expected,
                        _get_logical_list(emulator, f"cattorch {entry.name} {binding.name}"),
                        atol=atol, rtol=rtol,
                    ))
        state_results = {
            state.name: _compare_values(
                replay.states[state.name],
                _get_logical_list(emulator, f"cattorch state {state.name}"),
                atol=atol, rtol=rtol,
            )
            for state in program.states
        }
        passed = (
            status == OK
            and all(value.passed for value in output_results)
            and all(value.passed for value in state_results.values())
        )
        call_results.append(ProgramCallVerifyResult(
            entrypoint=entry.name,
            passed=passed,
            outputs=tuple(output_results),
            states=state_results,
            status=status,
        ))
        if tentative and status != OK:
            # The states in PyTorch have moved on without the sprite's.
            break
    return call_results


class _GenerationReference:
    """What a generation sprite should compute, run in PyTorch.

    ``logits(tokens)`` gives the next token's logits after reading
    ``tokens``, all the tokens since the last reset, as one row.
    """

    def __init__(self, model, program, registry, *, quantization, storage, sig_figs):
        lossy = _lossy(quantization, storage, sig_figs)
        self.whole: str | None = None
        if isinstance(program, GenerationProgram):
            from cattorch.autoregressive import MethodLeaves
            from cattorch.markers import generation_method

            method = program.method or generation_method(model) or "forward"
            if not callable(getattr(model, method, None)):
                raise ValueError(
                    f"model has no callable method {method!r}, which "
                    "GenerationProgram(method=...) names"
                )
            owned = copy.deepcopy(model).float()
            module = owned if method == "forward" else MethodLeaves(owned, method)
            prepared = registry.adapt_model(module, _copy=False)
            if lossy:
                prepared = fold_eval_batch_norms(prepared)
                report = None
                if quantization is not None:
                    report = quantize_model(
                        prepared, quantization, None,
                        orientation_inputs=[
                            (torch.zeros(1, program.max_context, dtype=torch.long),),
                        ],
                        _registry=registry,
                        # The logits are the output's first tensor, as in the export.
                        _logits_of=lambda output: tensor_output_leaves(output, warn=False)[0],
                    )
                round_as_stored(
                    prepared, storage or StorageConfig(), quantization=quantization,
                    sig_figs=sig_figs, report=report,
                )
            self.model = prepared
            self.max_context: int | None = program.max_context
            self.states = None
            return
        from cattorch.huggingface_recurrent import adapt_recurrent_model
        from cattorch.recurrent_generation import step_program

        states, step = program.states, program.step
        owned = model
        # A Hugging Face model's step is cattorch's own code, so its logits
        # come from the model's forward over every token read instead.
        if states is None:
            adapted = adapt_recurrent_model(model)
            if adapted is None:
                raise UnsupportedModelError(
                    "RecurrentGenerationProgram needs states=... unless the model is a "
                    "supported Hugging Face recurrent model (Mamba, Mamba-2, RWKV)"
                )
            owned, states, step = adapted.module, adapted.states, adapted.step
            self.whole = adapted.reference
        else:
            owned = copy.deepcopy(model).float()
        states = tuple(
            State(state.name, state.initial.float()) if state.initial.is_floating_point()
            else state for state in states
        )
        export = step_program(states, step)
        self.model = _program_reference(
            owned, export, registry, quantization=quantization, storage=storage,
            sig_figs=sig_figs, logits="logits",
        )
        self.step = step
        self.initial = tuple(state.initial for state in states)
        self.max_context = program.max_context
        self.states = self.initial
        self.read = 0

    def logits(self, tokens: list[int]) -> torch.Tensor:
        with torch.no_grad(), _reading_stored_uses(self.model):
            if self.states is None or self.whole is not None:
                value = torch.tensor([tokens], dtype=torch.long)
                run = self.model if self.whole is None else getattr(self.model, self.whole)
                output = tensor_output_leaves(run(value), warn=False)[0]
                return output.reshape(len(tokens), -1)[-1]
            if len(tokens) < self.read:
                self.states, self.read = self.initial, 0
            logits = None
            for token in tokens[self.read:]:
                returned = getattr(self.model, self.step)(
                    torch.tensor([[token]], dtype=torch.long), *self.states,
                )
                logits, *updates = tensor_output_leaves(returned, warn=False)
                self.states = tuple(updates)
            self.read = len(tokens)
            assert logits is not None
            return logits.reshape(-1)


def _generation_calls(vocabulary: int, max_context: int | None) -> tuple[ProgramCall, ...]:
    """A prefill and decodes over a fixed token sequence, as default ``calls``."""
    length = min(max_context or 8, 8)
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, vocabulary, (1, length), generator=generator)
    prompt = max(1, length // 2)
    return (
        ProgramCall("prefill", (tokens[:, :prompt],)),
        *(
            ProgramCall("decode", (tokens[:, position:position + 1],))
            for position in range(prompt, length)
        ),
    )


def _verify_generation(
    model: torch.nn.Module,
    program: GenerationProgram | RecurrentGenerationProgram,
    sprite: Sprite,
    *,
    calls: tuple[ProgramCall, ...] | None,
    adapters: tuple[ModuleAdapter, ...],
    quantization: QuantizationConfig | None,
    storage: StorageConfig | None,
    sig_figs: int | None,
    atol: float,
    rtol: float,
) -> ProgramVerifyResult:
    """Run generation block calls in the emulator and compare each with PyTorch."""
    from cattorch import interface

    project = _sprite_json(sprite, GenerationResult, "a generation")
    path = getattr(sprite, "path", sprite)
    emulator = ScratchEmulator(project)
    names = {entry[0] for entry in project.get("lists", {}).values()}
    if interface.LOGITS not in names:
        raise ValueError(f"{path} is not a generation sprite: it has no {interface.LOGITS!r} list")
    top_k = interface.TOP_K_VALUES in names
    _check_generation_program(project, sprite, program, top_k)
    registry = default_registry().for_export(adapters=adapters)
    reference = _GenerationReference(
        model, program, registry, quantization=quantization, storage=storage, sig_figs=sig_figs,
    )
    # The token IDs the sprite accepts.
    from cattorch.autoregressive import EmbeddingLookup, lookup_vocabulary

    vocabulary = sprite.vocabulary_size if isinstance(sprite, GenerationResult) else None
    if vocabulary is None:
        token = torch.zeros(1, 1, dtype=torch.long)
        lookup = EmbeddingLookup(token)
        run = (
            reference.model if reference.states is None
            else lambda value: getattr(reference.model, reference.step)(value, *reference.initial)
        )
        with torch.no_grad(), lookup:
            run(token)
        vocabulary = lookup_vocabulary(run, lookup) or int(reference.logits([0]).numel())
    if calls is None:
        calls = _generation_calls(vocabulary, reference.max_context)
    tokens: list[int] = []
    results = []
    for invocation in calls:
        if not isinstance(invocation, ProgramCall):
            raise TypeError(f"calls must contain ProgramCall values, not {type(invocation).__name__}")
        block = invocation.entrypoint
        if block not in {"prefill", "decode", "reset"}:
            raise ValueError(
                f"a generation sprite's calls name 'prefill', 'decode' or 'reset', not {block!r}"
            )
        if block == "reset":
            if invocation.inputs:
                raise ValueError("ProgramCall('reset') takes no inputs")
            values: list = []
        else:
            if len(invocation.inputs) != 1:
                raise ValueError(f"ProgramCall({block!r}, ...) takes one tensor of token IDs")
            values = invocation.inputs[0].detach().cpu().flatten().tolist()
            _set_logical_list(emulator, interface.TOKEN_IDS, values)
        emulator.run_procedure(f"cattorch {block}")
        status = str(emulator.variables.get(STATUS_VARIABLE, "missing runtime status"))
        expected_status = _expected_generation_status(
            block, values, len(tokens), vocabulary, reference.max_context,
        )
        if expected_status == OK:
            tokens = (
                [] if block == "reset"
                else [int(value) for value in values] if block == "prefill"
                else [*tokens, int(values[0])]
            )
        outputs: list[VerifyResult] = []
        if block != "reset" and status == OK == expected_status:
            expected = reference.logits(tokens)
            outputs.append(_compare_values(
                expected, _get_logical_list(emulator, interface.LOGITS), atol=atol, rtol=rtol,
            ))
            if top_k:
                values_found = _get_logical_list(emulator, interface.TOP_K_VALUES)
                ids_found = _get_logical_list(emulator, interface.TOP_K_IDS)
                largest = torch.topk(expected, len(values_found)).values
                outputs.append(_compare_values(largest, values_found, atol=atol, rtol=rtol))
                # Ties may list either ID, so the IDs are checked by the
                # logits they point at.
                try:
                    pointed = expected[[int(float(value)) for value in ids_found]]
                except (ValueError, IndexError):
                    pointed = torch.full((len(ids_found),), math.nan)
                outputs.append(_compare_values(largest, pointed.tolist(), atol=atol, rtol=rtol))
        results.append(ProgramCallVerifyResult(
            entrypoint=block,
            passed=status == expected_status and all(value.passed for value in outputs),
            outputs=tuple(outputs),
            states={},
            status=status,
            expected_status=expected_status,
        ))
    return ProgramVerifyResult(
        passed=all(value.passed for value in results), calls=tuple(results),
    )


def _check_generation_program(project: dict, sprite, program, top_k: bool) -> None:
    """The program must be the one the sprite was exported with.

    The expected statuses follow the program's ``max_context`` and the
    compared lists its ``top_k``, so another program would report false
    failures or skip checks.
    """
    from cattorch import interface

    stored = next(
        (entry[1] for entry in project.get("variables", {}).values()
         if entry[0] == interface.MAX_CONTEXT),
        None,
    )
    if stored is not None:
        number = ScratchEmulator._scratch_number(stored)
        sprite_context = None if number == math.inf else number
        if sprite_context != program.max_context:
            shown = "no limit" if sprite_context is None else sprite_context
            raise ValueError(
                f"the program's max_context={program.max_context} doesn't match the sprite's "
                f"({shown}); verify with the program the sprite was exported with"
            )
    sprite_top_k = sprite.top_k if isinstance(sprite, GenerationResult) else None
    if (program.top_k is not None) != top_k or (
        sprite_top_k is not None and sprite_top_k != program.top_k
    ):
        shown = sprite_top_k if sprite_top_k is not None else ("some" if top_k else None)
        raise ValueError(
            f"the program's top_k={program.top_k} doesn't match the sprite's ({shown}); "
            "verify with the program the sprite was exported with"
        )


def _expected_generation_status(
    block: str, values: list, cached: int, vocabulary: int, max_context: int | None,
) -> str:
    """The ``cattorch status`` a generation block should end with."""
    from cattorch import interface

    if block == "reset":
        return OK
    tokens = interface.TOKEN_IDS
    if block == "decode" and len(values) != 1:
        return interface.wrong_number_of_items(tokens)
    if block == "prefill":
        if not values:
            return interface.too_few_items(tokens)
        if max_context is not None and len(values) > max_context:
            return interface.too_many_items(tokens)
    for value in values:
        number = float(value)
        if not number.is_integer() or not 0 <= number < vocabulary:
            return interface.invalid_item(tokens)
    if block == "decode" and max_context is not None and cached >= max_context:
        return interface.CONTEXT_FULL
    return OK


def _tensor_reference(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    example_kwargs: dict[str, Any] | None,
    *,
    adapters: tuple[ModuleAdapter, ...],
    quantization: QuantizationConfig | None,
    storage: StorageConfig | None,
    sig_figs: int | None,
) -> torch.nn.Module:
    """The model a tensor-input sprite is compared with.

    An evaluation-mode copy with the Hugging Face defaults ``transpile``
    uses, when needed; with export options, a copy prepared as the export
    prepares its own: adapters, folded batch norms, quantized weights, and
    the rounding of storage and ``sig_figs``.
    """
    if adapters or _lossy(quantization, storage, sig_figs):
        registry = default_registry().for_export(adapters=adapters)
        owned = registry.adapt_model(model)
        if not _lossy(quantization, storage, sig_figs):
            return owned
        owned = fold_eval_batch_norms(owned)
        report = None
        if quantization is not None:
            report = quantize_model(
                owned, quantization, None, orientation_inputs=[inputs],
                call_kwargs=example_kwargs, _registry=registry,
            )
        round_as_stored(
            owned, storage or StorageConfig(), quantization=quantization, sig_figs=sig_figs,
            report=report,
        )
        return owned
    transformers = sys.modules.get("transformers")
    huggingface = transformers is not None and any(
        isinstance(module, transformers.PreTrainedModel) for module in model.modules()
    )
    training = any(module.training for module in model.modules())
    if training or huggingface:
        # Compare against the model transpile exported: an evaluation-mode
        # copy, with the same Hugging Face defaults (no cache or extra outputs).
        model = copy.deepcopy(model)
        if training:
            eval_mode_copy_warning(model)
            model.eval()
        if huggingface:
            prepare_huggingface_models(model)
    return model


def _verify_tensors(
    model: torch.nn.Module,
    example_inputs,
    sprite: Sprite,
    *,
    adapters: tuple[ModuleAdapter, ...],
    quantization: QuantizationConfig | None,
    storage: StorageConfig | None,
    sig_figs: int | None,
    example_kwargs: dict[str, Any] | None,
    atol: float,
    rtol: float,
) -> VerifyResult | MultiOutputVerifyResult:
    project = _sprite_json(sprite, TranspileResult, "a tensor-input export's")
    inputs = _normalise_inputs(example_inputs)
    if not inputs or any(not isinstance(value, torch.Tensor) for value in inputs):
        raise TypeError("example_inputs must be a tensor or tuple of tensors")

    emulator = ScratchEmulator(project)
    for index, value in enumerate(flat_example_inputs(inputs, example_kwargs)):
        name = input_name(index)
        _set_logical_list(emulator, name, value.detach().cpu().flatten().tolist())
    emulator.run_procedure(FORWARD)

    reference = _tensor_reference(
        model, inputs, example_kwargs, adapters=adapters, quantization=quantization,
        storage=storage, sig_figs=sig_figs,
    )
    try:
        with torch.no_grad(), _reading_stored_uses(reference):
            expected = reference(*inputs, **(example_kwargs or {}))
    except Exception as raised:  # noqa: BLE001 - the model failed; reported as such
        raise UnsupportedModelError(
            model_failure("the model", raised, flat_example_inputs(inputs, example_kwargs)),
        ) from raised
    leaves = tensor_output_leaves(expected, warn=False)
    output_names = tuple(output_name(index) for index in range(len(leaves)))
    actual_names = {
        entry[0] for entry in project.get("lists", {}).values()
        if re.fullmatch(r"output(?:_[1-9][0-9]*)?", entry[0])
    }
    if actual_names != set(output_names):
        raise UnsupportedModelError(
            "the sprite's output lists don't match the model's tensor outputs: "
            f"expected {output_names}, found {tuple(sorted(actual_names))}"
        )
    comparisons = tuple(
        _compare_values(
            value, _get_logical_list(emulator, name), atol=atol, rtol=rtol,
        )
        for value, name in zip(leaves, output_names)
    )
    approximate = sprite.approximate_operations if isinstance(sprite, TranspileResult) else ()
    if len(comparisons) == 1:
        return dataclasses.replace(comparisons[0], approximate_operations=approximate)
    return MultiOutputVerifyResult(
        passed=all(result.passed for result in comparisons),
        outputs=comparisons,
        approximate_operations=approximate,
    )


Sprite = str | Path | TranspileResult | ProgramResult | GenerationResult


def _sprite_json(sprite, result_type, kind: str) -> dict:
    """The sprite ``verify`` was given: an export's result, or its path.

    A path is read as ``transpile`` names its file, adding ``.sprite3``
    unless it ends in it, so the ``output_path`` given to ``transpile``
    works too. A file at the path exactly as given is read when that one
    doesn't exist, so a renamed sprite, or a project passed by mistake, is
    read and explained.
    """
    if isinstance(sprite, result_type):
        return read_json(sprite.path)
    if isinstance(sprite, (str, os.PathLike)) and not isinstance(sprite, bytes):
        path = output_file(sprite, ".sprite3", parameter="sprite")
        if not path.exists() and Path(sprite).is_file():
            path = Path(sprite)
        return read_user_archive(path, SPRITE_JSON, parameter="sprite")
    raise TypeError(
        f"{kind} sprite is a {result_type.__name__} or a path, not a {type(sprite).__name__}"
    )


def verify(
    model: torch.nn.Module,
    example_inputs: (
        torch.Tensor | tuple[torch.Tensor, ...] | ExportProgram | GenerationProgram
        | RecurrentGenerationProgram | None
    ),
    sprite: Sprite,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-5,
    calls: Sequence[ProgramCall] | None = None,
    adapters: Sequence[ModuleAdapter] = (),
    quantization: QuantizationConfig | None = None,
    storage: StorageConfig | None = None,
    sig_figs: int | None = None,
    example_kwargs: dict[str, Any] | None = None,
) -> VerifyResult | MultiOutputVerifyResult | ProgramVerifyResult:
    """Run an exported sprite in the emulator and compare it with PyTorch.

    Use it to check the conversion and the effect of lossy export settings.
    It says nothing about speed in browser Scratch. ``model`` and
    ``example_inputs`` are what was passed to ``transpile``, and ``sprite``
    is the export's result or its path; ``None`` takes the interface from
    the model's markers, as ``transpile`` does.

    For tensor inputs, the sprite runs ``cattorch forward`` on
    ``example_inputs`` (and the tensors in ``example_kwargs``, which must
    match the export's), and the result is a ``VerifyResult``, or a
    ``MultiOutputVerifyResult`` with one comparison per output. A length
    mismatch is a failed result with infinite error metrics.

    For an ``ExportProgram`` (or ``@entrypoint`` markers), ``calls`` runs a
    sequence of ``ProgramCall``s with state carried between them, comparing
    outputs and states after every call. By default each entry point runs
    once with its example inputs, then, with several entry points, again in
    reverse order from a fresh sprite and model, so each runs both before
    and after every other one. With markers, PyTorch runs the model's own methods on a
    copy whose buffers hold the state. For a ``GenerationProgram``
    or ``RecurrentGenerationProgram``, ``calls`` names the blocks
    ``"prefill"``, ``"decode"`` and ``"reset"`` with their token IDs, such as
    ``ProgramCall("prefill", (torch.tensor([[1, 2, 3]]),))``; by default a
    prefill and a few decodes. Each call's logits and top-k lists are
    compared with the model's next-token logits, and its ``cattorch status``
    with the status its input should give, such as ``context full``. Both
    return a ``ProgramVerifyResult``.

    ``adapters``, ``quantization``, ``storage`` and ``sig_figs`` are the
    export's options: given, PyTorch runs a copy of the model prepared as the
    export prepared its own, holding the values the sprite stores, so the
    comparison isolates the conversion from lossy storage and can use tight
    tolerances. Without them a lossy export needs a looser tolerance. Only
    symmetric quantization can be reproduced. A model in training mode is
    run as an evaluation-mode copy, with a warning. Non-tensor output leaves
    are ignored, as in ``transpile``.
    """
    for label, value in (("atol", atol), ("rtol", rtol)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{label} must be a number, not {type(value).__name__}")
        if not value >= 0:
            raise ValueError(f"{label} must be a non-negative number, not {value!r}")
    validate_model_and_inputs(model, example_inputs)
    validate_export_options(
        storage=storage, quantization=quantization, sig_figs=sig_figs, adapters=adapters,
        example_kwargs=example_kwargs,
    )
    adapters = tuple(adapters)
    _check_reproducible(quantization)
    if example_inputs is None:
        from cattorch.transpiler import interface_from_markers

        model, example_inputs = interface_from_markers(model)
    from cattorch.transpiler import check_head_bits

    check_head_bits(example_inputs, quantization)
    options: dict[str, Any] = dict(
        adapters=adapters, quantization=quantization, storage=storage, sig_figs=sig_figs,
        atol=atol, rtol=rtol,
    )
    stateful = (ExportProgram, GenerationProgram, RecurrentGenerationProgram)
    if isinstance(example_inputs, stateful):
        if example_kwargs is not None:
            raise ValueError(
                f"example_kwargs can't be used with a {type(example_inputs).__name__}, "
                "whose calls take their inputs from calls="
            )
        calls = None if calls is None else tuple(calls)
        if isinstance(example_inputs, ExportProgram):
            return _verify_program(model, example_inputs, sprite, calls=calls, **options)
        return _verify_generation(model, example_inputs, sprite, calls=calls, **options)
    if calls is not None:
        raise ValueError(
            "calls= applies to programs and generation programs; for tensor inputs, "
            "verify runs example_inputs once"
        )
    return _verify_tensors(
        model, example_inputs, sprite, example_kwargs=example_kwargs, **options,
    )
