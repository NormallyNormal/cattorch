"""Numerically verify generated Scratch sprites against PyTorch."""

from __future__ import annotations

import copy
import dataclasses
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence, overload

import torch

from cattorch.errors import UnsupportedModelError
from cattorch.adapters import ModuleAdapter
from cattorch.export_options import validate_export_options
from cattorch.frontend import flat_example_inputs, tensor_output_leaves
from cattorch.huggingface import prepare_huggingface_models
from cattorch.operator_registry import default_registry, eval_mode_copy_warning
from cattorch.quantization import (
    QuantizationConfig, expert_bank_effective_ndims, quantize_model,
)
from cattorch.storage import StorageConfig, quantized_storage
from cattorch.program import (
    ExportProgram, GenerationProgram, Input, Output, ProgramCall, RecurrentGenerationProgram, State,
)
from cattorch.program_runtime import ProgramReplay
from cattorch.graph_transforms import fold_eval_batch_norms
from cattorch.results import (
    GenerationResult, ProgramCallVerifyResult, ProgramResult, ProgramVerifyResult,
    TranspileResult, VerifyResult, MultiOutputVerifyResult,
)
from cattorch.util.scratch.archive import read_json
from cattorch.util.scratch.emulator import ScratchEmulator
from cattorch.util.scratch.interface import read_logical_list, write_logical_list
from cattorch.interface import OK, STATUS_VARIABLE, input_name, output_name


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
) -> None:
    """Round ``model``'s floating parameters and buffers as a sprite stores them.

    ``storage``, ``quantization`` and ``sig_figs`` are the export's options;
    a quantized model has already had its weights quantized. Values are cut
    to ``sig_figs`` first, then kept or rounded to float16 as storage
    decides: integer storage keeps biases, small tensors and anything it
    would not reproduce exactly in float16. A comparison with the result
    then isolates Scratch lowering from the independently lossy storage.
    Values computed in ``forward`` rather than held by the model, such as a
    table built there, aren't rounded.
    """
    from cattorch.sprite import _static_storage_group_size, _static_storage_precision
    from cattorch.storage import rounded_values

    if quantization is not None:
        storage = quantized_storage(storage, quantization)
    effective_ndims = expert_bank_effective_ndims(model)
    with torch.no_grad():
        for value in (*model.parameters(), *model.buffers()):
            if not value.is_floating_point():
                continue
            ndim = effective_ndims.get(value.untyped_storage().data_ptr())
            # Storage decides on the tensor as the model holds it; weights
            # used on the right of a matmul, and bilinear weights, are
            # stored transposed, so check that layout too.
            layout = value
            precision = _static_storage_precision(layout, storage, ndim)
            if precision == "float16" and value.ndim >= 2:
                layout = value.transpose(-1, -2).contiguous()
                precision = _static_storage_precision(layout, storage, ndim)
                if precision == "float16":
                    layout = value
            if quantization is not None and value.numel() < quantization.min_quantized_values:
                precision = "float16"
            if sig_figs is None and precision not in {"float16"}:
                continue
            flat = layout.detach().flatten().tolist()
            if sig_figs is not None:
                flat = [float(f"%.{sig_figs}g" % item) for item in flat]
            if precision != "float32":
                # Integer storage quantizes the cut values again.
                flat = rounded_values(
                    flat, precision, _static_storage_group_size(layout, storage),
                    storage.scale_precision,
                )
            stored = torch.tensor(flat, dtype=value.dtype).reshape(layout.shape)
            value.copy_(stored if layout is value else stored.transpose(-1, -2))


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
) -> torch.nn.Module:
    """A copy of ``model`` prepared as a program export prepares its own.

    Adapters, folded batch norms, quantized weights, and the rounding of
    storage and ``sig_figs``, so it holds the values the sprite stores.
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
        from cattorch.program_export import _ProgramCalibrationModule
        from cattorch.program_runtime import default_program_calls

        observed_calls = default_program_calls(program)
        quantize_model(
            _ProgramCalibrationModule(owned, program, observed_calls),
            quantization,
            None,
            orientation_inputs=[torch.zeros(1) for _ in observed_calls],
            _registry=registry,
            _names_from=owned,
        )
    round_as_stored(owned, storage or StorageConfig(), quantization=quantization, sig_figs=sig_figs)
    return owned


def _verify_program(
    model: torch.nn.Module,
    program: ExportProgram,
    sprite: str | Path | ProgramResult,
    *,
    calls: tuple[ProgramCall, ...] | None,
    adapters: tuple[ModuleAdapter, ...],
    quantization: QuantizationConfig | None,
    storage: StorageConfig | None,
    sig_figs: int | None,
    atol: float,
    rtol: float,
) -> ProgramVerifyResult:
    if isinstance(sprite, (TranspileResult, GenerationResult)):
        raise TypeError(
            f"an ExportProgram's sprite is a ProgramResult or a path, not a "
            f"{type(sprite).__name__}"
        )
    path = sprite.path if isinstance(sprite, ProgramResult) else Path(sprite)
    emulator = ScratchEmulator(read_json(path))
    registry = default_registry().for_export(adapters=adapters)
    owned = _program_reference(
        model, program, registry, quantization=quantization, storage=storage, sig_figs=sig_figs,
    )
    replay = ProgramReplay(owned, program)
    if calls is None:
        from cattorch.program_runtime import default_program_calls
        calls = default_program_calls(program)
    call_results = []
    for invocation in calls:
        if not isinstance(invocation, ProgramCall):
            raise TypeError(f"calls must contain ProgramCall values, not {type(invocation).__name__}")
        entry = replay.entry(invocation)
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
            from cattorch.experts import _expert_family_context
            with _expert_family_context(registry.expert_families):
                leaves = replay.run(invocation)
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
    return ProgramVerifyResult(
        passed=all(value.passed for value in call_results),
        calls=tuple(call_results),
    )


class _GenerationReference:
    """What a generation sprite should compute, run in PyTorch.

    ``logits(tokens)`` gives the next token's logits after reading
    ``tokens``, all the tokens since the last reset, as one row.
    """

    def __init__(self, model, program, registry, *, quantization, storage, sig_figs):
        lossy = _lossy(quantization, storage, sig_figs)
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
                if quantization is not None:
                    quantize_model(
                        prepared, quantization, None,
                        orientation_inputs=[
                            (torch.zeros(1, program.max_context, dtype=torch.long),),
                        ],
                        _registry=registry,
                    )
                round_as_stored(
                    prepared, storage or StorageConfig(), quantization=quantization,
                    sig_figs=sig_figs,
                )
            self.model = prepared
            self.max_context: int | None = program.max_context
            self.states = None
            return
        from cattorch.huggingface_recurrent import adapt_recurrent_model
        from cattorch.recurrent_generation import step_program

        states, step = program.states, program.step
        owned = model
        if states is None:
            adapted = adapt_recurrent_model(model)
            if adapted is None:
                raise UnsupportedModelError(
                    "RecurrentGenerationProgram needs states=... unless the model is a "
                    "supported Hugging Face recurrent model (Mamba, Mamba-2, RWKV)"
                )
            owned, states, step = adapted.module, adapted.states, adapted.step
        else:
            owned = copy.deepcopy(model).float()
        states = tuple(
            State(state.name, state.initial.float()) if state.initial.is_floating_point()
            else state for state in states
        )
        export = step_program(states, step)
        self.model = _program_reference(
            owned, export, registry, quantization=quantization, storage=storage,
            sig_figs=sig_figs,
        )
        self.step = step
        self.initial = tuple(state.initial for state in states)
        self.max_context = program.max_context
        self.states = self.initial
        self.read = 0

    def logits(self, tokens: list[int]) -> torch.Tensor:
        with torch.no_grad():
            if self.states is None:
                value = torch.tensor([tokens], dtype=torch.long)
                output = tensor_output_leaves(self.model(value), warn=False)[0]
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
    sprite: str | Path | GenerationResult,
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

    if isinstance(sprite, (TranspileResult, ProgramResult)):
        raise TypeError(
            f"a generation sprite is a GenerationResult or a path, not a "
            f"{type(sprite).__name__}"
        )
    path = sprite.path if isinstance(sprite, GenerationResult) else Path(sprite)
    project = read_json(path)
    emulator = ScratchEmulator(project)
    names = {entry[0] for entry in project.get("lists", {}).values()}
    if interface.LOGITS not in names:
        raise ValueError(f"{path} is not a generation sprite: it has no {interface.LOGITS!r} list")
    top_k = interface.TOP_K_VALUES in names
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
        if quantization is not None:
            quantize_model(
                owned, quantization, None, orientation_inputs=[inputs],
                call_kwargs=example_kwargs, _registry=registry,
            )
        round_as_stored(
            owned, storage or StorageConfig(), quantization=quantization, sig_figs=sig_figs,
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
    sprite: str | Path | TranspileResult,
    *,
    adapters: tuple[ModuleAdapter, ...],
    quantization: QuantizationConfig | None,
    storage: StorageConfig | None,
    sig_figs: int | None,
    example_kwargs: dict[str, Any] | None,
    atol: float,
    rtol: float,
) -> VerifyResult | MultiOutputVerifyResult:
    if isinstance(sprite, (ProgramResult, GenerationResult)):
        raise TypeError(
            f"a sprite from tensor inputs is a TranspileResult or a path, not a "
            f"{type(sprite).__name__}"
        )
    inputs = _normalise_inputs(example_inputs)
    if not inputs or any(not isinstance(value, torch.Tensor) for value in inputs):
        raise TypeError("example_inputs must be a tensor or tuple of tensors")

    path = sprite.path if isinstance(sprite, TranspileResult) else Path(sprite)
    project = read_json(path)
    emulator = ScratchEmulator(project)
    for index, value in enumerate(flat_example_inputs(inputs, example_kwargs)):
        name = input_name(index)
        _set_logical_list(emulator, name, value.detach().cpu().flatten().tolist())
    emulator.run_procedure("cattorch forward")

    reference = _tensor_reference(
        model, inputs, example_kwargs, adapters=adapters, quantization=quantization,
        storage=storage, sig_figs=sig_figs,
    )
    with torch.no_grad():
        expected = reference(*inputs, **(example_kwargs or {}))
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


@overload
def verify(
    model: torch.nn.Module,
    example_inputs: ExportProgram | GenerationProgram | RecurrentGenerationProgram,
    sprite: Sprite,
    *,
    atol: float = ...,
    rtol: float = ...,
    calls: Sequence[ProgramCall] | None = ...,
    adapters: Sequence[ModuleAdapter] = ...,
    quantization: QuantizationConfig | None = ...,
    storage: StorageConfig | None = ...,
    sig_figs: int | None = ...,
    example_kwargs: dict[str, Any] | None = ...,
) -> ProgramVerifyResult: ...


@overload
def verify(
    model: torch.nn.Module,
    example_inputs: torch.Tensor | tuple[torch.Tensor, ...] | None,
    sprite: Sprite,
    *,
    atol: float = ...,
    rtol: float = ...,
    calls: Sequence[ProgramCall] | None = ...,
    adapters: Sequence[ModuleAdapter] = ...,
    quantization: QuantizationConfig | None = ...,
    storage: StorageConfig | None = ...,
    sig_figs: int | None = ...,
    example_kwargs: dict[str, Any] | None = ...,
) -> VerifyResult | MultiOutputVerifyResult | ProgramVerifyResult: ...


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
    sequence of ``ProgramCall``s with state carried between them, each entry
    point once with its example inputs by default. For a ``GenerationProgram``
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
    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative")
    validate_export_options(
        storage=storage, quantization=quantization, sig_figs=sig_figs, adapters=adapters,
        example_kwargs=example_kwargs,
    )
    adapters = tuple(adapters)
    _check_reproducible(quantization)
    if example_inputs is None:
        from cattorch.transpiler import interface_from_markers

        model, example_inputs = interface_from_markers(model)
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
