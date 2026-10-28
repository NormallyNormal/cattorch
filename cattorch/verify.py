"""Numerically verify generated Scratch sprites against PyTorch."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

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
from cattorch.program import ExportProgram, Input, Output, ProgramCall
from cattorch.program_runtime import ProgramReplay
from cattorch.graph_transforms import fold_eval_batch_norms
from cattorch.results import (
    ProgramCallVerifyResult, ProgramResult, ProgramVerifyResult, TranspileResult,
    VerifyResult, MultiOutputVerifyResult,
)
from cattorch.util.scratch.emulator import ScratchEmulator
from cattorch.util.scratch.interface import read_logical_list, write_logical_list


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


def _verify_program(
    model: torch.nn.Module,
    program: ExportProgram,
    sprite: str | Path | ProgramResult,
    *,
    calls: tuple[ProgramCall, ...] | None,
    adapters: tuple[ModuleAdapter, ...],
    quantization: QuantizationConfig | None,
    atol: float,
    rtol: float,
) -> ProgramVerifyResult:
    path = sprite.path if isinstance(sprite, ProgramResult) else Path(sprite)
    with zipfile.ZipFile(path) as archive:
        project = json.loads(archive.read("sprite.json"))
    emulator = ScratchEmulator(project)
    registry = default_registry().clone()
    for adapter in adapters:
        registry.register_adapter(adapter)
    owned = registry.adapt_model(
        model, state_names=frozenset(state.name for state in program.states),
    )
    if quantization is not None:
        if quantization.method != "symmetric":
            raise ValueError(
                "program verification currently reconstructs symmetric quantization; "
                "GPTQ requires the exact export calibration sequence"
            )
        owned = fold_eval_batch_norms(
            owned, methods=tuple(entry.method for entry in program.entrypoints),
        )
        from cattorch.program_export import _ProgramCalibrationModule
        from cattorch.program_runtime import default_program_calls
        observed_calls = default_program_calls(program)
        quantize_model(
            _ProgramCalibrationModule(owned, program, observed_calls),
            quantization,
            None,
            orientation_inputs=[torch.zeros(1) for _ in observed_calls],
            _registry=registry,
        )
        # Integer storage keeps biases, small tensors, and anything it would
        # not reproduce exactly in float16. Mirror that one-time decoder result
        # so this comparison isolates Scratch lowering from the independently
        # lossy storage decision.
        from cattorch.sprite import _static_storage_precision
        storage = quantized_storage(StorageConfig(), quantization)
        effective_ndims = expert_bank_effective_ndims(owned)
        with torch.no_grad():
            for value in (*owned.parameters(), *owned.buffers()):
                if not value.is_floating_point():
                    continue
                precision = _static_storage_precision(
                    value, storage,
                    effective_ndims.get(value.untyped_storage().data_ptr()),
                )
                if precision == "float16" and value.ndim == 2:
                    # Weights used on the right of a matmul are stored
                    # transposed, so check that layout too.
                    precision = _static_storage_precision(
                        value.T.contiguous(), storage,
                        effective_ndims.get(value.untyped_storage().data_ptr()),
                    )
                if value.numel() < quantization.min_quantized_values:
                    precision = "float16"
                if precision == "float16":
                    value.copy_(value.to(torch.float16).to(value.dtype))
    replay = ProgramReplay(owned, program)
    if calls is None:
        from cattorch.program_runtime import default_program_calls
        calls = default_program_calls(program)
    call_results = []
    for invocation in calls:
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
        status = str(emulator.variables.get("cattorch status", "missing runtime status"))
        output_results = []
        if status == "ok":
            from cattorch.moe import _expert_family_context
            with _expert_family_context(registry.expert_families):
                leaves = replay.run(invocation)
            for binding, expected in zip(entry.returns, leaves):
                if isinstance(binding, Output):
                    output_results.append(_compare_values(
                        expected,
                        _get_logical_list(emulator, f"cattorch {entry.name} {binding.name}"),
                        atol=atol, rtol=rtol,
                    ))
        state_results = tuple(
            (
                state.name,
                _compare_values(
                    replay.states[state.name],
                    _get_logical_list(emulator, f"cattorch state {state.name}"),
                    atol=atol, rtol=rtol,
                ),
            )
            for state in program.states
        )
        passed = (
            status == "ok"
            and all(value.passed for value in output_results)
            and all(value.passed for _, value in state_results)
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


def verify(
    model: torch.nn.Module,
    example_inputs: torch.Tensor | tuple[torch.Tensor, ...] | ExportProgram | None,
    sprite: str | Path | TranspileResult | ProgramResult,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-5,
    calls: tuple[ProgramCall, ...] | None = None,
    adapters: tuple[ModuleAdapter, ...] = (),
    quantization: QuantizationConfig | None = None,
    example_kwargs: dict[str, Any] | None = None,
):
    """Run one generated forward in the emulator and compare it with PyTorch.

    Use it to check the conversion and the effect of lossy export settings.
    It says nothing about speed in browser Scratch.

    ``sprite`` may be a path or ``TranspileResult``. ``example_inputs`` must
    match the exported fixed interface. The returned ``VerifyResult`` reports
    output length, aggregate errors, and the worst flattened index for one
    tensor output; a length mismatch is a failed result with infinite error
    metrics. Multiple tensor leaves return ``MultiOutputVerifyResult``, with
    one comparison per leaf in export order and an overall ``passed`` flag.

    For an ``ExportProgram``, ``calls`` runs a sequence of ``ProgramCall``s
    with state carried between them, ``adapters`` must match the export, and a
    symmetric ``quantization`` config makes PyTorch use the quantized weights.
    The result is a ``ProgramVerifyResult``.

    ``example_kwargs`` must match the export: tensor values are written to the
    inputs after the positional ones, and all of them are passed to the model.
    A model in training mode is run as an evaluation-mode copy, with a warning.
    Non-tensor output leaves are ignored, as in ``transpile``.
    """
    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative")
    if example_inputs is None:
        # As in ``transpile``: the interface comes from the model's markers.
        from cattorch.markers import example_inputs as marked_example, has_entrypoints, marked_program
        if has_entrypoints(model):
            model, example_inputs = marked_program(model)
        else:
            example_inputs = marked_example(model)
            if example_inputs is None:
                raise ValueError("example_inputs is None, but the model has no marker for them")
    validate_export_options(example_kwargs=example_kwargs)
    if example_kwargs is not None and isinstance(example_inputs, ExportProgram):
        raise ValueError("example_kwargs cannot be used with an ExportProgram")
    if isinstance(example_inputs, ExportProgram):
        if isinstance(sprite, TranspileResult):
            raise TypeError("an ExportProgram must be verified with a ProgramResult or path")
        return _verify_program(
            model, example_inputs, sprite,
            calls=calls, adapters=adapters, quantization=quantization,
            atol=atol, rtol=rtol,
        )
    if calls is not None or adapters or quantization is not None:
        raise ValueError("calls, adapters, and quantization require an ExportProgram")
    if isinstance(sprite, ProgramResult):
        raise TypeError("ordinary tensor verification cannot use a ProgramResult")
    inputs = _normalise_inputs(example_inputs)
    if not inputs or any(not isinstance(value, torch.Tensor) for value in inputs):
        raise TypeError("example_inputs must be a tensor or tuple of tensors")

    path = sprite.path if isinstance(sprite, TranspileResult) else Path(sprite)
    with zipfile.ZipFile(path) as archive:
        project = json.loads(archive.read("sprite.json"))
    emulator = ScratchEmulator(project)
    for index, value in enumerate(flat_example_inputs(inputs, example_kwargs)):
        name = "input" if index == 0 else f"input_{index}"
        _set_logical_list(emulator, name, value.detach().cpu().flatten().tolist())
    emulator.run_procedure("cattorch forward")

    training = any(module.training for module in model.modules())
    transformers = sys.modules.get("transformers")
    huggingface = transformers is not None and any(
        isinstance(module, transformers.PreTrainedModel) for module in model.modules()
    )
    if training or huggingface:
        # Compare against the model transpile exported: an evaluation-mode
        # copy, with the same Hugging Face defaults (no cache or extra outputs).
        model = copy.deepcopy(model)
        if training:
            model.eval()
            eval_mode_copy_warning()
        if huggingface:
            prepare_huggingface_models(model)
    with torch.no_grad():
        expected = model(*inputs, **(example_kwargs or {}))
    leaves = tensor_output_leaves(expected, warn=False)
    output_names = tuple(
        "output" if index == 0 else f"output_{index}"
        for index in range(len(leaves))
    )
    actual_names = {
        entry[0] for entry in project.get("lists", {}).values()
        if re.fullmatch(r"output(?:_[1-9][0-9]*)?", entry[0])
    }
    if actual_names != set(output_names):
        raise UnsupportedModelError(
            "generated output lists do not match the model's tensor output leaves: "
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
