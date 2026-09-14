"""Lower experimental ``ExportProgram`` interfaces into one Scratch sprite."""

from __future__ import annotations

import math
import uuid
import zipfile
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from cattorch.adapters import ModuleAdapter
from cattorch.codegen import CodegenConfig
from cattorch.errors import UnsupportedModelError
from cattorch.export_options import validate_export_options
from cattorch.fast import FastConfig, prepare_fast_model
from cattorch.graph_transforms import fold_eval_batch_norms
from cattorch.operator_registry import OperatorRegistry, default_registry
from cattorch.program import (
    EntryPoint, ExportProgram, Input, Output, ProgramCall, StateInput, StateUpdate,
)
from cattorch.program_runtime import (
    MethodModule, ProgramReplay, default_program_calls, entry_examples,
    validate_entry_outputs,
)
from cattorch.results import (
    EntryPointResult, ProgramResult, StateResult, TensorSpec,
)
from cattorch.quantization import QuantizationConfig, quantize_model
from cattorch.storage import StorageConfig
from cattorch.sprite import (
    _add_warp_procedure, _merge_lists_by_name, _merge_variables_by_name,
)
from cattorch.util.scratch.block_combiner import _merge_slots
from cattorch.util.scratch.dsl import (
    Program, add, append, broadcast_and_wait, call, clear, eq, for_each, gt, if_,
    item, length, lt, mod, set_var, var,
)
from cattorch.util.scratch.finalize_scratch import finalize_sprite
from cattorch.util.scratch.interface import (
    rename_list as _rename_list, rename_variable as _rename_variable,
    rename_procedure as _rename_procedure, refresh_data_display_name,
)
from cattorch.util.scratch.remap import remap_ids
from cattorch.util.scratch.sharding import shard_sprite_lists


class _ProgramCalibrationModule(nn.Module):
    """Present a sequence of named program calls to the shared GPTQ observer."""

    def __init__(self, model: nn.Module, program: ExportProgram, calls: tuple[ProgramCall, ...]):
        super().__init__()
        self.model = model
        self.calls = calls
        self.position = 0
        self.replay = ProgramReplay(model, program)

    def forward(self, _sentinel: torch.Tensor):
        invocation = self.calls[self.position]
        self.position += 1
        return self.replay.run(invocation)


def _merge_sprite_targets(left: dict, right: dict) -> dict:
    """Merge independent lifecycle/procedure roots without chaining them."""
    result = left
    right = dict(right)
    block_collisions = set(result.get("blocks", {})) & set(right.get("blocks", {}))
    if block_collisions:
        suffix = uuid.uuid4().hex[:8]
        mapping = {identifier: f"{identifier}_{suffix}" for identifier in right["blocks"]}
        right = remap_ids(right, mapping)

    variables, variable_remap = _merge_slots(
        result.get("variables", {}), right.get("variables", {}),
    )
    lists, list_remap = _merge_slots(
        result.get("lists", {}), right.get("lists", {}),
    )
    blocks = remap_ids(right.get("blocks", {}), {**variable_remap, **list_remap})

    broadcasts = result.setdefault("broadcasts", {})
    by_message = {message: identifier for identifier, message in broadcasts.items()}
    broadcast_remap = {}
    for identifier, message in right.get("broadcasts", {}).items():
        winner = by_message.get(message)
        if winner is None:
            broadcasts[identifier] = message
            by_message[message] = identifier
        else:
            broadcast_remap[identifier] = winner
    if broadcast_remap:
        blocks = remap_ids(blocks, broadcast_remap)

    overlap = set(result.get("blocks", {})) & set(blocks)
    if overlap:
        raise ValueError(f"block IDs still collide after program merge: {sorted(overlap)[:3]}")
    result["variables"] = variables
    result["lists"] = lists
    result.setdefault("blocks", {}).update(blocks)
    return result


def _namespace_entry_storage(sprite: dict, namespace: str) -> dict[str, str]:
    """Keep captured weights and decoder streams owned by their entrypoint.

    Independent captures can assign the same graph name to different weights.
    Only byte-identical encoded payloads may be deduplicated after merging.
    """
    names = [
        entry[0]
        for entry in sprite.get("lists", {}).values()
        if entry[0].startswith(("W_", "cattorch payload ", "cattorch scale payload "))
    ]
    for old in names:
        _rename_list(sprite, old, f"_cattorch {namespace} {old}")
    variable_names = [
        entry[0]
        for entry in sprite.get("variables", {}).values()
        if entry[0].startswith(("cattorch payload ", "cattorch scale payload "))
        or (
            entry[0].startswith("cattorch ") and " bank" in entry[0]
            and entry[0].endswith((" payload", " table"))
        )
    ]
    for old in variable_names:
        _rename_variable(sprite, old, f"_cattorch {namespace} {old}")
    names.extend(variable_names)
    return {old: f"_cattorch {namespace} {old}" for old in names}


def _namespace_entry_procedures(sprite: dict, namespace: str) -> None:
    """Keep private helpers from different entrypoint compilations distinct."""
    protected = {
        f"_cattorch {namespace} compute",
        f"_cattorch {namespace} init",
        f"_cattorch {namespace} prepare for save",
    }
    names = {
        mutation["proccode"]
        for block in sprite.get("blocks", {}).values()
        if isinstance((mutation := block.get("mutation")), dict)
        and isinstance(mutation.get("proccode"), str)
        and mutation["proccode"] not in protected
    }
    for old in names:
        _rename_procedure(sprite, old, f"_cattorch {namespace} {old}")


def _deduplicate_program_payloads(sprite: dict, payload_names: set[str]) -> None:
    """Share byte-identical encoded banks across program entrypoints."""
    for category in ("lists", "variables"):
        slots = sprite.get(category, {})
        winners = {}
        remap: dict[str, str] = {}
        for identifier, entry in slots.items():
            if entry[0] not in payload_names:
                continue
            fingerprint = tuple(entry[1]) if isinstance(entry[1], list) else entry[1]
            winner = winners.get(fingerprint)
            if winner is None:
                winners[fingerprint] = identifier
            else:
                remap[identifier] = winner
        if not remap:
            continue
        sprite["blocks"] = remap_ids(sprite.get("blocks", {}), remap)
        for winner in set(remap.values()):
            refresh_data_display_name(sprite["blocks"], winner, slots[winner][0])
        for identifier in remap:
            slots.pop(identifier, None)


def _state_list(name: str) -> str:
    return f"cattorch state {name}"


def _input_list(entrypoint: str, name: str) -> str:
    return f"cattorch {entrypoint} {name}"


def _temporary_return(entrypoint: str, index: int) -> str:
    return f"_cattorch {entrypoint} return {index}"


def _copy_list(source: str, destination: str, index_name: str):
    return (
        clear(destination),
        for_each(index_name, length(source), (append(destination, item(source, var(index_name))),)),
    )


def _build_entry_wrapper(
    entrypoint: EntryPoint,
    result,
    program: ExportProgram,
) -> tuple[Program, tuple[TensorSpec, ...]]:
    states = {state.name: state for state in program.states}
    status = "cattorch status"
    internal = f"_cattorch {entrypoint.name} compute"
    completion = f"cattorch {entrypoint.name} complete"
    public_outputs = []
    lists = set()
    variables = {status}
    body = [set_var(status, "ok")]

    for index, (binding, spec) in enumerate(zip(entrypoint.arguments, result.inputs)):
        name = (
            _input_list(entrypoint.name, binding.name)
            if isinstance(binding, Input) else _state_list(binding.name)
        )
        lists.add(name)
        if spec.numel is None:
            trailing = math.prod(
                extent for extent in spec.shape[1:] if isinstance(extent, int)
            )
            body.extend((
                if_(gt(mod(length(name), trailing), 0), (
                    set_var(status, f"invalid dynamic length: {name}"),
                )),
                if_(gt(length(name), spec.max_numel), (
                    set_var(status, f"dynamic capacity exceeded: {name}"),
                )),
            ))
        else:
            expected = spec.numel
            body.extend((
                if_(gt(length(name), expected), (set_var(status, f"invalid length: {name}"),)),
                if_(lt(length(name), expected), (set_var(status, f"invalid length: {name}"),)),
            ))

    temporary_specs = result.outputs
    validate_entry_outputs(program, entrypoint, temporary_specs)
    for index, (binding, spec) in enumerate(zip(entrypoint.returns, temporary_specs)):
        temporary = _temporary_return(entrypoint.name, index)
        lists.add(temporary)
        if isinstance(binding, Output):
            destination = _input_list(entrypoint.name, binding.name)
            lists.add(destination)
            public_outputs.append(TensorSpec(
                list_name=destination,
                shape=spec.shape,
                dtype=spec.dtype,
                numel=spec.numel,
                max_numel=spec.max_numel,
            ))

    computation = [call(internal)]
    for index, (binding, spec) in enumerate(zip(entrypoint.returns, temporary_specs)):
        temporary = _temporary_return(entrypoint.name, index)
        if isinstance(binding, Output):
            if spec.numel is None:
                trailing = math.prod(
                    extent for extent in spec.shape[1:] if isinstance(extent, int)
                )
                computation.extend((
                    if_(gt(mod(length(temporary), trailing), 0), (
                        set_var(status, f"invalid dynamic return: {binding.name}"),
                    )),
                    if_(gt(length(temporary), spec.max_numel), (
                        set_var(status, f"dynamic return capacity exceeded: {binding.name}"),
                    )),
                ))
            else:
                expected = spec.numel
                computation.extend((
                    if_(gt(length(temporary), expected), (
                        set_var(status, f"invalid return length: {binding.name}"),
                    )),
                    if_(lt(length(temporary), expected), (
                        set_var(status, f"invalid return length: {binding.name}"),
                    )),
                ))
            continue
        state = states[binding.name]
        destination = _state_list(state.name)
        lists.add(destination)
        if state.mode == "replace":
            expected = state.initial.numel()
            computation.extend((
                if_(gt(length(temporary), expected), (
                    set_var(status, f"invalid state update: {state.name}"),
                )),
                if_(lt(length(temporary), expected), (
                    set_var(status, f"invalid state update: {state.name}"),
                )),
            ))
        else:
            item_size = math.prod(state.initial.shape[1:])
            maximum = state.capacity * item_size
            computation.extend((
                if_(gt(mod(length(temporary), item_size), 0), (
                    set_var(status, f"invalid append update: {state.name}"),
                )),
                if_(gt(add(length(destination), length(temporary)), maximum), (
                    set_var(status, f"state capacity exceeded: {state.name}"),
                )),
            ))

    commits = []
    for index, binding in enumerate(entrypoint.returns):
        temporary = _temporary_return(entrypoint.name, index)
        if isinstance(binding, Output):
            destination = _input_list(entrypoint.name, binding.name)
        else:
            state = states[binding.name]
            destination = _state_list(state.name)
        if isinstance(binding, StateUpdate) and states[binding.name].mode == "append":
            commits.append(for_each(
                f"{entrypoint.name} append {index}", length(temporary),
                (append(destination, item(temporary, var(f"{entrypoint.name} append {index}"))),),
            ))
        else:
            commits.extend(_copy_list(
                temporary, destination, f"{entrypoint.name} copy {index}",
            ))
    commits.append(broadcast_and_wait(completion))
    computation.append(if_(eq(var(status), "ok"), tuple(commits)))
    body.append(if_(eq(var(status), "ok"), tuple(computation)))
    return Program(
        f"entrypoint_{entrypoint.name}",
        variables=variables | {
            f"{entrypoint.name} copy {index}"
            for index, binding in enumerate(entrypoint.returns)
            if not (
                isinstance(binding, StateUpdate)
                and states[binding.name].mode == "append"
            )
        } | {
            f"{entrypoint.name} append {index}"
            for index, binding in enumerate(entrypoint.returns)
            if isinstance(binding, StateUpdate) and states[binding.name].mode == "append"
        },
        lists=lists,
        body=tuple(body),
    ), tuple(public_outputs)


def _set_list_contents(sprite: dict, name: str, values: list) -> None:
    for entry in sprite.setdefault("lists", {}).values():
        if entry[0] == name:
            entry[1] = values
            return
    identifier = f"cattorch_program_list_{uuid.uuid4().hex}"
    sprite["lists"][identifier] = [name, values]


def transpile_program(
    model: nn.Module,
    program: ExportProgram,
    output_path: str | Path,
    sig_figs: int | None = None,
    *,
    name: str | None = None,
    optimization="exact",
    fast_config=None,
    storage=None,
    quantization=None,
    calibration_calls: tuple[ProgramCall, ...] | None = None,
    codegen: CodegenConfig | None = None,
    adapters: tuple[ModuleAdapter, ...] = (),
) -> ProgramResult:
    """Compile all explicit program entrypoints into one sprite artifact."""
    if not isinstance(program, ExportProgram):
        raise TypeError("program must be an ExportProgram")
    validate_export_options(
        name=name, optimization=optimization, fast_config=fast_config,
        storage=storage, quantization=quantization, codegen=codegen,
        sig_figs=sig_figs, adapters=adapters,
    )
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".sprite3":
        output_path = Path(f"{output_path}.sprite3")
    sprite_name = output_path.stem if name is None else name
    codegen = codegen or CodegenConfig()
    registry: OperatorRegistry = default_registry().clone()
    for adapter in adapters:
        registry.register_adapter(adapter)

    states = {state.name: state for state in program.states}
    calls = tuple(calibration_calls or ())
    known_entries = {entry.name for entry in program.entrypoints}
    if any(call_value.entrypoint not in known_entries for call_value in calls):
        raise ValueError("calibration_calls contains an unknown entrypoint")
    if calibration_calls is not None and (quantization is None or quantization.method != "gptq"):
        raise ValueError("calibration_calls can only be used with GPTQ quantization")

    # Model ownership, transforms, and calibration belong to the whole
    # program. Every entrypoint below captures the same prepared module.
    prepared_model = registry.adapt_model(
        model, state_names=frozenset(states),
    )
    config = fast_config or FastConfig()
    if optimization == "fast":
        prepared_model = prepare_fast_model(prepared_model, config)
    prepared_model = fold_eval_batch_norms(
        prepared_model, methods=tuple(entry.method for entry in program.entrypoints),
    )
    quantization_report = None
    resolved_storage = storage or StorageConfig()
    if quantization is not None:
        if quantization.method == "gptq" and prepared_model.training:
            raise ValueError("GPTQ calibration requires model.eval()")
        gptq = quantization.method == "gptq"
        observed_calls = calls if gptq else default_program_calls(program)
        calibration_model = _ProgramCalibrationModule(
            prepared_model, program, observed_calls,
        )
        sentinels = [torch.zeros(1) for _ in observed_calls]
        quantization_report = quantize_model(
            calibration_model,
            quantization,
            sentinels if gptq else None,
            orientation_inputs=None if gptq else sentinels,
            _registry=registry,
        )
        prepared_model = calibration_model.model
        resolved_storage = replace(
            resolved_storage,
            precision=quantization.precision,
            group_size=quantization.group_size,
            min_quantized_values=quantization.min_quantized_values,
            scale_precision=quantization.scale_precision,
            grouping="row",
        )

    combined = None
    entry_results = []
    sub_warnings: list[str] = []
    sharded_names: set[str] = set()
    interface_capacities: dict[str, int] = {}
    payload_names: set[str] = set()
    with TemporaryDirectory(prefix="cattorch-program-") as directory:
        from cattorch.transpiler import _transpile

        for position, entrypoint in enumerate(program.entrypoints):
            if not hasattr(model, entrypoint.method) or not callable(getattr(model, entrypoint.method)):
                raise ValueError(
                    f"model has no callable method {entrypoint.method!r} for "
                    f"entrypoint {entrypoint.name!r}"
                )
            args, dynamic_inputs = entry_examples(program, entrypoint)
            wrapper = MethodModule(prepared_model, entrypoint.method)
            temporary_path = Path(directory) / f"entry_{position}.sprite3"
            sub_codegen = replace(
                codegen,
                compact_internal_names=False,
                compact_schema=False,
                id_namespace=None,
            )
            sub_result = _transpile(
                wrapper,
                tuple(args),
                temporary_path,
                sig_figs,
                name=sprite_name,
                optimization=optimization,
                fast_config=fast_config,
                storage=resolved_storage,
                quantization=quantization,
                calibration_inputs=None,
                codegen=sub_codegen,
                frontend="fx",
                _registry=registry,
                _prepared_model=True,
                _prepared_quantization_report=quantization_report,
                _dynamic_inputs=dynamic_inputs,
                _defer_interface_sharding=True,
            )
            if len(sub_result.inputs) != len(entrypoint.arguments):
                raise UnsupportedModelError(
                    f"entrypoint {entrypoint.name!r} captured an unexpected argument count"
                )
            validate_entry_outputs(program, entrypoint, sub_result.outputs)
            with zipfile.ZipFile(temporary_path) as archive:
                sprite = json.loads(archive.read("sprite.json"))

            storage_names = _namespace_entry_storage(sprite, entrypoint.name)
            payload_names.update(new for old, new in storage_names.items() if not old.startswith("W_"))

            internal = f"_cattorch {entrypoint.name} compute"
            _rename_procedure(sprite, "cattorch forward", internal)
            _rename_procedure(sprite, "cattorch init", f"_cattorch {entrypoint.name} init")
            _rename_procedure(
                sprite, "cattorch prepare for save",
                f"_cattorch {entrypoint.name} prepare for save",
            )
            _rename_variable(
                sprite, "cattorch initialized",
                f"_cattorch {entrypoint.name} initialized",
            )
            _namespace_entry_procedures(sprite, entrypoint.name)
            for index, binding in enumerate(entrypoint.arguments):
                old = "input" if index == 0 else f"input_{index}"
                new = (
                    _input_list(entrypoint.name, binding.name)
                    if isinstance(binding, Input) else _state_list(binding.name)
                )
                _rename_list(sprite, old, new)
                spec = sub_result.inputs[index]
                interface_capacities[new] = max(
                    interface_capacities.get(new, 0), spec.max_numel or 0,
                )
            for index in range(len(entrypoint.returns)):
                old = "output" if index == 0 else f"output_{index}"
                _rename_list(sprite, old, _temporary_return(entrypoint.name, index))
                interface_capacities[_temporary_return(entrypoint.name, index)] = (
                    sub_result.outputs[index].max_numel or 0
                )

            combined = sprite if combined is None else _merge_sprite_targets(combined, sprite)
            wrapper_program, output_specs = _build_entry_wrapper(
                entrypoint, sub_result, program,
            )
            for spec in output_specs:
                interface_capacities[spec.list_name] = max(
                    interface_capacities.get(spec.list_name, 0), spec.max_numel or 0,
                )
            combined = _add_warp_procedure(
                combined,
                f"cattorch {entrypoint.name}",
                wrapper_program,
                x=640,
                y=160 + 160 * position,
            )
            _merge_lists_by_name(
                combined,
                set(wrapper_program.lists),
            )
            _merge_variables_by_name(combined, set(wrapper_program.variables))
            public_inputs = tuple(
                TensorSpec(
                    list_name=_input_list(entrypoint.name, binding.name),
                    shape=spec.shape,
                    dtype=spec.dtype,
                    numel=spec.numel,
                )
                for binding, spec in zip(entrypoint.arguments, sub_result.inputs)
                if isinstance(binding, Input)
            )
            entry_results.append(EntryPointResult(
                name=entrypoint.name,
                procedure=f"cattorch {entrypoint.name}",
                completion_broadcast=f"cattorch {entrypoint.name} complete",
                inputs=public_inputs,
                outputs=output_specs,
                state_inputs=tuple(
                    binding.name for binding in entrypoint.arguments
                    if isinstance(binding, StateInput)
                ),
                state_updates=tuple(
                    binding.name for binding in entrypoint.returns
                    if isinstance(binding, StateUpdate)
                ),
            ))
            sub_warnings.extend(
                warning for warning in sub_result.warnings
                if warning.startswith("GPTQ fallback")
            )
            sharded_names.update(storage_names.get(name, name) for name in sub_result.sharded_lists)
            quantization_report = sub_result.quantization

    assert combined is not None
    _deduplicate_program_payloads(combined, payload_names)
    for state in program.states:
        values = state.initial.detach().cpu().flatten().tolist()
        _set_list_contents(combined, _state_list(state.name), values)
        _set_list_contents(combined, f"_cattorch initial state {state.name}", values)

    lifecycle_lists = {
        name
        for state in program.states
        for name in (_state_list(state.name), f"_cattorch initial state {state.name}")
    }
    lifecycle_variables = {
        f"cattorch reset {index}" for index, _state in enumerate(program.states)
    }
    reset_body = [set_var("cattorch status", "ok")]
    lifecycle_variables.add("cattorch status")
    for index, state in enumerate(program.states):
        reset_body.extend(_copy_list(
            f"_cattorch initial state {state.name}",
            _state_list(state.name),
            f"cattorch reset {index}",
        ))
    combined = _add_warp_procedure(
        combined,
        "cattorch reset",
        Program(
            "program_reset", variables=lifecycle_variables,
            lists=lifecycle_lists, body=reset_body,
        ),
        x=960,
        y=160,
    )
    init_body = [
        *(call(f"_cattorch {entry.name} init") for entry in program.entrypoints),
        broadcast_and_wait("cattorch init complete"),
    ]
    combined = _add_warp_procedure(
        combined,
        "cattorch init",
        Program("program_init", body=init_body),
        x=960,
        y=320,
    )
    combined = _add_warp_procedure(
        combined,
        "cattorch prepare for save",
        Program(
            "program_prepare",
            body=tuple(
                call(f"_cattorch {entry.name} prepare for save")
                for entry in program.entrypoints
            ),
        ),
        x=960,
        y=480,
    )
    _merge_lists_by_name(combined, lifecycle_lists)
    _merge_variables_by_name(combined, lifecycle_variables | {"cattorch status"})

    program_capacities = dict(interface_capacities)
    program_capacities.update({
        _state_list(state.name): (
            state.capacity * math.prod(state.initial.shape[1:])
            if state.mode == "append" else state.initial.numel()
        )
        for state in program.states
    })
    program_capacities.update({
        f"_cattorch initial state {state.name}": state.initial.numel()
        for state in program.states
    })
    program_layouts = shard_sprite_lists(combined, program_capacities)
    sharded_names.update(
        layout.name for layout in program_layouts.values()
        if len(layout.shards) > 1
    )

    public_data_names = {"cattorch status"}
    for entrypoint in entry_results:
        public_data_names.update(value.list_name for value in entrypoint.inputs)
        public_data_names.update(value.list_name for value in entrypoint.outputs)
    public_data_names.update(_state_list(state.name) for state in program.states)
    public_procedures = {
        "cattorch init", "cattorch reset", "cattorch prepare for save",
        *(f"cattorch {entry.name}" for entry in program.entrypoints),
    }
    finalized = finalize_sprite(
        combined,
        output_path,
        sprite_name=sprite_name,
        codegen=codegen,
        public_data_names=frozenset(public_data_names),
        public_procedures=frozenset(public_procedures),
    )
    state_results = tuple(
        StateResult(
            name=state.name,
            list_name=_state_list(state.name),
            mode=state.mode,
            shape=tuple(state.initial.shape),
            dtype=str(state.initial.dtype).removeprefix("torch."),
            capacity=state.capacity,
        )
        for state in program.states
    )
    procedures = (
        "cattorch init", "cattorch reset", "cattorch prepare for save",
        *(f"cattorch {entry.name}" for entry in program.entrypoints),
    )
    return ProgramResult(
        path=finalized.path,
        sprite_name=sprite_name,
        archive_bytes=finalized.archive_bytes,
        expanded_json_bytes=finalized.expanded_json_bytes,
        block_count=len(combined.get("blocks", {})),
        list_count=len(combined.get("lists", {})),
        sharded_lists=tuple(sorted(sharded_names)),
        warnings=tuple(dict.fromkeys((*sub_warnings, *finalized.warnings))),
        entrypoints=tuple(entry_results),
        states=state_results,
        procedures=procedures,
        quantization=quantization_report,
    )


__all__ = ["transpile_program"]
