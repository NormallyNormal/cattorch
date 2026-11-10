"""Lower experimental ``ExportProgram`` interfaces into one Scratch sprite."""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn

from cattorch.adapters import ModuleAdapter
from cattorch.codegen import CodegenConfig
from cattorch.errors import UnsupportedModelError
from cattorch.export_options import validate_export_options
from cattorch.fast import FastConfig, prepare_fast_model
from cattorch.graph_transforms import fold_eval_batch_norms
from cattorch.operator_registry import default_registry
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
from cattorch.quantization import quantize_model
from cattorch.interface import STATUS_VARIABLE, final_data_name, input_name, output_name
from cattorch.storage import StorageConfig, quantized_storage
from cattorch.sprite import (
    _add_warp_procedure, _merge_lists_by_name, _merge_variables_by_name,
)
from cattorch.util.scratch.block_combiner import _merge_slots
from cattorch.util.scratch.dsl import (
    Program, Statement, add, append, broadcast_and_wait, call, clear, eq, for_each, gt, if_,
    item, length, lt, mod, repeat, set_var, var,
)
from cattorch.util.scratch.finalize_scratch import FinalizedSprite, finalize_sprite
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


_ENSURE_STATES = "_cattorch ensure states"
_STATES_FILLED = "_cattorch states filled"


def _fill_procedure(index: int) -> str:
    return f"_cattorch fill state {index}"


def _constant_fill(state) -> int | float | None:
    """The one value a numeric replace state starts with everywhere, if any.

    Only replace states qualify: their list is never validly empty, so an
    empty one in a freshly loaded project is known to need filling. An empty
    append state is a valid value.
    """
    flat = state.initial.detach().cpu().flatten()
    if (
        state.mode != "replace" or flat.numel() == 0
        or flat.dtype == torch.bool or flat.is_complex()
    ):
        return None
    first = flat[0]
    same = flat == first
    if flat.is_floating_point():
        same &= torch.signbit(flat) == torch.signbit(first)
    return first.item() if bool(same.all()) else None


@dataclass(frozen=True)
class ProgramAssembly:
    """What a ``transpile_program(_edit_program=...)`` callback can see and extend.

    ``names`` maps every default interface name (``cattorch <entry>``,
    ``cattorch <entry> <binding>``, ``cattorch <entry> complete``,
    ``cattorch state <name>``, and the lifecycle procedures) to its name in
    the sprite. ``states`` maps each state to its list. The callback may add
    to ``capacities`` (logical list sizes, for sharding the lists it adds) and
    to the public name sets, which keep names from being marked internal or
    compacted; it may also remove names from them.
    """

    names: Mapping[str, str]
    states: Mapping[str, str]
    capacities: dict[str, int]
    public_data_names: set[str]
    public_procedures: set[str]


def _interface_renames(
    program: ExportProgram, public_names: Mapping[str, str] | None, private_states: bool,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Validated list, procedure, and broadcast renames for the program interface."""
    lists: dict[str, str] = {}
    procedures: dict[str, str] = {}
    broadcasts: dict[str, str] = {}
    defaults: dict[str, dict[str, str]] = {}
    for entrypoint in program.entrypoints:
        defaults[f"cattorch {entrypoint.name}"] = procedures
        defaults[f"cattorch {entrypoint.name} complete"] = broadcasts
        for binding in (*entrypoint.arguments, *entrypoint.returns):
            if isinstance(binding, (Input, Output)):
                defaults[_input_list(entrypoint.name, binding.name)] = lists
    for old, new in (public_names or {}).items():
        if old not in defaults:
            raise ValueError(f"_public_names has no entry point name {old!r}")
        if not isinstance(new, str) or not new.strip():
            raise ValueError(f"_public_names[{old!r}] must be a non-empty string")
        if new != old:
            defaults[old][old] = new
    if private_states:
        for state in program.states:
            lists[_state_list(state.name)] = f"_cattorch state {state.name}"
    targets = [*lists.values(), *procedures.values(), *broadcasts.values()]
    taken = set(defaults) | {_state_list(state.name) for state in program.states}
    if len(targets) != len(set(targets)) or set(targets) & taken:
        raise ValueError("_public_names must map to distinct, unused names")
    return lists, procedures, broadcasts


def _rename_broadcast(sprite: dict, old: str, new: str) -> None:
    for identifier, message in sprite.get("broadcasts", {}).items():
        if message == old:
            sprite["broadcasts"][identifier] = new
            refresh_data_display_name(sprite.get("blocks", {}), identifier, new)


def _build_entry_wrapper(
    entrypoint: EntryPoint,
    result,
    program: ExportProgram,
    ensure_states: bool = False,
) -> tuple[Program, tuple[TensorSpec, ...]]:
    states = {state.name: state for state in program.states}
    status = STATUS_VARIABLE
    internal = f"_cattorch {entrypoint.name} compute"
    completion = f"cattorch {entrypoint.name} complete"
    public_outputs = []
    lists = set()
    variables = {status}
    body: list[Statement] = [set_var(status, "ok")]
    if ensure_states:
        body.append(call(_ENSURE_STATES))

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
    kernels=(),
    _edit_program: Callable[[dict, ProgramAssembly], None] | None = None,
    _public_names: Mapping[str, str] | None = None,
    _private_states: bool = False,
) -> ProgramResult:
    """Compile all explicit program entrypoints into one sprite artifact.

    A replace state whose initial tensor holds one value everywhere is not stored:
    its list is saved empty and filled by a loop on ``cattorch reset``, on
    ``cattorch init`` and on the first entry point call if it is still empty.

    Private hooks for exports built on a program (default: no change):

    - ``_edit_program(sprite, assembly)`` runs on the assembled sprite after
      the entry points and lifecycle procedures exist, with every interface
      name final, and before lists are sharded and the sprite is written. It
      may add procedures, lists and variables that use them; ``assembly``
      (a ``ProgramAssembly``) gives the names, and takes the capacities and
      public names of what the callback adds.
    - ``_public_names`` renames entry point procedures (``cattorch <entry>``),
      their input and output lists (``cattorch <entry> <binding>``) and
      completion broadcasts (``cattorch <entry> complete``), keyed by the
      default name. A new name starting with ``_`` is private: it is left out
      of the public interface and ``ProgramResult.procedures``, and may be
      compacted, so the entry point results then name it only as built.
    - ``_private_states`` names state lists ``_cattorch state <name>`` and
      leaves them out of the public interface and of the result's states.

    The renaming hooks can't be combined with variable-length inputs, whose
    export check runs the entry points through their default names.
    """
    if not isinstance(program, ExportProgram):
        raise TypeError("program must be an ExportProgram")
    list_renames, procedure_renames, broadcast_renames = _interface_renames(
        program, _public_names, _private_states,
    )
    renames = {**list_renames, **procedure_renames, **broadcast_renames}
    if renames and any(
        isinstance(binding, Input) and binding.dynamic is not None
        for entry in program.entrypoints for binding in entry.arguments
    ):
        raise ValueError("_public_names and _private_states need fixed-length inputs")

    def name_of(default: str) -> str:
        return renames.get(default, default)

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
    registry = default_registry().for_export(adapters=adapters, kernels=kernels)

    states = {state.name: state for state in program.states}
    fills = {state.name: _constant_fill(state) for state in program.states}
    any_filled = any(value is not None for value in fills.values())
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
        resolved_storage = quantized_storage(resolved_storage, quantization)

    combined = None
    entry_results = []
    sub_warnings: list[str] = []
    sharded_names: set[str] = set()
    interface_capacities: dict[str, int] = {}
    payload_names: set[str] = set()
    # Each entry point's sprite stays in memory; only the program is
    # finalized and written.
    entry_sprites: list[dict] = []

    def keep_entry(sprite, output_path, **_options) -> FinalizedSprite:
        entry_sprites.append(sprite)
        return FinalizedSprite(
            path=Path(output_path), archive_bytes=0, expanded_json_bytes=0, asset_id="",
            warnings=(),
        )

    from cattorch.transpiler import _transpile

    for position, entrypoint in enumerate(program.entrypoints):
        if not hasattr(model, entrypoint.method) or not callable(getattr(model, entrypoint.method)):
            raise ValueError(
                f"model has no callable method {entrypoint.method!r} for "
                f"entrypoint {entrypoint.name!r}"
            )
        args, dynamic_inputs = entry_examples(program, entrypoint)
        wrapper = MethodModule(prepared_model, entrypoint.method)
        sub_codegen = replace(
            codegen,
            compact_internal_names=False,
            compact_schema=False,
            id_namespace=None,
        )
        def capture(arguments, dynamic, edit=None):
            return _transpile(
                wrapper,
                tuple(arguments),
                output_path,
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
                _dynamic_inputs=dynamic,
                _defer_interface_sharding=True,
                _edit_sprite=edit,
                _finalize=keep_entry,
            )

        try:
            sub_result = capture(args, dynamic_inputs)
        except UnsupportedModelError as error:
            sub_result = _entry_by_item(
                error, entrypoint, wrapper, args, dynamic_inputs, capture,
            )
            sub_warnings.append(
                f"entry point {entrypoint.name!r} runs once per item of its "
                "variable-length input, because its operations can't take a "
                "variable length directly"
            )
        if len(sub_result.inputs) != len(entrypoint.arguments):
            raise UnsupportedModelError(
                f"entrypoint {entrypoint.name!r} captured an unexpected argument count"
            )
        validate_entry_outputs(program, entrypoint, sub_result.outputs)
        sprite = entry_sprites.pop()

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
            old = input_name(index)
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
            old = output_name(index)
            _rename_list(sprite, old, _temporary_return(entrypoint.name, index))
            interface_capacities[_temporary_return(entrypoint.name, index)] = (
                sub_result.outputs[index].max_numel or 0
            )

        combined = sprite if combined is None else _merge_sprite_targets(combined, sprite)
        wrapper_program, output_specs = _build_entry_wrapper(
            entrypoint, sub_result, program, ensure_states=any_filled,
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
        if fills[state.name] is not None:
            # A constant state is filled by a loop at run time instead.
            _set_list_contents(combined, _state_list(state.name), [])
            continue
        values = state.initial.detach().cpu().flatten().tolist()
        _set_list_contents(combined, _state_list(state.name), values)
        _set_list_contents(combined, f"_cattorch initial state {state.name}", values)

    lifecycle_lists = {_state_list(state.name) for state in program.states} | {
        f"_cattorch initial state {state.name}"
        for state in program.states if fills[state.name] is None
    }
    lifecycle_variables = {
        f"cattorch reset {index}" for index, state in enumerate(program.states)
        if fills[state.name] is None
    }
    reset_body: list[Statement] = [set_var(STATUS_VARIABLE, "ok")]
    lifecycle_variables.add(STATUS_VARIABLE)
    for index, state in enumerate(program.states):
        if fills[state.name] is not None:
            reset_body.append(call(_fill_procedure(index)))
            continue
        reset_body.extend(_copy_list(
            f"_cattorch initial state {state.name}",
            _state_list(state.name),
            f"cattorch reset {index}",
        ))
    if any_filled:
        reset_body.append(set_var(_STATES_FILLED, 1))
        lifecycle_variables.add(_STATES_FILLED)
        # The first call or init fills the constant states still empty in a
        # freshly loaded project. Later, as before, only reset refills them.
        combined = _add_warp_procedure(
            combined,
            _ENSURE_STATES,
            Program(
                "program_ensure_states", variables={_STATES_FILLED},
                lists={_state_list(state.name) for state in program.states},
                body=(if_(eq(var(_STATES_FILLED), 0), (
                    *(
                        if_(eq(length(_state_list(state.name)), 0), (call(_fill_procedure(index)),))
                        for index, state in enumerate(program.states)
                        if fills[state.name] is not None
                    ),
                    set_var(_STATES_FILLED, 1),
                )),),
            ),
            x=960,
            y=640,
        )
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
        *((call(_ENSURE_STATES),) if any_filled else ()),
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
    _merge_variables_by_name(combined, lifecycle_variables | {STATUS_VARIABLE})

    for old, new in list_renames.items():
        _rename_list(combined, old, new)
    for old, new in procedure_renames.items():
        _rename_procedure(combined, old, new)
    for old, new in broadcast_renames.items():
        _rename_broadcast(combined, old, new)
    entry_results = [
        replace(
            entry_result,
            procedure=name_of(entry_result.procedure),
            completion_broadcast=name_of(entry_result.completion_broadcast),
            inputs=tuple(
                replace(value, list_name=name_of(value.list_name)) for value in entry_result.inputs
            ),
            outputs=tuple(
                replace(value, list_name=name_of(value.list_name)) for value in entry_result.outputs
            ),
            state_inputs=() if _private_states else entry_result.state_inputs,
            state_updates=() if _private_states else entry_result.state_updates,
        )
        for entry_result in entry_results
    ]

    program_capacities = {name_of(name): size for name, size in interface_capacities.items()}
    program_capacities.update({
        name_of(_state_list(state.name)): (
            state.capacity * math.prod(state.initial.shape[1:])
            if state.mode == "append" else state.initial.numel()
        )
        for state in program.states
    })
    program_capacities.update({
        f"_cattorch initial state {state.name}": state.initial.numel()
        for state in program.states if fills[state.name] is None
    })
    public_data_names = {STATUS_VARIABLE}
    for entry_result in entry_results:
        public_data_names.update(value.list_name for value in entry_result.inputs)
        public_data_names.update(value.list_name for value in entry_result.outputs)
    if not _private_states:
        public_data_names.update(_state_list(state.name) for state in program.states)
    lifecycle_procedures = ("cattorch init", "cattorch reset", "cattorch prepare for save")
    entry_procedures = tuple(name_of(f"cattorch {entry.name}") for entry in program.entrypoints)
    public_procedures = {*lifecycle_procedures, *entry_procedures}
    if _edit_program is not None:
        defaults = [
            *lifecycle_procedures,
            *(_state_list(state.name) for state in program.states),
            *(
                f"cattorch {entry.name}{suffix}" for entry in program.entrypoints
                for suffix in ("", " complete")
            ),
            *(
                _input_list(entry.name, binding.name) for entry in program.entrypoints
                for binding in (*entry.arguments, *entry.returns)
                if isinstance(binding, (Input, Output))
            ),
        ]
        _edit_program(combined, ProgramAssembly(
            names={name: name_of(name) for name in defaults},
            states={state.name: name_of(_state_list(state.name)) for state in program.states},
            capacities=program_capacities,
            public_data_names=public_data_names,
            public_procedures=public_procedures,
        ))
    public_data_names = {name for name in public_data_names if not name.startswith("_")}
    public_procedures = {name for name in public_procedures if not name.startswith("_")}

    program_layouts = shard_sprite_lists(combined, program_capacities)
    sharded_names = {name_of(name) for name in sharded_names}
    sharded_names.update(
        layout.name for layout in program_layouts.values()
        if len(layout.shards) > 1
    )
    for index, state in enumerate(program.states):
        value = fills[state.name]
        if value is None:
            continue
        # Built after sharding, so each physical list is filled directly up to
        # its capacity rather than through the routed append of a logical one.
        layout = program_layouts[name_of(_state_list(state.name))]
        remaining = state.initial.numel()
        fill_body: list[Statement] = []
        for shard in layout.shards:
            count = min(shard.capacity, remaining)
            remaining -= count
            fill_body.append(clear(shard.name))
            if count:
                fill_body.append(repeat(count, (append(shard.name, value),)))
        combined = _add_warp_procedure(
            combined,
            _fill_procedure(index),
            Program(
                f"program_fill_{index}",
                lists=[shard.name for shard in layout.shards],
                body=fill_body,
            ),
            x=1280,
            y=160 + 160 * index,
        )

    finalized = finalize_sprite(
        combined,
        output_path,
        sprite_name=sprite_name,
        codegen=codegen,
        public_data_names=frozenset(public_data_names),
        public_procedures=frozenset(public_procedures),
    )
    state_results = () if _private_states else tuple(
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
    procedures = tuple(
        name for name in (*lifecycle_procedures, *entry_procedures)
        if not name.startswith("_")
    )
    result = ProgramResult(
        path=finalized.path,
        sprite_name=sprite_name,
        archive_bytes=finalized.archive_bytes,
        expanded_json_bytes=finalized.expanded_json_bytes,
        block_count=len(combined.get("blocks", {})),
        list_count=len(combined.get("lists", {})),
        sharded_lists=tuple(sorted(
            final_data_name(name, frozenset(public_data_names)) for name in sharded_names
        )),
        warnings=tuple(dict.fromkeys((*sub_warnings, *finalized.warnings))),
        entrypoints=tuple(entry_results),
        states=state_results,
        procedures=procedures,
        quantization=quantization_report,
    )
    if any(
        isinstance(binding, Input) and binding.dynamic is not None
        for entry in program.entrypoints for binding in entry.arguments
    ):
        from cattorch.dynamic import check_entry_lengths, exact_export
        from cattorch.verify import verify

        check_entry_lengths(
            model, program, result, verify, adapters=adapters, quantization=quantization,
            exact=exact_export(
                optimization=optimization, storage=storage, quantization=quantization,
                sig_figs=sig_figs,
            ),
        )
    return result


def _entry_by_item(error, entrypoint, wrapper, args, dynamic_inputs, capture):
    """Export an entry point for one item and run it once per item.

    Used when its operations can't follow a variable-length input directly.
    The input must be its only variable-length one, with no append state,
    the method must update no state, and its items must be independent.
    """
    from cattorch.dynamic import per_unit_result, run_per_unit, unit_output_axes

    if len(dynamic_inputs) != 1 or any(
        isinstance(binding, StateUpdate) for binding in entrypoint.returns
    ):
        raise error
    ((index, (name, minimum, maximum)),) = dynamic_inputs.items()
    if not isinstance(entrypoint.arguments[index], Input):
        raise error
    spec = {index: (name, minimum, maximum, 0)}
    # The full example, whose items differ, shows mixing across them.
    examples = list(args)
    examples[index] = entrypoint.arguments[index].example
    try:
        axes = unit_output_axes(wrapper, examples, [], spec, None)
    except UnsupportedModelError:
        raise error from None
    input_shapes = [tuple(value.shape) for value in args]
    result = capture(
        args, None, lambda sprite: run_per_unit(sprite, spec, input_shapes, len(axes)),
    )
    return per_unit_result(
        result, spec, axes, input_shapes, [tuple(value.shape) for value in result.outputs],
    )


__all__ = ["transpile_program"]
