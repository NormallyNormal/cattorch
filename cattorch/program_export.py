"""Lower an ``ExportProgram`` interface into one Scratch sprite."""

from __future__ import annotations

import math
import uuid
from collections import Counter
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
from cattorch.graph import JointLengthError
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
    BoundedDimension, EntryPointResult, ProgramResult, StateResult, TensorSpec, fallback_warning,
)
from cattorch.quantization import quantize_model
from cattorch import interface
from cattorch.interface import STATUS_VARIABLE, input_name, is_public_data, output_name
from cattorch.layer_sharing import (
    NO_REPEATED_LAYERS, NO_REPEATED_LAYERS_IN_PROGRAM, layer_sharing_warning,
)
from cattorch.storage import StorageConfig, quantized_storage
from cattorch.guards import (
    _BUSY, _EPOCH, _add_epoch_procedure, _epoch_current, _init_outcome, _overtaken,
    _stop_overtaken_decoding,
)
from cattorch.procedures import (
    _add_warp_procedure, _merge_lists_by_name, _merge_variables_by_name, _remove_canvas_calls,
)
from cattorch.util.paths import output_file
from cattorch.util.scratch.block_combiner import _merge_slots
from cattorch.util.scratch.dsl import (
    Program, Statement, add, append, argument, broadcast_and_wait, call, change_var, clear, eq,
    for_each, gt, if_, if_else, item, length, mod, mul, not_, repeat, set_var, var,
)
from cattorch.util.scratch.finalize_scratch import finalize_sprite
from cattorch.util.scratch.interface import (
    rename_lists as _rename_lists, rename_variable as _rename_variable,
    rename_variables as _rename_variables, rename_procedure as _rename_procedure,
    refresh_data_display_names,
)
from cattorch.util.scratch.remap import remap_ids
from cattorch.util.scratch.sharding import SCRATCH_LIST_LIMIT, shard_sprite_lists


class _ProgramCalibrationModule(nn.Module):
    """Present a sequence of named program calls to the shared GPTQ observer."""

    def __init__(
        self, model: nn.Module, program: ExportProgram, calls: tuple[ProgramCall, ...],
        resets: frozenset[int] = frozenset(),
    ):
        super().__init__()
        self.model = model
        self.calls = calls
        self.resets = resets
        self.position = 0
        self.replay = ProgramReplay(model, program)

    def forward(self, _sentinel: torch.Tensor):
        if self.position in self.resets:
            self.replay.reset()
        invocation = self.calls[self.position]
        self.position += 1
        # A method may assign its module's buffers, as a marked entry point
        # installs each state as one. Calibration only observes: the calls
        # run under inference mode, and a buffer they replaced would stay an
        # inference tensor that later in-place updates can't touch.
        buffers = [(module, dict(module._buffers)) for module in self.model.modules()]
        try:
            return self.replay.run(invocation)
        finally:
            for module, saved in buffers:
                module._buffers.clear()
                module._buffers.update(saved)


def _program_output(
    program: ExportProgram, calls: tuple[ProgramCall, ...], name: str,
) -> Callable[[list[torch.Tensor]], torch.Tensor | None]:
    """Pick the output ``name`` from what each of ``calls`` returns, in turn."""
    positions = {
        entry.name: next(
            (
                index for index, binding in enumerate(entry.returns)
                if isinstance(binding, Output) and binding.name == name
            ),
            None,
        )
        for entry in program.entrypoints
    }
    remaining = iter(calls)

    def pick(leaves: list[torch.Tensor]) -> torch.Tensor | None:
        position = positions[next(remaining).entrypoint]
        return None if position is None else leaves[position]

    return pick


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
    _rename_lists(sprite, {old: f"_cattorch {namespace} {old}" for old in names})
    variable_names = [
        entry[0]
        for entry in sprite.get("variables", {}).values()
        if entry[0].startswith(("cattorch payload ", "cattorch scale payload "))
        or (
            entry[0].startswith("cattorch ") and " bank" in entry[0]
            and entry[0].endswith((" payload", " table"))
        )
    ]
    _rename_variables(sprite, {old: f"_cattorch {namespace} {old}" for old in variable_names})
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
        refresh_data_display_names(
            sprite["blocks"], {winner: slots[winner][0] for winner in set(remap.values())},
        )
        for identifier in remap:
            slots.pop(identifier, None)


_state_list = interface.state_list
_input_list = interface.entry_list


def _temporary_return(entrypoint: str, index: int) -> str:
    return f"_cattorch {entrypoint} return {index}"


def _copy_list(source: str, destination: str, index_name: str):
    return (
        clear(destination),
        for_each(index_name, length(source), (append(destination, item(source, var(index_name))),)),
    )


_ENSURE_STATES = "_cattorch ensure states"
# With the calling block's epoch.
_ENSURE_STATES_IN_EPOCH = "_cattorch ensure states %s"
_STATES_FILLED = "_cattorch states filled"


def _all_initialized(flags) -> tuple[Statement, ...]:
    """Set ``cattorch initialized`` to 1 when every entry point's weights are decoded.

    Each entry point decodes its own weights on its first call; the flags
    are each one's ``_cattorch ENTRY initialized``, 0 or 1.
    """
    product = var(flags[0])
    for flag in flags[1:]:
        product = mul(product, var(flag))
    return (set_var(interface.INITIALIZED_VARIABLE, product),)


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


def _rename_broadcasts(sprite: dict, renames: dict[str, str]) -> None:
    names: dict[str, str] = {}
    for old, new in renames.items():
        for identifier, message in sprite.get("broadcasts", {}).items():
            if message == old:
                sprite["broadcasts"][identifier] = names[identifier] = new
    refresh_data_display_names(sprite.get("blocks", {}), names)


def _add_split_checks(sprite: dict, split_checks, layouts, name_of) -> dict:
    """Add the procedures that check the input and state lists the program split.

    Each checks a list's total length, each shard's own length
    (``interface.shard_checks``), and then the items in every shard.
    """
    for position, (procedure, name, shown, length_options, kind) in enumerate(split_checks):
        layout = layouts[name_of(name)]
        shards = [shard.name for shard in layout.shards]
        ok = eq(var(STATUS_VARIABLE), interface.OK)
        body: list[Statement] = list(
            interface.length_checks(shown, lists=shards, **length_options),
        )
        if len(shards) > 1:
            body.append(if_(ok, interface.shard_checks(shown, shards, layout.shards[0].capacity)))
        if kind is not None:
            items = interface.item_checks(shown, *kind, lists=shards)
            if items:
                body.append(if_(ok, items))
        sprite = _add_warp_procedure(sprite, procedure, Program(
            f"program_split_check_{position}",
            variables=(STATUS_VARIABLE, "_cattorch check index"), lists=shards, body=body,
        ), x=2240, y=160 + 160 * position)
    return sprite


def _build_entry_wrapper(
    entrypoint: EntryPoint,
    result,
    program: ExportProgram,
    ensure_states: bool = False,
    hidden: frozenset[int] = frozenset(),
    gate: str = "",
    guarded: bool = False,
    display: Callable[[str], str] = lambda name: name,
    finish: tuple[Statement, ...] = (),
    finish_variables: tuple[str, ...] = (),
    item_kinds: Mapping[int, tuple] = {},
    index_checked: frozenset[int] = frozenset(),
    split_checks: list | None = None,
) -> tuple[Program, tuple[TensorSpec, ...]]:
    """The procedure body that checks, computes and commits one entry point.

    The ``hidden`` return positions are checked and copied only while the
    ``gate`` variable is 1. A ``guarded`` body is run by
    ``_add_epoch_procedure`` and commits nothing once another call started.
    A call that fails writes no output and changes no state. ``display``
    gives a list's final name, for the status; ``finish`` ends the body,
    whatever happened, with the ``finish_variables``. An unguarded body then
    sends its completion broadcast, which a guarded one's public block sends. ``item_kinds`` say
    what each input's items must be (``interface.item_checks``), by position;
    a guarded body computes nothing once another call started. The
    ``index_checked`` inputs, by position, are those indices the entry point
    computes come from: an index out of range fails the call naming the
    input as the compiled entry point knows it, ``input_name(position)``.
    An input or state list longer than a Scratch list is checked by a
    procedure the body calls; its ``(procedure, list, shown name, length
    options, item kind)`` is added to ``split_checks``, for
    ``_add_split_checks`` to build once the lists are split.
    """
    states = {state.name: state for state in program.states}
    status = STATUS_VARIABLE
    internal = f"_cattorch {entrypoint.name} compute"
    completion = interface.completion(f"cattorch {entrypoint.name}")
    public_outputs = []
    lists = set()
    variables = {status, gate, *finish_variables} if hidden else {status, *finish_variables}
    body: list[Statement] = [set_var(status, interface.OK)]
    if ensure_states:
        body.append(
            call(_ENSURE_STATES_IN_EPOCH, epoch=argument("epoch")) if guarded
            else call(_ENSURE_STATES)
        )

    ok = eq(var(status), interface.OK)
    for index, (binding, spec) in enumerate(zip(entrypoint.arguments, result.inputs)):
        name = (
            _input_list(entrypoint.name, binding.name)
            if isinstance(binding, Input) else _state_list(binding.name)
        )
        lists.add(name)
        if spec.numel is None:
            # The dimensions before the variable one have size 1, so its
            # items are rows of the other dimensions' values.
            trailing = math.prod(extent for extent in spec.shape if isinstance(extent, int))
            rows = next(extent for extent in spec.shape if isinstance(extent, BoundedDimension))
            minimum = rows.minimum
            # An append state can be empty.
            minimum = max(minimum, 1) if isinstance(binding, Input) else 0
            length_options: dict = dict(
                width=trailing, min_items=trailing * minimum, max_items=spec.max_numel,
            )
        else:
            length_options = dict(numel=spec.numel)
        kind = item_kinds.get(index) if isinstance(binding, Input) else None
        capacity = spec.numel if spec.numel is not None else spec.max_numel
        if split_checks is not None and capacity is not None and capacity > SCRATCH_LIST_LIMIT:
            # The list will be split: its shards are checked one by one, in
            # a procedure built once the program knows them.
            procedure = f"_cattorch check {name}"
            split_checks.append((procedure, name, display(name), length_options, kind))
            body.append(if_(ok, (call(procedure),)))
            continue
        body.append(if_(ok, interface.length_checks(display(name), lists=(name,), **length_options)))
        if kind is not None:
            items = interface.item_checks(display(name), *kind, lists=(name,))
            if items:
                variables.add("_cattorch check index")
                body.append(if_(ok, items))

    temporary_specs = result.outputs
    validate_entry_outputs(program, entrypoint, temporary_specs)
    output_lists = []
    for index, (binding, spec) in enumerate(zip(entrypoint.returns, temporary_specs)):
        temporary = _temporary_return(entrypoint.name, index)
        lists.add(temporary)
        if isinstance(binding, Output):
            destination = _input_list(entrypoint.name, binding.name)
            lists.add(destination)
            output_lists.append(destination)
            public_outputs.append(TensorSpec(
                list_name=destination,
                shape=spec.shape,
                dtype=spec.dtype,
                numel=spec.numel,
                max_numel=spec.max_numel,
            ))

    # What the model returned, checked against what the lists can hold.
    computation: list[Statement] = [call(internal)]
    for index in sorted(index_checked):
        binding = entrypoint.arguments[index]
        name = (
            _input_list(entrypoint.name, binding.name)
            if isinstance(binding, Input) else _state_list(binding.name)
        )
        computation.append(if_(eq(var(status), interface.invalid_item(input_name(index))), (
            set_var(status, interface.invalid_item(display(name))),
        )))
    for index, (binding, spec) in enumerate(zip(entrypoint.returns, temporary_specs)):
        temporary = _temporary_return(entrypoint.name, index)
        if isinstance(binding, Output):
            shown = display(_input_list(entrypoint.name, binding.name))
            if spec.numel is None:
                trailing = math.prod(extent for extent in spec.shape if isinstance(extent, int))
                checks = interface.length_checks(
                    shown, width=trailing, min_items=0, max_items=spec.max_numel,
                    lists=(temporary,),
                )
                computation.append(if_(ok, checks))
            else:
                checks = interface.length_checks(shown, numel=spec.numel, lists=(temporary,))
                computation.append(
                    if_(eq(var(gate), 1), checks) if index in hidden else if_(ok, checks),
                )
            continue
        state = states[binding.name]
        destination = _state_list(state.name)
        shown = display(destination)
        lists.add(destination)
        if state.mode == "replace":
            computation.append(if_(ok, interface.length_checks(
                shown, numel=state.initial.numel(), lists=(temporary,),
            )))
        else:
            item_size = math.prod(state.initial.shape[1:])
            maximum = state.capacity * item_size
            computation.append(if_(ok, (
                if_(gt(mod(length(temporary), item_size), 0), (
                    set_var(status, interface.wrong_number_of_items(shown)),
                )),
                if_(gt(add(length(destination), length(temporary)), maximum), (
                    set_var(status, interface.state_full(shown)),
                )),
            )))

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
                f"_cattorch {entrypoint.name} append {index}", length(temporary),
                (append(destination, item(
                    temporary, var(f"_cattorch {entrypoint.name} append {index}"),
                )),),
            ))
        else:
            copy = _copy_list(temporary, destination, f"_cattorch {entrypoint.name} copy {index}")
            commits.extend((if_(eq(var(gate), 1), copy),) if index in hidden else copy)
    if not guarded:
        computation.append(if_(ok, tuple(commits)))
        body.append(if_(ok, tuple(computation)))
        body.extend(finish)
        body.append(broadcast_and_wait(completion))
    else:
        # The commit is the call's last step, so a call that another one ran
        # during skips it. One paused part way, or stopped, leaves the states
        # marked busy until a reset; one another call started during leaves
        # its outputs mixed, and clears them.
        current = _epoch_current()
        variables.add(_EPOCH)
        mixed = tuple(clear(name) for name in output_lists)
        if any(isinstance(binding, StateUpdate) for binding in entrypoint.returns):
            variables.add(_BUSY)
            commits = [
                set_var(_BUSY, 1), *commits,
                if_else(
                    current,
                    # A call refused meanwhile set the status.
                    (set_var(_BUSY, 0), set_var(status, interface.OK)),
                    (set_var(_BUSY, 1), *mixed, *_overtaken()),
                ),
            ]
        else:
            commits.append(if_(not_(current), (*mixed, *_overtaken())))
        # A call taken over while it checked its inputs computes nothing.
        body.append(if_(ok, (if_(current, tuple(computation)),)))
        # Another call may have finished decoding the weights while this one
        # was, and this one then left them half overwritten.
        initialized = f"_cattorch {entrypoint.name} initialized"
        variables.add(initialized)
        body.append(if_else(
            current,
            (if_(ok, tuple(commits)),),
            (set_var(initialized, 0), *_overtaken()),
        ))
        body.extend(finish)
    return Program(
        f"entrypoint_{entrypoint.name}",
        variables=variables | {
            f"_cattorch {entrypoint.name} copy {index}"
            for index, binding in enumerate(entrypoint.returns)
            if not (
                isinstance(binding, StateUpdate)
                and states[binding.name].mode == "append"
            )
        } | {
            f"_cattorch {entrypoint.name} append {index}"
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
    _hidden_outputs: frozenset[str] = frozenset(),
    _calibration_resets: frozenset[int] = frozenset(),
    _calibration_name: str = "calibration",
    _guard_overlaps: bool = True,
    _owned_model: bool = False,
    _initialized_flag: bool = True,
    _check_items: bool = True,
    _logits: str | None = None,
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
    - ``_hidden_outputs`` names entry point output lists (by default name)
      that are computed, checked and written only while the
      ``cattorch project output`` variable is 1; the operations only they
      need are skipped otherwise.
    - ``_calibration_resets`` holds the positions in ``calibration_calls``
      before which GPTQ calibration resets every state.
    - ``_calibration_name`` is the caller's name for ``calibration_calls``,
      for its errors.
    - ``_guard_overlaps=False`` leaves out the busy flag and epoch that keep
      stopped and overlapping calls from changing the states unnoticed, for
      a caller that guards the entry points and ``cattorch reset`` itself.
    - ``_initialized_flag=False`` leaves out ``cattorch initialized``, and
      ``_check_items=False`` the check of each input item, for a caller that
      keeps the one and checks the inputs itself.
    - ``_owned_model=True`` says ``model`` is a copy the caller made and no
      longer needs, so it is adapted and quantized in place instead of
      copied again. Variable-length entry points still copy it: their
      check runs the original model afterwards.
    - ``_logits`` names the output binding that holds a generation export's
      logits, whose output head quantization stores with
      ``QuantizationConfig.head_bits``.

    The renaming and hidden-output hooks can't be combined with
    variable-length inputs, whose export check runs the entry points through
    their default names.
    """
    if not isinstance(program, ExportProgram):
        raise TypeError("program must be an ExportProgram")
    list_renames, procedure_renames, broadcast_renames = _interface_renames(
        program, _public_names, _private_states,
    )
    renames = {**list_renames, **procedure_renames, **broadcast_renames}
    output_lists = {
        _input_list(entry.name, binding.name): (entry.name, index)
        for entry in program.entrypoints
        for index, binding in enumerate(entry.returns) if isinstance(binding, Output)
    }
    unknown = sorted(set(_hidden_outputs) - set(output_lists))
    if unknown:
        raise ValueError(f"_hidden_outputs has no entry point output {unknown[0]!r}")
    if (renames or _hidden_outputs) and any(
        isinstance(binding, Input) and binding.dynamic is not None
        for entry in program.entrypoints for binding in entry.arguments
    ):
        raise ValueError(
            "_public_names, _private_states and _hidden_outputs need fixed-length inputs"
        )

    def name_of(default: str) -> str:
        return renames.get(default, default)

    validate_export_options(
        name=name, optimization=optimization, fast_config=fast_config,
        storage=storage, quantization=quantization, codegen=codegen,
        sig_figs=sig_figs, adapters=adapters,
    )
    output_path = Path(output_file(output_path, ".sprite3"))
    sprite_name = output_path.stem if name is None else name
    codegen = codegen or CodegenConfig()
    registry = default_registry().for_export(adapters=adapters, kernels=kernels)

    states = {state.name: state for state in program.states}
    # A state that starts as one constant is refilled by a loop on reset,
    # with no stored copy of its initial value. A private state's own list
    # is saved empty too, and filled by the first call or init. A public
    # state's list keeps its stored value: other sprites may read it before
    # any call, and a project has no hook that runs on load to fill it.
    fills = {state.name: _constant_fill(state) for state in program.states}
    empty_on_load = {
        name for name, value in fills.items() if value is not None and _private_states
    }
    any_filled = bool(empty_on_load)
    calls = tuple(calibration_calls or ())
    known_entries = {entry.name for entry in program.entrypoints}
    for call_value in calls:
        if call_value.entrypoint not in known_entries:
            raise ValueError(
                f"{_calibration_name} calls entry point {call_value.entrypoint!r}, which the "
                f"program doesn't have; its entry points are {sorted(known_entries)}"
            )
    if calibration_calls is not None and (quantization is None or quantization.method != "gptq"):
        raise ValueError(f"{_calibration_name}= is only used by GPTQ quantization")

    # Model ownership, transforms, and calibration belong to the whole
    # program. Every entrypoint below captures the same prepared module.
    dynamic_entries = any(
        isinstance(binding, Input) and binding.dynamic is not None
        for entry in program.entrypoints for binding in entry.arguments
    )
    prepared_model = registry.adapt_model(
        model, state_names=frozenset(states), _copy=not _owned_model or dynamic_entries,
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
            prepared_model, program, observed_calls, _calibration_resets if gptq else frozenset(),
        )
        sentinels = [torch.zeros(1) for _ in observed_calls]
        quantization_report = quantize_model(
            calibration_model,
            quantization,
            sentinels if gptq else None,
            orientation_inputs=None if gptq else sentinels,
            _registry=registry,
            _calibration_name=_calibration_name,
            _names_from=prepared_model,
            _logits_of=(
                None if _logits is None
                else _program_output(program, observed_calls, _logits)
            ),
        )
        prepared_model = calibration_model.model
        resolved_storage = quantized_storage(
            resolved_storage, quantization, quantization_report,
        )

    combined = None
    entry_results = []
    flags = [f"_cattorch {entry.name} initialized" for entry in program.entrypoints]
    all_initialized = _all_initialized(flags) if _initialized_flag else ()
    sub_warnings: list[str] = []
    # Why layer sharing skipped each entry point, when it was asked for.
    unshared_layers: dict[str, str | None] = {}
    fallbacks: Counter[str] = Counter()
    approximations: Counter[str] = Counter()
    sharded_names: set[str] = set()
    interface_capacities: dict[str, int] = {}
    payload_names: set[str] = set()
    storage_warnings: list[str] = []

    from cattorch.transpiler import (
        _checked_index_inputs, _compile_sprite, refuse_runtime_index_checks,
    )

    split_checks: list = []
    for position, entrypoint in enumerate(program.entrypoints):
        if not hasattr(model, entrypoint.method) or not callable(getattr(model, entrypoint.method)):
            raise ValueError(
                f"model has no callable method {entrypoint.method!r} for "
                f"entry point {entrypoint.name!r}"
            )
        args, dynamic_inputs = entry_examples(program, entrypoint)
        wrapper = MethodModule(prepared_model, entrypoint.method)
        sub_codegen = replace(
            codegen,
            compact_internal_names=False,
            compact_schema=False,
            id_namespace=None,
        )
        hidden = frozenset(
            index for name, (entry_name, index) in output_lists.items()
            if entry_name == entrypoint.name and name in _hidden_outputs
        )

        def capture(arguments, dynamic, edit=None):
            # Each entry point's sprite stays in memory; only the program is
            # finalized and written.
            return _compile_sprite(
                wrapper,
                tuple(arguments),
                sig_figs,
                optimization=optimization,
                fast_config=fast_config,
                storage=resolved_storage,
                quantization=quantization,
                codegen=sub_codegen,
                registry=registry,
                prepared_model=True,
                quantization_report=quantization_report,
                dynamic_inputs=dynamic,
                edit_sprite=edit,
                entry_point=True,
                hidden_outputs=hidden,
            )

        try:
            sub_result = capture(args, dynamic_inputs)
        except JointLengthError as error:
            raise UnsupportedModelError(f"entry point {entrypoint.name!r} {error}") from None
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
                f"entry point {entrypoint.name!r} captured an unexpected argument count"
            )
        validate_entry_outputs(program, entrypoint, sub_result.outputs)
        if not _guard_overlaps:
            # A recurrent generator's step, which gets the generation guards.
            refuse_runtime_index_checks(sub_result.graph.nodes, "RecurrentGenerationProgram")
        sprite = sub_result.sprite
        # An interface list's peak size, when the compiler used it as scratch
        # space for a larger intermediate first.
        peak_sizes = sub_result.interface_sizes

        storage_names = _namespace_entry_storage(sprite, entrypoint.name)
        payload_names.update(new for old, new in storage_names.items() if not old.startswith("W_"))

        # The entry point's own forward and init become private helpers of
        # the public blocks.
        _remove_canvas_calls(sprite, interface.FORWARD)
        _remove_canvas_calls(sprite, interface.INIT)
        internal = f"_cattorch {entrypoint.name} compute"
        _rename_procedure(sprite, interface.FORWARD, internal)
        _rename_procedure(sprite, interface.INIT, f"_cattorch {entrypoint.name} init")
        _rename_procedure(
            sprite, interface.PREPARE_FOR_SAVE,
            f"_cattorch {entrypoint.name} prepare for save",
        )
        _rename_variable(
            sprite, interface.INITIALIZED_VARIABLE,
            f"_cattorch {entrypoint.name} initialized",
        )
        _namespace_entry_procedures(sprite, entrypoint.name)
        interface_renames: dict[str, str] = {}
        for index, binding in enumerate(entrypoint.arguments):
            old = input_name(index)
            new = (
                _input_list(entrypoint.name, binding.name)
                if isinstance(binding, Input) else _state_list(binding.name)
            )
            interface_renames[old] = new
            spec = sub_result.inputs[index]
            interface_capacities[new] = max(
                interface_capacities.get(new, 0), spec.max_numel or 0,
            )
        for index in range(len(entrypoint.returns)):
            old = output_name(index)
            temporary = _temporary_return(entrypoint.name, index)
            private_unit = f"_cattorch unit {old}"
            if any(entry[0] == private_unit for entry in sprite.get("lists", {}).values()):
                # Run once per item: the entry's own output list became the
                # private unit list, and the public one only collects items.
                interface_capacities[private_unit] = max(
                    interface_capacities.get(private_unit, 0), peak_sizes.get(old, 0),
                )
                capacity = sub_result.outputs[index].max_numel or 0
            else:
                capacity = max(sub_result.outputs[index].max_numel or 0, peak_sizes.get(old, 0))
            interface_renames[old] = temporary
            interface_capacities[temporary] = capacity
        _rename_lists(sprite, interface_renames)

        combined = sprite if combined is None else _merge_sprite_targets(combined, sprite)
        wrapper_program, output_specs = _build_entry_wrapper(
            entrypoint, sub_result, program, ensure_states=any_filled,
            hidden=hidden, gate=interface.PROJECT_OUTPUT, guarded=_guard_overlaps, display=name_of,
            finish=all_initialized,
            finish_variables=(interface.INITIALIZED_VARIABLE, *flags) if _initialized_flag else (),
            item_kinds=sub_result.item_kinds if _check_items else {},
            index_checked=_checked_index_inputs(sub_result.graph),
            split_checks=split_checks,
        )
        for spec in output_specs:
            interface_capacities[spec.list_name] = max(
                interface_capacities.get(spec.list_name, 0), spec.max_numel or 0,
            )
        if _guard_overlaps:
            combined = _add_epoch_procedure(
                combined, f"cattorch {entrypoint.name}", wrapper_program,
                x=640, y=160 + 160 * position,
                refusal=(
                    (set_var(STATUS_VARIABLE, interface.INTERRUPTED),) if program.states else None
                ),
                variables=(STATUS_VARIABLE,),
                completion=interface.completion(f"cattorch {entrypoint.name}"),
            )
        else:
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
            procedure=interface.entry_procedure(entrypoint.name),
            completion_broadcast=interface.completion(interface.entry_procedure(entrypoint.name)),
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
            if warning.startswith(("GPTQ fallback", "tied weight"))
        )
        # Each entry point's report adds what it stored to the earlier ones',
        # so the last one's warnings say how the program stores each weight.
        storage_warnings = [
            warning for warning in sub_result.warnings
            if warning.startswith(("quantized weight", "weight computed from"))
        ]
        if codegen.layer_sharing:
            unshared_layers[entrypoint.name] = sub_result.unshared_layers
        fallbacks.update(dict(sub_result.fallback_operations))
        approximations.update(dict(sub_result.approximate_operations))
        sharded_names.update(storage_names.get(name, name) for name in sub_result.sharded_lists)
        quantization_report = sub_result.quantization

    assert combined is not None
    sub_warnings.extend(storage_warnings)
    _deduplicate_program_payloads(combined, payload_names)
    for state in program.states:
        if state.name in empty_on_load:
            # Filled by a loop at run time instead.
            _set_list_contents(combined, _state_list(state.name), [])
            continue
        values = state.initial.detach().cpu().flatten().tolist()
        _set_list_contents(combined, _state_list(state.name), values)
        if fills[state.name] is None:
            _set_list_contents(combined, f"_cattorch initial state {state.name}", values)

    lifecycle_lists = {_state_list(state.name) for state in program.states} | {
        f"_cattorch initial state {state.name}"
        for state in program.states if fills[state.name] is None
    }
    lifecycle_variables = {
        f"_cattorch reset {index}" for index, state in enumerate(program.states)
        if fills[state.name] is None
    }
    reset_body: list[Statement] = [set_var(STATUS_VARIABLE, interface.OK)]
    lifecycle_variables.add(STATUS_VARIABLE)
    for index, state in enumerate(program.states):
        if fills[state.name] is not None:
            reset_body.append(call(_fill_procedure(index)))
            continue
        reset_body.extend(_copy_list(
            f"_cattorch initial state {state.name}",
            _state_list(state.name),
            f"_cattorch reset {index}",
        ))
    ensure_call = (
        call(_ENSURE_STATES_IN_EPOCH, epoch=argument("epoch")) if _guard_overlaps
        else call(_ENSURE_STATES)
    )
    if any_filled:
        reset_body.append(set_var(_STATES_FILLED, 1))
        lifecycle_variables.add(_STATES_FILLED)
        fill_empty = tuple(
            if_(eq(length(_state_list(state.name)), 0), (call(_fill_procedure(index)),))
            for index, state in enumerate(program.states)
            if state.name in empty_on_load
        )
        # The first call or init fills the constant states still empty in a
        # freshly loaded project. Later, as before, only reset refills them.
        if _guard_overlaps:
            # Not over states a stopped or running call left busy.
            ensure = Program(
                "program_ensure_states", variables={_STATES_FILLED, _BUSY, _EPOCH},
                lists={_state_list(state.name) for state in program.states},
                body=(if_(eq(add(var(_STATES_FILLED), var(_BUSY)), 0), (
                    set_var(_BUSY, 1), *fill_empty, set_var(_STATES_FILLED, 1),
                    if_else(_epoch_current(), (set_var(_BUSY, 0),), (set_var(_BUSY, 1),)),
                )),),
            )
            combined = _add_warp_procedure(
                combined, _ENSURE_STATES_IN_EPOCH, ensure, x=1920, y=0, arguments=("epoch",),
            )
        else:
            ensure = Program(
                "program_ensure_states", variables={_STATES_FILLED},
                lists={_state_list(state.name) for state in program.states},
                body=(if_(eq(var(_STATES_FILLED), 0), (
                    *fill_empty, set_var(_STATES_FILLED, 1),
                )),),
            )
            combined = _add_warp_procedure(combined, _ENSURE_STATES, ensure, x=1920, y=0)
    init_body: list[Statement] = [*((ensure_call,) if any_filled else ())]
    for entry, flag in zip(program.entrypoints, flags):
        if _guard_overlaps:
            # A call running meanwhile decodes the weights itself.
            init_body.append(set_var(flag, 0))
        init_body.append(call(f"_cattorch {entry.name} init"))
    if _guard_overlaps:
        init_body.append(_init_outcome(flags))
        init_body.extend(all_initialized)
    else:
        init_body.extend((*all_initialized, set_var(STATUS_VARIABLE, interface.OK)))
    init_body.append(broadcast_and_wait(interface.completion(interface.INIT)))
    prepare_body: list[Statement] = [
        call(f"_cattorch {entry.name} prepare for save") for entry in program.entrypoints
    ]
    prepare_body.extend((
        *all_initialized, set_var(STATUS_VARIABLE, interface.OK),
        broadcast_and_wait(interface.completion(interface.PREPARE_FOR_SAVE)),
    ))
    if _guard_overlaps:
        lifecycle_variables.add(_EPOCH)
        if program.states:
            # A reset that another call ran during, or that is stopped, leaves
            # the states busy. A call refused meanwhile set the status.
            lifecycle_variables.add(_BUSY)
            reset_body.append(if_else(
                _epoch_current(), (set_var(_BUSY, 0), set_var(STATUS_VARIABLE, interface.OK)),
                (set_var(_BUSY, 1), *_overtaken()),
            ))
        else:
            reset_body.append(if_(not_(_epoch_current()), _overtaken()))
        # Clearing the weights takes them from a running call.
        prepare_body.insert(0, change_var(_EPOCH, 1))
    else:
        reset_body.extend((
            set_var(STATUS_VARIABLE, interface.OK),
            broadcast_and_wait(interface.completion(interface.RESET)),
        ))
    epoch = (_EPOCH,) if _guard_overlaps else ()
    initialized_variables = (
        (interface.INITIALIZED_VARIABLE, *flags) if _initialized_flag else tuple(flags)
    )
    lifecycle = (
        (interface.RESET, Program(
            "program_reset", variables=lifecycle_variables, lists=lifecycle_lists,
            body=reset_body,
        )),
        (interface.INIT, Program(
            "program_init", variables=(*epoch, *initialized_variables, STATUS_VARIABLE),
            body=init_body,
        )),
        (interface.PREPARE_FOR_SAVE, Program(
            "program_prepare", variables=(*epoch, *initialized_variables, STATUS_VARIABLE),
            body=prepare_body,
        )),
    )
    for position, (procedure, body_program) in enumerate(lifecycle):
        y = 160 + 160 * position
        if _guard_overlaps and procedure != interface.PREPARE_FOR_SAVE:
            combined = _add_epoch_procedure(
                combined, procedure, body_program, x=1280, y=y,
                take_over=(
                    (set_var(_BUSY, 1),) if procedure == interface.RESET and program.states
                    else ()
                ),
                completion=interface.completion(procedure) if procedure == interface.RESET else None,
            )
        else:
            combined = _add_warp_procedure(combined, procedure, body_program, x=1280, y=y)
    _merge_lists_by_name(combined, lifecycle_lists)
    _merge_variables_by_name(
        combined, lifecycle_variables | {
            STATUS_VARIABLE, *initialized_variables,
            # Each entry point's checks of its input items use one.
            "_cattorch check index",
        },
    )
    if _hidden_outputs:
        _merge_variables_by_name(combined, {interface.PROJECT_OUTPUT})
        for entry in combined["variables"].values():
            if entry[0] == interface.PROJECT_OUTPUT:
                entry[1] = 1

    _rename_lists(combined, list_renames)
    for old, new in procedure_renames.items():
        _rename_procedure(combined, old, new)
    _rename_broadcasts(combined, broadcast_renames)
    if _guard_overlaps:
        _stop_overtaken_decoding(combined)
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
    lifecycle_procedures = (interface.INIT, interface.RESET, interface.PREPARE_FOR_SAVE)
    entry_procedures = tuple(
        name_of(interface.entry_procedure(entry.name)) for entry in program.entrypoints
    )
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

    # Entry points have sharded their own lists already. A list still too
    # long, such as the storage decoder's powers table at a tiny limit, is a
    # fixed table: left whole, finalization rejects it as for a plain export.
    fixed_tables = frozenset(
        entry[0] for entry in combined.get("lists", {}).values()
        if entry[0] not in program_capacities and len(entry[1]) > SCRATCH_LIST_LIMIT
    )
    program_layouts = shard_sprite_lists(
        combined, program_capacities, deferred_names=fixed_tables,
    )
    combined = _add_split_checks(combined, split_checks, program_layouts, name_of)
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
            x=1920,
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
            name for name in sharded_names
            if is_public_data(name, frozenset(public_data_names))
        )),
        warnings=tuple(dict.fromkeys((
            *sub_warnings, *_layer_sharing_warnings(unshared_layers), *finalized.warnings,
            *filter(None, (fallback_warning(tuple(sorted(fallbacks.items()))),)),
        ))),
        entrypoints=tuple(entry_results),
        states=state_results,
        procedures=procedures,
        quantization=quantization_report,
        fallback_operations=tuple(sorted(fallbacks.items())),
        approximate_operations=tuple(sorted(approximations.items())),
    )
    if any(
        (isinstance(binding, Input) and binding.dynamic is not None)
        or (isinstance(binding, StateInput) and states[binding.name].mode == "append")
        for entry in program.entrypoints for binding in entry.arguments
    ):
        from cattorch.dynamic import check_entry_lengths

        check_entry_lengths(
            model, program, result, adapters=adapters, optimization=optimization,
            storage=storage, quantization=quantization, sig_figs=sig_figs,
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
    ((index, (name, minimum, maximum, axis)),) = dynamic_inputs.items()
    if not isinstance(entrypoint.arguments[index], Input):
        raise error
    spec = {index: (name, minimum, maximum, axis)}
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


def _layer_sharing_warnings(unshared: dict[str, str | None]) -> tuple[str, ...]:
    """Warnings for the entry points ``layer_sharing=True`` didn't share.

    ``unshared`` maps each entry point to why its layers weren't shared
    (``None`` if they were). An entry point without repeated layers, such as
    a reset, isn't reported, unless no entry point has any.
    """
    if not unshared:
        return ()
    if all(reason == NO_REPEATED_LAYERS for reason in unshared.values()):
        reason = NO_REPEATED_LAYERS if len(unshared) == 1 else NO_REPEATED_LAYERS_IN_PROGRAM
        return (layer_sharing_warning(reason),)
    if len(unshared) == 1:
        return tuple(
            layer_sharing_warning(reason) for reason in unshared.values() if reason is not None
        )
    return tuple(
        layer_sharing_warning(reason, name)
        for name, reason in unshared.items()
        if reason is not None and reason != NO_REPEATED_LAYERS
    )


__all__ = ["transpile_program"]

