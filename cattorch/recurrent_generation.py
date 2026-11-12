"""Token-by-token generation for models whose sequence mixing is a recurrence.

A ``RecurrentGenerationProgram`` is lowered as an ``ExportProgram`` with one
entry point (a token and the states in, logits and the new states out) and
the generation blocks of a ``GenerationProgram`` around it: the same
procedures, lists, statuses and broadcasts.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from cattorch.adapters import ModuleAdapter
from cattorch.autoregressive import (
    EmbeddingLookup, check_calibration_tokens, lookup_vocabulary, row_mismatch,
)
from cattorch.codegen import CodegenConfig
from cattorch.errors import UnsupportedModelError
from cattorch.fast import FastConfig
from cattorch.frontend import capture_model, tensor_output_leaves
from cattorch.huggingface_recurrent import (
    adapt_recurrent_model, find_recurrent_model, version_hint,
)
from cattorch import interface
from cattorch.interface import STATUS_VARIABLE
from cattorch.operator_registry import eval_mode_copy_warning
from cattorch.program import (
    EntryPoint, ExportProgram, Input, Output, ProgramCall, RecurrentGenerationProgram, State,
    StateInput, StateUpdate,
)
from cattorch.program_export import ProgramAssembly, transpile_program
from cattorch.program_runtime import MethodModule
from cattorch.quantization import QuantizationConfig
from cattorch.results import GenerationResult, generation_result
from cattorch.sprite import (
    _STEP_LOGITS, _STEP_TOKEN, _TOP_K_IDS, _TOP_K_VALUES, _add_generation_blocks,
    _add_warp_procedure, _finish_generation_prepare, _guard_generation_init,
    _remove_completion_broadcast,
)
from cattorch.storage import StorageConfig
from cattorch.util.scratch.dsl import Program, call, eq, if_, set_var, var
from cattorch.util.scratch.interface import rename_list, rename_procedure, rename_variable
from cattorch.util.scratch.sharding import SCRATCH_LIST_LIMIT

_ENTRY = "step"
_COMPUTE = "_cattorch token compute"
_STEP = "_cattorch token step"
_RESET = "_cattorch reset state"
_RESTORE = "_cattorch restore states"
# As in the check of a cached generation step.
_TOLERANCE = 1e-4


def _check_step(model: nn.Module, program: RecurrentGenerationProgram) -> tuple[int, int]:
    """Run the step once from the initial states; return its vocabulary and logit count."""
    assert program.states is not None
    step = getattr(model, program.step)
    token = torch.zeros(1, 1, dtype=torch.long)
    lookup = EmbeddingLookup(token)
    with torch.no_grad(), lookup:
        returned = step(token, *(state.initial for state in program.states))
    leaves = tensor_output_leaves(returned, warn=False)
    if len(leaves) != 1 + len(program.states):
        raise UnsupportedModelError(
            f"step returned {len(leaves)} tensors, expected 1 + number of states "
            f"({1 + len(program.states)})"
        )
    logits, *updates = leaves
    for state, update in zip(program.states, updates):
        if update.shape != state.initial.shape:
            raise UnsupportedModelError(
                f"step changes the shape of state {state.name!r} from "
                f"{tuple(state.initial.shape)} to {tuple(update.shape)}"
            )
    if logits.ndim == 0 or math.prod(logits.shape[:-1]) != 1:
        raise UnsupportedModelError(
            f"the step's logits must flatten to one vocabulary row, not shape "
            f"{tuple(logits.shape)}"
        )
    vocabulary = lookup_vocabulary(
        lambda token: step(token, *(state.initial for state in program.states)), lookup,
    )
    if vocabulary is None:
        raise UnsupportedModelError(
            "the step's token must index an embedding table, so token IDs can be "
            "checked against the vocabulary"
        )
    return vocabulary, logits.numel()


def _check_reference(
    model: nn.Module, program: RecurrentGenerationProgram, vocabulary: int, hint: str,
) -> None:
    """Run the traced step over random tokens and compare each logits row with the reference.

    The step is traced as the export traces it, one tensor argument per
    declared input, and the traced graph (what cattorch lowers to blocks)
    runs token by token from the initial states, so a step that disagrees
    with the reference and a trace that binds its inputs wrongly are both
    caught. Lowering the graph to blocks is not part of this check.
    ``hint`` ends a mismatch's message.
    """
    assert program.reference is not None and program.states is not None
    length = min(program.max_context or 32, 32)
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, vocabulary, (1, length), generator=generator)
    states = [state.initial for state in program.states]
    with torch.no_grad():
        expected = tensor_output_leaves(getattr(model, program.reference)(tokens), warn=False)[0]
    # Traced after the reference has run, as the export traces it: a
    # model's first forward may change it once, as RWKV rescales its weights.
    step = capture_model(
        MethodModule(model, program.step), (tokens[:, :1], *states),
    ).graph_module
    rows = []
    with torch.no_grad():
        for position in range(length):
            logits, *states = tensor_output_leaves(
                step(tokens[:, position:position + 1], *states), warn=False,
            )
            rows.append(logits.reshape(-1))
    width = rows[0].numel()
    if expected.numel() != length * width or expected.shape[-1] != width:
        raise UnsupportedModelError(
            f"{program.reference} returned logits of shape {tuple(expected.shape)} for "
            f"{length} tokens; expected [1, {length}, {width}] or [{length}, {width}]"
        )
    expected = expected.reshape(length, width)
    for row, actual in enumerate(rows):
        mismatch = row_mismatch(actual, expected[row], _TOLERANCE)
        if mismatch is not None:
            largest, scale = mismatch
            overall = max(
                float((rows[other].double() - expected[other].double()).abs().max())
                for other in range(length)
            )
            raise UnsupportedModelError(
                f"the recurrent step disagrees with {program.reference} at logits row "
                f"{row}: max error {largest:.3g} against values up to {scale:.3g}, "
                f"tolerance {_TOLERANCE:g} relative; max error over all rows {overall:.3g}"
                + hint
            )


def step_program(states: tuple[State, ...], step: str) -> ExportProgram:
    """The one-entry-point ``ExportProgram`` a recurrent generator is lowered as."""
    return ExportProgram(
        states=states,
        entrypoints=(EntryPoint(
            _ENTRY,
            method=step,
            arguments=(
                Input("token", torch.zeros(1, 1, dtype=torch.long)),
                *(StateInput(state.name) for state in states),
            ),
            returns=(Output("logits"), *(StateUpdate(state.name) for state in states)),
        ),),
    )


def _calibration_calls(
    sequences: Iterable[torch.Tensor | tuple[torch.Tensor, ...]], vocabulary: int,
) -> tuple[tuple[ProgramCall, ...], frozenset[int]]:
    """One step call per token, and the calls that start a sequence."""
    calls: list[ProgramCall] = []
    starts: set[int] = set()
    for sequence in check_calibration_tokens(sequences, vocabulary):
        if isinstance(sequence, tuple) and len(sequence) == 1:
            sequence = sequence[0]
        if not isinstance(sequence, torch.Tensor) or sequence.ndim != 2 or sequence.shape[0] != 1:
            raise ValueError(
                f"a RecurrentGenerationProgram's calibration= takes [1, length] token "
                f"tensors, and item {len(starts)} isn't one"
            )
        starts.add(len(calls))
        calls.extend(
            ProgramCall(_ENTRY, (sequence[:, position:position + 1].long(),))
            for position in range(sequence.shape[1])
        )
    return tuple(calls), frozenset(starts)


def _generation_interface(
    program: RecurrentGenerationProgram, vocabulary: int, *, compact_top_k: bool,
):
    """Wrap the step entry point in the public generation blocks."""

    def edit(sprite: dict, assembly: ProgramAssembly) -> None:
        # Like a GenerationProgram: no notification per step, and the
        # program's reset and prepare for save become generation's.
        completion = assembly.names[interface.completion(f"cattorch {_ENTRY}")]
        _remove_completion_broadcast(sprite, completion)
        sprite["broadcasts"] = {
            identifier: message for identifier, message in sprite["broadcasts"].items()
            if message != completion
        }
        # The program's reset restores the initial states; generation's
        # reset also clears the outputs and counters.
        rename_procedure(sprite, interface.RESET, _RESTORE)
        _remove_completion_broadcast(sprite, interface.completion(interface.RESET))
        _remove_completion_broadcast(sprite, interface.completion(interface.PREPARE_FOR_SAVE))
        _remove_completion_broadcast(sprite, interface.completion(interface.INIT))
        rename_variable(sprite, f"_cattorch {_ENTRY} initialized", interface.INITIALIZED_VARIABLE)

        top_k_lists = (_TOP_K_VALUES, _TOP_K_IDS) if program.top_k is not None else ()
        _add_warp_procedure(
            sprite,
            _STEP,
            Program(
                "recurrent_generation_step",
                variables=(STATUS_VARIABLE, interface.INITIALIZED_VARIABLE),
                body=(
                    # Decode the weights as cattorch init does, as a KV
                    # generation step does.
                    if_(eq(var(interface.INITIALIZED_VARIABLE), 0), (call(interface.INIT),)),
                    set_var(STATUS_VARIABLE, interface.OK),
                    call(_COMPUTE),
                ),
            ),
            x=1600,
            y=0,
        )
        _add_generation_blocks(
            sprite,
            step=_STEP,
            reset=_RESET,
            restore=_RESTORE,
            tokens=interface.TOKEN_IDS,
            step_tokens=_STEP_TOKEN,
            logits=interface.LOGITS,
            step_logits=_STEP_LOGITS,
            capacity=program.max_context,
            hidden_prefill=program.hidden_prefill,
            top_k=program.top_k,
            compact_top_k=compact_top_k,
            vocabulary=vocabulary,
        )
        if program.max_context is None:
            # The documented loop stops when the cache length reaches the
            # maximum context, which an unbounded generator never does.
            sprite["variables"]["cattorch_recurrent_max_context"] = [
                interface.MAX_CONTEXT, "Infinity",
            ]
        rename_list(sprite, "cattorch prefill buffer", "_cattorch prefill buffer")
        # The prompt arrives in the one `cattorch token ids` list: no more
        # than a Scratch list holds, however large max_context is.
        assembly.capacities["_cattorch prefill buffer"] = min(
            program.max_context or SCRATCH_LIST_LIMIT, SCRATCH_LIST_LIMIT,
        )
        logits = assembly.capacities.get(_STEP_LOGITS, 0)
        assembly.capacities[interface.LOGITS] = logits
        assembly.public_data_names.update((interface.TOKEN_IDS, interface.LOGITS, *top_k_lists))
        assembly.public_procedures.update((interface.PREFILL, interface.DECODE))
        _guard_generation_init(sprite)
        _finish_generation_prepare(sprite, _RESET)

    return edit


def transpile_recurrent_generation(
    model: nn.Module,
    program: RecurrentGenerationProgram,
    output_path: str | Path,
    sig_figs: int | None,
    *,
    name: str | None,
    optimization: Literal["exact", "fast"],
    fast_config: FastConfig | None,
    storage: StorageConfig | None,
    quantization: QuantizationConfig | None,
    calibration_sequences: Iterable[torch.Tensor | tuple[torch.Tensor, ...]] | None,
    codegen: CodegenConfig | None,
    adapters: tuple[ModuleAdapter, ...],
    kernels=(),
) -> GenerationResult:
    """Export ``program.step`` with the init, reset, prefill and decode blocks."""
    hint = ""
    # Whether ``model`` is a copy this export made, which the export can then
    # adapt and quantize in place instead of copying again.
    owned = False
    if program.states is None:
        adapted = adapt_recurrent_model(model)
        if adapted is None:
            found = find_recurrent_model(model)
            if found is not None:
                path, family = found
                raise UnsupportedModelError(
                    f"RecurrentGenerationProgram() exports a Hugging Face {family} model "
                    f"only on its own, and this {type(model).__name__} holds one at "
                    f"{path!r}: export model.{path} directly, or pass states=... and a "
                    "step method"
                )
            raise UnsupportedModelError(
                "RecurrentGenerationProgram needs states=... unless the model is a "
                "supported Hugging Face recurrent model (Mamba, Mamba-2, RWKV)"
            )
        model = adapted.module
        owned = True
        program = replace(
            program, states=adapted.states, step=adapted.step, reference=adapted.reference,
        )
        hint = version_hint()
    assert program.states is not None
    for method, field in ((program.step, "step"), (program.reference, "reference")):
        if method is not None and not callable(getattr(model, method, None)):
            raise ValueError(
                f"model has no callable method {method!r}, which "
                f"RecurrentGenerationProgram({field}=...) names"
            )
    if not owned:
        # The checks below run the step, which may change module state (an
        # embedding with max_norm renormalizes its rows): they run on the
        # export's copy, never on the caller's model.
        model = copy.deepcopy(model)
        owned = True
    eval_mode_copy_warning(model)
    model.eval()
    if any(
        value.is_floating_point() and value.dtype != torch.float32
        for value in (*model.parameters(), *model.buffers())
    ):
        # Scratch computes in doubles either way; in float32 the step can be
        # checked against the reference without bfloat16 rounding noise.
        model.float()
    states = tuple(
        State(state.name, state.initial.float())
        if state.initial.is_floating_point() else state
        for state in program.states
    )
    program = RecurrentGenerationProgram(
        states,
        step=program.step,
        max_context=program.max_context,
        hidden_prefill=program.hidden_prefill,
        top_k=program.top_k,
        reference=program.reference,
    )
    vocabulary, logit_count = _check_step(model, program)
    if program.top_k is not None and program.top_k > logit_count:
        raise ValueError(f"top_k ({program.top_k}) exceeds output size ({logit_count})")
    if program.reference is not None:
        _check_reference(model, program, vocabulary, hint)

    calls, resets = (
        _calibration_calls(calibration_sequences, vocabulary)
        if calibration_sequences is not None else (None, frozenset())
    )
    export = step_program(program.states, program.step)
    codegen = codegen or CodegenConfig()
    result = transpile_program(
        model, export, output_path, sig_figs,
        name=name, optimization=optimization, fast_config=fast_config, storage=storage,
        quantization=quantization, calibration_calls=calls, codegen=codegen,
        adapters=adapters, kernels=kernels,
        _edit_program=_generation_interface(
            program, vocabulary, compact_top_k=codegen.unrolling != "speed",
        ),
        _public_names={
            f"cattorch {_ENTRY}": _COMPUTE,
            f"cattorch {_ENTRY} token": _STEP_TOKEN,
            f"cattorch {_ENTRY} logits": _STEP_LOGITS,
        },
        _private_states=True,
        _hidden_outputs=(
            frozenset({f"cattorch {_ENTRY} logits"}) if program.hidden_prefill else frozenset()
        ),
        _calibration_resets=resets,
        _calibration_name="calibration",
        # The generation blocks guard the step and reset, and check the tokens.
        _guard_overlaps=False,
        _initialized_flag=False,
        _check_items=False,
        _owned_model=owned,
        _logits="logits",
    )
    (entry,) = result.entrypoints
    return generation_result(
        result,
        max_context=program.max_context,
        top_k=program.top_k,
        vocabulary_size=vocabulary,
        logits=logit_count,
    )


__all__ = ["step_program", "transpile_recurrent_generation"]
