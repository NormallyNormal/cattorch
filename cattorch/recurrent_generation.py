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
from cattorch.frontend import tensor_output_leaves
from cattorch.huggingface_recurrent import (
    adapt_recurrent_model, find_recurrent_model, version_hint,
)
from cattorch.interface import STATUS_VARIABLE
from cattorch.program import (
    EntryPoint, ExportProgram, Input, Output, ProgramCall, RecurrentGenerationProgram, State,
    StateInput, StateUpdate,
)
from cattorch.program_export import ProgramAssembly, transpile_program
from cattorch.quantization import QuantizationConfig
from cattorch.results import TranspileResult
from cattorch.sprite import (
    _TOP_K_IDS, _TOP_K_VALUES, _add_generation_blocks, _add_procedure_completion_broadcast,
    _add_warp_procedure, _remove_completion_broadcast,
)
from cattorch.storage import StorageConfig
from cattorch.util.scratch.dsl import Program, call, clear, eq, if_, not_, set_var, var
from cattorch.util.scratch.interface import rename_list, rename_procedure, rename_variable

_ENTRY = "step"
_COMPUTE = "_cattorch token compute"
_STEP = "_cattorch token step"
_RESET = "_cattorch reset state"
_RESTORE = "_cattorch restore states"
_TOKENS = "cattorch tokens"
_LOGITS = "cattorch logits"
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
    """Run the step over random tokens and compare each logits row with the reference.

    ``hint`` ends a mismatch's message.
    """
    assert program.reference is not None and program.states is not None
    length = min(program.max_tokens or 32, 32)
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, vocabulary, (1, length), generator=generator)
    step = getattr(model, program.step)
    states = [state.initial for state in program.states]
    rows = []
    with torch.no_grad():
        expected = tensor_output_leaves(getattr(model, program.reference)(tokens), warn=False)[0]
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


def _calibration_calls(
    sequences: Iterable[torch.Tensor | tuple[torch.Tensor, ...]], vocabulary: int,
) -> tuple[tuple[ProgramCall, ...], frozenset[int]]:
    """One step call per token, and the calls that start a sequence."""
    calls: list[ProgramCall] = []
    starts = set()
    for sequence in check_calibration_tokens(sequences, vocabulary):
        if isinstance(sequence, tuple) and len(sequence) == 1:
            sequence = sequence[0]
        if not isinstance(sequence, torch.Tensor) or sequence.ndim != 2 or sequence.shape[0] != 1:
            raise ValueError(
                "RecurrentGenerationProgram calibration_sequences must be [1, length] "
                "token tensors"
            )
        starts.add(len(calls))
        calls.extend(
            ProgramCall(_ENTRY, (sequence[:, position:position + 1].long(),))
            for position in range(sequence.shape[1])
        )
    return tuple(calls), frozenset(starts)


def _delete_procedure(sprite: dict, name: str) -> None:
    """Remove a procedure's definition and every block under it, if it exists."""
    blocks = sprite["blocks"]
    definitions = {
        identifier for identifier, block in blocks.items()
        if block.get("opcode") == "procedures_definition"
        and blocks.get(block["inputs"]["custom_block"][1], {}).get(
            "mutation", {},
        ).get("proccode") == name
    }
    if not definitions:
        return
    roots: dict[str, str] = {}

    def root(identifier: str) -> str:
        path = []
        while identifier not in roots and blocks[identifier].get("parent") is not None:
            path.append(identifier)
            identifier = blocks[identifier]["parent"]
        top = roots.get(identifier, identifier)
        for visited in (*path, identifier):
            roots[visited] = top
        return top

    for identifier in [identifier for identifier in blocks if root(identifier) in definitions]:
        del blocks[identifier]


def _generation_interface(
    program: RecurrentGenerationProgram, vocabulary: int, *, compact_top_k: bool,
):
    """Wrap the step entry point in the public generation blocks."""

    def edit(sprite: dict, assembly: ProgramAssembly) -> None:
        # Like a GenerationProgram: no prepare-for-save block and no
        # notification per step.
        _delete_procedure(sprite, assembly.names["cattorch prepare for save"])
        _delete_procedure(sprite, f"_cattorch {_ENTRY} prepare for save")
        assembly.public_procedures.discard("cattorch prepare for save")
        completion = assembly.names[f"cattorch {_ENTRY} complete"]
        _remove_completion_broadcast(sprite, completion)
        sprite["broadcasts"] = {
            identifier: message for identifier, message in sprite["broadcasts"].items()
            if message != completion
        }
        # The program's reset restores the initial states; generation's
        # reset also clears the outputs and counters.
        rename_procedure(sprite, "cattorch reset", _RESTORE)
        rename_variable(sprite, f"_cattorch {_ENTRY} initialized", "cattorch initialized")

        top_k_lists = (_TOP_K_VALUES, _TOP_K_IDS) if program.top_k is not None else ()
        outputs = (_LOGITS, *top_k_lists)
        _add_warp_procedure(
            sprite,
            _STEP,
            Program(
                "recurrent_generation_step",
                variables=(STATUS_VARIABLE,),
                lists=outputs,
                body=(
                    set_var(STATUS_VARIABLE, "ok"),
                    call(_COMPUTE),
                    # A failed step commits nothing; clear what it would have written.
                    if_(not_(eq(var(STATUS_VARIABLE), "ok")), tuple(clear(name) for name in outputs)),
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
            tokens=_TOKENS,
            logits=_LOGITS,
            capacity=program.max_tokens,
            hidden_prefill=program.hidden_prefill,
            top_k=program.top_k,
            compact_top_k=compact_top_k,
            vocabulary=vocabulary,
        )
        if program.max_tokens is None:
            # The documented loop stops when the cache length reaches the
            # maximum context, which an unbounded generator never does.
            sprite["variables"]["cattorch_recurrent_max_context"] = [
                "cattorch max context", "Infinity",
            ]
        rename_list(sprite, "cattorch prefill buffer", "_cattorch prefill buffer")
        if program.max_tokens is not None:
            assembly.capacities["_cattorch prefill buffer"] = program.max_tokens
        assembly.public_procedures.update(("cattorch prefill", "cattorch decode"))

        _remove_completion_broadcast(sprite, "cattorch init complete")
        _add_procedure_completion_broadcast(
            sprite, "cattorch init", "cattorch init complete", status="ok",
        )
        for procedure in ("cattorch reset", "cattorch prefill", "cattorch decode"):
            _add_procedure_completion_broadcast(sprite, procedure, f"{procedure} complete")

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
) -> TranspileResult:
    """Export ``program.step`` with the init, reset, prefill and decode blocks."""
    hint = ""
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
        program = replace(
            program, states=adapted.states, step=adapted.step, reference=adapted.reference,
        )
        hint = version_hint()
    assert program.states is not None
    for method, role in ((program.step, "recurrent step"), (program.reference, "reference")):
        if method is not None and not callable(getattr(model, method, None)):
            raise ValueError(f"model has no callable {role} {method!r}")
    if calibration_sequences is not None and (
        quantization is None or quantization.method != "gptq"
    ):
        raise ValueError("calibration_sequences can only be used with GPTQ quantization")
    if any(
        value.is_floating_point() and value.dtype != torch.float32
        for value in (*model.parameters(), *model.buffers())
    ):
        # Scratch computes in doubles either way; in float32 the step can be
        # checked against the reference without bfloat16 rounding noise.
        model = copy.deepcopy(model).float()
    states = tuple(
        State(state.name, state.initial.float())
        if state.initial.is_floating_point() else state
        for state in program.states
    )
    program = RecurrentGenerationProgram(
        states,
        step=program.step,
        max_tokens=program.max_tokens,
        hidden_prefill=program.hidden_prefill,
        top_k=program.top_k,
        reference=program.reference,
    )
    checked = (
        copy.deepcopy(model).eval() if any(module.training for module in model.modules())
        else model
    )
    vocabulary, logit_count = _check_step(checked, program)
    if program.top_k is not None and program.top_k > logit_count:
        raise ValueError(f"top_k ({program.top_k}) exceeds output size ({logit_count})")
    if program.reference is not None:
        _check_reference(checked, program, vocabulary, hint)

    calls, resets = (
        _calibration_calls(calibration_sequences, vocabulary)
        if calibration_sequences is not None else (None, frozenset())
    )
    export = ExportProgram(
        states=program.states,
        entrypoints=(EntryPoint(
            _ENTRY,
            method=program.step,
            arguments=(
                Input("token", torch.zeros(1, 1, dtype=torch.long)),
                *(StateInput(state.name) for state in program.states),
            ),
            returns=(Output("logits"), *(StateUpdate(state.name) for state in program.states)),
        ),),
    )
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
            f"cattorch {_ENTRY} token": _TOKENS,
            f"cattorch {_ENTRY} logits": _LOGITS,
        },
        _private_states=True,
        _hidden_outputs=(
            frozenset({f"cattorch {_ENTRY} logits"}) if program.hidden_prefill else frozenset()
        ),
        _calibration_resets=resets,
        _calibration_name="calibration_sequences",
        # The generation blocks guard the step and reset.
        _guard_overlaps=False,
    )
    (entry,) = result.entrypoints
    return TranspileResult(
        path=result.path,
        sprite_name=result.sprite_name,
        archive_bytes=result.archive_bytes,
        expanded_json_bytes=result.expanded_json_bytes,
        block_count=result.block_count,
        list_count=result.list_count,
        sharded_lists=result.sharded_lists,
        warnings=result.warnings,
        inputs=entry.inputs,
        output=entry.outputs[0],
        procedures=("cattorch init", "cattorch reset", "cattorch prefill", "cattorch decode"),
        quantization=result.quantization,
        fallback_operations=result.fallback_operations,
        approximate_operations=result.approximate_operations,
    )


__all__ = ["transpile_recurrent_generation"]
