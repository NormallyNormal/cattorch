"""Read-only FX compatibility and Scratch cost analysis."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace

import torch
from torch import nn

from cattorch.adapters import ModuleAdapter
from cattorch.export_options import validate_export_options
from cattorch.errors import UnsupportedModelError, UnsupportedOperationError
from cattorch.frontend import capture_model
from cattorch.graph import _prepare_graph
from cattorch.moe import (
    SparseMoE, StackedSwiGLUMoE, _BankedSparseMoE, _expert_family_context,
)
from cattorch.operator_registry import default_registry
from cattorch.program import ExportProgram, Output
from cattorch.program_runtime import MethodModule, entry_examples, validate_entry_outputs
from cattorch.results import TensorSpec
from cattorch.storage import StorageConfig


@dataclass(frozen=True)
class EntryPointAnalysis:
    """Capture and lowering summary for one callable entrypoint."""

    name: str
    method: str
    operations: tuple[tuple[str, int], ...]
    module_paths: tuple[str, ...]
    unsupported_operations: tuple[str, ...]
    outputs: tuple[TensorSpec, ...]


@dataclass(frozen=True)
class MoEAnalysis:
    """Stored and active parameter summary for one semantic sparse MoE."""

    module_path: str
    experts: int
    top_k: int
    stored_expert_parameters: int
    active_expert_parameters: int
    router_parameters: int
    executor: str = "generic"


@dataclass(frozen=True)
class AnalysisReport:
    """Read-only report produced without generating a sprite."""

    entrypoints: tuple[EntryPointAnalysis, ...]
    moe: tuple[MoEAnalysis, ...]
    unique_parameters: int
    unique_buffers: int
    estimated_payload_bytes: int
    estimated_json_bytes: int
    warnings: tuple[str, ...]


def _unique_values(values) -> int:
    seen = set()
    count = 0
    for value in values:
        identity = (
            value.untyped_storage().data_ptr(), value.storage_offset(),
            tuple(value.shape), tuple(value.stride()), value.dtype,
        )
        if identity not in seen:
            seen.add(identity)
            count += value.numel()
    return count


def _storage_density(storage: StorageConfig) -> float:
    return {
        "float32": 5.0,
        "float16": 2.5,
        "int8": 1.25,
        "int6": 0.9375,
        "int4": 0.625,
    }[storage.precision]


def analyze(
    model: nn.Module,
    program: ExportProgram,
    *,
    adapters: tuple[ModuleAdapter, ...] = (),
    storage: StorageConfig | None = None,
    optimization: str = "exact",
) -> AnalysisReport:
    """Capture every entrypoint and report compatibility and approximate size."""
    if not isinstance(program, ExportProgram):
        raise TypeError("program must be an ExportProgram")
    validate_export_options(optimization=optimization, storage=storage, adapters=adapters)
    registry = default_registry().clone()
    for adapter in adapters:
        registry.register_adapter(adapter)
    owned = registry.adapt_model(
        model, state_names=frozenset(state.name for state in program.states),
    )
    generic_expert_errors = []
    from cattorch.util.argument import Argument
    from cattorch.util.instruction.optimized_moe import ExpertFamilyMoEInstruction
    for path, module in owned.named_modules():
        if not isinstance(module, _BankedSparseMoE):
            continue
        family = module.expert_family
        probe_args = [
            Argument("value", family.example_input.shape),
            Argument("selection", torch.Size((1, family.expert_count))),
            Argument("combination", torch.Size((1, family.expert_count))),
            *(
                Argument(f"bank_{index}", bank.shape)
                for index, bank in enumerate(family.unique_banks())
            ),
        ]
        try:
            instruction = ExpertFamilyMoEInstruction(
                "cattorch.expert_family_moe.default", "output", *probe_args,
                family=family, top_k=module.top_k,
                normalize_selected=module.normalize_selected,
            )
            instruction._expert_program()
        except ValueError as error:
            generic_expert_errors.append(
                f"ExpertFamily at {path or '<root>'}: {error}"
            )
    entries = []
    for entrypoint in program.entrypoints:
        args, dynamic_inputs = entry_examples(program, entrypoint)
        wrapper = MethodModule(owned, entrypoint.method)
        with _expert_family_context(registry.expert_families):
            captured = capture_model(wrapper, args, frontend="fx")
        counts = Counter(
            str(node.target) for node in captured.graph.nodes if node.op == "call_function"
        )
        paths = {
            value[0]
            for node in captured.graph.nodes if node.op == "call_function"
            for value in node.meta.get("nn_module_stack", {}).values()
            if value and value[0]
        }
        unsupported = list(generic_expert_errors)
        outputs = ()
        try:
            graph = _prepare_graph(
                wrapper, args, optimization=optimization,
                _registry=registry, dynamic_inputs=dynamic_inputs, _captured=captured,
            )
            specs = graph.output_specs()
            validate_entry_outputs(program, entrypoint, specs)
            outputs = tuple(
                replace(spec, list_name=f"cattorch {entrypoint.name} {binding.name}")
                for binding, spec in zip(entrypoint.returns, specs)
                if isinstance(binding, Output)
            )
        except (UnsupportedModelError, UnsupportedOperationError) as error:
            unsupported.append(str(error))
        entries.append(EntryPointAnalysis(
            name=entrypoint.name,
            method=entrypoint.method,
            operations=tuple(sorted(counts.items())),
            module_paths=tuple(sorted(paths)),
            unsupported_operations=tuple(dict.fromkeys(unsupported)),
            outputs=outputs,
        ))

    moe_reports = []
    for path, module in owned.named_modules():
        if isinstance(module, StackedSwiGLUMoE):
            stored = sum(value.numel() for value in (
                module.gate_weight, module.up_weight, module.down_weight,
            ))
            one = stored // module.gate_weight.shape[0]
            moe_reports.append(MoEAnalysis(
                module_path=path,
                experts=module.gate_weight.shape[0],
                top_k=module.top_k,
                stored_expert_parameters=stored,
                active_expert_parameters=one * module.top_k,
                router_parameters=module.router_weight.numel(),
                executor="stacked_swiglu",
            ))
            continue
        if not isinstance(module, (SparseMoE, _BankedSparseMoE)):
            continue
        banks = module.expert_family.unique_banks()
        expert_values = sum(value.numel() for value in banks)
        one_expert = sum(value[0].numel() for value in banks)
        moe_reports.append(MoEAnalysis(
            module_path=path,
            experts=module.expert_family.expert_count,
            top_k=module.top_k,
            stored_expert_parameters=expert_values,
            active_expert_parameters=one_expert * module.top_k,
            router_parameters=sum(value.numel() for value in module.router.parameters()),
            executor="generic",
        ))
    storage = storage or StorageConfig()
    parameters = _unique_values(owned.parameters())
    buffers = _unique_values(owned.buffers())
    payload = math.ceil((parameters + buffers) * _storage_density(storage))
    warnings = []
    # Expert template errors are repeated in every entrypoint; count them once.
    unsupported_count = len({
        operation for entry in entries for operation in entry.unsupported_operations
    })
    if unsupported_count:
        warnings.append(f"{unsupported_count} captured operation contracts are unsupported")
    return AnalysisReport(
        entrypoints=tuple(entries),
        moe=tuple(moe_reports),
        unique_parameters=parameters,
        unique_buffers=buffers,
        estimated_payload_bytes=payload,
        estimated_json_bytes=payload + 650_000,
        warnings=tuple(warnings),
    )


__all__ = ["AnalysisReport", "EntryPointAnalysis", "MoEAnalysis", "analyze"]
