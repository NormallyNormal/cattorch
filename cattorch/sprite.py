"""Assemble a plain export's sprite around its compiled program.

The compiled program becomes the body of ``cattorch forward`` behind an
idempotent ``cattorch init`` (``_wrap_optimized_lifecycle``). This module
also gates parts of the chain, checks the forward's inputs, adds
``cattorch prepare for save``, sizes the lists before sharding and removes
what nothing uses. The other pieces of a sprite have their own modules:
``procedures`` (custom-block helpers), ``guards`` (stopped and overlapping
calls), ``static_storage`` (weights), ``generation_blocks`` (reset, prefill
and decode) and ``layer_sharing`` (one custom block for repeated layers).
"""

from __future__ import annotations

import logging
import math
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, AbstractSet

import torch

from cattorch import interface
from cattorch.graph import _GenerationSpec, _GraphInfo, _get_shape
from cattorch.guards import _EPOCH, _epoch_current
from cattorch.interface import STATUS_VARIABLE, is_public_data
from cattorch.procedures import _add_warp_procedure, _merge_variables_by_name, _procedure_mutation
from cattorch.util.scratch.dsl import Program, call, clear, eq, if_, set_var, var
from cattorch.util.scratch.interface import rename_procedure as _rename_procedure
from cattorch.util.scratch.remap import remap_ids
from cattorch.util.scratch.sharding import ListLayout

if TYPE_CHECKING:
    from cattorch.transpiler import _Compiler

log = logging.getLogger(__name__)


def _merge_duplicate_lists(sprite):
    """Merge local lists (names starting with '_') that have identical contents."""
    lists = sprite.get("lists", {})

    def local_base(name: str) -> str:
        for index in range(len(name) - 1, 0, -1):
            if name[index] == '_' and name[index + 1:].isdigit():
                return name[:index]
        return name

    # Only combine repeated instances of the same logical local.  Two
    # differently named empty work lists can be live simultaneously inside a
    # single instruction (notably the MoE router's logits, IDs, and weights),
    # so content alone is not a safe aliasing proof.
    by_content: dict[tuple[str, tuple], list[tuple[str, str]]] = {}
    for sid, entry in lists.items():
        display_name = entry[0]
        if not display_name.startswith("_"):
            continue
        content_key = (local_base(display_name), tuple(entry[1]))
        by_content.setdefault(content_key, []).append((sid, display_name))

    remap = {}
    for group in by_content.values():
        if len(group) < 2:
            continue
        winner_sid = group[0][0]
        for sid, display_name in group[1:]:
            log.info("Merging duplicate list %s (%s) -> %s (%s)",
                     display_name, sid, group[0][1], winner_sid)
            remap[sid] = winner_sid
            del lists[sid]

    if not remap:
        return

    sprite["blocks"] = remap_ids(sprite["blocks"], remap, share_unchanged=True)

    base_counts: dict[str, int] = {}
    for entry in lists.values():
        name = entry[0]
        if not name.startswith("_"):
            continue
        base = name
        for i in range(len(name) - 1, 0, -1):
            if name[i] == '_' and name[i+1:].isdigit():
                base = name[:i]
                break
        count = base_counts.get(base, 0) + 1
        base_counts[base] = count
        entry[0] = base if count == 1 else f"{base}_{count}"


def _remove_unused(sprite):
    """Remove unreachable blocks and unreferenced data slots."""
    blocks = sprite.get("blocks", {})
    reachable: set[str] = set()
    pending = [
        identifier
        for identifier, block in blocks.items()
        if block.get("topLevel") and block.get("parent") is None
    ]
    while pending:
        identifier = pending.pop()
        if identifier in reachable or identifier not in blocks:
            continue
        reachable.add(identifier)
        stack = [blocks[identifier]]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, (list, tuple)):
                stack.extend(value)
            elif isinstance(value, str) and value in blocks and value not in reachable:
                pending.append(value)
    for identifier in set(blocks) - reachable:
        log.info("Removing unreachable block: %s", identifier)
        del blocks[identifier]

    referenced: set[str] = set()
    stack = [blocks]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        elif isinstance(value, str):
            referenced.add(value)

    for section in ("lists", "variables"):
        slots = sprite.get(section, {})
        # Public names stay even when no block reads them, such as the input
        # of a model that only uses its shape.
        unused = [
            sid for sid, entry in slots.items()
            if sid not in referenced
            and not entry[0].startswith("cattorch ")
            and not is_public_data(entry[0], is_list=section == "lists")
        ]
        for sid in unused:
            log.info("Removing unused %s: %s", section.rstrip("s"), slots[sid][0])
            del slots[sid]


def _wrap_optimized_lifecycle(sprite: dict) -> None:
    """Put the compiled stack behind idempotent init and warp forward blocks."""
    blocks = sprite["blocks"]
    roots = [
        bid for bid, block in blocks.items()
        if block.get("topLevel") and block.get("parent") is None
        and block.get("opcode") != "procedures_definition"
    ]
    if len(roots) != 1:
        raise ValueError(f"expected one compiled root before lifecycle wrapping, got {roots}")

    suffix = uuid.uuid4().hex[:8]
    prefix = f"cattorch_lifecycle_{suffix}"
    initialized_id = f"{prefix}_initialized"
    initialized = interface.INITIALIZED_VARIABLE
    sprite.setdefault("variables", {})[initialized_id] = [initialized, 0]

    old_root = roots[0]
    blocks[old_root]["topLevel"] = False
    blocks[old_root].pop("x", None)
    blocks[old_root].pop("y", None)

    forward_name = interface.FORWARD
    init_name = interface.INIT
    forward_call = f"{prefix}_forward_call"
    init_call = f"{prefix}_init_call"
    forward_def = f"{prefix}_forward_definition"
    forward_proto = f"{prefix}_forward_prototype"
    init_def = f"{prefix}_init_definition"
    init_proto = f"{prefix}_init_prototype"
    init_check = f"{prefix}_init_check"
    init_check_expr = f"{prefix}_init_check_expr"
    nested_init_call = f"{prefix}_nested_init_call"
    set_initialized = f"{prefix}_set_initialized"

    for call_id, procedure, y in ((forward_call, forward_name, 0), (init_call, init_name, 80)):
        blocks[call_id] = {
            "next": None, "parent": None, "inputs": {}, "fields": {},
            "shadow": False, "topLevel": True,
            "opcode": "procedures_call",
            "mutation": _procedure_mutation(procedure),
            "x": 0, "y": y,
        }

    blocks[forward_def] = {
        "opcode": "procedures_definition", "next": init_check, "parent": None,
        "inputs": {"custom_block": [1, forward_proto]}, "fields": {},
        "shadow": False, "topLevel": True, "x": 320, "y": 0,
    }
    blocks[forward_proto] = {
        "opcode": "procedures_prototype", "next": None, "parent": forward_def,
        "inputs": {}, "fields": {}, "shadow": True, "topLevel": False,
        "mutation": _procedure_mutation(forward_name, warp=True),
    }
    blocks[init_def] = {
        "opcode": "procedures_definition", "next": set_initialized, "parent": None,
        "inputs": {"custom_block": [1, init_proto]}, "fields": {},
        "shadow": False, "topLevel": True, "x": 320, "y": 160,
    }
    blocks[init_proto] = {
        "opcode": "procedures_prototype", "next": None, "parent": init_def,
        "inputs": {}, "fields": {}, "shadow": True, "topLevel": False,
        "mutation": _procedure_mutation(init_name, warp=True),
    }

    blocks[init_check] = {
        "opcode": "control_if", "next": old_root, "parent": forward_def,
        "inputs": {
            "CONDITION": [2, init_check_expr],
            "SUBSTACK": [2, nested_init_call],
        },
        "fields": {}, "shadow": False, "topLevel": False,
    }
    blocks[init_check_expr] = {
        "opcode": "operator_equals", "next": None, "parent": init_check,
        "inputs": {
            "OPERAND1": [3, [12, initialized, initialized_id], [10, ""]],
            "OPERAND2": [1, [4, 0]],
        },
        "fields": {}, "shadow": False, "topLevel": False,
    }
    blocks[nested_init_call] = {
        "opcode": "procedures_call", "next": None, "parent": init_check,
        "inputs": {}, "fields": {}, "shadow": False, "topLevel": False,
        "mutation": _procedure_mutation(init_name),
    }
    blocks[set_initialized] = {
        "opcode": "data_setvariableto", "next": None, "parent": init_def,
        "inputs": {"VALUE": [1, [4, 1]]},
        "fields": {"VARIABLE": [initialized, initialized_id]},
        "shadow": False, "topLevel": False,
    }
    blocks[old_root]["parent"] = init_check


def _logical_list_sizes(
    compiler: _Compiler,
    graph: _GraphInfo,
    generation: _GenerationSpec | None,
    output_name: str | None,
) -> dict[str, int]:
    """Infer peak logical capacities for every tensor list before sharding."""
    sizes: dict[str, int] = {}

    def record(name: str, shape) -> None:
        try:
            size = math.prod(shape)
        except TypeError:
            return
        sizes[name] = max(sizes.get(name, 0), int(size))

    for node in graph.nodes:
        if node.name in compiler.scope.assignments:
            record(f"T{compiler.scope.assignments[node.name]}", _get_shape(node))
        if node.name in graph.input_names:
            record(graph.input_names[node.name], _get_shape(node))

    for key, tensor in compiler.static_lists.items():
        if isinstance(tensor, torch.Tensor):
            sizes[f"W_{key}"] = tensor.numel()

    if output_name and output_name in sizes:
        sizes["output"] = sizes.pop(output_name)

    if generation is not None:
        for name, cache_width in graph.generation.cache_widths.items():
            sizes[name] = generation.max_context * cache_width
        sizes[interface.PREFILL_BUFFER] = generation.max_context
        if generation.top_k is not None:
            sizes[interface.TOP_K_VALUES] = generation.top_k
            sizes[interface.TOP_K_IDS] = generation.top_k
    for name, size in compiler.logical_sizes.items():
        sizes[name] = max(sizes.get(name, 0), size)
    return sizes


def _add_prepare_for_save(
    sprite: dict,
    layouts: dict[str, ListLayout],
    *,
    static_names: set[str],
    compressed: bool,
    inputs: AbstractSet[str] = frozenset(),
) -> None:
    """Add the public procedure that drops rebuildable runtime list contents.

    It marks the weights undecoded and clears the decoded weights and the
    working lists. The ``inputs`` lists are kept: they hold the caller's
    values, or are public lists, and a program's persistent state is passed
    to its entry points as inputs.
    """
    names = [
        shard.name
        for layout in layouts.values()
        if not layout.name.startswith("_") and layout.name not in inputs
        and (compressed or layout.name not in static_names)
        for shard in layout.shards
    ]
    initialized = interface.INITIALIZED_VARIABLE
    program = Program(
        "prepare_for_save",
        variables=(initialized,),
        lists=names,
        body=(set_var(initialized, 0), *(clear(name) for name in names)),
    )
    _add_warp_procedure(sprite, interface.PREPARE_FOR_SAVE, program, x=640, y=640)
    _merge_variables_by_name(sprite, {initialized})


def _gate_chain(
    sprite: dict, start: str, gate: str = interface.PROJECT_OUTPUT, stop: str | None = None,
) -> None:
    """Run the command chain from ``start`` only while the ``gate`` variable is 1.

    The chain runs to its end, or up to ``stop``, which then follows the new
    ``if`` block. Hidden prefill uses it to skip what a prompt token's logits
    alone need, such as the final layer's attention output and the output
    projection. The gate variable is created, set to 1, if the sprite has none.
    """
    blocks = sprite["blocks"]
    predecessors = [
        identifier for identifier, block in blocks.items() if block.get("next") == start
    ]
    if len(predecessors) != 1:
        raise ValueError(f"expected one block before the gated chain, got {predecessors}")
    predecessor = predecessors[0]
    last = start
    while blocks[last].get("next") not in (stop, None):
        last = blocks[last]["next"]
    if blocks[last].get("next") != stop:
        raise ValueError("the gated chain does not end at the given block")
    blocks[last]["next"] = None

    variables = sprite.setdefault("variables", {})
    variable_id = next(
        (identifier for identifier, entry in variables.items() if entry[0] == gate), None,
    )
    if variable_id is None:
        variable_id = f"cattorch_gate_{uuid.uuid4().hex[:8]}"
        variables[variable_id] = [gate, 1]
    token = uuid.uuid4().hex[:8]
    guard = f"cattorch_gate_{token}_if"
    condition = f"cattorch_gate_{token}_condition"
    blocks[predecessor]["next"] = guard
    blocks[guard] = {
        "opcode": "control_if", "next": stop, "parent": predecessor,
        "inputs": {"CONDITION": [2, condition], "SUBSTACK": [2, start]},
        "fields": {}, "shadow": False, "topLevel": False,
    }
    blocks[condition] = {
        "opcode": "operator_equals", "next": None, "parent": guard,
        "inputs": {
            "OPERAND1": [3, [12, gate, variable_id], [10, ""]],
            "OPERAND2": [1, [4, 1]],
        },
        "fields": {}, "shadow": False, "topLevel": False,
    }
    blocks[start]["parent"] = guard
    if stop is not None:
        blocks[stop]["parent"] = guard


# What ``cattorch forward`` becomes once a wrapper checks its inputs first.
_CHECKED_FORWARD = "_cattorch checked forward"


def _add_input_checks(sprite: dict, checks: Sequence[Sequence], lists: Sequence[str]) -> None:
    """Make ``cattorch forward`` set the status, and run only if ``checks`` leave it ok.

    ``checks`` are groups of statements, each run only while the status is
    still ok, so that it reports the first problem. The forward becomes the
    internal ``_cattorch checked forward``. ``lists`` are the lists the
    checks read.
    """
    ok = eq(var(STATUS_VARIABLE), interface.OK)
    _rename_procedure(sprite, interface.FORWARD, _CHECKED_FORWARD)
    wrapper = Program(
        "checked_forward",
        variables=(STATUS_VARIABLE, "_cattorch check index", _EPOCH),
        lists=tuple(dict.fromkeys(lists)),
        body=(
            set_var(STATUS_VARIABLE, interface.OK),
            *(if_(ok, tuple(group)) for group in checks),
            # A forward taken over while it checked computes nothing; the
            # forward later runs in an epoch procedure (_guard_plain_calls).
            if_(ok, (if_(_epoch_current(), (call(_CHECKED_FORWARD),)),)),
        ),
    )
    _add_warp_procedure(sprite, interface.FORWARD, wrapper, x=640, y=960)
    # The loose forward block on the canvas runs the public block, not the
    # procedure it wraps.
    for block in sprite["blocks"].values():
        if (
            block.get("opcode") == "procedures_call" and block.get("topLevel")
            and block["mutation"].get("proccode") == _CHECKED_FORWARD
        ):
            block["mutation"] = _procedure_mutation(interface.FORWARD)

