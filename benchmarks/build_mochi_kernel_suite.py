"""Build focused Scratch-VM screens for instruction ideas found in Mochi."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from cattorch.benchmark import (
    SUITE_RESULTS,
    _StageBlocks,
    _add_broadcast_entry,
    _stage_target,
    _write_project,
)
from cattorch.templates.template import TEMPLATE_DIR
from cattorch.transpiler import _add_warp_procedure, _merge_lists_by_name
from cattorch.util.scratch.dsl import (
    Program,
    add,
    append,
    change_var,
    clear,
    gt,
    if_,
    if_else,
    item,
    mul,
    repeat,
    replace,
    set_var,
    sub,
    var,
)


OUTPUT = Path(__file__).parent / "artifacts" / "mochi_kernel_suite.sb3"


def _unrolled(count: int, advancing, final):
    """Four-way unroll with no dead final index updates."""
    chunks, remainder = divmod(count, 4)
    result = []
    if remainder:
        if chunks:
            result.append(repeat(chunks, tuple(advancing) * 4))
        result.extend(tuple(advancing) * (remainder - 1))
        result.extend(final)
    else:
        if chunks > 1:
            result.append(repeat(chunks - 1, tuple(advancing) * 4))
        result.extend(tuple(advancing) * 3)
        result.extend(final)
    return tuple(result)


def _values(size: int, multiplier: int, modulus: int, divisor: float):
    return [((index * multiplier) % modulus - modulus // 2) / divisor for index in range(size)]


def _linear_program(
    label: str,
    inner: int,
    outputs: int,
    *,
    indexing: str,
    preallocated: bool = False,
) -> Program:
    input_name = f"linear {inner}x{outputs} input"
    weight_name = f"linear {inner}x{outputs} weight"
    output_name = f"linear {inner}x{outputs} {label} output"
    variables = (
        "sum", "feature", "weight index", "weight start", "output index",
    )
    lists = (input_name, weight_name, output_name)
    setup = [] if preallocated else [clear(output_name)]
    setup.append(set_var("weight start", 1 if indexing == "dual" else 0))
    if preallocated:
        setup.append(set_var("output index", 1))

    if indexing == "dual":
        product = change_var(
            "sum",
            mul(item(input_name, var("feature")), item(weight_name, var("weight index"))),
        )
        advancing = (product, change_var("feature", 1), change_var("weight index", 1))
        dot = (
            set_var("feature", 1),
            set_var("weight index", var("weight start")),
            *_unrolled(inner, advancing, (product,)),
        )
    elif indexing == "row-base":
        product = change_var(
            "sum",
            mul(
                item(input_name, var("feature")),
                item(weight_name, add(var("weight start"), var("feature"))),
            ),
        )
        advancing = (product, change_var("feature", 1))
        dot = (
            set_var("feature", 1),
            *_unrolled(inner, advancing, (product,)),
        )
    else:
        raise ValueError(indexing)

    write = (
        replace(output_name, var("output index"), var("sum"))
        if preallocated else append(output_name, var("sum"))
    )
    advance_output = (change_var("output index", 1),) if preallocated else ()
    body = (
        *setup,
        repeat(outputs, (
            set_var("sum", 0),
            *dot,
            write,
            change_var("weight start", inner),
            *advance_output,
        )),
    )
    return Program(
        label,
        variables=variables,
        lists=lists,
        list_values={
            input_name: _values(inner, 17, 19, 8),
            weight_name: _values(inner * outputs, 29, 23, 16),
            output_name: [0] * outputs if preallocated else [],
        },
        body=body,
    )


def _head_program(label: str, *, fused_argmax: bool) -> Program:
    inner, outputs = 128, 1024
    input_name = "head input"
    weight_name = "head weight"
    logits_name = f"head {label} logits"
    result_name = f"head {label} result"
    lists = (input_name, weight_name, logits_name, result_name)
    variables = ("sum", "feature", "weight start", "output", "best", "best id")
    product = change_var(
        "sum",
        mul(
            item(input_name, var("feature")),
            item(weight_name, add(var("weight start"), var("feature"))),
        ),
    )
    dot = _unrolled(inner, (product, change_var("feature", 1)), (product,))
    per_output = [
        set_var("sum", 0),
        set_var("feature", 1),
        *dot,
    ]
    if fused_argmax:
        per_output.append(if_(gt(var("sum"), var("best")), (
            set_var("best", var("sum")),
            set_var("best id", var("output")),
        )))
    else:
        per_output.append(append(logits_name, var("sum")))
    per_output.extend((change_var("weight start", inner), change_var("output", 1)))

    finish = []
    if not fused_argmax:
        finish.extend((
            set_var("output", 1),
            repeat(outputs, (
                if_(gt(item(logits_name, var("output")), var("best")), (
                    set_var("best", item(logits_name, var("output"))),
                    set_var("best id", var("output")),
                )),
                change_var("output", 1),
            )),
        ))
    finish.extend((
        replace(result_name, 1, var("best id")),
        replace(result_name, 2, var("best")),
    ))
    return Program(
        label,
        variables=variables,
        lists=lists,
        list_values={
            input_name: _values(inner, 17, 19, 8),
            weight_name: _values(inner * outputs, 29, 23, 16),
            logits_name: [],
            result_name: [0, 0],
        },
        body=(
            clear(logits_name),
            set_var("weight start", 0),
            set_var("output", 1),
            set_var("best", -1e30),
            set_var("best id", 0),
            repeat(outputs, tuple(per_output)),
            *finish,
        ),
    )


def _insert_top_k(rank: int, logits_name: str, values_name: str, ids_name: str):
    shifts = []
    for position in range(5, rank, -1):
        shifts.extend((
            replace(values_name, position, item(values_name, position - 1)),
            replace(ids_name, position, item(ids_name, position - 1)),
        ))
    shifts.extend((
        replace(values_name, rank, var("current")),
        replace(ids_name, rank, var("index")),
    ))
    return tuple(shifts)


def _top_k_decision(rank: int, logits_name: str, values_name: str, ids_name: str):
    if rank == 5:
        return _insert_top_k(rank, logits_name, values_name, ids_name)
    return (
        if_else(
            gt(var("current"), item(values_name, rank)),
            _insert_top_k(rank, logits_name, values_name, ids_name),
            _top_k_decision(rank + 1, logits_name, values_name, ids_name),
        ),
    )


def _top_k_program(label: str, *, single_pass: bool) -> Program:
    logits_name = "topk logits"
    values_name = f"topk {label} values"
    ids_name = f"topk {label} ids"
    logits = [((index * 7919) % 10007) / 10007 + index * 1e-9 for index in range(4096)]
    initialization = (
        clear(values_name), clear(ids_name),
        *(statement for _ in range(5) for statement in (
            append(values_name, -1e30), append(ids_name, 0),
        )),
    )
    if single_pass:
        body = (
            *initialization,
            set_var("index", 1),
            repeat(4096, (
                set_var("current", item(logits_name, var("index"))),
                if_(gt(var("current"), item(values_name, 5)),
                    _top_k_decision(1, logits_name, values_name, ids_name)),
                change_var("index", 1),
            )),
        )
    else:
        body = [*initialization, set_var("rank", 1)]
        body.append(repeat(5, (
            set_var("best", -1e30),
            set_var("best id", 0),
            set_var("index", 1),
            repeat(4096, (
                if_(gt(item(logits_name, var("index")), var("best")), (
                    set_var("best", item(logits_name, var("index"))),
                    set_var("best id", var("index")),
                )),
                change_var("index", 1),
            )),
            replace(values_name, var("rank"), var("best")),
            replace(ids_name, var("rank"), var("best id")),
            replace(logits_name, var("best id"), -1e30),
            change_var("rank", 1),
        )))
        body.extend((
            set_var("rank", 1),
            repeat(5, (
                replace(
                    logits_name,
                    item(ids_name, var("rank")),
                    item(values_name, var("rank")),
                ),
                change_var("rank", 1),
            )),
        ))
        body = tuple(body)
    return Program(
        label,
        variables=("index", "current", "rank", "best", "best id"),
        lists=(logits_name, values_name, ids_name),
        list_values={logits_name: logits, values_name: [], ids_name: []},
        body=body,
    )


def _position_program(label: str, *, rope: bool) -> Program:
    output_name = f"position {label} output"
    if not rope:
        token_name, position_name = "position token", "position learned"
        add_value = append(
            output_name,
            add(item(token_name, var("index")), item(position_name, var("index"))),
        )
        body = (
            clear(output_name), set_var("index", 1),
            *_unrolled(128, (add_value, change_var("index", 1)), (add_value,)),
        )
        lists = (token_name, position_name, output_name)
        list_values = {
            token_name: _values(128, 17, 19, 8),
            position_name: _values(128, 13, 17, 16),
            output_name: [],
        }
        variables = ("index",)
    else:
        source_name, cos_name, sin_name = "position rope source", "position cos", "position sin"
        body = (
            clear(output_name),
            set_var("layer", 1),
            repeat(3, (
                set_var("index", 1),
                set_var("pair", 1),
                repeat(80, (
                    set_var("even", item(source_name, var("index"))),
                    set_var("odd", item(source_name, add(var("index"), 1))),
                    append(output_name, sub(
                        mul(var("even"), item(cos_name, var("pair"))),
                        mul(var("odd"), item(sin_name, var("pair"))),
                    )),
                    append(output_name, add(
                        mul(var("even"), item(sin_name, var("pair"))),
                        mul(var("odd"), item(cos_name, var("pair"))),
                    )),
                    change_var("index", 2),
                    change_var("pair", 1),
                )),
                change_var("layer", 1),
            )),
        )
        lists = (source_name, cos_name, sin_name, output_name)
        list_values = {
            source_name: _values(160, 17, 19, 8),
            cos_name: [math.cos(index / 80) for index in range(80)],
            sin_name: [math.sin(index / 80) for index in range(80)],
            output_name: [],
        }
        variables = ("layer", "index", "pair", "even", "odd")
    return Program(
        label,
        variables=variables,
        lists=lists,
        list_values=list_values,
        body=body,
    )


def _sprite_target(name: str):
    source = next((TEMPLATE_DIR / "sprite").glob("*.svg"))
    data = source.read_bytes()
    asset_id = hashlib.md5(data).hexdigest()
    md5ext = f"{asset_id}.svg"
    return {
        "isStage": False, "name": name,
        "variables": {}, "lists": {}, "broadcasts": {}, "blocks": {},
        "comments": {}, "currentCostume": 0,
        "costumes": [{
            "name": "cat", "bitmapResolution": 1, "dataFormat": "svg",
            "assetId": asset_id, "md5ext": md5ext,
            "rotationCenterX": 48, "rotationCenterY": 50,
        }],
        "sounds": [], "volume": 100, "visible": False,
        "x": 0, "y": 0, "size": 100, "direction": 90,
        "draggable": False, "rotationStyle": "all around", "layerOrder": 1,
    }, md5ext, data


def main():
    cases = []
    programs = []
    for inner, outputs, iterations in ((128, 384, 20), (384, 384, 8)):
        for label, indexing, preallocated in (
            ("dual_index_append", "dual", False),
            ("row_base_append", "row-base", False),
            ("row_base_replace", "row-base", True),
        ):
            procedure = f"{label}_{inner}x{outputs}"
            programs.append((procedure, _linear_program(
                procedure, inner, outputs,
                indexing=indexing, preallocated=preallocated,
            )))
            cases.append((procedure, iterations))

    for label, fused in (("head_materialize_scan", False), ("head_fused_argmax", True)):
        programs.append((label, _head_program(label, fused_argmax=fused)))
        cases.append((label, 8))
    for label, single_pass in (("topk_repeated_scan", False), ("topk_single_pass", True)):
        programs.append((label, _top_k_program(label, single_pass=single_pass)))
        cases.append((label, 30))
    for label, rope in (("position_learned_add", False), ("position_rope_3layer", True)):
        programs.append((label, _position_program(label, rope=rope)))
        cases.append((label, 1000))

    sprite, sprite_md5ext, sprite_bytes = _sprite_target("Mochi kernel screens")
    for index, (procedure, program) in enumerate(programs):
        _add_warp_procedure(sprite, procedure, program, x=640, y=index * 100)

    shared = {
        entry[0]
        for entry in sprite["lists"].values()
        if entry[0] in {
            "linear 128x384 input", "linear 128x384 weight",
            "linear 384x384 input", "linear 384x384 weight",
            "head input", "head weight", "topk logits",
        }
    }
    _merge_lists_by_name(sprite, shared)

    broadcasts = {}
    for procedure, _ in programs:
        message = f"benchmark {procedure}"
        broadcasts[message] = f"benchmark_{procedure}"
        _add_broadcast_entry(sprite, message, broadcasts[message], procedure)

    result_id = "cattorch_mochi_kernel_results"
    builder = _StageBlocks(broadcasts, {SUITE_RESULTS: result_id})
    specs = [("clear", SUITE_RESULTS)]
    for procedure, iterations in cases:
        message = f"benchmark {procedure}"
        specs.extend((
            ("broadcast", message),
            ("reset_timer",),
            ("repeat", iterations, (("broadcast", message),)),
            ("record_result", SUITE_RESULTS, f"{procedure} x{iterations}"),
        ))
    hat = builder._id()
    first, _ = builder._chain(tuple(specs), hat)
    builder.blocks[hat] = {
        "opcode": "event_whenflagclicked", "next": first, "parent": None,
        "inputs": {}, "fields": {}, "shadow": False, "topLevel": True,
        "x": 0, "y": 0,
    }
    stage, stage_md5ext, stage_bytes = _stage_target(
        builder.blocks, broadcasts, {result_id: [SUITE_RESULTS, []]},
    )
    monitor = {
        "id": result_id, "mode": "list", "opcode": "data_listcontents",
        "params": {"LIST": SUITE_RESULTS}, "spriteName": None, "value": [],
        "width": 460, "height": 240, "x": 10, "y": 10, "visible": True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    path = _write_project(
        OUTPUT, stage, [sprite], {sprite_md5ext: sprite_bytes},
        (stage_md5ext, stage_bytes), "Mochi kernel suite", monitors=[monitor],
    )
    print(f"Mochi kernel suite written: {path}")


if __name__ == "__main__":
    main()
