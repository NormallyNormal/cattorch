"""Build an isolated CatGPT-sized RoPE lowering benchmark.

This is a benchmark-only red-team screen.  It deliberately models all Q/K
heads in all six CatGPT layers per procedure call (6 * 2 * 8 rows, head
dimension 14), and compares the current dense signed-permutation matmul with
progressively more direct lowerings.  It is not imported by production code.
"""

from __future__ import annotations

import math
from pathlib import Path

from cattorch.benchmark import SUITE_RESULTS, _add_broadcast_entry, _stage_target, _write_project
from cattorch.sprite import _add_warp_procedure, _merge_lists_by_name
from cattorch.util.scratch.dsl import (
    Program,
    add,
    append,
    change_var,
    clear,
    item,
    mul,
    repeat,
    set_var,
    var,
)
from build_next_optimization_suite import (
    _HighResolutionStageBlocks,
    _sprite_target,
)


OUTPUT = Path("/tmp/cattorch_rope_redteam.sb3")
LAYERS = 6
HEADS = 8
Q_AND_K = 2
HEAD_DIM = 14
ROWS = LAYERS * HEADS * Q_AND_K
ELEMENTS = ROWS * HEAD_DIM
ITERATIONS = 160
TRIALS = 3

SOURCE = "rope source values"
COS = "rope cosine values"
SIN = "rope sine values"
ROTATION = "rope dense rotation"
START_DAYS = "rope benchmark start days"
START_DAYS_ID = "rope_benchmark_start_days"


def _unrolled(count: int, step: tuple, factor: int = 4) -> tuple:
    chunks, remainder = divmod(count, factor)
    statements = []
    if chunks:
        statements.append(repeat(chunks, step * factor))
    statements.extend(step * remainder)
    return tuple(statements)


def _inputs() -> tuple[list[float], list[float], list[float], list[int]]:
    source = [((index * 17) % 83 - 41) / 29 for index in range(ELEMENTS)]
    cosine = []
    sine = []
    for pair in range(ELEMENTS // 2):
        # Duplicate each frequency across its even/odd rotary pair, as in RoPE.
        angle = ((pair * 11) % 61 - 30) / 19
        cosine.extend((math.cos(angle), math.cos(angle)))
        sine.extend((math.sin(angle), math.sin(angle)))

    # Row-vector multiplication yields [-x_odd, x_even] for every pair.
    rotation = [0] * (HEAD_DIM * HEAD_DIM)
    for even in range(0, HEAD_DIM, 2):
        rotation[even * HEAD_DIM + even + 1] = 1
        rotation[(even + 1) * HEAD_DIM + even] = -1
    return source, cosine, sine, rotation


def _base_values(output: str, rotated: str | None = None, scaled: str | None = None):
    source, cosine, sine, rotation = _inputs()
    values = {SOURCE: source, COS: cosine, SIN: sine, ROTATION: rotation, output: []}
    if rotated is not None:
        values[rotated] = []
    if scaled is not None:
        values[scaled] = []
    return values


def _elementwise_tail(rotated: str, scaled: str, output: str) -> tuple:
    multiply_step = (
        append(scaled, mul(item(rotated, var("index")), item(SIN, var("index")))),
        change_var("index", 1),
    )
    add_step = (
        append(
            output,
            add(
                mul(item(SOURCE, var("index")), item(COS, var("index"))),
                item(scaled, var("index")),
            ),
        ),
        change_var("index", 1),
    )
    return (
        clear(scaled),
        set_var("index", 1),
        *_unrolled(ELEMENTS, multiply_step, factor=8),
        clear(output),
        set_var("index", 1),
        *_unrolled(ELEMENTS, add_step, factor=8),
    )


def _dense_program() -> Program:
    name = "rope_dense_current"
    rotated = f"{name} rotated"
    scaled = f"{name} scaled"
    output = f"{name} output"
    dot_step = (
        change_var(
            "sum",
            mul(item(SOURCE, var("left index")), item(ROTATION, var("right index"))),
        ),
        change_var("left index", 1),
        change_var("right index", HEAD_DIM),
    )
    return Program(
        name,
        variables=("sum", "left start", "left index", "right start", "right index", "index"),
        lists=(SOURCE, COS, SIN, ROTATION, rotated, scaled, output),
        list_values=_base_values(output, rotated, scaled),
        body=(
            clear(rotated),
            set_var("left start", 1),
            repeat(
                ROWS,
                (
                    set_var("right start", 1),
                    repeat(
                        HEAD_DIM,
                        (
                            set_var("sum", 0),
                            set_var("left index", var("left start")),
                            set_var("right index", var("right start")),
                            *_unrolled(HEAD_DIM, dot_step, factor=4),
                            append(rotated, var("sum")),
                            change_var("right start", 1),
                        ),
                    ),
                    change_var("left start", HEAD_DIM),
                ),
            ),
            *_elementwise_tail(rotated, scaled, output),
        ),
        optimize_loops=False,
    )


def _direct_program() -> Program:
    name = "rope_direct_rotation"
    rotated = f"{name} rotated"
    scaled = f"{name} scaled"
    output = f"{name} output"
    pair_step = (
        append(rotated, mul(-1, item(SOURCE, add(var("pair base"), 1)))),
        append(rotated, item(SOURCE, var("pair base"))),
        change_var("pair base", 2),
    )
    return Program(
        name,
        variables=("pair base", "index"),
        lists=(SOURCE, COS, SIN, ROTATION, rotated, scaled, output),
        list_values=_base_values(output, rotated, scaled),
        body=(
            clear(rotated),
            set_var("pair base", 1),
            *_unrolled(ELEMENTS // 2, pair_step, factor=4),
            *_elementwise_tail(rotated, scaled, output),
        ),
        optimize_loops=False,
    )


def _fused_program(*, cache_pair: bool) -> Program:
    suffix = "cached" if cache_pair else "list"
    name = f"rope_fused_pairwise_{suffix}"
    output = f"{name} output"

    if cache_pair:
        read_pair = (
            set_var("even", item(SOURCE, var("pair base"))),
            set_var("odd", item(SOURCE, add(var("pair base"), 1))),
        )
        even = var("even")
        odd = var("odd")
    else:
        read_pair = ()
        even = item(SOURCE, var("pair base"))
        odd = item(SOURCE, add(var("pair base"), 1))

    pair_step = (
        *read_pair,
        append(
            output,
            add(
                mul(even, item(COS, var("pair base"))),
                mul(mul(-1, odd), item(SIN, var("pair base"))),
            ),
        ),
        append(
            output,
            add(
                mul(odd, item(COS, add(var("pair base"), 1))),
                mul(even, item(SIN, add(var("pair base"), 1))),
            ),
        ),
        change_var("pair base", 2),
    )
    variables = ("pair base", "even", "odd") if cache_pair else ("pair base",)
    return Program(
        name,
        variables=variables,
        lists=(SOURCE, COS, SIN, ROTATION, output),
        list_values=_base_values(output),
        body=(
            clear(output),
            set_var("pair base", 1),
            *_unrolled(ELEMENTS // 2, pair_step, factor=4),
        ),
        optimize_loops=False,
    )


def main() -> None:
    programs = (
        _dense_program(),
        _direct_program(),
        _fused_program(cache_pair=False),
        _fused_program(cache_pair=True),
    )
    sprite, sprite_md5ext, sprite_bytes = _sprite_target("CatGPT RoPE red-team screen")
    for index, program in enumerate(programs):
        _add_warp_procedure(sprite, program.name, program, x=640, y=index * 100)
    _merge_lists_by_name(sprite, {SOURCE, COS, SIN, ROTATION})

    broadcasts = {}
    for program in programs:
        message = f"benchmark {program.name}"
        identifier = f"benchmark_{program.name}"
        broadcasts[message] = identifier
        _add_broadcast_entry(sprite, message, identifier, program.name)

    result_id = "cattorch_rope_redteam_results"
    builder = _HighResolutionStageBlocks(broadcasts, {SUITE_RESULTS: result_id})
    specs = [("clear", SUITE_RESULTS)]
    for program in programs:
        specs.append(("broadcast", f"benchmark {program.name}"))
    for trial in range(TRIALS):
        ordered = programs[trial:] + programs[:trial]
        for program in ordered:
            message = f"benchmark {program.name}"
            specs.extend((
                ("set_days", START_DAYS, START_DAYS_ID),
                ("repeat", ITERATIONS, (("broadcast", message),)),
                (
                    "record_days",
                    SUITE_RESULTS,
                    f"{program.name} trial{trial + 1} x{ITERATIONS}",
                    START_DAYS,
                    START_DAYS_ID,
                ),
            ))

    hat = builder._id()
    first, _ = builder._chain(tuple(specs), hat)
    builder.blocks[hat] = {
        "opcode": "event_whenflagclicked",
        "next": first,
        "parent": None,
        "inputs": {},
        "fields": {},
        "shadow": False,
        "topLevel": True,
        "x": 0,
        "y": 0,
    }
    stage, stage_md5ext, stage_bytes = _stage_target(
        builder.blocks,
        broadcasts,
        {result_id: [SUITE_RESULTS, []]},
        {
            START_DAYS_ID: [START_DAYS, 0],
            "rope_case_count": ["cattorch benchmark case count", len(programs)],
        },
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _write_project(
        OUTPUT,
        stage,
        [sprite],
        {sprite_md5ext: sprite_bytes},
        (stage_md5ext, stage_bytes),
        "CatGPT RoPE red-team suite",
    )
    print(f"Wrote {OUTPUT}")
    print(f"Geometry: {LAYERS} layers, Q+K, {HEADS} heads, head_dim={HEAD_DIM}")
    print(f"Each timing: {ITERATIONS} complete six-layer Q+K rotations")


if __name__ == "__main__":
    main()
