"""Build the second-generation vanilla Scratch optimization screen.

The candidates in this file are intentionally benchmark-only.  Production
kernels should adopt them only after the generated project has been measured
on scratch.mit.edu and its output lists have been checked.
"""

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
from cattorch.sprite import _add_warp_procedure, _merge_lists_by_name
from cattorch.templates.template import TEMPLATE_DIR
from cattorch.util.scratch.dsl import (
    Program,
    add,
    append,
    change_var,
    clear,
    div,
    for_each,
    gt,
    if_,
    item,
    mathop,
    mul,
    repeat,
    replace,
    set_var,
    sub,
    var,
)


OUTPUT = Path(__file__).parent / "artifacts" / "next_optimization_suite.sb3"
SHARED_WEIGHTS = "nextopt shared weights"
SHARD_2 = "nextopt shared weights shard 2"
SHARD_3 = "nextopt shared weights shard 3"
START_DAYS = "nextopt benchmark start days"
START_DAYS_ID = "nextopt_benchmark_start_days"


def _values(size: int, multiplier: int = 17, modulus: int = 37, divisor: float = 32):
    return [((index * multiplier) % modulus - modulus // 2) / divisor for index in range(size)]


def _unrolled(count: int, step, *, factor: int = 4):
    chunks, remainder = divmod(count, factor)
    statements = []
    if chunks:
        statements.append(repeat(chunks, tuple(step) * factor))
    statements.extend(tuple(step) * remainder)
    return tuple(statements)


def _dot(
    count: int,
    *,
    left: str,
    right: str,
    left_index: str,
    right_index: str,
    accumulator: str,
    factor: int = 4,
):
    return _unrolled(
        count,
        (
            change_var(
                accumulator,
                mul(item(left, var(left_index)), item(right, var(right_index))),
            ),
            change_var(left_index, 1),
            change_var(right_index, 1),
        ),
        factor=factor,
    )


def _runtime_loop_program(name: str, variant: str, factor: int = 1) -> Program:
    data = "nextopt loop data"
    output = f"{name} output"
    size = 1024
    work = (change_var("sum", item(data, var("index"))),)
    if variant == "repeat_change":
        body = (
            set_var("sum", 0),
            set_var("index", 0),
            repeat(size, (change_var("index", 1), *work)),
        )
    elif variant == "for_each":
        body = (set_var("sum", 0), for_each("index", size, work))
    elif variant == "dynamic_repeat":
        body = (
            set_var("sum", 0),
            set_var("index", 1),
            repeat(var("loop count"), (*work, change_var("index", 1))),
        )
    elif variant == "literal_repeat":
        body = (
            set_var("sum", 0),
            set_var("index", 1),
            repeat(size, (*work, change_var("index", 1))),
        )
    elif variant == "unroll":
        step = (*work, change_var("index", 1))
        body = (
            set_var("sum", 0),
            set_var("index", 1),
            repeat(size // factor, step * factor),
        )
    else:
        raise ValueError(variant)
    return Program(
        name,
        variables=("sum", "index", "loop count"),
        variable_values={"loop count": size},
        lists=(data, output),
        list_values={data: _values(size), output: []},
        body=(*body, clear(output), append(output, var("sum"))),
        optimize_loops=False,
    )


def _generic_sharded_head(name: str) -> Program:
    source = "nextopt head input"
    output = f"{name} output"
    logical = var("weight index")
    weight = add(
        add(item(SHARED_WEIGHTS, logical), item(SHARD_2, sub(logical, 200_000))),
        item(SHARD_3, sub(logical, 400_000)),
    )
    step = (
        change_var("sum", mul(item(source, var("input index")), weight)),
        change_var("input index", 1),
        change_var("weight index", 1),
    )
    return Program(
        name,
        variables=("sum", "input index", "weight index"),
        lists=(source, SHARED_WEIGHTS, SHARD_2, SHARD_3, output),
        list_values={
            source: _values(128),
            SHARED_WEIGHTS: [1] * 200_000,
            SHARD_2: [1] * 200_000,
            SHARD_3: [1] * 124_288,
            output: [],
        },
        body=(
            clear(output),
            set_var("weight index", 1),
            repeat(4096, (
                set_var("sum", 0),
                set_var("input index", 1),
                *_unrolled(128, step),
                append(output, var("sum")),
            )),
        ),
    )


def _direct_sharded_head(name: str) -> Program:
    source = "nextopt head input"
    output = f"{name} output"

    def full_row(shard: str):
        return (
            set_var("sum", 0),
            set_var("input index", 1),
            *_dot(
                128, left=source, right=shard,
                left_index="input index", right_index="weight index",
                accumulator="sum",
            ),
            append(output, var("sum")),
        )

    return Program(
        name,
        variables=("sum", "input index", "weight index"),
        lists=(source, SHARED_WEIGHTS, SHARD_2, SHARD_3, output),
        list_values={
            source: _values(128),
            SHARED_WEIGHTS: [1] * 200_000,
            SHARD_2: [1] * 200_000,
            SHARD_3: [1] * 124_288,
            output: [],
        },
        body=(
            clear(output),
            set_var("weight index", 1),
            repeat(1562, full_row(SHARED_WEIGHTS)),
            # Logical row 1563 crosses the fixed 200,000-item boundary.
            set_var("sum", 0),
            set_var("input index", 1),
            set_var("weight index", 199_937),
            *_dot(
                64, left=source, right=SHARED_WEIGHTS,
                left_index="input index", right_index="weight index",
                accumulator="sum",
            ),
            set_var("weight index", 1),
            *_dot(
                64, left=source, right=SHARD_2,
                left_index="input index", right_index="weight index",
                accumulator="sum",
            ),
            append(output, var("sum")),
            set_var("weight index", 65),
            repeat(1562, full_row(SHARD_2)),
            set_var("weight index", 1),
            repeat(971, full_row(SHARD_3)),
        ),
    )


def _attention_values(context: int, width: int):
    query = _values(width, 11, 31, 24)
    keys = _values(context * width, 19, 43, 32)
    values = _values(context * width, 23, 47, 28)
    return query, keys, values


def _attention_program(name: str, context: int, *, fused: bool) -> Program:
    width = 32
    query_name = f"nextopt attention {context} query"
    key_name = f"nextopt attention {context} keys"
    value_name = f"nextopt attention {context} values"
    score_name = f"{name} scores"
    probability_name = f"{name} probabilities"
    output_name = f"{name} output"
    query, keys, values = _attention_values(context, width)
    scale = 1 / math.sqrt(width)
    score_dot = _dot(
        width,
        left=query_name,
        right=key_name,
        left_index="query index",
        right_index="key index",
        accumulator="sum",
    )
    current_score = item(score_name, var("score index"))

    if fused:
        score_finish = (
            set_var("score", mul(var("sum"), scale)),
            append(score_name, var("score")),
            if_(gt(var("score"), var("maximum")), (set_var("maximum", var("score")),)),
        )
        softmax = (
            set_var("score index", 1),
            set_var("denominator", 0),
            repeat(context, (
                set_var("exponential", mathop("e ^", sub(current_score, var("maximum")))),
                replace(score_name, var("score index"), var("exponential")),
                change_var("denominator", var("exponential")),
                change_var("score index", 1),
            )),
        )
        probability = item(score_name, var("probability index"))
        final_value = div(var("sum"), var("denominator"))
        lists = (query_name, key_name, value_name, score_name, output_name)
        list_values = {
            query_name: query, key_name: keys, value_name: values,
            score_name: [], output_name: [],
        }
    else:
        score_finish = (append(score_name, mul(var("sum"), scale)),)
        softmax = (
            clear(probability_name),
            set_var("score index", 1),
            set_var("maximum", current_score),
            change_var("score index", 1),
            repeat(context - 1, (
                if_(gt(current_score, var("maximum")), (set_var("maximum", current_score),)),
                change_var("score index", 1),
            )),
            set_var("score index", 1),
            set_var("denominator", 0),
            repeat(context, (
                change_var(
                    "denominator",
                    mathop("e ^", sub(current_score, var("maximum"))),
                ),
                change_var("score index", 1),
            )),
            set_var("score index", 1),
            repeat(context, (
                append(
                    probability_name,
                    div(
                        mathop("e ^", sub(current_score, var("maximum"))),
                        var("denominator"),
                    ),
                ),
                change_var("score index", 1),
            )),
        )
        probability = item(probability_name, var("probability index"))
        final_value = var("sum")
        lists = (
            query_name, key_name, value_name, score_name,
            probability_name, output_name,
        )
        list_values = {
            query_name: query, key_name: keys, value_name: values,
            score_name: [], probability_name: [], output_name: [],
        }

    return Program(
        name,
        variables=(
            "query index", "key index", "key start", "score index", "score",
            "maximum", "denominator", "exponential", "feature",
            "probability index", "value index", "sum",
        ),
        lists=lists,
        list_values=list_values,
        body=(
            clear(score_name),
            set_var("key start", 1),
            # Scratch accepts the numeric string while strict project JSON
            # cannot contain JavaScript's bare ``-Infinity`` token.
            set_var("maximum", "-Infinity"),
            repeat(context, (
                set_var("sum", 0),
                set_var("query index", 1),
                set_var("key index", var("key start")),
                *score_dot,
                *score_finish,
                change_var("key start", width),
            )),
            *softmax,
            clear(output_name),
            set_var("feature", 0),
            repeat(width, (
                set_var("sum", 0),
                set_var("probability index", 1),
                set_var("value index", add(var("feature"), 1)),
                repeat(context, (
                    change_var(
                        "sum",
                        mul(probability, item(value_name, var("value index"))),
                    ),
                    change_var("probability index", 1),
                    change_var("value index", width),
                )),
                append(output_name, final_value),
                change_var("feature", 1),
            )),
        ),
    )


def _grouped_linear_program(name: str, *, interleaved: bool) -> Program:
    source = "nextopt transformer input"
    output = f"{name} output"
    inner, columns = 128, 384
    if interleaved:
        step = (
            set_var("input value", item(source, var("input index"))),
            change_var("sum a", mul(item(SHARED_WEIGHTS, var("weight")), var("input value"))),
            change_var("sum b", mul(item(SHARED_WEIGHTS, add(var("weight"), 1)), var("input value"))),
            change_var("sum c", mul(item(SHARED_WEIGHTS, add(var("weight"), 2)), var("input value"))),
            change_var("sum d", mul(item(SHARED_WEIGHTS, add(var("weight"), 3)), var("input value"))),
            change_var("weight", 4),
            change_var("input index", 1),
        )
        indices = ("weight",)
        setup = (set_var("weight", 1),)
        finish = ()
    else:
        step = (
            set_var("input value", item(source, var("input index"))),
            change_var("sum a", mul(item(SHARED_WEIGHTS, var("weight a")), var("input value"))),
            change_var("weight a", 1),
            change_var("sum b", mul(item(SHARED_WEIGHTS, var("weight b")), var("input value"))),
            change_var("weight b", 1),
            change_var("sum c", mul(item(SHARED_WEIGHTS, var("weight c")), var("input value"))),
            change_var("weight c", 1),
            change_var("sum d", mul(item(SHARED_WEIGHTS, var("weight d")), var("input value"))),
            change_var("weight d", 1),
            change_var("input index", 1),
        )
        indices = ("weight a", "weight b", "weight c", "weight d")
        setup = (
            set_var("weight a", 1), set_var("weight b", inner + 1),
            set_var("weight c", 2 * inner + 1), set_var("weight d", 3 * inner + 1),
        )
        finish = tuple(change_var(index, 3 * inner) for index in indices)
    return Program(
        name,
        variables=(
            "sum a", "sum b", "sum c", "sum d", "input value", "input index",
            "weight", "weight a", "weight b", "weight c", "weight d",
        ),
        lists=(source, SHARED_WEIGHTS, output),
        list_values={source: _values(inner), SHARED_WEIGHTS: [1] * 200_000, output: []},
        body=(
            clear(output), *setup,
            repeat(columns // 4, (
                set_var("sum a", 0), set_var("sum b", 0),
                set_var("sum c", 0), set_var("sum d", 0),
                set_var("input index", 1),
                repeat(inner, step),
                append(output, var("sum a")), append(output, var("sum b")),
                append(output, var("sum c")), append(output, var("sum d")),
                *finish,
            )),
        ),
    )


def _rms_linear_program(name: str, *, fused: bool) -> Program:
    source = "nextopt transformer input"
    normalized = f"{name} normalized"
    output = f"{name} output"
    width, columns = 128, 384
    stats = (
        set_var("sum", 0), set_var("input index", 1),
        *_unrolled(width, (
            change_var("sum", mul(item(source, var("input index")), item(source, var("input index")))),
            change_var("input index", 1),
        )),
        set_var("rms", mathop("sqrt", add(div(var("sum"), width), 1e-5))),
    )
    if fused:
        projection_source = source
        prefix = stats
        result = div(var("sum"), var("rms"))
        lists = (source, SHARED_WEIGHTS, output)
        values = {source: _values(width), SHARED_WEIGHTS: [1] * 200_000, output: []}
    else:
        projection_source = normalized
        prefix = (
            *stats, clear(normalized), set_var("input index", 1),
            *_unrolled(width, (
                append(normalized, div(item(source, var("input index")), var("rms"))),
                change_var("input index", 1),
            )),
        )
        result = var("sum")
        lists = (source, SHARED_WEIGHTS, normalized, output)
        values = {
            source: _values(width), SHARED_WEIGHTS: [1] * 200_000,
            normalized: [], output: [],
        }
    return Program(
        name,
        variables=("sum", "input index", "weight index", "weight start", "rms"),
        lists=lists,
        list_values=values,
        body=(
            *prefix, clear(output), set_var("weight start", 1),
            repeat(columns, (
                set_var("sum", 0), set_var("input index", 1),
                set_var("weight index", var("weight start")),
                *_dot(
                    width, left=projection_source, right=SHARED_WEIGHTS,
                    left_index="input index", right_index="weight index",
                    accumulator="sum",
                ),
                append(output, result), change_var("weight start", width),
            )),
        ),
    )


def _swiglu_program(name: str, *, fused: bool) -> Program:
    source = "nextopt transformer input"
    gate = f"{name} gate"
    up = f"{name} up"
    output = f"{name} output"
    width, hidden = 128, 384
    if fused:
        body = [clear(output), set_var("weight start", 1)]
        body.append(repeat(hidden, (
            set_var("gate sum", 0), set_var("up sum", 0),
            set_var("input index", 1), set_var("weight index", var("weight start")),
            *_unrolled(width, (
                set_var("input value", item(source, var("input index"))),
                change_var("gate sum", mul(var("input value"), item(SHARED_WEIGHTS, var("weight index")))),
                change_var("up sum", mul(var("input value"), item(SHARED_WEIGHTS, var("weight index")))),
                change_var("input index", 1), change_var("weight index", 1),
            )),
            append(
                output,
                mul(
                    mul(var("gate sum"), div(1, add(1, mathop("e ^", mul(-1, var("gate sum")))))),
                    var("up sum"),
                ),
            ),
            change_var("weight start", width),
        )))
        lists = (source, SHARED_WEIGHTS, output)
        values = {source: _values(width), SHARED_WEIGHTS: [1] * 200_000, output: []}
    else:
        def projection(destination: str):
            return (
                clear(destination), set_var("weight start", 1),
                repeat(hidden, (
                    set_var("sum", 0), set_var("input index", 1),
                    set_var("weight index", var("weight start")),
                    *_dot(
                        width, left=source, right=SHARED_WEIGHTS,
                        left_index="input index", right_index="weight index",
                        accumulator="sum",
                    ),
                    append(destination, var("sum")), change_var("weight start", width),
                )),
            )
        body = [
            *projection(gate), *projection(up), clear(output), set_var("feature", 1),
            repeat(hidden, (
                set_var("gate sum", item(gate, var("feature"))),
                append(
                    output,
                    mul(
                        mul(var("gate sum"), div(1, add(1, mathop("e ^", mul(-1, var("gate sum")))))),
                        item(up, var("feature")),
                    ),
                ),
                change_var("feature", 1),
            )),
        ]
        lists = (source, SHARED_WEIGHTS, gate, up, output)
        values = {
            source: _values(width), SHARED_WEIGHTS: [1] * 200_000,
            gate: [], up: [], output: [],
        }
    return Program(
        name,
        variables=(
            "sum", "gate sum", "up sum", "input value", "input index",
            "weight index", "weight start", "feature",
        ),
        lists=lists,
        list_values=values,
        body=tuple(body),
    )


def _prefill_program(name: str, *, cache_only: bool) -> Program:
    """Synthetic final-layer workload with cache-equivalent K/V outputs."""
    source = "nextopt transformer input"
    qkv = f"{name} qkv"
    gate = f"{name} gate"
    up = f"{name} up"
    hidden = f"{name} hidden"
    final = f"{name} final"
    k_cache = f"{name} K cache"
    v_cache = f"{name} V cache"

    def projection(input_name: str, input_size: int, count: int, destination: str):
        return (
            clear(destination), set_var("weight start", 1),
            repeat(count, (
                set_var("sum", 0), set_var("input index", 1),
                set_var("weight index", var("weight start")),
                *_dot(
                    input_size, left=input_name, right=SHARED_WEIGHTS,
                    left_index="input index", right_index="weight index",
                    accumulator="sum",
                ),
                append(destination, var("sum")), change_var("weight start", input_size),
            )),
        )

    body = [clear(k_cache), clear(v_cache)]
    if cache_only:
        body.extend(projection(source, 128, 32, k_cache))
        body.extend(projection(source, 128, 32, v_cache))
        lists = (source, SHARED_WEIGHTS, k_cache, v_cache)
    else:
        body.extend(projection(source, 128, 192, qkv))
        body.extend((
            set_var("copy index", 129),
            repeat(32, (append(k_cache, item(qkv, var("copy index"))), change_var("copy index", 1))),
            repeat(32, (append(v_cache, item(qkv, var("copy index"))), change_var("copy index", 1))),
        ))
        # Work still performed by the current final layer for discarded prompt tokens.
        body.extend(projection(source, 128, 128, final))
        body.extend(projection(source, 128, 384, gate))
        body.extend(projection(source, 128, 384, up))
        body.extend((
            clear(hidden), set_var("feature", 1),
            repeat(384, (
                append(hidden, mul(item(gate, var("feature")), item(up, var("feature")))),
                change_var("feature", 1),
            )),
        ))
        body.extend(projection(hidden, 384, 128, final))
        lists = (
            source, SHARED_WEIGHTS, qkv, gate, up, hidden, final, k_cache, v_cache,
        )
    values = {name: [] for name in lists}
    values[source] = _values(128)
    values[SHARED_WEIGHTS] = [1] * 200_000
    return Program(
        name,
        variables=(
            "sum", "input index", "weight index", "weight start",
            "copy index", "feature",
        ),
        lists=lists,
        list_values=values,
        body=tuple(body),
    )


class _HighResolutionStageBlocks(_StageBlocks):
    def _days_reporter(self, parent: str):
        identifier = self._id()
        self.blocks[identifier] = self._base("sensing_dayssince2000", parent)
        return identifier

    def _command(self, spec, parent):
        kind = spec[0]
        if kind == "set_days":
            name, var_id = spec[1:]
            block_id = self._id()
            block = self._base("data_setvariableto", parent)
            block["fields"]["VARIABLE"] = [name, var_id]
            reporter = self._days_reporter(block_id)
            block["inputs"]["VALUE"] = [3, reporter, [4, 0]]
            self.blocks[block_id] = block
            return block_id
        if kind == "record_days":
            list_name, label, var_name, var_id = spec[1:]
            block_id = self._id()
            block = self._base("data_addtolist", parent)
            block["fields"]["LIST"] = [list_name, self.list_ids[list_name]]
            join_id = self._id()
            multiply_id = self._id()
            subtract_id = self._id()
            days_id = self._days_reporter(subtract_id)
            block["inputs"]["ITEM"] = [3, join_id, [10, ""]]
            self.blocks[join_id] = {
                **self._base("operator_join", block_id),
                "inputs": {
                    "STRING1": [1, [10, f"{label}: "]],
                    "STRING2": [3, multiply_id, [10, ""]],
                },
            }
            self.blocks[multiply_id] = {
                **self._base("operator_multiply", join_id),
                "inputs": {
                    "NUM1": [3, subtract_id, [4, 0]],
                    "NUM2": [1, [4, 86_400]],
                },
            }
            self.blocks[subtract_id] = {
                **self._base("operator_subtract", multiply_id),
                "inputs": {
                    "NUM1": [3, days_id, [4, 0]],
                    "NUM2": [3, [12, var_name, var_id], [10, ""]],
                },
            }
            self.blocks[block_id] = block
            return block_id
        return super()._command(spec, parent)


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


def _programs_and_cases():
    programs = []
    cases = []

    def add_case(name: str, program: Program, iterations: int):
        programs.append((name, program))
        cases.append((name, iterations))

    # These runs are deliberately long enough that small dispatch savings do
    # not disappear into browser scheduling and clock noise.
    add_case("loop_repeat_change", _runtime_loop_program("loop_repeat_change", "repeat_change"), 2000)
    add_case("loop_for_each", _runtime_loop_program("loop_for_each", "for_each"), 2000)
    add_case("loop_dynamic_repeat", _runtime_loop_program("loop_dynamic_repeat", "dynamic_repeat"), 2000)
    add_case("loop_literal_repeat", _runtime_loop_program("loop_literal_repeat", "literal_repeat"), 2000)
    for factor in (1, 2, 4, 8):
        name = f"loop_unroll_{factor}"
        add_case(name, _runtime_loop_program(name, "unroll", factor), 2000)

    add_case("head_shards_generic", _generic_sharded_head("head_shards_generic"), 3)
    add_case("head_shards_direct", _direct_sharded_head("head_shards_direct"), 3)

    for context, iterations in (
        (1, 12_000), (4, 4_800), (16, 1_400),
        (32, 720), (64, 360), (128, 180),
    ):
        for variant, fused in (("current", False), ("fused", True)):
            name = f"attention_c{context}_{variant}"
            add_case(name, _attention_program(name, context, fused=fused), iterations)

    add_case("linear_grouped_current", _grouped_linear_program("linear_grouped_current", interleaved=False), 40)
    add_case("linear_grouped_interleaved", _grouped_linear_program("linear_grouped_interleaved", interleaved=True), 40)
    add_case("rms_linear_current", _rms_linear_program("rms_linear_current", fused=False), 40)
    add_case("rms_linear_fused", _rms_linear_program("rms_linear_fused", fused=True), 40)
    add_case("swiglu_current", _swiglu_program("swiglu_current", fused=False), 24)
    add_case("swiglu_fused", _swiglu_program("swiglu_fused", fused=True), 24)
    add_case("prefill_final_current", _prefill_program("prefill_final_current", cache_only=False), 5)
    add_case("prefill_final_cache_only", _prefill_program("prefill_final_cache_only", cache_only=True), 5)
    return programs, cases


def main():
    programs, cases = _programs_and_cases()
    sprite, sprite_md5ext, sprite_bytes = _sprite_target("cattorch next optimization screens")
    for index, (name, program) in enumerate(programs):
        _add_warp_procedure(sprite, name, program, x=640, y=index * 90)

    shared = {
        entry[0]
        for entry in sprite["lists"].values()
        if entry[0].startswith("nextopt ") and not entry[0].endswith(
            (" output", " scores", " probabilities", " normalized", " gate", " up", " hidden", " final", " K cache", " V cache", " qkv")
        )
    }
    _merge_lists_by_name(sprite, shared)

    broadcasts = {}
    for name, _ in programs:
        message = f"benchmark {name}"
        broadcasts[message] = f"benchmark_{name}"
        _add_broadcast_entry(sprite, message, broadcasts[message], name)

    result_id = "cattorch_next_optimization_results"
    builder = _HighResolutionStageBlocks(broadcasts, {SUITE_RESULTS: result_id})
    specs = [("clear", SUITE_RESULTS)]
    for name, iterations in cases:
        message = f"benchmark {name}"
        specs.extend((
            ("broadcast", message),
            ("set_days", START_DAYS, START_DAYS_ID),
            ("repeat", iterations, (("broadcast", message),)),
            ("record_days", SUITE_RESULTS, f"{name} x{iterations}", START_DAYS, START_DAYS_ID),
        ))
    hat = builder._id()
    first, _ = builder._chain(tuple(specs), hat)
    builder.blocks[hat] = {
        "opcode": "event_whenflagclicked", "next": first, "parent": None,
        "inputs": {}, "fields": {}, "shadow": False, "topLevel": True,
        "x": 0, "y": 0,
    }
    stage, stage_md5ext, stage_bytes = _stage_target(
        builder.blocks,
        broadcasts,
        {result_id: [SUITE_RESULTS, []]},
        {
            START_DAYS_ID: [START_DAYS, 0],
            "nextopt_case_count": ["cattorch benchmark case count", len(cases)],
        },
    )
    monitor = {
        "id": result_id, "mode": "list", "opcode": "data_listcontents",
        "params": {"LIST": SUITE_RESULTS}, "spriteName": None, "value": [],
        "width": 590, "height": 390, "x": 5, "y": 5, "visible": True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    path = _write_project(
        OUTPUT, stage, [sprite], {sprite_md5ext: sprite_bytes},
        (stage_md5ext, stage_bytes), "Next optimization suite", monitors=[monitor],
    )
    print(f"Next optimization suite written: {path}")
    print(f"Cases: {len(cases)}")


if __name__ == "__main__":
    main()
