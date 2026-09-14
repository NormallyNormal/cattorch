"""Build focused Scratch screens for implementation ideas found in MythicGPT."""

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
from cattorch.storage import (
    BASE92_ALPHABET,
    EncodedList,
    build_unpack_program,
    encode_values,
)
from cattorch.transpiler import _add_warp_procedure, _merge_lists_by_name
from cattorch.util.scratch.dsl import (
    Program,
    add,
    append,
    change_var,
    clear,
    div,
    gt,
    if_,
    if_else,
    item,
    lt,
    mathop,
    mod,
    mul,
    index_of,
    letter,
    repeat,
    replace,
    round_,
    set_var,
    sub,
    var,
)


OUTPUT = Path(__file__).parent / "artifacts" / "mythic_kernel_suite.sb3"
MYTHIC_ALPHABET = (
    "!#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`{|}~"
)
assert len(MYTHIC_ALPHABET) == 66


def _values(size: int, multiplier: int, modulus: int, divisor: float):
    return [
        ((index * multiplier) % modulus - modulus // 2) / divisor
        for index in range(size)
    ]


def _unrolled(count: int, advancing, final):
    """Four-way unroll with no dead final index mutations."""
    chunks, remainder = divmod(count, 4)
    statements = []
    if remainder:
        if chunks:
            statements.append(repeat(chunks, tuple(advancing) * 4))
        statements.extend(tuple(advancing) * (remainder - 1))
        statements.extend(final)
    else:
        if chunks > 1:
            statements.append(repeat(chunks - 1, tuple(advancing) * 4))
        statements.extend(tuple(advancing) * 3)
        statements.extend(final)
    return tuple(statements)


def _fully_unrolled_repeat(count: int, step):
    """Reduce loop dispatch while retaining every final index mutation."""
    chunks, remainder = divmod(count, 4)
    statements = []
    if chunks:
        statements.append(repeat(chunks, tuple(step) * 4))
    statements.extend(tuple(step) * remainder)
    return tuple(statements)


def _linear_names(inner: int, outputs: int):
    stem = f"linear {inner}x{outputs}"
    return f"{stem} input", f"{stem} weight"


def _linear_values(inner: int, outputs: int):
    input_values = _values(inner, 17, 37, 16)
    weight_values = _values(inner * outputs, 29, 43, 32)
    return input_values, weight_values


def _current_linear_program(
    label: str,
    inner: int,
    outputs: int,
    *,
    input_name: str,
    weight_name: str,
    input_values,
    weight_values,
    output_name: str,
    preallocated: bool,
    output_scale: float = 1.0,
) -> Program:
    product = change_var(
        "sum",
        mul(
            item(input_name, var("feature")),
            item(weight_name, add(var("weight offset"), var("feature"))),
        ),
    )
    advancing = (product, change_var("feature", 1))
    dot = _unrolled(inner, advancing, (product,))
    result = var("sum") if output_scale == 1 else mul(var("sum"), output_scale)
    write = (
        replace(output_name, var("output index"), result)
        if preallocated else append(output_name, result)
    )
    return Program(
        label,
        variables=("sum", "feature", "weight offset", "output index"),
        lists=(input_name, weight_name, output_name),
        list_values={
            input_name: input_values,
            weight_name: weight_values,
            output_name: [0] * outputs if preallocated else [],
        },
        body=(
            *((clear(output_name),) if not preallocated else ()),
            set_var("weight offset", 0),
            set_var("output index", 1),
            repeat(outputs, (
                set_var("sum", 0),
                set_var("feature", 1),
                *dot,
                write,
                change_var("weight offset", inner),
                change_var("output index", 1),
            )),
        ),
    )


def _mythic_linear_program(
    label: str,
    inner: int,
    outputs: int,
    *,
    input_name: str,
    weight_name: str,
    input_values,
    weight_values,
    output_name: str,
    unrolled: bool,
    preallocated: bool = True,
    output_scale: float = 1.0,
) -> Program:
    if outputs % 4:
        raise ValueError("Mythic four-output screen requires a multiple of four outputs")

    step = (
        set_var("input value", item(input_name, var("feature"))),
        change_var("sum a", mul(item(weight_name, var("weight a")), var("input value"))),
        change_var("weight a", 1),
        change_var("sum b", mul(item(weight_name, var("weight b")), var("input value"))),
        change_var("weight b", 1),
        change_var("sum c", mul(item(weight_name, var("weight c")), var("input value"))),
        change_var("weight c", 1),
        change_var("sum d", mul(item(weight_name, var("weight d")), var("input value"))),
        change_var("weight d", 1),
        change_var("feature", 1),
    )
    dot = _fully_unrolled_repeat(inner, step) if unrolled else (repeat(inner, step),)

    def result(name):
        return var(name) if output_scale == 1 else mul(var(name), output_scale)

    if preallocated:
        writes = (
            replace(output_name, var("output index"), result("sum a")),
            replace(output_name, add(var("output index"), 1), result("sum b")),
            replace(output_name, add(var("output index"), 2), result("sum c")),
            replace(output_name, add(var("output index"), 3), result("sum d")),
            change_var("output index", 4),
        )
        output_setup = (set_var("output index", 1),)
    else:
        writes = (
            append(output_name, result("sum a")),
            append(output_name, result("sum b")),
            append(output_name, result("sum c")),
            append(output_name, result("sum d")),
        )
        output_setup = (clear(output_name),)

    return Program(
        label,
        variables=(
            "sum a", "sum b", "sum c", "sum d", "input value", "feature",
            "weight a", "weight b", "weight c", "weight d", "output index",
        ),
        lists=(input_name, weight_name, output_name),
        list_values={
            input_name: input_values,
            weight_name: weight_values,
            output_name: [0] * outputs if preallocated else [],
        },
        body=(
            *output_setup,
            set_var("weight a", 1),
            set_var("weight b", inner + 1),
            set_var("weight c", 2 * inner + 1),
            set_var("weight d", 3 * inner + 1),
            repeat(outputs // 4, (
                set_var("sum a", 0),
                set_var("sum b", 0),
                set_var("sum c", 0),
                set_var("sum d", 0),
                set_var("feature", 1),
                *dot,
                *writes,
                change_var("weight a", 3 * inner),
                change_var("weight b", 3 * inner),
                change_var("weight c", 3 * inner),
                change_var("weight d", 3 * inner),
            )),
        ),
    )


def _softmax_scores(size: int):
    # Include a realistic spread plus values outside Mythic's -16 cutoff.
    return [
        ((index * 37) % 197) / 10 - 18 + math.sin(index * 0.7) * 0.17
        for index in range(size)
    ]


def _exact_softmax_program(label: str, size: int) -> Program:
    input_name = f"softmax {size} scores"
    output_name = f"softmax {size} exact output"
    current = item(input_name, var("index"))
    exponential = mathop("e ^", sub(current, var("maximum")))
    return Program(
        label,
        variables=("index", "maximum", "sum"),
        lists=(input_name, output_name),
        list_values={input_name: _softmax_scores(size), output_name: []},
        body=(
            clear(output_name),
            set_var("index", 1),
            set_var("maximum", current),
            change_var("index", 1),
            repeat(size - 1, (
                if_(gt(current, var("maximum")), (set_var("maximum", current),)),
                change_var("index", 1),
            )),
            set_var("index", 1),
            set_var("sum", 0),
            repeat(size, (
                change_var("sum", exponential),
                change_var("index", 1),
            )),
            set_var("index", 1),
            repeat(size, (
                append(output_name, div(exponential, var("sum"))),
                change_var("index", 1),
            )),
        ),
    )


def _mythic_softmax_program(label: str, size: int) -> Program:
    input_name = f"softmax {size} scores"
    table_name = "mythic exp table"
    output_name = f"softmax {size} mythic_lookup output"
    current = item(input_name, var("index"))
    table_index = add(round_(div(add(var("delta"), 16), 0.05)), 1)
    choose_exponential = if_else(
        lt(var("delta"), 0),
        (
            if_else(
                gt(var("delta"), -16),
                (set_var("exponential", item(table_name, table_index)),),
                (set_var("exponential", 0),),
            ),
        ),
        (set_var("exponential", 1),),
    )
    return Program(
        label,
        variables=("index", "maximum", "sum", "delta", "exponential"),
        lists=(input_name, table_name, output_name),
        list_values={
            input_name: _softmax_scores(size),
            table_name: [math.exp(-16 + index * 0.05) for index in range(321)],
            output_name: [],
        },
        body=(
            clear(output_name),
            set_var("index", 1),
            set_var("maximum", current),
            change_var("index", 1),
            repeat(size - 1, (
                if_(gt(current, var("maximum")), (set_var("maximum", current),)),
                change_var("index", 1),
            )),
            set_var("index", 1),
            set_var("sum", 0),
            repeat(size, (
                set_var("delta", sub(current, var("maximum"))),
                choose_exponential,
                append(output_name, var("exponential")),
                change_var("sum", var("exponential")),
                change_var("index", 1),
            )),
            set_var("index", 1),
            repeat(size, (
                replace(
                    output_name,
                    var("index"),
                    div(item(output_name, var("index")), var("sum")),
                ),
                change_var("index", 1),
            )),
        ),
    )


def _stored_exact_softmax_program(label: str, size: int) -> Program:
    """Exact stable softmax with Mythic's store-then-normalize structure."""
    input_name = f"softmax {size} scores"
    output_name = f"softmax {size} stored_exact output"
    current = item(input_name, var("index"))
    return Program(
        label,
        variables=("index", "maximum", "sum", "exponential"),
        lists=(input_name, output_name),
        list_values={input_name: _softmax_scores(size), output_name: []},
        body=(
            clear(output_name),
            set_var("index", 1),
            set_var("maximum", current),
            change_var("index", 1),
            repeat(size - 1, (
                if_(gt(current, var("maximum")), (set_var("maximum", current),)),
                change_var("index", 1),
            )),
            set_var("index", 1),
            set_var("sum", 0),
            repeat(size, (
                set_var("exponential", mathop("e ^", sub(current, var("maximum")))),
                append(output_name, var("exponential")),
                change_var("sum", var("exponential")),
                change_var("index", 1),
            )),
            set_var("index", 1),
            repeat(size, (
                replace(
                    output_name,
                    var("index"),
                    div(item(output_name, var("index")), var("sum")),
                ),
                change_var("index", 1),
            )),
        ),
    )


def _encode_mythic_int8(values) -> str:
    if len(values) % 3:
        raise ValueError("Mythic int8 benchmark values must be a multiple of three")
    encoded = []
    for start in range(0, len(values), 3):
        first, second, third = (int(value) + 127 for value in values[start:start + 3])
        packed = first * 255 * 255 + second * 255 + third
        digits = [0] * 4
        for index in range(3, -1, -1):
            digits[index] = packed % 66
            packed //= 66
        encoded.extend(MYTHIC_ALPHABET[digit] for digit in digits)
    return "".join(encoded)


def _mythic_decode_program(label: str, values) -> Program:
    payload_name = "mythic int8 payload"
    alphabet_name = "mythic base66 alphabet"
    output_name = "decode mythic int8 output"
    payload = _encode_mythic_int8(values)

    def digit(offset):
        return sub(
            index_of(
                alphabet_name,
                letter(var(payload_name), add(var("decode cursor"), offset)),
            ),
            1,
        )

    packed = digit(0)
    for offset in range(1, 4):
        packed = add(mul(packed, 66), digit(offset))
    return Program(
        label,
        variables=(
            payload_name, "decode cursor", "decode packed",
            "decode low", "decode middle",
        ),
        variable_values={payload_name: payload},
        lists=(alphabet_name, output_name),
        list_values={alphabet_name: list(MYTHIC_ALPHABET), output_name: []},
        body=(
            clear(output_name),
            set_var("decode cursor", 1),
            repeat(len(values) // 3, (
                set_var("decode packed", packed),
                set_var("decode low", mod(var("decode packed"), 255)),
                set_var(
                    "decode packed",
                    mathop("floor", div(var("decode packed"), 255)),
                ),
                set_var("decode middle", mod(var("decode packed"), 255)),
                append(
                    output_name,
                    sub(mathop("floor", div(var("decode packed"), 255)), 127),
                ),
                append(output_name, sub(var("decode middle"), 127)),
                append(output_name, sub(var("decode low"), 127)),
                change_var("decode cursor", 4),
            )),
        ),
    )


def _cattorch_f16_decode_program(label: str, values) -> Program:
    output_name = "decode cattorch f16 output"
    spec = EncodedList(
        name=output_name,
        payload_name="cattorch f16 benchmark payload",
        payload=encode_values(values, "float16"),
        precision="float16",
    )
    unpack = build_unpack_program([spec])
    return Program(
        label,
        variables=unpack.variables,
        variable_values=unpack.variable_values,
        lists=unpack.lists,
        list_values=unpack.list_values,
        body=(clear(output_name), *unpack.body),
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
            "name": character, "bitmapResolution": 1, "dataFormat": "svg",
            "assetId": asset_id, "md5ext": md5ext,
            "rotationCenterX": 48, "rotationCenterY": 50,
        } for character in BASE92_ALPHABET],
        "sounds": [], "volume": 100, "visible": False,
        "x": 0, "y": 0, "size": 100, "direction": 90,
        "draggable": False, "rotationStyle": "all around", "layerOrder": 1,
    }, md5ext, data


def main():
    programs: list[tuple[str, Program]] = []
    cases: list[tuple[str, int]] = []

    for inner, outputs, iterations in (
        (128, 128, 60),
        (128, 384, 20),
        (384, 128, 20),
    ):
        input_name, weight_name = _linear_names(inner, outputs)
        input_values, weight_values = _linear_values(inner, outputs)
        stem = f"linear_{inner}x{outputs}"
        variants = (
            ("current_append", "current", False, False),
            ("current_replace", "current", True, False),
            ("mythic_four", "mythic", True, False),
            ("mythic_four_unrolled", "mythic", True, True),
        )
        for suffix, kernel, preallocated, unrolled in variants:
            procedure = f"{stem}_{suffix}"
            output_name = f"linear {inner}x{outputs} {suffix} output"
            if kernel == "current":
                program = _current_linear_program(
                    procedure, inner, outputs,
                    input_name=input_name, weight_name=weight_name,
                    input_values=input_values, weight_values=weight_values,
                    output_name=output_name, preallocated=preallocated,
                )
            else:
                program = _mythic_linear_program(
                    procedure, inner, outputs,
                    input_name=input_name, weight_name=weight_name,
                    input_values=input_values, weight_values=weight_values,
                    output_name=output_name, unrolled=unrolled,
                )
            programs.append((procedure, program))
            cases.append((procedure, iterations))

    inner, outputs, iterations = 128, 384, 20
    quant_input_name = "quant linear 128x384 input"
    quant_float_weight = "quant linear 128x384 float weight"
    quant_int_weight = "quant linear 128x384 int weight"
    quant_input = _values(inner, 17, 37, 16)
    integers = [((index * 29) % 255) - 127 for index in range(inner * outputs)]
    scale = 0.00325
    floats = [value * scale for value in integers]
    quant_variants = (
        ("quant_float_current", "current", quant_float_weight, floats, 1.0, False),
        ("quant_int_current", "current", quant_int_weight, integers, scale, False),
        ("quant_int_mythic_four", "mythic", quant_int_weight, integers, scale, False),
        ("quant_int_mythic_four_unrolled", "mythic", quant_int_weight, integers, scale, True),
    )
    for procedure, kernel, weight_name, weight_values, output_scale, unrolled in quant_variants:
        output_name = f"{procedure.replace('_', ' ')} output"
        if kernel == "current":
            program = _current_linear_program(
                procedure, inner, outputs,
                input_name=quant_input_name, weight_name=weight_name,
                input_values=quant_input, weight_values=weight_values,
                output_name=output_name, preallocated=True,
                output_scale=output_scale,
            )
        else:
            program = _mythic_linear_program(
                procedure, inner, outputs,
                input_name=quant_input_name, weight_name=weight_name,
                input_values=quant_input, weight_values=weight_values,
                output_name=output_name, unrolled=unrolled,
                output_scale=output_scale,
            )
        programs.append((procedure, program))
        cases.append((procedure, iterations))

    for size, iterations in ((32, 12_000), (128, 3_000), (192, 2_000)):
        exact = f"softmax_{size}_exact"
        stored = f"softmax_{size}_stored_exact"
        lookup = f"softmax_{size}_mythic_lookup"
        programs.extend((
            (exact, _exact_softmax_program(exact, size)),
            (stored, _stored_exact_softmax_program(stored, size)),
            (lookup, _mythic_softmax_program(lookup, size)),
        ))
        cases.extend(((exact, iterations), (stored, iterations), (lookup, iterations)))

    decode_integers = [((index * 29) % 255) - 127 for index in range(49_152)]
    decode_floats = [value * 0.00325 for value in decode_integers]
    programs.extend((
        (
            "decode_cattorch_f16",
            _cattorch_f16_decode_program("decode_cattorch_f16", decode_floats),
        ),
        (
            "decode_mythic_int8",
            _mythic_decode_program("decode_mythic_int8", decode_integers),
        ),
    ))
    cases.extend((("decode_cattorch_f16", 5), ("decode_mythic_int8", 5)))

    sprite, sprite_md5ext, sprite_bytes = _sprite_target("MythicGPT kernel screens")
    for index, (procedure, program) in enumerate(programs):
        _add_warp_procedure(sprite, procedure, program, x=640, y=index * 100)

    shared_lists = {
        entry[0]
        for entry in sprite["lists"].values()
        if entry[0].endswith((" input", " weight", " scores"))
        or entry[0] == "mythic exp table"
    }
    _merge_lists_by_name(sprite, shared_lists)

    broadcasts = {}
    for procedure, _ in programs:
        message = f"benchmark {procedure}"
        broadcasts[message] = f"benchmark_{procedure}"
        _add_broadcast_entry(sprite, message, broadcasts[message], procedure)

    result_id = "cattorch_mythic_kernel_results"
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
        "width": 560, "height": 360, "x": 10, "y": 10, "visible": True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    path = _write_project(
        OUTPUT, stage, [sprite], {sprite_md5ext: sprite_bytes},
        (stage_md5ext, stage_bytes), "MythicGPT kernel suite", monitors=[monitor],
    )
    print(f"MythicGPT kernel suite written: {path}")


if __name__ == "__main__":
    main()
