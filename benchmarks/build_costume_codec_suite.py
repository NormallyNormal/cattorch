"""Build a vanilla-Scratch screen for case-sensitive costume Base85 decoding.

Scratch's ordinary string/list comparisons are case-insensitive, which limits
the current codec to one letter case.  Costume names are looked up with strict
JavaScript equality, so an 85-costume sprite can use both cases and decode four
bytes from five JSON characters.  Switching costumes also touches the renderer;
the scratch-vm command-line result is therefore only a lower bound.  Adoption
requires the browser benchmark generated here.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from cattorch.benchmark import (
    SUITE_RESULTS,
    _add_broadcast_entry,
    _stage_target,
    _write_project,
)
from cattorch.sprite import _add_warp_procedure
from cattorch.storage import BASE85_ALPHABET, _encode_bytes
from cattorch.util.scratch.dsl import (
    Program,
    add,
    append,
    change_var,
    clear,
    costume_number,
    div,
    index_of,
    letter,
    mathop,
    mod,
    mul,
    repeat,
    set_var,
    string_length,
    sub,
    switch_costume,
    var,
)

from build_next_optimization_suite import (
    START_DAYS,
    START_DAYS_ID,
    _HighResolutionStageBlocks,
    _sprite_target,
)


OUTPUT = Path(__file__).parent / "artifacts" / "costume_codec_suite.sb3"
RAW_SIZE = 49_152  # divisible by both the Base64 and Base85 source group sizes
ITERATIONS = 5

BASE64_ALPHABET = "!#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`{|"


def _raw_bytes() -> bytes:
    # Deterministic and deliberately not very compressible.  Expanded JSON,
    # not the surrounding .sb3 ZIP, is the resource this experiment measures.
    return bytes(((index * 73 + index // 251 * 19 + 41) & 255) for index in range(RAW_SIZE))


def _encode_base85(raw: bytes) -> str:
    return _encode_bytes(raw)


def _encode_base64(raw: bytes) -> str:
    standard = base64.b64encode(raw).decode("ascii").rstrip("=")
    return standard.translate(str.maketrans(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
        BASE64_ALPHABET,
    ))


def _base64_program(raw: bytes) -> Program:
    payload = "codec base64 payload"
    alphabet = "codec base64 alphabet"
    output = "codec base64 output"

    def sextet(name: str, offset: int):
        return set_var(
            name,
            sub(index_of(alphabet, letter(var(payload), add(var("codec cursor"), offset))), 1),
        )

    byte0 = add(mul(var("codec s0"), 4), mathop("floor", div(var("codec s1"), 16)))
    byte1 = add(
        mul(mod(var("codec s1"), 16), 16),
        mathop("floor", div(var("codec s2"), 4)),
    )
    byte2 = add(mul(mod(var("codec s2"), 4), 64), var("codec s3"))
    return Program(
        "codec_base64",
        variables=(payload, "codec cursor", "codec s0", "codec s1", "codec s2", "codec s3"),
        variable_values={payload: _encode_base64(raw)},
        lists=(alphabet, output),
        list_values={alphabet: list(BASE64_ALPHABET), output: []},
        body=(
            clear(output),
            set_var("codec cursor", 1),
            repeat(
                mathop("floor", div(string_length(var(payload)), 4)),
                (
                    sextet("codec s0", 0),
                    sextet("codec s1", 1),
                    sextet("codec s2", 2),
                    sextet("codec s3", 3),
                    append(output, byte0),
                    append(output, byte1),
                    append(output, byte2),
                    change_var("codec cursor", 4),
                ),
            ),
        ),
        optimize_loops=False,
    )


def _base64_packed_program(raw: bytes) -> Program:
    payload = "codec base64 packed payload"
    alphabet = "codec base64 alphabet"
    output = "codec base64 packed output"
    digit = sub(
        index_of(alphabet, letter(var(payload), var("codec64p cursor"))),
        1,
    )
    return Program(
        "codec_base64_packed",
        variables=(payload, "codec64p cursor", "codec64p packed", "codec64p digit"),
        variable_values={payload: _encode_base64(raw)},
        lists=(alphabet, output),
        list_values={alphabet: list(BASE64_ALPHABET), output: []},
        body=(
            clear(output),
            set_var("codec64p cursor", 1),
            repeat(
                mathop("floor", div(string_length(var(payload)), 4)),
                (
                    set_var("codec64p packed", 0),
                    repeat(4, (
                        set_var("codec64p digit", digit),
                        set_var(
                            "codec64p packed",
                            add(
                                mul(var("codec64p packed"), 64),
                                var("codec64p digit"),
                            ),
                        ),
                        change_var("codec64p cursor", 1),
                    )),
                    append(
                        output,
                        mathop("floor", div(var("codec64p packed"), 65_536)),
                    ),
                    append(
                        output,
                        mod(
                            mathop("floor", div(var("codec64p packed"), 256)),
                            256,
                        ),
                    ),
                    append(output, mod(var("codec64p packed"), 256)),
                ),
            ),
        ),
        optimize_loops=False,
    )


def _base85_program(raw: bytes) -> Program:
    payload = "codec base85 payload"
    output = "codec base85 output"

    def consume_digit(offset: int):
        return (
            switch_costume(letter(var(payload), add(var("codec85 cursor"), offset))),
            set_var(
                "codec85 packed",
                add(mul(var("codec85 packed"), 85), sub(costume_number(), 1)),
            ),
        )

    body = []
    for offset in range(5):
        body.extend(consume_digit(offset))
    body.extend((
        append(output, mathop("floor", div(var("codec85 packed"), 16_777_216))),
        append(output, mod(mathop("floor", div(var("codec85 packed"), 65_536)), 256)),
        append(output, mod(mathop("floor", div(var("codec85 packed"), 256)), 256)),
        append(output, mod(var("codec85 packed"), 256)),
        change_var("codec85 cursor", 5),
    ))
    return Program(
        "codec_base85_costumes",
        variables=(payload, "codec85 cursor", "codec85 packed"),
        variable_values={payload: _encode_base85(raw)},
        lists=(output,),
        list_values={output: []},
        body=(
            clear(output),
            set_var("codec85 cursor", 1),
            repeat(
                mathop("floor", div(string_length(var(payload)), 5)),
                (set_var("codec85 packed", 0), *body),
            ),
        ),
        optimize_loops=False,
    )


def _base85_compact_program(raw: bytes) -> Program:
    payload = "codec base85 compact payload"
    output = "codec base85 compact output"
    return Program(
        "codec_base85_costumes_compact",
        variables=(payload, "codec85c cursor", "codec85c packed", "codec85c digit"),
        variable_values={payload: _encode_base85(raw)},
        lists=(output,),
        list_values={output: []},
        body=(
            clear(output),
            set_var("codec85c cursor", 1),
            repeat(
                mathop("floor", div(string_length(var(payload)), 5)),
                (
                    set_var("codec85c packed", 0),
                    repeat(5, (
                        switch_costume(letter(var(payload), var("codec85c cursor"))),
                        set_var("codec85c digit", sub(costume_number(), 1)),
                        set_var(
                            "codec85c packed",
                            add(
                                mul(var("codec85c packed"), 85),
                                var("codec85c digit"),
                            ),
                        ),
                        change_var("codec85c cursor", 1),
                    )),
                    append(
                        output,
                        mathop("floor", div(var("codec85c packed"), 16_777_216)),
                    ),
                    append(
                        output,
                        mod(
                            mathop("floor", div(var("codec85c packed"), 65_536)),
                            256,
                        ),
                    ),
                    append(
                        output,
                        mod(
                            mathop("floor", div(var("codec85c packed"), 256)),
                            256,
                        ),
                    ),
                    append(output, mod(var("codec85c packed"), 256)),
                ),
            ),
        ),
        optimize_loops=False,
    )


def _codec_costumes(sprite: dict, md5ext: str) -> None:
    source = sprite["costumes"][0]
    sprite["costumes"] = [
        {**source, "name": character, "md5ext": md5ext}
        for character in BASE85_ALPHABET
    ]
    sprite["currentCostume"] = 0


def main() -> None:
    raw = _raw_bytes()
    programs = (
        ("codec_base64", _base64_program(raw)),
        ("codec_base64_packed", _base64_packed_program(raw)),
        ("codec_base85_costumes", _base85_program(raw)),
        ("codec_base85_costumes_compact", _base85_compact_program(raw)),
    )
    sprite, sprite_md5ext, sprite_bytes = _sprite_target("cattorch costume codec screen")
    _codec_costumes(sprite, sprite_md5ext)
    for index, (name, program) in enumerate(programs):
        _add_warp_procedure(sprite, name, program, x=640, y=index * 100)

    broadcasts = {}
    for name, _ in programs:
        message = f"benchmark {name}"
        broadcasts[message] = f"benchmark_{name}"
        _add_broadcast_entry(sprite, message, broadcasts[message], name)

    result_id = "cattorch_costume_codec_results"
    builder = _HighResolutionStageBlocks(broadcasts, {SUITE_RESULTS: result_id})
    specs = [("clear", SUITE_RESULTS)]
    for name, _ in programs:
        message = f"benchmark {name}"
        specs.extend((
            ("broadcast", message),
            ("set_days", START_DAYS, START_DAYS_ID),
            ("repeat", ITERATIONS, (("broadcast", message),)),
            ("record_days", SUITE_RESULTS, f"{name} x{ITERATIONS}", START_DAYS, START_DAYS_ID),
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
            "codec_raw_size": ["codec raw byte count", RAW_SIZE],
            "codec_iterations": ["codec benchmark iterations", ITERATIONS],
        },
    )
    monitor = {
        "id": result_id, "mode": "list", "opcode": "data_listcontents",
        "params": {"LIST": SUITE_RESULTS}, "spriteName": None, "value": [],
        "width": 590, "height": 180, "x": 5, "y": 5, "visible": True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    path = _write_project(
        OUTPUT, stage, [sprite], {sprite_md5ext: sprite_bytes},
        (stage_md5ext, stage_bytes), "Costume codec suite", monitors=[monitor],
    )
    print(f"Costume codec suite written: {path}")
    print(f"Raw bytes per decode: {RAW_SIZE}; timed decodes per case: {ITERATIONS}")
    print(f"Base64 payload chars: {len(_encode_base64(raw))}")
    print(f"Base85 payload chars: {len(_encode_base85(raw))}")
    print(f"Expected decoded SHA256: {hashlib.sha256(raw).hexdigest()}")


if __name__ == "__main__":
    main()
