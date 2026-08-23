#!/usr/bin/env python3
"""Add the calibrated TinyStories streaming sampler to a combined sprite."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from cattorch.sprite import (
    _add_warp_procedure,
    _find_procedure_definition,
    _merge_lists_by_name,
    _merge_variables_by_name,
    _procedure_mutation,
)
from cattorch.util.scratch.dsl import (
    Program,
    add,
    append,
    call,
    change_var,
    clear,
    div,
    eq,
    for_each,
    gt,
    if_,
    if_else,
    item,
    length,
    mathop,
    mul,
    random,
    repeat,
    repeat_until,
    replace,
    set_var,
    sub,
    var,
)
from cattorch.util.scratch.finalize_scratch import finalize_sprite


PROCEDURE = "tinystories sample streaming"
INITIALIZE_PROCEDURE = "tinystories sampler initialize"
ONE_TOKEN_PROCEDURE = "tinystories sampler one token"
LOAD_INPUT_PROCEDURE = "tinystories LOAD INPUT - EDIT HERE"
STREAM_OUTPUT_PROCEDURE = "tinystories STREAM OUTPUT - EDIT HERE"
TEXT_INPUT = "REPLACE ME: global story input"
TEXT_OUTPUT = "REPLACE ME: global story output"
TOP_K = 20
EOS_ID = 2
TEMPERATURE = 0.6
FREQUENCY_PENALTY = 0.3

MODEL_INPUT = "input"
MODEL_OUTPUT = "output"
TOKEN_IDS = "token_ids"
TOP_VALUES = "tinystories sampler top values"
TOP_IDS = "tinystories sampler top ids"
TOP_WEIGHTS = "tinystories sampler top weights"
GENERATED_IDS = "tinystories generated token ids"
FREQUENCIES = "tinystories generated token counts"
CACHE_LENGTH = "cattorch cache length"
MAX_CONTEXT = "cattorch max context"

INDEX = "sampler index"
CANDIDATE = "sampler candidate"
INSERT_RANK = "sampler insert rank"
SHIFT = "sampler shift"
MAXIMUM = "sampler maximum"
WEIGHT = "sampler weight"
TOTAL = "sampler total"
THRESHOLD = "sampler threshold"
CUMULATIVE = "sampler cumulative"
SELECTED = "sampler selected token"
HAS_SELECTED = "sampler has selected"
FINISHED = "sampler finished"


def _initialize_top_k():
    return (
        clear(TOP_VALUES),
        clear(TOP_IDS),
        repeat(
            TOP_K,
            (
                append(TOP_VALUES, -1e30),
                append(TOP_IDS, -1),
            ),
        ),
    )


def _consider_candidate():
    """Insert one frequency-adjusted, temperature-scaled logit into top 20."""
    return (
        set_var(
            CANDIDATE,
            div(
                sub(
                    item(MODEL_OUTPUT, var(INDEX)),
                    mul(
                        FREQUENCY_PENALTY,
                        item(FREQUENCIES, var(INDEX)),
                    ),
                ),
                TEMPERATURE,
            ),
        ),
        set_var(INSERT_RANK, 0),
        for_each(
            SHIFT,
            TOP_K,
            (
                if_(
                    eq(var(INSERT_RANK), 0),
                    (
                        if_(
                            gt(var(CANDIDATE), item(TOP_VALUES, var(SHIFT))),
                            (set_var(INSERT_RANK, var(SHIFT)),),
                        ),
                    ),
                ),
            ),
        ),
        if_(
            gt(var(INSERT_RANK), 0),
            (
                set_var(SHIFT, TOP_K + 1),
                repeat(
                    TOP_K,
                    (
                        change_var(SHIFT, -1),
                        if_(
                            gt(var(SHIFT), var(INSERT_RANK)),
                            (
                                replace(
                                    TOP_VALUES,
                                    var(SHIFT),
                                    item(TOP_VALUES, sub(var(SHIFT), 1)),
                                ),
                                replace(
                                    TOP_IDS,
                                    var(SHIFT),
                                    item(TOP_IDS, sub(var(SHIFT), 1)),
                                ),
                            ),
                        ),
                    ),
                ),
                replace(TOP_VALUES, var(INSERT_RANK), var(CANDIDATE)),
                replace(TOP_IDS, var(INSERT_RANK), sub(var(INDEX), 1)),
            ),
        ),
    )


def _sample_top_k():
    """Apply softmax to the selected logits and draw one token."""
    return (
        clear(TOP_WEIGHTS),
        set_var(MAXIMUM, item(TOP_VALUES, 1)),
        set_var(TOTAL, 0),
        for_each(
            INDEX,
            TOP_K,
            (
                set_var(
                    WEIGHT,
                    mathop(
                        "e ^",
                        sub(item(TOP_VALUES, var(INDEX)), var(MAXIMUM)),
                    ),
                ),
                append(TOP_WEIGHTS, var(WEIGHT)),
                change_var(TOTAL, var(WEIGHT)),
            ),
        ),
        # Integer random avoids Scratch's integer-vs-decimal random quirk.
        set_var(
            THRESHOLD,
            mul(div(random(0, 999_999_999), 1_000_000_000), var(TOTAL)),
        ),
        set_var(CUMULATIVE, 0),
        set_var(HAS_SELECTED, 0),
        set_var(SELECTED, item(TOP_IDS, TOP_K)),
        for_each(
            INDEX,
            TOP_K,
            (
                if_(
                    eq(var(HAS_SELECTED), 0),
                    (
                        change_var(CUMULATIVE, item(TOP_WEIGHTS, var(INDEX))),
                        if_(
                            gt(var(CUMULATIVE), var(THRESHOLD)),
                            (
                                set_var(SELECTED, item(TOP_IDS, var(INDEX))),
                                set_var(HAS_SELECTED, 1),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def _one_token():
    return (
        *_initialize_top_k(),
        for_each(INDEX, length(MODEL_OUTPUT), _consider_candidate()),
        *_sample_top_k(),
        append(TOKEN_IDS, var(SELECTED)),
        append(GENERATED_IDS, var(SELECTED)),
        replace(
            FREQUENCIES,
            add(var(SELECTED), 1),
            add(item(FREQUENCIES, add(var(SELECTED), 1)), 1),
        ),
        call("cattorch detokenize"),
        call(STREAM_OUTPUT_PROCEDURE),
        if_else(
            eq(var(SELECTED), EOS_ID),
            (set_var(FINISHED, 1),),
            (
                clear(MODEL_INPUT),
                append(MODEL_INPUT, var(SELECTED)),
                call("cattorch decode"),
            ),
        ),
    )


def _sampler_variables() -> tuple[str, ...]:
    return (
        CACHE_LENGTH,
        MAX_CONTEXT,
        INDEX,
        CANDIDATE,
        INSERT_RANK,
        SHIFT,
        MAXIMUM,
        WEIGHT,
        TOTAL,
        THRESHOLD,
        CUMULATIVE,
        SELECTED,
        HAS_SELECTED,
        FINISHED,
    )


def _sampler_lists() -> tuple[str, ...]:
    return (
        MODEL_INPUT,
        MODEL_OUTPUT,
        TOKEN_IDS,
        TOP_VALUES,
        TOP_IDS,
        TOP_WEIGHTS,
        GENERATED_IDS,
        FREQUENCIES,
    )


def build_initializer_program() -> Program:
    return Program(
        "TinyStories sampler initialization",
        variables=_sampler_variables(),
        lists=_sampler_lists(),
        body=(
            call("cattorch init"),
            call(LOAD_INPUT_PROCEDURE),
            call("cattorch tokenize"),
            clear(MODEL_INPUT),
            for_each(
                INDEX,
                length(TOKEN_IDS),
                (append(MODEL_INPUT, item(TOKEN_IDS, var(INDEX))),),
            ),
            call("cattorch prefill"),
            call("cattorch detokenize"),
            call(STREAM_OUTPUT_PROCEDURE),
            clear(GENERATED_IDS),
            clear(FREQUENCIES),
            repeat(length(MODEL_OUTPUT), (append(FREQUENCIES, 0),)),
            # Empty/invalid prompts leave no logits. Stop rather than sampling
            # the -1 placeholder IDs from an empty candidate set.
            set_var(FINISHED, eq(length(MODEL_OUTPUT), 0)),
        ),
        optimize_loops=True,
    )


def build_one_token_program() -> Program:
    return Program(
        "TinyStories calibrated one-token sampler",
        variables=_sampler_variables(),
        lists=_sampler_lists(),
        body=_one_token(),
        optimize_loops=True,
    )


def build_sampler_program() -> Program:
    return Program(
        "TinyStories streaming scheduler",
        variables=(CACHE_LENGTH, MAX_CONTEXT, FINISHED),
        body=(
            call(INITIALIZE_PROCEDURE),
            repeat_until(
                gt(
                    add(
                        var(CACHE_LENGTH),
                        mul(var(FINISHED), var(MAX_CONTEXT)),
                    ),
                    sub(var(MAX_CONTEXT), 1),
                ),
                (call(ONE_TOKEN_PROCEDURE),),
            ),
        ),
    )


def add_sampler(sprite: dict) -> dict:
    required = {
        "cattorch init",
        "cattorch tokenize",
        "cattorch detokenize",
        "cattorch prefill",
        "cattorch decode",
    }
    definitions = {
        block.get("mutation", {}).get("proccode")
        for block in sprite["blocks"].values()
        if block.get("opcode") == "procedures_prototype"
    }
    missing = required - definitions
    if missing:
        raise ValueError(f"Combined sprite is missing procedures: {sorted(missing)}")

    _add_warp_procedure(
        sprite,
        LOAD_INPUT_PROCEDURE,
        Program(
            "TinyStories editable input bridge",
            variables=("input", TEXT_INPUT),
            variable_values={TEXT_INPUT: "Once upon a time,"},
            body=(set_var("input", var(TEXT_INPUT)),),
        ),
        x=960,
        y=900,
    )
    _add_warp_procedure(
        sprite,
        STREAM_OUTPUT_PROCEDURE,
        Program(
            "TinyStories editable output bridge",
            variables=("output", TEXT_OUTPUT),
            variable_values={TEXT_OUTPUT: "Generated story streams here"},
            body=(set_var(TEXT_OUTPUT, var("output")),),
        ),
        x=960,
        y=1040,
    )
    _add_warp_procedure(
        sprite,
        INITIALIZE_PROCEDURE,
        build_initializer_program(),
        x=960,
        y=1180,
    )
    _add_warp_procedure(
        sprite,
        ONE_TOKEN_PROCEDURE,
        build_one_token_program(),
        x=960,
        y=1320,
    )
    _add_warp_procedure(
        sprite,
        PROCEDURE,
        build_sampler_program(),
        x=960,
        y=0,
    )
    _merge_variables_by_name(
        sprite,
        {
            "input",
            "output",
            TEXT_INPUT,
            TEXT_OUTPUT,
            CACHE_LENGTH,
            MAX_CONTEXT,
            INDEX,
            CANDIDATE,
            INSERT_RANK,
            SHIFT,
            MAXIMUM,
            WEIGHT,
            TOTAL,
            THRESHOLD,
            CUMULATIVE,
            SELECTED,
            HAS_SELECTED,
            FINISHED,
        },
    )
    _merge_lists_by_name(
        sprite,
        {
            MODEL_INPUT,
            MODEL_OUTPUT,
            TOKEN_IDS,
            TOP_VALUES,
            TOP_IDS,
            TOP_WEIGHTS,
            GENERATED_IDS,
            FREQUENCIES,
        },
    )

    # The outer sampler must yield between tokens so the output watcher can
    # visibly stream. Its expensive model/tokenizer callees remain warp blocks.
    definition = _find_procedure_definition(sprite, PROCEDURE)
    prototype_id = sprite["blocks"][definition]["inputs"]["custom_block"][1]
    sprite["blocks"][prototype_id]["mutation"]["warp"] = "false"

    call_id = "cattorch_tinystories_sampler_call"
    sprite["blocks"][call_id] = {
        "opcode": "procedures_call",
        "next": None,
        "parent": None,
        "inputs": {},
        "fields": {},
        "shadow": False,
        "topLevel": True,
        "x": 640,
        "y": 0,
        "mutation": _procedure_mutation(PROCEDURE),
    }
    return sprite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name", default="tinystories-500k complete")
    args = parser.parse_args()

    with zipfile.ZipFile(args.input) as archive:
        sprite = json.loads(archive.read("sprite.json"))
    sprite = add_sampler(sprite)
    print(finalize_sprite(sprite, args.output, sprite_name=args.name))


if __name__ == "__main__":
    main()
