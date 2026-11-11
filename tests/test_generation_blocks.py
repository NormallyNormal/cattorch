"""The generic generation blocks around a toy step that has no K/V cache."""

from __future__ import annotations

import pytest

from cattorch.interface import STATUS_VARIABLE
from cattorch.sprite import _add_generation_blocks, _wrap_optimized_lifecycle
from cattorch.util.scratch.dsl import (
    Program, add, append, clear, eq, if_, item, length, replace, set_var, sub,
)
from cattorch.util.scratch.emulator import ScratchEmulator

VOCABULARY = 10
FAILING_TOKEN = 9


def toy_emulator(*, capacity=None, hidden_prefill=False, top_k=2):
    """A step that sums its tokens into a state list.

    Its logits are ``[sum, 0.5, -token]``; token 9 reports a failure.
    """
    step = Program(
        "toy_step",
        variables=(STATUS_VARIABLE,),
        lists=("input", "output", "state"),
        body=(
            if_(eq(length("state"), 0), (append("state", 0),)),
            replace("state", 1, add(item("state", 1), item("input", 1))),
            clear("output"),
            append("output", item("state", 1)),
            append("output", 0.5),
            append("output", sub(0, item("input", 1))),
            if_(eq(item("input", 1), FAILING_TOKEN), (
                set_var(STATUS_VARIABLE, "toy failure"),
            )),
        ),
    )
    sprite = step.compile()
    _wrap_optimized_lifecycle(sprite)
    sprite = _add_generation_blocks(
        sprite,
        state_lists=("state",),
        capacity=capacity,
        hidden_prefill=hidden_prefill,
        top_k=top_k,
        compact_top_k=True,
        vocabulary=VOCABULARY,
    )
    emulator = ScratchEmulator(sprite)
    emulator.run_procedure("cattorch init")
    return emulator, sprite


@pytest.mark.parametrize("compact", [True, False])
def test_top_k_matches_torch_with_infinite_nan_and_tied_logits(compact):
    import math

    import torch

    from cattorch.sprite import _build_compact_top_k_program, _build_top_k_program

    build = _build_compact_top_k_program if compact else _build_top_k_program
    emulator = ScratchEmulator(build(5, logits="output").compile())
    inf = math.inf
    for logits in (
        # Fewer finite logits than top_k: -inf logits still get their IDs.
        [0.5, -inf, 2.0, -inf, 0.5, -1.0, -inf],
        # NaN ranks first, as in torch.topk.
        [1.0, math.nan, -inf, 3.0, 1.0, 0.0],
    ):
        emulator.lists["output"] = list(logits)
        emulator.run()
        # Ties keep the lower token ID first: a stable descending sort.
        expected = torch.sort(
            torch.tensor(logits, dtype=torch.float64), descending=True, stable=True,
        )
        assert emulator.lists["cattorch top k ids"] == expected.indices[:5].tolist()
        torch.testing.assert_close(
            torch.tensor(emulator.lists["cattorch top k values"], dtype=torch.float64),
            expected.values[:5], equal_nan=True,
        )


def decode(emulator, tokens):
    emulator.lists["input"] = list(tokens)
    emulator.run_procedure("cattorch decode")
    return emulator.variables[STATUS_VARIABLE]


def prefill(emulator, tokens):
    emulator.lists["input"] = list(tokens)
    emulator.run_procedure("cattorch prefill")
    return emulator.variables[STATUS_VARIABLE]


def count(emulator):
    return emulator.variables["cattorch cache length"]


@pytest.mark.parametrize("hidden_prefill", [False, True])
def test_reset_decode_and_prefill_without_a_cache(hidden_prefill):
    emulator, _ = toy_emulator(hidden_prefill=hidden_prefill)
    assert count(emulator) == 0

    assert decode(emulator, [3]) == "ok"
    assert count(emulator) == 1
    assert emulator.lists["output"] == [3, 0.5, -3]
    assert emulator.lists["cattorch top k ids"] == [0, 1]
    assert decode(emulator, [2]) == "ok"
    assert count(emulator) == 2
    assert emulator.lists["output"] == [5, 0.5, -2]

    emulator.variables[STATUS_VARIABLE] = "stale"
    emulator.run_procedure("cattorch reset cache")
    assert emulator.variables[STATUS_VARIABLE] == "ok"
    assert count(emulator) == 0
    assert emulator.lists["state"] == []
    assert emulator.lists["output"] == []
    assert emulator.lists["cattorch top k ids"] == []

    assert prefill(emulator, [1, 2, 4]) == "ok"
    assert count(emulator) == 3
    assert emulator.lists["state"] == [7]
    assert emulator.lists["output"] == [7, 0.5, -4]
    assert emulator.lists["cattorch top k values"] == [7, 0.5]

    # A prefill starts over from a reset state.
    assert prefill(emulator, [5]) == "ok"
    assert count(emulator) == 1
    assert emulator.lists["output"] == [5, 0.5, -5]


def test_decode_rejects_anything_but_one_valid_token():
    emulator, _ = toy_emulator()
    assert prefill(emulator, [1, 2]) == "ok"
    for tokens, status in (
        ([], "decode requires exactly one token"),
        ([1, 2], "decode requires exactly one token"),
        ([VOCABULARY], f"token ids must be whole numbers from 0 to {VOCABULARY - 1}"),
        ([-1], f"token ids must be whole numbers from 0 to {VOCABULARY - 1}"),
        ([1.5], f"token ids must be whole numbers from 0 to {VOCABULARY - 1}"),
    ):
        assert decode(emulator, tokens) == status
        assert emulator.lists["output"] == []
        assert emulator.lists["cattorch top k ids"] == []
        assert emulator.lists["state"] == [3]
        assert count(emulator) == 2
    assert decode(emulator, [4]) == "ok"
    assert count(emulator) == 3


def test_prefill_rejects_empty_and_invalid_prompts():
    emulator, _ = toy_emulator()
    assert prefill(emulator, []) == "prefill requires at least one token"
    assert prefill(emulator, [1, 12]) == (
        f"token ids must be whole numbers from 0 to {VOCABULARY - 1}"
    )
    assert count(emulator) == 0
    assert emulator.lists["state"] == []


def test_a_step_that_reports_failure_is_not_counted():
    emulator, _ = toy_emulator()
    assert decode(emulator, [1]) == "ok"
    assert decode(emulator, [FAILING_TOKEN]) == "toy failure"
    assert count(emulator) == 1
    assert decode(emulator, [1]) == "ok"
    assert count(emulator) == 2


def test_no_capacity_sets_no_token_limit():
    emulator, sprite = toy_emulator(top_k=None)
    names = {entry[0] for entry in sprite["variables"].values()}
    assert "cattorch max context" not in names
    assert "cattorch top k ids" not in emulator.lists
    assert prefill(emulator, [1] * 300) == "ok"
    assert count(emulator) == 300
    assert decode(emulator, [2]) == "ok"
    assert count(emulator) == 301
    assert emulator.lists["output"] == [302, 0.5, -2]


def test_capacity_limits_the_counted_tokens():
    emulator, sprite = toy_emulator(capacity=2)
    values = {entry[0]: entry[1] for entry in sprite["variables"].values()}
    assert values["cattorch max context"] == 2
    assert prefill(emulator, [1, 2, 3]) == "prefill exceeds maximum context"
    assert prefill(emulator, [1, 2]) == "ok"
    assert decode(emulator, [3]) == "maximum context exceeded"
    assert emulator.lists["output"] == []
    assert emulator.lists["cattorch top k ids"] == []
    assert emulator.lists["state"] == [3]
    assert count(emulator) == 2
    emulator.run_procedure("cattorch reset cache")
    assert decode(emulator, [3]) == "ok"
    assert count(emulator) == 1
