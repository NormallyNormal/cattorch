"""The generic generation blocks around a toy step that has no K/V cache."""

from __future__ import annotations

import pytest

from cattorch import interface
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
    emulator.lists[interface.TOKEN_IDS] = list(tokens)
    emulator.run_procedure(interface.DECODE)
    return emulator.variables[STATUS_VARIABLE]


def prefill(emulator, tokens):
    emulator.lists[interface.TOKEN_IDS] = list(tokens)
    emulator.run_procedure(interface.PREFILL)
    return emulator.variables[STATUS_VARIABLE]


def count(emulator):
    return emulator.variables[interface.CACHE_LENGTH]


def logits(emulator):
    return emulator.lists[interface.LOGITS]


TOKEN_IDS = interface.TOKEN_IDS


@pytest.mark.parametrize("hidden_prefill", [False, True])
def test_reset_decode_and_prefill_without_a_cache(hidden_prefill):
    emulator, _ = toy_emulator(hidden_prefill=hidden_prefill)
    assert count(emulator) == 0

    assert decode(emulator, [3]) == interface.OK
    assert count(emulator) == 1
    assert logits(emulator) == [3, 0.5, -3]
    assert emulator.lists["cattorch top k ids"] == [0, 1]
    assert decode(emulator, [2]) == interface.OK
    assert count(emulator) == 2
    assert logits(emulator) == [5, 0.5, -2]

    emulator.variables[STATUS_VARIABLE] = "stale"
    emulator.run_procedure(interface.RESET)
    assert emulator.variables[STATUS_VARIABLE] == interface.OK
    assert count(emulator) == 0
    assert emulator.lists["state"] == []
    assert logits(emulator) == []
    assert emulator.lists["cattorch top k ids"] == []

    assert prefill(emulator, [1, 2, 4]) == interface.OK
    assert count(emulator) == 3
    assert emulator.lists["state"] == [7]
    assert logits(emulator) == [7, 0.5, -4]
    assert emulator.lists["cattorch top k values"] == [7, 0.5]
    # The blocks read the token list without changing it.
    assert emulator.lists[TOKEN_IDS] == [1, 2, 4]

    # A prefill starts over from a reset state.
    assert prefill(emulator, [5]) == interface.OK
    assert count(emulator) == 1
    assert logits(emulator) == [5, 0.5, -5]


def test_decode_rejects_anything_but_one_valid_token():
    emulator, _ = toy_emulator()
    assert prefill(emulator, [1, 2]) == interface.OK
    for tokens, status in (
        ([], interface.wrong_number_of_items(TOKEN_IDS)),
        ([1, 2], interface.wrong_number_of_items(TOKEN_IDS)),
        ([VOCABULARY], interface.invalid_item(TOKEN_IDS)),
        ([-1], interface.invalid_item(TOKEN_IDS)),
        ([1.5], interface.invalid_item(TOKEN_IDS)),
        (["one"], interface.invalid_item(TOKEN_IDS)),
    ):
        emulator.broadcasts.clear()
        assert decode(emulator, tokens) == status
        # A decode that fails writes nothing and still says it ended.
        assert logits(emulator) == [3, 0.5, -2]
        assert emulator.lists["cattorch top k ids"] == [0, 1]
        assert emulator.lists["state"] == [3]
        assert count(emulator) == 2
        assert emulator.broadcasts == [interface.completion(interface.DECODE)]
    assert decode(emulator, [4]) == interface.OK
    assert count(emulator) == 3


def test_prefill_rejects_empty_and_invalid_prompts():
    emulator, _ = toy_emulator()
    assert prefill(emulator, [3]) == interface.OK
    for tokens, status in (
        ([], interface.too_few_items(TOKEN_IDS)),
        ([1, 12], interface.invalid_item(TOKEN_IDS)),
    ):
        assert prefill(emulator, tokens) == status
        # It changes nothing, not even resetting first.
        assert count(emulator) == 1
        assert emulator.lists["state"] == [3]
        assert logits(emulator) == [3, 0.5, -3]


def test_a_step_that_reports_failure_is_not_counted():
    emulator, _ = toy_emulator()
    assert decode(emulator, [1]) == interface.OK
    assert decode(emulator, [FAILING_TOKEN]) == "toy failure"
    assert count(emulator) == 1
    assert logits(emulator) == [1, 0.5, -1]
    assert decode(emulator, [1]) == interface.OK
    assert count(emulator) == 2


def test_no_capacity_sets_no_token_limit():
    emulator, sprite = toy_emulator(top_k=None)
    names = {entry[0] for entry in sprite["variables"].values()}
    assert interface.MAX_CONTEXT not in names
    assert "cattorch top k ids" not in emulator.lists
    assert prefill(emulator, [1] * 300) == interface.OK
    assert count(emulator) == 300
    assert decode(emulator, [2]) == interface.OK
    assert count(emulator) == 301
    assert logits(emulator) == [302, 0.5, -2]


def test_capacity_limits_the_counted_tokens():
    emulator, sprite = toy_emulator(capacity=2)
    values = {entry[0]: entry[1] for entry in sprite["variables"].values()}
    assert values[interface.MAX_CONTEXT] == 2
    assert prefill(emulator, [1, 2, 3]) == interface.too_many_items(TOKEN_IDS)
    assert prefill(emulator, [1, 2]) == interface.OK
    assert decode(emulator, [3]) == interface.CONTEXT_FULL
    assert logits(emulator) == [3, 0.5, -2]
    assert emulator.lists["cattorch top k ids"] == [0, 1]
    assert emulator.lists["state"] == [3]
    assert count(emulator) == 2
    emulator.run_procedure(interface.RESET)
    assert decode(emulator, [3]) == interface.OK
    assert count(emulator) == 1
