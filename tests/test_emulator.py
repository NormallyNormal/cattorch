"""Emulator semantics that must match the Scratch VM for verify() to be trusted."""

from __future__ import annotations

import math

import pytest

from cattorch.util.scratch.emulator import ScratchEmulator


def _literal(value):
    return [1, [10, value]] if isinstance(value, str) else [1, [4, value]]


def _reporter(opcode, inputs, fields=None):
    emulator = ScratchEmulator({"blocks": {
        "reporter": {
            "opcode": opcode, "next": None, "parent": None,
            "inputs": {name: _literal(value) for name, value in inputs.items()},
            "fields": fields or {}, "shadow": False, "topLevel": True,
        },
    }})
    return emulator._eval_reporter("reporter")


def _mathop(operator, value):
    return _reporter("operator_mathop", {"NUM": value}, {"OPERATOR": [operator, None]})


@pytest.mark.parametrize(
    ("operator", "value", "expected"),
    [
        ("ln", 0, -math.inf),
        ("log", 0, -math.inf),
        ("ln", -1, math.nan),
        ("log", -1, math.nan),
        ("sqrt", -1, math.nan),
        ("floor", math.inf, math.inf),
        # Scratch reads NaN as 0 before applying the function.
        ("floor", math.nan, 0),
        ("ceiling", -math.inf, -math.inf),
        ("10 ^", 400, math.inf),
    ],
)
def test_mathop_nonfinite_results_match_javascript(operator, value, expected):
    result = _mathop(operator, value)
    if math.isnan(expected):
        assert math.isnan(result)
    else:
        assert result == expected


def test_nan_is_zero_in_arithmetic():
    # As Scratch's Cast.toNumber: sqrt(-1) + 1 is 1 in Scratch.
    emulator = ScratchEmulator({"blocks": {
        "sum": {
            "opcode": "operator_add", "next": None, "parent": None,
            "inputs": {"NUM1": [3, "root", [4, ""]], "NUM2": _literal(1)},
            "fields": {}, "shadow": False, "topLevel": True,
        },
        "root": {
            "opcode": "operator_mathop", "next": None, "parent": "sum",
            "inputs": {"NUM": _literal(-1)}, "fields": {"OPERATOR": ["sqrt", None]},
            "shadow": False, "topLevel": False,
        },
    }})
    assert emulator._eval_reporter("sum") == 1


def test_mod_by_zero_is_nan():
    assert math.isnan(_reporter("operator_mod", {"NUM1": 5, "NUM2": 0}))


def test_empty_string_is_not_equal_to_zero():
    assert _reporter("operator_equals", {"OPERAND1": "", "OPERAND2": 0}) is False
    assert _reporter("operator_equals", {"OPERAND1": "1.0", "OPERAND2": 1}) is True
    assert _reporter("operator_equals", {"OPERAND1": "ABC", "OPERAND2": "abc"}) is True


def test_comparisons_fall_back_to_case_insensitive_strings():
    assert _reporter("operator_gt", {"OPERAND1": "", "OPERAND2": -1}) is False
    assert _reporter("operator_lt", {"OPERAND1": "apple", "OPERAND2": "Banana"}) is True
    assert _reporter("operator_gt", {"OPERAND1": 10, "OPERAND2": 9}) is True


@pytest.mark.parametrize(("text", "expected"), [
    ("0x1F", 32), ("0o7", 8), ("0b11", 4), (" 2 ", 3), (".5e1", 6), ("+1", 2),
    ("Infinity", math.inf), ("-Infinity", -math.inf),
    # Not numbers to JavaScript's Number, so 0.
    ("inf", 1), ("1_0", 1), ("-0x1", 1), ("nan", 1),
])
def test_text_is_read_as_javascript_numbers(text, expected):
    assert _reporter("operator_add", {"NUM1": text, "NUM2": 1}) == expected


@pytest.mark.parametrize(("value", "expected"), [
    ("Infinity", math.inf), ("-Infinity", -math.inf), (-2.5, -2), ("0x10", 16),
])
def test_round_matches_javascript(value, expected):
    assert _reporter("operator_round", {"NUM": value}) == expected


def test_change_by_reads_an_empty_variable_as_zero():
    emulator = ScratchEmulator({
        "variables": {"counter": ["counter", ""]},
        "blocks": {"change": {
            "opcode": "data_changevariableby", "next": None, "parent": None,
            "inputs": {"VALUE": _literal("0x2")},
            "fields": {"VARIABLE": ["counter", "counter"]},
            "shadow": False, "topLevel": True,
        }},
    })
    emulator.run()
    assert emulator.variables["counter"] == 2


@pytest.mark.parametrize(("times", "expected"), [(2.5, 3), (2.4, 2), ("", 0)])
def test_repeat_rounds_its_count(times, expected):
    emulator = ScratchEmulator({
        "variables": {"counter": ["counter", 0]},
        "blocks": {
            "loop": {
                "opcode": "control_repeat", "next": None, "parent": None,
                "inputs": {"TIMES": _literal(times), "SUBSTACK": [2, "body"]},
                "fields": {}, "shadow": False, "topLevel": True,
            },
            "body": {
                "opcode": "data_changevariableby", "next": None, "parent": "loop",
                "inputs": {"VALUE": _literal(1)},
                "fields": {"VARIABLE": ["counter", "counter"]},
                "shadow": False, "topLevel": False,
            },
        },
    })
    emulator.run()
    assert emulator.variables["counter"] == expected


def test_item_number_of_uses_scratch_comparison():
    emulator = ScratchEmulator({
        "lists": {"tokens": ["tokens", ["<unk>", " 1", "1", "A"]]},
        "blocks": {
            "reporter": {
                "opcode": "data_itemnumoflist", "next": None, "parent": None,
                "inputs": {"ITEM": _literal("1")},
                "fields": {"LIST": ["tokens", "tokens"]},
                "shadow": False, "topLevel": True,
            },
        },
    })
    assert emulator._eval_reporter("reporter") == 2
    emulator.blocks["reporter"]["inputs"]["ITEM"] = _literal("a")
    assert emulator._eval_reporter("reporter") == 4


def test_procedure_arguments_are_local_to_each_call():
    from cattorch.sprite import _add_warp_procedure
    from cattorch.util.scratch.dsl import Program, add, append, argument, call, gt, if_
    from cattorch.util.scratch.finalize_scratch import _compact_internal_display_names

    sprite = {"blocks": {}, "variables": {}, "lists": {}, "broadcasts": {}}
    # Counts down from its argument, recording it before and after the call
    # inside, which has its own.
    _add_warp_procedure(sprite, "count %s", Program("count", lists=("seen",), body=(
        append("seen", argument("n")),
        if_(gt(argument("n"), 0), (call("count %s", n=add(argument("n"), -1)),)),
        append("seen", argument("n")),
    )), x=0, y=0, arguments=("n",))
    _add_warp_procedure(sprite, "run", Program("run", lists=("seen",), body=(
        call("count %s", n=2), append("seen", argument("n")),
    )), x=0, y=200)
    _compact_internal_display_names(
        sprite, public_data_names=frozenset({"seen"}), public_procedures=frozenset({"run"}),
    )
    assert "p0 %s" in {
        block["mutation"]["proccode"] for block in sprite["blocks"].values() if "mutation" in block
    }
    emulator = ScratchEmulator(sprite)
    emulator.run_procedure("run")
    # Outside every procedure, an argument reports 0, as in Scratch.
    assert emulator.lists["seen"] == [2, 1, 0, 0, 1, 2, 0]


def test_stop_this_script_returns_from_the_procedure_running_it():
    from cattorch.sprite import _add_warp_procedure
    from cattorch.util.scratch.dsl import Program, append, call, eq, if_

    sprite = {"blocks": {}, "variables": {}, "lists": {}, "broadcasts": {}}
    _add_warp_procedure(sprite, "inner", Program("inner", lists=("seen",), body=(
        append("seen", 1), if_(eq(1, 1), (append("seen", "stop here"),)), append("seen", 2),
    )), x=0, y=0)
    _add_warp_procedure(sprite, "outer", Program("outer", lists=("seen",), body=(
        call("inner"), append("seen", 3),
    )), x=0, y=200)
    stop = next(
        block for block in sprite["blocks"].values()
        if block["opcode"] == "data_addtolist" and block["inputs"]["ITEM"][1][1] == "stop here"
    )
    stop.update(
        opcode="control_stop", inputs={}, fields={"STOP_OPTION": ["this script", None]},
        mutation={"tagName": "mutation", "children": [], "hasnext": "false"},
    )
    emulator = ScratchEmulator(sprite)
    # As in Scratch, the rest of the procedure doesn't run, and its caller
    # carries on; run directly, the procedure just ends.
    emulator.run_procedure("outer")
    assert emulator.lists["seen"] == [1, 3]
    emulator.run_procedure("inner")
    assert emulator.lists["seen"] == [1, 3, 1]
