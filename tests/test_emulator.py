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
        ("floor", math.nan, math.nan),
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
