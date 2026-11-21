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


# What scratch-vm (V8) gives; test_emulator_math_vm checks many more in the VM.
@pytest.mark.parametrize(("opcode", "operator", "value", "expected"), [
    # Math.round rounds once: floor(x + 0.5) would round these up.
    ("operator_round", None, 0.49999999999999994, 0),
    ("operator_round", None, 4503599627370497, 4503599627370497),
    # V8's fdlibm differs from the C library in the last bit.
    ("operator_mathop", "ln", 3, 1.0986122886681096),
    ("operator_mathop", "log", 3, 0.4771212547196623),
    ("operator_mathop", "e ^", 270, 1.8176493851391001e117),
    ("operator_mathop", "10 ^", 2.5, 316.2277660168379),
    ("operator_mathop", "10 ^", -4, 0.00009999999999999999),
    ("operator_mathop", "atan", 0.8756150886325909, 41.20587914346107),
    ("operator_mathop", "sin", 1e308, math.nan),
    ("operator_mathop", "cos", -1e308, math.nan),
])
def test_math_matches_scratch_vm_to_the_last_bit(opcode, operator, value, expected):
    fields = {"OPERATOR": [operator, None]} if operator else None
    result = _reporter(opcode, {"NUM": value}, fields)
    assert result == expected or (math.isnan(expected) and math.isnan(result))


@pytest.mark.parametrize(("opcode", "operator", "value"), [
    ("operator_round", None, -0.4),
    ("operator_mathop", "ceiling", -0.5),
    ("operator_mathop", "floor", "-0"),
    # toFixed(10) keeps the sign of a negative value that rounds to 0.
    ("operator_mathop", "sin", -1e-12),
])
def test_math_keeps_negative_zero_as_javascript_does(opcode, operator, value):
    fields = {"OPERATOR": [operator, None]} if operator else None
    result = _reporter(opcode, {"NUM": value}, fields)
    assert result == 0 and math.copysign(1, result) == -1


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


@pytest.mark.parametrize(("times", "expected"), [
    (2.5, 3), (2.4, 2), ("", 0), (0.49999999999999994, 0),
])
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
    from cattorch.procedures import _add_warp_procedure
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
    assert "_p0 %s" in {
        block["mutation"]["proccode"] for block in sprite["blocks"].values() if "mutation" in block
    }
    emulator = ScratchEmulator(sprite)
    emulator.run_procedure("run")
    # Outside every procedure, an argument reports 0, as in Scratch.
    assert emulator.lists["seen"] == [2, 1, 0, 0, 1, 2, 0]


def test_stop_this_script_returns_from_the_procedure_running_it():
    from cattorch.procedures import _add_warp_procedure
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


# Expected values below are scratch-vm's, from
# test_real_vm.py::test_the_emulator_casts_compares_and_formats_values_as_scratch_vm_does.

@pytest.mark.parametrize(("text", "expected"), [
    # JavaScript's Number reads ASCII digits only, and trims its own whitespace.
    ("１", 1), ("٣", 1), ("﻿3", 4), ("　4", 5), ("\x1c5", 1), ("\x855", 1),
    ("0x" + "f" * 300, math.inf),
])
def test_number_casts_follow_javascript(text, expected):
    assert _reporter("operator_add", {"NUM1": text, "NUM2": 1}) == expected


@pytest.mark.parametrize(("left", "right", "equal"), [
    ("1", "１", False), ("+5", "\x1c5", False), ("3", "﻿3", True), ("İ", "i̇", True),
    ("ΑΣ", "ας", True), ("K", "k", True), ("ß", "SS", False),
])
def test_equality_follows_scratch(left, right, equal):
    assert _reporter("operator_equals", {"OPERAND1": left, "OPERAND2": right}) is equal


def test_text_orders_by_utf16_code_units():
    # U+FFFD is one unit above the high surrogate that starts an emoji.
    assert _reporter("operator_gt", {"OPERAND1": "�", "OPERAND2": "😀"}) is True
    assert _reporter("operator_gt", {"OPERAND1": "a😀b", "OPERAND2": "abc"}) is True


@pytest.mark.parametrize(("value", "text"), [
    (0.5, "0.5"), (1e-7, "1e-7"), (1.5e-7, "1.5e-7"), (1e16, "10000000000000000"),
    (1e-5, "0.00001"), (1e21, "1e+21"), (12345678901234567890, "12345678901234567000"),
    ("1.0", "1.0"),
])
def test_numbers_join_as_javascript_writes_them(value, text):
    assert _reporter("operator_join", {"STRING1": "", "STRING2": value}) == text


def test_letters_and_lengths_count_utf16_units():
    assert _reporter("operator_length", {"STRING": "a😀b"}) == 4
    assert _reporter("operator_letter_of", {"LETTER": 2, "STRING": "😀"}) == "\ude00"
    assert _reporter("operator_letter_of", {"LETTER": "", "STRING": "abc"}) == ""
    assert _reporter("operator_letter_of", {"LETTER": 1.5, "STRING": "abc"}) == "a"


@pytest.mark.parametrize(("value", "expected"), [
    ("0", False), ("false", False), ("False", False), ("", False), ("true", True), ("x", True),
])
def test_not_reads_values_as_scratch_booleans(value, expected):
    assert _reporter("operator_not", {"OPERAND": value}) is not expected


@pytest.mark.parametrize(("left", "right", "expected"), [
    (-5, math.inf, -5), (5, -math.inf, 5), (math.inf, 3, math.nan), (-7, 3, 2),
    (7.5, -2, -0.5),
])
def test_mod_follows_scratch_with_infinities(left, right, expected):
    result = _reporter("operator_mod", {"NUM1": left, "NUM2": right})
    assert (math.isnan(result) and math.isnan(expected)) or result == expected


def _list_emulator(blocks, values):
    return ScratchEmulator({
        "blocks": blocks, "lists": {"list": ["letters", list(values)]},
        "costumes": [{"name": name} for name in ("a", "3", "Five", "x y", "b")],
    })


@pytest.mark.parametrize(("index", "expected"), [
    ("last", "e"), ("all", ""), (" 2 ", "b"), ("0x2", "b"), ("Infinity", ""), (2.9, "b"),
    ("0b101", "e"), ("１", ""),
])
def test_list_indices_follow_cast_to_list_index(index, expected):
    emulator = _list_emulator({
        "reporter": {
            "opcode": "data_itemoflist", "next": None, "parent": None,
            "inputs": {"INDEX": _literal(index)}, "fields": {"LIST": ["letters", "list"]},
            "shadow": False, "topLevel": True,
        },
    }, "abcde")
    assert emulator._eval_reporter("reporter") == expected


@pytest.mark.parametrize(("requested", "costume"), [
    ("3", 2), ("2", 2), ("0", 5), ("-1", 4), ("1.5", 2), ("0x2", 2), ("next costume", 1),
    ("previous costume", 4), ("five", 5), ("", 5), (3.0, 3), ("１", 5),
    # MathUtil.wrapClamp wraps in doubles; Math.round doesn't round 2**52 + 1 up.
    ("4503599627370498", 3), ("1e308", 1),
])
def test_switch_costume_follows_scratch_vm(requested, costume):
    emulator = _list_emulator({}, "")
    emulator._current_costume = 4
    emulator._switch_costume(requested)
    assert emulator._current_costume + 1 == costume


def test_for_each_reads_its_count_before_every_iteration():
    from cattorch.util.scratch.dsl import Program, append, for_each, if_else, length, lt

    program = Program("grow", lists=("grow",), variables=("k",), list_values={"grow": [1, 2]},
                      body=(for_each("k", length("grow"), (
                          if_else(lt(length("grow"), 5), (append("grow", 0),), ()),
                      )),))
    emulator = ScratchEmulator(program.compile())
    emulator.run()
    assert len(emulator.lists["grow"]) == 5
    assert emulator.variables["k"] == 5
