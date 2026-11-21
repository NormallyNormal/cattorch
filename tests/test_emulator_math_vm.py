"""The emulator's arithmetic against ``@scratch/scratch-vm``, bit for bit.

``verify`` and the rest of the suite trust the emulator's numbers, so each
``mathop``, ``round``, repeat count and costume number is computed in
scratch-vm and compared exactly, ``-0`` included (as ``1 / x``). Needs Node.js
and ``npm ci`` in ``benchmarks/``; skips without them unless
``CATTORCH_REQUIRE_VM`` is set, as it is in CI.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from cattorch.util.scratch.emulator import ProjectEmulator

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "benchmarks" / "run_project_vm.cjs"
MODULES = ROOT / "benchmarks" / "node_modules"

pytestmark = pytest.mark.vm

OPERATORS = (
    "abs", "floor", "ceiling", "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "ln",
    "log", "e ^", "10 ^",
)


@pytest.fixture(scope="module", autouse=True)
def scratch_vm():
    if shutil.which("node") is None or not (MODULES / "@scratch" / "scratch-vm").is_dir():
        message = "needs Node.js and `npm ci` in benchmarks/"
        if os.environ.get("CATTORCH_REQUIRE_VM"):
            pytest.fail(message)
        pytest.skip(message)


def _values() -> list:
    """Values each operation reads: edge cases, then random doubles of every size."""
    values: list = [
        0, "-0", 1, -1, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 0.49999999999999994,
        -0.49999999999999994, 4503599627370497, -4503599627370497, 9007199254740993, 0.975,
        -0.4, 2**-27, 2**-29, 5e-324, 1e-310, 1e-9, 3, 90, -90, 180, 270, -270, 360, 45,
        1e16, 1e21, 1e300, -1e300, 1e308, 1.7976931348623157e308, 709.78, 709.79, -745.1,
        -745.2, 308, 308.25, -323.3, -4, 242, 1023.9, 1024, 2.4375, 1.1875, 0.4375, 0.6744,
        0.78125, 0.3, 123.456, -0.0000001, "Infinity", "-Infinity", "", "abc", " 3 ", "0x10",
        True,
    ]
    rng = random.Random(4)
    for _ in range(160):
        kind = rng.random()
        if kind < 0.3:
            value = rng.uniform(-10, 10)
        elif kind < 0.5:
            value = rng.uniform(-1, 1)
        elif kind < 0.7:
            value = rng.uniform(-750, 750)
        elif kind < 0.85:
            value = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
        else:
            value = rng.uniform(-1e7, 1e7)
        if math.isfinite(value):
            values.append(value)
    return values


def _same(vm_value, emulated) -> bool:
    """Equal as the VM's JSON shows them: numbers as doubles, strings exactly."""
    def normal(value):
        if isinstance(value, float) and not math.isfinite(value):
            return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        return value

    vm_value, emulated = normal(vm_value), normal(emulated)
    if isinstance(vm_value, str) or isinstance(emulated, str):
        return vm_value == emulated
    return float(vm_value) == float(emulated)


def test_mathop_round_and_counts_match_scratch_vm_exactly(tmp_path):
    from cattorch.project import _Builder
    from cattorch.util.scratch.dsl import (
        append, change_var, clear, costume_number, div, for_each, gt, if_, item, length, lt,
        mathop, repeat, round_, set_var, switch_costume, var,
    )

    values = _values()
    builder = _Builder()
    sprite = builder.add_demo_sprite()
    sprite["name"] = "Probe"
    costume = sprite["costumes"][0]
    sprite["costumes"] = [dict(copy.deepcopy(costume), name=f"c{index}") for index in range(5)]
    inputs = builder.global_list("values", values)
    names = [*OPERATORS, "round", "repeat", "costume"]
    out = {name: builder.global_list(f"out {name}") for name in names}
    # 1 / x tells -0 from 0, which JSON can't.
    signs = {name: builder.global_list(f"sign {name}") for name in names[:-2]}
    value = item(inputs, var("i"))
    body = [clear(target) for target in (*out.values(), *signs.values())]
    body.append(for_each("i", length(inputs), (
        *(statement for operator in OPERATORS for statement in (
            append(out[operator], mathop(operator, value)),
            append(signs[operator], div(1, mathop(operator, value))),
        )),
        append(out["round"], round_(value)),
        append(signs["round"], div(1, round_(value))),
        set_var("count", 0),
        # Only counts Scratch can finish.
        if_(lt(value, 1000), (if_(gt(value, -1), (
            repeat(value, (change_var("count", 1),)),
        )),)),
        append(out["repeat"], var("count")),
        switch_costume("c0"),
        switch_costume(value),
        append(out["costume"], costume_number()),
    )))
    builder.script(
        sprite, {"opcode": "event_whenflagclicked", "fields": {}}, tuple(body),
        lists=(inputs, *out.values(), *signs.values()), variables=("i", "count"),
    )
    path = tmp_path / "math.sb3"
    builder.write(path)
    output = tmp_path / "vm.json"
    completed = subprocess.run(
        ["node", str(RUNNER), str(path), f"--out={output}"],
        capture_output=True, text=True, env={**os.environ, "NODE_PATH": str(MODULES)},
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    stage = json.loads(output.read_text())["stage"]
    emulator = ProjectEmulator(path)
    emulator.green_flag()
    differences = []
    for name, list_name in [*out.items(), *signs.items()]:
        expected, actual = stage[list_name], emulator.stage.lists[list_name]
        assert len(expected) == len(actual) == len(values), name
        differences += [
            (list_name, values[index], wanted, got)
            for index, (wanted, got) in enumerate(zip(expected, actual))
            if not _same(wanted, got)
        ]
    assert not differences, differences[:20]

