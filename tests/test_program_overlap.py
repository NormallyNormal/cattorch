"""Program calls from several scripts at once, interleaved as Scratch can, or stopped.

As in ``test_generation_overlap``, each script runs in its own thread, and
they switch where Scratch can pause a warp block: at a call to an internal
procedure, and here also at each repeat of a loop in the program's own
blocks, which commit and restore the states.
"""

from __future__ import annotations

from cattorch import interface
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from cattorch import (
    EntryPoint, ExportProgram, Input, Output, ProgramCall, State, StateInput, StateUpdate,
    transpile,
)
from cattorch.program_runtime import ProgramReplay
from cattorch.util.scratch.emulator import ScratchEmulator

sys.path.insert(0, str(Path(__file__).parent))
from test_generation_automatic import load_sprite  # noqa: E402
from test_generation_overlap import Script, Scripts  # noqa: E402

TOLERANCE = 1e-5
INTERRUPTED = interface.INTERRUPTED
OVERTAKEN = interface.OVERTAKEN
LIFECYCLE = ("cattorch init", "cattorch reset", "cattorch prepare for save")
LOOPS = ("control_repeat", "control_for_each", "control_repeat_until")


class Machine(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.mix = nn.Linear(3, 3)

    def step(self, delta, counter, total):
        counter = counter + self.mix(delta)
        return counter * 2, counter, total + counter.sum(), counter[None]

    def peek(self, x, counter):
        return self.mix(x) * counter

    def other(self, x):
        return torch.tanh(self.mix(x))


def three(name):
    return Input(name, torch.ones(3))


# A replace state stored with its initial values, one filled at run time
# because it holds one value, and an append state: a step updates all three.
STATEFUL = ExportProgram(
    states=(
        State("counter", torch.arange(3.0)), State("total", torch.zeros(2)),
        State("log", torch.zeros(0, 3), mode="append", capacity=8),
    ),
    entrypoints=(
        EntryPoint(
            "step", method="step",
            arguments=(three("delta"), StateInput("counter"), StateInput("total")),
            returns=(
                Output("value"), StateUpdate("counter"), StateUpdate("total"), StateUpdate("log"),
            ),
        ),
        EntryPoint(
            "peek", method="peek", arguments=(three("x"), StateInput("counter")),
            returns=(Output("y"),),
        ),
        EntryPoint("other", method="other", arguments=(three("x"),), returns=(Output("y"),)),
    ),
)
STATELESS = ExportProgram(entrypoints=(
    EntryPoint("peek", method="other", arguments=(three("x"),), returns=(Output("y"),)),
    EntryPoint("other", method="other", arguments=(three("x"),), returns=(Output("y"),)),
))


def pause_points(emulator, public):
    """Where Scratch can pause a script, and whether each is a loop.

    A script calling a public block isn't paused before it runs.
    """
    blocks = emulator.blocks
    points = {
        start: False for name, start in emulator._procedures.items()
        if name not in public and start is not None
    }
    definitions = {
        identifier: blocks[block["inputs"]["custom_block"][1]]["mutation"]["proccode"]
        for identifier, block in blocks.items() if block["opcode"] == "procedures_definition"
    }

    def procedure(identifier):
        while blocks[identifier].get("parent") is not None:
            identifier = blocks[identifier]["parent"]
        return definitions.get(identifier, "")

    for block in blocks.values():
        if block["opcode"] not in LOOPS or block["inputs"].get("SUBSTACK") is None:
            continue
        name = procedure(block["parent"])
        if name in public or name.endswith(" in epoch %s") or name.startswith(
            ("_cattorch ensure states", "_cattorch fill state"),
        ):
            points[block["inputs"]["SUBSTACK"][1]] = True
    return points


class Sprite:
    """An exported program and the results it should give."""

    def __init__(self, sprite, model, program):
        self.sprite = sprite
        self.model = model
        self.program = program
        self.entries = {f"cattorch {entry.name}": entry for entry in program.entrypoints}
        self.public = {*LIFECYCLE, *self.entries}

    def run_call(self, emulator, call):
        procedure, inputs = call
        entry = self.entries.get(procedure)
        for name, values in inputs.items():
            emulator.lists[f"{procedure} {name}"] = list(values)
        emulator.run_procedure(procedure)
        outputs = {} if entry is None else {
            binding.name: list(emulator.lists[f"{procedure} {binding.name}"])
            for binding in entry.returns if isinstance(binding, Output)
        }
        return procedure, inputs, emulator.variables["cattorch status"], outputs, self.states(
            emulator,
        )

    def states(self, emulator):
        return {
            state.name: list(emulator.lists[f"cattorch state {state.name}"])
            for state in self.program.states
        }

    def start(self, setup, prepare):
        """A new emulator with ``setup`` run, its replay and what ``prepare`` returned.

        ``prepare(emulator)`` runs first: the chains it replaces are compiled
        into the ones around them as they first run.
        """
        emulator = ScratchEmulator(self.sprite)
        prepared = prepare(emulator)
        replay = ProgramReplay(self.model, self.program)
        for call in setup:
            result = self.run_call(emulator, call)
            self.check(replay, result, ok=True)
        return emulator, replay, prepared

    def check(self, replay, result, ok=False):
        """Check one finished call against the replay of the ok calls before it."""
        procedure, inputs, status, outputs, states = result[-5:]
        if procedure in ("cattorch init", "cattorch prepare for save"):
            # They report nothing; the next call shows whether they left the
            # weights and states as they should be.
            return
        if not ok and status != "ok":
            assert status in {INTERRUPTED, OVERTAKEN}, result
            return
        assert status == "ok", result
        if procedure == "cattorch reset":
            replay.reset()
        else:
            entry = self.entries[procedure]
            expected = replay.run(ProgramCall(entry.name, tuple(
                torch.tensor(inputs[binding.name], dtype=torch.float32)
                for binding in entry.arguments if isinstance(binding, Input)
            )))
            for binding, value in zip(entry.returns, expected):
                if isinstance(binding, Output):
                    self.assert_close(outputs[binding.name], value, result)
        for name, value in replay.states.items():
            self.assert_close(states[name], value, result)

    @staticmethod
    def assert_close(actual, expected, result):
        torch.testing.assert_close(
            torch.tensor([float(value) for value in actual], dtype=torch.float64),
            expected.flatten().double(), atol=TOLERANCE, rtol=TOLERANCE, msg=lambda message: (
                f"{result[:3]}: {message}"
            ),
        )

    def interleave(self, setup, first, second, schedule):
        """Run two scripts after ``setup``, as ``schedule`` says.

        ``schedule`` is a list of ``(script index, pauses)``; None pauses runs
        the script to its end, and "stop" stops it where it is paused.
        Returns the emulator, the replay and the calls that finished, in order.

        Calls that report ok never overlap, so each one's results follow the
        ok calls that finished before it; a call that fails changes nothing,
        or leaves the states busy, so that every call fails until a reset.
        """
        finished: list[tuple] = []
        emulator, replay, scripts = self.start(setup, lambda emulator: Scripts(
            emulator, finished, self.run_call, pause_points(emulator, self.public),
        ))
        pair = (Script("first", first), Script("second", second))
        for index, pauses in schedule:
            if pauses == "stop":
                scripts.stop(pair[index])
            else:
                scripts.run(pair[index], pauses)
        for script in pair:
            scripts.run(script)
        for result in finished:
            self.check(replay, result)
        return emulator, replay, finished

    def check_afterwards(self, emulator, replay, allow_busy=True):
        """Check the next calls, then that reset recovers.

        Unless ``allow_busy``, the next calls must not be refused.
        """
        calls = [(name, {
            binding.name: [0.5, -1.0, 2.0] for binding in entry.arguments
            if isinstance(binding, Input)
        }) for name, entry in self.entries.items()]
        for call in calls:
            result = self.run_call(emulator, call)
            if result[2] == INTERRUPTED and allow_busy and self.program.states:
                before = self.states(emulator)
                assert self.run_call(emulator, call)[2:] == result[2:]
                assert self.states(emulator) == before
                break
            self.check(replay, result, ok=True)
        self.check(replay, self.run_call(emulator, ("cattorch reset", {})), ok=True)
        for call in calls:
            self.check(replay, self.run_call(emulator, call), ok=True)

    def pauses(self, setup, calls):
        """Whether each place ``calls`` can pause, run on their own after ``setup``, is a loop."""
        kinds: list[bool] = []

        def count(emulator):
            for start, loop in pause_points(emulator, self.public).items():
                emulator._chains[start] = counting(emulator, start, loop)

        def counting(emulator, start, loop):
            chain = None

            def run():
                nonlocal chain
                kinds.append(loop)
                if chain is None:
                    chain = emulator._compile_chain(start)
                chain()
            return run

        emulator, _, _ = self.start(setup, count)
        kinds.clear()
        for call in calls:
            self.run_call(emulator, call)
        return kinds


def commit_start(kinds):
    """Where a call's commit starts: its last loops, after its inputs' checks and computation."""
    return len(kinds) - next(index for index, loop in enumerate(reversed(kinds)) if not loop)


@pytest.fixture(scope="module")
def stateful(tmp_path_factory):
    model = Machine().eval()
    result = transpile(model, STATEFUL, tmp_path_factory.mktemp("stateful") / "machine")
    return Sprite(load_sprite(result.path), model, STATEFUL)


@pytest.fixture(scope="module")
def stateless(tmp_path_factory):
    model = Machine().eval()
    result = transpile(model, STATELESS, tmp_path_factory.mktemp("stateless") / "machine")
    return Sprite(load_sprite(result.path), model, STATELESS)


def step(*delta):
    return ("cattorch step", {"delta": list(delta)})


def peek(*x):
    return ("cattorch peek", {"x": list(x)})


def other(*x):
    return ("cattorch other", {"x": list(x)})


RESET = ("cattorch reset", {})
INIT = ("cattorch init", {})
PREPARE = ("cattorch prepare for save", {})
# Before the scripts run: initialized, with the states changed; or a freshly
# loaded project, whose first call decodes the weights and fills the states.
SETUPS = {"initialized": [INIT, step(1.0, 1.0, 1.0)], "fresh": []}
# A call that another script's calls start during.
OVERLAPS = {
    "step, then step": ([step(10.0, 0.0, -1.0)], [step(100.0, 2.0, 0.5), peek(1.0, 1.0, 1.0)]),
    "step, then peek": ([step(10.0, 0.0, -1.0)], [peek(1.0, 2.0, 3.0)]),
    "step, then other": ([step(10.0, 0.0, -1.0)], [other(1.0, 2.0, 3.0)]),
    "step, then reset": ([step(10.0, 0.0, -1.0)], [RESET, step(3.0, 2.0, 1.0)]),
    "step, then init": ([step(10.0, 0.0, -1.0)], [INIT, peek(1.0, 1.0, 1.0)]),
    "step, then prepare for save": ([step(10.0, 0.0, -1.0)], [PREPARE, step(3.0, 2.0, 1.0)]),
    "peek, then step": ([peek(1.0, 2.0, 3.0)], [step(100.0, 2.0, 0.5)]),
    "other, then other": ([other(1.0, 2.0, 3.0)], [other(-3.0, 0.0, 1.0)]),
    "reset, then step": ([RESET], [step(100.0, 2.0, 0.5)]),
    "reset, then reset": ([RESET], [RESET, step(3.0, 2.0, 1.0)]),
    "init, then step": ([INIT], [step(100.0, 2.0, 0.5)]),
    "prepare for save, then step": ([PREPARE], [step(100.0, 2.0, 0.5)]),
}


@pytest.mark.parametrize("setup", sorted(SETUPS))
@pytest.mark.parametrize("overlap", sorted(OVERLAPS))
def test_a_call_paused_while_another_runs_never_reports_wrong_results(stateful, overlap, setup):
    first, second = OVERLAPS[overlap]
    for pause in range(len(stateful.pauses(SETUPS[setup], first))):
        # The second script runs to its end while the first is paused, then
        # the first resumes; or the first is stopped where it was paused.
        for resume in (None, "stop"):
            emulator, replay, _ = stateful.interleave(
                SETUPS[setup], first, second, [(0, pause), (1, None), (0, resume)],
            )
            stateful.check_afterwards(emulator, replay)


@pytest.mark.parametrize("overlap", ["step, then step", "step, then reset", "reset, then reset"])
def test_a_resumed_call_that_ends_first_fails_the_call_running(stateful, overlap):
    first, second = OVERLAPS[overlap]
    setup = SETUPS["initialized"]
    taken_over = len(stateful.pauses(setup, second[:1]))
    kinds = stateful.pauses(setup, first)
    if second[0] != RESET:
        # A step is refused, rather than paused, during the first one's commit.
        kinds = kinds[:commit_start(kinds)]
    for pause in range(0, len(kinds), 2):
        for other_pause in range(0, taken_over, 2):
            # The first script resumes while the second is paused, and ends first.
            emulator, replay, finished = stateful.interleave(
                setup, first, second, [(0, pause), (1, other_pause), (0, None)],
            )
            assert [result[0] for result in finished[:2]] == ["first", "second"]
            assert [result[3] for result in finished[:2]] == [OVERTAKEN, OVERTAKEN]
            # Each call sent its completion broadcast, though it failed.
            for procedure in {result[1] for result in finished}:
                assert emulator.broadcasts.count(interface.completion(procedure)) == sum(
                    call[0] == procedure for call in (*setup, *(result[1:] for result in finished))
                )
            stateful.check_afterwards(emulator, replay)


@pytest.mark.parametrize("call", [step(10.0, 0.0, -1.0), peek(1.0, 2.0, 3.0), RESET, INIT, PREPARE])
@pytest.mark.parametrize("setup", sorted(SETUPS))
def test_a_stopped_call_changes_nothing_or_leaves_calls_failing_until_reset(
    stateful, call, setup,
):
    kinds = stateful.pauses(SETUPS[setup], [call])
    for pause, loop in enumerate(kinds):
        emulator, replay, finished = stateful.interleave(
            SETUPS[setup], [call], [], [(0, pause), (0, "stop")],
        )
        assert finished == []
        # Only a step's commit or a reset, and in a fresh project the first
        # filling of the states, leaves them busy; a call stopped before
        # changing any state doesn't.
        changing = (
            call in (RESET, INIT) or setup == "fresh"
            or call[0] == "cattorch step" and (loop or any(kinds[:pause]))
        )
        stateful.check_afterwards(emulator, replay, allow_busy=changing)


def test_a_call_refused_during_a_commit_leaves_it_to_finish(stateful):
    setup = SETUPS["initialized"]
    first, second = [step(10.0, 0.0, -1.0)], [step(100.0, 2.0, 0.5), peek(1.0, 1.0, 1.0)]
    commit = commit_start(stateful.pauses(setup, first))
    emulator, replay, finished = stateful.interleave(
        setup, first, second, [(0, commit + 1), (1, None), (0, None)],
    )
    assert [(result[0], result[3]) for result in finished] == [
        ("second", INTERRUPTED), ("second", INTERRUPTED), ("first", "ok"),
    ]
    stateful.check_afterwards(emulator, replay, allow_busy=False)


def test_a_reset_during_a_commit_leaves_calls_failing_until_the_next_reset(stateful):
    setup = SETUPS["initialized"]
    first = [step(10.0, 0.0, -1.0)]
    commit = commit_start(stateful.pauses(setup, first))
    emulator, replay, finished = stateful.interleave(
        setup, first, [RESET], [(0, commit + 1), (1, None), (0, None)],
    )
    assert [(result[0], result[3]) for result in finished] == [
        ("second", "ok"), ("first", OVERTAKEN),
    ]
    assert stateful.run_call(emulator, peek(1.0, 1.0, 1.0))[2] == INTERRUPTED
    stateful.check_afterwards(emulator, replay)


STATELESS_OVERLAPS = {
    "peek, then peek": ([("cattorch peek", {"x": [1.0, 2.0, 3.0]})],
                        [("cattorch peek", {"x": [-3.0, 0.0, 1.0]})]),
    "peek, then other": ([("cattorch peek", {"x": [1.0, 2.0, 3.0]})],
                         [("cattorch other", {"x": [-3.0, 0.0, 1.0]})]),
    "peek, then reset": ([("cattorch peek", {"x": [1.0, 2.0, 3.0]})], [RESET]),
    "reset, then peek": ([RESET], [("cattorch peek", {"x": [1.0, 2.0, 3.0]})]),
}


@pytest.mark.parametrize("overlap", sorted(STATELESS_OVERLAPS))
def test_stateless_calls_never_report_wrong_results_and_never_need_a_reset(stateless, overlap):
    first, second = STATELESS_OVERLAPS[overlap]
    for pause in range(len(stateless.pauses([], first))):
        for resume in (None, "stop"):
            emulator, replay, _ = stateless.interleave(
                [], first, second, [(0, pause), (1, None), (0, resume)],
            )
            stateless.check_afterwards(emulator, replay, allow_busy=False)
