"""Generation calls from several scripts at once, interleaved as Scratch can.

Scratch pauses a warp block that has run for 500 ms where it calls a
procedure or repeats a loop, and runs other scripts meanwhile. These tests
run each script in its own thread and switch between them where a
procedure is called, then check that no call reports ``ok`` with wrong
logits and that a later reset or prefill always recovers.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest
import torch

from cattorch import CodegenConfig, GenerationProgram, RecurrentGenerationProgram, transpile
from cattorch.util.scratch.emulator import ScratchEmulator

sys.path.insert(0, str(Path(__file__).parent))
from test_generation_automatic import INTERRUPTED, load_sprite, logits_of  # noqa: E402
from test_optimization import CachedLM  # noqa: E402
from test_recurrent_generation import make, reference_rows  # noqa: E402

TOLERANCE = 1e-4
# Pause places tried per case, spread over the call.
SAMPLES = 8
OVERTAKEN = "another reset, prefill, decode or init ran during this one; reset or prefill first"
SETUP = [1, 2, 3, 4]


def spread(count, samples=SAMPLES):
    """Up to ``samples`` of ``range(count)``, first and last included."""
    if count <= samples:
        return list(range(count))
    return sorted({round(index * (count - 1) / (samples - 1)) for index in range(samples)})


class Stopped(Exception):
    """A script stopped where it was paused."""


PUBLIC = ("cattorch init", "cattorch reset", "cattorch prefill", "cattorch decode")


def paused_procedures(emulator):
    """Where Scratch can pause a script: calls inside the warp blocks.

    A script calling a public block isn't paused before it runs.
    """
    return {
        start for name, start in emulator._procedures.items() if name not in PUBLIC
    }


def run_generation_call(emulator, call):
    """Run one ``(procedure, tokens)`` call and return what it left."""
    procedure, tokens = call
    if tokens is not None:
        emulator.lists["cattorch tokens"] = list(tokens)
    emulator.run_procedure(procedure)
    return (
        procedure, tokens, emulator.variables["cattorch status"],
        list(emulator.lists["cattorch logits"]), list(emulator.lists["cattorch top k ids"]),
        emulator.variables["cattorch cache length"],
    )


class Script:
    """A Scratch script making calls, by default generation calls ``(procedure, tokens)``."""

    def __init__(self, name, calls):
        self.name = name
        self.calls = calls
        self.results: list[tuple] = []
        self.pauses = 0
        self.budget = 0
        self.stop = False
        self.done = False
        self.error: BaseException | None = None
        self.thread: threading.Thread | None = None


class Scripts:
    """Run scripts on one sprite, switching between them where a procedure is called.

    ``run_call(emulator, call)`` runs each call and returns what to record of
    it; the chains starting at ``pauses`` are where scripts can pause.
    """

    def __init__(self, emulator, finished, run_call=run_generation_call, pauses=None):
        self.emulator = emulator
        self.finished = finished
        self.run_call = run_call
        self.condition = threading.Condition()
        self.running: Script | None = None
        self.local = threading.local()
        for start in paused_procedures(emulator) if pauses is None else pauses:
            emulator._chains[start] = self._pausing(start)

    def _pausing(self, start):
        # Compiled when it first runs, so that it runs the pausing chains
        # nested in it.
        chain = None

        def run():
            nonlocal chain
            script = getattr(self.local, "script", None)
            if script is not None:
                script.pauses += 1
                if script.pauses > script.budget:
                    self._pause(script)
            if chain is None:
                chain = self.emulator._compile_chain(start)
            chain()
        return run

    def _pause(self, script):
        # Each script has its own procedure arguments in Scratch.
        frames = self.emulator._arguments
        saved = frames[:]
        frames.clear()
        with self.condition:
            self.running = None
            self.condition.notify_all()
            self.condition.wait_for(lambda: self.running is script)
        frames[:] = saved
        if script.stop:
            raise Stopped

    def _body(self, script):
        self.local.script = script
        with self.condition:
            self.condition.wait_for(lambda: self.running is script)
        try:
            for call in script.calls:
                result = (script.name, *self.run_call(self.emulator, call))
                script.results.append(result)
                self.finished.append(result)
        except Stopped:
            pass
        except BaseException as error:  # noqa: BLE001 - reraised in the main thread
            script.error = error
        with self.condition:
            script.done = True
            self.running = None
            self.condition.notify_all()

    def run(self, script, pauses=None):
        """Let ``script`` run until it pauses ``pauses`` more times, or to its end."""
        if script.done:
            return
        script.budget = script.pauses + pauses if pauses is not None else float("inf")
        if script.thread is None:
            script.thread = threading.Thread(target=self._body, args=(script,), daemon=True)
            script.thread.start()
        with self.condition:
            self.running = script
            self.condition.notify_all()
            self.condition.wait_for(lambda: self.running is None)
        if script.error is not None:
            raise script.error

    def stop(self, script):
        """Stop ``script`` where it is paused."""
        script.stop = True
        self.run(script)


class Generator:
    """An exported generator and the logits it should give."""

    def __init__(self, sprite, expected):
        self.sprite = sprite
        self.expected = expected

    def check(self, result, prefix):
        name, procedure, _, status, logits, top_k, length = result
        assert status == "ok", result[:4]
        if procedure == "cattorch reset":
            assert logits == top_k == [] and length == 0
            return
        expected = self.expected(prefix)
        torch.testing.assert_close(
            torch.tensor(logits, dtype=torch.float64), expected.double(),
            atol=TOLERANCE, rtol=0, msg=lambda message: f"{name} {procedure}: {message}",
        )
        assert top_k == expected.topk(2).indices.tolist()
        assert length == len(prefix)

    def interleave(self, first, second, schedule, fresh=False):
        """Run two scripts after a prefill of ``SETUP``, as ``schedule`` says.

        ``schedule`` is a list of ``(script index, pauses)``; None pauses runs
        the script to its end, and "stop" stops it where it is paused.
        Returns the emulator and the calls that finished, in order. A
        ``fresh`` sprite, as in a freshly loaded project, has no setup and
        its weights aren't decoded.

        Calls that report ok never overlap, so each one's logits follow the
        tokens of the ok calls that finished before it. Any other status
        leaves the generator stopped until a reset or prefill.
        """
        emulator = self.start(fresh)
        finished: list[tuple] = []
        scripts = Scripts(emulator, finished)
        pair = (Script("first", first), Script("second", second))
        for index, pauses in schedule:
            if pauses == "stop":
                scripts.stop(pair[index])
            else:
                scripts.run(pair[index], pauses)
        for script in pair:
            scripts.run(script)
        self.prefix: list | None = [] if fresh else list(SETUP)
        for result in finished:
            procedure, tokens, status = result[1:4]
            if status == "ok":
                if procedure == "cattorch decode":
                    assert self.prefix is not None, ("ok while stopped", finished)
                    self.prefix = [*self.prefix, *tokens]
                else:
                    self.prefix = list(tokens or [])
                self.check(result, self.prefix)
            else:
                assert status in {INTERRUPTED, OVERTAKEN}, result[:4]
                assert result[4] == result[5] == [], result[:4]
                self.prefix = None
        return emulator, finished

    def check_afterwards(self, emulator, finished):
        """Check the next decode, then that prefill recovers."""
        emulator.lists["cattorch tokens"] = [6]
        emulator.run_procedure("cattorch decode")
        status = emulator.variables["cattorch status"]
        if self.prefix is None:
            assert status == INTERRUPTED, finished
        else:
            self.check(self._result(emulator, "after"), [*self.prefix, 6])
        emulator.lists["cattorch tokens"] = [5, 3]
        emulator.run_procedure("cattorch prefill")
        assert emulator.variables["cattorch status"] == "ok"
        emulator.lists["cattorch tokens"] = [7]
        emulator.run_procedure("cattorch decode")
        self.check(self._result(emulator, "recovered"), [5, 3, 7])

    @staticmethod
    def _result(emulator, name):
        return (
            name, "cattorch decode", None, emulator.variables["cattorch status"],
            list(emulator.lists["cattorch logits"]), list(emulator.lists["cattorch top k ids"]),
            emulator.variables["cattorch cache length"],
        )

    def start(self, fresh=False):
        """A new emulator, after a prefill of ``SETUP`` unless ``fresh``."""
        emulator = ScratchEmulator(self.sprite)
        if not fresh:
            emulator.run_procedure("cattorch init")
            emulator.lists["cattorch tokens"] = list(SETUP)
            emulator.run_procedure("cattorch prefill")
            assert emulator.variables["cattorch status"] == "ok"
        return emulator

    def pauses(self, calls, fresh=False):
        """How many procedure calls ``calls`` make, run on their own after the setup."""
        counter = {"calls": 0}
        emulator = self.start(fresh)
        for start in paused_procedures(emulator):
            chain = emulator._chain_runner(start)

            def counted(chain=chain):
                counter["calls"] += 1
                chain()
            emulator._chains[start] = counted
        for procedure, tokens in calls:
            if tokens is not None:
                emulator.lists["cattorch tokens"] = list(tokens)
            emulator.run_procedure(procedure)
        return counter["calls"]


def kv_generator(tmp_path, sharing, hidden_prefill=True):
    torch.manual_seed(8)
    model = CachedLM(context=16, layers=3).eval()
    result = transpile(
        model, GenerationProgram(max_context=16, top_k=2, hidden_prefill=hidden_prefill),
        tmp_path / f"kv_{sharing}",
        codegen=CodegenConfig(layer_sharing=sharing, unrolling="compact"),
    )
    return Generator(load_sprite(result.path), lambda prefix: logits_of(model, prefix))


def recurrent_generator(tmp_path, hidden_prefill=True):
    model = make("gru")
    result = transpile(
        model, RecurrentGenerationProgram(model.states(), top_k=2, hidden_prefill=hidden_prefill),
        tmp_path / "recurrent",
    )
    return Generator(load_sprite(result.path), lambda prefix: reference_rows(model, prefix)[-1])


@pytest.fixture(scope="module", params=["banked", "per_layer", "recurrent"])
def generator(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp(request.param)
    if request.param == "recurrent":
        return recurrent_generator(directory)
    return kv_generator(directory, "auto" if request.param == "banked" else "off")


DECODE = ("cattorch decode", [5])
PREFILL = ("cattorch prefill", [2, 6, 3])
RESET = ("cattorch reset", None)
# A call that another script's call starts during, and what that script runs.
OVERLAPS = {
    "decode, then reset": ([DECODE], [RESET, ("cattorch decode", [6])]),
    "decode, then prefill": ([DECODE], [PREFILL, ("cattorch decode", [6])]),
    "decode, then decode": ([DECODE], [("cattorch decode", [6])]),
    "prefill, then prefill": ([PREFILL], [("cattorch prefill", [7, 1])]),
    "prefill, then reset": ([PREFILL], [RESET]),
    "prefill, then decode": ([PREFILL], [("cattorch decode", [6])]),
    "reset, then decode": ([RESET], [("cattorch decode", [6])]),
    "reset, then prefill": ([RESET], [PREFILL]),
}


@pytest.mark.parametrize("overlap", sorted(OVERLAPS))
def test_a_call_paused_while_another_runs_never_reports_wrong_logits(generator, overlap):
    first, second = OVERLAPS[overlap]
    for pause in spread(generator.pauses(first)):
        # The second script runs to its end while the first is paused, then
        # the first resumes; or the first is stopped where it was paused.
        for resume in (None, "stop"):
            state = generator.interleave(first, second, [(0, pause), (1, None), (0, resume)])
            generator.check_afterwards(*state)


@pytest.mark.parametrize("overlap", ["decode, then prefill", "prefill, then prefill"])
def test_a_call_taken_over_while_it_decodes_the_weights_leaves_them_to_decode_again(
    generator, overlap,
):
    # In a freshly loaded project the first call decodes the weights; one that
    # resumes doing so after another call decoded them leaves them wrong.
    first, second = OVERLAPS[overlap]
    for pause in spread(generator.pauses(first, fresh=True)):
        for resume in (None, "stop"):
            state = generator.interleave(
                first, second, [(0, pause), (1, None), (0, resume)], fresh=True,
            )
            generator.check_afterwards(*state)


@pytest.mark.parametrize("overlap", ["decode, then prefill", "prefill, then prefill"])
def test_a_resumed_call_that_ends_first_fails_the_call_running(generator, overlap):
    first, second = OVERLAPS[overlap]
    taken_over = generator.pauses(second[:1])
    for pause in spread(generator.pauses(first), 4):
        for other in spread(taken_over, 4):
            # The first script resumes while the second is paused in its
            # takeover, and ends first.
            emulator, finished = generator.interleave(
                first, second, [(0, pause), (1, other), (0, None)],
            )
            assert [result[0] for result in finished[:2]] == ["first", "second"]
            assert [result[3] for result in finished[:2]] == [OVERTAKEN, OVERTAKEN]
            generator.check_afterwards(emulator, finished)


def test_overlapping_calls_report_why(tmp_path):
    generator = kv_generator(tmp_path, "auto")
    emulator, finished = generator.interleave(
        [DECODE], [PREFILL, ("cattorch decode", [6])], [(0, 3), (1, None), (0, None)],
    )
    assert [result[3] for result in finished] == ["ok", "ok", OVERTAKEN]
    emulator.lists["cattorch tokens"] = [6]
    emulator.run_procedure("cattorch decode")
    assert emulator.variables["cattorch status"] == INTERRUPTED
    assert emulator.lists["cattorch logits"] == []
    emulator, finished = generator.interleave(
        [DECODE], [("cattorch decode", [6])], [(0, 3), (1, None), (0, None)],
    )
    # The refusal also stops the paused decode, which reports it too.
    assert [result[3] for result in finished] == [INTERRUPTED, OVERTAKEN]
    generator.check_afterwards(emulator, finished)


@pytest.mark.parametrize("kind", ["kv", "recurrent"])
def test_overlaps_without_hidden_prefill(tmp_path, kind):
    generator = (
        kv_generator(tmp_path, "off", hidden_prefill=False) if kind == "kv"
        else recurrent_generator(tmp_path, hidden_prefill=False)
    )
    first, second = OVERLAPS["prefill, then prefill"]
    for pause in range(generator.pauses(first)):
        state = generator.interleave(first, second, [(0, pause), (1, None), (0, None)])
        generator.check_afterwards(*state)
