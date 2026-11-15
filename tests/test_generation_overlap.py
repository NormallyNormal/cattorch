"""Generation calls from several scripts at once, interleaved as Scratch can.

Scratch pauses a warp block that has run for 500 ms where it calls a
procedure or repeats a loop, and runs other scripts meanwhile. These tests
run each script in its own thread and switch between them where a
procedure is called, then check that no call reports ``ok`` with wrong
logits and that a later reset or prefill always recovers.
"""

from __future__ import annotations

import pytest
import torch

from cattorch import CodegenConfig, GenerationProgram, RecurrentGenerationProgram, transpile
from cattorch.util.scratch.emulator import ScratchEmulator

from overlap_harness import Script, Scripts, pause_points, record_pauses, spread
from scratch_status import INTERRUPTED, OVERTAKEN
from sprite_helpers import load_sprite
from test_generation_automatic import logits_of
from shared_models import CachedLM, recurrent_model, reference_rows
from cattorch.interface import TOKEN_IDS

TOLERANCE = 1e-4
# Pause places tried per case, spread over the call.
SAMPLES = 8
SETUP = [1, 2, 3, 4]
# Scripts pause only at calls inside the warp blocks, not before these run.
PUBLIC = ("cattorch init", "cattorch reset", "cattorch prefill", "cattorch decode")


def run_generation_call(emulator, call):
    """Run one ``(procedure, tokens)`` call and return what it left."""
    procedure, tokens = call
    if tokens is not None:
        emulator.lists[TOKEN_IDS] = list(tokens)
    before = list(emulator.lists["cattorch logits"]), list(emulator.lists["cattorch top k ids"])
    emulator.run_procedure(procedure)
    return (
        procedure, tokens, emulator.variables["cattorch status"],
        list(emulator.lists["cattorch logits"]), list(emulator.lists["cattorch top k ids"]),
        emulator.variables["cattorch cache length"], before,
    )


class Generator:
    """An exported generator and the logits it should give."""

    def __init__(self, sprite, expected):
        self.sprite = sprite
        self.expected = expected

    def check(self, result, prefix):
        name, procedure, _, status, logits, top_k, length = result[:7]
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
        scripts = Scripts(
            emulator, finished, run_generation_call, pause_points(emulator, PUBLIC, loops=False),
        )
        scripts.play((Script("first", first), Script("second", second)), schedule)
        self.prefix: list | None = [] if fresh else list(SETUP)
        # A call that fails writes no output: the logits stay as the last ok
        # call left them, or are empty if one was taken over while it copied.
        outputs = ([], []) if fresh else self.setup_outputs
        for result in finished:
            procedure, tokens, status = result[1:4]
            if status == "ok":
                outputs = (result[4], result[5])
                if procedure == "cattorch decode":
                    assert self.prefix is not None, ("ok while stopped", finished)
                    self.prefix = [*self.prefix, *tokens]
                else:
                    self.prefix = list(tokens or [])
                self.check(result, self.prefix)
            else:
                assert status in {INTERRUPTED, OVERTAKEN}, result[:4]
                self.prefix = None
        # Once every call has ended, the outputs are the last ok call's, or
        # empty if a call was taken over while it copied them; a call that
        # fails writes none. (While a call copies them, one that ends meanwhile
        # can find them part copied, and one stopped meanwhile leaves them so.)
        if not any(pauses == "stop" for _, pauses in schedule):
            assert (
                list(emulator.lists["cattorch logits"]), list(emulator.lists["cattorch top k ids"]),
            ) in (outputs, ([], [])), finished
        return emulator, finished

    def check_afterwards(self, emulator, finished):
        """Check the next decode, then that prefill recovers."""
        emulator.lists[TOKEN_IDS] = [6]
        emulator.run_procedure("cattorch decode")
        status = emulator.variables["cattorch status"]
        if self.prefix is None:
            assert status == INTERRUPTED, finished
        else:
            self.check(self._result(emulator, "after"), [*self.prefix, 6])
        emulator.lists[TOKEN_IDS] = [5, 3]
        emulator.run_procedure("cattorch prefill")
        assert emulator.variables["cattorch status"] == "ok"
        emulator.lists[TOKEN_IDS] = [7]
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
            emulator.lists[TOKEN_IDS] = list(SETUP)
            emulator.run_procedure("cattorch prefill")
            assert emulator.variables["cattorch status"] == "ok"
            self.setup_outputs = (
                list(emulator.lists["cattorch logits"]), list(emulator.lists["cattorch top k ids"]),
            )
        return emulator

    def pauses(self, calls, fresh=False):
        """How many procedure calls ``calls`` make, run on their own after the setup."""
        hits: list = []
        emulator = self.start(fresh)
        record_pauses(emulator, pause_points(emulator, PUBLIC, loops=False), hits)
        for procedure, tokens in calls:
            if tokens is not None:
                emulator.lists[TOKEN_IDS] = list(tokens)
            emulator.run_procedure(procedure)
        return len(hits)


def kv_generator(tmp_path, sharing, hidden_prefill=True):
    torch.manual_seed(8)
    model = CachedLM(context=16, layers=3).eval()
    result = transpile(
        model, GenerationProgram(max_context=16, top_k=2, hidden_prefill=hidden_prefill),
        tmp_path / f"kv_{sharing}",
        codegen=CodegenConfig(layer_sharing=sharing == "auto", unrolling="compact"),
    )
    return Generator(load_sprite(result.path), lambda prefix: logits_of(model, prefix))


def recurrent_generator(tmp_path, hidden_prefill=True):
    model = recurrent_model("gru")
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
    for pause in spread(generator.pauses(first), SAMPLES):
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
    for pause in spread(generator.pauses(first, fresh=True), SAMPLES):
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
    emulator.lists[TOKEN_IDS] = [6]
    emulator.run_procedure("cattorch decode")
    assert emulator.variables["cattorch status"] == INTERRUPTED
    # The overtaken decode and the refused one wrote nothing.
    assert emulator.lists["cattorch logits"] == finished[1][4]
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
