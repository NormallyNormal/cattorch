"""``cattorch init`` from one script while another runs generation blocks, or stopped.

As in ``test_plain_overlap``, each script runs in its own thread, and they
switch where Scratch can pause a warp block: at a call to an internal
procedure, and at each repeat of a loop, the weight decoder's included.
Compressed weights are decoded by ``cattorch init``, or by the first block
that finds them undecoded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from cattorch import (
    CodegenConfig, GenerationProgram, QuantizationConfig, RecurrentGenerationProgram,
    StorageConfig, transpile,
)
from cattorch.util.scratch.emulator import ScratchEmulator

sys.path.insert(0, str(Path(__file__).parent))
from test_generation_automatic import INTERRUPTED, load_sprite  # noqa: E402
from test_generation_overlap import OVERTAKEN, SETUP, Script, Scripts  # noqa: E402
from test_optimization import CachedLM  # noqa: E402
from test_plain_overlap import spread  # noqa: E402
from test_program_overlap import LOOPS  # noqa: E402
from test_recurrent_generation import make  # noqa: E402

INIT, RESET = ("cattorch init", None), ("cattorch reset", None)
DECODE, PREFILL = ("cattorch decode", [5]), ("cattorch prefill", [2, 6, 3])
PUBLIC = ("cattorch init", "cattorch reset", "cattorch prefill", "cattorch decode")
# Pause places tried per case, spread over the call.
SAMPLES = 6
TOLERANCE = 1e-9
STORAGES = {
    "compressed": {},
    "int8": {"quantization": QuantizationConfig(bits=8, min_quantized_values=1)},
    "uncompressed": {"storage": StorageConfig(compression=False)},
}


def pause_points(emulator):
    """Where Scratch can pause a script: internal procedure calls and loop repeats.

    A script calling a public block isn't paused before it runs.
    """
    points = {
        start for name, start in emulator._procedures.items()
        if name not in PUBLIC and start is not None
    }
    points.update(
        block["inputs"]["SUBSTACK"][1] for block in emulator.blocks.values()
        if block["opcode"] in LOOPS and block["inputs"].get("SUBSTACK") is not None
    )
    return points


def run_call(emulator, call):
    procedure, tokens = call
    if tokens is not None:
        emulator.lists["cattorch tokens"] = list(tokens)
    emulator.run_procedure(procedure)
    return (
        procedure, tokens, emulator.variables["cattorch status"],
        list(emulator.lists["cattorch logits"]),
    )


class Generator:
    """A generation sprite, checked against what it gives when called alone."""

    def __init__(self, path, name):
        self.sprite = load_sprite(path)
        self.name = name
        self._expected: dict = {}

    def expected(self, prefix):
        """The logits after ``prefix``, from a prefill in a sprite of its own."""
        key = tuple(prefix)
        if key not in self._expected:
            emulator = ScratchEmulator(self.sprite)
            assert run_call(emulator, ("cattorch prefill", prefix))[2] == "ok"
            self._expected[key] = list(emulator.lists["cattorch logits"])
        return self._expected[key]

    def assert_logits(self, logits, prefix, result):
        torch.testing.assert_close(
            torch.tensor([float(value) for value in logits], dtype=torch.float64),
            torch.tensor(self.expected(prefix), dtype=torch.float64),
            atol=TOLERANCE, rtol=TOLERANCE, msg=lambda message: f"{result}: {message}",
        )

    def start(self, fresh, prepare):
        """A new emulator, after ``prepare(emulator)`` and, unless ``fresh``, init and a prefill."""
        emulator = ScratchEmulator(self.sprite)
        prepared = prepare(emulator)
        if not fresh:
            for call in (INIT, ("cattorch prefill", SETUP)):
                assert run_call(emulator, call)[2] == "ok"
        return emulator, prepared

    def pauses(self, calls, fresh):
        """How many times ``calls``, run on their own, can pause."""
        count = [0]

        def counting(emulator, start):
            chain = None

            def run():
                nonlocal chain
                count[0] += 1
                if chain is None:
                    chain = emulator._compile_chain(start)
                chain()
            return run

        def prepare(emulator):
            for start in pause_points(emulator):
                emulator._chains[start] = counting(emulator, start)

        emulator, _ = self.start(fresh, prepare)
        count[0] = 0
        for call in calls:
            run_call(emulator, call)
        return count[0]

    def interleave(self, first, second, schedule, fresh):
        """Run two scripts as ``schedule`` says, check their blocks, and return them.

        ``schedule`` is a list of ``(script index, pauses)``; None pauses runs
        the script to its end, and "stop" stops it where it is paused. A
        generation block that reports ok ran alone, so its logits follow the
        ok blocks before it; any other status stops the generator until a
        reset or prefill. Init reports nothing; the blocks after it show
        whether it left the weights right.
        """
        finished: list[tuple] = []
        emulator, scripts = self.start(fresh, lambda emulator: Scripts(
            emulator, finished, run_call, pause_points(emulator),
        ))
        pair = (Script("first", first), Script("second", second))
        for index, pauses in schedule:
            if pauses == "stop":
                scripts.stop(pair[index])
            else:
                scripts.run(pair[index], pauses)
        for script in pair:
            scripts.run(script)
        prefix: list | None = [] if fresh else list(SETUP)
        for result in finished:
            _, procedure, tokens, status, logits = result
            if procedure == "cattorch init":
                continue
            if status != "ok":
                assert status in {OVERTAKEN, INTERRUPTED}, result[:4]
                assert logits == [], result[:4]
                prefix = None
                continue
            if procedure == "cattorch reset":
                prefix = []
                continue
            if procedure == "cattorch decode":
                assert prefix is not None, ("ok while stopped", finished)
                prefix = [*prefix, *tokens]
            else:
                prefix = list(tokens)
            self.assert_logits(logits, prefix, result[:4])
        # A generation block stopped part way, last in every schedule, stops
        # the generator too.
        if any(
            script.stop and script.calls[len(script.results)] != INIT for script in pair
        ):
            prefix = None
        self.check_afterwards(emulator, prefix)
        return finished

    def check_afterwards(self, emulator, prefix):
        """The next decode continues, or is refused; a prefill then always recovers."""
        result = run_call(emulator, ("cattorch decode", [6]))
        if prefix is None:
            assert result[2] == INTERRUPTED, result[:3]
        else:
            assert result[2] == "ok", result[:3]
            self.assert_logits(result[3], [*prefix, 6], result[:3])
        for call, expected in ((("cattorch prefill", [5, 3]), [5, 3]), (DECODE, [5, 3, 5])):
            result = run_call(emulator, call)
            assert result[2] == "ok", result[:3]
            self.assert_logits(result[3], expected, result[:3])


# Each storage's decoder in the KV sprite; the recurrent sprite shares them.
@pytest.fixture(scope="module", params=[
    "kv compressed", "kv int8", "kv uncompressed", "recurrent int8",
])
def generator(request, tmp_path_factory):
    kind, storage = request.param.split()
    path = tmp_path_factory.mktemp(request.param.replace(" ", "_")) / kind
    if kind == "kv":
        torch.manual_seed(8)
        model = CachedLM(context=8, layers=1).eval()
        result = transpile(
            model, GenerationProgram(max_context=8), path,
            codegen=CodegenConfig(unrolling="compact"), **STORAGES[storage],
        )
    else:
        # Its cell state, one value everywhere, is filled at run time too.
        model = make("lstm")
        result = transpile(
            model, RecurrentGenerationProgram(model.states()), path, **STORAGES[storage],
        )
    return Generator(result.path, request.param)


SETUPS = ("fresh", "initialized")
# Init, and a block that another script's blocks start during.
OVERLAPS = {
    "init, then prefill": ([INIT], [PREFILL, ("cattorch decode", [6])]),
    "init, then decode": ([INIT], [DECODE, ("cattorch decode", [6])]),
    "init, then init": ([INIT], [INIT, PREFILL]),
    "decode, then init": ([DECODE], [INIT]),
    "prefill, then init": ([PREFILL], [INIT, ("cattorch decode", [6])]),
    "reset, then init": ([RESET], [INIT, DECODE]),
}


@pytest.mark.parametrize("overlap", sorted(OVERLAPS))
@pytest.mark.parametrize("setup", SETUPS)
def test_init_overlapping_a_generation_block_never_leaves_wrong_logits(generator, setup, overlap):
    first, second = OVERLAPS[overlap]
    fresh = setup == "fresh"
    for pause in spread(generator.pauses(first, fresh), SAMPLES):
        for resume in (None, "stop"):
            finished = generator.interleave(
                first, second, [(0, pause), (1, None), (0, resume)], fresh,
            )
            if first[0] != INIT and resume is None:
                # Init changed the weights under it.
                assert finished[-1][:4] == ("first", *first[0], OVERTAKEN), finished


@pytest.mark.parametrize("setup", SETUPS)
def test_a_stopped_init_leaves_the_weights_to_decode_again(generator, setup):
    fresh = setup == "fresh"
    for pause in spread(generator.pauses([INIT], fresh), SAMPLES):
        assert generator.interleave([INIT], [], [(0, pause), (0, "stop")], fresh) == []


def test_a_decode_taken_over_while_decoding_the_weights_by_init_fails(tmp_path):
    # In a freshly loaded project the first decode decodes the weights. An
    # init run meanwhile reused the decoder's variables, so the decode
    # resumed with wrong weights, reported ok and left them wrong for every
    # later block.
    torch.manual_seed(8)
    result = transpile(
        CachedLM(context=8, layers=1).eval(), GenerationProgram(max_context=8),
        tmp_path / "kv", codegen=CodegenConfig(unrolling="compact"),
    )
    generator = Generator(result.path, "kv")
    count = generator.pauses([DECODE], fresh=True)
    finished = generator.interleave(
        [DECODE], [INIT], [(0, count // 8), (1, None), (0, None)], fresh=True,
    )
    assert [(result[0], result[3]) for result in finished] == [
        ("second", "ok"), ("first", OVERTAKEN),
    ]
