"""Decoding the weights while other scripts' calls run, in every kind of sprite.

As in ``test_plain_overlap``, each script runs in its own thread, and they
switch where Scratch can pause a warp block: at a call to an internal
procedure, and at each repeat of a loop, the weight decoder's included. A
decode that another call started during must stop before it writes to the
weights or to the decoder's shared lists and variables again: one that
resumed after that call ended, or while it was decoding the weights itself,
left them half overwritten, and later calls reported ok with them.

That guard is switched off by default, for its cost in blocks and init
time (see ``cattorch.guards._GUARD_DECODERS``); the sprites of ``kind``
have it switched on.
"""

from __future__ import annotations

import pytest
import torch

from cattorch import (
    CodegenConfig, GenerationProgram, QuantizationConfig, RecurrentGenerationProgram, transpile,
)

import test_generation_init_overlap as generation
import test_plain_overlap as plain
import test_program_overlap as program
from overlap_harness import Script, Scripts, pause_points, record_pauses, spread
from scratch_status import INTERRUPTED, OVERTAKEN
from sprite_helpers import SpriteEmulator, load_sprite
from shared_models import CachedLM, recurrent_model

INT8 = {"quantization": QuantizationConfig(bits=8, min_quantized_values=1)}
STORAGES = {"compressed": {}, "int8": INT8}


class Kind:
    """A sprite, three calls that use its weights, and how to check what they gave."""

    public: tuple
    init: tuple
    calls: list
    setup: object

    def start(self, prepare):
        raise NotImplementedError

    def run_call(self, emulator, call):
        raise NotImplementedError

    def check(self, emulator, finished, scripts):
        raise NotImplementedError

    def pauses(self, calls):
        """How many times ``calls``, run on their own after the setup, can pause."""
        hits: list = []
        emulator = self.start(
            lambda emulator: record_pauses(emulator, pause_points(emulator, self.public), hits),
        )
        hits.clear()
        for call in calls:
            self.run_call(emulator, call)
        return len(hits)

    def interleave(self, calls, schedule):
        """Run a script per list of ``calls`` as ``schedule`` says, then check them.

        ``schedule`` is a list of ``(script index, pauses)``; None pauses runs
        the script to its end.
        """
        finished: list[tuple] = []
        holder = {}

        def prepare(emulator):
            holder["scripts"] = Scripts(
                emulator, finished, self.run_call, pause_points(emulator, self.public),
            )

        emulator = self.start(prepare)
        scripts = [Script(chr(ord("A") + index), each) for index, each in enumerate(calls)]
        holder["scripts"].play(scripts, schedule)
        self.check(emulator, finished, scripts)
        return finished


class Plain(Kind):
    public = plain.PUBLIC
    init = (plain.INIT, {})

    def __init__(self, path, options):
        self.plain, _ = plain.export(
            _Directory(path), "plain", plain.Tiny(), {"input": torch.zeros(1, 4)}, **options,
        )
        self.plain.name = "static"
        self.calls = plain.calls(self.plain, [10, 20, 30])
        self.setup = [self.init, *plain.calls(self.plain, [3])]

    def start(self, prepare):
        return self.plain.start(self.setup, prepare)[0]

    def run_call(self, emulator, call):
        return self.plain.run_call(emulator, call)

    def check(self, emulator, finished, scripts):
        for result in finished:
            self.plain.check(result)
        self.plain.check_afterwards(emulator)


class Program(Kind):
    init = program.INIT

    def __init__(self, path, options):
        model = program.Machine().eval()
        result = transpile(model, program.STATELESS, path / "machine", **options)
        self.sprite = program.Sprite(load_sprite(result.path), model, program.STATELESS)
        self.public = tuple(self.sprite.public)
        self.calls = [
            program.peek(1.0, 2.0, 3.0), program.other(-1.0, 0.5, 2.0),
            program.peek(0.5, -2.0, 1.0),
        ]
        self.setup = [self.init]
        self._expected: dict = {}

    def start(self, prepare):
        return self.sprite.start(self.setup, prepare)[0]

    def run_call(self, emulator, call):
        return self.sprite.run_call(emulator, call)

    def expected(self, call):
        """The outputs of ``call`` made alone, in a sprite of its own."""
        key = (call[0], tuple(call[1]["x"]))
        if key not in self._expected:
            result = self.run_call(SpriteEmulator(self.sprite.sprite), call)
            assert result[2] == "ok"
            self._expected[key] = result[3]
        return self._expected[key]

    def check_result(self, result):
        procedure, inputs, status, outputs = result[-5:-1]
        if procedure in program.LIFECYCLE:
            return
        if status != "ok":
            assert status in {INTERRUPTED, OVERTAKEN}, result
            return
        for name, values in self.expected((procedure, inputs)).items():
            torch.testing.assert_close(
                torch.tensor([float(value) for value in outputs[name]], dtype=torch.float64),
                torch.tensor([float(value) for value in values], dtype=torch.float64),
                atol=1e-9, rtol=0, msg=lambda message: f"{result[:3]}: {message}",
            )

    def check(self, emulator, finished, scripts):
        for result in finished:
            self.check_result(result)
        for call in self.calls:
            result = self.run_call(emulator, call)
            assert result[2] == "ok", result[:4]
            self.check_result(result)


class Generator(Kind):
    public = generation.PUBLIC
    init = generation.INIT
    calls = [("cattorch prefill", [2, 6, 3]), ("cattorch prefill", [1, 2]), ("cattorch prefill", [4, 1])]

    def __init__(self, path, kind, options):
        if kind == "kv":
            torch.manual_seed(8)
            result = transpile(
                CachedLM(context=8, layers=1).eval(), GenerationProgram(max_context=8), path / kind,
                codegen=CodegenConfig(unrolling="compact"), **options,
            )
        else:
            model = recurrent_model("lstm")
            result = transpile(
                model, RecurrentGenerationProgram(model.states()), path / kind, **options,
            )
        self.generator = generation.Generator(result.path, kind)

    def start(self, prepare):
        return self.generator.start(False, prepare)[0]

    def run_call(self, emulator, call):
        return generation.run_call(emulator, call)

    def check(self, emulator, finished, scripts):
        prefix: list | None = list(generation.SETUP)
        for result in finished:
            _, procedure, tokens, status, logits = result
            if procedure == "cattorch init":
                continue
            if status != "ok":
                assert status in {OVERTAKEN, INTERRUPTED}, result[:4]
                assert logits == [], result[:4]
                prefix = None
                continue
            prefix = list(tokens)
            self.generator.assert_logits(logits, prefix, result[:4])
        self.generator.check_afterwards(emulator, prefix)


class _Directory:
    def __init__(self, path):
        self.path = path

    def mktemp(self, name):
        path = self.path / name
        path.mkdir(parents=True, exist_ok=True)
        return path


KINDS = [
    f"{kind} {storage}" for kind in ("plain", "program", "kv", "recurrent")
    for storage in STORAGES
]


@pytest.fixture(scope="module", params=KINDS)
def kind(request, tmp_path_factory):
    name, storage = request.param.split()
    path = tmp_path_factory.mktemp(request.param.replace(" ", "_"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("cattorch.guards._GUARD_DECODERS", True)
        if name == "plain":
            return Plain(path, STORAGES[storage])
        if name == "program":
            return Program(path, STORAGES[storage])
        return Generator(path, name, STORAGES[storage])


def test_an_init_resumed_after_the_call_that_took_over_writes_no_weights(kind):
    # The init resumed after the call or init that took over had decoded the
    # weights and ended, and wrote some of them again before it paused once
    # more; a call that then ran, and found them decoded, reported ok with them.
    first, second, third = kind.calls
    for pause in spread(kind.pauses([kind.init]), 12)[:-1]:
        for resumed in (1, 10, 100):
            for taking_over in (second, kind.init):
                kind.interleave(
                    [[kind.init], [taking_over], [third]],
                    [(0, pause), (1, None), (0, resumed), (2, None), (0, None)],
                )


def test_a_decode_resumed_while_another_decodes_writes_nothing(kind):
    # Both decode with the same lists and variables: the init resumed while
    # the second script's init was paused, decoding too, overwrote them, and
    # the second script's call then reported ok with what that init wrote.
    first, second, third = kind.calls
    count = kind.pauses([kind.init])
    for pause in spread(count, 5)[:-1]:
        for other in spread(count, 5)[1:-1]:
            for resumed in (1, 10):
                kind.interleave(
                    [[kind.init], [kind.init, second]],
                    [(0, pause), (1, other), (0, resumed), (1, None), (0, None)],
                )


@pytest.mark.parametrize("storage", sorted(STORAGES))
def test_a_call_taken_over_before_it_decodes_decodes_nothing(tmp_path, storage):
    # In a freshly loaded project each entry point's first call decodes its
    # weights. The first call, paused as it started and resumed while the
    # second decoded another entry point's weights with the same lists,
    # decoded its own in the epoch the second had started, as if it were
    # current, and the second reported ok with wrong outputs. This check is
    # on by default.
    sprite = Program(tmp_path, STORAGES[storage])
    sprite.setup = []
    first, second, _ = sprite.calls
    count = sprite.pauses([second])
    for other in spread(count, 12)[1:-1]:
        for resumed in (1, 10, 40):
            sprite.interleave(
                [[first], [second]], [(0, 0), (1, other), (0, resumed), (1, None), (0, None)],
            )


def test_the_decoders_are_unguarded_by_default(tmp_path):
    # The guard costs blocks in every compressed sprite, and init time. By
    # default only the unpacking checks the epoch, as it starts.
    result = transpile(plain.Tiny().eval(), torch.zeros(1, 4), tmp_path / "tiny")
    blocks = load_sprite(result.path)["blocks"].values()
    assert sum(block["opcode"] == "control_stop" for block in blocks) == 1
    assert not any(
        "decode" in block["mutation"]["proccode"]
        for block in blocks if block["opcode"] == "procedures_prototype"
        and "epoch" in block["mutation"]["argumentnames"]
    )
