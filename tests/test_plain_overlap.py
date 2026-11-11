"""Forward calls of a plain export from several scripts at once, or stopped.

As in ``test_program_overlap``, each script runs in its own thread, and they
switch where Scratch can pause a warp block: at a call to an internal
procedure, and at each repeat of a loop.
"""

from __future__ import annotations

import sys
import threading
from collections import defaultdict
from pathlib import Path

import pytest
import torch
from torch import nn

from cattorch import QuantizationConfig, StorageConfig, interface, transpile
from cattorch.util.scratch.emulator import ScratchEmulator

sys.path.insert(0, str(Path(__file__).parent))
from test_generation_automatic import load_sprite  # noqa: E402
from test_generation_overlap import Script, Scripts, spread  # noqa: E402
from test_program_overlap import LOOPS, OVERTAKEN  # noqa: E402

TOLERANCE = 1e-5
FORWARD, INIT, PREPARE = "cattorch forward", "cattorch init", "cattorch prepare for save"
PUBLIC = (FORWARD, INIT, PREPARE)
# Pause places tried per case, spread over the call.
SAMPLES = 24


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


class Plain:
    """A plain export and the outputs it should give."""

    def __init__(self, path, model, inputs):
        self.sprite = load_sprite(path)
        self.model = model
        # The input lists' names, with an example of each.
        self.inputs = inputs
        with torch.no_grad():
            outputs = model(*inputs.values())
        self.output_count = 1 if isinstance(outputs, torch.Tensor) else len(outputs)
        self._expected: dict = {}

    def output_names(self):
        return ["output", *(f"output_{index}" for index in range(1, self.output_count))]

    def expected(self, inputs):
        """The outputs of a call made alone, in a sprite of its own.

        It matches the model, within the error of how the weights are stored.
        """
        key = tuple((name, tuple(values)) for name, values in inputs.items())
        if key not in self._expected:
            emulator = ScratchEmulator(self.sprite)
            procedure, _, status, outputs, _ = self.run_call(emulator, (FORWARD, inputs))
            assert status == "ok"
            values = [
                torch.tensor(inputs[name], dtype=example.dtype).reshape(-1, *example.shape[1:])
                for name, example in self.inputs.items()
            ]
            with torch.no_grad():
                reference = self.model(*values)
            for actual, value in zip(
                outputs, [reference] if isinstance(reference, torch.Tensor) else reference,
            ):
                torch.testing.assert_close(
                    torch.tensor([float(item) for item in actual]), value.flatten().float(),
                    atol=1e-2, rtol=1e-2,
                )
            self._expected[key] = outputs
        return self._expected[key]

    def outputs(self, emulator):
        return [_logical(emulator, name) for name in self.output_names()]

    def run_call(self, emulator, call):
        """Run a call; what it left, with the broadcasts its own script sent."""
        procedure, inputs = call
        for name, values in inputs.items():
            emulator.lists[name] = list(values)
        sent = _sent(emulator)[threading.get_ident()]
        before = len(sent)
        emulator.run_procedure(procedure)
        return (
            procedure, inputs, emulator.variables["cattorch status"], self.outputs(emulator),
            tuple(sent[before:]),
        )

    def check(self, result, ok=False, previous=None):
        """Check a finished call; ``previous`` are the outputs the last ok call left.

        Every call that ends sends its completion broadcast, whatever its
        status. One that another call ran during writes no output: the
        outputs stay as the last ok call left them, or are empty if it was
        taken over while it copied them.
        """
        procedure, inputs, status, outputs, sent = result[-5:]
        assert sent[-1:] == (interface.completion(procedure),), result
        if procedure != FORWARD:
            return
        if not ok and status != "ok":
            assert status == OVERTAKEN, result
            assert all(values == [] for values in outputs) or outputs == previous, result
            return
        assert status == "ok", result
        for actual, expected in zip(outputs, self.expected(inputs)):
            torch.testing.assert_close(
                torch.tensor([float(value) for value in actual], dtype=torch.float64),
                torch.tensor([float(value) for value in expected], dtype=torch.float64),
                atol=TOLERANCE, rtol=TOLERANCE,
                msg=lambda message: f"{result[:3]}: {message}",
            )

    def start(self, setup, prepare):
        emulator = ScratchEmulator(self.sprite)
        prepared = prepare(emulator)
        emulator.previous = None
        for call in setup:
            result = self.run_call(emulator, call)
            self.check(result, ok=True)
            if call[0] == FORWARD:
                emulator.previous = result[3]
        return emulator, prepared

    def pauses(self, setup, calls):
        """How many times ``calls``, run on their own after ``setup``, can pause."""
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

        emulator, _ = self.start(setup, prepare)
        count[0] = 0
        for call in calls:
            self.run_call(emulator, call)
        return count[0]

    def interleave(self, setup, first, second, schedule):
        """Run two scripts after ``setup``, as ``schedule`` says, and check their calls.

        ``schedule`` is a list of ``(script index, pauses)``; None pauses runs
        the script to its end, and "stop" stops it where it is paused.
        """
        finished: list[tuple] = []
        emulator, scripts = self.start(setup, lambda emulator: Scripts(
            emulator, finished, self.run_call, pause_points(emulator),
        ))
        pair = (Script("first", first), Script("second", second))
        for index, pauses in schedule:
            if pauses == "stop":
                scripts.stop(pair[index])
            else:
                scripts.run(pair[index], pauses)
        for script in pair:
            scripts.run(script)
        previous = emulator.previous
        for result in finished:
            self.check(result, previous=previous)
            if result[1] == FORWARD and result[3] == "ok":
                previous = result[4]
        return emulator, finished

    def check_afterwards(self, emulator):
        """The next calls run as usual: nothing was left half done."""
        for call in (self.call(1), self.call(2)):
            self.check(self.run_call(emulator, call), ok=True)

    def call(self, seed, length=None):
        generator = torch.Generator().manual_seed(seed)
        inputs = {}
        for name, example in self.inputs.items():
            shape = example.shape if length is None else (length, *example.shape[1:])
            if example.dtype.is_floating_point:
                values = torch.randn(shape, generator=generator)
            else:
                values = torch.randint(0, 4, shape, generator=generator)
            inputs[name] = values.flatten().tolist()
        return (FORWARD, inputs)


def _sent(emulator):
    """The broadcasts each script (thread) of ``emulator`` sent."""
    if not hasattr(emulator, "sent"):
        emulator.sent = defaultdict(list)
        emulator.on_broadcast = lambda message: emulator.sent[threading.get_ident()].append(message)
    return emulator.sent


def _logical(emulator, name):
    """A list's values, joined across its shards."""
    values = list(emulator.lists[name])
    shard = 2
    while True:
        try:
            values += emulator.lists[f"{name} shard {shard}"]
        except KeyError:
            return values
        shard += 1


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.first = nn.Linear(4, 6)
        self.second = nn.Linear(6, 3)

    def forward(self, x):
        return self.second(torch.softmax(self.first(x), -1))


def export(tmp_path_factory, name, model, inputs, **options):
    model = model.eval()
    path = tmp_path_factory.mktemp(name) / name
    result = transpile(model, tuple(inputs.values()), path, **options)
    return Plain(result.path, model, inputs), result


EXPORTS = {
    # Stored compressed: the first call decodes the weights.
    "static": lambda: (Tiny(), {"input": torch.zeros(1, 4)}, {}),
    "uncompressed": lambda: (Tiny(), {"input": torch.zeros(1, 4)}, {
        "storage": StorageConfig(compression=False),
    }),
    "int8": lambda: (Tiny(), {"input": torch.zeros(1, 4)}, {
        "quantization": QuantizationConfig(bits=8, min_quantized_values=1),
    }),
    "variable length": lambda: (Tiny(), {"input": torch.zeros(2, 4)}, {"dynamic": {0: (1, 3)}}),
    "variable length per item": lambda: (
        nn.TransformerEncoderLayer(4, 1, 4, dropout=0.0, batch_first=True),
        {"input": torch.zeros(2, 2, 4)}, {"dynamic": {0: (1, 3)}},
    ),
    "recurrent step": lambda: (
        _Cell(), {"input": torch.zeros(1, 3), "input_1": torch.zeros(1, 4)}, {},
    ),
}


class _Cell(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.cell = nn.GRUCell(3, 4)

    def forward(self, x, h):
        h = self.cell(x, h)
        return h, h.sum(-1)


_EXPORTED: dict = {}


def _exported(name, tmp_path_factory):
    if name not in _EXPORTED:
        model, inputs, options = EXPORTS[name]()
        sprite, _ = export(tmp_path_factory, name.replace(" ", "_"), model, inputs, **options)
        sprite.name = name
        _EXPORTED[name] = sprite
    return _EXPORTED[name]


@pytest.fixture(scope="module", params=sorted(EXPORTS))
def plain(request, tmp_path_factory):
    return _exported(request.param, tmp_path_factory)


# Init and prepare for save decode or keep the weights, which depends on the
# storage, and wrap the forward procedure, which a variable length changes.
@pytest.fixture(scope="module", params=["int8", "static", "uncompressed", "variable length"])
def stored(request, tmp_path_factory):
    return _exported(request.param, tmp_path_factory)


def calls(plain, seeds):
    length = 2 if plain.name.startswith("variable length") else None
    return [plain.call(seed, length) for seed in seeds]


SETUPS = ("fresh", "initialized")


def setup_calls(plain, setup):
    return [] if setup == "fresh" else [(INIT, {}), *calls(plain, [3])]


@pytest.mark.parametrize("setup", SETUPS)
def test_a_forward_paused_while_another_runs_never_reports_wrong_outputs(plain, setup):
    before = setup_calls(plain, setup)
    first, second = calls(plain, [10]), calls(plain, [20, 21])
    for pause in spread(plain.pauses(before, first), SAMPLES):
        for resume in (None, "stop"):
            emulator, finished = plain.interleave(
                before, first, second, [(0, pause), (1, None), (0, resume)],
            )
            assert [result[3] for result in finished if result[0] == "second"] == ["ok", "ok"]
            plain.check_afterwards(emulator)


@pytest.mark.parametrize("setup", SETUPS)
def test_a_stopped_forward_leaves_the_next_one_to_run_as_usual(plain, setup):
    before = setup_calls(plain, setup)
    first = calls(plain, [10])
    for pause in spread(plain.pauses(before, first), SAMPLES):
        emulator, finished = plain.interleave(before, first, [], [(0, pause), (0, "stop")])
        assert finished == []
        plain.check_afterwards(emulator)


@pytest.mark.parametrize(("first", "second"), [
    ("init", "forward"), ("forward", "init"), ("forward", "prepare for save"),
    ("prepare for save", "forward"),
])
@pytest.mark.parametrize("setup", SETUPS)
def test_init_and_prepare_for_save_never_leave_wrong_outputs_reported_ok(
    stored, setup, first, second,
):
    plain = stored
    if PREPARE not in ScratchEmulator(plain.sprite)._procedures and "prepare for save" in (
        first, second,
    ):
        pytest.skip("this export has no prepare for save")
    before = setup_calls(plain, setup)
    named = {"init": [(INIT, {})], "prepare for save": [(PREPARE, {})], "forward": calls(plain, [10])}
    first_calls, second_calls = named[first], [*named[second], *calls(plain, [20])]
    for pause in spread(plain.pauses(before, first_calls), SAMPLES):
        for resume in (None, "stop"):
            emulator, _ = plain.interleave(
                before, first_calls, second_calls, [(0, pause), (1, None), (0, resume)],
            )
            plain.check_afterwards(emulator)


def test_a_resumed_forward_that_ends_first_fails_the_one_running(plain):
    # The first script resumes while the second is paused, and ends first: it
    # changes the epoch, so the second, which read what it wrote, fails too.
    before = setup_calls(plain, "initialized")
    first, second = calls(plain, [10]), calls(plain, [20])
    count = plain.pauses(before, first)
    for pause in spread(count, 6)[1:-1]:
        for other in spread(plain.pauses(before, second), 6)[1:-1]:
            emulator, finished = plain.interleave(
                before, first, second, [(0, pause), (1, other), (0, None)],
            )
            assert [(result[0], result[3]) for result in finished] == [
                ("first", OVERTAKEN), ("second", OVERTAKEN),
            ]
            plain.check_afterwards(emulator)


def test_sharded_outputs_are_cleared_by_a_forward_another_ran_during(
    tmp_path_factory, small_list_limit,
):
    small_list_limit(300)
    torch.manual_seed(2)
    model = nn.Sequential(nn.Linear(4, 640), nn.Tanh())
    plain, result = export(tmp_path_factory, "sharded", model, {"input": torch.zeros(1, 4)})
    plain.name = "sharded"
    assert "output shard 3" in {entry[0] for entry in plain.sprite["lists"].values()}
    first, second = calls(plain, [10]), calls(plain, [20])
    overtaken = 0
    for pause in spread(plain.pauses([], first), 12):
        emulator, finished = plain.interleave([], first, second, [(0, pause), (1, None), (0, None)])
        overtaken += finished[-1][3] == OVERTAKEN
        plain.check_afterwards(emulator)
    assert overtaken


def test_the_blocks_on_the_canvas_run_the_public_blocks(plain):
    # A variable-length export's clickable forward ran the procedure it wraps.
    loose = sorted(
        block["mutation"]["proccode"] for block in plain.sprite["blocks"].values()
        if isinstance(block, dict) and block.get("topLevel")
        and block["opcode"] == "procedures_call"
    )
    assert loose == [FORWARD, INIT]
