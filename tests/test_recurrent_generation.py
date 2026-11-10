"""Recurrent token-by-token generation end to end in the Scratch emulator."""

from __future__ import annotations

import contextlib
import json
import sys
import zipfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from cattorch import (
    CharTokenizer, CodegenConfig, QuantizationConfig, RecurrentGenerationProgram, State,
    UnsupportedModelError, build_project, transpile, verify,
)
from cattorch.util.scratch.emulator import ProjectEmulator, ScratchEmulator
from cattorch.util.scratch.interface import read_logical_list

sys.path.insert(0, str(Path(__file__).parent))

TOLERANCE = 1e-4


class GRULM(nn.Module):
    """A GRU language model that starts from a learned, non-constant state."""

    def __init__(self, vocab=16, width=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.gru = nn.GRU(width, width, batch_first=True)
        self.head = nn.Linear(width, vocab)
        self.register_buffer("h0", torch.randn(1, 1, width))

    def states(self):
        return (State("hidden", self.h0.clone()),)

    def forward(self, tokens):
        output, _ = self.gru(self.embed(tokens), self.h0)
        return self.head(output)

    def step(self, token, hidden):
        output, hidden = self.gru(self.embed(token), hidden)
        return self.head(output), hidden


class LSTMLM(nn.Module):
    """Two states; the cell starts at one non-zero value everywhere."""

    def __init__(self, vocab=16, width=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.lstm = nn.LSTM(width, width, batch_first=True)
        self.head = nn.Linear(width, vocab)
        self.width = width

    def states(self):
        return (
            State("hidden", torch.zeros(1, 1, self.width)),
            State("cell", torch.full((1, 1, self.width), 0.25)),
        )

    def forward(self, tokens):
        hidden, cell = (state.initial for state in self.states())
        output, _ = self.lstm(self.embed(tokens), (hidden, cell))
        return self.head(output)

    def step(self, token, hidden, cell):
        output, (hidden, cell) = self.lstm(self.embed(token), (hidden, cell))
        return self.head(output)[:, 0], hidden, cell


def rms(value, weight, eps=1e-5):
    return weight * value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps)


class SSMBlock(nn.Module):
    """A Mamba-2 mixer: gated depthwise causal convolution and a selective scan."""

    def __init__(self, width, heads, head_dim, state, kernel):
        super().__init__()
        self.inner, self.heads, self.head_dim = heads * head_dim, heads, head_dim
        self.state, self.kernel = state, kernel
        self.conv_dim = self.inner + 2 * state
        self.norm = nn.Parameter(1 + 0.1 * torch.randn(width))
        self.in_proj = nn.Linear(width, self.inner + self.conv_dim + heads, bias=False)
        self.conv_weight = nn.Parameter(0.5 * torch.randn(self.conv_dim, kernel))
        self.conv_bias = nn.Parameter(0.1 * torch.randn(self.conv_dim))
        self.dt_bias = nn.Parameter(0.1 * torch.randn(heads))
        self.A_log = nn.Parameter(torch.log(torch.linspace(1, 4, heads)))
        self.D = nn.Parameter(torch.ones(heads))
        self.gate_norm = nn.Parameter(1 + 0.1 * torch.randn(self.inner))
        self.out_proj = nn.Linear(self.inner, width, bias=False)

    def project(self, h):
        return self.in_proj(rms(h, self.norm)).split(
            [self.inner, self.conv_dim, self.heads], dim=-1,
        )

    def scan(self, ssm, xbc, dt):
        x, B, C = F.silu(xbc).split([self.inner, self.state, self.state], dim=-1)
        x = x.reshape(self.heads, self.head_dim)
        dt = F.softplus(dt.reshape(self.heads) + self.dt_bias)
        decay = torch.exp(-dt * torch.exp(self.A_log))
        ssm = ssm * decay[:, None, None] + (dt[:, None] * x)[:, :, None] * B[None, None, :]
        y = (ssm * C[None, None, :]).sum(-1) + self.D[:, None] * x
        return ssm, y.reshape(1, self.inner)

    def finish(self, h, y, gate):
        return h + self.out_proj(rms(y * F.silu(gate), self.gate_norm))

    def step(self, h, conv, ssm):
        gate, xbc, dt = self.project(h)
        window = torch.cat([conv, xbc.reshape(self.conv_dim, 1)], dim=-1)
        mixed = (window * self.conv_weight).sum(-1) + self.conv_bias
        ssm, y = self.scan(ssm, mixed, dt)
        return self.finish(h, y, gate), window[:, 1:], ssm

    def forward(self, h):
        gate, xbc, dt = self.project(h)
        length = h.shape[0]
        mixed = F.conv1d(
            xbc.T[None], self.conv_weight[:, None], self.conv_bias,
            padding=self.kernel - 1, groups=self.conv_dim,
        )[0, :, :length].T
        ssm = torch.zeros(self.heads, self.head_dim, self.state)
        rows = []
        for position in range(length):
            ssm, y = self.scan(ssm, mixed[position], dt[position])
            rows.append(y)
        return self.finish(h, torch.cat(rows), gate)


class SSMLM(nn.Module):
    def __init__(self, vocab=24, width=8, layers=2, heads=2, head_dim=4, state=4, kernel=3):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList(
            SSMBlock(width, heads, head_dim, state, kernel) for _ in range(layers)
        )
        self.norm = nn.Parameter(1 + 0.1 * torch.randn(width))
        self.head = nn.Linear(width, vocab, bias=False)

    def states(self):
        block = self.blocks[0]
        return (
            State("conv", torch.zeros(len(self.blocks), block.conv_dim, block.kernel - 1)),
            State("ssm", torch.zeros(len(self.blocks), block.heads, block.head_dim, block.state)),
        )

    def forward(self, tokens):
        h = self.embed(tokens)[0]
        for block in self.blocks:
            h = block(h)
        return self.head(rms(h, self.norm))[None]

    def step(self, token, conv, ssm):
        h = self.embed(token).reshape(1, -1)
        convs, ssms = [], []
        for index, block in enumerate(self.blocks):
            h, new_conv, new_ssm = block.step(h, conv[index], ssm[index])
            convs.append(new_conv)
            ssms.append(new_ssm)
        return self.head(rms(h, self.norm)), torch.stack(convs), torch.stack(ssms)


MODELS = {"gru": GRULM, "lstm": LSTMLM, "ssm": SSMLM}


def make(name):
    torch.manual_seed({"gru": 1, "lstm": 2, "ssm": 3}[name])
    return MODELS[name]().eval()


def load_sprite(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("sprite.json"))


PROGRAM_FIELDS = {"step", "max_tokens", "hidden_prefill", "top_k", "reference"}


def export(model, path, **options):
    """Export with the model's states; program fields and transpile options mixed."""
    fields = {key: options.pop(key) for key in PROGRAM_FIELDS & set(options)}
    result = transpile(model, RecurrentGenerationProgram(model.states(), **fields), path, **options)
    return ScratchEmulator(load_sprite(result.path)), result


def reference_rows(model, tokens):
    with torch.no_grad():
        return model(torch.tensor([tokens])).reshape(len(tokens), -1)


def logits(emulator):
    return torch.tensor(emulator.lists["cattorch logits"], dtype=torch.float64)


def run(emulator, procedure, tokens=None):
    if tokens is not None:
        emulator.lists["cattorch tokens"] = list(tokens)
    emulator.run_procedure(procedure)
    return emulator.variables["cattorch status"]


def states(emulator, model):
    return {state.name: list(emulator.lists[f"_cattorch state {state.name}"]) for state in model.states()}


@pytest.fixture(scope="module", params=sorted(MODELS))
def exported(request, tmp_path_factory):
    model = make(request.param)
    emulator, result = export(
        model, tmp_path_factory.mktemp(request.param) / "lm", top_k=3, reference="forward",
    )
    return model, emulator, result


def test_prefill_then_decode_matches_the_stateless_forward(exported):
    model, emulator, result = exported
    tokens = [1, 3, 5, 2, 7, 4, 6, 9, 11, 2]
    expected = reference_rows(model, tokens).double()
    assert run(emulator, "cattorch init") == "ok"
    assert run(emulator, "cattorch prefill", tokens[:5]) == "ok"
    rows = [logits(emulator)]
    for token in tokens[5:]:
        assert run(emulator, "cattorch decode", [token]) == "ok"
        rows.append(logits(emulator))
    torch.testing.assert_close(torch.stack(rows), expected[4:], atol=TOLERANCE, rtol=0)
    assert emulator.variables["cattorch cache length"] == 10
    assert emulator.variables["cattorch max context"] == "Infinity"
    assert emulator.lists["cattorch top k ids"] == expected[-1].topk(3).indices.tolist()
    torch.testing.assert_close(
        torch.tensor(emulator.lists["cattorch top k values"], dtype=torch.float64),
        expected[-1].topk(3).values, atol=TOLERANCE, rtol=0,
    )
    assert result.procedures == ("cattorch init", "cattorch reset", "cattorch prefill", "cattorch decode")
    assert result.inputs[0].list_name == "cattorch tokens"
    assert result.output.list_name == "cattorch logits"
    assert result.output.numel == expected.shape[1]


def test_prefill_equals_the_same_tokens_decoded_one_by_one(exported):
    model, emulator, _ = exported
    run(emulator, "cattorch prefill", [4, 1, 8, 2])
    prefilled = (logits(emulator), states(emulator, model))
    run(emulator, "cattorch reset")
    for token in (4, 1, 8, 2):
        assert run(emulator, "cattorch decode", [token]) == "ok"
    assert torch.equal(logits(emulator), prefilled[0])
    assert states(emulator, model) == prefilled[1]
    assert emulator.variables["cattorch cache length"] == 4


def test_reset_restores_the_initial_states(exported):
    model, emulator, _ = exported
    initial = {state.name: state.initial.flatten().tolist() for state in model.states()}
    run(emulator, "cattorch prefill", [2, 3, 4])
    assert states(emulator, model) != initial
    assert run(emulator, "cattorch reset") == "ok"
    assert states(emulator, model) == pytest.approx(initial)
    assert emulator.variables["cattorch cache length"] == 0
    assert emulator.lists["cattorch logits"] == []
    assert emulator.lists["cattorch top k ids"] == []
    assert run(emulator, "cattorch decode", [5]) == "ok"
    torch.testing.assert_close(
        logits(emulator), reference_rows(model, [5])[0].double(), atol=TOLERANCE, rtol=0,
    )


def test_invalid_tokens_leave_every_state_untouched(exported):
    model, emulator, result = exported
    vocabulary = result.output.numel
    run(emulator, "cattorch prefill", [1, 2])
    before = states(emulator, model)
    for tokens, status in (
        ([], "decode requires exactly one token"),
        ([1, 2], "decode requires exactly one token"),
        ([vocabulary], "token ids must be whole numbers"),
        ([-1], "token ids must be whole numbers"),
        ([2.5], "token ids must be whole numbers"),
        (["abc"], "token ids must be whole numbers"),
    ):
        assert run(emulator, "cattorch decode", tokens).startswith(status), tokens
        assert states(emulator, model) == before
        assert emulator.variables["cattorch cache length"] == 2
        assert emulator.lists["cattorch logits"] == []
        assert emulator.lists["cattorch top k ids"] == []
    # As for a K/V cache, prefill resets before it checks its tokens.
    initial = {state.name: state.initial.flatten().tolist() for state in model.states()}
    for tokens in ([], [1, vocabulary], [3, -2]):
        assert run(emulator, "cattorch prefill", tokens) != "ok"
        assert states(emulator, model) == pytest.approx(initial)
        assert emulator.variables["cattorch cache length"] == 0
        assert emulator.lists["cattorch logits"] == []
    assert run(emulator, "cattorch decode", [vocabulary - 1]) == "ok"


def test_decode_in_a_freshly_loaded_sprite_fills_constant_states(tmp_path):
    model = make("lstm")
    emulator, result = export(model, tmp_path / "lstm")
    sprite = load_sprite(result.path)
    stored = {entry[0]: entry[1] for entry in sprite["lists"].values()}
    assert stored["_cattorch state cell"] == []
    assert run(emulator, "cattorch decode", [3]) == "ok"
    torch.testing.assert_close(
        logits(emulator), reference_rows(model, [3])[0].double(), atol=TOLERANCE, rtol=0,
    )


def test_max_tokens_limits_decode_and_prefill(tmp_path):
    model = make("gru")
    emulator, _ = export(model, tmp_path / "bounded", max_tokens=3, top_k=2)
    assert emulator.variables["cattorch max context"] == 3
    assert run(emulator, "cattorch prefill", [1, 2]) == "ok"
    assert run(emulator, "cattorch decode", [3]) == "ok"
    before = states(emulator, model)
    assert run(emulator, "cattorch decode", [4]) == "maximum context exceeded"
    assert emulator.lists["cattorch logits"] == [] and emulator.lists["cattorch top k ids"] == []
    assert emulator.variables["cattorch cache length"] == 3
    assert states(emulator, model) == before
    assert run(emulator, "cattorch prefill", [1, 2, 3, 4]) == "prefill exceeds maximum context"
    assert emulator.lists["cattorch logits"] == []
    assert run(emulator, "cattorch prefill", [5, 6, 7]) == "ok"
    torch.testing.assert_close(
        logits(emulator), reference_rows(model, [5, 6, 7])[-1].double(), atol=TOLERANCE, rtol=0,
    )


def test_unbounded_generation_runs_past_any_small_limit(tmp_path):
    model = make("gru")
    emulator, _ = export(model, tmp_path / "unbounded", hidden_prefill=False)
    tokens = [(7 * index + 3) % 16 for index in range(40)]
    assert run(emulator, "cattorch prefill", tokens[:1]) == "ok"
    for token in tokens[1:]:
        assert run(emulator, "cattorch decode", [token]) == "ok"
    assert emulator.variables["cattorch cache length"] == 40
    assert emulator.variables["cattorch max context"] == "Infinity"
    torch.testing.assert_close(
        logits(emulator), reference_rows(model, tokens)[-1].double(), atol=TOLERANCE, rtol=0,
    )


@pytest.mark.parametrize("name", ["gru", "ssm"])
def test_hidden_prefill_skips_the_logits_of_prompt_tokens(tmp_path, name):
    model = make(name)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    counts, results = {}, {}
    for hidden in (False, True):
        emulator, _ = export(model, tmp_path / f"{name}_{hidden}", hidden_prefill=hidden)
        run(emulator, "cattorch init")
        before = sum(emulator.opcode_counts.values())
        assert run(emulator, "cattorch prefill", prompt) == "ok"
        counts[hidden] = sum(emulator.opcode_counts.values()) - before
        prefilled = (logits(emulator), states(emulator, model))
        assert run(emulator, "cattorch decode", [9]) == "ok"
        results[hidden] = (*prefilled, logits(emulator))
        if hidden:
            # A decode computes the logits a hidden prompt token skips.
            before = sum(emulator.opcode_counts.values())
            run(emulator, "cattorch decode", [9])
            full = sum(emulator.opcode_counts.values()) - before
            emulator.variables["_cattorch project output"] = 0
            before = sum(emulator.opcode_counts.values())
            run(emulator, "cattorch decode", [9])
            skipped = full - (sum(emulator.opcode_counts.values()) - before)
    assert torch.equal(results[True][0], results[False][0])
    assert results[True][1] == results[False][1]
    assert torch.equal(results[True][2], results[False][2])
    assert skipped > 0
    # Every prompt token but the last skips that work; the loops differ slightly.
    assert counts[False] - counts[True] == pytest.approx((len(prompt) - 1) * skipped, rel=0.02)


def test_a_hidden_step_still_commits_its_states(tmp_path):
    model = make("ssm")
    emulator, _ = export(model, tmp_path / "ssm")
    run(emulator, "cattorch prefill", [1])
    emulator.variables["_cattorch project output"] = 0
    kept = list(emulator.lists["cattorch logits"])
    assert run(emulator, "cattorch decode", [2]) == "ok"
    assert emulator.lists["cattorch logits"] == kept
    emulator.variables["_cattorch project output"] = 1
    assert run(emulator, "cattorch decode", [3]) == "ok"
    torch.testing.assert_close(
        logits(emulator), reference_rows(model, [1, 2, 3])[-1].double(), atol=TOLERANCE, rtol=0,
    )


def test_layer_sharing_option_behaves_as_for_programs(tmp_path):
    model = make("ssm")
    emulator, _ = export(model, tmp_path / "shared", codegen=CodegenConfig(layer_sharing="auto"))
    run(emulator, "cattorch prefill", [3, 1, 4])
    run(emulator, "cattorch decode", [1])
    torch.testing.assert_close(
        logits(emulator), reference_rows(model, [3, 1, 4, 1])[-1].double(), atol=TOLERANCE, rtol=0,
    )


def test_lifecycle_broadcasts(tmp_path):
    model = make("gru")
    emulator, result = export(model, tmp_path / "gru")
    messages = set(load_sprite(result.path)["broadcasts"].values())
    assert messages == {
        "cattorch init complete", "cattorch reset complete", "cattorch prefill complete",
        "cattorch decode complete",
    }
    for procedure, tokens in (
        ("init", None), ("reset", None), ("prefill", [1, 2, 3]), ("decode", [4]),
    ):
        emulator.broadcasts.clear()
        run(emulator, f"cattorch {procedure}", tokens)
        assert emulator.broadcasts == [f"cattorch {procedure} complete"]


def test_the_reference_check_catches_a_wrong_step(tmp_path):
    class Wrong(GRULM):
        def step(self, token, hidden):
            output, hidden = self.gru(self.embed(token), hidden)
            return self.head(output) + 0.01 * hidden[0], hidden

    torch.manual_seed(0)
    model = Wrong(width=16).eval()
    with pytest.raises(
        UnsupportedModelError,
        match=r"the recurrent step disagrees with forward at logits row 0: max error .* "
              r"tolerance 0.0001 relative; max error over all rows",
    ):
        export(model, tmp_path / "wrong", reference="forward")
    # Without a reference nothing is checked.
    export(model, tmp_path / "unchecked")


def test_reference_must_return_one_row_per_token(tmp_path):
    class LastOnly(GRULM):
        def last(self, tokens):
            return self(tokens)[:, -1]

    with pytest.raises(UnsupportedModelError, match="last returned logits of shape"):
        export(LastOnly().eval(), tmp_path / "last", reference="last")


def test_program_validation():
    state = State("h", torch.zeros(2))
    program = RecurrentGenerationProgram((state,))
    assert (program.step, program.max_tokens, program.hidden_prefill, program.top_k,
            program.reference) == ("step", None, True, None, None)
    assert RecurrentGenerationProgram([state]).states == (state,)
    with pytest.raises(TypeError, match="must be a tuple of State values"):
        RecurrentGenerationProgram(state)
    cases = [
        ((), {}, ValueError, "needs at least one state"),
        (("h",), {}, TypeError, "must contain State values"),
        ((state, State("h", torch.ones(1))), {}, ValueError, "state names must be unique"),
        ((State("log", torch.zeros(0, 2), mode="append", capacity=4),), {}, ValueError,
         "recurrent state 'log' must be a replace state"),
        ((state,), {"step": " "}, ValueError, "recurrent step must be a non-empty string"),
        ((state,), {"reference": ""}, ValueError, "reference method must be a non-empty string"),
        ((state,), {"max_tokens": 0}, ValueError, "max_tokens must be positive"),
        ((state,), {"max_tokens": 2.0}, TypeError, "max_tokens must be an integer or None"),
        ((state,), {"max_tokens": True}, TypeError, "max_tokens must be an integer or None"),
        ((state,), {"hidden_prefill": 1}, TypeError, "hidden_prefill must be a boolean"),
        ((state,), {"top_k": 0}, ValueError, "top_k must be between 1 and 64"),
        ((state,), {"top_k": 65}, ValueError, "top_k must be between 1 and 64"),
        ((state,), {"top_k": "3"}, TypeError, "top_k must be an integer or None"),
    ]
    for arguments, options, error, message in cases:
        with pytest.raises(error, match=message):
            RecurrentGenerationProgram(arguments, **options)
    with pytest.raises(TypeError):
        RecurrentGenerationProgram((state,), "step")
    assert RecurrentGenerationProgram().states is None
    for options in ({"step": "next"}, {"reference": "forward"}):
        with pytest.raises(ValueError, match="step= and reference= need states="):
            RecurrentGenerationProgram(**options)


class Broken(GRULM):
    def only_logits(self, token, hidden):
        return self.step(token, hidden)[0]

    def reshaped(self, token, hidden):
        logits, hidden = self.step(token, hidden)
        return logits, hidden.reshape(1, -1)

    def two_rows(self, token, hidden):
        logits, hidden = self.step(token, hidden)
        return logits.expand(2, 1, -1), hidden

    def one_hot(self, token, hidden):
        embedded = F.one_hot(token, 16).float() @ self.embed.weight
        output, hidden = self.gru(embedded, hidden)
        return self.head(output), hidden


@pytest.mark.parametrize(("options", "error", "message"), [
    ({"step": "missing"}, ValueError, "model has no callable recurrent step 'missing'"),
    ({"reference": "missing"}, ValueError, "model has no callable reference 'missing'"),
    ({"step": "only_logits"}, UnsupportedModelError,
     r"step returned 1 tensors, expected 1 \+ number of states \(2\)"),
    ({"step": "reshaped"}, UnsupportedModelError,
     r"step changes the shape of state 'hidden' from \(1, 1, 8\) to \(1, 8\)"),
    ({"step": "two_rows"}, UnsupportedModelError,
     r"logits must flatten to one vocabulary row, not shape \(2, 1, 16\)"),
    ({"step": "one_hot"}, UnsupportedModelError, "token must index an embedding table"),
    ({"top_k": 17}, ValueError, r"top_k \(17\) exceeds output size \(16\)"),
    ({"dynamic": {0: (1, 4)}}, ValueError, "dynamic= can't be used with a RecurrentGenerationProgram"),
    ({"example_kwargs": {}}, ValueError, "example_kwargs cannot be used with a RecurrentGenerationProgram"),
    ({"frontend": "export"}, ValueError, "RecurrentGenerationProgram is supported only by the FX"),
    ({"calibration_inputs": [torch.zeros(1, 2)]}, ValueError,
     "RecurrentGenerationProgram uses calibration_sequences instead of calibration_inputs"),
    ({"calibration_calls": ()}, ValueError, "calibration_calls require an ExportProgram"),
    ({"calibration_sequences": [torch.zeros(1, 2, dtype=torch.long)]}, ValueError,
     "calibration_sequences can only be used with GPTQ quantization"),
    ({"calibration_sequences": [torch.zeros(3, dtype=torch.long)],
      "quantization": QuantizationConfig(bits=8, method="gptq")}, ValueError,
     r"calibration_sequences must be \[1, length\] token tensors"),
])
def test_export_validation(tmp_path, options, error, message):
    torch.manual_seed(0)
    model = Broken().eval()
    deprecated = options.get("frontend") == "export"
    with pytest.warns(DeprecationWarning) if deprecated else contextlib.nullcontext():
        with pytest.raises(error, match=message):
            export(model, tmp_path / "broken", **options)


def max_error(emulator, model, tokens):
    run(emulator, "cattorch prefill", tokens[:3])
    rows = [logits(emulator)]
    for token in tokens[3:]:
        run(emulator, "cattorch decode", [token])
        rows.append(logits(emulator))
    expected = reference_rows(model, tokens)[2:].double()
    return float((torch.stack(rows) - expected).abs().max()), float(expected.abs().max())


def test_int8_quantized_export_stays_close(tmp_path):
    model = make("ssm")
    emulator, result = export(
        model, tmp_path / "q8", quantization=QuantizationConfig(bits=8, min_quantized_values=16),
    )
    assert result.quantization is not None and result.quantization.tensors
    error, scale = max_error(emulator, model, [1, 5, 9, 13, 17, 21])
    assert error < 0.04 * scale


def test_gptq_calibrates_over_token_sequences(tmp_path):
    model = make("ssm")
    generator = torch.Generator().manual_seed(0)
    sequences = [torch.randint(0, 24, (1, 48), generator=generator) for _ in range(3)]
    emulator, result = export(
        model, tmp_path / "gptq",
        quantization=QuantizationConfig(bits=8, method="gptq", min_quantized_values=16),
        calibration_sequences=sequences,
    )
    methods = {item.method for item in result.quantization.tensors}
    assert "gptq" in methods
    error, scale = max_error(emulator, model, [2, 4, 6, 8, 10, 12])
    assert error < 0.04 * scale


def test_gptq_resets_the_states_between_sequences(monkeypatch, tmp_path):
    from cattorch import program_runtime

    resets = []
    original = program_runtime.ProgramReplay.reset

    def counting(self):
        resets.append(True)
        original(self)

    monkeypatch.setattr(program_runtime.ProgramReplay, "reset", counting)
    model = make("gru")
    sequences = [torch.tensor([[1, 2, 3]]), (torch.tensor([[4, 5]]),)]
    export(
        model, tmp_path / "gptq",
        quantization=QuantizationConfig(bits=8, method="gptq", min_quantized_values=16),
        calibration_sequences=sequences,
    )
    # When the replay is created, and before each sequence.
    assert len(resets) == 1 + len(sequences)


class NormedGRULM(GRULM):
    """A GRU language model with a LayerNorm and GELU feed-forward before its head,
    operations fast mode approximates, and mish, which runs as a decomposition."""

    def __init__(self, vocab=16, width=8):
        super().__init__(vocab, width)
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, 2 * width)
        self.down = nn.Linear(2 * width, width)

    def project(self, output):
        output = self.norm(output)
        return self.head(output + self.down(F.mish(F.gelu(self.up(output)))))

    def forward(self, tokens):
        output, _ = self.gru(self.embed(tokens), self.h0)
        return self.project(output)

    def step(self, token, hidden):
        output, hidden = self.gru(self.embed(token), hidden)
        return self.project(output), hidden


def test_fast_mode_reports_its_approximations(tmp_path):
    torch.manual_seed(6)
    model = NormedGRULM().eval()
    tokens = [1, 2, 3, 4, 5, 6]
    errors = {}
    approximated = {
        "exact": (), "fast": (("aten.gelu.default", 1), ("aten.layer_norm.default", 1)),
    }
    for optimization in ("exact", "fast"):
        emulator, result = export(model, tmp_path / optimization, optimization=optimization)
        errors[optimization], scale = max_error(emulator, model, tokens)
        assert result.approximate_operations == approximated[optimization]
        assert result.fallback_operations == (("aten.mish.default", 1),)
        assert any("aten.mish.default (1 call)" in warning for warning in result.warnings)
    # QuickGELU moves the logits measurably, but stays close.
    assert errors["exact"] < TOLERANCE < errors["fast"] < 0.02 * scale


class Memory(nn.Module):
    """A large state, so the state list is sharded."""

    def __init__(self, size=600, vocab=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, size)
        self.head = nn.Linear(size, vocab)
        self.register_buffer("start", torch.randn(1, size))

    def states(self):
        return (State("memory", self.start.clone()),)

    def forward(self, tokens):
        memory, rows = self.start, []
        for position in range(tokens.shape[1]):
            memory = 0.5 * memory + self.embed(tokens[:, position])
            rows.append(self.head(memory))
        return torch.stack(rows, 1)

    def step(self, token, memory):
        memory = 0.5 * memory + self.embed(token)[:, 0]
        return self.head(memory), memory


def test_a_state_longer_than_a_list_is_sharded(tmp_path, small_list_limit):
    small_list_limit(300)
    torch.manual_seed(4)
    model = Memory().eval()
    emulator, result = export(model, tmp_path / "memory", reference="forward")
    assert "_cattorch state memory" in result.sharded_lists
    run(emulator, "cattorch prefill", [1, 2, 3])
    run(emulator, "cattorch decode", [4])
    torch.testing.assert_close(
        logits(emulator), reference_rows(model, [1, 2, 3, 4])[-1].double(), atol=TOLERANCE, rtol=0,
    )
    run(emulator, "cattorch reset")
    restored = read_logical_list(emulator.lists, "_cattorch state memory")
    assert restored == pytest.approx(model.start.flatten().tolist())


class Readout(Memory):
    """Logits reduced from a product as long as the state, plus a bias."""

    def __init__(self, size=650, vocab=5):
        super().__init__(size, vocab)
        self.weight = nn.Parameter(torch.randn(vocab, size // vocab))
        self.bias = nn.Parameter(torch.randn(vocab))

    def forward(self, tokens):
        memory, rows = self.start, []
        for position in range(tokens.shape[1]):
            memory, row = self._next(memory, self.embed(tokens[:, position]))
            rows.append(row)
        return torch.stack(rows, 1)

    def _next(self, memory, row):
        memory = 0.5 * memory + row
        vocab = self.bias.numel()
        return memory, (memory.reshape(vocab, -1) * self.weight).sum(-1).reshape(1, vocab) + self.bias

    def step(self, token, memory):
        memory, row = self._next(memory, self.embed(token)[:, 0])
        return row, memory


def test_a_logits_list_used_for_a_product_longer_than_a_list_is_sharded(
    tmp_path, small_list_limit,
):
    small_list_limit(300)
    torch.manual_seed(4)
    model = Readout().eval()
    emulator, _ = export(model, tmp_path / "readout", reference="forward")
    assert run(emulator, "cattorch prefill", [1, 2]) == "ok"
    assert run(emulator, "cattorch decode", [3]) == "ok"
    torch.testing.assert_close(
        logits(emulator), reference_rows(model, [1, 2, 3])[-1].double(), atol=TOLERANCE, rtol=0,
    )


def test_a_fixed_table_longer_than_a_list_is_rejected_as_for_plain_exports(
    tmp_path, small_list_limit,
):
    # The storage decoder's table of powers of two can't be sharded.
    small_list_limit(200)
    model = make("gru")
    with pytest.raises(ValueError, match="_cattorch f32 powers.* exceeds Scratch's limit of 200"):
        export(model, tmp_path / "tiny")


def test_build_project_runs_the_generation_demo(tmp_path):
    from test_project import CHARACTERS, char_tokenizer

    torch.manual_seed(5)
    model = GRULM(vocab=len(CHARACTERS)).eval()
    _, result = export(model, tmp_path / "lm")
    tokenizer = CharTokenizer(char_tokenizer()).save(tmp_path / "chars", name="chars")
    project = build_project(
        tmp_path / "demo", result, tokenizer=tokenizer, max_new_tokens=4, temperature=0,
    )
    emulator = ProjectEmulator(project.path, answers=["hi", ""])
    emulator.green_flag()
    ids = [CHARACTERS.index(character) for character in "hi"]
    with torch.no_grad():
        for _ in range(4):
            ids.append(int(model(torch.tensor([ids]))[0, -1].argmax()))
    text = "".join(CHARACTERS[index] for index in ids)
    assert text in emulator.sprites["Demo"].said


def test_verify_rejects_the_program_as_for_generation(tmp_path):
    model = make("gru")
    program = RecurrentGenerationProgram(model.states())
    result = transpile(model, program, tmp_path / "gru")
    with pytest.raises(TypeError):
        verify(model, program, result)
