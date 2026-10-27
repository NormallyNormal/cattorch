"""Recurrent layers: looped sequence kernels, cells, and step entry points."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cattorch import entrypoint, transpile, verify
from cattorch.experimental import ProgramCall


class Called(nn.Module):
    """Return everything a recurrent module returns, so verify checks it all."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, value, *state):
        return self.module(value, *state)


class LSTMWithState(nn.Module):
    def __init__(self, lstm):
        super().__init__()
        self.lstm = lstm

    def forward(self, value, hidden, cell):
        return self.lstm(value, (hidden, cell))


SEQUENCES = {
    "gru": lambda: (nn.GRU(3, 5, batch_first=True), (torch.randn(2, 4, 3),)),
    "stacked bidirectional gru": lambda: (
        nn.GRU(3, 4, num_layers=2, bidirectional=True), (torch.randn(4, 2, 3),),
    ),
    "lstm with initial state": lambda: (
        LSTMWithState(nn.LSTM(3, 5)),
        (torch.randn(4, 1, 3), torch.randn(1, 1, 5), torch.randn(1, 1, 5)),
    ),
    "projected bidirectional lstm": lambda: (
        nn.LSTM(3, 5, proj_size=2, bidirectional=True, batch_first=True),
        (torch.randn(1, 4, 3),),
    ),
    "unbatched relu rnn": lambda: (nn.RNN(3, 5, nonlinearity="relu", bias=False),
                                   (torch.randn(4, 3),)),
    "gru cell": lambda: (nn.GRUCell(3, 5), (torch.randn(2, 3), torch.randn(2, 5))),
    "lstm cell": lambda: (nn.LSTMCell(3, 5), (torch.randn(2, 3),)),
    "unbatched rnn cell": lambda: (nn.RNNCell(3, 5), (torch.randn(3),)),
}


@pytest.mark.parametrize("name", sorted(SEQUENCES))
def test_recurrent_modules_match(tmp_path, name):
    torch.manual_seed(0)
    module, example = SEQUENCES[name]()
    model = Called(module).eval()
    result = transpile(model, example, tmp_path / "recurrent")
    assert verify(model, example, result).passed
    assert result.fallback_operations == ()


def test_nested_example_inputs_are_rejected_clearly(tmp_path):
    model = Called(nn.LSTM(3, 5)).eval()
    state = (torch.zeros(1, 1, 5), torch.zeros(1, 1, 5))
    with pytest.raises(TypeError, match="example input 1 is a tuple"):
        transpile(model, (torch.randn(4, 1, 3), state), tmp_path / "nested")


def test_the_sprite_does_not_grow_with_the_sequence(tmp_path):
    torch.manual_seed(1)
    model = Called(nn.LSTM(3, 4, batch_first=True)).eval()
    sizes = []
    for steps in (3, 60):
        example = torch.randn(1, steps, 3)
        result = transpile(model, example, tmp_path / f"steps{steps}")
        assert verify(model, example, result).passed
        sizes.append(result.path.stat().st_size)
    assert abs(sizes[0] - sizes[1]) < 200


class Classifier(nn.Module):
    def __init__(self, recurrent, final):
        super().__init__()
        self.embedding = nn.Embedding(12, 3)
        self.recurrent = recurrent
        self.final = final
        width = recurrent.hidden_size * (2 if recurrent.bidirectional and not final else 1)
        self.head = nn.Linear(width, 2)

    def forward(self, tokens):
        outputs, state = self.recurrent(self.embedding(tokens))
        if isinstance(state, tuple):
            state = state[0]
        return self.head(state[-1] if self.final else outputs)


@pytest.mark.parametrize(("recurrent", "final"), [
    (lambda: nn.GRU(3, 4, batch_first=True), True),
    (lambda: nn.LSTM(3, 4, batch_first=True, bidirectional=True, num_layers=2), False),
], ids=["final state", "every step"])
def test_a_variable_length_sequence(tmp_path, recurrent, final):
    torch.manual_seed(2)
    model = Classifier(recurrent(), final).eval()
    result = transpile(
        model, torch.randint(0, 12, (1, 4)), tmp_path / "classifier", dynamic={0: {1: (1, 24)}},
    )
    for steps in (1, 9, 24):
        assert verify(model, torch.randint(0, 12, (1, steps)), result).passed, steps
    if final:
        # The final state has no runtime extent.
        assert result.output.shape == (1, 2)


class Streaming(nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.cell = cell
        self.head = nn.Linear(5, 2)
        self.register_buffer("hidden", torch.zeros(1, 5))
        self.register_buffer("memory", torch.zeros(1, 5))

    @entrypoint(example=lambda: (torch.randn(1, 3),))
    def step(self, value):
        if isinstance(self.cell, nn.LSTMCell):
            hidden, memory = self.cell(value, (self.hidden, self.memory))
            self.memory.copy_(memory)
        else:
            hidden = self.cell(value, self.hidden)
        self.hidden.copy_(hidden)
        return self.head(hidden)


@pytest.mark.parametrize("cell", [nn.GRUCell, nn.LSTMCell], ids=["gru", "lstm"])
def test_a_step_method_keeps_its_hidden_state(tmp_path, cell):
    torch.manual_seed(3)
    model = Streaming(cell(3, 5)).eval()
    result = transpile(model, None, tmp_path / "streaming")
    calls = tuple(ProgramCall("step", (torch.randn(1, 3),)) for _ in range(4))
    assert verify(model, None, result, calls=calls).passed
