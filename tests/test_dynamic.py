"""Bounded variable-length inputs: ``transpile(..., dynamic=...)``."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cattorch import BoundedDimension, UnsupportedModelError, entrypoint, transpile, verify
from cattorch.experimental import ProgramCall


class Lambda(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, value):
        return self.function(value)


def _check(tmp_path, model, example, dynamic, lengths, axis=0):
    model = model.eval()
    result = transpile(model, example, tmp_path / "dynamic", dynamic=dynamic)
    for length in lengths:
        shape = list(example.shape)
        shape[axis] = length
        value = (
            torch.randn(shape) if example.is_floating_point()
            else torch.randint(0, 10, shape)
        )
        assert verify(model, value, result).passed, length
    return result


def test_a_variable_batch_has_a_bounded_interface(tmp_path):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.LayerNorm(8), nn.Linear(8, 3))
    result = _check(tmp_path, model, torch.randn(5, 4), {0: (1, 16)}, (1, 3, 16))
    assert result.inputs[0].shape == (BoundedDimension("input", 1, 16), 4)
    assert result.output.shape == (BoundedDimension("input", 1, 16), 3)
    assert result.output.max_numel == 48


ROW_BY_ROW = {
    "sum": lambda x: x.sum(-1),
    "topk": lambda x: x.topk(2, -1).values,
    "pad": lambda x: F.pad(x, (1, 2)),
    "flip": lambda x: x.flip(-1),
    "slice": lambda x: x[:, 1:4],
    "cumsum": lambda x: x.cumsum(-1),
    "cat": lambda x: torch.cat([x, x * 2], -1),
    "group norm": lambda x: F.group_norm(x.unsqueeze(-1), 2),
    "rms norm": lambda x: F.rms_norm(x, (6,)),
}


@pytest.mark.parametrize("name", sorted(ROW_BY_ROW))
def test_operations_without_runtime_length_kernels_run_row_by_row(tmp_path, name):
    torch.manual_seed(1)
    _check(tmp_path, Lambda(ROW_BY_ROW[name]), torch.randn(3, 6), {0: (1, 8)}, (2, 8))


def test_a_token_dimension_after_a_size_one_batch(tmp_path):
    torch.manual_seed(2)
    model = nn.Sequential(nn.Embedding(10, 8), nn.Linear(8, 10), nn.LogSoftmax(-1))
    result = _check(
        tmp_path, model, torch.tensor([[1, 2, 3]]), {0: {1: (1, 12)}}, (1, 5, 12), axis=1,
    )
    assert result.inputs[0].shape == (1, BoundedDimension("input", 1, 12))


def test_a_model_that_reorders_its_batch_runs_once_per_item(tmp_path):
    torch.manual_seed(3)
    layer = nn.TransformerEncoderLayer(8, 2, 16, dropout=0.0, batch_first=True)
    result = _check(tmp_path, layer, torch.randn(2, 4, 8), {0: (1, 5)}, (1, 4))
    assert any("once per item" in warning for warning in result.warnings)


def test_a_model_using_the_length_itself_is_rejected(tmp_path):
    class Positions(nn.Module):
        def __init__(self):
            super().__init__()
            self.tokens = nn.Embedding(10, 4)
            self.positions = nn.Embedding(16, 4)

        def forward(self, tokens):
            return self.tokens(tokens) + self.positions(torch.arange(tokens.shape[1]))

    with pytest.raises(UnsupportedModelError, match="uses the length itself"):
        transpile(Positions().eval(), torch.tensor([[1, 2, 3]]), tmp_path / "p", dynamic={0: {1: (1, 16)}})


def test_a_model_with_two_runtime_dimensions_is_rejected(tmp_path):
    with pytest.raises(UnsupportedModelError, match="two runtime dimensions"):
        transpile(
            Lambda(lambda x: x @ x.T).eval(), torch.randn(3, 4), tmp_path / "m",
            dynamic={0: (1, 8)},
        )


@pytest.mark.parametrize(("dynamic", "error"), [
    ({3: (1, 4)}, ValueError),
    ({0: (0, 4)}, ValueError),
    ({0: (5, 4)}, ValueError),
    ({0: {2: (1, 4)}}, ValueError),
    ({"missing": (1, 4)}, ValueError),
    ([(1, 4)], TypeError),
])
def test_dynamic_arguments_are_validated(tmp_path, dynamic, error):
    with pytest.raises(error):
        transpile(nn.Linear(4, 2).eval(), torch.randn(2, 4), tmp_path / "v", dynamic=dynamic)


def test_a_keyword_input_can_be_dynamic(tmp_path):
    class Gated(nn.Module):
        def forward(self, value, *, gate):
            return value * torch.sigmoid(gate)

    model = Gated().eval()
    result = transpile(
        model, (torch.randn(1, 3),), tmp_path / "k",
        example_kwargs={"gate": torch.randn(2, 3)}, dynamic={"gate": (1, 6)},
    )
    assert result.inputs[1].shape[0] == BoundedDimension("input_1", 1, 6)
    report = verify(model, (torch.randn(1, 3),), result, example_kwargs={"gate": torch.randn(5, 3)})
    assert report.passed


def test_entry_points_take_dynamic_inputs(tmp_path):
    class Scorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 2)
            self.register_buffer("calls", torch.zeros(1))

        @entrypoint(example=lambda: (torch.ones(4, 3),), dynamic={"rows": (1, 10)})
        def score(self, rows):
            self.calls += 1
            return torch.softmax(self.linear(rows), -1) * self.calls

    torch.manual_seed(4)
    model = Scorer().eval()
    result = transpile(model, None, tmp_path / "scorer")
    assert result.entrypoints[0].inputs[0].shape == (BoundedDimension("rows", 1, 10), 3)
    calls = (ProgramCall("score", (torch.randn(2, 3),)), ProgramCall("score", (torch.randn(9, 3),)))
    assert verify(model, None, result, calls=calls).passed


def test_an_entry_point_that_cant_follow_the_length_runs_per_item(tmp_path):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.TransformerEncoderLayer(8, 2, 16, dropout=0.0, batch_first=True)

        @entrypoint(example=lambda: (torch.randn(3, 4, 8),), dynamic={"sequences": (1, 6)})
        def encode(self, sequences):
            return self.layer(sequences)

    torch.manual_seed(0)
    model = Encoder().eval()
    result = transpile(model, None, tmp_path / "encoder")
    assert any("runs once per item" in warning for warning in result.warnings)
    assert result.entrypoints[0].outputs[0].shape == (BoundedDimension("sequences", 1, 6), 4, 8)
    calls = tuple(ProgramCall("encode", (torch.randn(length, 4, 8),)) for length in (1, 5, 2))
    assert verify(model, None, result, calls=calls).passed


def test_an_entry_point_that_updates_state_doesnt_run_per_item(tmp_path):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.TransformerEncoderLayer(8, 2, 16, dropout=0.0, batch_first=True)
            self.register_buffer("calls", torch.zeros(1))

        @entrypoint(example=lambda: (torch.randn(3, 4, 8),), dynamic={"sequences": (1, 6)})
        def encode(self, sequences):
            self.calls += 1
            return self.layer(sequences) * self.calls

    with pytest.raises(UnsupportedModelError):
        transpile(Encoder().eval(), None, tmp_path / "encoder")


def test_an_entry_point_using_the_length_itself_is_rejected(tmp_path):
    class Positions(nn.Module):
        def __init__(self):
            super().__init__()
            self.table = nn.Parameter(torch.randn(10, 3))

        @entrypoint(example=lambda: (torch.randn(4, 3),), dynamic={"rows": (1, 8)})
        def add(self, rows):
            return rows + self.table[torch.arange(rows.shape[0])]

    with pytest.raises(UnsupportedModelError, match="input 'rows' of entry point 'add'"):
        transpile(Positions().eval(), None, tmp_path / "positions")
