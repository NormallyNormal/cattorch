"""Operations that mix values along a variable-length axis."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cattorch import BoundedDimension, UnsupportedModelError, entrypoint, transpile, verify
from cattorch import ProgramCall


class Mixing(nn.Module):
    """A linear layer, so the mixing operation reads computed values."""

    def __init__(self, function, features=4):
        super().__init__()
        self.function = function
        self.linear = nn.Linear(features, features)

    def forward(self, value):
        return self.function(self.linear(value))


def _export(tmp_path, model, example, dynamic, lengths, axis=0):
    """Export, then check against PyTorch at lengths the export check skips."""
    model = model.eval()
    result = transpile(model, example, tmp_path / "mixing", dynamic=dynamic)
    for length in lengths:
        shape = list(example.shape)
        shape[axis] = length
        value = torch.randn(shape) if example.is_floating_point() else torch.randint(0, 10, shape)
        assert verify(model, value, result).passed, length
    return result


ROWS = {
    "sum": lambda x: x.sum(0),
    "mean": lambda x: x.mean(0),
    "centered": lambda x: x - x.mean(0, keepdim=True),
    "amax": lambda x: x.amax(0),
    "min": lambda x: x.min(0).values,
    "prod": lambda x: (x.sigmoid() + 0.5).prod(0),
    "all dimensions": lambda x: x.sum(),
    "two dimensions": lambda x: x.mean((0, 1)),
    "softmax": lambda x: x.softmax(0),
    "log softmax": lambda x: x.log_softmax(0),
    "logsumexp": lambda x: x.logsumexp(0),
    "variance": lambda x: x.var(0),
    "standard deviation": lambda x: x.std(0, keepdim=True),
    "mean from the shape": lambda x: x.sum(0) / x.shape[0],
    "reciprocal of the length": lambda x: x.sum(0, keepdim=True) * (1.0 / x.shape[0]),
    "softmax along columns": lambda x: x.T.softmax(-1),
    "maximum along columns": lambda x: x.T.amax(1),
    "contraction": lambda x: x.T @ x,
    "vector contraction": lambda x: x[:, 0] @ x,
    "columns": lambda x: torch.ones(2, 4) @ x.T,
    "attention pooling": lambda x: (x.softmax(0) * x).sum(0),
    "first": lambda x: x[0],
    "last": lambda x: x[-1] - x[:1],
    "cat before": lambda x: torch.cat([torch.zeros(2, 4), x]).sum(0),
    "cat around": lambda x: torch.cat([torch.ones(1, 4), x, -x[-1:]]).softmax(0),
}


# The trace has one item, which leaves a variance no degrees of freedom.
@pytest.mark.filterwarnings("ignore:.*degrees of freedom")
@pytest.mark.parametrize("name", ROWS)
def test_mixing_along_a_variable_batch(tmp_path, name):
    torch.manual_seed(0)
    # PyTorch's standard deviation of one item is NaN, which Scratch's square
    # root reads as 0.
    shortest = 2 if name == "standard deviation" else 1
    _export(tmp_path, Mixing(ROWS[name]), torch.randn(5, 4), {0: (shortest, 12)}, (7, 12))


def test_a_slice_past_the_traced_item_is_rejected(tmp_path):
    with pytest.raises(UnsupportedModelError, match="traces it with 1"):
        transpile(
            Mixing(lambda x: x[-2:].sum(0)).eval(), torch.randn(5, 4), tmp_path / "two",
            dynamic={0: (2, 12)},
        )


def test_a_slice_that_drops_the_traced_item_is_rejected(tmp_path):
    with pytest.raises(UnsupportedModelError, match="leaves no items"):
        transpile(
            Mixing(lambda x: x[1:].sum(0)).eval(), torch.randn(5, 4), tmp_path / "drop",
            dynamic={0: (2, 12)},
        )


def test_a_cat_along_the_extent_widens_the_output_bounds(tmp_path):
    torch.manual_seed(0)
    model = Mixing(lambda x: torch.cat([torch.zeros(1, 4), x, torch.ones(2, 4)]))
    result = _export(tmp_path, model, torch.randn(3, 4), {0: (1, 10)}, (6,))
    assert result.output.shape == (BoundedDimension("input", 4, 13), 4)
    assert result.output.max_numel == 52


def test_mean_pooling_over_tokens(tmp_path):
    class Bag(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10, 6)
            self.head = nn.Linear(6, 3)

        def forward(self, tokens):
            return self.head(self.embedding(tokens).mean(1))

    torch.manual_seed(0)
    _export(tmp_path, Bag(), torch.randint(0, 10, (1, 4)), {0: {1: (1, 16)}}, (9, 16), axis=1)


def test_attention_from_a_query_over_a_variable_memory(tmp_path):
    class Pool(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.MultiheadAttention(8, 2, batch_first=True)
            self.query = nn.Parameter(torch.randn(1, 1, 8))

        def forward(self, memory):
            return self.attention(self.query, memory, memory, need_weights=False)[0]

    torch.manual_seed(0)
    _export(tmp_path, Pool(), torch.randn(1, 5, 8), {0: {1: (1, 12)}}, (9,), axis=1)


def test_a_class_token_joined_to_a_sequence(tmp_path):
    class Classify(nn.Module):
        def __init__(self):
            super().__init__()
            self.token = nn.Parameter(torch.randn(1, 1, 4))
            self.linear = nn.Linear(4, 6)

        def forward(self, sequence):
            hidden = self.linear(torch.cat([self.token, sequence], 1))
            return hidden.softmax(1).amax(1) + hidden[:, 0] + hidden[:, -1]

    torch.manual_seed(0)
    _export(tmp_path, Classify(), torch.randn(1, 5, 4), {0: {1: (1, 12)}}, (8,), axis=1)


def test_a_fixed_operand_with_the_traced_shape_broadcasts(tmp_path):
    class Shift(nn.Module):
        def __init__(self):
            super().__init__()
            self.shift = nn.Parameter(torch.randn(1, 4))

        def forward(self, value):
            return (value + self.shift) * self.shift

    torch.manual_seed(0)
    _export(tmp_path, Shift(), torch.randn(5, 4), {0: (1, 8)}, (8,))


def test_self_attention_over_the_extent_is_rejected(tmp_path):
    layer = nn.TransformerEncoderLayer(8, 2, 16, dropout=0.0, batch_first=True).eval()
    with pytest.raises(UnsupportedModelError, match="two runtime dimensions"):
        transpile(layer, torch.randn(1, 5, 8), tmp_path / "encoder", dynamic={0: {1: (1, 8)}})


def test_a_running_sum_is_caught_by_the_length_check(tmp_path):
    with pytest.raises(UnsupportedModelError, match="uses the length itself"):
        transpile(
            Mixing(lambda x: x.cumsum(0)).eval(), torch.randn(5, 4), tmp_path / "running",
            dynamic={0: (1, 8)},
        )


class Memory(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("seen", torch.zeros(0, 3))
        self.query = nn.Parameter(torch.randn(3))

    @entrypoint(example=lambda: (torch.ones(3),), capacity={"seen": 6})
    def remember(self, row):
        self.seen = torch.cat([self.seen, row.reshape(1, 3)])
        weights = (self.seen @ self.query).softmax(0)
        return weights @ self.seen, self.seen.mean(0)


def test_an_entry_point_reads_the_state_it_appends_to(tmp_path):
    torch.manual_seed(0)
    model = Memory().eval()
    result = transpile(model, None, tmp_path / "memory")
    calls = tuple(ProgramCall("remember", (torch.randn(3),)) for _ in range(6))
    assert verify(model, None, result, calls=calls).passed



def test_other_functions_of_the_length_are_rejected(tmp_path):
    with pytest.raises(UnsupportedModelError, match="a \\+ b \\* n and a \\+ b / n"):
        transpile(
            Mixing(lambda x: x.sum(0) * (x.shape[0] * x.shape[0])).eval(), torch.randn(1, 4),
            tmp_path / "square", dynamic={0: (1, 8)},
        )


def test_mixing_integer_items_with_a_one_item_example_is_rejected(tmp_path):
    class Sorted(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10, 4)

        def forward(self, tokens):
            return torch.sort(self.embedding(tokens), dim=0).values

    torch.manual_seed(0)
    with pytest.raises(UnsupportedModelError):
        transpile(Sorted().eval(), torch.tensor([3]), tmp_path / "sorted", dynamic={0: (1, 8)})
