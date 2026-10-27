"""gather and scatter whose indices are computed at run time."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cattorch import transpile, verify


class TopKGather(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, scores):
        ids = torch.topk(scores, 2, dim=self.dim)[1]
        return scores.exp().gather(self.dim, ids)


class GroupMask(nn.Module):
    """DeepSeek-V3's group mask: scatter ones at the chosen groups."""

    def forward(self, scores):
        ids = torch.topk(scores, 2, dim=-1)[1]
        mask = torch.zeros_like(scores).scatter(1, ids, 1.0)
        return scores.masked_fill(mask == 0, -1.0)


class ScatterSource(nn.Module):
    def forward(self, scores):
        ids = torch.topk(scores, 3, dim=-1)[1]
        picked = scores.gather(-1, ids)
        return torch.full((4, 6), 0.5).scatter(-1, ids, picked * 2)


class ScatterAlongRows(nn.Module):
    def forward(self, scores):
        ids = torch.topk(scores, 1, dim=0)[1]
        return torch.zeros(4, 6).scatter(0, ids, scores[:1] + 1)


@pytest.mark.parametrize("model", [
    TopKGather(-1), TopKGather(0), GroupMask(), ScatterSource(), ScatterAlongRows(),
], ids=lambda model: type(model).__name__ + str(getattr(model, "dim", "")))
def test_runtime_index_gather_and_scatter_match_pytorch(tmp_path, model):
    torch.manual_seed(0)
    value = torch.randn(4, 6)
    result = transpile(model.eval(), value, tmp_path / "index")
    assert verify(model, value, result).passed


@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
@pytest.mark.parametrize("include_self", [True, False])
def test_scatter_reductions(tmp_path, reduce, include_self):
    """Duplicate targets combine; untouched positions keep the input."""
    class Reduce(nn.Module):
        def forward(self, scores):
            ids = torch.topk(scores, 2, dim=-1)[1] % 3
            return scores.scatter_reduce(1, ids, scores[:, :2] * 2, reduce, include_self=include_self)

    value = torch.randn(4, 6)
    result = transpile(Reduce().eval(), value, tmp_path / "reduce")
    assert verify(Reduce().eval(), value, result).passed


def test_legacy_scatter_reduce_argument(tmp_path):
    class Reduce(nn.Module):
        def forward(self, scores):
            ids = torch.topk(scores, 2, dim=-1)[1]
            return torch.zeros(4, 6).scatter(1, ids, 1.0, reduce="add")

    value = torch.randn(4, 6)
    assert verify(Reduce().eval(), value, transpile(Reduce().eval(), value, tmp_path / "add")).passed


class Indexing(nn.Module):
    """Indices computed from the input, as in a router or a lookup."""

    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, value, scores):
        return self.operation(value, torch.topk(scores, 3, dim=-1)[1])


INDEXING = {
    "index_select": lambda x, ids: x.index_select(1, ids),
    "rows": lambda x, ids: x[ids],
    "columns": lambda x, ids: x[:, ids],
    "pairs": lambda x, ids: x[ids, ids.flip(0)],
    "non-adjacent": lambda x, ids: x[ids, :, ids % 2],
    "take": lambda x, ids: torch.take(x, ids * 7),
    "index_add": lambda x, ids: x.index_add(0, ids, torch.ones(3, 5, 2), alpha=2),
    "index_fill": lambda x, ids: x.index_fill(1, ids, -1.5),
    "index_put": lambda x, ids: x.index_put((ids, ids), torch.ones(3, 2) * 4),
    "index_put accumulate": lambda x, ids: x.index_put((ids,), torch.ones(5, 2), accumulate=True),
    "setitem": lambda x, ids: x.clone().index_put_((ids,), torch.tensor(0.0)),
    "mask assignment": lambda x, ids: x.index_put((x[:, 0, 0] > 0,), torch.ones(5, 2)),
}


@pytest.mark.parametrize("name", sorted(INDEXING))
def test_input_dependent_indexing(tmp_path, name):
    torch.manual_seed(4)
    model = Indexing(INDEXING[name]).eval()
    inputs = (torch.randn(5, 5, 2), torch.randn(5))
    result = transpile(model, inputs, tmp_path / "indexing")
    assert verify(model, inputs, result).passed
