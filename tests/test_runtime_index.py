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


def test_scatter_reductions_are_rejected(tmp_path):
    class Reduce(nn.Module):
        def forward(self, scores):
            ids = torch.topk(scores, 2, dim=-1)[1]
            return torch.zeros(4, 6).scatter(1, ids, 1.0, reduce="add")

    with pytest.raises(Exception, match="scatter"):
        transpile(Reduce().eval(), torch.randn(4, 6), tmp_path / "reduce")
