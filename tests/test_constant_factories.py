"""Constants a model builds in forward from factories export as constants."""

import pytest
import torch

from cattorch import transpile, verify


class _Constant(torch.nn.Module):
    def __init__(self, make):
        super().__init__()
        self.make = make

    def forward(self, value):
        return value + self.make().to(value.dtype).reshape(-1)[:64]


_FACTORIES = {
    "eye(512, 64)": lambda: torch.eye(512, 64),
    "eye(300)": lambda: torch.eye(300),
    "eye(8, 64)": lambda: torch.eye(8, 64),
    "eye(256, 64, dtype=long)": lambda: torch.eye(256, 64, dtype=torch.long),
    "eye(300, 64, dtype=bool)": lambda: torch.eye(300, 64, dtype=torch.bool),
    "diag": lambda: torch.diag(torch.arange(64.0)),
    "diag_embed": lambda: torch.diag_embed(torch.ones(1, 64)),
    "tril": lambda: torch.tril(torch.ones(64, 64)),
    "triu": lambda: torch.triu(torch.ones(64, 64), 1),
    "ones": lambda: torch.ones(1, 64),
    "zeros": lambda: torch.zeros(1, 64),
    "full": lambda: torch.full((1, 64), 2.5),
    "arange": lambda: torch.arange(0, 128, 2.0),
    "linspace": lambda: torch.linspace(0, 1, 64),
}


@pytest.mark.parametrize("name", list(_FACTORIES))
def test_constant_factory_in_forward_exports(tmp_path, name):
    model = _Constant(_FACTORIES[name]).eval()
    example = torch.randn(1, 64)
    result = transpile(model, example, tmp_path / "constant")
    assert verify(model, example, result).passed
