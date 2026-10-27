"""Classify points by quadrant with a small MLP, and export it to Scratch.

    python examples/mlp.py

writes ``build/quadrants.sprite3`` and a runnable ``build/quadrants.sb3``:
edit the two numbers in the "quadrants input" list and press space to see the
predicted quadrant. The same model exports from the command line with

    cattorch export examples/mlp.py:trained --project --labels "top right,top left,bottom left,bottom right"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

import cattorch

LABELS = ["top right", "top left", "bottom left", "bottom right"]


@cattorch.example(lambda: (torch.tensor([[0.5, -0.25]]),))
class Quadrants(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 4))

    def forward(self, point):
        return self.layers(point)


def quadrant(points):
    right, top = points[:, 0] > 0, points[:, 1] > 0
    return torch.where(top, torch.where(right, 0, 1), torch.where(right, 3, 2))


def trained(steps: int = 300) -> Quadrants:
    torch.manual_seed(0)
    model = Quadrants()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(steps):
        points = torch.randn(64, 2)
        loss = nn.functional.cross_entropy(model(points), quadrant(points))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


def main(output: Path = Path("build"), steps: int = 300) -> cattorch.ProjectResult:
    model = trained(steps)
    # The @example marker supplies the inputs.
    result = cattorch.transpile(model, None, output / "quadrants")
    assert cattorch.verify(model, None, result).passed
    example = torch.tensor([[0.5, -0.25]])
    return cattorch.build_project(
        output / "quadrants", result, example_inputs=example, labels=LABELS,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("build"))
    print(main(parser.parse_args().output).path)
