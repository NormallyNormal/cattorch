"""Tell horizontal from vertical lines in 8x8 images with a small CNN.

    python examples/cnn.py

writes ``build/lines.sprite3`` and a runnable ``build/lines.sb3``. Its
"lines input" list holds the 64 pixels, row by row; set some to 1 and press
space to classify the picture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

import cattorch

LABELS = ["horizontal", "vertical"]


class Lines(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(4, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classify = nn.Linear(8, 2)

    def forward(self, image):
        return self.classify(self.features(image).flatten(1))


def pictures(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Images with one full line each, plus a little noise."""
    labels = torch.randint(0, 2, (count,))
    images = torch.rand(count, 1, 8, 8) * 0.2
    positions = torch.randint(0, 8, (count,))
    for index, (label, position) in enumerate(zip(labels, positions)):
        if label == 0:
            images[index, 0, position, :] = 1
        else:
            images[index, 0, :, position] = 1
    return images, labels


def trained(steps: int = 200) -> Lines:
    torch.manual_seed(0)
    model = Lines()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        images, labels = pictures(32)
        loss = nn.functional.cross_entropy(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


def main(output: Path = Path("build"), steps: int = 200) -> cattorch.ProjectResult:
    model = trained(steps)
    example = pictures(1)[0]
    result = cattorch.transpile(model, example, output / "lines")
    assert cattorch.verify(model, example, result).passed
    return cattorch.build_project(
        output / "lines", result, example_inputs=example, labels=LABELS,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("build"))
    print(main(parser.parse_args().output).path)
