# Getting started

## Export a model

Install cattorch from PyPI:

```bash
pip install cattorch
```

Export an evaluation-mode model with inputs that have the shapes the Scratch
project will use:

```python
from pathlib import Path

import torch
import torch.nn as nn

from cattorch import transpile, verify


class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


model = TwoLayerNet().eval()
example = torch.randn(1, 4)

artifact = transpile(model, example, Path("build/two_layer_net"))
print(artifact.path)                 # build/two_layer_net.sprite3
print(artifact.archive_bytes)
print(artifact.inputs, artifact.output)

check = verify(model, example, artifact)
assert check.passed, check
```

`transpile` creates missing parent directories and appends `.sprite3` if the
path does not already have that suffix. The Scratch sprite name defaults to
the output filename stem; pass `name="My model"` to choose it independently.

The returned `TranspileResult` describes the artifact without requiring code
to parse the zip file. It includes the final path and sprite name, archive and
expanded-JSON sizes, block and list counts, sharded list names, warnings,
input/output tensor specifications, and generated procedure names.

## Use the sprite in Scratch

1. Import the generated `.sprite3` into a Scratch 3 project.
2. Put flattened tensor values in `input`. For multiple arguments, use
   `input_1`, `input_2`, and so on.
3. Run the `cattorch forward` custom block.
4. Read the flattened tensor result from `output` and reshape it according to
   `artifact.output.shape` in the surrounding project if needed.

`cattorch forward` initializes the sprite automatically on its first call.
Calling `cattorch init` explicitly is useful when startup decoding should be
kept out of a timed region. Both blocks run without screen refresh.

Before saving a project that has run, call `cattorch prepare for save`. It
clears decoded weights and runtime data while preserving compressed payloads,
preventing Scratch from serializing both forms of the weights.

## Export options

```python
from cattorch import FastConfig, StorageConfig, transpile

artifact = transpile(
    model,
    example,
    "build/two_layer_net_fast",
    name="Two-layer classifier",
    optimization="fast",
    fast_config=FastConfig(),
    storage=StorageConfig(precision="float16"),
    sig_figs=6,
)
```

Kernel optimization and serialized storage precision are independent choices.
See [optimization modes](optimization.md) and [storage](storage.md) before
selecting lossy settings.
