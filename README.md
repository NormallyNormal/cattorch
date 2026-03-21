# cattorch

Export PyTorch neural networks to [Scratch](https://scratch.mit.edu) sprites.

cattorch transpiles a `torch.nn.Module` into a `.sprite3` file that can be
imported directly into any Scratch project. The generated sprite uses only
standard Scratch blocks — no extensions or modifications required.

## Install

```bash
pip install cattorch
```

Requires Python 3.10+ and PyTorch 2.0+.

## Usage

```python
import torch
import torch.nn as nn
from cattorch import transpile

class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = TwoLayerNet()
transpile(model, torch.Size([1, 4]), "two_layer_net")
# => two_layer_net.sprite3
```

In Scratch, the sprite reads its input from a list called `input` and writes
results to a list called `output`. Click the green flag to run inference.

## Supported operations

| Category | Operations |
|---|---|
| Linear layers | `nn.Linear` (with and without bias) |
| Matrix multiply | `@` / `torch.matmul` |
| Activations | ReLU, Sigmoid, Tanh, GELU, SiLU, LeakyReLU, ELU |
| Normalization | `nn.LayerNorm` |
| Softmax | `F.softmax` (any dim) |
| Arithmetic | tensor add, scalar multiply, scalar divide |
| Shape | view, reshape, flatten, contiguous, clone (no-ops on flat data) |
| Transpose | `transpose`, `permute`, `.T` (arbitrary dimensions) |

These are sufficient for architectures like MLPs and single-head transformers
(including full pre-norm transformer blocks with residual connections).

## Scratch limits

- **Project size**: Scratch limits projects to 5 MB. cattorch warns at 4 MB
  and errors at 5 MB.
- **List length**: Scratch lists can hold at most 200,000 items. cattorch
  raises an error if any weight tensor or intermediate list exceeds this.

## License

MIT
