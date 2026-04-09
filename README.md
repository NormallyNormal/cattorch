# cattorch

Export PyTorch neural networks to [Scratch](https://scratch.mit.edu) sprites.

cattorch transpiles a `torch.nn.Module` into a `.sprite3` file that can be
imported directly into any Scratch project. The generated sprite uses only
standard Scratch blocks, so no extensions or modifications are required.

cattorch does not export training scripts, you will need to train your model
with torch before exporting to a Scratch sprite.

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
# train your model first! then:
# transpile(model, example input, sprite name)
transpile(model, torch.randn(1, 4), "two_layer_net")
# => two_layer_net.sprite3

# optionally reduce file size by rounding weights
transpile(model, torch.randn(1, 4), "two_layer_net", sig_figs=6)
```

In Scratch, the sprite reads its input from a list called `input` and writes
results to a list called `output`. It is up to you to add logic to fill the
input tensor and run the generated code blocks.

If the model takes multiple input tensors, the additional inputs are named
`input_1`, `input_2`, etc.

## Supported operations

| Category | Operations |
|---|---|
| Linear layers | `nn.Linear` (with and without bias) |
| Matrix multiply | `@` / `torch.matmul` |
| Activations | ReLU, Sigmoid, Tanh, GELU (tanh approx. only), SiLU, LeakyReLU, ELU |
| Normalization | `nn.LayerNorm`, `nn.RMSNorm` |
| Softmax | `F.softmax` (any dim) |
| Embedding | `nn.Embedding` |
| Masking | `masked_fill` (for causal attention masks via `register_buffer`) |
| Arithmetic | tensor add, tensor subtract, tensor multiply, scalar multiply, scalar divide, negate |
| Shape | view, reshape, flatten, contiguous, clone (no-ops on flat data) |
| Transpose | `transpose`, `permute`, `.T` (arbitrary dimensions) |
| Split / Chunk | `split`, `split_with_sizes`, `chunk` |
| Concatenation | `torch.cat` (any dim, any number of inputs) |
| Slice | `tensor[:n]` style slicing along any dimension |

These are sufficient for architectures like MLPs and transformer LLMs,
including multi-head attention, combined QKV projections, rotary position
embeddings (RoPE), causal masking, pre-norm blocks with residual connections,
and SwiGLU-style gating. CNN and RNN support is planned for the future.

## Scratch limits

- **Project size**: Scratch limits projects to 5 MB. cattorch warns at 4 MB
  and errors at 5 MB.
- **List length**: Scratch lists can hold at most 200,000 items. cattorch
  raises an error if any weight tensor or intermediate list exceeds this.

## License

MIT
