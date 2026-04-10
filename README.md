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

cattorch uses `torch.export` under the hood, which requires a single code path
with no data-dependent control flow. If your model has conditional returns
(e.g. returning loss during training), add an inference-only forward method:

```python
# won't work: conditional return
def forward(self, x, targets=None):
    logits = self.head(x)
    if targets is None:
        return logits
    return logits, F.cross_entropy(logits, targets)

# will work: single return path
def forward_inference(self, x):
    return self.head(x)

model.eval()
model.forward = model.forward_inference
transpile(model, example_input, "my_model")
```

Some modules (e.g. HuggingFace transformer blocks) return tuples instead of
plain tensors. `torch.export` will fail if a downstream layer receives a tuple
where it expects a tensor. Unpack the output in your wrapper's forward method:

```python
# won't work: block returns (hidden_states, attention_weights, ...)
x = block(x)

# will work: extract the tensor you need
x = block(x)[0]
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
| Activations | `F.relu`, `torch.sigmoid`, `torch.tanh`, `F.gelu` (tanh approx. only), `F.silu`, `F.leaky_relu`, `F.elu` |
| Normalization | `nn.LayerNorm`, `nn.RMSNorm`, `torch.rsqrt` |
| Softmax | `F.softmax` (any dim) |
| Embedding | `nn.Embedding` |
| Masking | `masked_fill` (for causal attention masks via `register_buffer`) |
| Arithmetic | `+`, `-`, `*` (tensor and scalar), `/` (scalar), unary `-`, `torch.pow` |
| Reduction | `torch.mean` (along a dim) |
| Tensor creation | `torch.arange`, `torch.ones`, `torch.zeros`, `torch.full`, `torch.ones_like`, `torch.zeros_like` |
| Shape | `view`, `reshape`, `flatten`, `contiguous`, `clone` (no-ops on flat data) |
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
