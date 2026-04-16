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
| Convolution | `nn.Conv1d`, `nn.Conv2d` (with and without bias, stride, padding) |
| Pooling | `nn.MaxPool1d/2d`, `nn.AvgPool1d/2d`, `nn.AdaptiveAvgPool2d` |
| Linear layers | `nn.Linear` (with and without bias) |
| Matrix multiply | `@` / `torch.matmul` |
| Activations | `F.relu`, `torch.sigmoid`, `torch.tanh`, `F.gelu` (tanh approx. only), `F.silu`, `F.leaky_relu`, `F.elu` |
| Normalization | `nn.BatchNorm1d`, `nn.BatchNorm2d`, `nn.LayerNorm`, `nn.RMSNorm`, `torch.rsqrt` |
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

These are sufficient for architectures like MLPs, CNNs, and transformer LLMs,
including multi-head attention, combined QKV projections, rotary position
embeddings (RoPE), causal masking, pre-norm blocks with residual connections,
and SwiGLU-style gating. RNN support is planned for the future.

## Tokenizers

cattorch can also transpile HuggingFace tokenizers into Scratch sprites, so the
full text → token IDs → model → token IDs → text pipeline can run inside a
Scratch project. Two tokenizer types are supported:

- `CharTokenizer` — character-level lookup. Each character maps to one ID.
- `BPETokenizer` — byte-pair encoding. Merges are applied iteratively over the
  full input string, including spaces.

Off-the-shelf tokenizers from large models will not work here. Production
tokenizers like GPT-2's or Llama's use byte-level pre-tokenization, regex
splits, and other preprocessing steps that the Scratch templates don't
implement, and their 30k–100k+ token vocabularies would cause embeddings to blow past Scratch's
200,000 list item limit. In practice you'll
want to train a small custom BPE tokenizer on your own corpus (with no
pre-tokenizer), so BPE operates on the raw input string, sized to match the
small model you're transpiling.

```python
from transformers import AutoTokenizer
from cattorch import CharTokenizer, BPETokenizer

tokenizer = AutoTokenizer.from_pretrained("my-model")
BPETokenizer(tokenizer).save("my_tokenizer")
# => my_tokenizer.sprite3
```

cattorch does not train tokenizers itself, the classes only transpile an
existing HuggingFace tokenizer. To train a small BPE tokenizer from scratch,
use the `tokenizers` library directly and wrap the result:

```python
from tokenizers import Tokenizer, models, trainers
from transformers import PreTrainedTokenizerFast
from cattorch import BPETokenizer

corpus = ["the cat sat on the mat", "the dog sat on the log"]

tok = Tokenizer(models.BPE())
trainer = trainers.BpeTrainer(vocab_size=100, min_frequency=1, special_tokens=[])
tok.train_from_iterator(corpus, trainer=trainer)

# no pre-tokenizer: BPE operates on the raw input string, including spaces
BPETokenizer(PreTrainedTokenizerFast(tokenizer_object=tok)).save("my_tokenizer")
```

The generated sprite has two top-level block stacks:

- **Encode**: reads the `input` variable (a string) and writes token IDs to
  the `token_ids` list.
- **Decode**: reads the `token_ids` list and writes the decoded string to the
  `output` variable.

Token IDs are 0-based, matching PyTorch embedding conventions, so the output
of the encode stack can be fed directly into a transpiled model. If you don't
care about the tokenizer type, use `transpile_tokenizer(tokenizer, name)` and
cattorch will pick `BPETokenizer` or `CharTokenizer` based on the tokenizer's
backend.

## Scratch limits

- **Project size**: Scratch limits projects to 5 MB. cattorch warns at 4 MB
  and errors at 5 MB.
- **List length**: Scratch lists can hold at most 200,000 items. cattorch
  raises an error if any weight tensor or intermediate list exceeds this.

## License

MIT
