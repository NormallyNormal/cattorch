# cattorch

cattorch exports inference-ready PyTorch neural networks as Scratch 3 sprites.
It lowers a `torch.nn.Module` to ordinary Scratch blocks and lists, so the
result can be imported into a project without a Scratch extension or modified
runtime.

Try the [published TinyStories example](https://scratch.mit.edu/projects/1374224416/)
to see an exported transformer running in vanilla Scratch.

## Quick start

```bash
pip install cattorch
```

cattorch requires Python 3.10 or newer and PyTorch 2.6 or newer.

```python
import torch
from torch import nn

from cattorch import transpile, verify

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).eval()
example = torch.randn(1, 4)

artifact = transpile(model, example, "build/model")
assert verify(model, example, artifact).passed
print(artifact.path)  # build/model.sprite3
```

Import the file using Scratch's **Upload Sprite** option. On the imported
sprite, fill the `input` list, click `cattorch forward`, and read `output`.
The first call initializes the model automatically.

cattorch exports inference only; train your model in PyTorch and call
`model.eval()` before exporting. The [getting-started guide](https://github.com/NormallyNormal/cattorch/blob/main/docs/getting-started.md)
covers input layout, connecting other sprites, and saving after a run.

## What it supports

- MLPs, CNNs, and decoder-style transformers built from
  [supported operations](https://github.com/NormallyNormal/cattorch/blob/main/docs/supported-models.md).
- Exact inference by default, with optional approximate kernels and smaller
  weight storage.
- Cached text generation and character, BPE, or SentencePiece tokenizer sprites.
- Experimental multi-method models, persistent state, and sparse
  mixture-of-experts models.
- Numerical verification and benchmark projects that measure speed in Scratch.

## Documentation

- [All guides](https://github.com/NormallyNormal/cattorch/blob/main/docs/index.md)
- [Getting started](https://github.com/NormallyNormal/cattorch/blob/main/docs/getting-started.md)
- [API reference](https://github.com/NormallyNormal/cattorch/blob/main/docs/api-reference.md)
- [Troubleshooting](https://github.com/NormallyNormal/cattorch/blob/main/docs/troubleshooting.md)

## License

MIT
