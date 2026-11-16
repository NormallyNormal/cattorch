# Getting started

[Documentation home](index.md)

This guide exports a small model, checks its output, imports it into Scratch,
and connects it to the rest of a project.

## Install

```bash
pip install cattorch
```

cattorch requires Python 3.11 or newer and PyTorch 2.10 or newer. To use a
specific PyTorch build, such as CPU-only, install it with the
[PyTorch installer](https://pytorch.org/get-started/locally/) first. Tokenizer
export needs extra packages; see [tokenizers](tokenizers.md).

## Export and verify a model

`example_inputs` fixes the input shapes and dtypes the sprite will accept. It
is not a training batch. Data for quantization is passed separately as
[calibration inputs](storage.md#gptq-calibration).

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
for warning in artifact.warnings:
    print("warning:", warning)

check = verify(model, example, artifact)
assert check.passed, check
```

`transpile` creates missing parent directories and appends `.sprite3` if the
path does not already have that suffix; a path that names no file, such as
an existing directory, raises `ValueError`. The export's name, NAME below,
defaults to the output filename stem (`two_layer_net` here); pass
`name="My model"` to choose it. It is the sprite's name in Scratch, and
every list, variable and broadcast a project uses the sprite through starts
with it.

The returned `TranspileResult` names the sprite's input and output lists
and its broadcasts, and gives its size and warnings. See the
[result-type reference](api-reference.md#result-types) for all fields.

> **Before distribution:** inspect `artifact.warnings`, run `verify`, and test
> an import/run/save cycle in the same Scratch client the project will use.

## Use the sprite in a project

You never need to open the sprite: a large model makes the Scratch editor
slow while its sprite is selected. Use it from another sprite, or the stage.

1. Open the target Scratch 3 project.
2. Choose **Upload Sprite** from the sprite chooser and select the generated
   `.sprite3` file. Scratch creates the global lists and variables the
   sprite uses, here `two_layer_net input`, `two_layer_net output` and
   `two_layer_net status`, and selects the sprite: select another sprite,
   or the stage, right away.
3. In a script of that other sprite, fill `two_layer_net input` with the
   model's input values, then `broadcast [two_layer_net forward] and wait`.
4. Read `two_layer_net status`. When it's `ok`, the flattened result is in
   `two_layer_net output`; otherwise the status says why the call failed,
   such as an input list with the wrong number of items.

In Scratch-like pseudocode:

```text
when green flag clicked
delete all of [two_layer_net input]
add (0.5) to [two_layer_net input]   // ... one item per input value
broadcast [two_layer_net forward] and wait
if <(two_layer_net status) = [ok]> then
  say (item (1) of [two_layer_net output])
end
```

For multiple model arguments, later flattened tensors use `NAME input_1`,
`NAME input_2`, and so on. Values are flattened in normal PyTorch contiguous order;
`artifact.inputs` records each list's name and the original shapes and
dtypes. Scratch list positions are one-based, but integer values such as
embedding token IDs remain the zero-based values expected by PyTorch.

Models returning multiple tensors use `NAME output`, `NAME output_1`, and so
on in PyTorch's nested-container leaf order, recorded in `artifact.outputs`.
If an input or output exceeds 200,000 values, use all its
[shard lists](storage.md#list-sharding).

`NAME forward` decodes the weights on its first call. Broadcast
`NAME init` yourself to decode them ahead of time, for example before
timing a run. Calls run without screen refresh. While one runs,
`NAME status` is `not finished`; when it ends, `NAME status` says how, and
the call broadcasts `NAME forward complete` (or `NAME init complete`), for
scripts that react to events rather than wait. A forward that fails writes
no output. The [Scratch interface](scratch-interface.md) lists every name a
sprite offers and every status; they won't change within cattorch 1.x.

To use two copies of one model in a project, export them with different
`name`s: two sprites of one name share their lists and both answer the
same broadcasts.

### Overlapping calls

Call the sprite from one script at a time, and let each call end before
starting another. A call that another call overlaps, or that is stopped
part way, reports it in `NAME status`, and a sprite with state can then
refuse some calls until it is reset, which ones depending on the kind of
sprite; see [overlapping calls](scratch-interface.md#overlapping-calls).

[`build_project`](projects-and-cli.md) writes a complete project around a
sprite, with the globals declared and a small Demo sprite that runs it.
Tokenizer and generation sprites work the same way; see
[tokenizer integration](tokenizers.md) and [cached generation](generation.md)
for their calls.

## Prepare the project for saving

Before saving a project that has run, `broadcast [cattorch prepare for save]
and wait`: every model, generation and program sprite answers it (each
also has its own `NAME prepare for save`). It clears the decoded weights
and working data but keeps the compressed weights, so the saved project
doesn't store both. Input and output lists and a program's state are kept.
The next call decodes the weights again.

Don't edit the generated sprite's costumes. The weights are stored in lists,
but the costumes are named after the characters of their encoding, and the
sprite decodes the weights by looking characters up among them. See
[preserve processor costumes](storage.md#preserve-processor-costumes).

## Add export options

This continues the example above:

```python
from cattorch import StorageConfig, transpile, verify

artifact = transpile(
    model,
    example,
    "build/two_layer_net_f16",
    name="Two-layer classifier",
    storage=StorageConfig(precision="float16"),
)

check = verify(model, example, artifact)
print(check.passed, check.max_abs_error)
```

Float16 storage makes the sprite smaller but can change its output. Check the
error on representative inputs before using it. For more options, see
[storage and quantization](storage.md), [optimization modes](optimization.md),
and [code generation](code-generation.md).

Next: [supported models and operations](supported-models.md) or the full
[`transpile` API](api-reference.md#transpile).
