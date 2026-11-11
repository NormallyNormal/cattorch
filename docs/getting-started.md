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
path does not already have that suffix. The Scratch sprite name defaults to
the output filename stem; pass `name="My model"` to choose it independently.

The returned `TranspileResult` describes the sprite's input/output lists,
size, and warnings. See the [result-type reference](api-reference.md#result-types)
for all fields.

> **Before distribution:** inspect `artifact.warnings`, run `verify`, and test
> an import/run/save cycle in the same Scratch client the project will use.

## Import and run the sprite

1. Open the target Scratch 3 project.
2. Choose **Upload Sprite** from the sprite chooser and select the generated
   `.sprite3` file.
3. Select the imported sprite. Its Code area has the `cattorch init` and
   `cattorch forward` My Block definitions, plus a clickable block for each.
4. Show the sprite-local `input` list and replace its contents with the model
   input values.
5. Click the `cattorch forward` call stack.
6. Read the flattened result from the sprite-local `output` list.

For multiple model arguments, later flattened tensors use `input_1`,
`input_2`, and so on. Values are flattened in normal PyTorch contiguous order;
`artifact.inputs` records the original shapes and dtypes. Scratch list
positions are one-based, but integer values such as embedding token IDs remain
the zero-based values expected by PyTorch.

Models returning multiple tensors use `output`, `output_1`, and so on in
PyTorch's nested-container leaf order, recorded in `artifact.outputs`.
If an input or output exceeds 200,000 values, use all its
[shard lists](storage.md#list-sharding) when copying data.

`cattorch forward` initializes the sprite on its first call. Call
`cattorch init` yourself to load the weights ahead of time, for example
before timing a run. Both blocks run without screen refresh, and when each
finishes it sets `cattorch status` to `ok` and broadcasts
`cattorch forward complete` or `cattorch init complete`. The
[Scratch interface](scratch-interface.md) lists every name a sprite offers;
they won't change within cattorch 1.x.

### Overlapping calls

Scratch pauses a block that has run for half a second where it calls a
procedure or repeats a loop, and runs other scripts meanwhile, so another
script can call `cattorch forward` while one is still running. Calls share
the sprite's working lists and decoded weights. Scratch's stop sign, or a
`stop all` block, can stop a block part way.

- Every block runs as it starts; none is refused. A `cattorch forward` that
  another `cattorch forward`, `cattorch init` or `cattorch prepare for save`
  started during sets `cattorch status` to `another call ran during this one`
  when it ends, clears its outputs, and sends no `cattorch forward complete`
  broadcast. If a forward it then resumed during is still running, that one
  fails the same way.
- A `cattorch init` that another block started during, or a forward taken
  over while it decoded the weights, leaves them for the next forward to
  decode again. A forward started while `cattorch init` decodes the weights
  decodes them itself.
- A forward stopped part way leaves nothing for the next call to fix, but its
  outputs are partly written and `cattorch status` keeps its earlier value.

Apart from three cases, a forward that reports `ok` ran alone and its outputs
are right when it ends. First, as for
[generation](generation.md#overlapping-calls): if the forward that was taken
over resumes while the newer block is itself paused, and is still unfinished
when the newer block ends, or is stopped after it resumed, the newer block
can report `ok` with results from working lists the old forward changed.
Second, a forward writes its outputs as it computes, not at its end: if the
forward that was taken over resumes after the newer one ended, it overwrites
the newer one's outputs, and then clears them, after the newer one reported
`ok`. Third, a `cattorch init` or forward taken over while it decoded the
weights carries on decoding them when it resumes: a forward that runs while
it does, whether it finds the weights decoded or decodes them too, can
report `ok` with weights the old block half overwrote; the next forward
after the old block ends decodes them again. Call `cattorch forward` from
one script at a time; to call it from another script, stop the calling
script first.

## Connect another sprite

My Blocks and sprite-local lists belong to one Scratch sprite. A controller
sprite cannot call `cattorch forward` or directly read the generated local
`input` and `output` lists.

[`build_project`](projects-and-cli.md) writes a project with these bridges
already in place: `broadcast [NAME run] and wait` after filling the global
`NAME input` list, then read `NAME output`. To build a bridge by hand, go
through global lists:

1. Create global input and output lists with your own names.
2. On the generated processor sprite, add a `when I receive` script.
3. In that receiver, copy the global input into local `input`, call
   `cattorch forward`, then copy local `output` to the global output list.
4. From the controller, use `broadcast [run model] and wait`. When it returns,
   the global output is ready.

In Scratch-like pseudocode, the receiver is:

```text
when I receive [run model]
delete all of [input]
copy every item of [my global input] to [input]
cattorch forward
delete all of [my global output]
copy every item of [output] to [my global output]
```

Don't make a new list named `input` or `output` to replace the generated one.
Generated blocks refer to lists by ID, not by name, so they will keep using the
original.

Tokenizer sprites have the same locality rule. See
[tokenizer integration](tokenizers.md) and
[cached generation](generation.md) for their additional procedures.

## Prepare the project for saving

Before saving a project that has run, call `cattorch prepare for save`. It
clears the decoded weights and working data but keeps the compressed weights,
so the saved project doesn't store both. Input lists and a program's state are
kept. The next forward call initializes the
sprite again.

Don't edit the generated sprite's costumes: they store the compressed weights.
See [preserve processor costumes](storage.md#preserve-processor-costumes).

## Add export options

```python
from cattorch import StorageConfig, transpile

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
