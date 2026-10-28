# Markers

[Documentation home](index.md)

Markers are decorators on your own classes and methods that tell cattorch
what it can't work out from the graph: which methods to export, which buffers
are persistent state, what a block's mixture-of-experts routing is, or what a
module should look like during export. They replace building
[programs and MoE classes](programs-and-moe.md) by hand.

A marker only attaches metadata. Your model runs, trains, and saves exactly
as before, and an unmarked model exports as it always has. Import them from
`cattorch`, as in `@cattorch.entrypoint`.

## Entry points and state

`@entrypoint` exports a method as a Scratch block. A buffer the method updates
becomes persistent state:

```python
import torch
from torch import nn
from cattorch import entrypoint, transpile, verify
from cattorch import ProgramCall

class Tally(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("total", torch.zeros(3))
        self.register_buffer("seen", torch.zeros(0, 3))

    @entrypoint(example=lambda: (torch.ones(3),))
    def add(self, value):
        self.total += value
        return self.total.sum()

    @entrypoint(example=lambda: (torch.ones(3),), capacity={"seen": 8})
    def remember(self, row):
        self.seen = torch.cat([self.seen, row.reshape(1, 3)])
        return row.max()

    @entrypoint(outputs=("scores",), example=lambda: (torch.ones(3),))
    def score(self, weights):
        return torch.relu(self.seen @ weights)

model = Tally().eval()
result = transpile(model, None, "tally.sprite3")
report = verify(model, None, result, calls=(
    ProgramCall("add", (torch.tensor([1.0, 2.0, 3.0]),)),
    ProgramCall("remember", (torch.tensor([0.5, -1.0, 2.0]),)),
    ProgramCall("remember", (torch.tensor([1.0, 1.0, 0.0]),)),
    ProgramCall("score", (torch.tensor([1.0, 0.0, -1.0]),)),
))
assert report.passed
```

Passing `None` as the example tells `transpile` and `verify` to take the
interface from the markers. Each entry point becomes a block named
`cattorch <name>`, with a list per input named after its parameter
(`cattorch add value`) and a list per output (`cattorch score scores`).

- `example=` gives the arguments to export with: a tuple, a tensor, or a
  function returning one. They must be tensors.
- `name=` names the block, `inputs=` and `outputs=` name its lists. Outputs
  default to `output`, `output_1`, and so on. Only the method's tensor
  results become outputs, as with `forward`; other results, such as `None`,
  are skipped with a warning.
- `dynamic={"rows": (1, 32)}` lets an input's first dimension vary between
  those bounds at run time, checked at several lengths during export (see
  [variable-length inputs](supported-models.md#variable-length-inputs)).
- cattorch runs each entry point once on a copy of the model to find the
  buffers it changes. Each one becomes a state list, `cattorch state total`,
  starting from the buffer's value; `cattorch reset` restores it. Every entry
  point can read every state.
- A buffer that grows along its first dimension, like `seen`, needs its
  largest length in `capacity=`. Another entry point can read it, as `score`
  does, and so can the method that grows it, after appending, with the
  [operations that follow a variable length](supported-models.md#variable-length-inputs).
- A buffer that changes shape without a capacity, or a capacity for a buffer
  that never grows, fails the export.

`marked_program(model)` returns the export copy and the
[`ExportProgram`](programs-and-moe.md#named-entrypoints-and-state) the markers
describe, if you want to inspect or extend it.

## Example inputs

`@example` gives a class example inputs for `forward`, so tools such as the
command line can export it without being told its shapes:

```python
from cattorch import example

@example(lambda: (torch.zeros(1, 1, 28, 28),))
class Digits(nn.Module):
    ...

transpile(Digits().eval(), None, "digits.sprite3")
```

## An export-only forward

`@export_forward` marks a method cattorch traces instead of `forward`, for
every instance of the class wherever it is in the model. Use it for a block
whose `forward` has training branches or control flow that depends on tensor
values:

```python
from cattorch import export_forward

class Gate(nn.Module):
    def forward(self, value):
        if float(value.max()) > 1:       # can't be traced
            return value.clamp(max=1)
        return value

    @export_forward
    def clamped(self, value):
        return value.clamp(max=1)
```

It takes the same arguments as `forward` and should compute the same result;
`verify` compares the export with `forward`.

## Mixture-of-experts routing

cattorch recognizes many hand-written MoE blocks by itself. When it can't,
for example with a shared expert or a scaled routed output, `@moe` states the
routing:

```python
from cattorch import moe

@moe(router="gate", experts="experts", top_k=2, normalize=True, shared="shared")
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(16, 4, bias=False)
        self.experts = nn.ModuleList(nn.Linear(16, 16) for _ in range(4))
        self.shared = nn.Linear(16, 16)

    def forward(self, x):
        ...  # your own topk and loop over the experts
```

`router` and `experts` name submodules: one score per expert, and a
`ModuleList` of identical experts. `scoring` (`"softmax"`, the default,
`"sigmoid"`, or `"raw"`) turns scores into weights, and `normalize` rescales
the chosen weights to sum to 1. `shared` names an expert every token runs,
and `scale` multiplies the routed output before it is added. The block's
`forward` still runs in PyTorch; at export cattorch replaces it with a
[`SparseMoE`](programs-and-moe.md#mixture-of-experts) and checks the two
match on probe inputs, so wrong options fail the export.

## Generation methods

`@generate` marks the method a `GenerationProgram` exports when it isn't
`forward`, instead of `GenerationProgram(method=...)`:

```python
from cattorch import GenerationProgram
from cattorch import generate

class Chat(nn.Module):
    @generate
    def logits(self, tokens):
        ...

transpile(Chat().eval(), GenerationProgram(max_context=64), "chat.sprite3")
```

## Classes you don't own

A marker needs your own source. For a third-party class, `@adapter` turns a
function into a [`ModuleAdapter`](programs-and-moe.md) to pass to one export:

```python
from cattorch import adapter

@adapter(ThirdPartyBlock)
def export_block(module, context):
    return MyTraceableBlock(module)

transpile(model, program, "model.sprite3", adapters=[export_block])
```

## Inheritance

Markers on a base class apply to its subclasses, and a subclass can redeclare
one to replace it. A method overridden without a marker loses it.
