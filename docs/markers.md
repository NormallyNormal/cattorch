# Markers

[Documentation home](index.md)

Markers are decorators on your own classes and methods that tell cattorch
what it can't work out from the graph: which methods to export, which buffers
are persistent state, what a block's mixture-of-experts routing is, or what a
module should look like during export. They replace building
[programs and MoE classes](programs-and-moe.md) by hand.

A marker only attaches metadata. Your model runs, trains, and saves exactly
as before, and markers change nothing for a model without them. Import them from
`cattorch`, as in `@cattorch.entrypoint`.

## Entry points and state

`@entrypoint` exports a method as a Scratch call. A buffer the method updates
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

    @entrypoint(example_inputs=lambda: (torch.ones(3),))
    def add(self, value):
        self.total += value
        return self.total.sum()

    @entrypoint(example_inputs=lambda: (torch.ones(3),), capacity={"seen": 8})
    def remember(self, row):
        self.seen = torch.cat([self.seen, row.reshape(1, 3)])
        return row.max()

    @entrypoint(output_names=("scores",), example_inputs=lambda: (torch.ones(3),))
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
interface from the markers. Each entry point becomes a broadcast,
`NAME <name>` (NAME being the export's name), with a global list per input
named after its parameter (`NAME add value`) and a list per output
(`NAME score scores`).

- `example_inputs=` gives the arguments to export with: a tuple, a tensor,
  or a function returning one. They must be tensors. `example_inputs=()`
  exports a method with no arguments, computed from the weights and the
  state.
- `name=` names the entry point, `input_names=` and `output_names=` name its
  lists. Outputs default to `output`, `output_1`, and so on. Only the
  method's tensor results become outputs, as with `forward`; other results,
  such as `None`, are skipped with a warning, and tensors inside a dataclass
  fail the export. A name must be a non-empty string without control
  characters such as newlines, and the names are tuples of strings.
- `dynamic={"rows": {0: (1, 32)}}` lets an input's dimension vary between
  those bounds at run time, checked at several lengths during export, in the
  same form as `transpile(dynamic=...)`: keys are input names, and
  `(1, 32)` is short for the first dimension (see
  [variable-length inputs](supported-models.md#variable-length-inputs)).
- cattorch runs each entry point once on a copy of the model to find the
  buffers it changes. Each one becomes a state list, `NAME state total`,
  starting from the buffer's value; `NAME reset` restores it. Each entry
  point takes, and checks, only the states it reads or writes; the others
  aren't part of its interface.
- A buffer that grows along its first dimension, like `seen`, needs its
  largest length in `capacity=`, the same on every entry point that grows
  it. Another entry point can read it, as `score`
  does, and so can the method that grows it, after appending, with the
  [operations that follow a variable length](supported-models.md#variable-length-inputs).
- A buffer that changes shape without a capacity, or a capacity for a buffer
  that never grows, fails the export. `capacity=` is a dict from buffer name
  to a positive integer, checked when the decorator is applied.
- A method that raises on its `example_inputs` fails the export with
  `UnsupportedModelError`, giving the inputs' shapes.
- A state can't share memory with another buffer or parameter, as views of
  one tensor do (`self.register_buffer("k", self.cache[0])`): each state is
  a list of its own, so a write through one wouldn't reach the other.
  Register only `cache`, and take `k = self.cache[0]` inside the method. A
  tensor registered under several names, by the model and a submodule, is
  one state, as long as the method updates it in place; reassigning it
  under one of its names fails the export.
- A state is a list of numbers from the start: a buffer registered as
  `None` that a method sets to a tensor, or one a method sets to `None`,
  fails the export. Give the buffer an initial tensor, and keep whether it
  has been set in another buffer, such as `torch.tensor(False)`. A method
  that returns no tensor and changes no buffer fails too, since its call
  would do nothing.
- Only buffers become state. Every entry point is traced from the model as
  you gave it, so the sprite runs every call of every entry point as a
  first call, from the initial values of everything else. An entry point
  that changes a parameter fails the export, and so does one that changes,
  on every call, a tensor kept as a plain attribute
  (`self.total = torch.zeros(3)` in `__init__`, then
  `self.total = self.total + x`), a tensor in a list or dictionary
  attribute, or a Python value such as `self.count += 1` with an `int` or
  appending to a list, of the model or a submodule; cattorch calls the
  method a second time, with other inputs, to tell. Register a tensor with
  `register_buffer` instead, and keep a number in a buffer such as
  `torch.tensor(0)`. A change made on the first call only, such as a cache
  filled once, or the weight old-style `nn.utils.spectral_norm` computes on
  its first forward, is allowed if later calls of every entry point give
  what they would give from the initial values, with a warning; one that
  changes their results, as `self.scale = 2.0` read by another entry point
  does, fails the export.

`marked_program(model)` returns a `MarkedProgram`: the export copy of the
model and the
[`ExportProgram`](programs-and-moe.md#named-entry-points-and-state) the markers
describe, if you want to inspect or extend them. `model, program =
marked_program(model)` unpacks it.

## Example inputs

`@example_inputs` gives a class example inputs for `forward`, so tools such
as the command line can export it without being told its shapes:

```python
from cattorch import example_inputs

@example_inputs(lambda: (torch.zeros(1, 1, 28, 28),))
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
