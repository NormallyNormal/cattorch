# Programs and sparse MoE

[Documentation home](index.md)

`ExportProgram` puts several model methods, and state that persists between
calls, into one sprite. Any model can keep state this way, not only a
language model: a running total, a memory of past inputs, or the hidden
state of a recurrent network fed one reading per call. Programs and adapters are stable; the
[mixture-of-experts classes](#mixture-of-experts) and `analyze` are
[experimental](stability.md). For a model with just a `forward` method and no state, use the
[ordinary export](getting-started.md) instead.

[Markers](markers.md) declare the same things with decorators on your own
model: `@entrypoint` methods and the buffers they update instead of
`ExportProgram`, and `@moe` instead of rebuilding a block with `ExpertFamily`
and `SparseMoE`. The classes here remain for code you can't decorate.

## Named entry points and state

Import the interface types from `cattorch`:

```python
import torch
import cattorch
from cattorch import (
    EntryPoint, ExportProgram, Input, Output, State, StateInput, StateUpdate,
)


class Counter(torch.nn.Module):
    def step(self, value, state):
        updated = state + value
        return updated, updated


model = Counter().eval()
program = ExportProgram(
    states=(State("counter", torch.zeros(4)),),
    entrypoints=(EntryPoint(
        "step",
        method="step",
        arguments=(Input("value", torch.ones(4)), StateInput("counter")),
        returns=(Output("result"), StateUpdate("counter")),
    ),),
)

result = cattorch.transpile(model, program, "model.sprite3")
```

`arguments` maps the method's parameters, in order, to input lists or stored
state. `returns` maps each returned tensor to an output list or a state update.
Here, each call adds `value` to the stored counter and outputs the new total.

The sprite gets `cattorch step`, `cattorch init`, `cattorch reset`, and
`cattorch prepare for save` blocks. Lists are named after the entry point and
state, such as `cattorch step value`, `cattorch step result`, and
`cattorch state counter`. When a call ends, the sprite sets `cattorch status`
to `ok`, or to why it failed, such as an input list with the wrong number of
items, and broadcasts `cattorch step complete`. The
[Scratch interface](scratch-interface.md#program-sprites) lists every name.

Rules:

- `init`, `reset`, and `prepare for save` can't be used as entry point names.
- List names must be unique across entry points and states, and can't end in
  `shard N`, which is used for [sharding](storage.md#list-sharding).
- An input and an output of the same entry point can share a name only if
  they have the same shape. The output then replaces the input in place.

### State

A `State(name, initial)` is a tensor the sprite keeps between calls, in the
list `cattorch state NAME`. An entry point reads it with `StateInput(name)`
among its `arguments`, and writes it with `StateUpdate(name)` among its
`returns`. Every entry point can read every state.

| | `mode="replace"` (the default) | `mode="append"` |
|---|---|---|
| Shape | Fixed: the shape of `initial`. | The first dimension grows, from `initial`'s rows up to `capacity` rows; the others are fixed. `capacity=` is required. |
| An update | The returned tensor, with the state's shape, replaces it. | The returned rows are added at the end. |
| When full | | The call is refused with `state full: cattorch state NAME` and changes nothing. |

- A call can update each state at most once. It writes its updates at its
  end, all together, and only if it succeeds: a call that fails leaves the
  outputs and states as they were.
- `cattorch reset` restores every state to its `initial` value, so a learned
  start state works.
- A state list belongs to the sprite. A project may read it, and may save
  and restore it, with values the sprite wrote. Each call checks its length:
  a replace state must hold its number of items, and an append state whole
  rows, from `initial`'s rows up to `capacity`; otherwise the call fails with
  `wrong number of items`, `too few items` or `too many items` naming it.
  Its items aren't checked, since the sprite rewrites the state every call
  and checking 200,000 items each time would cost more than the call, and
  writing anything else into it gives undefined results.
- Each entry point method is traced once, like any export, so it follows one
  path: it can't branch on tensor values, and every shape is fixed when you
  export, except an append state's first dimension.

That growing dimension is called a runtime extent. Elementwise, linear,
normalization, matrix, and reshaping operations can use it, with these
limits, all checked at export:

- A tensor can have only one runtime extent, and it can't be multiplied by
  another.
- It can't be partly sliced, and output shapes can't depend on tensor values.
- Convolution and pooling accept it only as the batch dimension.
- Arithmetic with a length, as in `self.history.sum(0) / self.history.shape[0]`
  or the `n - 1` of `var`, follows that length at run time, whichever of an
  entry point's variable-length inputs and append states it comes from. A
  number computed from two of those lengths together, as in
  `x / (a.shape[0] + b.shape[0])`, fails the export; keep such a count in a
  state of its own.

Export checks each entry point at several lengths of each of its
variable-length inputs and append states, one at a time and then all
together, with the append state's list filled to that length, and compares
its outputs and updated states with the method's. A method that builds
something from a length that the sprite can't follow, such as
`torch.arange(self.history.shape[0])`, fails that check.

With [markers](markers.md#entry-points-and-state), each buffer an
`@entrypoint` method changes becomes a state: a replace state, or an append
state if it grows along its first dimension, with the `capacity=` you give.

### Stopped and overlapping calls

Call the sprite's blocks from one script at a time. What the sprite detects
when calls overlap, and how to recover, is in
[overlapping calls](scratch-interface.md#overlapping-calls). A call stopped,
or taken over, while it writes the states leaves them partly written: every
entry point then reports `an earlier call was stopped or is still running;
reset first` and changes nothing, until `cattorch reset`. A program without
state has nothing to reset; its next call simply runs.

To test a sequence of calls, pass `ProgramCall` values to `verify()`:

```python
from cattorch import ProgramCall

check = cattorch.verify(
    model,
    program,
    result,
    calls=(
        ProgramCall("step", (torch.ones(4),)),
        ProgramCall("step", (torch.full((4,), 2.0),)),
    ),
)
assert check.passed, check
```

Each entry in `check.calls` includes the Scratch `status`. A call Scratch
rejects fails verification even if its outputs happen to match. Later calls
continue from the state after the last successful call.

To check support and estimated size without writing a sprite, call
`analyze(model, program)` from `cattorch.experimental`; `analyze(model, None)` reads a marked model's program from its markers.

## Scoped module adapters

`ModuleAdapter` swaps every module of a given type for a replacement during
export, for example to turn a third-party block into supported operations:

```python
from cattorch import ModuleAdapter

adapter = ModuleAdapter(ThirdPartyBlock, convert_block)
result = cattorch.transpile(model, program, "model.sprite3", adapters=(adapter,))
```

The function receives `(module, context)`, where `context.module_path` is the
module's name in the model. Adapters run on a copy of the model, affect only
that export, and apply to child modules before their parents. A module used in
several places is replaced once, and every use gets the same replacement.

## Mixture of experts

These classes are experimental, imported from `cattorch.experimental`, and may
change in any release; [`@moe`](markers.md) and automatic detection cover most
models.

`ExpertFamily` holds a set of experts with identical structure. `SparseMoE`
sends each token to its top-k experts and combines their outputs. In Scratch,
only the selected experts run.

For models with separate expert modules:

```python
import torch
from torch import nn
from cattorch import transpile, verify
from cattorch.experimental import ExpertFamily, SparseMoE

example = torch.zeros(1, 4)
experts = [
    nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
    for _ in range(4)
]
family = ExpertFamily.from_modules(experts, example_input=example)
router = nn.Sequential(nn.Linear(4, 4), nn.Softmax(dim=-1))
moe = SparseMoE(router, family, top_k=2).eval()

artifact = transpile(moe, example, "moe")
assert verify(moe, example, artifact).passed
```

The router can return:

- One score tensor, used both to pick experts and to weight their outputs.
- A `(selection_scores, combination_scores)` tuple, to pick and weight with
  different scores.
- A `RoutingScores` object, which can also normalize the selected weights.

On exact ties, the lower expert ID wins. If normalization would divide by zero,
the weights are zero.

If your model already stores expert weights stacked in one tensor per
parameter, pass one expert module as a template plus a dictionary of the
stacked tensors:

```python
family = ExpertFamily(
    expert_template,
    {
        "up.weight": up_bank,       # [experts, hidden, width]
        "up.bias": up_bias_bank,    # [experts, hidden]
        "down.weight": down_bank,   # [experts, width, hidden]
        "down.bias": down_bias_bank,
    },
    example_input=torch.zeros(1, width),
)
moe = SparseMoE(router, family, top_k=2)
```

The first dimension of each stacked tensor is the number of experts.
`ExpertFamily.from_stacked` does the same thing.

An expert must process each token independently and keep no state. It can use
linear layers (with or without bias), elementwise arithmetic, residual
connections, ReLU, GELU, SiLU, sigmoid, and tanh, `clamp` with constant
bounds, powers with integer exponents from -4 to 4 or ±0.5, LayerNorm and
RMSNorm, slicing, `chunk`, `split`, transposes and float casts, and low-rank
linear layers written as two matrix products. Convolution, attention,
in-place modification, and data-dependent control flow are rejected.

For SwiGLU experts without biases, `StackedSwiGLUMoE` is a faster
implementation, which `SparseMoE` and recognized MoE blocks switch to by
themselves. A block whose experts' weights are stacked into one tensor each
isn't recognized; an [`@adapter`](markers.md#classes-you-dont-own) converts it:

```python
import torch
from torch import nn

from cattorch import adapter
from cattorch.experimental import StackedSwiGLUMoE


class MyStackedMoE(nn.Module):
    def __init__(self, width=8, hidden=16, experts=4):
        super().__init__()
        self.top_k = 2
        self.router = nn.Linear(width, experts, bias=False)
        self.gate = nn.Linear(1, 1)  # each weight holds every expert's matrix
        self.up = nn.Linear(1, 1)
        self.down = nn.Linear(1, 1)
        self.gate.weight = nn.Parameter(torch.randn(experts, hidden, width))
        self.up.weight = nn.Parameter(torch.randn(experts, hidden, width))
        self.down.weight = nn.Parameter(torch.randn(experts, width, hidden))

    # forward routes each token through its top_k experts.


@adapter(MyStackedMoE)
def stacked_experts(module, context):
    return StackedSwiGLUMoE(
        module.router.weight, module.gate.weight, module.up.weight, module.down.weight,
        top_k=module.top_k,
    )
```

Pass it to the export with `transpile(..., adapters=[stacked_experts])`.

### Hand-written MoE blocks

A block that holds a `ModuleList` of identical experts and a linear router
with one output per expert, and loops over the experts the usual way, usually
exports without changes. cattorch runs the block on probe inputs next to
`SparseMoE` with each routing configuration it knows: every `top_k`, softmax,
sigmoid or raw router scores, and selected weights renormalized or not. The
first one that gives the same output replaces the block, and a
`CustomMoEWarning` names it, for example `top_k=2, softmax scores,
renormalized weights`. Softmax over the top-k logits, as in Mixtral, is the
same as softmax scores with renormalized weights. Run `verify` on real inputs
afterwards.

Anything else in the block, such as a shared expert, a routed scaling factor,
or router bias added only for selection, doesn't match. Such a block is kept
as it is if it traces; otherwise export stops with an error pointing here.
Rebuild it with `ExpertFamily` and `SparseMoE`, or keep its router and call
`routed_experts` as described next.

### Routing with your own router code

A typical MoE block picks experts with `torch.topk` and then loops over the
experts with `torch.where` and `index_add_`. That loop depends on tensor
values, so it can't be traced. Keep the router as it is and replace only the
loop with `routed_experts(value, expert_ids, weights, experts)`:

```python
import torch
from torch import nn
from cattorch import transpile, verify
from cattorch.experimental import ExpertFamily, routed_experts

class Block(nn.Module):
    def __init__(self, width=4, experts=4, top_k=2):
        super().__init__()
        self.gate = nn.Linear(width, experts, bias=False)
        self.experts = ExpertFamily.from_modules(
            [nn.Sequential(nn.Linear(width, 8), nn.SiLU(), nn.Linear(8, width))
             for _ in range(experts)],
            example_input=torch.zeros(1, width),
        )
        self.top_k = top_k

    def forward(self, value):
        probabilities = torch.softmax(self.gate(value), dim=-1)
        weights, ids = torch.topk(probabilities, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return routed_experts(value, ids, weights, self.experts)

model = Block().eval()
example = torch.randn(3, 4)
artifact = transpile(model, example, "routed-moe")
assert verify(model, example, artifact).passed
```

`expert_ids` and `weights` have shape `[..., top_k]`, where `value` has shape
`[..., width]`. Each token's output is the sum of `weights[t, r]` times
expert `expert_ids[t, r]` applied to the token, over its routes `r`. The router
is ordinary exported code, so softmax or sigmoid scoring, normalization,
shared experts, and other variations need no special support. Expert IDs are
zero-based, as `torch.topk` returns them. In PyTorch, IDs outside the expert
bank raise `IndexError`; in Scratch they aren't checked.

When router scores tie exactly, Scratch's `topk` picks the lower expert ID.
PyTorch doesn't define which of the tied experts `torch.topk` returns, so
`verify` can fail on inputs that make scores tie, such as all zeros.

For GPTQ, use calibration data that routes tokens the way real use does.
Experts that receive too little data fall back to symmetric quantization; see
[GPTQ calibration](storage.md#gptq-calibration).

## Limitations

- Adapters must return PyTorch modules; they can't emit Scratch blocks directly.
- A state can be updated in place like any tensor the method computes:
  `self.buffer[0] = x`, `self.buffer[1:] = self.buffer[:-1].clone()`,
  `self.memory[:, 2].mul_(0.5)`, `self.memory[1].add_(x)`, boolean-mask
  assignment (`self.buffer[self.buffer > 1] = 0`), assignment at index
  tensors, `masked_fill_`, `copy_`, `fill_`, `zero_` and `clamp_`. To write at
  a position held in a state, as a ring buffer does, index with a one-item
  tensor: `self.buffer[self.position.reshape(1)] = x.unsqueeze(0)`; a 0-d
  index (`self.buffer[self.position] = x`) becomes a Python number when
  traced and fails the export. Writing through a strided or transposed view
  (`self.memory.t()[0] = 1`), and reading a view taken before the write
  (`row = self.buffer[0]; self.buffer[0] = x; return row`), fail the export
  too; write those out of place, with `torch.cat` or `torch.where`.
- Training and autograd aren't supported.
