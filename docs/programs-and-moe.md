# Experimental programs and sparse MoE

[Documentation home](index.md)

`ExportProgram` puts several model methods, and state that persists between
calls, into one sprite. Everything in this guide is experimental and may
change. For a model with just a `forward` method and no state, use the
[ordinary export](getting-started.md) instead.

## Named entrypoints and state

Import the interface types from `cattorch.experimental`:

```python
import torch
import cattorch
from cattorch.experimental import (
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
`cattorch prepare for save` blocks. Lists are named after the entrypoint and
state, such as `cattorch step value`, `cattorch step result`, and
`cattorch state counter`. After a successful call, the sprite broadcasts
`cattorch step complete` and sets `cattorch status` to `ok`. If the input is
invalid, `cattorch status` holds an error message instead.

Rules:

- A call can update each state at most once.
- `init`, `reset`, and `prepare for save` can't be used as entrypoint names.
- List names must be unique across entrypoints and states, and can't end in
  `shard N`, which is used for [sharding](storage.md#list-sharding).
- An input and an output of the same entrypoint can share a name only if they
  have the same shape. The output then replaces the input in place.

`State(mode="replace")` has a fixed shape, and each update replaces it.
`State(mode="append")` grows along its first dimension with each update, up to
its `capacity`. A call that fails leaves the outputs and state as they were.
`cattorch reset` restores every state to its initial value.

Because an append state's first dimension changes at runtime, it is the one
exception to fixed shapes. This variable-length dimension is called a runtime
extent. Elementwise, linear, normalization, matrix, and reshaping operations
can use it, with these limits, all checked at export:

- A tensor can have only one runtime extent, and it can't be multiplied by
  another.
- It can't be partly sliced, and output shapes can't depend on tensor values.
- Convolution and pooling accept it only as the batch dimension.

To test a sequence of calls, pass `ProgramCall` values to `verify()`:

```python
from cattorch.experimental import ProgramCall

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
`analyze(model, program)` from `cattorch.experimental`.

## Scoped module adapters

`ModuleAdapter` swaps every module of a given type for a replacement during
export, for example to turn a third-party block into supported operations:

```python
from cattorch.experimental import ModuleAdapter

adapter = ModuleAdapter(ThirdPartyBlock, convert_block)
result = cattorch.transpile(model, program, "model.sprite3", adapters=(adapter,))
```

The function receives `(module, context)`, where `context.module_path` is the
module's name in the model. Adapters run on a copy of the model, affect only
that export, and apply to child modules before their parents. A module used in
several places is replaced once, and every use gets the same replacement.

## Mixture of experts

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
connections, ReLU, GELU, SiLU, sigmoid, and tanh, LayerNorm and RMSNorm, and
low-rank linear layers written as two matrix products. Convolution, attention,
in-place modification, and data-dependent control flow are rejected.

For SwiGLU experts without biases, `StackedSwiGLUMoE` is a faster
implementation. If your MoE module has `router.weight`, `gate.weight`,
`up.weight`, and `down.weight` attributes, `stacked_swiglu_moe_adapter(YourMoEClass)`
converts it.

For GPTQ, use calibration data that routes tokens the way real use does.
Experts that receive too little data fall back to symmetric quantization; see
[GPTQ calibration](storage.md#gptq-calibration).

## Limitations

- Programs require the FX frontend (`frontend="fx"`, the default).
- Adapters must return PyTorch modules; they can't emit Scratch blocks directly.
- State can't be updated in place by index.
- Training and autograd aren't supported.
