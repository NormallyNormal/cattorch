# Kernel development

[Documentation home](index.md)

This guide is for contributors adding or optimizing Scratch kernels. For
export options, see [optimization modes](optimization.md).

Scratch runs one block at a time, so kernel speedups come from doing fewer list
operations, less loop bookkeeping, fewer temporary lists, and fewer repeated
calculations. Techniques that rely on threads, SIMD, memory bandwidth, or fast
integer arithmetic don't carry over.

## Production path

Both `exact` and `fast` kernels are written in Python with the
`cattorch.util.scratch.dsl` module. The DSL describes Scratch expressions,
statements, loops, conditions, variables, and lists, and compiles them to
Scratch JSON. `Program.pseudocode()` prints a readable version for review.

Don't write Scratch JSON by hand. Write new operations in the DSL or with the
small block builders, check them in the Python emulator, and benchmark them in
Scratch.

Where things live:

| Module | Responsibility |
|---|---|
| `cattorch.frontend` | Turns FX and `torch.export` traces into one graph format. |
| `cattorch.graph` | Constant folding, aliases, validation, and recognizing fusable patterns. |
| `cattorch.operator_registry` | Operation validation, exact/fast kernel selection, quantization hooks, and internal module adapters. |
| `cattorch.transpiler` | Turns graph nodes into instruction programs. |
| `cattorch.sprite` | Adds init, storage, sharding, and generation blocks around the compiled program. |
| `cattorch.util.instruction.optimized_*` | Kernels, split by domain: `linear`, `elementwise`, `normalization`, `tensor`, `convolution`, and `moe`. |
| `cattorch.util.instruction.dispatch`, `cattorch.util.instruction.optimized` | Compatibility re-exports only. Add no new code here. |

A change to an operation should come with:

1. Tests against PyTorch covering shapes, edge cases, and broadcasting.
2. Emulator tests of the generated blocks.
3. A benchmark project whose timer starts after `cattorch init`.
4. Results from real Scratch, with enough iterations to be well above the
   timer's resolution.

Fast kernels also need tests that measure how far the approximation drifts, and
that the exact kernel is used when the approximation is turned off.

## Draft context interface (internal)

`cattorch.kernel_api` is a private draft of a friendlier way to write a kernel.
It isn't exported or supported for users, and it may change before the public
interface in 0.8. Contributors can use it for new kernels; rotary embedding and
`topk` (`cattorch/builtin_kernels.py`) are written this way.

A `Kernel` pairs an operator with an `emit` function. `emit` receives a
`KernelContext` and returns a body of DSL statements; the context names every
input instead of numbering `T` slots by hand.

```python
from cattorch.kernel_api import Kernel
from cattorch.util.scratch.dsl import add, append, change_var, clear, item, mul, repeat, set_var, var

def emit_scale_shift(ctx):
    value, weight = ctx.inputs["value"], ctx.inputs["weight"]  # schema argument names
    index = ctx.temp_var("index")
    out = ctx.output.list
    return (
        clear(out),
        set_var(index, 1),
        repeat(value.numel, (
            append(out, add(
                mul(item(value.list, var(index)), item(weight.list, var(index))),
                ctx.scalars["shift"],
            )),
            change_var(index, 1),
        )),
    )

registry.register_kernel(Kernel("mylib.scale_shift.default", emit_scale_shift))
```

- `ctx.inputs` maps schema argument names to `TensorInfo` (`list`, `shape`,
  `dtype`, `static`, `numel`, `strides`). `ctx.scalars` holds the other
  arguments, with schema defaults filled in. `ctx.output` is the result.
- `ctx.temp_list(name, length)` returns a private scratch list and records its
  capacity so it is sharded past 200,000 items. `ctx.static_list(name, values)`
  returns a private table filled at export time. Private names start with `_`
  and are never merged across instructions. `ctx.temp_var(name)` declares a
  variable; variables are shared by name across the sprite.
- Lists are declared in request order, after the inputs, and the output is
  placed when `ctx.output` is first read. That order shows up in the sprite,
  so keep it stable when porting an existing kernel.
- Don't put per-layer data (a layer index, say) in a kernel's name or body.
  Layer sharing relies on identical layers producing identical code.
- `Kernel(validator=...)` takes the same context and returns a rejection
  reason or `None`; it surfaces in `UnsupportedOperationError`.
  `Kernel(fast_flag="activations")` marks an approximate kernel used only in
  fast mode with that `FastConfig` flag on.
- `ctx.options` holds facts the compiler decides after the operation is
  known, such as the row stride and base variable of banked generation caches,
  or that this is the final layer during hidden prefill. It is empty in
  validators. An option that differs between repeated layers prevents layer
  sharing, so pass one only when it is identical for them.
- A tensor input that the compiler binds to a persistent named list, such as a
  generation K/V cache, is read through `TensorInfo.list` like any other input.
  The kernel appends to or reads from it; the generation wrapper clears,
  rolls back, and shards it. The `cattorch.cached_attention` and
  `cattorch.sequence_row` kernels in `cattorch/builtin_kernels.py` are written
  this way, and read the current position from the `cattorch cache length`
  variable.
- `register_kernel` merges into the operation's existing rule, keeping its
  quantization observer and the other (exact or fast) kernel. Register on a
  cloned registry, never on `default_registry()`.

## Performance evidence

The emulator is the reference for correctness, not speed. The official
Scratch VM gives repeatable local timings, but users run the browser editor,
where browser, OS, hardware, Turbo Mode, and the rest of the project all affect
speed. Record the environment, and only compare candidates within one suite.

Related: [benchmark API reference](api-reference.md#benchmark-api) and the
[repository benchmark guide](../benchmarks/README.md).
