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

## Context kernels

New kernels are written against the public, experimental interface described
in [custom kernels and operations](custom-kernels.md): a `Kernel` with an
`emit(ctx)` function returning `cattorch.scratch` statements. Rotary
embedding, `topk`, runtime gather and scatter, and generation's cached
attention (`cattorch/builtin_kernels.py`) are written this way, and register
in `_build_default_registry`. New built-in operations are declared with
`cattorch.custom_ops.op`, as `cattorch/ops.py` does. Notes that only matter
inside cattorch:

- Lists are declared in request order, after the inputs, and the output is
  placed when `ctx.output` is first read. That order shows up in the sprite,
  so keep it stable when porting an existing kernel.
- `temp_list` and `static_list` names start with `_` and are never merged
  across instructions; `temp_list` records its capacity so it is sharded past
  200,000 items.
- `Kernel(fast_flag="activations")` ties an approximate kernel to a
  `FastConfig` flag.
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
  cloned registry, never on `default_registry()`; `transpile(kernels=...)`
  does this per export.

## Operator conformance

`tests/opinfo_conformance.py` runs PyTorch's OpInfo samples, up to four per
operator, through `transpile` and `verify`. Each sample's tensors are Scratch
inputs, so a sample that only passes with constant indices is reported
separately. A full run takes about two minutes on four cores:

```bash
python tests/opinfo_conformance.py --check   # every recorded case still passes
python tests/opinfo_conformance.py --write   # record new passes, regenerate the page
python tests/opinfo_conformance.py sign round   # try a few operators
```

`--write` updates `tests/opinfo_passing.json` and
[`docs/operator-coverage.md`](operator-coverage.md). The test suite runs the
first recorded sample of each operator, and CI runs `--check`, so a case that
passes once has to keep passing. Record new passes with `--write` in the same
change that adds them. The list is tied to one PyTorch minor version, because
OpInfo's samples change between releases; with another version both checks
skip and ask for a refresh.

## Performance evidence

The emulator is the reference for correctness, not speed. The official
Scratch VM gives repeatable local timings, but users run the browser editor,
where browser, OS, hardware, Turbo Mode, and the rest of the project all affect
speed. Record the environment, and only compare candidates within one suite.

Related: [benchmark API reference](api-reference.md#benchmark-api) and the
[repository benchmark guide](../benchmarks/README.md).
