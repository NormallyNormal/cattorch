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

## Performance evidence

The emulator is the reference for correctness, not speed. The official
Scratch VM gives repeatable local timings, but users run the browser editor,
where browser, OS, hardware, Turbo Mode, and the rest of the project all affect
speed. Record the environment, and only compare candidates within one suite.

Related: [benchmark API reference](api-reference.md#benchmark-api) and the
[repository benchmark guide](../benchmarks/README.md).
