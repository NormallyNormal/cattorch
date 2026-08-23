# Kernel development

cattorch kernels target sequential Scratch execution. Useful optimizations
usually reduce list operations, loop bookkeeping, materialized intermediates,
or repeated calculations. Desktop CPU/GPU techniques based on threads, SIMD,
memory bandwidth, or conventional quantized arithmetic generally do not
transfer directly.

## Production path

The public `exact` and `fast` modes use code-backed kernels built with
`cattorch.util.scratch.dsl`. The DSL represents Scratch expressions,
statements, loops, conditions, variables, and lists in Python, then lowers that
representation to Scratch JSON. Its `Program.pseudocode()` output is intended
for review and benchmark discussion.

Treat JSON as a lowering target, not source code. New production operations
should be expressed in the DSL or small structured block builders, tested in
the Python emulator for semantics, and benchmarked in Scratch for performance.

The main implementation boundaries are:

- `cattorch.graph` owns `torch.export`, constant folding, aliases, and fusion
  recognition;
- `cattorch.transpiler` turns analyzed nodes into instruction programs;
- `cattorch.sprite` assembles lifecycle, storage, sharding, and generation
  blocks around the compiled program;
- `cattorch.util.instruction.dispatch` selects ordinary exact/fast kernels;
- `optimized_linear`, `optimized_elementwise`, `optimized_normalization`,
  `optimized_tensor`, and `optimized_convolution` contain domain kernels;
- `optimized` is a compatibility facade, not an implementation module.

An operation-level change should normally have:

1. exact PyTorch comparisons covering shapes, edge cases, and broadcasting;
2. Scratch-emulator comparisons for generated blocks;
3. a paired or suite benchmark whose timer starts after `cattorch init`;
4. real-Scratch results with sufficient iterations to exceed timer resolution.

Fast kernels also need accuracy tests that show the intended degradation and a
fallback to exact behavior when the approximation is not enabled.

## Historical benchmark results

The retired JSON-template backend is not included in cattorch 0.4. Historical
legacy/exact Scratch results remain useful measurements, but new benchmark
projects compare the two supported production modes: exact and fast. New
kernels belong in the DSL-backed production registry.

## Performance evidence

The emulator is authoritative for the supported Scratch semantics, not speed.
The official VM can be useful for stable local comparisons, but the deployment
target is the real Scratch runtime. Browser, OS, hardware, Turbo Mode, and
surrounding project overhead can affect timings. Record the environment and
compare candidates within the same suite.
