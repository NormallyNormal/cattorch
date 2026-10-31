# Stability

[Documentation home](index.md)

From 1.0, code written against cattorch and Scratch projects built around
exported sprites keep working across 1.x releases.

## What 1.x keeps

- **The Python API:** every name importable from `cattorch` itself, with its
  documented parameters, fields, and behavior. This includes `transpile`,
  `verify`, `build_project`, the tokenizers, the configuration classes, the
  result types, the exceptions, programs (`ExportProgram`, `EntryPoint`,
  `Input`, `Output`, `State`, `StateInput`, `StateUpdate`, `ProgramCall`), the
  markers, and `ModuleAdapter`.
- **The Scratch interface:** every block, list, variable, and broadcast in the
  [Scratch interface](scratch-interface.md), including the names
  [`build_project`](projects-and-cli.md#bridges) adds to a project.
- **The command line:** the `cattorch` commands and options in
  [projects and the command line](projects-and-cli.md).

A 1.x release can add names, parameters with defaults, result fields, Scratch
names, and supported operations. A name that is replaced keeps working, with a
`DeprecationWarning`, until 2.0.

## What can change

- **Experimental APIs**, imported from `cattorch.experimental`: the kernel
  interface (`Kernel`, `KernelContext`, `TensorInfo`, `check_kernel`,
  `build_kernel_benchmark`, `cattorch.scratch`, and `transpile(kernels=...)`),
  `op`, the explicit mixture-of-experts classes (`ExpertFamily`, `SparseMoE`,
  `StackedSwiGLUMoE`, `routed_experts`, `select_routes`, `RoutingScores`),
  and `analyze`. They can change in any release;
  the [changelog](../CHANGELOG.md) notes when they do.
- **Internals:** modules and names not importable from `cattorch` or
  `cattorch.experimental`, and names starting with `_`.
- **Sprite internals:** blocks, lists, and variables that aren't part of the
  Scratch interface, and the exact sprite size, block count, and speed.
- **Numbers within tolerance:** results of `optimization="fast"`, float16, or
  quantization, which can change as approximations improve. Exact exports keep
  matching PyTorch.
- **Wording:** the text of warnings and error messages. Exception classes
  don't change.

Programs, markers, and adapters were imported from `cattorch.experimental`
before 1.0. Those imports still work, with a `DeprecationWarning`.

Related: [Scratch interface](scratch-interface.md), [API reference](api-reference.md).
