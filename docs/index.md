# cattorch documentation

cattorch exports PyTorch models as Scratch 3 sprites that run in unmodified
Scratch, with no extensions.

If you're new, start with the [getting-started guide](getting-started.md). It
covers exporting a model, checking its output, importing the sprite, and
connecting it to the rest of a project.

## Use cattorch

- [Getting started](getting-started.md): export and integrate a small model.
- [Supported models and operations](supported-models.md): which models and
  PyTorch operations can be exported.
- [Operator coverage](operator-coverage.md): how PyTorch's own operator samples
  fare, generated from a test run.
- [Optimization modes](optimization.md): exact or faster approximate kernels.
- [Storage and quantization](storage.md): float16, integer quantization, GPTQ,
  and Scratch's size limits.
- [KV-cached generation](generation.md): export a language model and generate
  text one token at a time.
- [Tokenizers](tokenizers.md): export character, BPE, or SentencePiece
  tokenizers.
- [Projects and the command line](projects-and-cli.md): build a runnable
  Scratch project around exported sprites, and export from the terminal.
- [Verification and benchmarking](verification-and-benchmarking.md): check
  output against PyTorch and measure speed in Scratch.
- [Troubleshooting](troubleshooting.md): fix export failures and report
  issues.

## Reference

- [API reference](api-reference.md): functions, configuration classes, result
  types, benchmark builders, and exceptions.
- [Scratch interface](scratch-interface.md): the blocks, lists, variables, and
  broadcasts of exported sprites, versioned for 1.x.
- [Code generation and JSON size](code-generation.md): trade sprite size for
  speed.
- [Programs and sparse MoE](programs-and-moe.md): multiple methods,
  persistent state, and mixture-of-experts models.
- [Markers](markers.md): declare entry points, state, MoE routing, and
  export-only methods with decorators on your own classes.
- [Custom kernels and operations](custom-kernels.md): write Scratch kernels
  and declare new operations (experimental).
- [Stability](stability.md): what cattorch 1.x keeps compatible, and what's
  experimental.
- [Glossary](glossary.md): cattorch and Scratch terms used in these guides.

## Contribute

- [Kernel development](kernels.md): code layout, tests, and performance
  measurement for contributors.
- [Benchmarks](../benchmarks/README.md): benchmark scripts and recorded
  results.
- [Changelog](../CHANGELOG.md): release history.
