# Verification and benchmarking

[Documentation home](index.md)

`verify` checks that a sprite's output matches PyTorch. Benchmark projects
measure its speed in Scratch. `verify` runs in a Python emulator of Scratch, so
it can't tell you how fast the sprite is in a browser or whether it imports and
saves correctly.

## Numerical verification

```python
from cattorch import transpile, verify

artifact = transpile(model, example, "model")
result = verify(model, example, artifact, atol=1e-4, rtol=1e-5)

print(result.passed)
print(result.max_abs_error, result.max_rel_error)
print(result.worst_index)
```

`verify` loads the sprite, runs `cattorch forward` once in the emulator, and
compares the result with PyTorch value by value. Sharded lists are handled.
Run it after every change that can affect accuracy: float16 or integer storage,
`sig_figs`, fast mode, pruning, or low-rank transforms.

For a model with several tensor outputs, `verify` returns a
`MultiOutputVerifyResult`. `result.passed` is true only if every output passes,
and `result.outputs` has one `VerifyResult` per output, in the same order as
`artifact.outputs`:

```python
artifact = transpile(multi_output_model, example, "multi_output_model")
result = verify(multi_output_model, example, artifact)
for spec, comparison in zip(artifact.outputs, result.outputs):
    print(spec.list_name, comparison.passed, comparison.max_abs_error)
```

A model returning one tensor gets a `VerifyResult`, even if the tensor is
wrapped in a tuple, list, or dictionary. For stateful call sequences, see
[program verification](programs-and-moe.md#named-entrypoints-and-state).

## Scratch benchmark projects

Compare exact and fast mode for one model:

```python
from cattorch import build_paired_benchmark

build_paired_benchmark(model, example, "model_benchmark.sb3")
```

Time several models in one project. Each result is added to a list shown on
the stage:

```python
from cattorch import build_benchmark_suite

build_benchmark_suite(
    [
        ("small", small_model, small_input),
        ("large", large_model, large_input),
    ],
    "benchmark_suite.sb3",
    iterations=100,
)
```

Separate builders compare storage formats and cached generation:

```python
from cattorch import build_generation_benchmark_suite, build_storage_benchmark_suite

build_storage_benchmark_suite(
    [("model", model, example)],
    "storage_suite.sb3",
    iterations=10,
)

build_generation_benchmark_suite(
    [("decoder", decoder, stateless_input, prompt)],
    "generation_suite.sb3",
    iterations=10,
)
```

Load the project in Scratch, turn on Turbo Mode, click the green flag, and
wait until the results list is complete. Initialization happens before timing
starts. Save the finished project and read the results with:

```bash
cattorch-benchmark downloaded-project.sb3
```

Scratch's timer is coarse (about 0.033-second steps in browser runs so far),
so raise `iterations` until each result is many times longer than that. Repeat runs, and compare results only from the same
browser, machine, and Scratch settings.

Related: [benchmark API reference](api-reference.md#benchmark-api) and the
[repository benchmark guide](../benchmarks/README.md).
