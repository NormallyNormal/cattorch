# Verification and benchmarking

Correctness and speed need different test environments. The included emulator
is useful for deterministic functional checks. Performance must ultimately be
timed in Scratch itself; emulator or browser measurements from another machine
are not assumed to be proportional.

## Numerical verification

```python
from cattorch import transpile, verify

artifact = transpile(model, example, "model")
result = verify(model, example, artifact, atol=1e-4, rtol=1e-5)

print(result.passed)
print(result.max_abs_error, result.max_rel_error)
print(result.worst_index)
```

`verify` loads the generated sprite, runs one `cattorch forward` in the Python
emulator, and compares the flattened result with PyTorch. It supports sharded
input/output lists. It is especially useful after selecting float16 storage,
significant-figure rounding, fast kernels, pruning, or low-rank transforms.

Verification does not estimate Scratch performance.

## Scratch benchmark projects

Create a paired exact/fast project:

```python
from cattorch import build_paired_benchmark

build_paired_benchmark(model, example, "model_benchmark.sb3")
```

Bundle models into one sequential suite, with each result appended to a visible
list:

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

Storage and generation have dedicated suite builders:

```python
from cattorch import build_generation_benchmark_suite, build_storage_benchmark_suite

build_storage_benchmark_suite(
    [("model", model, example)],
    "storage_suite.sb3",
    iterations=10,
)

build_generation_benchmark_suite(
    [("decoder", decoder, token_example)],
    "generation_suite.sb3",
    iterations=10,
)
```

In Scratch, enable Turbo Mode, click the green flag, and wait for the results
list to complete. Initialization occurs before the timer, and generated calls
use no-refresh custom blocks. Save the completed project and inspect it with:

```bash
cattorch-benchmark downloaded-project.sb3
```

Scratch's timer can advance in coarse increments (about 0.033 seconds in one
reference browser). Increase `iterations` until fast kernels run comfortably
longer than the timer resolution. Browser results remain specific to that OS,
browser, and hardware; use the same target environment for comparisons.

See the
[benchmark development guide](https://github.com/NormallyNormal/cattorch/blob/main/benchmarks/README.md)
for repository scripts, suites, artifacts, and result formats.
