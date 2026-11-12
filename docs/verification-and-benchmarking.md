# Verification and benchmarking

[Documentation home](index.md)

`verify` checks that a sprite's output matches PyTorch. Benchmark projects
measure its speed in Scratch. `verify` runs in a Python emulator of Scratch, so
it can't tell you how fast the sprite is in a browser or whether it imports and
saves correctly.

## Numerical verification

```python
import torch
from torch import nn

from cattorch import QuantizationConfig, transpile, verify

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).eval()
example = torch.randn(2, 4)

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

Give `verify` the export's `adapters`, `quantization`, `storage` and
`sig_figs`, and PyTorch runs a copy of the model holding the values the
sprite stores. The comparison then measures the conversion alone, with tight
tolerances, however lossy the storage:

```python
quantization = QuantizationConfig(bits=8, min_quantized_values=1)
quantized = transpile(model, example, "model_q8", quantization=quantization)
assert verify(model, example, quantized, quantization=quantization).passed
# Against the float model, the same sprite needs a looser tolerance.
print(verify(model, example, quantized).max_abs_error)
```

For a model with several tensor outputs, `verify` returns a
`MultiOutputVerifyResult`. `result.passed` is true only if every output passes,
and `result.outputs` has one `VerifyResult` per output, in the same order as
`artifact.outputs`:

```python
class TwoOutputs(nn.Module):
    def forward(self, value):
        return value.relu(), value.sum(-1)

multi_output_model = TwoOutputs()
artifact = transpile(multi_output_model, example, "multi_output_model")
result = verify(multi_output_model, example, artifact)
for spec, comparison in zip(artifact.outputs, result.outputs):
    print(spec.list_name, comparison.passed, comparison.max_abs_error)
```

A model returning one tensor gets a `VerifyResult`, even if the tensor is
wrapped in a tuple, list, or dictionary.

`verify` works for every kind of export. For a program, `calls` runs a
sequence of `ProgramCall`s with state carried between them (see
[program verification](programs-and-moe.md#named-entrypoints-and-state)). For
a `GenerationProgram` or `RecurrentGenerationProgram`, `calls` names the
blocks `"prefill"`, `"decode"` and `"reset"`; each call's logits and top-k
lists are compared with the model's, and its `cattorch status` with the one
its input should give (see [generation](generation.md#verify-a-generator)).

`verify` uses cattorch's emulator, not Scratch itself. cattorch's own tests
also run a set of exported sprites in the official `@scratch/scratch-vm`: an
MLP, a CNN, a transformer layer, each weight storage format, a
variable-length input, a stateful program, and cached generation with its
tokenizer. To check your own project the same way, use
`benchmarks/run_project_vm.cjs` (see the
[benchmarks README](../benchmarks/README.md)).

## Scratch benchmark projects

Every builder takes `repeats`, the timed calls per variant, `warmups`,
untimed calls first, and `transpile`'s export options (`storage`,
`quantization`, `fast_config`, `sig_figs`, `codegen`, `adapters`, ...).
Compare exact and fast mode for one model:

```python
from cattorch import build_paired_benchmark

build_paired_benchmark(model, example, "model_benchmark.sb3", repeats=7)
```

Time several models in one project. Each result is added to a list shown on
the stage:

```python
from cattorch import build_benchmark_suite

small_model, small_input = model, example
large_model = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 3)).eval()
build_benchmark_suite(
    [
        ("small", small_model, small_input),
        ("large", large_model, example),
    ],
    "benchmark_suite.sb3",
    repeats=100,
)
```

Separate builders compare storage formats, and a text generator's cached
decode with a stateless forward. A generation case takes the program
`transpile` would, and a prompt:

```python
import sys
from pathlib import Path

from cattorch import (
    GenerationProgram, build_generation_benchmark_suite, build_storage_benchmark_suite,
)

build_storage_benchmark_suite(
    [("model", model, example)],
    "storage_suite.sb3",
    repeats=10,
)

class Decoder(nn.Module):
    def __init__(self, vocabulary=16, width=8):
        super().__init__()
        self.embed = nn.Embedding(vocabulary, width)
        self.head = nn.Linear(width, vocabulary)

    def forward(self, tokens):
        hidden = self.embed(tokens)
        attended = nn.functional.scaled_dot_product_attention(
            hidden, hidden, hidden, is_causal=True,
        )
        return self.head(hidden + attended)

decoder, prompt = Decoder().eval(), torch.tensor([[1, 2, 3, 4]])
build_generation_benchmark_suite(
    [("decoder", decoder, GenerationProgram(max_context=16), prompt)],
    "generation_suite.sb3",
    repeats=10,
)
```

Load the project in Scratch, turn on Turbo Mode, click the green flag, and
wait until the results list is complete. Initialization happens before timing
starts. Save the finished project and read the results with
`analyze_benchmark`, which returns a `BenchmarkAnalysis`, or from the shell:

```bash
cattorch analyze-benchmark downloaded-project.sb3
```

Each `BenchmarkCase` gives the seconds one call of each variant took and the
speedup. Scratch's timer is coarse (about 0.033-second steps in browser runs
so far), so raise `repeats` until each result is many times longer than
that. Repeat runs, and compare results only from the same browser, machine,
and Scratch settings.

Related: [benchmark API reference](api-reference.md#benchmark-api) and the
[repository benchmark guide](../benchmarks/README.md).
