# Storage, quantization, and Scratch limits

[Documentation home](index.md)

Storage settings control how parameters, buffers, and constants are saved in
the sprite. Compressed values are decoded once during `cattorch init`, and
inference then runs on ordinary Scratch number lists.

## Choose a storage mode

| Goal | Configuration | Lossy? |
|---|---|---|
| Preserve PyTorch float32 values | `StorageConfig()` | No |
| Reduce size with float16 weights | `StorageConfig(precision="float16")` | Yes |
| Quantize without calibration | `QuantizationConfig(bits=6)`; also accepts 4 or 8 | Yes |
| Use representative data to choose low-bit weights | `QuantizationConfig(bits=4, method="gptq")` | Yes |
| Disable compression for debugging | `StorageConfig(compression=False)` | No |

Storage affects file size, startup time, and accuracy. It doesn't make
inference faster, because inference uses the decoded weights.

Start with the default float32 storage and run `verify`. Try float16 next. For
a large model, compare symmetric Q6, symmetric Q4, and GPTQ Q4 on
representative data. Fewer bits give a smaller sprite and can lower model
quality. Lossy storage stays lossy with `optimization="exact"`.

## Examples

```python
from cattorch import QuantizationConfig, StorageConfig, transpile

# Lossless for PyTorch float32 values.
transpile(model, example, "model", storage=StorageConfig())

# Smaller, lossy storage.
transpile(
    model,
    example,
    "model_f16",
    storage=StorageConfig(precision="float16"),
)

# Six-bit weights without calibration.
transpile(
    model,
    example,
    "model_q6",
    quantization=QuantizationConfig(bits=6),
)
```

See the [configuration reference](api-reference.md#storageconfig) for every
field and default.

## Symmetric quantization behavior

Each row of a weight matrix is split into groups, and each group gets its own
scale. `group_size=64` is a maximum: cattorch uses the largest divisor of the
matrix input width that doesn't exceed it, such as 56 for a width of 112.
Smaller groups usually reduce error but store more scales. Scales are float16 by
default.

Matrices with fewer than `min_quantized_values=16384` values stay float16,
because the integer decoder costs more than it saves on small tensors. Set
`min_quantized_values=1` to quantize every eligible matrix. Biases,
normalization parameters, and buffers such as RoPE tables and masks use
float16. Values that can't safely be stored lossily stay float32.

Inspect `result.warnings` and `result.quantization.tensors`, then verify
model-level output or generation quality.

## GPTQ calibration

GPTQ runs the model on representative inputs to choose better integer weights
during export. It doesn't modify your model or slow down inference in
Scratch.

```python
result = transpile(
    model,
    example,
    "model_gptq_q4",
    quantization=QuantizationConfig(bits=4, method="gptq"),
    calibration_inputs=calibration_batches,
)

for tensor in result.quantization.tensors:
    if tensor.fallback_reason:
        print(tensor.name, tensor.fallback_reason)
```

Choose the calibration argument to match the export interface:

| Export interface | Calibration argument | Items |
|---|---|---|
| Tensor example inputs | `calibration_inputs` | Tensors or tuples matching the eager model's inputs |
| `ExportProgram` | `calibration_calls` | Ordered `ProgramCall` values exercising the model's methods and state |
| `GenerationProgram` | `calibration_sequences` | Full token sequences accepted by the generation method |

Calibration data is read once, so a generator works. For cached generation,
keep the export example at one token (`[1, 1]`) but calibrate with realistic
full sequences. The model method must accept those sequence lengths.

A matrix that saw too little calibration data, such as an expert the router
never picked, falls back to symmetric quantization. Small matrices stay
float16. The quantization report shows which method each tensor used.

The [`QuantizationConfig` reference](api-reference.md#quantizationconfig)
lists the GPTQ settings. GPTQ reduces per-layer error, but that doesn't always
mean better results on your task, so compare symmetric and GPTQ exports on real
use.

## Significant-figure rounding

`sig_figs=N` rounds stored values to N significant figures before encoding.
It is lossy and works in either optimization mode.

When trying several lossy options (storage precision, quantization, `sig_figs`,
fast mode), test each one on its own before combining them, so you can tell
which one causes a drop in accuracy.

## Preserve processor costumes

Compressed weights are stored in the generated sprite's costume names. Don't
rename, reorder, remove, or add costumes on that sprite, or the weights will
load incorrectly. Put any visible costumes on a different sprite.

## List sharding

Scratch limits each list to 200,000 items. cattorch splits larger tensors
across several lists, called shards:

- The first shard keeps the original name.
- Later shards are named `<name> shard 2`, `<name> shard 3`, and so on.
- The generated blocks read and write every shard.
- For each split public list, the variables `cattorch <name> shard count` and
  `cattorch <name> logical length` hold the number of shards and total items.

If your own scripts copy a sharded input or output, copy every shard in order,
not only the first. `result.sharded_lists` names the lists that were split.
`verify` handles shards for you.

## Project size

cattorch reports `archive_bytes` (the compressed file) and
`expanded_json_bytes` (the uncompressed sprite data) separately. A small file
can still hold too much JSON for Scratch.

The exporter warns when expanded JSON passes 4,000,000 bytes, again past
5,000,000 bytes (the limit Scratch's online editor uses when saving), and when
it exceeds `CodegenConfig.target_json_bytes`. No warning doesn't guarantee the
project will upload or save: the stage and other sprites add to the total.

To reduce size, try lower-precision storage and
[compact code generation](code-generation.md).

## Save after running

After a run, the sprite holds both the compressed weights and the decoded
copies. Model and `ExportProgram` sprites have a `cattorch prepare for save`
block that clears the decoded weights and working data. Call it before saving;
the next model call initializes the sprite again.

`GenerationProgram` sprites don't have this block. Their `cattorch reset`
clears the cache and outputs but leaves the decoded weights loaded, so a
project saved after generating will be larger.

Related: [verification](verification-and-benchmarking.md) and
[result types](api-reference.md#result-types).
