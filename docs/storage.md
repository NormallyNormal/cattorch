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

A matrix whose integer storage wouldn't be smaller than float16 stays float16,
unrounded: the codes and a scale per group would take 16 bits per value or
more, as with `group_size=1`, a prime input width such as 97 (whose only
divisor up to 64 is 1), or int8 with float32 scales and groups of 4 or fewer.
The cost is the storage codec's own for that tensor (for int4, its Huffman
code). The report gives such a matrix `method="float16"` and says why.

cattorch runs the model once to find each matrix's input width. A weight read
by `nn.Linear` or `F.linear`, on either side of a matrix product (`x @ weight`,
`weight @ x`, `torch.mm`, `torch.bmm`, `torch.mv`), by `torch.addmm` (as in
GPT-2's `Conv1D` layers), `torch.baddbmm` or `torch.addmv`, by a two-operand
`torch.einsum` or `torch.tensordot` that contracts it over one dimension, or
by `F.bilinear` and `nn.Bilinear` is grouped over its inputs, and GPTQ
calibrates it. So are `nn.MultiheadAttention`'s input projection, its output
projection when the call passes no mask, and the input and hidden weights of
every layer and direction of `nn.RNN`, `nn.LSTM` and `nn.GRU`. A projected
LSTM's projection (`weight_hr_l*`) reads a value no output shows, so it uses
symmetric quantization, and so does every layer but the first when the
module runs with dropout in training mode. A 1D or 2D convolution is a
linear map of each input window, so its kernels are grouped like a linear
layer's, over each output channel's fan-in (input channels times kernel
positions): a 3x3 convolution from 64 channels has groups of 64 of its 576
inputs. A 3D convolution is lowered to a 2D convolution per kernel depth,
and a transposed convolution to a plain one with flipped kernels, so their
groups stay within the last two kernel dimensions (the last one, in 1D),
such as 9 values of a 3x3x3 kernel. GPTQ calibrates each group of kernels
over the input windows it reads, for 1D, 2D and 3D, grouped, depthwise and
transposed convolutions. Other matrices, such as embeddings, are grouped
along their last dimension and use symmetric quantization; with GPTQ,
`result.warnings` names each of them.

Each part of a parameter that a use reads, such as the slices of a fused
QKV weight, even and odd rows, or one half read by `F.linear` and the other
on the right of a matrix product, is quantized once, grouped over that use's
inputs, and reported as its own entry (`qkv[offset=...,shape=...]`). Uses
that read the same values the same way, such as `F.linear(h, weight)` and
`h @ weight.T`, share one entry and GPTQ calibrates it with both. Values no
such use reads are grouped along the parameter's last dimension, reported as
`name[remaining]` when the rest of the parameter is quantized for its uses.
When two uses read the same values grouped along different axes, as
`F.linear(x, weight)` and `x @ weight` do, only one layout can be stored in
integers: the larger use's, or the first one's.

A weight the model copies before using it, such as
`F.linear(x, weight.t().contiguous())` or `x @ weight.clone()`, is quantized
and calibrated as the copy is used, and the copy is stored in integers.

A weight computed from parameters before any input arrives, such as a
`weight_norm` or `spectral_norm` weight (`torch.nn.utils.parametrizations` or
the older `torch.nn.utils.spectral_norm`), `F.linear(x, w * g)`, a LoRA sum
`w + b @ a`, `w.clamp(...)`, a masked `w * mask`, a scaled tied embedding or
an average of a weight and a buffer, is quantized as computed: grouped as
the matrix product reads it and calibrated by GPTQ like a parameter, while
the parameters it is computed from stay unrounded. The model then computes
with the quantized weight in place of the computation, both in the sprite
and in every PyTorch comparison cattorch makes (`verify` with an
`ExportProgram`, the length check of `dynamic=`). The report names a
parametrized or `spectral_norm` weight by its attribute (`fc.weight`), and
any other one `weight computed from <parameters>`, and lists the
parameters in `sources`. A weight computed outside the forward of any of the
model's modules, as in an `ExportProgram` entry point method, or within an
operation that also reads an input, is stored as computed, in float16, with
a warning and a `method="float16"` entry. The older
`torch.nn.utils.weight_norm` usually can't be exported, because PyTorch
can't copy a module that uses it; use
`torch.nn.utils.parametrizations.weight_norm`.

The report says what storage stores. Storage keeps a list in integers only
when one of the integer encodings quantization uses reproduces it exactly, so
a quantized value read in another layout is kept in float16. Each stored list is traced
back to the parameter values it holds: an entry whose values are all stored
in float16 is reported with `method` `"float16"` and the reason, and
values also or only stored in float16 elsewhere get a `name[float16]` entry.
`result.warnings` names the parameter each time.

Matrices with fewer than `min_quantized_values=16384` values stay float16,
because the integer decoder costs more than it saves on small tensors. Set
`min_quantized_values=1` to quantize every eligible matrix. A matrix is a
parameter read as one by the uses above, or one with at least two
dimensions larger than one (for an expert bank, per expert). Vectors stay
float16: biases, normalization parameters, `nn.MultiheadAttention`'s
`(1, 1, E)` `bias_k` and `bias_v`, and `(N, 1)` gains such as
`weight_norm`'s. Buffers such as RoPE tables and masks use float16 too.
Values that can't safely be stored lossily stay float32.

Inspect `result.warnings` and `result.quantization.tensors`, then verify
model-level output or generation quality.

## Per-weight settings

`QuantizationConfig.overrides` gives some weights other settings. Each key
is a module path or a parameter name, as `model.named_parameters()` spells
it, or a computed attribute such as a parametrized `fc.weight`; the value is
bits (4, 6 or 8), `None` to keep the weights float16, or a
`QuantizationOverride` with `bits`, `method` and `group_size`:

```python
from cattorch import QuantizationConfig, QuantizationOverride

QuantizationConfig(
    bits=4,
    method="gptq",
    overrides={
        "lm_head": 8,                      # the output head in int8
        "transformer.h.0": QuantizationOverride(bits=6, group_size=32),
        "transformer.h.11.mlp.c_proj": None,  # float16
    },
)
```

A key selects the parameters under it, and the longest matching key wins.
A key that selects no parameter is an error. A weight read under several
names, such as GPT-2's token embedding `transformer.wte.weight` and its tied
output head `lm_head.weight`, takes the highest precision its names select
(float16 above 8 bits above 6 and 4), so `{"lm_head": 8}` keeps the shared
table in int8. A weight computed from parameters takes the highest precision
of the overrides that select its parameters or its attribute name.
`method="gptq"` needs a GPTQ config; `method="symmetric"` within one skips
calibration for those weights. Each `TensorQuantizationReport` names the
`override` it used. One sprite can hold several integer formats: each is
decoded by its own bank, and the int4 Huffman decoder is added only when
some weight uses int4.

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
calibrate with realistic full sequences of up to `max_context` tokens.

A matrix that saw too little calibration data, such as an expert the router
never picked, falls back to symmetric quantization, and so does one with
more than 8,192 inputs, whose GPTQ statistics would take over 512 MiB. Small
matrices stay float16. The quantization report shows which method each
tensor used.

GPTQ applies its error feedback within a block of `block_size` columns
column by column, and to the columns after the block as one matrix product
(the lazy batch update of the GPTQ paper), which makes wide matrices several
times faster to calibrate.

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

Compressed weights are stored in lists as text. To decode them, the sprite
switches to the costume named after each character and reads the costume
number, so its costumes are a lookup table for the decoder. Don't rename,
reorder, remove, or add costumes on that sprite, or the weights will load
incorrectly. Put any visible costumes on a different sprite.

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

A token embedding tied to the output head, as in GPT-2 and Mamba, is stored
once even when it spans several lists: the embedding reads the head's shards.
When another operation or a custom kernel needs the table in a different
layout from the head's, it is stored twice and `result.warnings` says so.

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
copies. Model, program and generation sprites have a `cattorch prepare for save`
block that clears the decoded weights and working data. Call it before saving;
the next call initializes the sprite again. A generation sprite's
`cattorch prepare for save` also resets it, as `cattorch reset` does, since its
cache is too large to save: prefill again afterwards.

Related: [verification](verification-and-benchmarking.md) and
[result types](api-reference.md#result-types).
