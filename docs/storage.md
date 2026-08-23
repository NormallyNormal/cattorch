# Storage and Scratch limits

Static tensors use a Scratch-safe costume-name Base85 codec by default.
`cattorch init` decodes the payload once into ordinary numeric lists, so
forward kernels do not parse encoded values. The processor sprite contains 85
costume entries which all reference the same physical SVG asset; vanilla
Scratch's case-sensitive costume lookup supplies the digit value.

Those 85 costume entries are reserved codec metadata. Do not rename, reorder,
delete, or insert costumes in a generated processor sprite: changing their
one-based positions changes decoded bytes. Keep visual costumes on another
sprite.

```python
from cattorch import StorageConfig, transpile

# Lossless for PyTorch float32 values.
transpile(model, example, "model", storage=StorageConfig())

# Smaller, explicitly lossy payload.
transpile(
    model,
    example,
    "model_f16",
    storage=StorageConfig(precision="float16"),
)

# Groupwise weight quantization, decoded once before the first forward pass.
transpile(
    model,
    example,
    "model_int8",
    storage=StorageConfig(precision="int8", group_size=64),
)

# Bit-packed signed 6-bit weights: just under one character each.
transpile(
    model,
    example,
    "model_int6",
    storage=StorageConfig(precision="int6", group_size=64),
)

# Two signed weights are packed into each byte.
transpile(
    model,
    example,
    "model_int4",
    storage=StorageConfig(precision="int4", group_size=64),
)

# Useful as an uncompressed debugging or benchmark baseline.
transpile(
    model,
    example,
    "model_plain",
    storage=StorageConfig(compression=False),
)
```

Float32 storage uses four bytes per value before Base85 representation; float16
uses two, int8 uses one, int6 packs four six-bit codes into three bytes, and
packed int4 uses half a byte. Before scale metadata, that is approximately
5.000, 2.500, 1.250, 0.938, and 0.625 safe characters per weight respectively.
Because Scratch project JSON is textual, Base85 also
avoids the long decimal spellings of ordinary JSON numbers. The encoded
representation costs five characters per four bytes, not one
character per arbitrary byte, because Scratch and JSON cannot safely carry
every possible raw byte as a single character.

Int8, int6, and int4 use symmetric quantization independently in each flat
group. Each group's scale is compressed separately. Integer codes are centered
so startup decoding needs no signed-value branch. Matrix-like tensors are
dequantized into ordinary Scratch numeric lists during `cattorch init`, so
forward kernels do not parse packed values, access scale metadata, or perform
integer-specific arithmetic. Small one-dimensional tensors such as biases and
normalization parameters use float16 rather than carrying their own integer
metadata. Integer storage therefore reduces
project size and startup decoding work, but it does not bypass the 200,000-item
limit of each decoded list and does not inherently accelerate a forward pass.

The default group size is 64. Scales default to float32; setting
`scale_precision="float16"` halves their payload at the cost of a small,
explicit additional approximation. Smaller groups usually reduce quantization error
at the cost of more scale data. All integer formats are explicitly lossy;
verify model-level outputs or generation quality before distribution.

By default, matrix shards with fewer than 16,384 values remain float16. For
small tensors, the Scratch blocks and scale payload needed by an integer
decoder can cost more project space and startup time than they save. Set
`min_quantized_values=1` to force integer storage for every matrix shard,
for example when measuring the formats themselves. The selected storage
precision therefore describes the requested maximum compression; an export
may contain a mixture of integer matrices, float16 small matrices, and float32
one-dimensional parameters.

`sig_figs=N` rounds static values before storage encoding. It is an independent,
lossy size/precision control and can be used in either optimization mode.

Static tensors with bit-identical flattened values share one physical payload,
even when their PyTorch shapes or graph names differ. Tied token embeddings and
four-row-grouped output heads also share one physical representation. Oversized
tied matrices use row-group-aligned physical shards; embedding lookup routes the
global index across those shards and reads the grouped layout with a stride-four
loop. Startup decoding is shared per precision rather than being regenerated
for every tensor. Encoded tensor bytes are coalesced into one payload bank per
precision. Quantization scales are coalesced within the corresponding capped
weight bank, and both temporary byte streams remain below Scratch's list
limit. Tensor wrappers carry only byte/scale offsets, so adding tensors no longer duplicates
the large Base85/IEEE decoder or creates a separate scale-payload variable for
every matrix. Forward kernels still receive ordinary tensor-specific numeric
lists, avoiding bank-routing work in each multiply-accumulate loop.

An experimental learned 64-entry scalar codebook was screened on the current
TinyStories checkpoint. It was smaller in principle but lost substantially
more top-1 agreement than symmetric int6, so it is not exposed as a production
`StorageConfig` precision.

## List sharding

Scratch limits each physical list to 200,000 items. cattorch automatically
splits larger logical tensors:

- shard one retains the original name;
- later lists are named `<name> shard 2`, `<name> shard 3`, and so on;
- generated kernels route reads, writes, lengths, clears, and sequential
  appends across those physical lists;
- visible `cattorch <name> shard count` and `cattorch <name> logical length`
  variables describe oversized non-local tensors.

Normal one-list kernels are retained when sharding is unnecessary, avoiding
the extra branch and index arithmetic for typical tensors. KV caches use the
same mechanism.

Large static Linear, static-matmul, paired-SwiGLU, and cached-QKV matrices
receive a stronger specialization: cattorch aligns physical lists to complete
output rows or four-row groups and emits separate sequential loops for each
list. No shard branch, modulo, or multi-list read remains inside their dot
products. The generic router remains the correctness fallback for irregular
dynamic access.

## Project size

Scratch clients and hosting paths impose different archive, asset, and
expanded-JSON constraints. The ordinary online editor save/upload path has an
approximately 5 MiB expanded `project.json` limit; that is not a universal
5 MiB cap on compressed `.sb3` or `.sprite3` archives. Legacy/offline archive
paths have accepted larger compressed projects, while individual assets have
their own limits. cattorch reports compressed archive and expanded sprite JSON
sizes separately and warns as expanded JSON approaches the ordinary online
limit. Treat a real import and save through the intended Scratch path as the
final acceptance test.

## Saving a project after running it

Decoded lists coexist with compressed payloads while a project is running.
Before saving, call `cattorch prepare for save`. It clears decoded weights and
runtime tensor data, preserves payloads, and rearms initialization. The next
forward call reconstructs the weights.
