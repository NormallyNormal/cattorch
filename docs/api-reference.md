# Public API reference

[Documentation home](index.md)

Functions, configuration classes, and result types are imported from
`cattorch`. Paths can be strings or `pathlib.Path` objects.

Experimental programs, state, adapters, analysis, and MoE classes are imported
from `cattorch.experimental`; see
[experimental programs and sparse MoE](programs-and-moe.md).

## Model export

### `transpile`

```text
transpile(
    model,
    example_inputs,
    output_path,
    sig_figs=None,
    *,
    name=None,
    optimization="exact",
    fast_config=None,
    storage=None,
    quantization=None,
    calibration_inputs=None,
    codegen=None,
    frontend="fx",
    adapters=(),
    calibration_calls=None,
    calibration_sequences=None,
) -> TranspileResult
```

Export an evaluation-mode `torch.nn.Module` as a Scratch `.sprite3` file.

| Parameter | Meaning |
|---|---|
| `model` | Module to export, in evaluation mode. Weight transforms and GPTQ work on a copy. |
| `example_inputs` | A tensor or tuple of tensors, an `ExportProgram`, or a `GenerationProgram`. Programs carry their own examples. |
| `output_path` | Destination. `.sprite3` is appended when absent and missing parent directories are created. |
| `sig_figs` | Round stored values to this many significant figures. Lossy. |
| `name` | Scratch sprite display name. Defaults to the destination filename stem. |
| `optimization` | `"exact"` (default) or `"fast"`. See [optimization modes](optimization.md). |
| `fast_config` | A `FastConfig`; valid only with `optimization="fast"`. |
| `storage` | A `StorageConfig` for weight compression and float precision. |
| `quantization` | A `QuantizationConfig` for symmetric or GPTQ integer weights. Don't combine with an integer `StorageConfig.precision`. |
| `calibration_inputs` | Iterable of tensors or tensor tuples for GPTQ with tensor inputs. Read once. Only valid with `method="gptq"`. |
| `codegen` | A `CodegenConfig` for JSON size target, loop unrolling, IDs, and compaction. |
| `frontend` | How the model is traced: `"fx"` (default) or `"export"` (`torch.export`). Programs require `"fx"`. |
| `adapters` | `ModuleAdapter`s for an `ExportProgram` or `GenerationProgram`. |
| `calibration_calls` | `ProgramCall`s for GPTQ with an `ExportProgram`. |
| `calibration_sequences` | Token sequences for GPTQ with a `GenerationProgram`. |

Tensor-input and generation exports return a `TranspileResult`. An
`ExportProgram` export returns a `ProgramResult`. Export raises
`UnsupportedOperationError` for an operation cattorch can't convert,
`UnsupportedModelError` for other unsupported models, and `ValueError` or
`TypeError` for invalid options. See [troubleshooting](troubleshooting.md).

With an `ExportProgram`, the call looks like this and returns a
`ProgramResult`:

```text
transpile(
    model,
    program,
    output_path,
    *,
    adapters=(),
    calibration_calls=None,
    ...
) -> ProgramResult
```

See [programs and sparse MoE](programs-and-moe.md) for interface names,
state updates, and calibration.

A `GenerationProgram` export also accepts `adapters`, for example to convert
your MoE module to `SparseMoE` or `StackedSwiGLUMoE`.

### `rotary_embedding`

```text
rotary_embedding(value, cosine, sine) -> Tensor
```

Apply rotary position embeddings (RoPE) to adjacent pairs of features. The
last dimension of `value` must be even. `cosine` and `sine` must broadcast to
`value`'s shape without making it larger.

## Tokenizer export

### `transpile_tokenizer`

```text
transpile_tokenizer(
    tokenizer,
    output_path,
    *,
    name=None,
    scratch_casefold=False,
    codegen=None,
) -> TokenizerResult
```

Detect the tokenizer type (character, raw-text Hugging Face BPE, or
SentencePiece BPE) and export it. Use `scratch_casefold=True` only for a
SentencePiece model trained with case-folding normalization.

### `CharTokenizer` and `BPETokenizer`

```text
CharTokenizer(tokenizer).save(output_path, *, name=None, codegen=None)
BPETokenizer(tokenizer).save(output_path, *, name=None, codegen=None)
```

Export a Hugging Face tokenizer as a character or BPE tokenizer, without
auto-detection. The tokenizer must have `get_vocab()`. `BPETokenizer` also
needs a fast (Rust-backed) BPE tokenizer so cattorch can read its merges.

### `SentencePieceBPETokenizer`

```text
SentencePieceBPETokenizer(
    tokenizer,
    *,
    scratch_casefold=False,
).save(output_path, *, name=None, codegen=None)
```

`tokenizer` can be a `SentencePieceProcessor` or an object with one in its
`.processor` attribute. See [tokenizers](tokenizers.md) for normalization,
Unicode, and byte fallback limits.

## Verification

### `verify`

```text
verify(
    model,
    example_inputs,
    sprite,
    *,
    atol=1e-4,
    rtol=1e-5,
    calls=None,
    adapters=(),
    quantization=None,
) -> VerifyResult | MultiOutputVerifyResult | ProgramVerifyResult
```

Run `cattorch forward` once in cattorch's Python emulator and compare the
result with PyTorch. `sprite` can be a path or a `TranspileResult`. Tolerances
must be non-negative. `verify` checks correctness, not speed.

A model with one tensor output returns a `VerifyResult`, even if the tensor is
inside a container. A model with several returns a `MultiOutputVerifyResult`,
whose `outputs` are in the same order as `TranspileResult.outputs` and whose
`passed` requires all of them to pass. Non-tensor outputs, or outputs that
don't match the sprite's lists, raise `UnsupportedModelError`.

The remaining arguments apply only to an `ExportProgram`:

- `calls`: a tuple of `ProgramCall`s run in order, keeping state between them
  in both PyTorch and Scratch. The result is a `ProgramVerifyResult`.
- `adapters`: the same adapters used at export.
- `quantization`: a symmetric `QuantizationConfig`. PyTorch then runs with the
  quantized weights, so the comparison shows conversion error separately from
  quantization error. GPTQ isn't supported here.

## Configuration objects

Configuration classes are immutable dataclasses. Invalid values raise an
error when the object is created.

### `FastConfig`

| Field | Default | Meaning |
|---|---:|---|
| `activations` | `True` | Use fast activation approximations (currently QuickGELU for GELU). |
| `layer_norm` | `True` | Compute LayerNorm variance in one pass. |
| `softmax` | `True` | Skip max subtraction in softmax. |
| `neuron_pruning` | `0.0` | Fraction in `[0, 1)` of hidden units to remove from Linear → activation → Linear paths and `StackedSwiGLUMoE` experts. |
| `weights` | `FastLayerConfig()` | Weight transforms applied to every eligible layer. |
| `overrides` | `{}` | Per-module `FastLayerConfig`s, keyed by module name, that replace `weights` for that module. |

### `FastLayerConfig`

| Field | Default | Meaning |
|---|---:|---|
| `pruning` | `0.0` | Fraction in `[0, 1)` of weight groups to remove, smallest magnitude first. |
| `rank` | `None` | Truncated-SVD rank. Can't be combined with `rank_ratio`. |
| `rank_ratio` | `None` | Rank as a fraction in `(0, 1]` of the matrix's full rank. Can't be combined with `rank`. |

A transform is skipped for a layer when it wouldn't reduce the number of
multiplications. See [optimization modes](optimization.md#weight-transforms).

### `GenerationProgram`

| Field | Default | Meaning |
|---|---:|---|
| `method` | required | Name of the model method that takes token IDs and returns next-token logits. |
| `example_token` | required | A `[1, 1]` integer tensor used to trace the method. |
| `max_context` | required | Maximum number of tokens the cache holds. |
| `hidden_prefill` | `True` | During prefill, compute logits only for the last prompt token. |
| `top_k` | `None` | If set (1 to 64), also write the top-k logits and their token IDs after each prefill or decode. |

See [KV-cached generation](generation.md) for model requirements and the
Scratch blocks.

### `StorageConfig`

| Field | Default | Meaning |
|---|---:|---|
| `compression` | `True` | Compress stored tensors and decode them during initialization. |
| `precision` | `"float32"` | `"float32"` or lossy `"float16"`. |

The integer precisions `"int4"`, `"int6"`, and `"int8"` are deprecated; use
`QuantizationConfig` instead. These fields only apply to the deprecated
integer precisions:

| Field | Default | Meaning |
|---|---:|---|
| `group_size` | `64` | Integer group size. |
| `min_quantized_values` | `16384` | Threshold below which eligible tensors remain float16. |
| `scale_precision` | `"float32"` | Integer scale precision. |
| `grouping` | `"flat"` | Integer grouping, `"flat"` or `"row"`. |

### `QuantizationConfig`

| Field | Default | Meaning |
|---|---:|---|
| `bits` | required | Integer weight width: `4`, `6`, or `8`. |
| `method` | `"symmetric"` | `"symmetric"` (round to nearest) or `"gptq"` (uses calibration data). |
| `group_size` | `64` | Maximum group size. The actual size is the largest divisor of the matrix input width that doesn't exceed this. |
| `scale_precision` | `"float16"` | `"float16"` or `"float32"` scales. |
| `min_quantized_values` | `16384` | Matrices with fewer values stay float16. |
| `damp_percent` | `0.01` | GPTQ damping added to the Hessian diagonal, as a fraction of its mean. GPTQ only. |
| `block_size` | `128` | Number of columns GPTQ processes at a time. GPTQ only. |
| `min_calibration_rows` | `128` | Minimum calibration rows a matrix needs to use GPTQ; below this it uses `fallback`. |
| `fallback` | `"symmetric"` | Method for matrices without enough calibration data. Only `"symmetric"` is supported. |

GPTQ needs a model in evaluation mode and at least one calibration example,
passed as `calibration_inputs`, `calibration_calls`, or `calibration_sequences`
depending on the export. See [GPTQ calibration](storage.md#gptq-calibration).

### `CodegenConfig`

| Field | Default | Meaning |
|---|---:|---|
| `target_json_bytes` | `4 * 1024 * 1024` | Expanded JSON size that `"auto"` unrolling aims for; a warning is issued above it. `None` turns it off. |
| `unrolling` | `"auto"` | `"auto"`, `"compact"` (smallest code), or `"speed"` (most unrolling). |
| `id_namespace` | `None` | Three characters from `A-Z a-z 0-9 - _` included in block IDs. `None` picks a random one. `""` gives shorter IDs for a sprite that is the only generated sprite in its project. |
| `compact_internal_names` | `False` | Shorten internal variable, list, and custom block names. |
| `compact_schema` | `False` | Leave redundant fields out of the saved block data. |
| `layer_sharing` | `"off"` | `"auto"` shares one copy of the code across repeated transformer layers. |

See [code generation](code-generation.md) before turning on `compact_schema` or
`layer_sharing`.

## Result types

An `ExportProgram` export returns a `ProgramResult`: the `ArtifactResult`
fields plus `EntryPointResult` and `StateResult` records. Verifying a program
returns a `ProgramVerifyResult`, with one `ProgramCallVerifyResult` per call.
These result types are imported from `cattorch`; the classes that define
programs and calls come from `cattorch.experimental`.

Each `ProgramCallVerifyResult` has `entrypoint`, `passed`, `outputs`, `states`,
and the Scratch `status`. A call whose status isn't `"ok"` fails, and doesn't
change the state used for later calls.

### `TensorSpec`

Describes one input or output list:

| Field | Meaning |
|---|---|
| `list_name` | Name of the list in Scratch. |
| `shape` | Tensor shape. |
| `dtype` | PyTorch dtype, as a string. |
| `numel` | Number of values, or `None` if the shape has a runtime extent. |
| `max_numel` | Largest number of values the list can hold. Equal to `numel` for fixed shapes. |

A shape with a runtime extent contains one `BoundedDimension` in its place.

### `BoundedDimension`

The `name`, `minimum`, and `maximum` (both inclusive) of a dimension whose
length can change at runtime. See
[programs](programs-and-moe.md#named-entrypoints-and-state).

### `ArtifactResult`

Fields shared by model and tokenizer results:

| Field | Meaning |
|---|---|
| `path` | Path of the written file. |
| `sprite_name` | Sprite name shown in Scratch. |
| `archive_bytes` | Size of the `.sprite3` file. |
| `expanded_json_bytes` | Size of the uncompressed `sprite.json`. |
| `block_count`, `list_count` | Number of generated blocks and lists. |
| `sharded_lists` | Names of lists split into [shards](storage.md#list-sharding). |
| `warnings` | Size, compatibility, and fallback warnings. Check these before sharing the sprite. |

### `TranspileResult`

`ArtifactResult` plus `inputs`, `output`, `additional_outputs`, `procedures`
(the custom block names), and `quantization` (a `QuantizationReport`, or
`None`). The `outputs` property lists every output's `TensorSpec` in order.

### `TokenizerResult`

`ArtifactResult` plus `tokenizer_type` and `token_count`.

### `VerifyResult`

| Field | Meaning |
|---|---|
| `passed` | Whether every value met `atol + rtol * abs(expected)`. |
| `values_compared` | Number of values compared. |
| `expected_shape` | Shape of the PyTorch output. |
| `actual_values` | Number of flattened values returned by the sprite. |
| `max_abs_error` | Largest absolute difference. |
| `max_rel_error` | Largest relative difference. |
| `mean_abs_error` | Mean absolute difference. |
| `worst_index` | Zero-based flattened index of the largest absolute error, or `None` for an empty result. |

If the sprite returns the wrong number of values, `passed` is `False` and the
error fields are infinite; no exception is raised.

### `MultiOutputVerifyResult`

`passed`, true only if every output passes, and `outputs`, a tuple with one
`VerifyResult` per output in export order.

### `QuantizationReport`

Summary of a quantized export: the requested `method` and `bits`, the
`calibration_batches` and `calibration_rows` used, `quantized_values` in total,
how many of those used GPTQ (`gptq_values`) or symmetric quantization
(`symmetric_values`), and a `TensorQuantizationReport` per tensor in
`tensors`.

### `TensorQuantizationReport`

One weight tensor, or one part of a tensor the model uses as a separate
matrix: `name`, the
`method` used, `bits`, number of `values`, `calibration_rows` seen,
`effective_group_size`, and `fallback_reason` if GPTQ wasn't used. A `method`
of `"float16"` means the tensor was smaller than `min_quantized_values`.

## Benchmark API

Each builder writes an `.sb3` project (adding the suffix if needed) and returns
its `Path`. Timing happens when you run the project in Scratch.

### `build_paired_benchmark`

```text
build_paired_benchmark(
    model,
    example_inputs,
    output_path,
    *,
    sig_figs=None,
    warmups=2,
    repeats=7,
    fast_config=None,
) -> Path
```

Build a project that times one model in exact and fast mode.

### `build_benchmark_suite`

```text
build_benchmark_suite(
    cases,
    output_path,
    *,
    iterations=100,
    warmups=0,
    sig_figs=None,
    fast_config=None,
    storage=None,
) -> Path
```

Each case is `(name, model, example_inputs)`, or
`(name, model, example_inputs, fast_config)` to set fast mode options for that
case. Names must be unique.

### `build_generation_benchmark_suite`

```text
build_generation_benchmark_suite(
    cases,
    output_path,
    *,
    iterations=16,
    sig_figs=None,
) -> Path
```

Each case is `(name, model, stateless_input, prompt)`. `prompt` must be a
`[1, T]` tensor. Prefill isn't timed; only the decode loop is.

### `build_storage_benchmark_suite`

```text
build_storage_benchmark_suite(
    cases,
    output_path,
    *,
    iterations=10,
    sig_figs=None,
) -> Path
```

Each case is `(name, model, example_inputs)`. The project times initialization
and forward calls for uncompressed float32, compressed float32 and float16, and
symmetric 8-, 6-, and 4-bit quantization.

### `analyze_benchmark`

```text
analyze_benchmark(path) -> dict
```

Read the results from a benchmark project saved after running in Scratch.
Returns a dictionary with the project's size and SHA-256, benchmark type, the
comparison, the raw result list, and parsed per-case timings. Storage suites also include sprite sizes, and
generation suites include output comparisons. Raises `ValueError` if the
project hasn't been run or can't be parsed. The `cattorch-benchmark` command
does the same from the shell; see
[verification and benchmarking](verification-and-benchmarking.md).

## Exceptions

| Exception | Also a | Raised when |
|---|---|---|
| `CattorchError` | `Exception` | Base class for the errors below. |
| `UnsupportedModelError` | `ValueError` | The model can't be exported, for example because it returns a non-tensor value. |
| `UnsupportedOperationError` | `NotImplementedError` | The model uses an operation cattorch can't convert. The message names the operation and module. |

Invalid configuration values raise ordinary `TypeError` or `ValueError`. See
[troubleshooting](troubleshooting.md).
