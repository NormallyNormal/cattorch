# Public API reference

[Documentation home](index.md)

Functions, configuration classes, and result types are imported from
`cattorch`. Paths can be strings or `pathlib.Path` objects. Every function
takes its leading arguments (the model, what it's called with, the output
path) by position and everything after them by keyword.

Programs (`ExportProgram`, `EntryPoint`, `Input`, `Output`, `State`,
`StateInput`, `StateUpdate`, `ProgramCall`), markers, and `ModuleAdapter` are
described in [programs and sparse MoE](programs-and-moe.md) and
[markers](markers.md). [Experimental](stability.md) APIs are imported from
`cattorch.experimental`: the explicit MoE classes, `analyze`, and the kernel
interface (`Kernel`, `op`, `check_kernel`, `build_kernel_benchmark`, and
`cattorch.scratch`), described in
[custom kernels and operations](custom-kernels.md).

## Model export

### `transpile`

```text
transpile(
    model,
    example_inputs,
    output_path,
    *,
    name=None,
    optimization="exact",
    fast_config=None,
    storage=None,
    quantization=None,
    calibration=None,
    sig_figs=None,
    codegen=None,
    adapters=(),
    kernels=(),
    example_kwargs=None,
    dynamic=None,
) -> TranspileResult | ProgramResult | GenerationResult
```

Export a `torch.nn.Module` as a Scratch `.sprite3` file.

| Parameter | Meaning |
|---|---|
| `model` | Module to export. Export works on a copy. A model left in training mode with modules that behave differently in it (dropout, batch norm) is exported in evaluation mode, with a warning. |
| `example_inputs` | What the sprite is called with: a tensor or tuple of tensors for `forward`, an `ExportProgram`, a `GenerationProgram`, or a `RecurrentGenerationProgram`. `None` takes it from the model's [markers](markers.md). |
| `output_path` | Destination. `.sprite3` is appended when absent and missing parent directories are created. |
| `name` | Scratch sprite display name. Defaults to the destination filename stem. |
| `optimization` | `"exact"` (default) or `"fast"`. See [optimization modes](optimization.md). |
| `fast_config` | A `FastConfig`; valid only with `optimization="fast"`. |
| `storage` | A `StorageConfig` for weight compression and float precision. |
| `quantization` | A `QuantizationConfig` for symmetric or GPTQ integer weights. |
| `calibration` | GPTQ calibration data, read once, in the form the export is called with: an iterable of tensors or tensor tuples for tensor inputs, of `ProgramCall`s for an `ExportProgram`, and of `[1, length]` token tensors for a generation program. Required with `method="gptq"` and invalid without it. |
| `sig_figs` | Round stored values to this many significant figures. Lossy. |
| `codegen` | A `CodegenConfig` for JSON size target, loop unrolling, IDs, and compaction. |
| `adapters` | `ModuleAdapter`s that replace module types in the export's copy of the model. |
| `kernels` | Experimental `Kernel`s for this export; see [custom kernels](custom-kernels.md). Each replaces the built-in lowering of its operation for this export. |
| `example_kwargs` | Keyword arguments for the model call, as a dict. Tensor values become input lists after the positional ones, in dict order. Other values, such as `use_cache=False`, are fixed for the export. Only for tensor inputs, and not with GPTQ when a value is a tensor. |
| `dynamic` | One variable-length input, as `{input: {dimension: (minimum, maximum)}}`, or `{input: (minimum, maximum)}` for its first dimension; `input` is its position or tensor keyword name. See [variable-length inputs](supported-models.md#variable-length-inputs). Only for tensor inputs: a program's entry points take `Input(dynamic=...)`, in the same form. |

Tensor inputs return a `TranspileResult`, an `ExportProgram` (or
`@entrypoint` markers) a `ProgramResult`, and a `GenerationProgram` or
`RecurrentGenerationProgram` a `GenerationResult`. Every option works with
every kind of export except where the table says otherwise; an option a kind
doesn't take raises `ValueError`.

Errors follow one rule. Invalid arguments raise `TypeError` (the wrong type)
or `ValueError` (an invalid value or combination), before the model runs.
`UnsupportedOperationError` means the traced model uses an operation
cattorch can't convert. `UnsupportedModelError` means the model can't be
exported as it is: something about its structure, a requirement of the
export kind it doesn't meet (such as a `GenerationProgram`'s causal
attention), a marker that doesn't match what it does, or a failure when it
runs on the example inputs. See [troubleshooting](troubleshooting.md).

Every warning an export gives is in its result's `warnings`, and is also
emitted as a `CattorchWarning`.

For GPTQ, use calibration data that routes tokens the way real use does.
Experts that receive too little data fall back to symmetric quantization; see
[GPTQ calibration](storage.md#gptq-calibration).

### `rotary_embedding`

```text
rotary_embedding(value, cosine, sine) -> Tensor
```

Apply rotary position embeddings (RoPE) to adjacent pairs of features. The
last dimension of `value` must be even. `cosine` and `sine` must broadcast to
`value`'s shape without making it larger.

## Markers and programs

Decorators on your own classes and methods. They only attach metadata, so the
model runs unchanged in PyTorch; see [markers](markers.md).

| Name | Use |
|---|---|
| `entrypoint` | On a method: export it as a Scratch block. Buffers it updates become state; `capacity=` for a buffer that grows; `name=`, `input_names=`, `output_names=`, `example_inputs=`, `dynamic=`. |
| `example_inputs` | On a class: example inputs for `forward`, so `transpile(model, None, path)` needs none. |
| `export_forward` | On a method: trace it instead of `forward` for every instance of the class. |
| `moe` | On a class: export it as a sparse mixture of experts with the given `router`, `experts`, `top_k`, `scoring`, `normalize`, `shared`, and `scale`, checked against `forward`. |
| `generate` | On a method: the method a `GenerationProgram` exports when its `method` isn't set. |
| `adapter` | On a function: build a `ModuleAdapter` for a class you don't own. |
| `marked_program` | `marked_program(model)` returns a `MarkedProgram`: the export copy of the model (`model`) and the `ExportProgram` (`program`) its `entrypoint` markers describe. |

Pass `None` as `example_inputs` to `transpile`, `verify` and `analyze` to take
the interface from `entrypoint` or `example_inputs` markers.

The classes markers compile to can also be used directly:

| Name | Use |
|---|---|
| `ExportProgram`, `EntryPoint`, `Input`, `Output`, `State`, `StateInput`, `StateUpdate` | Declare entry points and state by hand; see [programs](programs-and-moe.md#named-entrypoints-and-state). |
| `ProgramCall` | One block call: in `verify(..., calls=...)`, and in an `ExportProgram`'s `calibration=`. |
| `ModuleAdapter`, `AdapterContext` | Replace a module type during export; the factory gets the module and an `AdapterContext` with its `module_path` and the program's `state_names`. See [adapters](programs-and-moe.md#scoped-module-adapters). |
| `CattorchWarning` | Base class of every warning cattorch emits. |
| `CustomMoEWarning` | A `CattorchWarning` warned when a hand-written block was detected and exported as a sparse mixture of experts. Filter it with `warnings` if that's intended. |
| `INTERFACE_VERSION` | The version of the [Scratch interface](scratch-interface.md) exported sprites implement, `1`. |

`op`, for declaring a new operation, is [experimental](stability.md); see
[custom kernels and operations](custom-kernels.md).

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
SentencePiece BPE) and export it with `CharTokenizer`, `BPETokenizer` or
`SentencePieceBPETokenizer`. Any other type, such as WordPiece, Unigram, or a
SentencePiece Unigram model, raises `ValueError`; see
[tokenizers](tokenizers.md). `scratch_casefold=True` is for a SentencePiece
model trained with case-folding normalization; with any other tokenizer it
raises `ValueError`.

### `CharTokenizer`, `BPETokenizer` and `SentencePieceBPETokenizer`

```text
CharTokenizer(tokenizer).save(output_path, *, name=None, codegen=None)
BPETokenizer(tokenizer).save(output_path, *, name=None, codegen=None)
SentencePieceBPETokenizer(tokenizer, *, scratch_casefold=False).save(
    output_path, *, name=None, codegen=None,
)
```

Export a tokenizer as the kind you choose, without detection; each returns a
`TokenizerResult`. `CharTokenizer` and `BPETokenizer` take a Hugging Face
tokenizer, which must have `get_vocab()`; anything else raises `TypeError`.
`BPETokenizer` also needs a fast (Rust-backed) BPE tokenizer so cattorch can
read its merges. `CharTokenizer` checks that the tokenizer's tokens are
single characters and that it encodes text one character at a time, and
raises `ValueError` otherwise. `SentencePieceBPETokenizer` takes a
`SentencePieceProcessor` or an object with one in its `.processor`
attribute; the model must be trained with `model_type="bpe"`. See
[tokenizers](tokenizers.md) for normalization, case, Unicode, and byte
fallback limits.

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
    storage=None,
    sig_figs=None,
    example_kwargs=None,
) -> VerifyResult | MultiOutputVerifyResult | ProgramVerifyResult
```

Run an exported sprite in cattorch's Python emulator and compare it with
PyTorch. `model` and `example_inputs` are what was passed to `transpile`
(`None` takes them from the markers), and `sprite` is the export's result or
its path. Tolerances must be non-negative. `verify` checks correctness, not
speed. It works for every kind of export:

- **Tensor inputs**: the sprite runs `cattorch forward` on `example_inputs`.
  A model with one tensor output returns a `VerifyResult`, even if the tensor
  is inside a container; a model with several returns a
  `MultiOutputVerifyResult`, whose `outputs` are in the same order as
  `TranspileResult.outputs`. Pass the same `example_kwargs` used at export.
  Non-tensor outputs are ignored, as in `transpile`. Outputs that don't match
  the sprite's lists raise `UnsupportedModelError`.
- **Programs** (`ExportProgram` or `@entrypoint` markers): `calls` is a
  sequence of `ProgramCall`s run in order, keeping state between them in both
  PyTorch and Scratch; by default each entry point once with its example
  inputs.
- **Generation** (`GenerationProgram` or `RecurrentGenerationProgram`):
  `calls` names the blocks `"prefill"`, `"decode"` and `"reset"` with their
  token IDs, such as `ProgramCall("prefill", (torch.tensor([[1, 2, 3]]),))`;
  by default a prefill and a few decodes. Each call's `cattorch logits` and
  top-k lists are compared with the model's next-token logits, and its
  `cattorch status` with the status its input should give, such as
  `context full` or `too many items: cattorch token ids`.

Programs and generation return a `ProgramVerifyResult`.

`adapters`, `quantization`, `storage` and `sig_figs` are the export's
options. Given them, PyTorch runs a copy of the model prepared as the export
prepared its own (adapted, batch norms folded, weights quantized and rounded
as stored), so the comparison isolates the conversion from lossy storage and
can use tight tolerances. Without them, compare a lossy export with a looser
tolerance. Only symmetric quantization can be reproduced; for GPTQ, verify
against the float model with a tolerance such as `atol=0.05`. A model in
training mode is compared as an evaluation-mode copy, with a warning.

## Projects

### `build_project`

```text
build_project(
    model,
    output_path,
    *,
    tokenizer=None,
    example_inputs=(),
    labels=None,
    max_new_tokens=40,
    temperature=0.8,
    eos_token_id=None,
    prompt=None,
) -> ProjectResult
```

Write a runnable `.sb3` around a `transpile` result, adding `.sb3` if it's
missing. Every operation gets a bridge to global lists that other sprites can
run with `broadcast and wait`. A `TranspileResult` gets list monitors starting
from `example_inputs` (or zeros) and runs on the green flag or space key;
`labels` names a classifier's outputs. A `GenerationResult` needs `tokenizer`,
a `TokenizerResult`, and gets a Demo sprite that continues typed prompts. A
`ProgramResult` gets bridges only. See
[projects and the command line](projects-and-cli.md).

### `ProjectResult`

| Field | Meaning |
|---|---|
| `path` | Path of the written `.sb3`. |
| `sprites` | Names of the project's sprites. |
| `broadcasts` | The project's public broadcasts: each bridge's message followed by its completion broadcast, such as `points run`, `points run complete`. |
| `lists`, `variables` | The global lists and variables the bridges read and write. |
| `warnings` | Size warnings; each is also emitted as a `CattorchWarning`. |

All names are Scratch names.

## Configuration objects

Configuration classes are immutable dataclasses. Invalid values raise an
error when the object is created.

### `FastConfig`

| Field | Default | Meaning |
|---|---:|---|
| `activations` | `True` | Use fast activation approximations (currently QuickGELU for exact and tanh GELU). |
| `layer_norm` | `True` | Compute LayerNorm variance in one pass. |
| `softmax` | `True` | Skip max subtraction in softmax. |
| `neuron_pruning` | `0.0` | Fraction in `[0, 1)` of hidden units to remove from Linear → activation → Linear paths and `StackedSwiGLUMoE` experts. Finding the paths needs a model `torch.fx` can trace symbolically; one it can't raises `UnsupportedModelError`. |
| `layers` | `FastLayerConfig()` | Weight transforms applied to every eligible layer. |
| `overrides` | `{}` | Per-module `FastLayerConfig`s, keyed by module name, that replace `layers` for that module. |

### `FastLayerConfig`

| Field | Default | Meaning |
|---|---:|---|
| `block_pruning` | `0.0` | Fraction in `[0, 1)` of weight blocks to remove, smallest magnitude first. |
| `rank` | `None` | Truncated-SVD rank. Can't be combined with `rank_ratio`. |
| `rank_ratio` | `None` | Rank as a fraction in `(0, 1]` of the matrix's full rank. Can't be combined with `rank`. |

A low-rank transform is skipped for a layer when it wouldn't reduce the
number of multiplications, with a warning. See
[optimization modes](optimization.md#weight-transforms).

### `GenerationProgram`

| Field | Default | Meaning |
|---|---:|---|
| `max_context` | required | Maximum number of tokens the cache holds. The only positional field; the others are keyword-only. |
| `method` | `None` | Name of the model method that takes token IDs and returns next-token logits for every position. `None` means the method marked `@generate`, or `forward`. |
| `hidden_prefill` | `True` | During prefill, compute logits only for the last prompt token. The final layer's query, attention output and MLP are skipped too. |
| `top_k` | `None` | If set (1 to 64), also write the top-k logits and their token IDs after each prefill or decode. |

See [KV-cached generation](generation.md) for model requirements and the
Scratch blocks.

### `RecurrentGenerationProgram`

| Field | Default | Meaning |
|---|---:|---|
| `states` | `None` | A tuple of replace-mode `State`s with their initial values, in the order the step takes and returns them. The only positional field; the others are keyword-only. `None` exports a Hugging Face Mamba, Mamba-2 or RWKV causal LM through cattorch's own step, states and reference check; `step` and `reference` must then keep their defaults. Any other model needs `states`. |
| `step` | `"step"` | Name of the model method `step(token, *states) -> (logits, *new_states)`. `token` is a `[1, 1]` tensor of one token ID, looked up in an embedding; the logits flatten to one vocabulary row; each new state keeps its shape. |
| `max_context` | `None` | Maximum number of tokens read between resets, as for a `GenerationProgram`. `None` sets no limit, and `cattorch max context` is `Infinity`. |
| `hidden_prefill` | `True` | During prefill, skip the operations that only compute the logits for every prompt token but the last. State updates still run. |
| `top_k` | `None` | If set (1 to 64), also write the top-k logits and their token IDs after each prefill or decode. |
| `reference` | `None` | Name of a stateless model method `reference(tokens) -> logits` (`[1, n, V]` or `[n, V]`). At export, the step runs over up to 32 random tokens from the initial states and each logits row must match it within `1e-4` of the row's largest value. |

For models whose tokens mix through a recurrence rather than attention: GRU
and LSTM language models, Mamba, RWKV. The sprite has the same blocks, lists,
statuses and broadcasts as a `GenerationProgram`'s, with
`cattorch cache length` counting the tokens read; `cattorch reset` and
prefill restore every state's initial value. A decode with a token that
isn't valid, or whose step fails, leaves the states unchanged; a prefill that
fails restores the initial states, as a `GenerationProgram`'s empties its
cache. Every `transpile` option works as for a `GenerationProgram`. See
[recurrent generation](recurrent-generation.md) for the step contract, the
supported Hugging Face models and the Scratch interface.

### `StorageConfig`

| Field | Default | Meaning |
|---|---:|---|
| `compression` | `True` | Compress stored tensors and decode them during initialization. |
| `precision` | `"float32"` | `"float32"` or lossy `"float16"`. |

For integer weights, pass `quantization=QuantizationConfig(bits=...)` to
`transpile`; `StorageConfig` rejects integer precisions.

### `QuantizationConfig`

| Field | Default | Meaning |
|---|---:|---|
| `bits` | required | Integer weight width: `4`, `6`, or `8`. |
| `method` | `"symmetric"` | `"symmetric"` (round to nearest) or `"gptq"` (uses calibration data). |
| `group_size` | `64` | Maximum group size. The actual size is the largest divisor of the matrix input width (for a convolution, its fan-in) that doesn't exceed this. A matrix whose integer storage wouldn't be smaller than float16 stays float16. |
| `scale_precision` | `"float16"` | `"float16"` or `"float32"` scales. |
| `min_quantized_values` | `16384` | Matrices with fewer values stay float16. |
| `damp_percent` | `0.01` | GPTQ damping added to the Hessian diagonal, as a fraction of its mean. GPTQ only. |
| `block_size` | `128` | Number of columns GPTQ processes at a time. GPTQ only. |
| `min_calibration_rows` | `128` | Minimum calibration rows a matrix needs to use GPTQ; below this it uses symmetric quantization. |
| `overrides` | `{}` | Other settings for some weights: a module path or parameter name mapped to bits (`4`, `6`, `8`), `None` for float16, or a `QuantizationOverride`. See [per-weight settings](storage.md#per-weight-settings). |
| `head_bits` | `"auto"` | The output head of a generation export: `"auto"` for int8 when `bits` is 4 or 6 and `bits` otherwise, `4`, `6` or `8`, or `None` for float16. An override that selects the head wins. Other exports accept only `"auto"` and raise `ValueError` for any other value. See [the output head](storage.md#the-output-head-of-a-language-model). |

### `QuantizationOverride`

| Field | Default | Meaning |
|---|---:|---|
| `bits` | required | `4`, `6`, `8`, or `None` to keep the selected weights float16. |
| `method` | `None` | `"symmetric"`, `"gptq"` (only within a GPTQ config), or `None` for the config's. |
| `group_size` | `None` | Maximum group size, or `None` for the config's. |

The longest key that matches a parameter's name wins; a weight read under
several names, such as a tied embedding and output head, takes the highest
precision they select. A key that matches no parameter is an error.

GPTQ needs at least one calibration example, passed as `transpile`'s
`calibration`. See [GPTQ calibration](storage.md#gptq-calibration).

### `CodegenConfig`

| Field | Default | Meaning |
|---|---:|---|
| `target_json_bytes` | `4 * 1024 * 1024` | Expanded JSON size that `"auto"` unrolling aims for; a warning is issued above it. `None` turns it off. |
| `unrolling` | `"auto"` | `"auto"`, `"compact"` (smallest code), or `"speed"` (most unrolling). |
| `id_namespace` | `None` | Three characters from `A-Z a-z 0-9 - _` included in block IDs. `None` picks a random one. `""` gives shorter IDs for a sprite that is the only generated sprite in its project. |
| `compact_internal_names` | `False` | Shorten internal variable, list, and custom block names. |
| `compact_schema` | `False` | Leave redundant fields out of the saved block data. |
| `layer_sharing` | `False` | Share one copy of the code across repeated transformer layers where the layout allows it, falling back with a warning where it doesn't. |

See [code generation](code-generation.md) before turning on `compact_schema` or
`layer_sharing`.

## Result types

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

Fields shared by every sprite result:

| Field | Meaning |
|---|---|
| `path` | Path of the written file. |
| `sprite_name` | Sprite name shown in Scratch. |
| `archive_bytes` | Size of the `.sprite3` file. |
| `expanded_json_bytes` | Size of the uncompressed `sprite.json`. |
| `block_count`, `list_count` | Number of generated blocks and lists. |
| `sharded_lists` | Names of lists split into [shards](storage.md#list-sharding). |
| `warnings` | Every warning of the export: size, compatibility, fallbacks, and anything that changed what was exported. Each is also emitted as a `CattorchWarning`. Check these before sharing the sprite. |

### `TranspileResult`

`ArtifactResult` plus:

| Field | Meaning |
|---|---|
| `inputs` | A `TensorSpec` per input list, in argument order. |
| `outputs` | A `TensorSpec` per output list, in output order. The `output` property is `outputs[0]`. |
| `procedures` | The public blocks. |
| `quantization` | A `QuantizationReport`, or `None`. |
| `fallback_operations` | Operations that have no kernel of their own and ran as PyTorch's decomposition into simpler operations, as `(operation, calls)` pairs such as `("aten.mish.default", 2)`. These usually run slower than a dedicated kernel would, and `warnings` names them too. Calls that depend only on weights are computed at export and aren't listed. |
| `approximate_operations` | The operations that used an approximate kernel in fast mode, the same way. |

### `GenerationResult`

`ArtifactResult` plus, the same for a `GenerationProgram` and a
`RecurrentGenerationProgram`:

| Field | Meaning |
|---|---|
| `inputs` | The `cattorch token ids` list: a prompt of 1 to `max_context` tokens for `cattorch prefill` (to 200,000 without a limit), or the one token `cattorch decode` reads. |
| `outputs` | `cattorch logits`, one value per token ID, then, with `top_k`, `cattorch top k values` and `cattorch top k ids`. |
| `max_context` | The most tokens read between resets; `None` for no limit. |
| `vocabulary_size` | The number of token IDs the sprite accepts. |
| `top_k` | The number of largest logits listed, or `None`. |
| `procedures` | `cattorch init`, `cattorch reset`, `cattorch prefill`, `cattorch decode` and `cattorch prepare for save`. |
| `quantization`, `fallback_operations`, `approximate_operations` | As for `TranspileResult`. |

### `ProgramResult`

`ArtifactResult` plus `entrypoints` (an `EntryPointResult` per entry point),
`states` (a `StateResult` per state), `procedures` (the public blocks), and
`quantization`, `fallback_operations` and `approximate_operations` as for a
`TranspileResult`, summed over the entry points.

### `EntryPointResult`

| Field | Meaning |
|---|---|
| `name` | The entry point's name. |
| `procedure` | Its block, `cattorch ENTRY`. |
| `completion_broadcast` | The broadcast the block sends when it ends, `cattorch ENTRY complete`. |
| `inputs`, `outputs` | A `TensorSpec` per input and output list, in declaration order. |
| `state_inputs`, `state_updates` | The names of the states it reads and writes. |

### `StateResult`

| Field | Meaning |
|---|---|
| `name` | The state's name. |
| `list_name` | Its list, `cattorch state STATE`. |
| `mode` | `"replace"` or `"append"`. |
| `shape`, `dtype` | Its initial tensor's shape and dtype. |
| `capacity` | The most rows an append state holds; `None` for a replace state. |

### `TokenizerResult`

`ArtifactResult` plus `tokenizer_type`, the kind exported (`"char"`,
`"bpe"` or `"sentencepiece-bpe"`), and `token_count`, its vocabulary size.

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
| `approximate_operations` | The export's `approximate_operations`, when `verify` was given its result. Approximate kernels explain small differences. |

If the sprite returns the wrong number of values, `passed` is `False` and the
error fields are infinite; no exception is raised.

### `MultiOutputVerifyResult`

`passed`, true only if every output passes; `outputs`, a tuple with one
`VerifyResult` per output in export order; and `approximate_operations`, as
for `VerifyResult`.

### `ProgramVerifyResult` and `ProgramCallVerifyResult`

A `ProgramVerifyResult` has `passed`, true only if every call passes, and
`calls`, one `ProgramCallVerifyResult` per call, with:

| Field | Meaning |
|---|---|
| `entrypoint` | The block called: an entry point, or `prefill`, `decode` or `reset`. |
| `passed` | Whether `status` is `expected_status` and every comparison passed. |
| `outputs` | A `VerifyResult` per output list, when the call succeeded. For generation: `cattorch logits`, then the top-k values, then the logits at the top-k IDs compared with the top-k values (ties may list either ID). |
| `states` | A `VerifyResult` per state list, by state name. |
| `status` | The `cattorch status` the call ended with. |
| `expected_status` | The status its input should give: `ok`, or for generation the failure an invalid call reports. A call that fails doesn't change the state used for later calls. |

### `QuantizationReport`

Summary of a quantized export: the requested `method` and `bits`, the
`calibration_batches` and `calibration_rows` used, `quantized_values` stored
in integers in total, how many of those used GPTQ (`gptq_values`) or
symmetric quantization (`symmetric_values`), and a `TensorQuantizationReport`
per tensor in `tensors`. The report says what storage stores, so a value
quantized and then kept in float16 doesn't count as quantized.

### `TensorQuantizationReport`

One weight tensor, or one part of a tensor the model uses as a separate
matrix (`name[offset=...,shape=...]`): `name`, the `method` used, `bits`,
number of `values`, `calibration_rows` seen, `effective_group_size`,
`fallback_reason` if GPTQ wasn't used, `sources`, the parameters a weight
computed from parameters is computed from (empty otherwise), and
`override`, the `QuantizationConfig.overrides` key whose settings it used,
if any, or `"head_bits"` for a generation export's output head whose
settings `QuantizationConfig.head_bits` chose. A weight computed from parameters is named by its attribute, such
as a parametrized `fc.weight`, or `weight computed from <parameters>`. A
`method` of `"float16"` means the values are stored in float16, and
`fallback_reason` says why: the tensor was smaller than
`min_quantized_values`, integer storage wouldn't be smaller, an override
or `head_bits=None` keeps it float16, storage reads it grouped along a different axis than it
was quantized in, or it is a weight computed from parameters where
quantization couldn't compute it from quantized values instead.
Values of a quantized entry that are also, or only, stored in float16, such
as the copy a second use reads in another layout, are reported in a
`name[float16]` entry. `name[remaining]` holds a parameter's values that no
use reads as a matrix when the rest is quantized for its uses.

## Benchmark API

Each builder writes an `.sb3` project (adding the suffix if needed) and returns
its `Path`. Timing happens when you run the project in Scratch; read the
results with `analyze_benchmark`. Every builder takes `repeats` (the timed
calls per variant), `warmups` (untimed calls first), and the export options
of `transpile`: `fast_config`, `storage`, `quantization`, `calibration` (a
list, since every variant is exported), `sig_figs`, `codegen`, `adapters` and
`kernels`. A project's public names are its results lists, such as
`cattorch benchmark results`, and the stage variables
`cattorch benchmark repeats`, `cattorch benchmark warmups` and
`cattorch benchmark comparison`; everything else is internal.

### `build_paired_benchmark`

```text
build_paired_benchmark(
    model, example_inputs, output_path, *,
    repeats=7, warmups=2, fast_config=None, storage=None, quantization=None,
    calibration=None, sig_figs=None, codegen=None, adapters=(), kernels=(),
) -> Path
```

Time one model in exact and fast mode, alternating, recording each call in
the `cattorch exact timings` and `cattorch fast timings` lists.

### `build_benchmark_suite`

```text
build_benchmark_suite(
    cases, output_path, *,
    repeats=100, warmups=0, fast_config=None, storage=None, quantization=None,
    calibration=None, sig_figs=None, codegen=None, adapters=(), kernels=(),
) -> Path
```

Each case is `(name, model, example_inputs)`, or
`(name, model, example_inputs, fast_config)` to set fast mode options for that
case, timed in exact and fast mode. Names must be unique.

### `build_generation_benchmark_suite`

```text
build_generation_benchmark_suite(
    cases, output_path, *,
    repeats=16, warmups=0, fast_config=None, storage=None, quantization=None,
    calibration=None, sig_figs=None, codegen=None, adapters=(), kernels=(),
) -> Path
```

Each case is `(name, model, program, prompt)`: `program` a `GenerationProgram`
or `RecurrentGenerationProgram`, and `prompt` a `[1, length]` token tensor.
The `cached` variant prefills the prompt outside the timer, then times
decodes; `max_context` must hold the prompt and every decode. The
`stateless` variant times the model's method (a recurrent program's
`reference`) on the prompt plus one token, the work each token costs without
a cache.

### `build_storage_benchmark_suite`

```text
build_storage_benchmark_suite(
    cases, output_path, *,
    repeats=10, warmups=0, fast_config=None, sig_figs=None, codegen=None,
    adapters=(), kernels=(),
) -> Path
```

Each case is `(name, model, example_inputs)`. The project times weight
decoding and forward calls for uncompressed float32, compressed float32 and
float16, and symmetric 8-, 6-, and 4-bit quantization, so it takes no
`storage` or `quantization`.

### `analyze_benchmark`

```text
analyze_benchmark(path) -> BenchmarkAnalysis
```

Read the results from a benchmark project saved after running in Scratch.
Raises `ValueError` if the project hasn't run, or ran only part of the way.
`cattorch analyze-benchmark PROJECT` prints the same as JSON; see
[verification and benchmarking](verification-and-benchmarking.md).

`BenchmarkAnalysis` has `kind` (`"paired"`, `"suite"` or `"storage"`), the
compared variants' names `baseline` and `candidate` (such as `"exact"` and
`"fast"`), `repeats` and `warmups`, `cases`, `storage`, the `raw_results`
list, and the file's `project_bytes` and `project_sha256`. Each
`BenchmarkCase` in `cases` has its `name`, `baseline_seconds` and
`candidate_seconds` (the seconds one call took), `speedup`, each sprite's
`baseline_blocks` and `candidate_blocks`, `baseline_output_values`,
`candidate_output_values`, `output_length_match` and `output_max_abs_error`
comparing their outputs, and, for a paired benchmark, every run's
`baseline_timings` and `candidate_timings` and the bootstrap
`speedup_95_percent_interval`. A storage suite's `storage` holds a
`StorageBenchmarkCase` per case, with its `name` and `variants`: a
`StorageBenchmarkVariant` (`init_seconds`, `forward_seconds`,
`sprite_bytes`) per storage mode.

## Exceptions

| Exception | Also a | Raised when |
|---|---|---|
| `CattorchError` | `Exception` | Base class for the errors below. |
| `UnsupportedModelError` | `ValueError` | The model can't be exported as it is; see [`transpile`](#transpile). |
| `UnsupportedOperationError` | `NotImplementedError` | The model uses an operation cattorch can't convert. The message names the operation and module. |
| `CattorchWarning` | `UserWarning` | Base class of every warning cattorch emits. |

Invalid arguments raise ordinary `TypeError` or `ValueError`. See
[troubleshooting](troubleshooting.md).
