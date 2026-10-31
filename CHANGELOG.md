# Unreleased

## New features

- Experimental custom kernels: `transpile(..., kernels=[Kernel(...)])` lowers
  an operation with your own Scratch code for that export, replacing a
  built-in kernel or a PyTorch decomposition. Kernels receive a context
  naming each input's list, shape, dtype, and whether it is a stored weight,
  and return statements from the new `cattorch.scratch` module. A validator
  can reject calls, and `approximate=True` kernels apply only in fast mode.
  `check_kernel` compares a kernel with PyTorch in the emulator, and
  `build_kernel_benchmark` times it in Scratch. See
  [custom kernels and operations](docs/custom-kernels.md).
- `cattorch.op` declares a new operation from an annotated function. It
  exports whole, with a kernel attached through `@operation.kernel`, or
  through its function when it has none.
- `transpile(..., dynamic={0: (1, 32)})` exports a model whose input length
  varies at run time within bounds, such as a batch or a sequence of
  independent tokens. Operations without a runtime-length kernel run once per
  item, and a model whose operations can't follow the length runs whole once
  per item. The export is checked against the model at several lengths, so a
  model that uses the length itself fails instead of exporting wrong numbers.
  Reductions, softmax, matrix products over the items, `cat` with fixed-size
  tensors, fixed slices at either end (`x[-1]`) and arithmetic with
  `x.shape[0]` follow the length too, which covers mean pooling and attention
  from fixed queries over a variable-length memory. An entry point that
  appends to state can read the grown state in the same call.
  Entry points take `Input(..., dynamic=(1, 32))` or
  `@entrypoint(dynamic=...)`, with the same check at several lengths and,
  for one that updates no state, the same run once per item. See
  [variable-length inputs](docs/supported-models.md#variable-length-inputs).
- Markers (`cattorch.entrypoint` and others), decorators that describe a model's export interface
  without changing how it runs in PyTorch. `@entrypoint` exports methods as
  Scratch blocks and turns the buffers they update into persistent state,
  with `capacity=` for a buffer that grows. `@example` gives `forward` example
  inputs, `@export_forward` swaps in an export-only method for a class,
  `@moe` declares a block's routing (including shared experts and scaled
  outputs), `@generate` picks a generation method, and `@adapter` builds a
  `ModuleAdapter` from a function. `transpile(model, None, path)` and
  `verify(model, None, result)` take the interface from the markers. See
  [markers](docs/markers.md).
- `build_project` writes a runnable Scratch project around exported sprites.
  Each operation is bridged to global lists, so other sprites can run it with
  `broadcast and wait`. A plain model gets list monitors and runs on the green
  flag, with an optional label for a classifier's prediction. A text generator
  gets its tokenizer and a Demo sprite that asks for a prompt and continues
  it, sampling at an adjustable temperature. See
  [projects and the command line](docs/projects-and-cli.md).
- A `cattorch` command: `cattorch export MODEL` exports and verifies a model
  from a Python file, module, or saved `.pt`, with inputs from its markers or
  `--shape`; `cattorch generate` exports a text generator, from a file or
  `--hf REPO`, with its tokenizer. `--project` also builds the project.
- An `examples/` directory: an MLP, a CNN, and a character-level text
  generator, each trained in seconds and exported with its project.
- Recurrent layers (`nn.GRU`, `nn.LSTM`, `nn.RNN`) run as one Scratch loop
  over time instead of unrolling every step, so the sprite no longer grows
  with the sequence length. With `dynamic=` the sequence length can vary at
  run time, and the final state is a fixed-shape output. `nn.GRUCell` and
  `nn.LSTMCell` now export, so a step method with its hidden state in a
  buffer works as an entry point.
- `TranspileResult.approximate_operations` lists the operations that used an
  approximate kernel in fast mode, and `verify` repeats it in its result.
- `GenerationProgram(max_context=64)` exports an ordinary stateless
  `forward(tokens) -> logits` model as a KV-cached generator, with no
  `example_token`, combined QKV projection, or position wrapper. Separate
  q/k/v projections, `arange` positions, RoPE, grouped-query and
  single-KV-head attention work, and so do unwrapped Hugging Face GPT-2,
  Llama, and Qwen2. cattorch checks the cached step against `forward` during
  export. See [KV-cached generation](docs/generation.md).
- Hand-written mixture-of-experts blocks, with a `ModuleList` of identical
  experts, a linear router, `topk`, and a loop over the experts, export
  without changes. cattorch finds the `SparseMoE` routing configuration that
  reproduces the block on probe inputs and warns with a `CustomMoEWarning`
  naming it. See [hand-written MoE blocks](docs/programs-and-moe.md#hand-written-moe-blocks).
- Slice and index assignment into a tensor the model created, such as
  `x[..., :4] = y` or `x[1] = y` into a `torch.empty` or `torch.zeros`
  buffer, exports as ordinary dataflow. Uninitialized buffers become zeros.
- `gather` and `scatter` accept indices computed at run time, such as
  `topk` results, along any dimension. This covers DeepSeek-V3's
  group-limited router, and DeepSeek-V3 models now export, with or without a
  `GenerationProgram`.
- `GenerationProgram` accepts sliding-window masks (Mistral, GPT-OSS), ALiBi
  biases, and GPT-OSS attention sinks. A windowed decode step reads only the
  window's cache rows.
- Hidden prefill also skips the final layer's query projection and RoPE for
  prompt tokens before the last.
- Hugging Face GPT-2, Llama, Qwen2, and BERT models export directly, without
  a wrapper. On the export copy, the cache and extra outputs are turned off
  and an attention implementation that traces is selected, so no
  `example_kwargs` are needed. Your model isn't modified.
- `transpile` and `verify` take `example_kwargs`. Tensor values become input
  lists after the positional inputs; other values are fixed for the export.
- Non-tensor outputs, such as `None` or a `DynamicCache`, are skipped with a
  warning instead of rejected, and `ModelOutput` is unwrapped.
- `.item()` and Python branches on tensors that don't depend on the model's
  inputs, such as masks built from `torch.arange`, are evaluated during export.
- `torch.no_grad()` and `torch.autocast` blocks inside `forward` no longer
  block export.
- A model with modules in training mode is exported as an evaluation-mode
  copy with a warning, instead of being rejected for training-mode dropout.
  `verify` compares against the same evaluation-mode copy.
- Computations whose inputs are all parameters, buffers, or constants now run
  during export, whatever operations they use. nanoGPT's `bias == 0` causal
  mask and RoPE tables built with `torch.outer` in `forward` no longer need
  rewriting.
- New elementwise operations: `exp`, `log`, `log10`, `sqrt`, `abs`, `floor`,
  `ceil`, `reciprocal`, `erf`, `clamp`, `hardtanh`/`relu6`, comparisons,
  logical not (including `~` on bool tensors), `torch.where`, `torch.maximum`,
  and `torch.minimum`.
  `torch.pow` accepts integer exponents from -4 to 4 and ±0.5.
- Exact GELU (the `nn.GELU()` default) is supported, so Hugging Face BERT
  models export without changes. This includes `ExpertFamily` experts, which
  previously required `approximate="tanh"`.
- `F.log_softmax` is supported.
- `unflatten` and `squeeze` are supported. With them, `nn.MultiheadAttention`
  and `nn.TransformerEncoder` export.
- Conversions to floating dtypes, such as `x.float()` on integer inputs, are
  no longer rejected. Conversions to integer dtypes truncate, and conversions
  to `bool` produce 0 or 1.
- Operations without a dedicated kernel now export through PyTorch's own
  decompositions when those lead to supported operations. Among them are
  `nn.LSTM` and `nn.GRU` (unrolled over the example's sequence length),
  GroupNorm, InstanceNorm, `F.normalize`, Mish, softplus, hardswish, `var`,
  `std`, and `cumsum`.
- New reductions over any dimensions: `sum`, `mean`, `prod`, `amax`, `amin`,
  `max`/`min` with indices, `argmax`, `argmin`, `any`, and `all`. `torch.topk`
  is supported.
- New data-movement operations: `expand`, `repeat`, `flip`, `roll`, padding,
  `tril`/`triu`, nearest upsampling, and `gather`/`index_select`/indexing with
  indices that don't depend on the input.
- `sin` and `cos`, adaptive max pooling, and
  `F.scaled_dot_product_attention` with a static `attn_mask`.
- BatchNorm without running statistics uses batch statistics, as PyTorch does.
- Experimental: `routed_experts(value, expert_ids, weights, experts)` runs an
  `ExpertFamily` on routes computed by your own router code, such as a linear
  layer, softmax or sigmoid, and `torch.topk`. It replaces the value-dependent
  expert loop of a typical MoE block, so shared experts and other router
  variations export without rebuilding the block as a `SparseMoE`. Only the
  selected experts run in Scratch. See
  [routing with your own router code](docs/programs-and-moe.md#routing-with-your-own-router-code).
- Hugging Face mixture-of-experts models, such as Mixtral, Qwen2-MoE, Qwen3-MoE,
  and OLMoE, export directly, with `transpile` or `GenerationProgram`. Experts
  classes marked with transformers' `@use_experts_implementation` run through
  `routed_experts` on the export copy, whatever their weight layout, biases,
  gating, or post-expert norm. Experts with their own gate function, such as
  GPT-OSS's clamped GLU, run it inside the expert program.
- In-place updates of computed tensors, such as `x /= x.sum()` or `x.mul_(2)`,
  export as ordinary operations. Updates seen through a view taken before the
  update are still rejected.
- `ExpertFamily` experts can use `torch.pow` with the same exponents as
  elsewhere, so ReLU² experts work. `SparseMoE` and `ExpertFamily` also work
  with automatic `GenerationProgram` export.

- `TranspileResult.fallback_operations` lists the operations without a kernel
  of their own that ran as PyTorch decompositions, with how many calls the
  sprite runs, and `warnings` names them. `analyze` reports them for each
  entrypoint. Calls computed at export aren't counted.
- [Operator coverage](docs/operator-coverage.md) lists how PyTorch's own
  operator samples (OpInfo) fare, generated by
  `tests/opinfo_conformance.py`. About 1,390 of 1,550 samples pass, and the
  test suite keeps every recorded pass passing.
- New operations: `torch.sign`, `trunc`, `round` (half to even, as in
  PyTorch), `%` and `torch.remainder`, `torch.fmod`, floor and truncating
  division (`//`, `rounding_mode=`), `&`, `|`, and `^` on boolean masks,
  `torch.logical_and`/`or`/`xor`, and `torch.full_like`.
- Tensors with no elements, and anything computed only from them, are
  computed during export, so empty inputs and dimensions no longer fail.
- Every convolution option: dilation, groups (including depthwise), unbatched
  inputs, and `padding="same"`. `nn.ConvTranspose1d/2d/3d` and `nn.Conv3d`
  export too, built from the 1D and 2D kernels.
- Every pooling option: `ceil_mode`, dilation, `count_include_pad=False`,
  `divisor_override`, `return_indices=True`, unbatched inputs, adaptive max
  pooling, and 3D pooling.
- `torch.pow` accepts any exponent, including a tensor; negative bases with
  fractional exponents give NaN, as in PyTorch.
- New operations: `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`,
  `asinh`, `acosh`, `atanh`, `exp2`, `log2`, `erfc`, `hypot`, `fmax`,
  `fmin`, `isfinite`, `copysign`, `logaddexp`, `logsumexp`, `logcumsumexp`,
  `round(decimals=)`, `cumprod`, `torch.cdist`, `F.pdist`, `F.bilinear`,
  `addbmm`, and `real`/`angle` on real tensors.
- Results that the model computes but never uses no longer run in Scratch.
- Indexing with indices computed from the input: `index_select`, `take`,
  and tensor indexing such as `x[ids]`, `x[:, ids]`, or `x[i, j]` read through
  a runtime `gather`. `scatter_add`, `scatter_reduce` (every reduction, with or
  without `include_self`), `index_add`, `index_copy`, `index_fill`,
  `index_reduce`, `index_put`, `x[ids] = y`, and `x[mask] = value` work with
  any integer indices.
- `torch.sort`, `argsort`, `msort`, `kthvalue`, `median`, `cummax`, `cummin`,
  and `torch.isin`.
- CI runs a set of exported sprites in the official Scratch VM and compares
  them with PyTorch: an MLP, a CNN, a transformer layer, every weight storage
  format, a variable-length input, a stateful program, and cached generation
  with its tokenizer. `benchmarks/run_project_vm.cjs` takes `--list`,
  `--var`, `--broadcast`, `--no-flag` and `--out` to drive a project's
  bridges.
- The Scratch interface is versioned: [Scratch interface](docs/scratch-interface.md)
  lists every public block, list, variable, and broadcast, and within interface
  version 1 none is renamed or removed. Each sprite records its version in the
  `cattorch interface version` variable, and `cattorch.interface` holds the
  version and the public names.
- Plain model sprites set `cattorch status` and broadcast
  `cattorch init complete` and `cattorch forward complete` when those blocks
  finish, as generation and program sprites do.
- Programs (`ExportProgram`, `EntryPoint`, `Input`, `Output`, `State`,
  `StateInput`, `StateUpdate`, `ProgramCall`), `marked_program`,
  `ModuleAdapter`, `AdapterContext`, and `CustomMoEWarning` are stable and
  imported from `cattorch`. Importing them from `cattorch.experimental` still
  works with a `DeprecationWarning`. [Stability](docs/stability.md) lists what
  1.x keeps compatible; the kernel interface, `op`, the explicit MoE classes,
  and `analyze` stay experimental.
- An `@entrypoint` method's non-tensor results, such as `None`, are skipped
  with a warning, as they are for `forward`, instead of silently.
- `torch.matmul` with broadcast batch dimensions, such as `[3, 4] @ [2, 4, 5]`
  or `[2, 1, 3, 4] @ [5, 4, 6]`, and a vector times a batch of matrices.

## Performance

- `torch.addmm` with a bias vector, which GPT-2's `Conv1D` layers use, runs on
  the linear kernel instead of a separate multiply and add. A two-layer GPT-2
  sprite has about 22% fewer blocks.
- Hugging Face style RoPE (`x * cos + rotate_half(x) * sin`, as in Llama,
  Qwen2 and Mistral) runs as one rotary loop instead of slices, negation,
  `cat` and three elementwise passes. A two-layer Llama sprite has about 14%
  fewer blocks.
- Hidden prefill skips the final layer's query, attention output and MLP for
  prompt tokens when layers are shared too, not just the output head. A
  three-layer shared model runs about 18% fewer Scratch operations for an
  8-token prompt.

## Fixes

- Layer sharing merged layers that differ only in a constant, such as an
  attention window or a residual scale, and ran every layer with the first
  layer's value. Such layers are no longer shared.
- With layer sharing, a `cattorch decode` before any prefill wrote the hidden
  state to `cattorch logits` instead of the logits.
- A model that reads input-dependent values with `tolist()` or `.numpy()`
  exported the example's values as constants. It is now rejected, like
  `.item()`.
- With `dynamic=`, a broadcast that adds dimensions to the variable-length
  input, such as `x.unsqueeze(-1) * weight`, gave wrong numbers at every
  length.
- The export-time check of variable-length exports compared each length with
  the error at the shortest one, so a sprite equally wrong everywhere passed,
  and it never checked lengths past 32. Exact exports must now match at every
  checked length, including the maximum (up to 1024 items).
- An `@entrypoint` whose growing buffer also changed its old rows, as in
  `cat([seen * 0.5, row])` or prepending, exported and then lost the change.
  It is now rejected.
- A buffer registered under two names in an `@entrypoint` model was read as a
  constant through the second name.
- The emulator kept NaN through arithmetic, where Scratch reads it as 0, so
  `verify` could pass a sprite that gives different numbers in Scratch.
- Hugging Face dynamic-NTK RoPE past `max_position_embeddings` exported for
  generation with wrong logits. Generation now checks shorter prompts too and
  rejects models whose positions depend on the prompt length.
- `BPETokenizer` accepted normalizers, pre-tokenizers, `fuse_unk` and
  `byte_fallback`, then tokenized differently from Hugging Face; it now
  rejects them. `transpile_tokenizer` exported WordPiece, WordLevel and Unigram
  tokenizers one character at a time; it now rejects those with
  multi-character tokens.
- SentencePiece sprites with `scratch_casefold=True` used the uppercase byte
  token for a lowercase letter without a vocabulary entry, and encoded a run
  of unknown characters as one unknown token each.
- `nn.LayerNorm(bias=False)`, `elementwise_affine=False`, and `F.layer_norm`
  with a weight but no bias failed with an `IndexError`.
- A BatchNorm used more than once, as in a Siamese encoder, normalized with
  the wrong standard deviation after its first use.

- `@cattorch.moe` stopped working once anything imported
  `cattorch.experimental`, because the module holding the MoE classes had the
  same name and replaced the marker. That module is now `cattorch.experts`.
- With a variable-length input or append state, three cases gave wrong values:
  - adding a fixed `[1, features]` tensor to the variable-length rows;
  - transposing across the variable axis while it had one item in the trace;
  - reshaping a tensor whose variable axis isn't its first dimension, as
    attention heads do.
- A linear layer whose result could be computed at export, such as one
  applied to `torch.zeros`, followed by an addition and an activation made
  export fail when a view of the result was the output.
- A tuple or other non-tensor positional example input fails with a
  `TypeError` that says to pass the tensors separately, instead of an
  `AttributeError`.
- `nn.SELU`, `nn.CELU`, and `F.elu` with a `scale` argument ignored the
  scale and gave wrong output.
- A view of a constant, such as a reshaped index table, passed to a
  data-movement operation like `index_select` lost its shape and failed.
- With a runtime-length input (append state or a dynamic input), embedding
  lookups and fused arithmetic processed only the traced length.
- `F.rms_norm` without a weight failed during export.
- An arithmetic operation with a constant on the left, such as `1 - x`, on a
  0-dimensional tensor gave an empty or wrong output.
- `F.batch_norm` with running statistics passed as inputs, not buffers, gave
  wrong output.
- `nn.AdaptiveAvgPool2d` on an unbatched `(C, H, W)` input gave wrong output.
- `transpose`, `.t()`, and `mean` on 0-dimensional tensors failed during
  export.
- Operations whose PyTorch decomposition leads back to themselves, such as
  `x ** y` with a tensor exponent, failed with a recursion error. They now
  raise `UnsupportedOperationError` naming the operation.
- `verify` ran a Hugging Face model with its own config, so a model with
  `output_hidden_states` or `use_cache` on failed with an output-count
  mismatch. It now compares against a copy with the same export defaults as
  `transpile`.
- Exports with many operations were slow: each operation copied everything
  generated so far. Export of a small two-layer MoE model is about 3 times
  faster.
- When `forward` fails at `max_context` tokens with an `IndexError`, the
  export error now says that `max_context` is probably larger than a position
  table.
- A linear layer followed by two residual additions, as in
  `x + (shared(x) + routed)`, added the first residual twice and gave wrong
  output.
- Layer sharing could fail with a `KeyError` when one stored tensor would
  fill more than one slot of a layer bank. Sharing is now skipped instead.
- A 0-dimensional parameter, buffer, or constant failed during export.
  Hugging Face models with `eager` attention hit this.
- Concatenating with an empty tensor, as Hugging Face caches do on their
  first step, failed during code generation.
- A tensor read through a view after its last direct use could be overwritten
  by a later operation, giving wrong output. Graph outputs that are also used
  later were affected the same way.
- `F.max_pool2d(x, 2)` and other pooling calls without an explicit stride
  failed during export.

## Breaking changes

- cattorch needs Python 3.11 or newer and PyTorch 2.10 or newer, up from
  Python 3.10 and PyTorch 2.6. Python 3.10 reached its end of life in
  October 2026.
- `cattorch.experimental.stacked_swiglu_moe_adapter` is removed. MoE blocks
  with SwiGLU experts switch to `StackedSwiGLUMoE` by themselves. For a block
  whose expert weights are stacked into one tensor each, an `@adapter` that
  returns `StackedSwiGLUMoE` does the same; see
  [mixture of experts](docs/programs-and-moe.md#mixture-of-experts).
- `op` moved back to `cattorch.experimental`, next to the kernel interface it
  uses. Only names outside `cattorch.experimental` are stable; see
  [stability](docs/stability.md).
- Tokenizer sprites read and write the `cattorch text` variable and the
  `cattorch token ids` list, instead of the `input` and `output` variables and
  the `token_ids` list, so every public name starts with `cattorch `. The
  `tokens` list is internal.
- Internal blocks, lists, and variables whose names started with `cattorch `,
  such as the weight decoders and the generation caches (`cattorch K cache 1`),
  now start with `_cattorch`. Names starting with `cattorch ` are the public
  interface.
- `StorageConfig` no longer takes the deprecated `"int8"`, `"int6"` and
  `"int4"` precisions or their `group_size`, `min_quantized_values`,
  `scale_precision` and `grouping` fields. Pass
  `quantization=QuantizationConfig(bits=8)` instead, which groups weights by
  row and uses float16 scales by default.
- The legacy cached-generation recognizer is removed, along with
  `GenerationProgram.example_token`. Export the model's stateless
  `forward(tokens) -> logits` with `GenerationProgram(max_context=...)`
  instead: replace a one-token decoder that reads its position from a zero
  buffer with a forward over the whole sequence. In Scratch timings on a
  TinyStories-sized model, the automatic path matched or beat the legacy one.
- `max_context` is now the first field of `GenerationProgram` and the only
  positional one; `method`, `hidden_prefill`, and `top_k` are keyword-only.

# 0.5.0 - 2026-09-14

## Breaking changes

- `GenerationConfig` and the `generation=` argument are replaced by
  `GenerationProgram`, passed in place of `example_inputs`. It names the model
  method to export and takes an example token and `max_context`.
- Generation sprites now read tokens from `cattorch tokens` and write logits to
  `cattorch logits` instead of `input` and `output`. `cattorch reset cache` is
  renamed `cattorch reset`, and every generation block broadcasts a
  `... complete` message when it finishes. `top_k` now accepts up to 64.
- Integer `StorageConfig.precision` values (`"int8"`, `"int6"`, `"int4"`) are
  deprecated in favor of `QuantizationConfig`.

## New features

- `QuantizationConfig` configures 4-, 6-, or 8-bit weights using symmetric
  quantization or GPTQ. GPTQ reads representative data from
  `calibration_inputs`, `calibration_calls`, or `calibration_sequences`, and
  `TranspileResult.quantization` reports what each tensor used.
- `cattorch.rotary_embedding` applies RoPE and exports to a single loop, about
  10x faster in the Scratch VM than the equivalent matrix expression.
- Models can return several tensors, including nested tuples, lists, and
  dictionaries. They are written to `output`, `output_1`, and so on, and
  `verify` checks each one (`MultiOutputVerifyResult`).
- Fixed slices with a positive step are supported.
- The FX frontend now handles causal `scaled_dot_product_attention`,
  multi-query and grouped-query attention, `stack`, and `select` as written in
  common decoder implementations.
- Experimental: `cattorch.experimental.ExportProgram` exports several model
  methods and persistent state in one sprite. State can be replaced or
  appended to, with a variable-length first dimension up to a fixed capacity.
  Failed calls leave state unchanged. `ModuleAdapter` swaps unsupported module
  types during export, `analyze` checks a model without exporting it, and
  `verify` accepts a sequence of `ProgramCall`s.
- Experimental: `ExpertFamily` and `SparseMoE` export top-k mixture-of-experts
  layers. Only the selected experts run in Scratch. Bias-free SwiGLU experts
  automatically use the faster `StackedSwiGLUMoE` implementation, and GPTQ
  calibrates each expert only on the tokens routed to it.
- `CodegenConfig(layer_sharing="auto")` now works with cached generation.

## Performance

- Top-k selection in generation sprites skips most candidates without scanning,
  matching the speed of the much larger unrolled selector.
- Top-1 `StackedSwiGLUMoE` experts run about a third faster, and fast mode can prune
  their hidden channels with `neuron_pruning`.
- `ExportProgram` sprites store identical weights once even when several methods
  use them.

## Fixes

- Separate temporary lists that start with the same contents are no longer
  merged into one.
- A full slice of a variable-length tensor followed by a view no longer reads an
  empty list.
- `compact_internal_names` no longer renames program inputs, outputs, state, or
  blocks.
- Layer sharing now rolls back cleanly on failure, keeps each layer's final
  output separate, and no longer changes 4-bit MoE weights that cross the
  200,000-item list boundary.
- KV caches larger than 200,000 items can now roll back a failed step.
- `ExpertFamily` rejects experts that differ in structure, parameter roles,
  aliasing, train/eval mode, or dtype, or that modify tensors in place. Its
  template follows `.eval()` and `.to()` and keeps non-persistent buffers.
- MoE export handles batched linear layers inside experts, grouped and
  affine-free normalization, general broadcasting, non-finite router scores,
  and 3D GPTQ calibration data.
- Weight payloads ending exactly at the 200,000-byte list limit no longer lose
  their last values.
- Exact tanh-based kernels stay stable for very large finite inputs.
- Concurrent exports in one process no longer interfere with each other
  through PyTorch's global tracing state.
- `QuantizationConfig` no longer quantizes buffers and constants such as RoPE
  tables and attention masks. They're stored as float16, and masks containing
  `-inf` no longer fail to export.
- A `SparseMoE` whose experts normalize or rescale their input before a SwiGLU
  block is no longer converted to `StackedSwiGLUMoE`, which dropped those
  operations.
- `verify(quantization=...)` for programs now uses the same float16 rules as
  export, including for MoE bias banks.
- An entrypoint input and output that share a name must now have the same
  shape.
- `verify` now matches Scratch for `ln`, `log`, `sqrt`, `floor`, and `ceiling`
  of out-of-range values, for `mod` by zero, for fractional repeat counts, and
  for comparisons involving empty strings.
- `python -m cattorch.benchmark` reports a clear error for projects saved before
  the benchmark finished.
- Quantized weights used on the right of a matmul, including tied LM heads, are
  now grouped the same way in PyTorch and in storage. Before, storage quantized
  GPTQ's result again along the other axis, which could change outputs by
  whole units. They also no longer appear twice in the quantization report.
- BPE and character tokenizers no longer map a digit to a longer token that
  Scratch considers numerically equal, such as `" 1"` or `"01"`.
- 4-bit Huffman weight banks are split when their compressed stream would exceed
  Scratch's 200,000-item list limit, instead of losing their last bytes.

# 0.4.0 - 2026-08-22

## New features

- `exact` applies Scratch-specific transformations that preserve model
  behavior, while `fast` enables optional approximate kernels, structured
  pruning, and low-rank weight transforms.
- Float16, groupwise int8, bit-packed int6, and packed int4 weights are
  encoded with a costume-name Base92 codec and decoded once when the sprite
  initializes. Int4 banks additionally use lossless, length-limited canonical
  Huffman coding.
- Logical tensors and encoded payloads are automatically sharded around
  vanilla Scratch's 200,000-item list limit.
- KV-cached generation: recognized causal language models can export reset,
  prompt-prefill, and single-token decode procedures, lifecycle broadcasts,
  public context state, and a compact top-k streaming sampler.
- SentencePiece BPE tokenizer and detokenizer sprites, with control tokens,
  whitespace conversion, ASCII byte fallback, and optional Scratch-native ASCII
  case folding.
- Exports return structured metadata and can be checked numerically against
  PyTorch. Benchmark builders support both the Scratch VM and timing inside
  vanilla Scratch.
- `CodegenConfig` exposes storage, JSON-budget, loop-unrolling, repeated-layer
  sharing, internal-name compaction, and compact ID namespace settings.
- A fixed-shape FX frontend is now the default, with the previous
  `torch.export` frontend retained temporarily through `frontend="export"` for
  compatibility comparisons.

## Technical changes

- Scratch programs are generated through a typed DSL instead of the private
  JSON-template backend.
- Production kernels use compile-time loop unrolling, fused sequential work,
  caching, and precomputation tailored to Scratch's execution model.
- Generated JSON is reduced through minification, compact schema and IDs,
  shared decoders, tensor deduplication, tied-weight reuse, and coalesced
  weight and scale banks.
- Character and raw-text BPE tokenizers now use the typed DSL and preserve
  backend unknown-token and merge-rank behavior.
- Model and tokenizer exports return structured result objects and raise
  contextual errors for unsupported models and operations.
- Output paths and Scratch sprite names can be configured independently.
- The minimum supported PyTorch version is now 2.6.
- Graph analysis, compilation, sprite assembly, storage, and kernel domains
  have separate implementation modules.
- Frontends normalize into one internal graph contract, and a private operator
  registry now coordinates validation, exact/fast lowering, quantization
  observation, and future semantic module adapters.
- Detailed examples and use cases moved from the README into `docs/`.

## Fixes

- Fixed broadcasting, identity outputs, tensor slicing and splitting, multi-
  input concatenation, transposes, fusion ownership, BatchNorm folding,
  pruning, dropout, dtype casts, and repeated TorchDynamo exports.
- Unsupported operation variants and unsafe optimization shapes now fail
  during export instead of producing plausible but incorrect Scratch output.
- Large tensors, scale banks, quantization edge cases, and non-floating
  constants now preserve Scratch list limits and numeric precision.
- Scratch-incompatible case, Unicode, BPE, and SentencePiece normalization
  contracts are rejected before export.
- Cached generation validates context and positional-table capacity and
  rejects malformed prefill or decode inputs.
- Compact IDs avoid Scratch's fixed toolbox IDs, and costume switches include
  the menu shadows required by the editor.
- Diagnostics distinguish the ordinary expanded `project.json` limit from
  compressed archive-path projects.
- Benchmark reports handle timer-resolution zeros and output-length mismatches
  explicitly.
