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
