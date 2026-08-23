# Changelog

## 0.4.0 - 2026-08-22

### Added

- Structured model and tokenizer export results.
- Public numerical verification against generated Scratch sprites.
- Contextual cattorch exception types for unsupported models and operations.
- Exact and fast production-kernel dispatch, compressed static storage, list
  sharding, KV-cached generation, and real-Scratch benchmark suites.
- Scratch-safe groupwise int8 and packed int4 weight storage, decoded once at
  initialization with a float16 fallback for matrices too small to repay the
  extra decoder blocks.
- Bit-packed groupwise int6 storage, optional float16 quantization
  scales, and public `CodegenConfig` controls for JSON budget, loop unrolling,
  and compact identifier namespaces.
- DSL-generated SentencePiece BPE tokenizer and detokenizer sprites with
  callable warp procedures, control-token handling, whitespace conversion,
  ASCII byte fallback, and Scratch-native ASCII case folding.
- Optional internal display-name and schema compaction, encoded weight/scale
  banks, an opt-in repeated-layer sharing pass for compatible stateless
  transformers, and a compact loop-based generation top-k selector.
- Generation lifecycle completion broadcasts and public cache-length/context
  state for coordinating Scratch user interfaces with initialization and
  prompt prefill.

### Changed

- Compressed byte payloads now use the browser-benchmarked case-sensitive
  costume-name Base85 decoder. Int6 codes are bit-packed through the same path
  while retaining approximately one payload character per weight.
- Output paths and Scratch sprite names are independently configurable.
- `exact` and `fast` are the only optimization modes. The obsolete private
  JSON-template backend and its duplicated kernel implementations were removed.
- The supported PyTorch minimum is now 2.6, the earliest version exercised by
  the full test suite and current export contract.
- Graph analysis, compilation, sprite assembly, and kernel domains now have
  separate implementation modules.
- Detailed examples and use cases moved from the README into `docs/`.

### Fixed

- State-name collisions, unsafe BatchNorm folding and neuron pruning,
  arbitrary elementwise broadcasting, identity outputs, linear fusion ownership,
  dropout and dtype-cast semantics, and repeated-export TorchDynamo limits.
- Project-size diagnostics now apply the ordinary online limit to expanded
  `project.json` rather than treating 5 MiB as a universal compressed-archive
  cap; large VM and archive-path benchmarks warn instead of being rejected.
- Generated sprite and benchmark JSON now omit insignificant separator
  whitespace, reducing project size without changing Scratch semantics.
- Final sprite serialization compacts internal block and data IDs under a
  per-export namespace, removing repeated construction-time labels from JSON.
- Internal IDs now use a Base64url counter and a configurable three-character
  default namespace instead of a ten-character hexadecimal prefix.
- Static storage deduplicates bit-identical tensors, reuses eligible tied
  embedding/output-head layouts, coalesces encoded bytes and scales by
  precision, and shares one startup decoder per precision.
- Production character and raw-text BPE tokenizers now compile from the typed
  Scratch DSL, and symbolic kernel loops select their unroll factor at JSON
  lowering time according to the configured size budget.
- The TinyStories streaming sampler now runs until EOS or the total context is
  full instead of using a separate output-token limit, and sizes its frequency
  table from the exported vocabulary at runtime.
- Compact IDs now retain a `ct` prefix and skip Scratch's fixed toolbox IDs;
  this prevents large sprites from colliding with the sensing `of` flyout block
  and making the vanilla editor's block sidebar disappear. Costume switches
  also emit the menu-shadow blocks required by scratch-blocks.
- ID remapping is now schema-aware, so compacting block/data IDs cannot rewrite
  equal literal strings in tokenizer vocabularies, variable values, or list
  payloads.
- Coalesced encoded-weight banks now split before their temporary decoded-byte
  list reaches vanilla Scratch's 200,000-item cap, preventing later tensors
  from silently decoding as zeros in larger models.
- Encoded scale banks and individual float32/float16 tensors now obey the same
  temporary-list cap; precision-aware static sharding prevents oversized
  decoder streams, including across several otherwise-small tensors.
- Float16 quantization-scale underflow remains finite, scale overflow reports a
  clear error, and non-floating constants can no longer silently round through
  compressed float32 storage.
- Explicit compact-ID namespaces are now either three Base64url characters or
  empty, preventing ambiguous namespace/counter concatenations across sprites.
- Slice bounds and split/chunk offsets now use correct flat tensor geometry;
  multi-input concatenation retains its accumulator; and singleton transpose,
  SwiGLU, and embedding-add aliases/fusions no longer accept unsafe broadcast
  or layout cases.
- Unsupported matmul broadcasting, convolution/pooling modifiers, reduction
  dimensions, power domains, BatchNorm state, add/subtract `alpha`, and exact
  GELU variants now fail at export instead of producing plausible wrong output.
- Character and raw BPE tokenizers now handle unknown input according to their
  backend, use canonical BPE merge ranks, and reject case/Unicode contracts
  vanilla Scratch cannot preserve. SentencePiece preprocessing configuration is
  validated before export.
- Cached generation now validates positional-table capacity, rejects malformed
  decode and prefill inputs, clears invalid top-k output, and stops the bundled
  TinyStories sampler when prefill produces no logits.
- Benchmark analysis now handles timer-resolution zeros and reports output
  length mismatches explicitly.
- SentencePiece BPE export now considers the complete merge table when it is
  longer than the vocabulary, preventing valid late-ranked word merges from
  being silently skipped.
- The production TinyStories build now uses its dynamic-position cached wrapper
  so RoPE advances with the KV cache instead of reusing position zero, and
  explicitly declares its case-folded SentencePiece normalization.
