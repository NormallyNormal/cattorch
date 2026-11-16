# Unreleased

cattorch 1.0 freezes the public Python API, the `cattorch` command line and
the Scratch interface for 1.x; see [stability](docs/stability.md). Upgrading
from 0.5.0? [Migrating to 1.0](docs/migrating-to-1.0.md) shows what to change
in your code and your Scratch projects.

## Breaking changes

Requirements:

- cattorch needs Python 3.11 or newer (was 3.10), PyTorch 2.10 or newer (was
  2.6), and now depends on NumPy 1.24 or newer.

Python API:

- `transpile`: every parameter after `model, example_inputs, output_path` is
  keyword-only, including `sig_figs` (`transpile(m, x, path, 3)` →
  `transpile(m, x, path, sig_figs=3)`).
- `transpile(frontend=...)` is removed, with the `torch.export` path that
  `frontend="export"` selected. Drop the argument; every export is traced
  with FX.
- `calibration_inputs=`, `calibration_calls=` and `calibration_sequences=` →
  `calibration=`, whose items take the form the export is called with:
  tensors or tuples, `ProgramCall`s, or `[1, length]` token tensors.
- `StorageConfig` no longer takes the integer precisions `"int8"`, `"int6"`
  and `"int4"`, deprecated in 0.5.0, or their `group_size`,
  `min_quantized_values`, `scale_precision` and `grouping` fields. Pass
  `quantization=QuantizationConfig(bits=8)` instead.
- `QuantizationConfig.fallback` is removed; its only value, `"symmetric"`,
  is the behaviour.
- `FastConfig.weights` → `FastConfig.layers`; `FastLayerConfig.pruning` →
  `FastLayerConfig.block_pruning`.
- `CodegenConfig(layer_sharing="auto" | "off")` → `layer_sharing=True |
  False`.
- `GenerationProgram`: `max_context` is the first field and the only
  positional one, and must be at least 3; `example_token` is removed; `method` is optional (the
  method marked `@generate`, else `forward`). The method is now a stateless
  `forward(tokens) -> logits` over the whole sequence, from which cattorch
  derives the cached step; a 0.5.0 method that takes one token must become
  an ordinary forward over the sequence. See
  [KV-cached generation](docs/generation.md#requirements).
- Generation exports return a `GenerationResult` instead of a
  `TranspileResult`: `inputs` describes `cattorch token ids` as
  `[1, 1..max_context]` (it was `cattorch tokens`, `(1, 1)`), `outputs` holds
  `cattorch logits` as `(vocabulary,)` (it was `(1, 1, vocabulary)`) and the
  top-k lists, and `max_context`, `vocabulary_size` and `top_k` are fields.
- A generation export with `QuantizationConfig(bits=4)` or `bits=6` stores
  the output head (such as `lm_head`, or GPT-2's tied embedding table) in
  int8. The sprite is larger and its logits much closer to the float
  model's. `head_bits=4` (or `6`) restores the previous export; see
  [the output head](docs/storage.md#the-output-head-of-a-language-model).
- `TranspileResult.output` and `.additional_outputs` → `.outputs`, every
  output in order; `.output` remains as a property for `outputs[0]`.
- `TokenizerResult.tokenizer_type` values `"CharTokenizer"`,
  `"BPETokenizer"`, `"SentencePieceBPETokenizer"` → `"char"`, `"bpe"`,
  `"sentencepiece-bpe"`.
- `ProgramCallVerifyResult.states`, a tuple of `(name, VerifyResult)`
  pairs → a mapping from state name to `VerifyResult`.
- Benchmark builders: `iterations=` → `repeats=`.
  `build_generation_benchmark_suite` cases are `(name, model, program,
  prompt)` instead of `(name, model, stateless_input, prompt)`; the program's
  `max_context` must hold the prompt and every decode. `analyze_benchmark`
  returns a `BenchmarkAnalysis` instead of a dict, and a case's seconds are
  per call instead of the total of all timed calls. Benchmark projects' stage
  variable `cattorch benchmark iterations` (`cattorch storage benchmark
  iterations`) is `cattorch benchmark repeats`.
- The `cattorch-benchmark` script → `cattorch analyze-benchmark PROJECT`.
- Programs, adapters and their classes are imported from `cattorch`;
  importing them from `cattorch.experimental` warns with a
  `DeprecationWarning`. In `cattorch.experimental`,
  `stacked_swiglu_moe_adapter` is removed (SwiGLU expert blocks switch to
  `StackedSwiGLUMoE` by themselves, and an `@adapter` returning one covers
  stacked expert weights), and `analyze`'s `program` parameter is
  `example_inputs`.
- Tokenizers that a sprite would tokenize differently are rejected with
  `ValueError` instead of exported: `BPETokenizer` with a normalizer,
  pre-tokenizer, `fuse_unk` or `byte_fallback`; `CharTokenizer` and
  `transpile_tokenizer` with multi-character tokens, or a tokenizer that
  doesn't encode one character at a time; SentencePiece models that aren't
  BPE or have user-defined symbols. `scratch_casefold=True` on a tokenizer
  that isn't SentencePiece raises `ValueError` instead of being ignored.
- `FastConfig(neuron_pruning=...)` on a model `torch.fx` can't trace
  symbolically, such as any generation export, raises
  `UnsupportedModelError` instead of skipping the pruning.

Scratch interface (re-export a 0.5.0 sprite to get these):

- Generation sprites' `cattorch tokens` list → `cattorch token ids`.
- Tokenizer sprites read and write the `cattorch text` variable and the
  `cattorch token ids` list instead of the `input` and `output` variables and
  the `token_ids` list.
- `cattorch storage compression` values `costume-base92` and
  `costume-base92-huffman-int4` → `base92` and `base92-huffman-int4`.
- `cattorch status` values, one per situation, naming the list at fault:
  `invalid length: LIST` and `invalid dynamic length: LIST` →
  `wrong number of items: LIST`; `dynamic capacity exceeded: LIST` →
  `too many items: LIST`; `state capacity exceeded: STATE` →
  `state full: cattorch state STATE`; `maximum context exceeded` →
  `context full`; `decode requires exactly one token`,
  `prefill requires at least one token` and `prefill exceeds maximum context`
  → `wrong number of items`, `too few items` and `too many items` of
  `cattorch token ids`. `cattorch status` starts empty instead of `0`. See
  [status values](docs/scratch-interface.md#status-values).
- A program entry point sends `cattorch ENTRY complete` whenever it ends,
  not only after a successful call, so a script reacting to it must check
  `cattorch status`.
- A block that fails changes nothing: a generation prefill or decode with
  invalid input no longer empties `cattorch logits` and the top-k lists, and
  a failed prefill no longer resets the cache first. `cattorch prefill` no
  longer empties `cattorch token ids`.
- Input lists are checked: a model sprite's `cattorch forward` with the
  wrong number of items or a non-number item, an integer list item that isn't
  a whole number from -2^53 to 2^53 (`Infinity` isn't one), and a generation
  block with a token ID that isn't a whole number in the vocabulary, fail
  with a status instead of computing meaningless output. A bool list must
  hold Scratch's `true` and `false` or the numbers 1 and 0; text such as
  `true`, which some operations read as true and others as false, fails with
  `invalid item: LIST`, and a bool input converted to numbers reads 1 and 0
  instead of passing `true` and `false` through to the output.
- Indices are checked as PyTorch checks them: an integer input used as an
  index, by tensor indexing, `gather`, `index_select`, `take`,
  `take_along_dim`, `one_hot`, `scatter`, `index_add`, `index_put`, an
  embedding or an embedding bag's offsets, must hold indices valid for every
  operation that uses it, and an index computed from the inputs, such as
  `x[ids + 1]` or a loss's targets, is checked as the block runs; either
  fails with `invalid item: LIST` naming the input. `index_select`,
  `index_add`, `index_copy`, `index_reduce` and `take_along_dim` refuse
  negative indices, as PyTorch does, and `F.one_hot` needs `num_classes`.
  A generator can't index with an index computed from the token IDs. See
  [index inputs](docs/supported-models.md#index-inputs).
- A program checks the length of its state lists, not their items: an
  append state must hold from its initial rows up to its capacity. State
  lists belong to the sprite; see
  [state](docs/programs-and-moe.md#state).
- `ExportProgram` refuses entry point names whose broadcasts Scratch would
  match as one, since it matches them ignoring case: names that differ only
  in case, a name like another's completion (`x complete` beside `x`), and
  `init`, `reset` and `prepare for save` in any case.
- An input or state list split into shards is checked shard by shard: each
  must hold its part, every shard but the last holding items holding exactly
  200,000, or the block fails with `wrong number of items: LIST`; a
  program's split lists check every shard's items. See
  [long lists](docs/scratch-interface.md#long-lists).
- `cattorch prepare for save` keeps the public lists. It used to empty a
  model's input and output lists and a program's input and state lists.
- Every internal block, list and variable starts with `_`, including the
  0.5.0 ones that started with `cattorch ` (`cattorch decode float32`,
  `cattorch K cache 1`, `cattorch select top k`, ...) or had no prefix
  (`index`, `sum`, `T1`, a tokenizer's `tokens`). Only names starting with
  `cattorch ` and a model's `input` and `output` lists are public.

## New features

### Interface and stability

- The [Scratch interface](docs/scratch-interface.md) is versioned. It lists
  every public block, list, variable and broadcast and every
  `cattorch status` value; within interface version 1 (all of 1.x) none is
  renamed or removed. Each sprite has a `cattorch interface version`
  variable, and Python has `cattorch.INTERFACE_VERSION`.
- Every public block, in every kind of sprite, sets `cattorch status` and
  does a `broadcast and wait` of its `... complete` broadcast whenever it
  ends, unless it is stopped. Model and tokenizer sprites gain `cattorch status` and their
  broadcasts; program sprites gain `cattorch initialized`,
  `cattorch reset complete` and `cattorch prepare for save complete`;
  generation sprites gain `cattorch prepare for save`.
- [Stability](docs/stability.md) says what 1.x keeps compatible and what is
  experimental. `cattorch.__version__` and `cattorch --version` give the
  installed version.
- `CattorchWarning` is the base class of every warning cattorch emits, and
  every warning an export gives is also in its result's `warnings`.
- `CodegenConfig(layer_sharing=True)` warns, saying why, whenever it shares
  nothing, for every kind of export, including a model with no `blocks.N`
  layers. A program reports each entry point whose layers couldn't be
  shared, and warns once if no entry point has layers. See
  [layer sharing](docs/code-generation.md#layer-sharing).

### Generation

- `GenerationProgram(max_context=...)` exports an ordinary stateless
  `forward(tokens) -> logits` as a KV-cached generator and checks the cached
  step against `forward` at export, and that `forward`'s answer for a token
  doesn't depend on the prompt's length, at lengths spread up to
  `max_context`. Separate or combined Q/K/V projections,
  learned positions, RoPE, grouped-query and single-KV-head attention,
  sliding windows (Mistral, GPT-OSS), ALiBi and attention sinks work, and so
  do unwrapped Hugging Face GPT-2, Llama and Qwen2. Models with float16 or
  bfloat16 weights are exported in float32. See
  [KV-cached generation](docs/generation.md).
- `verify` runs a generation sprite's `prefill`, `decode` and `reset` calls
  and compares each call's logits, top-k lists and `cattorch status` with
  PyTorch; see [verify a generator](docs/generation.md#verify-a-generator).

### Recurrent generation

- `RecurrentGenerationProgram` exports token-by-token generation for models
  whose tokens mix through a recurrence, such as GRU and LSTM language
  models: a `step(token, *states) -> (logits, *new_states)` method and the
  states' initial values. Hugging Face Mamba, Mamba-2 and RWKV causal
  language models need no step: `RecurrentGenerationProgram()` builds one.
  The sprite has the same blocks, lists and statuses as a
  `GenerationProgram`'s, and `max_context` is optional. See
  [recurrent generation](docs/recurrent-generation.md).

### Programs, markers and mixture of experts

- Programs (`ExportProgram`, `EntryPoint`, `Input`, `Output`, `State`,
  `StateInput`, `StateUpdate`, `ProgramCall`), `ModuleAdapter` and
  `AdapterContext` are stable and imported from `cattorch`.
- [Markers](docs/markers.md) describe the export interface on your own
  classes without changing how they run: `@entrypoint` exports a method as a
  block and turns the buffers it updates into state, which each entry point
  takes only if it reads or writes it (a parameter, another tensor
  attribute, or a Python value changed on every call, fails the export
  instead of being baked in), `@example_inputs` gives
  `forward` example inputs, `@export_forward` swaps in an export-only method,
  `@moe` declares a block's routing, `@generate` picks a generation method,
  and `@adapter` builds a `ModuleAdapter`. `transpile(model, None, path)` and
  `verify(model, None, result)` read them; `marked_program` returns the
  `ExportProgram` they describe.
- `adapters=` works with every kind of export, not only programs.
- Hand-written mixture-of-experts blocks with a `ModuleList` of experts and a
  linear router usually export without changes, with a `CustomMoEWarning` naming the
  routing found. Hugging Face Mixtral, Qwen2-MoE, Qwen3-MoE, OLMoE,
  DeepSeek-V3 and GPT-OSS export directly. Experimental `routed_experts`
  runs an `ExpertFamily` on routes from your own router code. See
  [mixture of experts](docs/programs-and-moe.md#mixture-of-experts).
- An entry point can take no arguments.

### Models and operations

- Many more operations, among them exact GELU, `log_softmax`, the
  elementwise math, trigonometric and comparison functions, reductions over
  any dimensions, `topk` and `sort`, `cumsum` and `cumprod`, every
  convolution and pooling option (dilation, groups, `padding="same"`,
  transposed and 3D), `gather`, `scatter` and indexing with indices computed
  at run time, `nn.EmbeddingBag`, broadcast batched `matmul`, and in-place
  updates of tensors the model creates or of its inputs, with every in-place
  method and augmented assignment whose out-of-place form exports (`x %= 2`
  included). `torch.pow` takes
  any exponent. See [supported models](docs/supported-models.md) and
  [operator coverage](docs/operator-coverage.md).
- Operations without a kernel of their own export through PyTorch's
  decompositions when those reach supported operations;
  `TranspileResult.fallback_operations` lists them, and fast mode's
  approximations are in `approximate_operations`.
- `nn.RNN`, `nn.GRU` and `nn.LSTM` (stacked, bidirectional, `proj_size`) and
  their cells export, running as one Scratch loop over time.
- Hugging Face GPT-2, Llama, Qwen2 and BERT models export without a wrapper;
  `ModelOutput` is unwrapped and non-tensor outputs are skipped with a
  warning.
- `transpile` and `verify` take `example_kwargs`.
- Computations that depend only on parameters, buffers and constants, such
  as masks and RoPE tables built in `forward`, run at export, and so do
  Python branches and `.item()` on them.
- A model in training mode is exported as an evaluation-mode copy, with a
  warning, instead of being rejected.
- `torch.no_grad()` and `torch.autocast` blocks inside `forward` are
  ignored.

### Variable-length inputs

- `transpile(..., dynamic={0: (1, 32)})` lets one input's length vary at run
  time within bounds, checked against the model at several lengths at
  export; program entry points take `Input(dynamic=...)`. See
  [variable-length inputs](docs/supported-models.md#variable-length-inputs).

### Quantization

- `QuantizationConfig(overrides=...)` gives some weights other settings: a
  module path or parameter name mapped to bits, `None` for float16, or a
  `QuantizationOverride`. `head_bits` sets a generation export's output head.
  Keys name the model's own parameters: a layer fast mode factorizes gives
  its overrides to its factors. See
  [per-weight settings](docs/storage.md#per-weight-settings).
- More weights are stored in integers and calibrated by GPTQ: convolution
  kernels, weights the model
  copies or slices before use (such as a fused QKV weight; a weight read
  directly and through a copy is quantized once), weights computed
  from parameters (`weight_norm`, `spectral_norm`, `w * g`, LoRA sums, and
  a weight masked by a buffer, a tensor attribute or a constant built in
  `forward`), and
  weights of `bmm`, `einsum`, `tensordot`, `F.bilinear`, `nn.MultiheadAttention`,
  recurrent layers and `nn.RNNCell`, `nn.LSTMCell` and `nn.GRUCell`. A 1D or 2D convolution's kernels are grouped over
  each output channel's fan-in, as a linear layer's are, instead of along
  the kernel width. See
  [symmetric quantization behavior](docs/storage.md#symmetric-quantization-behavior).
- A matrix whose integer storage wouldn't be smaller than float16, such as
  one with `group_size=1` or a prime input width, stays float16. GPTQ uses
  symmetric quantization for a matrix with more than 8,192 inputs. The
  report says so in both cases.
- `result.quantization` reports what storage stores, read from storage's own
  record of each list it stored, with `name[float16]` and `name[remaining]`
  entries and `TensorQuantizationReport.sources` and `override`. A
  program's entry points are reported together, and a position table read
  at constant positions has an entry for the rows stored.

### Command line and projects

- `build_project` writes a runnable `.sb3` around exported sprites, with
  bridges other sprites can call by broadcast, list monitors for a model,
  and a prompt-and-continue Demo sprite for a text generator.
- The `cattorch` command: `cattorch export` exports and verifies a model from
  a Python file, module or `.pt`, `cattorch generate` exports a text
  generator and its tokenizer (also `--hf REPO`), and `--project` builds the
  project. See [projects and the command line](docs/projects-and-cli.md).
- An `examples/` directory with small models that train in seconds and
  export with their projects.

### Verification and benchmarking

- `verify` works for every kind of export, and takes the export's
  `adapters`, `quantization`, `storage` and `sig_figs` so PyTorch computes
  with the values the sprite stores, each matrix product reading the list it
  reads in the sprite, such as a float16 copy of a quantized weight.
  `VerifyResult.approximate_operations` names fast-mode approximations. Its
  reference is your own code: a marked model's methods run on its buffers, a
  generation model's forward, and a Hugging Face recurrent model's forward
  rather than the step cattorch writes.
- Every benchmark builder takes `repeats`, `warmups` and `transpile`'s export
  options, and generation benchmarks take a `GenerationProgram` or
  `RecurrentGenerationProgram`. See
  [verification and benchmarking](docs/verification-and-benchmarking.md).
- `benchmarks/run_project_vm.cjs` runs a project in the official Scratch VM.
- Every export checks that the code it generates computes each value before
  reading it, and that every list a block reads exists. A failure raises a
  `RuntimeError` naming an internal error instead of writing a sprite that
  would compute wrong numbers.

### Custom kernels (experimental)

- `transpile(..., kernels=[Kernel(...)])` lowers an operation with your own
  Scratch code from `cattorch.scratch`; `check_kernel` and
  `build_kernel_benchmark` test and time it, and `op` declares a new
  operation. All are in `cattorch.experimental`; see
  [custom kernels and operations](docs/custom-kernels.md).

## Performance

- Exports of large models are several times faster and use a fraction of the
  memory. GPTQ uses the blocked (lazy batch) update and is several times
  faster.
- A token embedding tied to the output head is stored once even when it
  spans several lists.
- A program state that starts as one value everywhere, such as zeros, no
  longer stores its initial copy.
- Hidden prefill also skips the final layer's query, attention output and
  MLP for prompt tokens before the last, with or without layer sharing.
- Layer sharing also shares repeated layers at the start of a model, which
  read the model's input.
- Rotate-half RoPE (`x * cos + rotate_half(x) * sin`) runs as one rotary
  loop.
- In-place operations, such as `x.clamp_(-1, 1)`, `x.sigmoid_()` and
  `x[0] = 0`, run the kernel of their out-of-place operation instead of a
  PyTorch decomposition.

## Fixes

Python:

- Combined-QKV attention gave wrong output with a batch of more than one, or
  when Q, K or V was also used elsewhere.
- Layer sharing gave wrong output when the layers did more than pass one
  tensor along: a skip connection across layers, two tensors passed between
  layers, every layer reading the model's input, or a tensor entering the
  first layer that was also used later. These models are now exported
  unshared, with a warning; see
  [layer sharing](docs/code-generation.md#layer-sharing).
- Layer sharing gave wrong output when code outside the layers read a
  layer's weight other than `blocks.0`'s, such as an output head tied to a
  layer's weight or a tensor with the same values, and when the model's loop
  ran code between two layers that read no layer's output. That code reads
  the layer's rows of the shared bank now, and code between layers keeps the
  model unshared, with a warning that says so (it used to blame weight
  deduplication, and a layer whose operations were all fused into code
  outside it was said to be misnumbered).
- Layer sharing with one layer quantized differently, such as
  `overrides={"blocks.3.fc2": 8}`, stored every layer's weights in float16,
  while the report and warning blamed `blocks.0` alone. Such layers are now
  exported unshared with a warning, each keeping the storage it asked for,
  and the report of a shared bank names every layer's weights.
- `masked_fill` on an integer or boolean tensor wrote a fractional fill
  value as it was instead of casting it to the tensor's type.
- A SiLU multiplied by itself (`s * s`) or by another SiLU
  (`F.silu(a) * F.silu(b)`) gave zeros, and causal attention whose scores are
  `t @ t` for one transposed tensor read values that were never computed.
- A Linear followed by two residual additions added the first one twice, and
  a residual add that broadcasts the Linear's output gave too few values.
- A Linear followed by BatchNorm1d was folded even when the BatchNorm
  normalized another axis, and a BatchNorm used twice normalized wrongly
  after its first use.
- A tensor read through a view after its last direct use could be
  overwritten by a later operation.
- Reading a list of more than 200,000 values at positions taken from another
  such list returned zeros past the first 200,000.
- Program inputs, states and outputs longer than a Scratch list could give
  wrong values with status `ok`.
- A result's `sharded_lists` also named internal lists, such as weights and
  K/V caches; it names only the public lists, whose shards your scripts read
  and write.
- `x @ W` with a large stored weight could fail with an internal error.
- Crashes: `MaxPool1d(1)`, `Conv2d(stride=(2,))`, pooling without an
  explicit stride, softmax and `transpose` of 0-d tensors, `mean(dim=())`,
  `nn.LayerNorm(bias=False)`, `F.rms_norm` without a weight, 0-d parameters,
  `torch.eye` with 256 rows or more,
  concatenating an empty tensor, `F.linear` with a vector weight, and a
  `RecursionError` reading a tensor split into hundreds of lists.
- Wrong output: `1 - x` on a 0-d tensor, `F.batch_norm` with statistics
  passed as inputs, unbatched `AdaptiveAvgPool2d`, `F.elu` with a `scale`,
  and `bool + bool`.
- In a mixture-of-experts layer, operations on an expert's own parameters
  read the first expert's values for every expert.
- Layer sharing merged layers that differ only in a constant, could fail
  with a `KeyError`, and a decode before any prefill wrote the hidden state
  to the logits.
- Layer sharing gave wrong output when the tensor entering the first shared
  layer was also returned or used after the layers, and dropped an output
  copied after the last layer, such as a returned input.
- `FastConfig(neuron_pruning=...)` also pruned parameters the model reads
  directly elsewhere, such as `x @ self.up.weight.T`, changing those outputs'
  values and shapes, and low-rank factorization and block pruning changed
  such reads too. A module whose parameters are read outside it, in any
  exported method, is now left untransformed, with a warning.
- A tied embedding was read in the wrong layout in fast mode with weight
  transforms, and fast weight pruning of `x @ W` changed `W` for other uses.
- Fast low-rank factorization replaced a layer with two plain layers whatever
  else it did, so a subclass of `nn.Linear` or a convolution with its own
  `forward`, such as one that clamps its output, lost those steps and gave
  wrong output, and a layer's forward hooks were dropped. A layer inside
  `nn.MultiheadAttention`, `nn.TransformerEncoderLayer` or another `torch.nn`
  module that reads its weight itself made the export fail with a false
  "check that example_inputs match" error. Such layers now keep their
  weights, with a warning; neuron pruning skips layers with forward hooks
  the same way.
- Reading input-dependent values with `tolist()` or `.numpy()` exported the
  example's values as constants; it is rejected, like `.item()`.
- `sig_figs` rounded integer buffers.
- Two states sharing one initial tensor read one state's list for both.
- An entry point that computed something from an append state's length,
  such as `history.sum(0) / history.shape[0]`, `history.var(0)` or
  `torch.arange(history.shape[0])`, used the length it was traced with.
  Arithmetic with a length now follows it at run time, for each of an entry
  point's append states and variable-length inputs, a number computed from
  two lengths together is rejected, and export checks every entry point at
  several lengths of each append state it reads, rejecting what the sprite
  can't follow.
- Quantization: the report could claim int8 or int4 for weights storage
  kept in float16, a weight with small values or groups of one could be
  stored in float16 without a warning, `min_quantized_values` was applied to
  each list of a split weight, vector parameters were quantized like
  matrices, and sliced weights were quantized twice.
- GPTQ errors name the calibration item that failed, whatever the model
  raised, and calibration items are checked before calibration: a token
  sequence longer than `max_context` or not `[1, length]`, or a tensor item
  that doesn't match the example inputs, is a `ValueError` naming it.
- A tuple passed as one example input raises a `TypeError` that says to pass
  the tensors separately, and a wrongly shaped one reports the model's own
  error.
- The emulator behind `verify` kept NaN through arithmetic, where Scratch
  reads it as 0, and read numbers from text differently from Scratch: it
  read non-ASCII digits such as `１` and trimmed Python's whitespace instead
  of JavaScript's. It now follows scratch-vm, checked against it value by
  value, in its casts, comparisons (by UTF-16 code units), number-to-text
  forms (`1e-7`, `0.00001`, `true`), text lengths and letters (in UTF-16
  units), booleans (`"0"` and `"false"` are false), list positions (`last`,
  `all`, `0x2`), `mod` with infinities, `pick random` bounds, costume
  switching by number, and `for each` counts read before every iteration.
- Mistakes that ended in an internal exception raise one naming the
  problem: a `model` that isn't a module or `example_inputs` that are a dict
  or a number (`TypeError`), a NaN or non-numeric `verify` tolerance, a
  `verify` sprite path given without `.sprite3` (now read, as `transpile`
  named it) or that names a project, `analyze_benchmark` on a file that
  isn't a benchmark project (`ValueError`), and a benchmark case whose
  `example_inputs` aren't tensors (`TypeError`, before anything is
  exported). A model that raises in `verify` or within its `dynamic=`
  bounds raises `UnsupportedModelError` with the inputs' shapes, as
  `transpile` does.
- A model returning tensors inside a dataclass failed with "requires at
  least one tensor output", or lost them with a warning; the error now names
  the container and the supported ones.
- Tokenizer exports accepted a falsy `codegen` of another type, such as
  `{}`, and any truthy `scratch_casefold`, such as `"no"`; they raise
  `TypeError` as `transpile` does. `CodegenConfig`'s `layer_sharing` and
  `unrolling`, which a tokenizer ignored, raise `ValueError` for one.
  `CharTokenizer` and `BPETokenizer` given a SentencePiece model point to
  `SentencePieceBPETokenizer` instead of saying they take one.
- SentencePiece tokenizer sprites decoded the unknown token as nothing; they
  decode it as SentencePiece does (` ⁇ ` by default).
- `QuantizationConfig`'s GPTQ-only `damp_percent`, `block_size` and
  `min_calibration_rows` were ignored with `method="symmetric"`; setting
  them raises `ValueError`. A `FastLayerConfig` rank at or above a weight's
  full rank was skipped silently; it warns, as other skipped
  factorizations do.
- An `output_path` that names no file (empty, ending in a separator, or an
  existing directory) raises `ValueError` instead of writing a hidden
  `..sprite3` or a file beside the directory, and so does such a `-o`. The
  benchmark builders append `.sb3` as every other writer does, instead of
  replacing the last suffix (`results.v2` wrote `results.sb3`).
- Character and BPE tokenizer sprites ignored Hugging Face added tokens, so
  a special token such as `</s>` in the text, or an ordinary added token
  such as `dog cat`, gave other token IDs with no warning. They now match
  added tokens in the text first, as the tokenizer does; see
  [added and special tokens](docs/tokenizers.md#added-and-special-tokens).
  Added tokens with `single_word=True` raise `ValueError`.
- Tokenizer sprites looked characters up case-insensitively, so `The`
  encoded like `the`. Sprites whose vocabulary has letters now look
  characters up by costume name, so don't change their costumes. SentencePiece `scratch_casefold=True`
  sprites mishandled lowercase letters without a vocabulary entry and runs of
  unknown characters.

Scratch:

- Calls from several scripts at once, or stopped part way, could report `ok`
  with wrong outputs, lose a reset, or leave the weights, cache or states
  corrupted for every later call. They now report
  `another call ran during this one` or
  `an earlier call was stopped or is still running; reset first`; see
  [overlapping calls](docs/scratch-interface.md#overlapping-calls).
- Generation top-k could list ID -1 with `-Infinity` when fewer than `top_k`
  logits were finite.
- Generation sprites had a second, unused `cattorch status` variable.
- Initialization left the last compressed weights in a variable, so a saved
  project stored them twice.
- `cattorch prepare for save` clears every internal working list, including
  those of recurrent layers, experts and cached attention and a generator's
  copy of the prompt, which it used to leave holding the last call's values.
- Loose call blocks on the canvas ran internal procedures directly.
- A tokenizer sprite's `cattorch text` started as `0`, so tokenizing before
  setting it tokenized "0"; it starts empty.

## Known limitations

- Call a sprite's blocks from one script at a time. Two rare overlaps are
  still not detected; see
  [overlapping calls](docs/scratch-interface.md#overlapping-calls).
- Tokenizers other than character, raw BPE and SentencePiece BPE, such as
  WordPiece and Unigram, are rejected; see
  [tokenizers](docs/tokenizers.md#compatibility).
- Generation needs causal attention as the only place tokens mix, and a
  fixed `max_context`; see
  [requirements](docs/generation.md#requirements).
- Training and autograd aren't supported. State can be updated in place by
  slices, index tensors and masks, but not through a 0-d index tensor
  (`buffer[position] = x`; use `buffer[position.reshape(1)]`) or a strided
  or transposed view, nor read through a view taken before the update; see
  [limitations](docs/programs-and-moe.md#limitations).
- Scratch reads NaN as 0, and integer arithmetic doesn't wrap around; see
  [inputs and outputs](docs/supported-models.md#inputs-and-outputs).

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
- An entry point input and output that share a name must now have the same
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
