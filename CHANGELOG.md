# Unreleased

## New features

- Experimental custom kernels: `transpile(..., kernels=[Kernel(...)])` lowers
  an operation with your own Scratch code for that export, replacing a
  built-in kernel or a PyTorch decomposition. Kernels receive a context
  naming each input's list, shape, dtype, and whether it is a stored weight,
  and return statements from the new `cattorch.scratch` module. A validator
  can reject calls, and `approximate=True` kernels apply only in fast mode.
  `check_kernel` compares a kernel with PyTorch in the emulator, and
  `build_kernel_benchmark` times it in Scratch. `ctx.argument_names` gives
  the arguments in order, for a kernel that reads them by position. Every
  built-in kernel is written the same way. See
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
- `RecurrentGenerationProgram(states, step="step")` exports token-by-token
  generation for a model whose tokens mix through a recurrence, such as a GRU
  or LSTM language model, Mamba or RWKV: a method `step(token, *states) ->
  (logits, *new_states)` and the states' initial values. The sprite has the
  same init, reset, prefill and decode blocks, lists, statuses and
  notifications as a `GenerationProgram`, so `build_project` and Scratch
  projects written for one drive the other. Reset restores each initial
  state, a failed decode leaves the states unchanged and a failed prefill
  restores the initial ones,
  `max_tokens` optionally limits the tokens (by default `cattorch max context`
  is `Infinity`), hidden prefill skips what only the logits need for prompt
  tokens, and `reference=` checks the step against a stateless method at
  export. Hugging Face Mamba, Mamba-2 and RWKV causal language models need no
  step: `transpile(model, RecurrentGenerationProgram(), path)` builds one from
  the model's weights, with one state per layer and kind, and always checks it
  against the model's forward, which catches transformers versions the step
  doesn't follow. The caller's model isn't changed, including RWKV's in-place
  rescaling. A `GenerationProgram` given one of these models points to
  `RecurrentGenerationProgram`, and `cattorch generate` exports them as
  recurrent generators, with `--max-context` optional. See
  [recurrent generation](docs/recurrent-generation.md).
- `ProgramResult` has `fallback_operations` and `approximate_operations`, and
  warns about decomposed operations, as `TranspileResult` does. A
  `RecurrentGenerationProgram` export reports both.
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
- `nn.EmbeddingBag` and `F.embedding_bag`, in `sum`, `mean` and `max` modes,
  with `per_sample_weights`, `padding_idx` and `include_last_offset`. Bags
  of one size, such as a 2D input, cost about as much as an embedding and a
  reduction; other offsets, fixed in the model or passed as an input, add a
  scatter into the bags. The table is stored and quantized like an
  `nn.Embedding` table, and a 2D input's number of bags can be
  [variable-length](docs/supported-models.md#variable-length-inputs).
  `max_norm` on an embedding is rejected with a message naming it.
- A variable-length input times a batched weight, such as `[n, 4] @
  [3, 4, 5]` or `[n, 4] @ [2, 3, 4, 5]`, follows the length: the weight's
  batches are folded into its columns, so one matmul keeps the items as
  rows, at the cost of the fixed-shape product. A variable batch against a
  batched weight, such as `[n, 1, 3, 4] @ [3, 4, 5]`, runs once per item
  instead of failing the length check, and rows inside a batch against a
  batched weight are rejected with a message saying so.
- `torch.lgamma` (also `Tensor.lgamma` and `torch.special.gammaln`), and
  through it `torch.mvlgamma` and `torch.special.multigammaln`. It agrees
  with float64 PyTorch to about 1e-14 over the whole domain, including
  negative arguments, and is exactly zero at 1 and 2 with full
  relative accuracy around them; poles and both infinities give Infinity.
- `cattorch --version` and `cattorch.__version__` give the installed
  version. `cattorch generate --help` now explains `--top-k`,
  `--max-new-tokens`, and `--temperature`, `analyze-benchmark --help` its
  `PROJECT`, and the help shows each option's default.

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
- A program `State(mode="replace")` that starts with one value everywhere,
  such as zeros, is no longer stored in the project, where it was written
  twice (the state and its initial copy). Its list is saved empty and filled
  by `cattorch init`, `cattorch reset`, or the first entry point call. A
  450,000-value zero state now adds almost nothing to the project instead of
  about 3.6 MB.
- Exports of large models are several times faster and write the same
  sprite as before. Renaming a program's weights and interface lists walks
  the blocks once instead of once per list, attaching weights no longer
  copies every block, merging a new procedure's data only remaps its own
  blocks, and weight quantization and encoding are vectorized. On a 4-core
  machine, random-weight Mamba2-130m shapes with the 50,288-token vocabulary
  in int8 took 32 s instead of 258 s for 2 layers and 35 s instead of 300 s
  for 4; a 2-layer GPT-2 `GenerationProgram` in float32 took 46 s instead of
  243 s. cattorch now depends on numpy.
- Large exports use a fraction of the memory, and write the same sprite as
  before. Weight lists stay as arrays until storage encodes them instead of
  holding every value as a Python float; the checks and quantization that
  run over a whole weight work a few rows at a time; and a model copy the
  export made itself is adapted in place instead of copied again. Measured
  on a 4-core machine with random weights in Mamba2-130m's shape, the
  50,288-token tied embedding and int8: 2 layers peaked at 1.3 GB instead of
  6.2 GB (10 s instead of 25 s) and 8 layers at 1.7 GB instead of 7.4 GB
  (16 s instead of 38 s). All 24 layers peaked at 2.7 GB in 35 s; that
  export wasn't repeated before the change, where the smaller ones
  extrapolate to about 11 GB. A 2-layer, 768-wide GPT-2
  `GenerationProgram` in int8 peaked at 0.7 GB instead of 1.3 GB.

## Fixes

- With `quantization`, the weight of `torch.addmm(bias, x, weight)`, which
  GPT-2's `Conv1D` layers use, was quantized along its outputs instead of
  its inputs, so storage couldn't hold it in integers and kept it in float16,
  and GPTQ skipped it. It is now quantized and calibrated like any linear
  weight and stored in int8 or int4 as requested. Such a weight larger than
  1,048,576 values, as in GPT-2 at full width, was also transposed in Scratch
  at run time; it is now stored transposed. In a random-weight 2-layer GPT-2
  `GenerationProgram` with a 1,000-token vocabulary, `sprite.json` at width
  256 shrinks from 4.9 MB to 3.0 MB in int8 and from 4.8 MB to 1.8 MB in
  int4, and at width 768 from 24.8 MB to 20.7 MB in int8 and from 15.3 MB to
  10.3 MB in int4, with about a third of the blocks. Quantized GPT-2 exports
  compute slightly different outputs than before.
- With `quantization`, more weights that storage kept in float16 are stored
  in int8 or int4 as requested: those of `torch.einsum` with the weight's
  input dimension last (`'bi,oi->bo'`, per-head `'bhi,hoi->bho'`, and their
  batched and `...` forms), `torch.tensordot(x, weight, dims=([1], [0]))`,
  and `F.bilinear` and `nn.Bilinear`. In small test models, an int4
  `nn.Bilinear(64, 32, 16)` export shrinks from 82.6 KB to 36.5 KB and two
  einsum layers from 34.0 KB to 25.1 KB. Weights on the right of `torch.bmm`
  or a batched matmul, and of einsums such as `'bi,io->bo'` that were
  already stored in integers, are now grouped over their inputs like other
  weights. GPTQ now calibrates all of these, and weights used through
  `torch.mv`, `torch.addmv`, `weight @ x`, `torch.bmm` with the weight on
  either side, `torch.baddbmm`, `nn.MultiheadAttention`'s fused inference
  path, and the input weights of `nn.RNN`, `nn.LSTM` and `nn.GRU` (and their
  hidden weights with one layer in one direction), instead of falling back
  to symmetric quantization. An einsum, `tensordot` or bilinear weight larger
  than 1,048,576 values is now rearranged at export instead of in Scratch at
  run time, also without quantization. A quantized weight that storage still
  keeps in float16, because the model copies it into another layout itself,
  such as `F.linear(x, weight.t().contiguous())`, now gets a warning in
  `result.warnings`.
- GPTQ now calibrates convolution kernels (`nn.Conv1d`, `nn.Conv2d`,
  `nn.Conv3d`, grouped and depthwise ones, and their transposed forms) over
  the input windows each group of kernels reads, instead of falling back to
  symmetric quantization with a warning. Its groups run along the kernel's
  last dimension, as storage reads them, so the kernels stay in int8 or
  int4. In small CNNs trained briefly on synthetic images, int4 GPTQ instead
  of int4 symmetric lowers the output error relative to the float model from
  1.4% to 0.8% for plain 3x3 convolutions, 4.5% to 1.2% with grouped ones,
  3.5% to 1.1% with depthwise ones, 4.0% to 0.6% for a 1D network and 10.7%
  to 4.0% for an autoencoder with transposed convolutions.
- With `quantization`, a weight the model copies before using it, as in
  `F.linear(x, weight.t().contiguous())`, `weight.T.clone()` or
  `x @ weight.T.contiguous().T`, is now quantized over the inputs of the
  copy and stored in int8 or int4 as requested, instead of in float16 with
  a warning, and GPTQ calibrates it. The model's own copy is made from the
  quantized weight, so PyTorch and Scratch compute with the same values.
- `result.quantization.tensors` reported a quantized weight that storage
  keeps in float16, because it reads it grouped along another axis, as
  `"symmetric"` or `"gptq"`. It now reports `method="float16"`, with a
  `fallback_reason` naming the rounding applied and why it is stored in
  float16, and leaves it out of `quantized_values`.
- A quantized export could fail with `grouped weight sharding must preserve
  complete four-row groups` when compressed storage keeps a weight that runs
  four output rows at a time in float16 or float32, whose lists hold fewer
  values than Scratch's limit, such as a model with a quantized and an
  unquantized matrix product weight of the same shape. Such weights are now
  cut into lists at the size storage gives each one.
- An export failed with `RecursionError` when it read a stored tensor split
  into several hundred lists into a list larger than Scratch's limit,
  for example an embedding of 31 million values in float32 storage looked up
  for 300 tokens. `verify` and the emulator also handle such sprites.
- A token embedding tied to the output head and larger than one Scratch list
  was stored twice, as the head's weights and again as the embedding's own
  table, in float16 beside an int8 head when quantized. Now the embedding
  reads the head's lists. In a random-weight 2-layer Mamba2-130m shape with
  the 50,288-token vocabulary, `sprite.json` shrinks from 161 MB to 65 MB in
  int8 and from 428 MB to 238 MB in float32, and in a 2-layer GPT-2
  `GenerationProgram` in float32 from 470 MB to 287 MB (with about a third
  more blocks: the float32 head reads more, smaller lists). A table that must
  still be stored twice, because another operation or a custom kernel reads
  it in a different layout from the head's, is reported in `warnings`.
- `ExportProgram` and `@entrypoint` sprites had no protection against calls
  from several scripts at once (Scratch pauses a block after 500 ms and runs
  other scripts) or stopped part way. A `cattorch reset` or another call made
  while a call was paused was lost or undone, and the paused call reported
  `ok`; calls share temporary lists, so even a sprite without state could
  report `ok` with wrong outputs, and a call paused while decoding the
  weights could leave them wrong for every later call. A call stopped while
  writing the states, or filling them in a freshly loaded project, left them
  partly written. Now a call that another block started during commits
  nothing and reports `another call ran during this one`. While a call or
  reset writes the states, and after one was stopped then until
  `cattorch reset`, entry point calls are refused with `an earlier call or
  reset was stopped while changing the states, or is still changing them;
  reset first`. A call stopped before it writes the states changes nothing,
  as before. Two overlaps are still not caught. See
  [stopped and overlapping calls](docs/programs-and-moe.md#stopped-and-overlapping-calls).
- In `GenerationProgram` and `RecurrentGenerationProgram` sprites,
  `cattorch reset` or `cattorch prefill` run from a second script while
  Scratch had paused a decode or prefill (it pauses a block after 500 ms)
  let the paused block resume and write into the new caches or states, and
  it and later decodes reported `ok` with wrong logits. Such a block now
  reports `another reset, prefill, decode or init ran during this one; reset
  or prefill first`, and decode fails until the next reset or prefill. A
  decode refused because another block is running stops that block too; its
  status now reads `an earlier reset, prefill or decode was stopped or is
  still running; reset or prefill first`. A block taken over while it
  decoded the weights, in a freshly loaded project, could leave them wrong
  for every later block, reset and prefill included; it now leaves them for
  the next block to decode again. One overlap is still not caught. See
  [overlapping calls](docs/generation.md#overlapping-calls).
- Sprites from `transpile(model, example)`, with fixed or variable-length
  inputs, had no such protection either. A `cattorch forward` that Scratch
  paused, then resumed after another script's forward, finished with that
  call's temporary values and reported `ok` with wrong outputs, overwriting
  the other call's. A `cattorch init` paused while it decoded the weights
  let a forward meanwhile use them half decoded and report `ok`, and, if it
  resumed after a forward had decoded them, left them wrong for every later
  call. Now a forward that another `cattorch forward`, `cattorch init` or
  `cattorch prepare for save` started during reports `another call ran during
  this one`, as program calls do, clears its outputs, sends no completion
  broadcast and fails a forward it resumed during, and an init that another
  call started during leaves the weights for the next forward to decode
  again. A sprite has about 24 more blocks, and a forward runs 3 more. The
  clickable forward on the canvas of a variable-length export ran the
  procedure it wraps, without the length checks; it runs `cattorch forward`.
  Two overlaps are still not caught. See
  [overlapping calls](docs/getting-started.md#overlapping-calls).
- `cattorch init` in `GenerationProgram` and `RecurrentGenerationProgram`
  sprites with compressed weights (the default, and quantized ones) had no
  such protection. An init paused while it decoded the weights let a
  prefill or decode meanwhile use them half decoded and report `ok` with
  wrong logits; resumed after a block had decoded them, or run twice at
  once, or stopped part way, it left them wrong for every later block. In a
  freshly loaded project, a decode or prefill paused while it decoded the
  weights, then resumed after an init from another script, reported `ok`
  with wrong logits and left the weights wrong; in scratch-vm the next
  prefill and decode were off by up to 1.4. Now init marks the weights
  undecoded while it decodes them, and a reset, prefill or decode that an
  init started during fails as one that another of them started during
  does; that status now reads `another reset, prefill, decode or init ran
  during this one; reset or prefill first`. A sprite has 12 more blocks;
  reset, prefill and decode run no more blocks, init 6 more.
  Uncompressed sprites decode nothing, so they had no such problem, and
  `ExportProgram` sprites already guarded init. One more overlap, in these
  sprites, plain exports and programs alike, is still not caught: a block
  that runs while a taken-over init, resumed, still decodes the weights. See
  [overlapping calls](docs/generation.md#overlapping-calls).
- In every kind of sprite with compressed weights (the default, and
  quantized ones), a block that Scratch paused before it began decoding the
  weights in a freshly loaded project, and resumed after another block had
  started, decoded them in that block's epoch, as if current; a block
  decoding the weights meanwhile, such as another entry point of an
  `ExportProgram` sprite, could then report `ok` with weights decoded from
  the lists it changed. Now the procedures `cattorch init` runs directly
  take the epoch of the block that started them and check it as they
  start, and a step, compute or checked forward passes its own: such a
  block decodes nothing. A compressed sprite has 8 to 30 more blocks; init
  runs 1 to 3 more; calls run none more, except 1 per token in a recurrent
  generator with a state filled at run time. A `cattorch init`, or a block
  in a freshly loaded project, taken over part way through decoding still
  carries on when it resumes, as above: a check at every loop of the
  decoders would stop it too, but costs 110 to 380 blocks in small and
  medium sprites, nearly 1,000 in a four-layer generator, and makes init
  run 6 to 23% more blocks, so it is left off for now. Uncompressed sprites
  are unchanged. See
  [overlapping calls](docs/generation.md#overlapping-calls).
- Generation, recurrent generation and `ExportProgram` sprites had loose call
  blocks on the canvas that ran private procedures directly. The one in a
  `GenerationProgram` sprite ran a step without checking its token or marking
  the generator busy. They are gone; a `GenerationProgram` sprite keeps its
  clickable `cattorch init`.
- The emulator behind `verify` read text as Scratch does: `0x`, `0o` and `0b`
  numbers and `Infinity` are numbers, while Python-only forms such as `inf`
  and `1_0` aren't. Rounding an infinity no longer crashes, and changing an
  empty variable starts from 0.
- GPTQ errors name the argument to fix: a `GenerationProgram` without
  calibration sequences asked for `calibration_inputs`, which it then
  rejected, and an `ExportProgram` without calls did the same; non-GPTQ
  `calibration_sequences` were reported as `calibration_inputs`. A
  calibration token outside the vocabulary, or not a whole number, is a
  `ValueError` naming the sequence and position, instead of an `IndexError`
  or a PyTorch error.
- Stopping a `GenerationProgram` sprite during a hidden prefill made every
  later `cattorch decode` report `ok` and advance `cattorch cache length`
  without writing logits, even after `cattorch reset`. Decode now computes its
  logits whatever a stopped prefill left behind, and after any stopped reset,
  prefill or decode it refuses with a status naming the problem until
  `cattorch reset` or `cattorch prefill` runs, instead of decoding from a
  half-updated cache. See
  [cache behavior](docs/generation.md#cache-behavior).
- An `ExportProgram` entry point, or a variable-length export, could return
  wrong values with status `ok` when the compiler first used an output's list,
  or a list shared with a variable-length tensor, for a larger intermediate
  longer than a Scratch list, as in `(state.reshape(5, -1) * w).sum(-1) + b`
  with a state over 200,000 values: that list wasn't split, and Scratch
  dropped the items past its limit.
- An `ExportProgram` entry point computed wrong values when one of its
  inputs, states or outputs was longer than a Scratch list and stored
  weights or intermediate lists were split too: splitting the interface
  lists reused block IDs from splitting the others.
- Variable-length exports:
  - Elementwise operations on an input with one value per item, such as a
    1-D input or a `(T, 1)` column, read only the first item.
  - `silu(x) * y` failed to export, or crashed with a `(1, D)` parameter.
  - Fast mode's low-rank matmul sized its loop from the example.
  - A model that can't run on one item now says so, instead of raising a
    PyTorch error.
- Max pooling crashed when every window held one value, as in `MaxPool1d(1)`.
- `Conv2d` with a one-value tuple, such as `stride=(2,)`, crashed.
- Softmax of a 0-d tensor and `mean(dim=())` or `mean(dim=None)` crashed.
- `x @ W` with a stored `W` whose inner dimension is wider than a Scratch
  list raised an internal error; it now exports as `nn.Linear` does.
- Fast mode lists the approximations used inside mixture-of-experts layers in
  `approximate_operations`.
- Experts take `x @ W`, so low-rank expert layers work as documented, and
  `x ** 0`.
- `torch.nan_to_num` replaced nothing. Its default `nan=0` and infinity
  replacements now work; other `nan` values are rejected, since Scratch can't
  tell NaN from 0.
- A kernel passed in `kernels=` now always runs. Before, fusions skipped it
  without a word: a Linear with its activation, scalar or residual, `silu(a) *
  b`, arithmetic chains, token plus position embeddings, rotary embeddings,
  causal attention, and large stored matmuls. In fast mode, the built-in
  approximate GELU, softmax and layer norm won over an exact kernel. On a
  variable-length input, a kernel for a reduction was replaced by the
  built-in one, and kernels for softmax, layer norm and mean ran on the
  traced size instead of once per item. A user embedding or Linear kernel on
  a tied table read the library's interleaved layout.
- A kernel's validator gets the same context as its `emit`, including
  `dynamic_axis` and stored values, and a user Linear kernel gets the call's
  own arguments, without the library's extra slots.
- `Kernel` rejects targets that name no single tensor operation, such as
  `torch.ops.aten.add` (whose `default` adds two numbers) or `torch.relu`, and
  an export warns about a passed kernel it never used.
- Combined-QKV attention (one projection split into Q, K and V) gave wrong
  output with a batch of more than one, or when Q, K or V was also returned
  or used elsewhere.
- A token embedding tied to the output Linear was read in the wrong layout in
  fast mode with weight options, for very wide tables, and by any third
  operation reading the table.
- Fast weight pruning of `x @ W` no longer changes `W` for other operations
  that read the same buffer.
- A Linear's residual add that broadcasts the Linear's output gave too few
  values.
- In a mixture-of-experts layer, operations on an expert's own parameters
  (such as `torch.sigmoid(self.gate)` or a slice of it) and a Linear with a
  computed weight read the first expert's values for every expert.
- `bool + bool` adds as a logical or, as in PyTorch.
- Integer powers up to 64 are computed by repeated multiplication, so whole
  results such as `3 ** 5` are exact.
- `F.linear` with a vector weight or a one-value bias.
- A custom kernel for `aten.linear` crashed with `TypeError` when the weight
  was large enough for the library to group or split it. It now gets the
  weight as PyTorch stores it.
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
- Sharing identical weights could keep a copy no block read, so another
  weight's list was left empty and read as zeros. This showed up with
  quantization or fast-mode low rank as an unexpectedly large error.
- A Linear followed by BatchNorm1d was folded even when the BatchNorm
  normalized another axis, as for `(batch, tokens, features)` inputs, giving
  wrong numbers or an `AssertionError`.
- Reading a list of more than 200,000 values at positions taken from another
  such list, as broadcasting and large transposes do, returned zeros past the
  first 200,000 values. `x[i]` on such an input copied it into a list that
  lost the same values.
- `cattorch prepare for save` emptied input lists, which in a program sprite
  hold its state, so the next call failed until `cattorch reset`.
- `sig_figs` rounded integer buffers, such as permutations and token offsets.
- A kernel passed with `kernels=` for an elementwise operation ran on the
  traced size when its input grew at run time. It now runs once per item, and
  export fails where it can't.
- Automatic MoE detection swapped a block that caps how many tokens an
  expert takes for plain top-k routing, which is wrong once an expert is full.
- `cumsum` and `cumprod` ran as a decomposition whose size grows with the
  square of the length: a 1,600-item `cumsum` made 11.8 MB of JSON, and a
  210,000-item one didn't finish. They now run as one pass.
- float16 and bfloat16 models, including Hub checkpoints loaded by
  `cattorch generate --hf`, were rejected for generation with a misleading
  message. They are exported in float32.
- The command line printed a traceback for an unsupported operation instead
  of an error message.
- Generation models with `-inf` logits, as for banned tokens, were rejected.
- `build_project` gave a sprite named `cattorch` global lists that collided
  with its own, and could make two sprites named `Demo`. Such sprites are
  renamed, like duplicates.
- Generation sprites had a second `cattorch status` variable, declared by a
  procedure meant to undo a step that failed partway. Nothing called that
  procedure. Both are gone, which saves about 13 blocks plus 17 per cache
  list.
- Generation sprites accepted token IDs outside the vocabulary, fractions and
  text, and reported `ok`. They now report an error.
- A variable-length `cattorch forward` accepted any input length and reported
  `ok`. It now reports too few items, too many, or a partial item.
- A variable-length export whose output gives several rows per item, as
  `x.reshape(-1, 2)` does, crashed.
- `frontend="export"` leaked raw `torch.export` errors.
- `transpile` rejected `adapters=` for plain and variable-length exports.

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
- A SentencePiece Unigram model was exported as a BPE tokenizer with
  different token IDs, and SentencePiece `word` and `char` models were also
  accepted. `SentencePieceBPETokenizer` and `transpile_tokenizer` now read the
  model type and reject anything but BPE, as well as models with user-defined
  symbols, which SentencePiece matches before BPE.
- `CharTokenizer` exported any tokenizer, so a WordPiece, word-level or
  Unigram tokenizer gave a sprite that looked up characters with different
  token IDs. It now rejects multi-character tokens, a normalizer, and any
  tokenizer that doesn't encode text one character at a time, such as a
  single-character vocabulary behind a `Whitespace` pre-tokenizer.
- Tokenizer sprites looked characters up with Scratch's case-insensitive
  `item # of`, so a letter whose case wasn't in the vocabulary was read as the
  other case (`The` encoded like `the`), and vocabularies with both cases
  were rejected. That included every real SentencePiece model with
  `byte_fallback=True` that lacked some ASCII letter as a normal piece, since
  its byte pieces for `B` and `b` collide. Sprites whose vocabulary has
  letters now get a costume per character and find characters by costume
  name, which Scratch matches exactly; `scratch_casefold=True` remains for
  case-folding models. Such tokenizer sprites change; one without letters is
  unchanged. The sprite adds 4 blocks and about 140 bytes of JSON per
  character, and runs at about the same speed.
- A SentencePiece byte-fallback export now also warns that a non-ASCII
  character outside the vocabulary encodes as the unknown token, where
  SentencePiece emits its UTF-8 bytes.
- `build_project` and the benchmark builders wrote an infinite or NaN example
  value as a bare `Infinity` or `NaN`, which isn't JSON, so the Scratch VM
  refused to load the project. Every Scratch archive now holds strict JSON,
  with these values as the strings `"Infinity"`, `"-Infinity"` and `"NaN"`,
  which Scratch reads as numbers. The command line's JSON output is strict
  too, and `benchmarks/run_project_vm.cjs` and the other VM runners write
  these values as the same strings instead of `null`.
- `cattorch generate -o build/demo.sprite3 --project` wrote the tokenizer as
  `demo.sprite3_tokenizer.sprite3`, and `-o y.sb3 --project` wrote
  `y.sb3.sprite3` and `y.sb3.sb3`. Every output path now comes from `-o` with
  any `.sprite3` or `.sb3` removed: `PATH.sprite3`, `PATH_tokenizer.sprite3`,
  and `PATH.sb3`.
- The command line showed a raw traceback for common mistakes: a missing,
  unreadable, or wrong kind of file (including `analyze-benchmark` on a
  missing file), an unwritable output, a `.pt` whose model class can't be
  imported, an unknown module, a Hugging Face model or tokenizer that can't
  be found, and `--vocab 0`. These are now `cattorch: error:` messages; the
  `.pt` one says the class must be importable and how. Loading a `.pt` also
  looks for the class's module in the working directory and beside the file.
  `--shape 1,x` was read as `1`, and a repeated `--dynamic` for the same
  dimension was accepted silently; both are now errors.
- `transpile_tokenizer`, `CharTokenizer`, and `BPETokenizer` given an object
  that isn't a tokenizer raised `AttributeError: ... 'get_vocab'`; they now
  raise a `TypeError` naming the supported tokenizer types.
- With `quantization`, a weight used through slices, such as a fused QKV
  weight split with `qkv[:d]`, even and odd rows, or a half read by
  `F.linear` and a half on the right of a matrix product, was quantized for
  its slices and then again as a whole along its rows. The report counted
  its values twice, GPTQ reported a false "no compatible linear activation
  statistics" fallback, and a slice rounded twice along different axes
  ended up in float16. Each part a use reads is now quantized once, in the
  layout that use reads it in, and the rest of the parameter once along its
  rows. In small test models, an int4 export of two column halves on the
  right of matrix products shrinks from 32.0 KB to 22.1 KB, with the output
  error relative to the float model down from 0.31 to 0.25, and one of a
  linear half and a right-hand half from 28.0 KB to 22.0 KB. GPTQ now also
  calibrates disjoint views, which it called "overlapping matrix views", and
  a weight read both by `F.linear(h, weight)` and `h @ weight.T`.
- `result.quantization` now says what storage stores: each stored list is
  traced back to the parameter values it holds. It reported int4 or int8 for
  weights storage kept in float16, such as a weight with more than 50,000
  rows on the right of a matrix product, and the warning named an internal
  list such as `'_param_constant0'` instead of the parameter. Values stored
  in float16 besides an integer copy, as when two uses read one weight along
  different axes, get a `name[float16]` entry.
- With `quantization`, a weight with small values, such as a row whose
  largest value is 0.0008 in int8, or any weight quantized in groups of one
  (`group_size=1`, or an input width that is prime, such as 97), could be
  stored entirely in float16 without a warning while the report said int8.
  A group's float16 scale, rounded coarsely below 6e-5, left its largest
  code one short, and storage then derived a different scale and rejected
  the tensor. Quantization, the encoder and storage now share one scale rule
  that gives a quantized group's scale back from its values. GPTQ uses it
  too, so a few GPTQ groups whose scale PyTorch's float16 conversion rounded
  differently now keep the values GPTQ chose: GPTQ exports change in about
  50 of every million values.
- With `quantization`, the parameters of a weight computed from them, such
  as `weight_norm`, `spectral_norm` or `F.linear(x, w * g)`, were rounded to
  integers, and the weight export computed from them was stored in float16
  without a warning, about 7% off the float model in int4. The parameters
  are no longer rounded; the weight is stored in float16 as computed, with a
  warning and a report entry named after its parameters.
- With `quantization`, parameters that are vectors, such as
  `nn.MultiheadAttention`'s `(1, 1, E)` `bias_k` and `bias_v` and `(N, 1)`
  gains, were quantized like matrices. A matrix is now a parameter read as
  one, or one with at least two dimensions larger than one; other parameters
  stay float16, as for biases.
- GPTQ now calibrates the input and hidden weights of every layer and
  direction of `nn.RNN`, `nn.LSTM` and `nn.GRU`, including projected LSTMs,
  instead of only the first layer's input weights (and hidden weights for
  one layer in one direction). A projected LSTM's projection, which reads a
  value no output shows, uses symmetric quantization.
- `ExportProgram` quantization reports named parameters `model.l.weight`
  instead of `l.weight`, and left out the warning for a quantized weight
  stored in float16.
- A GPTQ calibration item the model can't run, such as one of the wrong
  width, raised PyTorch's error without saying which item. The error now
  names the item's index and its input shapes. "requires finite values"
  errors now name the parameter.
- `x @ W` failed with "grouped Linear shards must contain four-row groups"
  when a list couldn't hold four of the weight's output rows, as for a
  60,000 x 16 weight, in every storage format. With quantization it also hit
  smaller weights stored in float16, such as a 25,600 x 96 buffer. Such a
  weight is now read through the generic list sharder, kept output-major
  under integer storage.
- Initialization left the last compressed weight bank in a variable, so a
  project saved after `cattorch prepare for save` still stored the
  compressed weights twice. Initialization now clears it when done.
- A quantized weight split across several lists could have its smaller
  lists stored in float16, because `min_quantized_values` was applied to
  each list as well as to the matrix: the last 4,160 values of a
  203,840-value int8 Linear, for example, while the quantization report said
  int8. A quantized weight's storage is now decided once, for the whole
  weight.
- A sprite whose model sums or otherwise reduces a tensor, such as
  `x.sum()`, had a variable named `output` beside the public `output` list,
  which the interface helpers counted as public. The reduction kernel's
  working variable has a private name now.
- Two states sharing one initial tensor, as in `zeros = torch.zeros(...)`;
  `State("hidden", zeros), State("cell", zeros)`, exported a step that read
  one state's list for both and gave wrong logits from the second token on.
  The same happened for any tensor passed as two traced arguments, or an
  argument that is also a parameter or buffer of the model: every argument
  is now traced from its own copy. The reference check of a
  `RecurrentGenerationProgram` now runs the traced step graph rather than the
  model's `step` method, so it checks what is exported.
- `torch.cdist` with its default `p=2` (or `p=2` given explicitly) failed to
  export: the tracer leaves out arguments at their schema default, and the
  decomposition required `p`. A test now checks every decomposition cattorch
  writes against its operator's schema defaults, which also found
  `prims.var`'s `correction`. The OpInfo conformance run also tries later
  samples whose non-tensor arguments differ from the first four, which all
  used `p=1` for `cdist`.
- A marked program whose entry point assigns a buffer (`self.h = ...`)
  failed with "Inplace update to inference tensor outside InferenceMode" when
  exported with `quantization=` and a variable-length entry point, or
  verified with `quantization=`. Calibration runs the entry points under
  inference mode, and the buffers they assigned stayed inference tensors;
  each calibration call now leaves the model's buffers as it found them.
- `nn.MultiheadAttention`, `nn.TransformerEncoderLayer` and
  `nn.TransformerDecoderLayer` with a boolean `attn_mask` or
  `key_padding_mask` held in a buffer failed with an internal
  `KeyError: 'tensor_meta'`. `zeros_like` and `ones_like` of a buffer now
  fold to a constant of the right shape and dtype.
- Generation top-k reported an ID of -1 and a value of `-Infinity`, with
  status ok, when fewer than `top_k` logits were finite. It now always lists
  `top_k` real token IDs, as `torch.topk` does, ranks a NaN logit first, and
  lists equal logits lowest token ID first (documented in
  [generation](docs/generation.md)).
- A `RecurrentGenerationProgram` export ran the step and reference checks on
  the caller's own model, so a model whose methods change its state (an
  embedding with `max_norm`, a buffer the step reassigns) was changed by
  exporting it, even when the export then failed. A variable-length
  (`dynamic=`) export ran `forward` on the caller's model too, in its length
  check and its check that items are independent. Both now run on copies; a
  test checks that every kind of export leaves the caller's parameters,
  buffers, dtypes and training flags as they were, whether it succeeds or
  fails.
- Valid variable-length (`dynamic=`) exports with `quantization=`, float16
  storage or `sig_figs` were rejected at random, depending on the weights,
  with a false "probably uses torch.arange" message: the length check
  allowed ten times the storage error it saw at the shortest length, which
  can be tiny. The check now compares the sprite with a copy of the model
  holding the values the sprite stores, with the same tolerance as an exact
  export, for plain models and entry points alike. Only fast kernels and
  GPTQ, whose effect it can't reproduce, still use the shortest length's
  difference.
- `nn.EmbeddingBag` with a `padding_idx`, varying along the indices of a
  `(1, indices)` input, failed with "aten.sum.dim_IntList mixes values across
  the runtime extent". Traced with one index, the padding mask `[1]` is
  viewed as `[1, 1, 1]`, and the variable-length dimension was taken to be
  the last of those; the shape traced at two indices now tells which one
  grows.
- After rows times a batched weight with `dynamic=` (`[n, 4] @ [3, 4, 5]`,
  giving `[3, n, 5]`), a per-head bias `[3, 1, 5]` or scale `[3, 1, 1]`, a
  softmax over the heads, and sums or maxima over a fixed dimension were
  rejected, with no fallback to running once per item. Fixed operands are
  now repeated to the variable length, and reductions and softmax over a
  fixed dimension run as runtime kernels. Broadcasting a variable-length
  operand there is still rejected, with an error that says why and suggests
  transposing the items to the front.
- A `RecurrentGenerationProgram` with a large `max_tokens` sharded its
  private prefill buffer into `max_tokens / 200,000` lists (22,920 blocks
  instead of 744 at 20 million, and 52 s to export at 100 million), although
  a prompt arrives in the one `cattorch tokens` list, which holds at most
  200,000 items. The buffer is now sized by what can arrive.

## Deprecations

- `transpile(..., frontend="export")` is deprecated and will be removed in
  2.0. It supports a subset of what the default `frontend="fx"` does, without
  its decompositions and pattern recognition. It now warns with a
  `DeprecationWarning`.

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
