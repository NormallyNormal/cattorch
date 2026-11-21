# Glossary

[Documentation home](index.md)

## cattorch terms

- **base92, base92-huffman-int4**: values of the
  `cattorch storage compression` variable. `base92` stores weights as text
  in a 92-character alphabet; `base92-huffman-int4` also packs 4-bit weights
  with a Huffman code; `none` keeps one list item per weight. See
  [every sprite](scratch-interface.md#every-sprite) and
  [preserve processor costumes](storage.md#preserve-processor-costumes).
- **Call**: running one of a sprite's operations, by broadcasting its
  message, `NAME OP` (such as `NAME forward`), and waiting. See
  [using a sprite without opening it](scratch-interface.md#using-a-sprite-without-opening-it).
- **Calibration data**: realistic inputs, passed as `calibration=`, that
  GPTQ uses to choose quantized weights. They don't affect the sprite's
  input shapes. See [GPTQ calibration](storage.md#gptq-calibration).
- **Completion broadcast**: the `NAME OP complete` message a call sends with
  `broadcast and wait` when it ends, whatever the outcome. See
  [how every call behaves](scratch-interface.md#how-every-call-behaves).
- **Entry point**: a model method exported as its own call,
  `NAME ENTRY`, in a program. See
  [programs](programs-and-moe.md#named-entry-points-and-state).
- **Exact mode**: the default kernel mode, which matches PyTorch's results
  up to Scratch's floating-point behavior. Lossy storage or `sig_figs` can
  still change the output. See [optimization modes](optimization.md).
- **Example inputs**: the tensor or tuple passed to `transpile`. Their shapes
  and dtypes become the sprite's input shapes, fixed unless an input is
  [variable-length](supported-models.md#variable-length-inputs).
- **Expanded JSON**: the uncompressed `sprite.json` or `project.json` inside a
  Scratch file. Scratch's size limits apply to it, not to the compressed
  file. See [project size](storage.md#project-size).
- **Export kinds**: what a sprite was exported from, which sets its calls: a
  model (`transpile` with example inputs), a program (`ExportProgram` or
  `@entrypoint` markers), a generator (`GenerationProgram`), a recurrent
  generator (`RecurrentGenerationProgram`), or a tokenizer
  (`transpile_tokenizer`). See the [Scratch interface](scratch-interface.md).
- **Fast mode**: a kernel mode that allows approximations, plus optional
  pruning and low-rank weight transforms. See
  [optimization modes](optimization.md).
- **Head bits**: `QuantizationConfig.head_bits`, the precision of a
  generator's output head, int8 by default when `bits` is 4 or 6. See
  [the output head](storage.md#the-output-head-of-a-language-model).
- **Interface version**: the sprite's `cattorch interface version` variable,
  `1` for cattorch 1.x sprites. See the [Scratch interface](scratch-interface.md).
- **K/V cache**: attention keys and values saved from earlier tokens, so
  generation doesn't recompute the whole sequence for each new token. See
  [KV-cached generation](generation.md).
- **One-token step**: a model method that reads one token and the recurrent
  state and returns the next token's logits and the new state; what a
  `RecurrentGenerationProgram` exports. See
  [the step](recurrent-generation.md#the-step).
- **Override**: a `QuantizationConfig.overrides` entry that gives some
  weights other quantization settings. See
  [per-weight settings](storage.md#per-weight-settings).
- **NAME**: an export's name, the `name` given to `transpile` (by default
  the file name). The sprite is named after it, and its globals and
  messages start with it. See
  [using a sprite without opening it](scratch-interface.md#using-a-sprite-without-opening-it).
- **Prefill, decode**: a generator's calls. `NAME prefill` reads a
  whole prompt; `NAME decode` reads one more token. See
  [KV-cached generation](generation.md#scratch-interface).
- **Processor sprite**: a generated model or tokenizer sprite, used for
  computation rather than display.
- **Pytree leaf order**: the order PyTorch uses to flatten nested tuples,
  lists, and dictionaries. cattorch uses it to number output lists. See
  [inputs and outputs](supported-models.md#inputs-and-outputs).
- **Recurrent state**: values a recurrent model, such as a GRU, Mamba, or
  RWKV, carries from one token to the next. Unlike a K/V cache, it doesn't
  grow with the number of tokens. See
  [states](recurrent-generation.md#states).
- **Runtime extent**: the one dimension of a tensor whose length can change
  while the sprite runs: the rows of an append state. See
  [state](programs-and-moe.md#state).
- **Shard**: one of several Scratch lists holding a tensor too large for a
  single list of 200,000 items. See
  [list sharding](storage.md#list-sharding).
- **State**: a tensor a sprite keeps between calls. A replace state has a
  fixed shape; an append state grows up to its capacity. See
  [state](programs-and-moe.md#state).
- **Static tensor**: a parameter, buffer, or constant stored in the sprite.
- **Restarted**: what happens to a call when its message is broadcast again
  while it runs: Scratch stops it and starts the call again. See
  [overlapping calls](scratch-interface.md#overlapping-calls).
- **Status**: the global `NAME status` variable: `not finished` while a call
  runs, then how it ended. See [status values](scratch-interface.md#status-values).
- **Taken over**: what happens to a call when another call of the same
  sprite starts while it is paused. See
  [overlapping calls](scratch-interface.md#overlapping-calls).

## Scratch terms

- **My Block / custom block**: a procedure that belongs to one Scratch sprite.
  Other sprites can't run it, so an exported sprite's blocks are internal
  and its calls are broadcasts.
- **Run without screen refresh / warp**: a custom block option that runs the
  block without pausing to redraw the screen, for up to about 500 ms at a
  time. cattorch turns it on for generated blocks. See
  [overlapping calls](scratch-interface.md#overlapping-calls).
- **Local data**: a variable or list that belongs to one sprite. Other sprites
  can't change it; they can read a variable with the `([VAR] of [SPRITE])`
  sensing block.
- **Global data**: a variable or list available to every sprite in a
  project, kept on the stage. Uploading a sprite creates the globals it uses
  but doesn't define.
- **Turbo Mode**: Scratch's faster execution mode, turned on by shift-clicking
  the green flag. Recommended when running generated models.

Related: [getting started](getting-started.md),
[Scratch interface](scratch-interface.md), and
[storage and quantization](storage.md).
