# Glossary

[Documentation home](index.md)

## cattorch terms

- **Artifact**: a generated `.sprite3` sprite or `.sb3` benchmark project.
- **Calibration inputs**: realistic model inputs that GPTQ uses to choose
  quantized weights. They don't affect the sprite's input shapes.
- **Entrypoint**: a named model method exported as a custom block in an
  `ExportProgram`.
- **Example inputs**: the tensor or tuple passed to `transpile`. Their shapes
  and dtypes become the sprite's fixed input shapes.
- **Exact mode**: the default kernel mode, which matches PyTorch's results
  up to Scratch's floating-point behavior.
  Lossy storage or `sig_figs` can still change the output.
- **Expanded JSON**: the uncompressed `sprite.json` or `project.json` inside a
  Scratch file. Scratch's size limits apply to it, not to the compressed file.
- **Fast mode**: a kernel mode that allows approximations, plus optional
  pruning and low-rank weight transforms.
- **KV cache**: attention keys and values saved from earlier tokens, so
  generation doesn't recompute the whole sequence for each new token.
- **Processor sprite**: a generated model or tokenizer sprite, used for
  computation rather than display.
- **Pytree leaf order**: the order PyTorch uses to flatten nested tuples,
  lists, and dictionaries. cattorch uses it to number output lists.
- **Runtime extent**: the one dimension of a tensor whose length can change
  while the sprite runs, such as the row count of an append state.
- **Static tensor**: a parameter, buffer, or constant stored in the sprite.

## Scratch terms

- **My Block / custom block**: a procedure that belongs to one Scratch sprite.
  cattorch's blocks take no arguments; they read and write lists and variables.
- **Run without screen refresh / warp**: a custom block option that runs the
  whole block without pausing to redraw the screen. cattorch turns it on for
  generated blocks.
- **Local data**: a variable or list that belongs to one sprite. Other sprites
  can't read it, even through a variable with the same name.
- **Global data**: a variable or list available to every sprite in a project.
- **Shard**: one of several Scratch lists holding a tensor too large for a
  single list. cattorch shards any list longer than 200,000 items.
- **Turbo Mode**: Scratch's faster execution mode, turned on by shift-clicking
  the green flag. Recommended when running generated models.

Related: [getting started](getting-started.md),
[storage and quantization](storage.md), and
[verification and benchmarking](verification-and-benchmarking.md).
