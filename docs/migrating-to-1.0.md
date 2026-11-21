# Migrating from 0.5 to 1.0

[Documentation home](index.md)

cattorch 1.0 freezes the Python API, the command line and the Scratch
interface (see [stability](stability.md)). To get there, some names and
behaviours from 0.5.0 changed. This page lists every change that can break
0.5.0 code or a Scratch project built around a 0.5.0 sprite, in the order you
meet them, and what to write instead. New features are in the
[changelog](../CHANGELOG.md).

The examples below share this setup:

```python
# Setup shared by the examples on this page.
import torch
from torch import nn

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).eval()
example = torch.randn(1, 4)


class Decoder(nn.Module):
    """A tiny causal language model: forward(tokens) -> logits."""

    def __init__(self, vocabulary=16, width=8, context=8):
        super().__init__()
        self.token = nn.Embedding(vocabulary, width)
        self.position = nn.Embedding(context, width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.head = nn.Linear(width, vocabulary, bias=False)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context, context, dtype=torch.bool), 1),
        )

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.token(tokens) + self.position(torch.arange(length))
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scores = (q @ k.transpose(-2, -1)).masked_fill(
            self.mask[:length, :length], float("-inf"),
        )
        return self.head(x + torch.softmax(scores, dim=-1) @ v)


decoder = Decoder().eval()
```

## Requirements

| | 0.5.0 | 1.0 |
|---|---|---|
| Python | 3.10 or newer | 3.11 or newer |
| PyTorch | 2.6 or newer | 2.10 or newer |
| NumPy | not required | 1.24 or newer, installed with cattorch |

## Python API

### Imports

Programs, adapters and their classes are stable in 1.0 and imported from
`cattorch`. Importing them from `cattorch.experimental` still works until 2.0,
with a `DeprecationWarning`:

```python
# cattorch 0.5.0
from cattorch.experimental import (
    EntryPoint, ExportProgram, Input, ModuleAdapter, Output, ProgramCall, State,
    StateInput, StateUpdate,
)
```

```python
from cattorch import (
    EntryPoint, ExportProgram, Input, ModuleAdapter, Output, ProgramCall, State,
    StateInput, StateUpdate,
)
```

The same goes for `AdapterContext` and `GenerationProgram`. The explicit MoE
classes (`ExpertFamily`, `SparseMoE`, `StackedSwiGLUMoE`, `RoutingScores`,
`select_routes`) and `analyze` stay in `cattorch.experimental`.
`cattorch.experimental.stacked_swiglu_moe_adapter` is removed: a sparse MoE
block with SwiGLU experts uses `StackedSwiGLUMoE` by itself, and for experts
whose weights are stacked into one tensor each, an
[`@adapter`](markers.md#classes-you-dont-own) that returns a
`StackedSwiGLUMoE` does the same. `analyze`'s second parameter is now called
`example_inputs` instead of `program`, matching `transpile`.

### `transpile` arguments

Everything after `model, example_inputs, output_path` is keyword-only, so
`sig_figs` can't be passed by position. `frontend` is removed: every export is
traced with `torch.fx`, and `frontend="export"` has no replacement. The three
calibration parameters are one, `calibration`, whose items take the form the
export is called with.

| 0.5.0 | 1.0 |
|---|---|
| `transpile(model, example, path, 3)` | `transpile(model, example, path, sig_figs=3)` |
| `transpile(..., frontend="fx")` | Leave it out. |
| `transpile(..., frontend="export")` | Leave it out; FX tracing supports more models. |
| `calibration_inputs=batches` | `calibration=batches` (tensors or tuples of tensors) |
| `calibration_calls=calls` (an `ExportProgram`) | `calibration=calls` (`ProgramCall`s) |
| `calibration_sequences=sequences` (a `GenerationProgram`) | `calibration=sequences` (`[1, length]` token tensors) |

```python
# cattorch 0.5.0
from cattorch import QuantizationConfig, transpile

batches = [torch.randn(1, 4) for _ in range(8)]
transpile(model, example, "model", 3,
          quantization=QuantizationConfig(bits=8, method="gptq", min_calibration_rows=1),
          calibration_inputs=batches)
```

```python
from cattorch import QuantizationConfig, transpile

batches = [torch.randn(1, 4) for _ in range(8)]
transpile(model, example, "model", sig_figs=3,
          quantization=QuantizationConfig(bits=8, method="gptq", min_calibration_rows=1),
          calibration=batches)
```

GPTQ without `calibration`, or `calibration` without GPTQ, raises
`ValueError`.

### Configuration classes

| 0.5.0 | 1.0 |
|---|---|
| `StorageConfig(precision="int8")` (also `"int6"`, `"int4"`), deprecated in 0.5.0 | `transpile(..., quantization=QuantizationConfig(bits=8))`. `StorageConfig` takes only `compression` and `precision` (`"float32"` or `"float16"`); its `group_size`, `min_quantized_values`, `scale_precision` and `grouping` fields are removed. |
| `QuantizationConfig(..., fallback="symmetric")` | Leave it out; symmetric was its only value and is what happens. |
| `FastConfig(weights=...)` | `FastConfig(layers=...)` |
| `FastLayerConfig(pruning=0.5)` | `FastLayerConfig(block_pruning=0.5)` |
| `CodegenConfig(layer_sharing="auto")` | `CodegenConfig(layer_sharing=True)` |
| `CodegenConfig(layer_sharing="off")` | `CodegenConfig(layer_sharing=False)`, the default |

`QuantizationConfig` groups weights by row and uses float16 scales by
default, where the deprecated integer `StorageConfig` used flat groups and
float32 scales, so the quantized values differ slightly.

```python
# cattorch 0.5.0
from cattorch import CodegenConfig, FastConfig, FastLayerConfig, StorageConfig, transpile

transpile(model, example, "model", optimization="fast",
          fast_config=FastConfig(weights=FastLayerConfig(pruning=0.25)),
          storage=StorageConfig(precision="int8"),
          codegen=CodegenConfig(layer_sharing="auto"))
```

```python
from cattorch import (
    CodegenConfig, FastConfig, FastLayerConfig, QuantizationConfig, transpile,
)

transpile(model, example, "model", optimization="fast",
          fast_config=FastConfig(layers=FastLayerConfig(block_pruning=0.25)),
          quantization=QuantizationConfig(bits=8),
          codegen=CodegenConfig(layer_sharing=True))
```

`FastConfig(neuron_pruning=...)` needs a model `torch.fx` can trace
symbolically. One it can't, such as any `GenerationProgram` export, used to
skip the pruning with a log message; it now raises `UnsupportedModelError`.

### `GenerationProgram`

`GenerationProgram` takes `max_context` first, and it is the only positional
field. `example_token` is removed, and `method` is optional: it defaults to
the method marked [`@generate`](markers.md#generation-methods), or `forward`.

The method is now a stateless `forward(tokens) -> logits` over the whole
sequence, which cattorch traces at `max_context` tokens and turns into a
cached one-token step itself, checking the step against the method at
export. It must take a `[1, length]` tensor of token IDs and return logits
for every position. A 0.5.0 method written to take one token, such as one
that reads its position from a buffer instead of from the input's length,
needs to become an ordinary forward over the sequence. See
[KV-cached generation](generation.md#requirements) for what the model may do.

```python
# cattorch 0.5.0
from cattorch import GenerationProgram, transpile

result = transpile(
    decoder,
    GenerationProgram(method="forward", example_token=torch.tensor([[0]]),
                      max_context=8, top_k=3),
    "decoder",
)
logits = result.output
```

```python
from cattorch import GenerationProgram, transpile

result = transpile(decoder, GenerationProgram(max_context=8, top_k=3), "decoder")
logits = result.outputs[0]
assert logits.list_name == "decoder logits"
assert result.vocabulary_size == 16 and result.max_context == 8
```

A generation export returns a `GenerationResult` instead of a
`TranspileResult`. Its `inputs` describes the global `NAME token ids` list
(NAME is the export's name, here `decoder`) with a `BoundedDimension` of 1
to `max_context` tokens (it was a `(1, 1)` list named `cattorch tokens`),
and `outputs` holds `NAME logits` with shape
`(vocabulary,)` (it was `(1, 1, vocabulary)`) followed by the top-k lists.
It also has `max_context`, `vocabulary_size` and `top_k`.

With `QuantizationConfig(bits=4)` or `bits=6`, a generation export now stores
the output head, the weight that computes the logits (in GPT-2, the tied
embedding table), in int8. The sprite is larger and its logits much closer
to the float model's. Pass
`QuantizationConfig(bits=4, head_bits=4)` for the 0.5.0 behaviour; see
[the output head](storage.md#the-output-head-of-a-language-model).

### Results

| 0.5.0 | 1.0 |
|---|---|
| `result.output`, `result.additional_outputs` | `result.outputs`, every output in order. `result.output` still works and is `result.outputs[0]`. |
| `TokenizerResult.tokenizer_type == "CharTokenizer"` | `"char"` |
| `TokenizerResult.tokenizer_type == "BPETokenizer"` | `"bpe"` |
| `TokenizerResult.tokenizer_type == "SentencePieceBPETokenizer"` | `"sentencepiece-bpe"` |
| `ProgramCallVerifyResult.states`, a tuple of `(name, VerifyResult)` pairs | A mapping from state name to `VerifyResult`. |
| `procedures` of `TranspileResult` and `ProgramResult`: the sprite's public blocks (`cattorch forward`, ...) | `broadcasts`: the messages that run it (`NAME forward`, ...); see [Scratch projects](#scratch-projects). |
| `EntryPointResult.procedure` (`cattorch ENTRY`) | `EntryPointResult.broadcast` (`NAME ENTRY`); `completion_broadcast` is `NAME ENTRY complete`. |
| `TensorSpec.list_name`, `StateResult.list_name` and `sharded_lists`: the sprite's own lists (`output`, `cattorch step value`, `cattorch state counter`) | The global lists a project uses (`NAME output`, `NAME step value`, `NAME state counter`). |

```python
# cattorch 0.5.0
from cattorch import transpile

result = transpile(model, example, "model")
specs = (result.output, *result.additional_outputs)
```

```python
from cattorch import transpile

result = transpile(model, example, "model")
specs = result.outputs
```

```python
# cattorch 0.5.0
from cattorch import transpile, verify
from cattorch.experimental import (
    EntryPoint, ExportProgram, Input, Output, ProgramCall, State, StateInput, StateUpdate,
)


class Counter(nn.Module):
    def step(self, value, state):
        updated = state + value
        return updated, updated


counter_program = ExportProgram(
    states=(State("counter", torch.zeros(4)),),
    entrypoints=(EntryPoint(
        "step", method="step",
        arguments=(Input("value", torch.ones(4)), StateInput("counter")),
        returns=(Output("result"), StateUpdate("counter")),
    ),),
)
counter = Counter().eval()
sprite = transpile(counter, counter_program, "counter")
report = verify(counter, counter_program, sprite,
                calls=(ProgramCall("step", (torch.ones(4),)),))
counter_check = dict(report.calls[0].states)["counter"]
```

```python
from cattorch import (
    EntryPoint, ExportProgram, Input, Output, ProgramCall, State, StateInput, StateUpdate,
    transpile, verify,
)


class Counter(nn.Module):
    def step(self, value, state):
        updated = state + value
        return updated, updated


counter_program = ExportProgram(
    states=(State("counter", torch.zeros(4)),),
    entrypoints=(EntryPoint(
        "step", method="step",
        arguments=(Input("value", torch.ones(4)), StateInput("counter")),
        returns=(Output("result"), StateUpdate("counter")),
    ),),
)
counter = Counter().eval()
sprite = transpile(counter, counter_program, "counter")
report = verify(counter, counter_program, sprite,
                calls=(ProgramCall("step", (torch.ones(4),)),))
counter_check = report.calls[0].states["counter"]
```

### Tokenizers

Tokenizers that a sprite would tokenize differently from the original are
rejected with `ValueError` instead of exported:

- `BPETokenizer` rejects a normalizer, a pre-tokenizer, `fuse_unk` and
  `byte_fallback`.
- `CharTokenizer` rejects a tokenizer with multi-character tokens or a
  normalizer, or one that doesn't encode text one character at a time.
- `SentencePieceBPETokenizer` rejects SentencePiece models that aren't BPE
  (`unigram`, `word`, `char`) and models with user-defined symbols.
- `transpile_tokenizer` rejects WordPiece, Unigram and word-level tokenizers
  with multi-character tokens, and `scratch_casefold=True` for a tokenizer
  that isn't a SentencePiece model, which 0.5.0 ignored.
- An object that isn't a tokenizer raises `TypeError` instead of
  `AttributeError`.

See [tokenizers](tokenizers.md#compatibility).

### Benchmarks

| 0.5.0 | 1.0 |
|---|---|
| `iterations=` on `build_benchmark_suite`, `build_generation_benchmark_suite` and `build_storage_benchmark_suite` | `repeats=` |
| `build_generation_benchmark_suite` cases `(name, model, stateless_input, prompt)` | `(name, model, program, prompt)`, where `program` is a `GenerationProgram` or `RecurrentGenerationProgram` whose `max_context` holds the prompt plus `repeats` decodes |
| `analyze_benchmark(path)` returns a dict | It returns a `BenchmarkAnalysis`, whose `cases` hold a `BenchmarkCase` each. |
| A suite case's `exact_seconds` and `fast_seconds`: the time of all iterations together | `BenchmarkCase.baseline_seconds` and `candidate_seconds`: the time of one call. |
| `cattorch-benchmark PROJECT` | `cattorch analyze-benchmark PROJECT` |
| Stage variable `cattorch benchmark iterations` (`cattorch storage benchmark iterations` in storage suites) | `cattorch benchmark repeats` |

`analyze_benchmark` still reads projects built by 0.5.0.

```python
# cattorch 0.5.0
from cattorch import build_benchmark_suite, build_generation_benchmark_suite

build_benchmark_suite([("mlp", model, example)], "suite", iterations=100)
build_generation_benchmark_suite(
    [("decoder", decoder, torch.tensor([[1, 2, 3, 4]]), torch.tensor([[1, 2, 3, 4]]))],
    "generation_suite", iterations=4,
)
```

```python
from cattorch import (
    GenerationProgram, build_benchmark_suite, build_generation_benchmark_suite,
)

build_benchmark_suite([("mlp", model, example)], "suite", repeats=100)
build_generation_benchmark_suite(
    [("decoder", decoder, GenerationProgram(max_context=8), torch.tensor([[1, 2, 3, 4]]))],
    "generation_suite", repeats=4,
)
```

## Scratch projects

A sprite keeps the interface of the cattorch version that exported it, and a
0.5.0 sprite doesn't become a 1.0 one by itself. To use the names and
behaviour below, export the model again with 1.0 and replace the sprite in
your project, then update the scripts that use it. A 1.0 sprite has a
`cattorch interface version` variable, `1`; a sprite without it is from
before 1.0. The full contract is in the [Scratch interface](scratch-interface.md).

### Calls are broadcasts, data is global

In 0.5.0, a sprite's blocks (`cattorch forward`, ...) were custom blocks
and its lists (`input`, `output`, ...) were its own, so only scripts inside
the sprite could use them, and you had to open the sprite to add them. In
1.0 you never open the sprite: a large one makes the editor slow while it's
selected. Each export has a NAME, the `name` you pass (by default the file
name), which is also the sprite's name:

- Each call is a broadcast, `NAME forward`, that a `when I receive` script
  in the sprite answers. Call it from any sprite with
  `broadcast [NAME forward] and wait`, then read `NAME status`. The blocks
  are internal (`_cattorch forward`); scripts must not call them.
- What a project reads or writes is global and named after the export.
  Uploading the sprite creates the globals on the stage, or uses the ones
  of the same name the project has. Constants stay in the sprite; read them
  with `([cattorch max context] of [NAME])`.
- Uploading a sprite selects it: select another sprite right away.
- Two sprites exported under one NAME share their globals and both answer
  the same broadcasts. To use two copies of one model, export them with
  different `name`s.

```text
// cattorch 0.5.0, a script inside the model sprite
delete all of [input]
add (0.5) to [input]
cattorch forward
say (item (1) of [output])

// cattorch 1.0, a script in any sprite (NAME is "points")
delete all of [points input]
add (0.5) to [points input]
broadcast [points forward] and wait
if <(points status) = [ok]> then
  say (item (1) of [points output])
end
```

| Sprite | 0.5.0 | 1.0 |
|---|---|---|
| Model | `cattorch forward`, `cattorch init`, `cattorch prepare for save` blocks | `NAME forward`, `NAME init`, `NAME prepare for save` broadcasts |
| Model | `input`, `input_1`, ..., `output`, ... lists | `NAME input`, `NAME input_1`, ..., `NAME output`, ... |
| Program | `cattorch ENTRY`, `cattorch init`, `cattorch reset`, `cattorch prepare for save` blocks | `NAME ENTRY`, `NAME init`, `NAME reset`, `NAME prepare for save` broadcasts |
| Program | `cattorch ENTRY INPUT`, `cattorch ENTRY OUTPUT`, `cattorch state STATE` lists | `NAME ENTRY INPUT`, `NAME ENTRY OUTPUT`, `NAME state STATE` |
| Program | `cattorch ENTRY complete` broadcast | `NAME ENTRY complete` |
| Generation | `cattorch init`, `cattorch reset`, `cattorch prefill`, `cattorch decode` blocks | `NAME init`, `NAME reset`, `NAME prefill`, `NAME decode` broadcasts, and the new `NAME prepare for save` |
| Generation | `cattorch tokens`, `cattorch logits`, `cattorch top k values`, `cattorch top k ids` lists | `NAME token ids`, `NAME logits`, `NAME top k values`, `NAME top k ids` |
| Generation | `cattorch cache length` variable | `NAME cache length` |
| Generation | `cattorch ... complete` broadcasts | `NAME ... complete` |
| Tokenizer | `cattorch tokenize`, `cattorch detokenize` blocks | `NAME tokenize`, `NAME detokenize` broadcasts |
| Tokenizer | `input` variable (text to tokenize) and `output` variable (detokenized text) | `NAME text`, for both |
| Tokenizer | `token_ids` list | `NAME token ids` |
| Program, generation | `cattorch status` variable, starting as `0` | `NAME status` (Scratch creates it holding `0`) |
| Every list over 200,000 items | `LIST shard 2`, ... | `NAME LIST shard 2`, ...; the shard counts stay in the sprite as `cattorch LIST shard count` |
| Model, program, generation | `cattorch storage compression` value `costume-base92` | `base92` |
| Model, program, generation | `cattorch storage compression` value `costume-base92-huffman-int4` | `base92-huffman-int4` |

A program's state lists are empty when the sprite is uploaded, until its
first call or `NAME init` fills them with their initial values; a project
you save keeps them. Every model, generation and program sprite also
answers `cattorch prepare for save`, so one `broadcast and wait` of it
prepares a whole project for saving.

Every internal name now starts with `_`. In 0.5.0, some internal blocks,
lists and variables started with `cattorch ` or had no prefix, such as the
weight decoders (`cattorch decode float32`, `cattorch unpack weights`), the
generation caches (`cattorch K cache 1`), `cattorch select top k`,
`cattorch project output`, `cattorch reset 0`, `T1`, `index` and `sum`, and
a tokenizer's `tokens`, `merge_ids` and `splitlist`. Scripts must not use
them.

### Status values

A call sets `NAME status` to one value per situation, naming the global
list at fault. LIST is that list, such as `NAME step value`, and TOK is
`NAME token ids`:

| 0.5.0 | 1.0 |
|---|---|
| `invalid length: LIST` | `wrong number of items: LIST` |
| `invalid dynamic length: LIST` | `wrong number of items: LIST` |
| `dynamic capacity exceeded: LIST` | `too many items: LIST` |
| `state capacity exceeded: STATE` | `state full: NAME state STATE` |
| `decode requires exactly one token` | `wrong number of items: TOK` |
| `prefill requires at least one token` | `too few items: TOK` |
| `prefill exceeds maximum context` | `too many items: TOK` |
| `maximum context exceeded` | `context full` |

`ok` still means success. New values cover a call that hasn't finished
(`not finished`), items that aren't numbers or whole numbers in range
(`invalid item: LIST`) and calls from several scripts at once; see
[status values](scratch-interface.md#status-values).

### Completion broadcasts and failed calls

Every call now sets `NAME status` and does a `broadcast and wait` of its
`NAME OP complete` broadcast whenever it ends, whether it succeeded or
not. Only a call that is stopped, or restarted by a second broadcast of its
message, sends none. In 0.5.0, a program entry point broadcast
`cattorch ENTRY complete` only after a successful call, so a script that
reacts to that broadcast must now check that `NAME status` is `ok` before
reading the outputs. Better, call with `broadcast [NAME ENTRY] and wait`
and read `NAME status` right after.

Model and tokenizer sprites had no status or broadcasts in 0.5.0. They now
have `NAME status`, and model sprites send `NAME init complete`,
`NAME forward complete` and `NAME prepare for save complete`; tokenizer
sprites `NAME tokenize complete` and `NAME detokenize complete`.

A call that fails now changes nothing:

- A model's forward checks its input lists. With the wrong number of items,
  or an item that isn't a number, it used to compute and write meaningless
  outputs; now it writes no outputs and sets `NAME status`, as program
  sprites do.
- A generation sprite's prefill or decode with invalid input used to empty
  the logits and the top-k lists, and a failed prefill reset the cache
  first. Now both leave the logits and the cache as they were. A token ID
  outside the vocabulary, a fraction or text is rejected with
  `invalid item: NAME token ids` instead of reporting `ok`.
- Prefill no longer empties `NAME token ids`. No call changes its input
  lists.

Prepare for save keeps the global lists: in 0.5.0 it emptied a model's input
and output lists, and a program's input and state lists, so a program
needed `cattorch reset` before its next call. Generation sprites now have
`NAME prepare for save` too; it also resets the sprite.

Calls from several scripts at once, which 0.5.0 didn't detect, now fail with
`another call ran during this one` or
`an earlier call was stopped or is still running; reset first`. Call a
sprite from one script at a time; see
[overlapping calls](scratch-interface.md#overlapping-calls).

## Command line

0.5.0 had one command, `cattorch-benchmark PROJECT`. It is now
`cattorch analyze-benchmark PROJECT`, and it prints the fields of
`BenchmarkAnalysis` as JSON. The new `cattorch export` and `cattorch generate`
commands are described in
[projects and the command line](projects-and-cli.md#the-command-line).

Related: [stability](stability.md), [Scratch interface](scratch-interface.md),
[API reference](api-reference.md).
