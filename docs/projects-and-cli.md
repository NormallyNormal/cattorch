# Projects and the command line

[Documentation home](index.md)

An exported sprite is one piece of a Scratch project: something still has to
fill its input list, broadcast its call, and show the result. `build_project`
writes that project for you, and the `cattorch` command exports a model and
builds its project in one step.

## Build a project

```python
import torch
from torch import nn

import cattorch

model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 3)).eval()
example = torch.tensor([[0.5, -0.25]])

result = cattorch.transpile(model, example, "build/points")
project = cattorch.build_project(
    result, "build/points", example_inputs=example, labels=["red", "green", "blue"],
)
print(project.path)  # build/points.sb3
```

Open the `.sb3` with Scratch's **File > Load from your computer**. It opens
with its small Demo sprite selected, not the model's: a large model sprite
makes the editor slow while it's selected, and you never need to open it.
The stage shows the `points input` and `points output` lists, starting from
`example_inputs` (or zeros). Click the green flag or press space to run the
model. Edit the input list's items to try other inputs. With `labels`, the
`points prediction` monitor shows the label of the largest output.

For a [`GenerationProgram`](generation.md) or
[`RecurrentGenerationProgram`](recurrent-generation.md) export, pass the
exported [tokenizer](tokenizers.md) too:

```python
words = cattorch.CharTokenizer(hf_tokenizer).save("build/tokenizer", name="tokenizer")
project = cattorch.build_project(
    result, "build/story", tokenizer=words, max_new_tokens=40, temperature=0.8,
)
```

The project then has a Demo sprite. Click the green flag and it asks for the
start of some text, continues it one token at a time with the model, and
shows the text so far in a speech bubble. An empty answer stops it. The
temperature slider sets how random the choices are; 0 always picks the
likeliest token. Generation stops after `max_new_tokens`, at `eos_token_id`
when you give one, or when the model's context is full (never, for a recurrent
generator without `max_context`). `prompt` replaces the
question it asks. When the model or the tokenizer fails, such as a prompt
longer than the context or a token ID the tokenizer doesn't have, the Demo
stops, says which call failed with its status (`chars detokenize: invalid
item: chars token ids`), and repeats it before the next question.

The model's vocabulary and the tokenizer's token IDs should match.
`build_project` warns when they don't: a vocabulary padded past the
tokenizer's (50304 for 50257) works until the model picks a padding ID, and a
tokenizer with an ID past the model's, such as an added padding token, works
until the text contains it. Options of another kind of project, such as
`temperature` for a classifier, raise `ValueError`; see
[`build_project`](api-reference.md#build_project).

A model with [entry points](markers.md) gets monitors of its lists and a
note on how to call it, but no demo, since only you know what its calls
mean.

## Projects

An exported sprite already has the interface other sprites use: its
broadcasts (`NAME forward`, ...) and its global lists and variables
(`NAME input`, `NAME status`, ...), NAME being its name; see
[using a sprite without opening it](scratch-interface.md#using-a-sprite-without-opening-it).
A project from `build_project` holds the sprites as they are and declares
their globals on the stage, as uploading them would. Any sprite in the
project can fill the inputs, `broadcast [NAME forward] and wait`, and read
`NAME status` and the outputs.

The project adds a small sprite, Demo, first among the sprites so that the
project opens with it selected. It runs the demo, and holds the project's
own messages, which a project of your own can use too:

| Project | Broadcast | Reads | Writes |
|---|---|---|---|
| Classifier (with `labels`) | `NAME predict`, after `NAME forward` | `NAME output`, `NAME labels` | `NAME prediction` |
| Text generator | `NAME sample`, after a `NAME prefill` or `NAME decode` that ended `ok` | `NAME logits`, `NAME temperature` | `NAME next token` |

Each ends with a `broadcast and wait` of the message followed by
` complete`. `NAME sample` picks the next token, zero-based, from the
softmax of the logits divided by the temperature (0 picks the likeliest
token). The [Scratch interface](scratch-interface.md#projects) lists every
name, and `ProjectResult` every sprite, broadcast, list and variable of a
project. No two of a project's sprites may share a name, nor two of its
globals or broadcasts: export the tokenizer under another name than the
model, for one.

To use the sprites in your own project, upload the `.sprite3` files into it
(select another sprite after each upload), or start from a built project
and add your own sprites.

## The command line

Installing cattorch adds a `cattorch` command.

```bash
cattorch export my_models.py:Classifier --project
cattorch export model.pt --shape 1,28,28 --labels "0,1,2,3,4,5,6,7,8,9" --project
cattorch generate my_models.py:TinyGPT --tokenizer path/to/tokenizer --project
cattorch generate --hf some-user/tiny-model --max-context 64 --project
cattorch generate --hf AntonV/mamba2-130m-hf --quantize 8 --project
```

`cattorch export MODEL` exports a model, checks it against PyTorch with
[`verify`](api-reference.md#verify), and prints a JSON summary. `MODEL` is
one of:

- `file.py:name` or `package.module:name`, naming a module, a class to
  construct without arguments, or a function returning a module.
- A `.pt` file saved with `torch.save(model)`. Loading it unpickles it, which
  can run code, so only load files you trust. The file names the model's
  class by module, so that module must be importable: cattorch also looks in
  the working directory and the `.pt` file's directory. A class defined in
  the script that saved the model can't be found; define it in a module
  instead.

The inputs come from the model's [`@cattorch.example_inputs`](markers.md) marker or
`@entrypoint` methods, or from `--shape 1,28,28` for random float inputs and
`--tokens 1,8` for random token IDs below `--vocab` (default 2; it needs
`--tokens`). Repeat them
for several inputs, floats first. `--dynamic 0=1:32` lets input 0's first dimension vary
from 1 to 32 at run time, and `--dynamic 0.1=1:64` its second; see
[variable-length inputs](supported-models.md#variable-length-inputs).

`cattorch generate` exports a `forward(tokens) -> logits` model as a
[`GenerationProgram`](generation.md) with `--max-context` (default 64) and
`--top-k`. A Hugging Face Mamba, Mamba-2, or RWKV model is recognized and
exported as a [`RecurrentGenerationProgram`](recurrent-generation.md)
instead; there `--max-context` is optional, and without
it the generator takes any number of tokens (the summary's `max_context` is
`null`). `--hf REPO` loads a Hugging Face causal language model and its
tokenizer by name or directory, instead of `MODEL` (giving both is an
error); `--tokenizer` names a Hugging Face tokenizer for a model given as
`MODEL`, or replaces `--hf`'s. The tokenizer is exported too when its type is
[supported](tokenizers.md#compatibility). Generation exports aren't verified.

The options are named after the Python parameters they set. Both commands
take:

| Option | Effect |
|---|---|
| `-o PATH` | Output path, with or without `.sprite3` or `.sb3`. The sprite is `PATH.sprite3` and `generate`'s tokenizer `PATH_tokenizer.sprite3`. Defaults to the model's name, lowercased: `name` in `file.py:name` or `module:name`, the `.pt` file's stem, or the last part of the `--hf` repo. |
| `--name NAME` | The sprite's name in Scratch, `transpile(name=...)`. Defaults to the file's name. |
| `--project` | Also write `PATH.sb3` with `build_project`. |
| `--optimization {exact,fast}` | `fast` uses [approximate kernels](optimization.md). |
| `--precision {float32,float16}` | How stored weights are kept, `StorageConfig(precision=...)`: `float32` (the default) exactly, `float16` at half precision. |
| `--quantize {4,6,8}` | [Quantize](storage.md) matrix weights to that many bits, `QuantizationConfig(bits=...)`. |
| `--quantize-method {symmetric,gptq}` | How `--quantize` quantizes; `symmetric` by default. `gptq` needs `--calibration`. |
| `--calibration FILE` | GPTQ calibration data: a list saved with `torch.save`, of input tensors for `export` or `[1, length]` token tensors for `generate`. |
| `--sig-figs N` | Round stored values to `N` significant figures. |

`export` verifies against a copy of the model holding the values the sprite
stores, so float16, symmetric quantization and `--sig-figs` are checked with
`atol=1e-3, rtol=1e-4`, and an exact float32 export with `atol=1e-4,
rtol=1e-5`. A `fast` export is checked with `atol=0.05, rtol=0.05`, and so
is a GPTQ export, which can't be reproduced and is compared with the model
itself. It also takes `--labels` for its project and
`--no-verify`. `generate` takes `--max-new-tokens` (default 40),
`--temperature` (default 0.8), and `--eos-token-id` (defaulting to the
tokenizer's end token) for its project's demo, as `build_project` does. A
tokenizer's end token past the model's token IDs, such as an added `</s>`,
can never be generated, so the demo then has none, and a warning says so.
These project options, and `--labels`, need `--project`: without it they
are an error rather than ignored. `cattorch analyze-benchmark PROJECT` prints a saved
[benchmark project's](verification-and-benchmarking.md) results as JSON, the
fields of `analyze_benchmark`'s `BenchmarkAnalysis`. `cattorch --version`
prints the installed version, which Python code can read as
`cattorch.__version__`, and `--help` after a command lists its options with
their defaults.

The JSON summary uses the result's field names: `path`, `sprite_name`,
`archive_bytes`, `expanded_json_bytes`, `block_count`, `list_count`,
`sharded_lists`, `warnings`, `fallback_operations`,
`approximate_operations` and `broadcasts` (the messages that run the
sprite, such as `net forward`), and for `generate` also `max_context`,
`vocabulary_size` and `top_k`. `lists` names the global lists a project
uses the sprite through, such as `net input` and `net output`. `export` adds `verify`, with `passed` and
`max_abs_error`, the largest error over every output (and, for a model with
entry points, every call and state; a call that ended with an unexpected
status counts as infinite, written as the string `"Infinity"`);
`--project` adds `project`, with its `path` and `warnings`; `generate` adds `tokenizer`,
with the tokenizer result's fields, or its `error` when it couldn't be
exported.

The commands exit with one of these statuses:

| Status | Meaning |
|---|---|
| `0` | Success. |
| `1` | The export failed: an unsupported model or operation, or a file that can't be read or written. |
| `2` | Invalid arguments, found by the command's parser or by cattorch's checks of them, including a `MODEL`, `--hf` model or `--tokenizer` that doesn't exist or can't be loaded (or needs transformers, which isn't installed), and options that can't apply together. |
| `3` | `export`'s check found that the sprite doesn't match PyTorch. |

## Examples

The repository's [`examples/`](../examples/README.md) directory has four
scripts that train a small model in a few seconds, export it, and build its
project: an MLP that names a point's quadrant, a CNN that tells horizontal
lines from vertical ones, and a character-level transformer and GRU that
continue a nursery rhyme.

Related: [getting started](getting-started.md), [API reference](api-reference.md#build_project).
