# Projects and the command line

[Documentation home](index.md)

An exported sprite is one piece of a Scratch project: something still has to
fill its input list, call its blocks, and show the result. `build_project`
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
    "build/points", result, example_inputs=example, labels=["red", "green", "blue"],
)
print(project.path)  # build/points.sb3
```

Open the `.sb3` with Scratch's **File > Load from your computer**. The stage
shows the `points input` and `points output` lists, starting from
`example_inputs` (or zeros). Click the green flag or press space to run the
model. Edit the input list's items to try other inputs. With `labels`, the
`points prediction` monitor shows the label of the largest output.

For a [`GenerationProgram`](generation.md) export, pass the exported
[tokenizer](tokenizers.md) too:

```python
words = cattorch.CharTokenizer(hf_tokenizer).save("build/tokenizer", name="tokenizer")
project = cattorch.build_project(
    "build/story", result, tokenizer=words, max_new_tokens=40, temperature=0.8,
)
```

The project then has a Demo sprite. Click the green flag and it asks for the
start of some text, continues it one token at a time with the model, and
shows the text so far in a speech bubble. An empty answer stops it. The
temperature slider sets how random the choices are; 0 always picks the
likeliest token. Generation stops after `max_new_tokens`, at `eos_token_id`
when you give one, or when the model's context is full. `prompt` replaces the
question it asks.

A model with [entry points](markers.md) gets a bridge for each one but no
demo, since only you know what its calls mean.

## Bridges

Every operation in the project is bridged to global lists and variables, so
any sprite can use the model with `broadcast and wait`. A bridge copies the
global inputs into the sprite's own lists, runs the operation, and copies the
results back, inside a block that runs without screen refresh. The names
start with the sprite's name:

| Model | Broadcast | Reads | Writes |
|---|---|---|---|
| Plain model | `NAME run` | `NAME input`, `NAME input_1`, ... | `NAME output`, ... |
| Classifier (with `labels`) | `NAME predict`, after `NAME run` | `NAME output` | `NAME prediction` |
| Entry point | `NAME ENTRY` | `NAME ` + each input list | `NAME ` + each output list |
| Text generator | `NAME prefill`, `NAME decode` | `NAME tokens`, `NAME temperature` | `NAME next token`, `NAME status` |
| Tokenizer | `NAME tokenize` | `NAME text` | `NAME token ids` |
| Tokenizer | `NAME detokenize` | `NAME token ids` | `NAME text` |

`NAME prefill` takes a whole prompt and `NAME decode` one token, as the
[generation blocks](generation.md#scratch-interface) do. Both sample the next
token from the logits and write it to `NAME next token`, zero-based. `NAME
status` is `ok` or an error message. `ProjectResult` lists every message, list,
and variable a project defines.

To use the bridges in your own project, build one with `build_project`, then
drag its sprites into your project, or start from it and add your own sprites.
In the editor, the bridges are the `cattorch bridge ...` blocks on each sprite.

## The command line

Installing cattorch adds a `cattorch` command.

```bash
cattorch export my_models.py:Classifier --project
cattorch export model.pt --shape 1,28,28 --labels "0,1,2,3,4,5,6,7,8,9" --project
cattorch generate my_models.py:TinyGPT --tokenizer path/to/tokenizer --project
cattorch generate --hf some-user/tiny-model --max-context 64 --project
```

`cattorch export MODEL` exports a model, checks it against PyTorch with
[`verify`](api-reference.md#verify), and prints a JSON summary. It exits with
status 1 if the check fails. `MODEL` is one of:

- `file.py:name` or `package.module:name`, naming a module, a class to
  construct without arguments, or a function returning a module.
- A `.pt` file saved with `torch.save(model)`. Loading it unpickles it, which
  can run code, so only load files you trust.

The inputs come from the model's [`@cattorch.example`](markers.md) marker or
`@entrypoint` methods, or from `--shape 1,28,28` for random float inputs and
`--tokens 1,8` (with `--vocab`) for token IDs. Repeat them for several
inputs, floats first. `--dynamic 0=1:32` lets input 0's first dimension vary
from 1 to 32 at run time, and `--dynamic 0.1=1:64` its second; see
[variable-length inputs](supported-models.md#variable-length-inputs).

`cattorch generate` exports a `forward(tokens) -> logits` model as a
[`GenerationProgram`](generation.md) with `--max-context` (default 64) and
`--top-k`. `--hf REPO` loads a Hugging Face causal language model and its
tokenizer by name or directory; `--tokenizer` names a Hugging Face tokenizer
for a model given as `MODEL`. The tokenizer is exported too when its type is
[supported](tokenizers.md#compatibility). Generation exports aren't verified.

Both commands take:

| Option | Effect |
|---|---|
| `-o PATH` | Output path without an extension; defaults to the model's name. |
| `--project` | Also write `PATH.sb3` with `build_project`. |
| `--fast` | Use [approximate kernels](optimization.md). |
| `--precision` | `float16` stores weights at half precision, and `int8` or `int4` [quantize](storage.md) matrices. Lossy exports are verified to within 0.05. |

`export` also takes `--labels` for its project and `--no-verify`, and
`generate` takes `--max-new-tokens`, `--temperature`, and `--eos` (defaulting
to the tokenizer's end token). `cattorch analyze-benchmark PROJECT` prints a
saved [benchmark project's](verification-and-benchmarking.md) timings, like
`cattorch-benchmark`.

## Examples

The repository's [`examples/`](../examples/README.md) directory has three
scripts that train a small model in a few seconds, export it, and build its
project: an MLP that names a point's quadrant, a CNN that tells horizontal
lines from vertical ones, and a character-level transformer that continues
a nursery rhyme.

Related: [getting started](getting-started.md), [API reference](api-reference.md#buildproject).
