# Recurrent generation

[Documentation home](index.md)

Some language models carry the past in a fixed-size state instead of a cache
of attention keys and values: GRU and LSTM language models, Mamba, Mamba-2 and
RWKV. A `RecurrentGenerationProgram` exports such a model as a sprite that
reads one token at a time, updates the state, and writes the next token's
logits. The sprite has the same blocks and lists as a
[KV-cached generator](generation.md), so a Scratch project or
[`build_project`](projects-and-cli.md) demo written for one drives the other.

| | [`GenerationProgram`](generation.md) | `RecurrentGenerationProgram` |
|---|---|---|
| Model | A transformer whose tokens mix only in causal attention. | A model whose tokens mix through a recurrence. |
| What you give | The stateless `forward(tokens)`. cattorch derives the step. | A one-token `step` and its states, or a Hugging Face Mamba, Mamba-2 or RWKV model as it is. |
| Memory per token | Grows: every token adds a row to each layer's K/V cache. | Fixed: the states have the same size after any number of tokens. |
| Context limit | `max_context`, required. | `max_tokens`, optional. Without it the sprite takes any number of tokens. |

## Quick start

A hand-written model needs a `step` method and the states' initial values:

```python
import torch
from torch import nn

from cattorch import RecurrentGenerationProgram, State, transpile


class CharGRU(nn.Module):
    def __init__(self, vocab=32, width=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.gru = nn.GRU(width, width, batch_first=True)
        self.head = nn.Linear(width, vocab)

    def forward(self, tokens):  # [1, n] -> [1, n, vocab]
        output, _ = self.gru(self.embed(tokens))
        return self.head(output)

    def step(self, token, hidden):  # [1, 1] and [1, 1, width]
        output, hidden = self.gru(self.embed(token), hidden)
        return self.head(output), hidden


model = CharGRU().eval()
program = RecurrentGenerationProgram(
    (State("hidden", torch.zeros(1, 1, 64)),), reference="forward",
)
artifact = transpile(model, program, "build/char_gru")
```

`reference="forward"` makes the export check `step` against `forward`; see
[the reference check](#the-reference-check).

A Hugging Face Mamba, Mamba-2 or RWKV causal language model needs nothing
else. cattorch supplies the step, the states and the reference check:

```python
from transformers import AutoModelForCausalLM

from cattorch import RecurrentGenerationProgram, transpile

model = AutoModelForCausalLM.from_pretrained("AntonV/mamba2-130m-hf").eval()
artifact = transpile(model, RecurrentGenerationProgram(), "build/mamba2")
```

From the terminal, `cattorch generate --hf REPO` does the same; see
[the command line](#the-command-line).

## The step

`step(token, *states) -> (logits, *new_states)`:

- `token` is a `[1, 1]` tensor holding one token ID. The step must look it up
  in an embedding table; cattorch uses the table's size as the vocabulary when
  it checks token IDs in Scratch. The token may be reshaped, indexed,
  transposed, expanded or cast first, as in `self.embed(token[:, 0])`, but not
  computed with, as in `self.embed(token + 1)`.
- The states come in the order of `RecurrentGenerationProgram.states`, and the
  step returns the new values in the same order, each with its state's shape.
- The logits flatten to one row of vocabulary scores: `[V]`, `[1, V]` and
  `[1, 1, V]` all work.
- `step` is the method's name by default; `step="next_token"` names another.

The step is traced once, like any export, so it must follow one path: it can't
branch on tensor values. Any [supported operation](supported-models.md#operations)
can be used. A stack of layers can loop over them in Python.

## States

Each state is a `State(name, initial)` in replace mode: the step's return
value replaces it. `RecurrentGenerationProgram` rejects append states.

- **Initial values.** `cattorch reset` and `cattorch prefill` set every state
  back to its `initial` tensor, so a learned start state works.
- **Constant states aren't stored.** A state whose initial value is one number
  everywhere, such as zeros, is saved as an empty list and filled by a loop on
  `cattorch init`, on `cattorch reset`, and on the first prefill or decode in a
  freshly loaded project. Any other state is stored twice: the live list and a
  copy of its initial value.
- **Large states are sharded.** A state over 200,000 values is split into
  several lists, like any [large list](storage.md#list-sharding). This costs
  extra blocks and time on every step, so a model with a big state per layer
  is better off with one state per layer than one stacked state; the Hugging
  Face adapters do this.
- **Names are internal.** The states are the sprite's `_cattorch state NAME`
  lists, which aren't part of the [Scratch interface](scratch-interface.md).

## The reference check

With `reference="forward"`, or another method name, export runs the step from
the initial states over up to 32 random tokens (fewer if `max_tokens` is
smaller) and compares each logits row with
`reference(tokens)`, which must return `[1, n, V]` or `[n, V]` logits. Every
row must agree within `1e-4` of its largest value, or export stops with an
`UnsupportedModelError` naming the first row that differs. Without
`reference`, the step is only run once, to check its outputs.

Use it whenever the model has a stateless forward: a step that drops a term or
starts from the wrong state is otherwise exported without complaint. For a
Hugging Face model, the reference is always the model's own forward, so the
check catches a transformers version whose models compute something the
adapter doesn't.

## Requirements

- The model is in evaluation mode. A model left in training mode is exported
  as an evaluation-mode copy, with a warning.
- The step takes a `[1, 1]` token and returns the logits and every state, as
  [above](#the-step).
- States can have any shape. Floating-point states and weights are converted
  to float32, the weights on a copy; Scratch computes in doubles either way.

## Errors

| Message contains | Cause |
|---|---|
| `RecurrentGenerationProgram needs states=...` | No states were given, and the model isn't a supported Hugging Face recurrent model. |
| `RecurrentGenerationProgram() exports a Hugging Face ... model only on its own` | No states were given, and the model holds a supported Hugging Face model inside it, at the path named. Export that model directly, or write a step and pass states. |
| `step= and reference= need states=` | `step` or `reference` was given without `states`. |
| `model has no callable recurrent step`, `model has no callable reference` | The named method doesn't exist. |
| `step returned N tensors, expected 1 + number of states` | The step doesn't return the logits and one new value per state. |
| `step changes the shape of state` | A new state has a different shape from its initial value. |
| `the step's logits must flatten to one vocabulary row` | The logits have more than one row, for example a batch of two. |
| `the step's token must index an embedding table` | The token isn't looked up in an `nn.Embedding` (or `F.embedding`), for example a one-hot product. |
| `REFERENCE returned logits of shape` | The reference doesn't return one row per token. |
| `the recurrent step disagrees with REFERENCE at logits row N` | The reference check failed. The message gives the first differing row and the largest errors. For a Hugging Face model, it adds that the installed transformers version may be unsupported. |
| `top_k (N) exceeds output size (V)` | `top_k` is larger than the vocabulary. |
| `has no NAME, which the recurrent step reads` | A Hugging Face model is missing a module or attribute the adapter uses, usually because of a different transformers version. |
| `with config FIELD=VALUE isn't supported` | A Hugging Face config variant the adapter can't export; see [supported models](#hugging-face-models). |
| `without a language-model head` | A Hugging Face base model such as `Mamba2Model`. Export the `*ForCausalLM` model. |

Passing a Mamba, Mamba-2 or RWKV model to a `GenerationProgram` raises an
`UnsupportedModelError` that points here.

## Hidden prefill

Prefill runs the step once per prompt token. With `hidden_prefill=True` (the
default), the operations that only compute logits, such as the final norm and
the output head, run only for the last prompt token; state updates run for
every token. The logits and states after prefill are the same either way. Set
it to `False` only for debugging or measurement.

## Quantization and fast mode

`QuantizationConfig`, `optimization="fast"`, `adapters` and the
[storage options](storage.md) work as for any export. GPTQ takes
`calibration_sequences`: `[1, length]` token tensors. Calibration runs the step
over each sequence token by token and resets the states between sequences.

An int8 Mamba-2 stays within 4% of the largest logit in the tests. Fast mode
approximates GELU, LayerNorm and softmax in the step as it does anywhere else,
and `TranspileResult.approximate_operations` lists what it approximated;
`fallback_operations` lists operations that ran as decompositions. See
[optimization modes](optimization.md).

## Scratch interface

The blocks, lists, variables, statuses and broadcasts are those of a
[KV-cached generator](generation.md#scratch-interface), and the
[generation loop](generation.md#generation-loop) there works unchanged. The
differences:

- There is no K/V cache. The states live in internal lists.
- `cattorch cache length` counts the tokens consumed since the last reset or
  prefill.
- `cattorch max context` is `max_tokens`, or `Infinity` when `max_tokens` is
  `None`. The loop's stop condition `(cattorch cache length) = (cattorch max
  context)` then never ends generation; stop on a token count or an end token
  instead.
- `cattorch reset` restores every state's initial value and clears the logits
  and top-k lists, and so does `cattorch prefill` before it reads the prompt.

## Invalid input

A failed prefill or decode changes no state partway: the step's results are
committed only after it succeeds.

Scratch's stop sign, or a `stop all` block, can stop `cattorch reset`,
`cattorch prefill` or `cattorch decode` partway, even while a step commits its
states, so some states may be updated and others not. Every `cattorch decode`
after that fails as in the last row below and changes nothing, until
`cattorch reset` or `cattorch prefill` runs; both start from the initial
states. `cattorch init` doesn't clear the condition. `cattorch cache length`
keeps whatever value the stopped block left.

Calls from several scripts at once behave as for a K/V cache: see
[overlapping calls](generation.md#overlapping-calls). A block that another
block ran during reports the last status below, and decode then fails until
reset or prefill, which always recover once nothing else is running. The same
case isn't caught: a block that was
taken over, resumed while the newer block is paused, and unfinished when it
ends, or stopped after resuming, can leave the newer block or a later decode
reporting `ok` with logits from states it changed.

| Situation | Result | `cattorch status` |
|---|---|---|
| Decode with zero or several tokens | States unchanged; logits and top-k lists cleared | `decode requires exactly one token` |
| Decode after `max_tokens` tokens | States unchanged; logits and top-k lists cleared | `maximum context exceeded` |
| Prefill with no tokens | States restored to their initial values; logits empty | `prefill requires at least one token` |
| Prefill longer than `max_tokens` | States restored; logits empty | `prefill exceeds maximum context` |
| A token ID that isn't a whole number from 0 to the vocabulary size minus 1 | Prefill: states restored, logits empty. Decode: states unchanged, logits cleared | `token ids must be whole numbers from 0 to N` |
| Decode after a reset, prefill or decode was stopped partway, or while one runs in another script | States unchanged; logits and top-k lists cleared | `an earlier reset, prefill or decode was stopped or is still running; reset or prefill first` |
| A reset, prefill or decode that another one ran during, from another script | Logits and top-k lists cleared; every decode fails as above until reset or prefill | `another reset, prefill or decode ran during this one; reset or prefill first` |

## Hugging Face models

`RecurrentGenerationProgram()` with no states exports these `*ForCausalLM`
models, written and tested against transformers 5.20.2:

| Model | Class | States per layer | Values per layer |
|---|---|---|---|
| Mamba-2 | `Mamba2ForCausalLM` | `conv i`, `ssm i` | `(I + 2·G·N)·(K − 1)` and `H·P·N` |
| Mamba | `MambaForCausalLM` | `conv i`, `ssm i` | `I·(K − 1)` and `I·N` |
| RWKV (RWKV-4) | `RwkvForCausalLM` | `attention shift i`, `feed forward shift i`, `numerator i`, `denominator i`, `maximum i` | `2·D + 3·A` |

`I` is the intermediate size (`expand · hidden_size`), `N` the `state_size`,
`K` the `conv_kernel`, `G` the `n_groups`, `H` the `num_heads`, `P` the
`head_dim`, `D` the `hidden_size` and `A` the `attention_hidden_size`.

The step is computed from the model's weights and config, not through the
model's cache objects, whose API changed between transformers versions. It
reads module, parameter and attribute names such as `backbone.layers.N.mixer.in_proj`
and `rwkv.blocks.N.attention.time_decay`, so an export fails early with an
`UnsupportedModelError` naming anything missing. The model's stateless forward
is the reference, so other upstream changes fail the reference check instead
of exporting wrong numbers. The export works on a float32 copy; your model,
its weights and its config aren't changed.

Config variants:

| Model | Supported | Not supported |
|---|---|---|
| Mamba-2 | Any `n_groups` that divides `num_heads`, `time_step_limit` (clamping the time step as transformers' forward does), `use_conv_bias`, `use_bias`, `conv_kernel` of 1 or more, `residual_in_fp32` (no effect in float32), `hidden_act`, `chunk_size`, tied embeddings. The gated RMSNorm runs over the whole intermediate width, as in transformers. | `n_groups` that doesn't divide `num_heads`; a `time_step_limit` that isn't a `(minimum, maximum)` pair. |
| Mamba | Any `time_step_rank`, `use_mambapy` and `use_associative_scan` (they only change how transformers runs its forward), `use_conv_bias`, `use_bias`, `conv_kernel`, `residual_in_fp32`, `hidden_act`. | |
| RWKV | `rescale_every` of any value, `attention_hidden_size` different from `hidden_size`, any `intermediate_size`, tied embeddings. | |

RWKV's `rescale_every` deserves a note. In evaluation, transformers divides
some layers' output weights by powers of two in place the first time the model
runs, and halves the hidden state every `rescale_every` layers. The step does
the same arithmetic on its own copy, whether or not your model has run yet,
and your model's weights are left as they are. transformers' own `state=` path
fails when `attention_hidden_size` differs from `hidden_size`; the step
doesn't use it.

### Size

Every decode reads every weight once and rewrites every state. Mamba2-130m
(`hidden_size` 768, 24 layers, `state_size` 128, 24 heads of 64, one group,
`conv_kernel` 4) has 4,847,616 state values: per layer, a `conv` state of
5,376 and an `ssm` state of 196,608, just under the 200,000-item list limit,
so no state list is sharded. Mamba-2 models with more heads or larger states,
such as the 2.7B model's 655,360-value `ssm` states, shard them. Mamba-130m
has 29,184 state values per layer and RWKV-4 169M 3,840.

Measured with random weights in Mamba2-130m's shape, on a 4-core machine; the
first three rows use a 256-token vocabulary:

| Export | Time | `.sprite3` | `sprite.json` | Blocks |
|---|---|---|---|---|
| One layer, float32 | 3 s | 16 MB | 23 MB | 3,447 |
| Two layers, float32 | 4 s | 31 MB | 42 MB | 6,129 |
| Two layers, int8 | 3 s | 8 MB | 13 MB | 4,849 |
| All 24 layers and the 50,288-token tied embedding, int8 | 35 s | 136 MB | 180 MB | 66,659 |

The full int8 export peaked at 2.7 GB of memory, the model itself included.
Each layer adds about 3.8 million weights, 2 seconds and 15 MB in float32. The
whole float32 model, about 129 million weights, would be roughly 500 MB and
was not exported. Either is far beyond what Scratch's
website accepts; see [project size](storage.md#project-size). Each generated
token is about 129 million multiply-adds, so expect a full-size model to be
very slow; it wasn't timed in Scratch.

### The command line

`cattorch generate --hf REPO`, or `cattorch generate MODEL` when `MODEL` loads
one, recognizes these models and exports them with a
`RecurrentGenerationProgram`. `--max-context` sets `max_tokens`; without it
the generator takes any number of tokens. See
[projects and the command line](projects-and-cli.md#the-command-line).

Related: [KV-cached generation](generation.md),
[`RecurrentGenerationProgram`](api-reference.md#recurrentgenerationprogram),
[tokenizers](tokenizers.md), [storage and GPTQ](storage.md).
