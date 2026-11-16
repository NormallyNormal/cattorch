# KV-cached generation

[Documentation home](index.md)

A decoder-only language model can be exported as a sprite that keeps a KV
cache and produces one token at a time. Pass the model you already have, a
stateless `forward(tokens) -> logits`, and cattorch derives the cached
one-token step itself.

```python
from cattorch import GenerationProgram, transpile

artifact = transpile(model, GenerationProgram(max_context=64), "cached_model")
```

The model can be a hand-written decoder or an unwrapped Hugging Face causal
language model (GPT-2, Llama, Qwen2) loaded as it is. Models without attention,
whose tokens mix through a recurrence (GRU and LSTM language models, Mamba,
RWKV), use a [`RecurrentGenerationProgram`](recurrent-generation.md) instead.
As with any export, the model is exported as an
[evaluation-mode copy](troubleshooting.md#put-the-model-in-evaluation-mode).
`max_context` is fixed at export, and at least 3. The other fields are
`hidden_prefill`, `top_k`, and `method`, which names the method to export
when it isn't `forward` (by default the method marked
[`@generate`](markers.md), if any).
See [`GenerationProgram`](api-reference.md#generationprogram) for all fields.
The export returns a `GenerationResult`, which describes the token list, the
logits and top-k lists, `max_context`, and the blocks.

## How it works

cattorch traces `forward` at `max_context` tokens and again at one token
fewer, and compares the two graphs. It finds the causal attention blocks and
the tables indexed by position (learned position embeddings, RoPE sine and
cosine), then rewrites the graph to process a single token:

- each position table becomes a lookup of the current row;
- each attention block reads and writes a per-layer K/V cache. Caches hold the
  key and value heads before grouped-query expansion, so a model with fewer KV
  heads gets a smaller cache.

The first tensor the model returns is the interface, which is `logits` for a
Hugging Face model. It is flattened to one vocabulary row.

At export, cattorch runs the one-token step over a random token sequence of up
to 32 tokens and compares every output row with the original `forward`. They
must agree within `1e-4` of the row's largest value, or export stops with an
`UnsupportedModelError` naming the first row that differs. This catches models
that break the requirements below in ways the analysis can't see.

The step is built from a trace at `max_context` tokens, so later positions
run the same operations as the first 32, with their own rows of any position
table. What the step can't see is a model whose answer for a token depends
on the prompt's length. So export also compares the model's answer for a
token in a prompt of `max_context` tokens with its answer in shorter prompts,
at about a dozen lengths spread up to `max_context` (1, 2, 16, 31, 32,
`max_context - 1`, and each eighth of `max_context`). A Python branch on one
exact length outside those, such as `if tokens.shape[1] == 37:`, isn't
caught.

## Requirements

- Attention is causal: each token attends to itself and earlier tokens, through
  a mask or `is_causal=True`. The attention is matmul, optional scale, mask,
  softmax, matmul, which is also what `F.scaled_dot_product_attention` becomes
  in a trace. Multi-head, grouped-query (`repeat_kv`), and single-KV-head
  attention, including one KV head broadcast against all query heads, are
  recognized.
- Tokens mix only in attention. Everything else works on each position
  separately, so a `cumsum`, mean, `roll`, or convolution along the sequence is
  rejected. A model that mixes tokens through a recurrence or a causal
  convolution, such as an RNN, Mamba, or RWKV, can be exported with a
  [`RecurrentGenerationProgram`](recurrent-generation.md) and a one-token step.
- The graph doesn't depend on the sequence length. A Python `if
  tokens.shape[1] > 16:` that changes the operations is rejected, and so is a
  position table whose first rows change with the length.
- The model returns logits for every position, not only the last. Slicing to
  the last position inside `forward` isn't supported.
- The input is a `[1, length]` tensor of token IDs, looked up in an embedding.
- Attention masks are causal masks, optionally limited to a sliding window of
  the last N tokens (Mistral, GPT-OSS). The scores may also carry an ALiBi
  bias, `slope * (key - query)` per head, possibly plus a constant per row, and
  per-head attention sinks: an extra softmax logit concatenated to the scores
  and dropped afterwards, as GPT-OSS does. Any other float bias is rejected.
  With a window, each decode step only reads the window's rows of the cache.

## Errors

| Message contains | Cause |
|---|---|
| `attention over the sequence isn't causal` | A bidirectional model such as BERT, or a mask that lets a token see a later one. The message names the token pair. |
| `allows mixing tokens only in causal attention` | An operation combines positions: `cumsum`, a mean over the sequence, `roll`, `flip`, a convolution along it. The message names the operation and its module. |
| `the model computes something else when the sequence is one token shorter` | A length-dependent Python branch that takes another path at `max_context - 1` tokens. Remove it or fix it to one path. |
| `changes when more tokens follow it` | The output for a token depends on the prompt's length: a length-dependent Python branch that takes the same path at the two traced lengths, or positions scaled by the length (dynamic NTK RoPE). |
| `isn't a recognised attention pattern` | A softmax over the sequence that isn't matmul, scale, causal mask, softmax, matmul. The message says which part didn't match. |
| `the table's first N rows ... differ` | A position table whose rows depend on the sequence length. |
| `the cached step disagrees with the model` | The export-time check failed. The message gives the first differing row and the largest error. |
| `GenerationProgram caches attention keys and values, and this ... model has no attention` | A Hugging Face Mamba, Mamba-2, or RWKV model. Use [`RecurrentGenerationProgram()`](recurrent-generation.md). |

## Hidden prefill

With `hidden_prefill=True` (the default), prefill computes logits only for the
last prompt token. For the other prompt tokens, the output head is skipped,
and the final layer keeps only its K/V append: it skips its attention output,
MLP, and final norm, and the operations that only compute its query, such as
the query projection and its RoPE. With [shared](#layer-sharing) layers, the
shared code checks whether it is running as the final layer. Set it to
`False` only for debugging or measurement.

## Layer sharing

`CodegenConfig(layer_sharing=True)` shares one block of code across repeated
layers when the attention blocks sit in consecutive modules named `blocks.0`,
`blocks.1`, and so on. The caches are then banked into one list per K and V
(the internal lists `_cattorch K cache` and `_cattorch V cache`). Models with other layer names, such as
Hugging Face's `model.layers.N`, export without sharing and with one list per
layer (`_cattorch K cache 1`, `_cattorch V cache 1`, ...). If the layers
can't be shared, for example because the MLP widths differ, or the model has
no `blocks.N` layers, cattorch exports without sharing, with a warning in
`result.warnings` that says why. Sharing makes sprites smaller. See
[layer sharing](code-generation.md#layer-sharing) for what can be shared.

## Quantization

The model is prepared the way an ordinary export is, before the step is built.
`QuantizationConfig`, including GPTQ with token sequences as `calibration`,
`optimization="fast"`, `adapters`, and the [storage options](storage.md) work
as they do for any model. With `bits=4` or `bits=6`, the output head, the
weight that computes the logits (GPT-2's tied embedding table), is stored
in int8 unless `QuantizationConfig.head_bits` or an override says
otherwise; see [the output head](storage.md#the-output-head-of-a-language-model).

## Verify a generator

`verify` runs prefill, decode and reset calls in the emulator and compares
each with PyTorch: the logits, the top-k lists, and `cattorch status`,
including the statuses an invalid call should give:

```python
import torch
from torch import nn

from cattorch import GenerationProgram, ProgramCall, transpile, verify

class Decoder(nn.Module):
    def __init__(self, vocabulary=16, width=8):
        super().__init__()
        self.embed = nn.Embedding(vocabulary, width)
        self.head = nn.Linear(width, vocabulary)

    def forward(self, tokens):
        hidden = self.embed(tokens)
        attended = nn.functional.scaled_dot_product_attention(
            hidden, hidden, hidden, is_causal=True,
        )
        return self.head(hidden + attended)

decoder = Decoder().eval()
program = GenerationProgram(max_context=4, top_k=3)
result = transpile(decoder, program, "decoder")
report = verify(decoder, program, result, calls=[
    ProgramCall("prefill", (torch.tensor([[1, 2, 3]]),)),
    ProgramCall("decode", (torch.tensor([[4]]),)),
    ProgramCall("decode", (torch.tensor([[5]]),)),  # the cache is full
])
assert report.passed
print([call.status for call in report.calls])  # ['ok', 'ok', 'context full']
```

Without `calls`, `verify` prefills a few tokens and decodes a few more. Pass
the export's `quantization`, `storage` and `sig_figs` too, and PyTorch uses
the values the sprite stores.

## Scratch interface

The sprite has `cattorch init`, `cattorch reset`, `cattorch prefill`,
`cattorch decode` and `cattorch prepare for save` blocks. Put the prompt's
token IDs in the `cattorch token ids` list and run `cattorch prefill`; it
resets the cache, reads the prompt, and writes the next token's logits to
`cattorch logits`, one per zero-based token ID. Then put one token ID in
`cattorch token ids` and run `cattorch decode` for each token after that.
`cattorch cache length` counts the tokens in the cache, and
`cattorch max context` holds `max_context`. Each block sets `cattorch status`
and sends its completion broadcast, such as `cattorch prefill complete`, when
it ends; check that the status is `ok` before you read the logits. The
[Scratch interface](scratch-interface.md#generation-sprites) page lists every
name and status.

`top_k` can be `None` (the default) or an integer from 1 to 64. With an
integer, each prefill and decode also writes the `top_k` largest logits,
largest first, to `cattorch top k values`, and their token IDs to
`cattorch top k ids`. Top-k only sorts candidates; it doesn't pick a token.
Like `torch.topk`, it always gives `top_k` real token IDs, including tokens
whose logit is `-Infinity`, and ranks a NaN logit first. Equal logits are
listed lowest token ID first; `torch.topk` doesn't specify their order, so
the IDs of tied logits can differ from it.
Sampling, temperature, repetition penalties, stopping at EOS, and turning
tokens into text are up to your Scratch project.

These blocks and lists belong to the generated sprite. To drive generation from
another sprite, use the
[global-list bridge](getting-started.md#connect-another-sprite), or let
[`build_project`](projects-and-cli.md) write a project with the model, its
tokenizer, bridges, and a generation loop already wired together.

## Generation loop

A typical loop looks like this:

```text
copy prompt token IDs into local [cattorch token ids]
cattorch prefill

set [finished] to [false]
repeat until <(finished) = [true] or (cattorch status) != [ok] or
              (cattorch cache length) = (cattorch max context)>
    sample one token from [cattorch logits] or [cattorch top k values/ids]
    if <(sampled token) = (EOS)> then
        set [finished] to [true]
    else
        stream or detokenize the sampled token
        delete all of [cattorch token ids]
        add (sampled token) to [cattorch token ids]
        cattorch decode
    end
end
```

The prompt counts toward the context, so stop when `cattorch cache length`
reaches `cattorch max context`.

## Invalid input

Prefill and decode check `cattorch token ids` before they change anything. A
block that fails writes no logits and leaves the cache as it was: a prefill
that fails doesn't reset first.

| Situation | `cattorch status` |
|---|---|
| Decode with zero or several tokens | `wrong number of items: cattorch token ids` |
| Decode when the cache is full | `context full` |
| Prefill with no tokens | `too few items: cattorch token ids` |
| Prefill longer than `max_context` | `too many items: cattorch token ids` |
| A token ID that isn't a whole number from 0 to the vocabulary size minus 1 | `invalid item: cattorch token ids` |
| Decode after a reset, prefill or decode was stopped or taken over partway, or while one runs in another script | `an earlier call was stopped or is still running; reset first` |
| A block that another of the sprite's blocks started during | `another call ran during this one`; see [overlapping calls](scratch-interface.md#overlapping-calls) |

## Cache behavior

A step past `max_context` is refused before it changes any cache. Caches
longer than 200,000 items are [sharded](storage.md#list-sharding)
automatically. To shrink the sprite, see [layer sharing](#layer-sharing).

Scratch's stop sign, or a `stop all` block, can stop `cattorch reset`,
`cattorch prefill` or `cattorch decode` partway, leaving some layers' caches a
token longer than others or holding part of a token. Every `cattorch decode`
after that fails as in the table above and changes nothing, until
`cattorch reset` or `cattorch prefill` runs; both start from empty caches.
`cattorch init` doesn't clear the condition. `cattorch cache length` keeps
whatever value the stopped block left. Call the sprite's blocks from one
script at a time; what the sprite detects when calls overlap, and how to
recover, is in [overlapping calls](scratch-interface.md#overlapping-calls).

Before saving a project that has generated, call `cattorch prepare for save`;
see [save after running](storage.md#save-after-running).

Related: [recurrent generation](recurrent-generation.md),
[tokenizers](tokenizers.md), [storage and GPTQ](storage.md),
[supported models](supported-models.md), and
[generation benchmarking](verification-and-benchmarking.md#scratch-benchmark-projects).
