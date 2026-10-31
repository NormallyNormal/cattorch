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
language model (GPT-2, Llama, Qwen2) loaded as it is. It must be in evaluation
mode. `max_context` is fixed at export. Other fields are `hidden_prefill`,
`top_k`, and `method`, which names the method to export when it isn't
`forward`. See [`GenerationProgram`](api-reference.md#generationprogram) for
all fields.

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

## Requirements

- Attention is causal: each token attends to itself and earlier tokens, through
  a mask or `is_causal=True`. The attention is matmul, optional scale, mask,
  softmax, matmul, which is also what `F.scaled_dot_product_attention` becomes
  in a trace. Multi-head, grouped-query (`repeat_kv`), and single-KV-head
  attention, including one KV head broadcast against all query heads, are
  recognized.
- Tokens mix only in attention. Everything else works on each position
  separately, so a `cumsum`, mean, `roll`, or convolution along the sequence is
  rejected.
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
| `mixes tokens outside attention`, `allows mixing tokens only in causal attention` | An operation combines positions: `cumsum`, a mean over the sequence, `roll`, `flip`, a convolution along it. The message names the operation and its module. |
| `the model's graph changes with the sequence length` | A length-dependent Python branch. Remove it or fix it to one path. |
| `not a recognised attention pattern` | A softmax over the sequence that isn't matmul, scale, causal mask, softmax, matmul. The message says which part didn't match. |
| `the table's first N rows ... differ` | A position table whose rows depend on the sequence length. |
| `the cached step disagrees with the model` | The export-time check failed. The message gives the first differing row and the largest error. |

## Hidden prefill

With `hidden_prefill=True` (the default), prefill computes logits only for the
last prompt token. For the other prompt tokens, the output head is skipped,
and the final layer keeps only its K/V append: it skips its attention output,
MLP, and final norm, and the operations that only compute its query, such as
the query projection and its RoPE. With [shared](#layer-sharing) layers, the
shared code checks whether it is running as the final layer. Set it to
`False` only for debugging or measurement.

## Layer sharing

`CodegenConfig(layer_sharing="auto")` shares one block of code across repeated
layers when the attention blocks sit in consecutive modules named `blocks.0`,
`blocks.1`, and so on. The caches are then banked into one list per K and V
(`_cattorch K cache` and `_cattorch V cache`). Models with other layer names, such as
Hugging Face's `model.layers.N`, export without sharing and with one list per
layer (`_cattorch K cache 1`, `_cattorch V cache 1`, ...). If the layers look
repeated but can't be factored, for example because the MLP widths differ,
cattorch logs a warning and exports again without sharing. Sharing makes
sprites smaller.

## Quantization

The model is prepared the way an ordinary export is, before the step is built.
`QuantizationConfig`, including GPTQ with `calibration_sequences`,
`optimization="fast"`, `adapters`, and the [storage options](storage.md) work
as they do for any model.

## Scratch interface

The sprite has these My Blocks, all of which run without screen refresh:

| Procedure | Input | Result |
|---|---|---|
| `cattorch init` | none | Decodes static storage and prepares the model. |
| `cattorch reset` | none | Clears the K/V cache, logits, and top-k lists. |
| `cattorch prefill` | one or more prompt IDs in `cattorch tokens` | Resets state, caches the prompt, and leaves final-token logits in `cattorch logits`. |
| `cattorch decode` | exactly one token ID in `cattorch tokens` | Adds the token to the cache and writes next-token logits. |

The variables and lists you use are:

| Name | Contents |
|---|---|
| `cattorch cache length` | Number of tokens currently in the cache. |
| `cattorch max context` | The `max_context` set at export. |
| `cattorch status` | `ok` after a successful reset, prefill, or decode; otherwise an error message. |
| `cattorch tokens` | Prompt tokens for prefill, or one token for decode. |
| `cattorch logits` | Logits for every token, indexed by zero-based token ID. |
| `cattorch top k values`, `cattorch top k ids` | When `top_k` is set, the largest logits in descending order and their zero-based token IDs. |

`top_k` can be `None` (the default) or an integer from 1 to 64. With `None`,
no top-k work is done. Top-k only sorts candidates; it doesn't pick a token.
Sampling, temperature, repetition penalties, stopping at EOS, and turning
tokens into text are up to your Scratch project.

These blocks and lists belong to the generated sprite. To drive generation from
another sprite, use the
[global-list bridge](getting-started.md#connect-another-sprite), or let
[`build_project`](projects-and-cli.md) write a project with the model, its
tokenizer, bridges, and a generation loop already wired together.

## Lifecycle notifications

When each block finishes, it does a `broadcast and wait` of `cattorch init
complete`, `cattorch reset complete`, `cattorch prefill complete`, or
`cattorch decode complete`.

Other sprites can use these to update a UI, for example showing "loading" until
`cattorch init complete` and "thinking" until `cattorch prefill complete`.
After prefill, check that `cattorch status` is `ok` before decoding. The
broadcasts don't give other sprites access to the generated sprite's lists.

## Generation loop

A typical loop looks like this:

```text
copy prompt token IDs into local [cattorch tokens]
cattorch prefill

set [finished] to [false]
repeat until <(finished) = [true] or (cattorch status) != [ok] or
              (cattorch cache length) = (cattorch max context)>
    sample one token from [cattorch logits] or [cattorch top k values/ids]
    if <(sampled token) = (EOS)> then
        set [finished] to [true]
    else
        stream or detokenize the sampled token
        delete all of [cattorch tokens]
        add (sampled token) to [cattorch tokens]
        cattorch decode
    end
end
```

The prompt counts toward the context, so stop when `cattorch cache length`
reaches `cattorch max context`.

## Invalid input

| Situation | Result | `cattorch status` |
|---|---|---|
| Decode with zero or several tokens | Cache unchanged; logits and top-k lists cleared | `decode requires exactly one token` |
| Decode when the cache is full | Cache unchanged; logits and top-k lists cleared | `maximum context exceeded` |
| Prefill with no tokens | Cache reset; logits empty | `prefill requires at least one token` |
| Prefill longer than `max_context` | Cache reset; logits empty | `prefill exceeds maximum context` |

## Cache behavior

If a step fails partway, every layer's cache is rolled back, so the cache never
holds a partial token.

Caches longer than 200,000 items are [sharded](storage.md#list-sharding) automatically. To
shrink the sprite, see layer sharing in
[code generation](code-generation.md#optional-compaction).

Generation sprites don't have `cattorch prepare for save`; see
[save after running](storage.md#save-after-running).

Related: [tokenizers](tokenizers.md), [storage and GPTQ](storage.md),
[supported models](supported-models.md), and
[generation benchmarking](verification-and-benchmarking.md#scratch-benchmark-projects).
