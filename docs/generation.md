# KV-cached generation

[Documentation home](index.md)

A decoder-only language model can be exported as a sprite that keeps a KV
cache and produces one token at a time, as long as cattorch recognizes its
attention pattern (see [graph requirements](#recognized-graph-requirements)).

```python
import torch

from cattorch import GenerationProgram, transpile

artifact = transpile(
    model,
    GenerationProgram(
        method="next_token_logits",
        example_token=torch.tensor([[0]]),
        max_context=64,
        hidden_prefill=True,
        top_k=5,
    ),
    "cached_model",
)
```

The model must be in evaluation mode. `method` names the model method to
export, and `example_token` must be a single integer token with shape `[1, 1]`.
`max_context` is fixed at export. See
[`GenerationProgram`](api-reference.md#generationprogram) for all fields.

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
[global-list bridge](getting-started.md#connect-another-sprite).

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

## Recognized graph requirements

Cached generation works when the model:

- Uses a combined QKV projection with causal attention in a form cattorch
  recognizes.
- Looks up positions from a table, either learned position embeddings or
  precomputed RoPE sine and cosine tables, so cattorch can index the table with
  the current cache length.
- Returns one tensor, and otherwise uses [supported operations](supported-models.md).

Watch out for stateless wrappers that slice the position or RoPE table by
sequence length (`table[:length]`). Traced with a `[1, 1]` example, that slice
only contains position zero, so every later token gets the wrong position.
Instead, have the generation method index the full table with an explicit
position tensor.

If a step fails partway, every layer's cache is rolled back, so the cache never
holds a partial token.

`hidden_prefill=True` skips computing logits for every prompt position except
the last. Set it to `False` only for debugging or measurement. Caches longer
than 200,000 items are [sharded](storage.md#list-sharding) automatically. To
shrink the sprite, see layer sharing in
[code generation](code-generation.md#optional-compaction).

Generation sprites don't have `cattorch prepare for save`; see
[save after running](storage.md#save-after-running).

Related: [tokenizers](tokenizers.md), [storage and GPTQ](storage.md),
[supported models](supported-models.md), and
[generation benchmarking](verification-and-benchmarking.md#scratch-benchmark-projects).
