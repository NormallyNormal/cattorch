# KV-cached generation

Decoder-only models using a recognized combined-QKV causal-attention graph can
be exported as stateful, single-token decoders.

```python
import torch

from cattorch import GenerationConfig, transpile

artifact = transpile(
    model,
    torch.tensor([[0]]),
    "cached_model",
    generation=GenerationConfig(max_context=64, top_k=5),
)
```

The sprite retains `cattorch forward` and adds:

- `cattorch reset cache`, which clears layer K/V lists and output;
- `cattorch prefill`, which reads a prompt from `input`, resets state, and
  leaves final-token logits in `output`;
- `cattorch decode`, which reads one token from `input`, appends K/V entries,
  and writes next-token logits to `output`.

`cattorch cache length` reports the number of cached tokens. At `max_context`,
decode clears `output` and sets `cattorch status` to `maximum context exceeded`.
`cattorch decode` also requires exactly one item in `input`. Empty or multi-item
decode inputs leave the cache unchanged, clear generated outputs, and report a
descriptive status. Prefill similarly rejects empty prompts and prompts longer
than `max_context` before running the model.

`top_k` may be `None` or an integer from 1 through 16. When enabled, the
project writes selected logits in descending order to `cattorch top k values`
and their zero-based token IDs to `cattorch top k ids`. Full logits remain in
`output`. Leaving it disabled adds no selection overhead.

Generation currently requires:

- batch size one and a `[1, 1]` integer example input;
- a standard combined-QKV causal-attention pattern recognized by cattorch;
- positional values expressed as a dynamic embedding lookup that cattorch can
  tie to the current cache length; this may be a learned position embedding or
  precomputed RoPE sine/cosine tables;
- a context size fixed at export.

Do not trace a one-token stateless attention wrapper that slices a positional
or RoPE table with `:length`. With a `[1, 1]` example, that graph contains only
position zero and cannot recover later positions during cached decoding. Use a
generation wrapper that indexes the full exported table from an explicit
position tensor, as the production TinyStories builder does.

Cache lists larger than 200,000 items are automatically sharded. A separate
hidden-only prefill path avoids computing projected logits for every prompt
position when the recognized graph permits it.
