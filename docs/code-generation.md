# Code generation and JSON size

`CodegenConfig` controls generated block density and internal Scratch IDs:

```python
from cattorch import CodegenConfig, transpile

transpile(
    model,
    example,
    "model",
    codegen=CodegenConfig(
        target_json_bytes=4 * 1024 * 1024,
        unrolling="auto",
        id_namespace=None,
        compact_internal_names=True,
    ),
)
```

`auto` estimates the stored-weight payload and reduces partial loop unrolling
as that payload consumes the JSON budget. `compact` always uses factor-one
loops; `speed` retains each kernel's maximum tested factor. The target is a
soft code-generation budget, not a universal Scratch upload limit. cattorch
warns if the final expanded JSON exceeds it.

Internal IDs use the fixed `ct` marker followed by a namespace and Base64url
counter. The marker keeps generated IDs distinct from Scratch's fixed toolbox
block IDs. The default namespace is three random Base64url characters (18
bits), which suits the common case of one cattorch sprite and still protects
small multi-sprite projects from likely collisions. Set an explicit
three-character namespace for reproducible exports:

```python
CodegenConfig(id_namespace="m01")
```

Set `id_namespace=""` for the smallest safe IDs (`ctA`, `ctB`, ...) only when
the sprite will not be combined with another export whose internal IDs may
overlap. Allowed namespace characters are `A-Z`, `a-z`, `0-9`, `-`, and `_`.

`compact_internal_names=True` renames private variables, lists, and custom
blocks to short sequential display names. The documented input/output,
lifecycle, generation, and tokenizer interfaces retain their stable names.
`compact_schema=True` additionally omits redundant false-valued block fields;
it is opt-in because a browser import/edit/save cycle remains the final
compatibility authority for schema-level trimming.

`layer_sharing="auto"` recognizes compatible stateless `blocks.N` transformer
stacks, banks corresponding layer weights, and emits one shared warp procedure.
It falls back without changing the graph when the stack differs, a bank would
cross Scratch's 200,000-item list limit, or generation uses layer-specific K/V
caches. The option is currently opt-in (`"off"` by default) pending another
vanilla-browser correctness pass; cattorch's emulator verifies exact output for
accepted stacks.

Generation top-k selection also follows the code-size policy. `auto` and
`compact` use a constant-size loop-based insertion selector; `speed` retains
the larger fully unrolled selector.

The same `codegen=` argument is accepted by `transpile_tokenizer` and tokenizer
`.save()` methods. Character and raw-text BPE tokenizers are generated from the
typed Scratch DSL. Their older JSON files are no longer part of production
lowering. The pre-0.4 JSON-template backend has been removed; historical
benchmark results remain documented in `benchmarks/`.
