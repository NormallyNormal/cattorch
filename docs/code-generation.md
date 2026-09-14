# Code generation and JSON size

[Documentation home](index.md)

`CodegenConfig` trades generated code size against speed. The defaults suit
most models. Use the compact settings when the sprite's expanded JSON is too
large.

```python
from cattorch import CodegenConfig, transpile

artifact = transpile(
    model,
    example,
    "model_compact",
    codegen=CodegenConfig(
        unrolling="compact",
        compact_internal_names=True,
    ),
)
print(artifact.expanded_json_bytes)
```

## Choose a size policy

Loop unrolling copies a loop's body several times so each pass does more
work, which runs faster in Scratch but adds blocks.

| `unrolling` | Behavior |
|---|---|
| `"auto"` (default) | Adjusts unrolling to the estimated weight size and JSON budget. |
| `"compact"` | Uses the smallest loops. |
| `"speed"` | Uses each kernel's maximum supported unrolling, producing more blocks. |

`target_json_bytes` defaults to `4 * 1024 * 1024`. `"auto"` unrolling aims to
stay under it, and the exporter warns if the final JSON is larger. It is a
target, not Scratch's upload limit; see [project size](storage.md#project-size).
Set it to `None` to turn it off.

To compare settings, time them in Scratch; see
[verification and benchmarking](verification-and-benchmarking.md).

## Optional compaction

All three options are off by default.

- `compact_internal_names=True` shortens the names of internal variables,
  lists, and custom blocks. The documented input, output, and lifecycle names
  don't change.
- `compact_schema=True` leaves redundant fields out of the saved block data.
- `layer_sharing="auto"` generates one copy of the code for repeated
  transformer layers named `blocks.N` and shares it across them. For cached
  generation, every layer must also have the same cache width. Layers that
  don't fit keep their own code.

Schema compaction and layer sharing are newer. After enabling either, run
`verify` and check that the project imports, edits, runs, and saves in your
Scratch client.

## Reproducible exports and multiple sprites

Generated block IDs include a random three-character namespace. For
reproducible exports, set `id_namespace` to three characters from `A-Z`,
`a-z`, `0-9`, `-`, or `_`:

```python
CodegenConfig(id_namespace="m01")
```

Give each generated sprite in a project a different namespace so their IDs
don't collide. `id_namespace=""` makes IDs shorter, but only use it when the
sprite is the only generated sprite in the project.

`transpile_tokenizer` and the tokenizer `.save()` methods accept the same
`codegen=` argument.

Related: [`CodegenConfig` reference](api-reference.md#codegenconfig).
