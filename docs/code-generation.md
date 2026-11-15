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
- `layer_sharing=True` generates one copy of the code for repeated layers
  and calls it once per layer, with each layer's weights. See
  [layer sharing](#layer-sharing).

Schema compaction and layer sharing are tested less widely than the
defaults. After enabling either, run `verify` and check that the project
imports, edits, runs, and saves in your Scratch client.

### Layer sharing

Layer sharing looks for submodules named `blocks.0`, `blocks.1`, and so on,
such as the items of an `nn.ModuleList` attribute called `blocks`. It shares
them when:

- they are in one `blocks` list, numbered consecutively from `blocks.0`, and
  run one after another, with no code between them;
- each layer passes one tensor on: it reads the previous layer's output
  (`blocks.0`: its input) and otherwise only the same tensors from before the
  layers, such as a mask, and nothing reads its tensors but the next layer
  (the last layer: the rest of the model). A layer that takes or returns two
  tensors, a skip connection across layers, a layer reading the model's
  input at every layer, or returning every layer's output keeps the model
  unshared. So does an elementwise operation fused with a layer's last
  operation, such as `x * 2` right after the last layer or at the start of
  each layer; a normalization or linear layer there is fine;
- every layer computes the same operations, with the same shapes and
  constants, and weights of the same sizes. Layers with a
  [mixture of experts](programs-and-moe.md#mixture-of-experts) block aren't
  shared, since each layer's experts are an expert family of their own;
- for [cached generation](generation.md#layer-sharing), every attention block
  sits in its own layer and every layer's cache has the same width.

The layers can be anywhere in the model, including at its start, where they
read the input list; the input list keeps the values you put in it.

Sharing is all or nothing: if any layer doesn't fit, the model is exported
without sharing. Whenever `layer_sharing=True` shares nothing, the export
warns once, in `result.warnings` and as a `CattorchWarning`, and the warning
says why, such as which layer differs from `blocks.0`. A model with no
`blocks.N` layers warns too, since the option had no effect. In a
[program](programs-and-moe.md), each entry point is shared on its own: an
entry point without repeated layers, such as one that resets state, isn't
reported, an entry point whose layers couldn't be shared is reported by
name, and a program in which no entry point has repeated layers warns once.

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
