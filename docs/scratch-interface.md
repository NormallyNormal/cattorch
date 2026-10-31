# Scratch interface

[Documentation home](index.md)

Scratch projects use an exported sprite through its blocks, lists, variables,
and broadcasts. This page lists them. They're a versioned contract:

- Every name that starts with `cattorch ` (with the space) is public, as are a
  model's `input` and `output` lists and their numbered forms.
- Within interface version 1, which covers cattorch 1.x, no public name is
  renamed or removed, and its meaning doesn't change. New names may be added.
- Every other name, including names that start with `_cattorch`, is internal:
  temporary lists, weight storage, helper blocks. Internal names can change in
  any release, so don't use them in your scripts.
- Each sprite has a `cattorch interface version` variable holding the version
  it implements, currently `1`. `cattorch.interface.INTERFACE_VERSION` is the
  version an installed cattorch writes.

Blocks run without screen refresh. When one finishes, the sprite does a
`broadcast and wait` of its completion message, so other sprites can react.

## Every model sprite

A sprite from [`transpile`](getting-started.md) with an example input:

| Name | Kind | Meaning |
|---|---|---|
| `cattorch forward` | block | Runs the model on `input` and writes `output`. Initializes the sprite first if needed. |
| `cattorch init` | block | Decodes the stored weights. Optional; `cattorch forward` calls it once. |
| `cattorch prepare for save` | block | Re-encodes the weights so a saved project stays small. |
| `input`, `input_1`, ... | lists | The model's input tensors, flattened in PyTorch order. |
| `output`, `output_1`, ... | lists | The model's outputs, flattened the same way. |
| `cattorch status` | variable | `ok` after a block finishes, or why `cattorch forward` didn't run, such as a variable-length input with too many items. |
| `cattorch initialized` | variable | `1` once the weights are decoded, `0` before. |
| `cattorch init complete`, `cattorch forward complete` | broadcasts | Sent when each block finishes. |

`TranspileResult.inputs` and `outputs` record each list's tensor shape. A
[variable-length](supported-models.md#variable-length-inputs) input or output
takes any whole number of items within its bounds.

## Generation sprites

A sprite from a [`GenerationProgram`](generation.md#scratch-interface) has no
`cattorch forward`, `input` or `output`. Instead:

| Name | Kind | Meaning |
|---|---|---|
| `cattorch init` | block | Decodes the stored weights. |
| `cattorch reset` | block | Clears the cache. |
| `cattorch prefill` | block | Takes a prompt from `cattorch tokens` and writes the next token's logits. |
| `cattorch decode` | block | Takes one token from `cattorch tokens`, adds it to the cache, and writes the next logits. |
| `cattorch tokens` | list | Zero-based token IDs to read. |
| `cattorch logits` | list | Logits for every token ID. |
| `cattorch top k values`, `cattorch top k ids` | lists | With `top_k`, the largest logits and their token IDs. |
| `cattorch cache length` | variable | Tokens in the cache. |
| `cattorch max context` | variable | The `max_context` it was exported with. |
| `cattorch status` | variable | `ok`, or an error message such as a full context or a token ID outside the vocabulary. |
| `cattorch initialized` | variable | `1` once the weights are decoded. |
| `cattorch init complete`, `cattorch reset complete`, `cattorch prefill complete`, `cattorch decode complete` | broadcasts | Sent when each block finishes. |

## Program sprites

A sprite from an [`ExportProgram`](programs-and-moe.md#named-entrypoints-and-state)
or [`@entrypoint` markers](markers.md) has a block per entry point, named
after it:

| Name | Kind | Meaning |
|---|---|---|
| `cattorch ENTRY` | block | Calls the entry point. |
| `cattorch ENTRY INPUT`, `cattorch ENTRY OUTPUT` | lists | Its inputs and outputs, named after the parameters and outputs. |
| `cattorch state STATE` | list | Each piece of persistent state. |
| `cattorch init`, `cattorch reset`, `cattorch prepare for save` | blocks | Decode the weights, restore every state to its initial value, and re-encode the weights. |
| `cattorch status` | variable | `ok` after a call, or why it failed; a failed call changes nothing. |
| `cattorch ENTRY complete`, `cattorch init complete` | broadcasts | Sent when a call or `cattorch init` finishes. |

## Tokenizer sprites

| Name | Kind | Meaning |
|---|---|---|
| `cattorch tokenize` | block | Reads `cattorch text` and writes its token IDs to `cattorch token ids`. |
| `cattorch detokenize` | block | Reads `cattorch token ids` and writes the text to `cattorch text`. |
| `cattorch text` | variable | The text to tokenize, or the detokenized text. |
| `cattorch token ids` | list | Zero-based token IDs. |

## Shared details

- **Lists over 200,000 items** are split into `NAME`, `NAME shard 2`,
  `NAME shard 3`, and so on, with `cattorch NAME shard count` and
  `cattorch NAME logical length` variables. See
  [list sharding](storage.md#list-sharding).
- **Storage** is described by the read-only variables
  `cattorch storage compression`, `cattorch storage precision`,
  `cattorch storage scale precision`, `cattorch quantization method`, and
  `cattorch quantization bits`.
- **Token IDs and indices** are zero-based, as in PyTorch, though Scratch list
  positions start at 1.
- **Projects** from [`build_project`](projects-and-cli.md#bridges) add global
  lists, variables, and broadcasts that start with the sprite's name. The same
  promise covers the names on that page.

Related: [getting started](getting-started.md), [glossary](glossary.md).
