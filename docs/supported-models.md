# Supported models and operations

[Documentation home](index.md)

cattorch traces one fixed inference path through the model, using the shapes
and dtypes in `example_inputs`. The Python code can't branch on tensor values.
Parameters, buffers, and constants are stored in the sprite, and tensor
arguments become Scratch input lists.

For persistent state and bounded variable-length tensors, use an experimental
[`ExportProgram`](programs-and-moe.md#named-entrypoints-and-state).

## Operations

| Category | Operations |
|---|---|
| Convolution | Batched `nn.Conv1d`, `nn.Conv2d`, with optional bias, stride, and padding; dilation and groups must be 1 |
| Pooling | Batched `nn.MaxPool1d/2d`, `nn.AvgPool1d/2d`, `nn.AdaptiveAvgPool1d/2d`, `nn.AdaptiveMaxPool1d/2d`; non-default dilation, ceiling, padding-count, and divisor options are rejected |
| Linear algebra | `nn.Linear`, `@`, `torch.matmul`; two batched operands must have identical batch dimensions |
| Activations | ReLU, ReLU6, hardtanh, hardswish, sigmoid, tanh, GELU (exact and tanh forms), SiLU, Mish, softplus, leaky ReLU, ELU |
| Normalization | BatchNorm1d/2d (including without running statistics), LayerNorm, RMSNorm, GroupNorm, InstanceNorm, `F.normalize`, `torch.rsqrt` |
| Attention primitives | softmax, log-softmax, embeddings, registered-buffer `masked_fill`, `F.scaled_dot_product_attention` (causal or with a static mask), `cattorch.rotary_embedding` |
| Arithmetic | tensor/scalar `+`, `-`, `*`, `/`; unary `-`; `exp`, `log`, `log10`, `sqrt`, `rsqrt`, `abs`, `floor`, `ceil`, `reciprocal`, `erf`, `sin`, `cos`; `clamp`; `torch.pow` with an integer exponent from -4 to 4 or ±0.5 |
| Reduction | `sum`, `mean`, `prod`, `amax`, `amin`, `max`/`min` (with indices), `argmax`, `argmin`, `any`, `all`, `var`, `std`, `cumsum`, over any dimensions; `torch.topk` |
| Comparison and selection | `==`, `!=`, `<`, `<=`, `>`, `>=`, `~` on comparisons, `torch.where`, `torch.maximum`, `torch.minimum`; results are 0/1 tensors |
| Tensor creation | `arange`, `ones`, `zeros`, `full`, `ones_like`, `zeros_like` |
| Shape and layout | `view`, `reshape`, `flatten`, `unflatten`, `squeeze`, `unsqueeze`, `contiguous`, `clone`, transpose, permute, `.T` |
| Tensor composition | split, split-with-sizes, chunk, concat, stack, fixed dimension slicing with a positive step |
| Data movement | `expand`, `repeat`, `flip`, `roll`, `F.pad` (constant, reflect, replicate), `tril`, `triu`, nearest upsampling, and `gather`, `index_select`, or indexing with indices that don't depend on the input |

Any computation whose inputs are all parameters, buffers, or constants runs
once during export, whatever operations it uses. Masks such as
`self.bias[:, :, :T, :T] == 0` and RoPE tables built with `torch.arange` and
`torch.outer` in `forward` are stored as weights and never run in Scratch.

These operations cover many MLPs and CNNs, along with transformer components
such as multi-head attention, combined QKV projections, rotary position
embeddings, causal masks, pre-norm residual blocks, and SwiGLU gates. Support
depends on the operations the model actually runs, not on its architecture
name. `nn.MultiheadAttention`, `nn.TransformerEncoder`, and nanoGPT- or
Llama-style attention written in plain PyTorch export without changes. A
wrapper around a supported module can still fail, for example when a block
returns a tuple; see [troubleshooting](troubleshooting.md#unpack-module-tuples-in-the-wrapper).

Hugging Face GPT-2, Llama, Qwen2, and BERT models export directly, without a
wrapper. Their `ModelOutput` is unwrapped, and the cache is skipped:

```python
cattorch.transpile(model, (input_ids,), "model.sprite3")
```

On the export copy, cattorch sets `use_cache`, `output_attentions`, and
`output_hidden_states` to `False`, so only the model's real outputs remain
(`logits` first for `*ForCausalLM`). It also switches attention to `sdpa`,
which gives smaller sprites than `eager`, or to `eager` when the model has no
SDPA support. Your model and its config aren't modified, and an explicit
`use_cache=` in `example_kwargs` still wins.

This exports a stateless forward pass over the example's sequence length. It
doesn't add a KV cache. For cached generation, pass the same unwrapped model to
a [`GenerationProgram`](generation.md):

```python
cattorch.transpile(model, cattorch.GenerationProgram(max_context=64),
                   "model.sprite3")
```

cattorch derives the one-token step and its per-layer caches, and checks it
against the model's own `forward` during export. GPT-2, Llama, and Qwen2
causal language models work this way; BERT doesn't, because its attention isn't
causal.

Use `cattorch.rotary_embedding(value, cosine, sine)` for adjacent-pair RoPE.
It accepts broadcastable sine/cosine tables that do not expand `value` and
requires an even final dimension.

Mixture-of-experts layers are supported through the experimental
`ExpertFamily` and `SparseMoE` classes; see
[experimental programs and sparse MoE](programs-and-moe.md#mixture-of-experts).

Operations without their own entry, such as `F.normalize`, `nn.Mish`, or
`var`, export when PyTorch can express them in terms of supported
operations. They may run slower than an operation with a dedicated kernel.

`nn.LSTM` and `nn.GRU` export for the fixed sequence length of the example,
including stacked and bidirectional layers. Every time step is unrolled into
its own blocks, so sprite size grows with sequence length and layer count.
Check `expanded_json_bytes` for longer sequences.

A listed operation can still be rejected for an unsupported shape,
broadcasting pattern, or option. The error names the operation that failed.

## Inputs and outputs

- The first tensor argument is flattened into `input`; later tensor arguments
  use `input_1`, `input_2`, and so on. Tensor values in `example_kwargs`
  follow the positional ones, in dict order.
- The model may return a tensor or a nested tuple, list, dictionary, or
  Hugging Face `ModelOutput` of tensors. Leaves are flattened into `output`,
  `output_1`, and so on in order. Non-tensor leaves are skipped with a
  warning; at least one tensor is required.
- Input shapes are fixed to the shapes of the example values.
- A model with modules in training mode is exported as an evaluation-mode
  copy, with a warning.
- `torch.no_grad()` and `torch.autocast` blocks inside `forward` are ignored.
- Scratch stores every value as a double. Conversions to a floating dtype,
  such as `x.float()` on integer token IDs, keep the value unchanged, including
  conversions to `float16`. Conversions to an integer dtype truncate toward zero,
  and conversions to `bool` produce 0 or 1.

An unsupported operation raises `UnsupportedOperationError`, naming the
operation and the module it came from. Other export problems, such as a
return value without any tensors, raise `UnsupportedModelError`.

Run [`verify`](api-reference.md#verify) whenever you change the model or export
options. If export fails, see [troubleshooting](troubleshooting.md).
