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
| Convolution | `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`, and `nn.ConvTranspose1d/2d/3d`, batched or unbatched, with any stride, padding (including `"same"`), dilation, groups (including depthwise), and output padding. 3D and transposed convolutions are built from the 1D and 2D kernels, so they cost more blocks |
| Pooling | `nn.MaxPool1d/2d/3d`, `nn.AvgPool1d/2d/3d`, `nn.AdaptiveAvgPool1d/2d/3d`, `nn.AdaptiveMaxPool1d/2d/3d`, batched or unbatched, with every option, including `return_indices=True` |
| Linear algebra | `nn.Linear`, `@`, `torch.matmul`; two batched operands must have identical batch dimensions |
| Activations | ReLU, ReLU6, hardtanh, hardswish, sigmoid, tanh, GELU (exact and tanh forms), SiLU, Mish, softplus, leaky ReLU, ELU |
| Normalization | BatchNorm1d/2d (including without running statistics), LayerNorm, RMSNorm, GroupNorm, InstanceNorm, `F.normalize`, `torch.rsqrt` |
| Attention primitives | softmax, log-softmax, embeddings, registered-buffer `masked_fill`, `F.scaled_dot_product_attention` (causal or with a static mask), `cattorch.rotary_embedding` |
| Arithmetic | tensor/scalar `+`, `-`, `*`, `/`, `//`, `%`; unary `-`; `exp`, `exp2`, `log`, `log2`, `log10`, `sqrt`, `rsqrt`, `abs`, `sign`, `floor`, `ceil`, `trunc`, `round`, `reciprocal`, `erf`, `erfc`; trigonometric, inverse trigonometric, and hyperbolic functions, `atan2`, `hypot`; `clamp`; `torch.pow` with any exponent, including a tensor; `logsumexp`, `logaddexp` |
| Reduction | `sum`, `mean`, `prod`, `amax`, `amin`, `max`/`min` (with indices), `argmax`, `argmin`, `any`, `all`, `var`, `std`, `median`, `kthvalue`, over any dimensions; `cumsum`, `cumprod`, `cummax`, `cummin`; `torch.topk`, `torch.sort`, `argsort` |
| Comparison and selection | `==`, `!=`, `<`, `<=`, `>`, `>=`, `~` on comparisons, `torch.where`, `torch.maximum`, `torch.minimum`; results are 0/1 tensors |
| Tensor creation | `arange`, `ones`, `zeros`, `full`, `ones_like`, `zeros_like` |
| Shape and layout | `view`, `reshape`, `flatten`, `unflatten`, `squeeze`, `unsqueeze`, `contiguous`, `clone`, transpose, permute, `.T` |
| Tensor composition | split, split-with-sizes, chunk, concat, stack, fixed dimension slicing with a positive step |
| Data movement | `expand`, `repeat`, `flip`, `roll`, `F.pad` (constant, reflect, replicate), `tril`, `triu`, nearest upsampling; `gather`, `index_select`, `take`, and tensor indexing such as `x[ids]` or `x[:, ids]`; `scatter`, `scatter_add`, `scatter_reduce`, `index_add`, `index_copy`, `index_fill`, `index_reduce`, and `index_put`, with any integer indices, including ones computed from the input; `x[mask] = value` with a boolean mask |
| In-place updates | `x /= y`, `x.copy_(y)`, and slice or index assignment such as `x[..., :4] = y` on a tensor the model created, with step-one slices; updates of inputs, parameters, or a tensor read through an earlier view are rejected |

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

Mixture-of-experts models from transformers, such as Mixtral, Qwen2-MoE,
Qwen3-MoE, OLMoE, and DeepSeek-V3, export the same way, with or without a
`GenerationProgram`. Their experts classes, marked with transformers'
`@use_experts_implementation`, are replaced on the export copy by banked experts
that take their routes from the model's own router, and only the selected experts
run in Scratch. Experts with their own gate function, such as GPT-OSS's
clamped GLU, run that function in Scratch. Experts using operations that
[experts](programs-and-moe.md#mixture-of-experts) don't support, such as a
hand-written normalization, can't export yet. GPT-OSS, with its attention
sinks and sliding-window layers, exports with or without a `GenerationProgram`.

Use `cattorch.rotary_embedding(value, cosine, sine)` for adjacent-pair RoPE.
It accepts broadcastable sine/cosine tables that do not expand `value` and
requires an even final dimension.

Hand-written mixture-of-experts blocks with a `ModuleList` of experts and a
linear router are usually recognized automatically. Others are supported
through the experimental `ExpertFamily` and `SparseMoE` classes, or
`routed_experts` with your own router code; see
[experimental programs and sparse MoE](programs-and-moe.md#mixture-of-experts).

Operations without their own entry, such as `F.normalize`, `nn.Mish`, or
`var`, export when PyTorch can express them in terms of supported
operations. They may run slower than an operation with a dedicated kernel;
`TranspileResult.fallback_operations` and `warnings` list the ones a sprite
uses. [Operator coverage](operator-coverage.md) shows which operators pass
PyTorch's own test samples.

`nn.GRU`, `nn.LSTM` (including `proj_size`) and `nn.RNN` export with stacked
and bidirectional layers, and so do `nn.GRUCell`, `nn.LSTMCell` and
`nn.RNNCell`. A recurrent layer runs as one Scratch loop over time, so the
sprite doesn't grow with the sequence length, and the sequence can vary at
run time (see [variable-length inputs](#variable-length-inputs)). To process
one step per call, write a step method around a cell and keep the hidden
state in a buffer; [`@cattorch.entrypoint`](markers.md) exports it with the
state kept between calls.

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
- Input shapes are fixed to the shapes of the example values, unless one
  input is [variable-length](#variable-length-inputs).
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

## Variable-length inputs

`dynamic=` lets one input's length vary at run time, within bounds:

```python
result = cattorch.transpile(model, torch.randn(4, 16), "model.sprite3",
                            dynamic={0: (1, 32)})
```

`{0: (1, 32)}` lets input 0's first dimension hold 1 to 32 items. Use
`{0: {1: (1, 32)}}` for another dimension, after dimensions of size 1, such
as the tokens in a `(1, tokens)` input; name a tensor in `example_kwargs` by
its key. The Scratch input list then takes any whole number of items, and
the output lists grow with it. `TensorSpec.shape` shows the bounded
dimension as a `BoundedDimension`.

The model must treat the items independently, as most models treat a batch,
or tokens processed one at a time. Recurrent layers are the exception: a
sequence's length can vary, and the final state (`h_n`) is an output of fixed
shape. Other operations that mix items, such as a softmax across them or
`x - x.mean(0)`, are rejected. When a model's
operations can't follow the length directly, for example attention layers
that reorder their batch internally, cattorch runs the whole model once per
item and adds a warning.

cattorch traces the model with one item, then checks the sprite against the
model at several lengths during export. A model that uses the length itself,
such as `torch.arange(tokens.shape[1])` for positions, fails that check.
Language models should use a [`GenerationProgram`](generation.md) instead,
which handles prompts of any length up to `max_context`.

Entry points take the same bounds with `Input(..., dynamic=(1, 32))`, or
`@cattorch.entrypoint(dynamic={"rows": (1, 32)})` in [markers](markers.md),
for an input's first dimension.

Run [`verify`](api-reference.md#verify) whenever you change the model or export
options. If export fails, see [troubleshooting](troubleshooting.md).
