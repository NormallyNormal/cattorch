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
| Pooling | Batched `nn.MaxPool1d/2d`, `nn.AvgPool1d/2d`, `nn.AdaptiveAvgPool2d`; non-default dilation, ceiling, padding-count, and divisor options are rejected |
| Linear algebra | `nn.Linear`, `@`, `torch.matmul`; two batched operands must have identical batch dimensions |
| Activations | ReLU, sigmoid, tanh, GELU (tanh form), SiLU, leaky ReLU, ELU |
| Normalization | BatchNorm1d/2d, LayerNorm, RMSNorm, `torch.rsqrt` |
| Attention primitives | softmax, embeddings, registered-buffer `masked_fill`, `cattorch.rotary_embedding` |
| Arithmetic | tensor/scalar `+`, `-`, `*`; scalar `/`; unary `-`; `torch.pow` with exponent 0 or 2 |
| Reduction | `torch.mean` along one dimension or consecutive dimensions |
| Tensor creation | `arange`, `ones`, `zeros`, `full`, `ones_like`, `zeros_like` |
| Shape and layout | `view`, `reshape`, `flatten`, `contiguous`, `clone`, transpose, permute, `.T` |
| Tensor composition | split, split-with-sizes, chunk, concat, fixed dimension slicing with a positive step |

These operations cover many MLPs and CNNs, along with transformer components
such as multi-head attention, combined QKV projections, rotary position
embeddings, causal masks, pre-norm residual blocks, and SwiGLU gates. Support
depends on the operations the model actually runs, not on its architecture
name. A wrapper around a supported module can still fail, for example when a
block returns a tuple; see [troubleshooting](troubleshooting.md#unpack-module-tuples-in-the-wrapper).

Use `cattorch.rotary_embedding(value, cosine, sine)` for adjacent-pair RoPE.
It accepts broadcastable sine/cosine tables that do not expand `value` and
requires an even final dimension.

Mixture-of-experts layers are supported through the experimental
`ExpertFamily` and `SparseMoE` classes; see
[experimental programs and sparse MoE](programs-and-moe.md#mixture-of-experts).

RNNs are not supported.

A listed operation can still be rejected for an unsupported shape,
broadcasting pattern, or option. The error names the operation that failed.

## Inputs and outputs

- The first tensor argument is flattened into `input`; later tensor arguments
  use `input_1`, `input_2`, and so on.
- The model may return a tensor or a nested tuple, list, or dictionary of
  tensors. Leaves are flattened into `output`, `output_1`, and so on in
  PyTorch pytree order; non-tensor leaves are rejected.
- Input shapes are fixed to the shapes of the example values.
- Training-mode behavior is rejected rather than dropped. For example,
  training-mode dropout raises an error, so call `model.eval()` first.
- Casting a runtime tensor to a different dtype, such as `x.float()` on an
  integer input, is rejected.

An unsupported operation raises `UnsupportedOperationError`, naming the
operation and the module it came from. Other export problems, such as a
non-tensor return value, raise `UnsupportedModelError`.

Run [`verify`](api-reference.md#verify) whenever you change the model or export
options. If export fails, see [troubleshooting](troubleshooting.md).
