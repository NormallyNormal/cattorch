# Supported models and operations

cattorch exports one fixed-shape inference graph with one tensor output. It is
built on `torch.export`, so graph construction must not depend on tensor data.
Parameters, buffers, and constants are embedded in the sprite; tensor arguments
become Scratch input lists.

## Operations

| Category | Operations |
|---|---|
| Convolution | Batched `nn.Conv1d`, `nn.Conv2d`, with optional bias, stride, and padding; dilation and groups must be 1 |
| Pooling | Batched `nn.MaxPool1d/2d`, `nn.AvgPool1d/2d`, `nn.AdaptiveAvgPool2d`; non-default dilation, ceiling, padding-count, and divisor options are rejected |
| Linear algebra | `nn.Linear`, `@`, `torch.matmul`; two batched operands must have identical batch dimensions |
| Activations | ReLU, sigmoid, tanh, GELU (tanh form), SiLU, leaky ReLU, ELU |
| Normalization | BatchNorm1d/2d, LayerNorm, RMSNorm, `torch.rsqrt` |
| Attention primitives | softmax, embeddings, registered-buffer `masked_fill` |
| Arithmetic | tensor/scalar `+`, `-`, `*`; scalar `/`; unary `-`; `torch.pow` with exponent 0 or 2 |
| Reduction | `torch.mean` along one dimension or consecutive dimensions |
| Tensor creation | `arange`, `ones`, `zeros`, `full`, `ones_like`, `zeros_like` |
| Shape and layout | `view`, `reshape`, `flatten`, `contiguous`, `clone`, transpose, permute, `.T` |
| Tensor composition | split, split-with-sizes, chunk, concat, dimension slicing with step 1 |

These operations cover many MLPs and CNNs, along with transformer components
such as multi-head attention, combined QKV projections, rotary position
embeddings, causal masks, pre-norm residual blocks, and SwiGLU gates. Support is
defined by the exported operation graph, not by a model family name: wrappers
around otherwise compatible modules may still need adjustment.

RNNs and multiple tensor outputs are not currently supported.

## Interface contract

- The first tensor argument is flattened into `input`; later tensor arguments
  use `input_1`, `input_2`, and so on.
- The model must return one tensor, flattened into `output`.
- Input shapes are specialized from the example values supplied at export.
- Training-only semantics are not silently removed. For example, training-mode
  dropout is rejected; call `model.eval()` before exporting.
- Dtype conversions that would change values are rejected when no faithful
  Scratch implementation exists.

Unsupported operations raise `UnsupportedOperationError` with the exported
operation, node, and module context when available. Unsupported graph-level
contracts raise `UnsupportedModelError`.
