# Optimization modes

[Documentation home](index.md)

cattorch defaults to `exact` mode. Use `fast` to trade some numerical accuracy
for speed, then verify the output and benchmark in your target Scratch client.

## Exact mode

`exact` is the default. It computes the same results as PyTorch, up to
Scratch's floating-point behavior.

"Exact" only describes the kernels. Float16 or integer
[storage](storage.md) and `sig_figs` still change the weights, so an exact-mode
export with those options is not lossless.

```python
from cattorch import transpile

artifact = transpile(model, example, "model", optimization="exact")
```

In both modes, the exporter combines compatible operations and removes
redundant work automatically.

## Fast mode

`fast` allows approximations that run faster in Scratch. Operations without a
faster approximation use the exact kernel.

```python
artifact = transpile(model, example, "model_fast", optimization="fast")
```

By default, fast mode:

- Replaces GELU, exact or tanh, with QuickGELU, `x * sigmoid(1.702 * x)`.
- Computes LayerNorm variance in one pass as `E[x²] - E[x]²`.
- Skips subtracting the maximum in softmax and reuses exponentials.

QuickGELU differs from GELU by up to about 0.02 per activation, which
accumulates through the layers: a small language model whose logits reach
about 18 can be off by about 0.05, all of it from this replacement. Turn
`activations` off if that matters. Fast softmax can overflow on large positive logits. One-pass LayerNorm can lose
precision when all values share a large offset. Sigmoid, tanh, SiLU, and ELU
stay exact.

Each approximation can be turned off separately:

```python
from cattorch import FastConfig, transpile

config = FastConfig(activations=True, layer_norm=False, softmax=False)
transpile(model, example, "model_fast", optimization="fast", fast_config=config)
```

## Weight transforms

Low-rank factorization and structured pruning are fast-mode options that are
off by default, because how much accuracy they cost depends on the model.

```python
from cattorch import FastConfig, FastLayerConfig, transpile

config = FastConfig(
    layers=FastLayerConfig(rank_ratio=0.5, block_pruning=0.25),
    neuron_pruning=0.25,
    overrides={
        "head": FastLayerConfig(),
        "blocks.0.mlp.up": FastLayerConfig(rank=16),
    },
)
transpile(model, example, "model_fast", optimization="fast", fast_config=config)
```

`rank` sets a truncated-SVD rank, and `rank_ratio` sets it as a fraction of
the matrix's full rank. A matrix is only factored if the two smaller matrices
need fewer multiplications than the original.

`block_pruning` removes the smallest-magnitude weights in whole groups so the
generated loops can skip them: groups of four input weights in linear layers
and `x @ weight` products, and whole input-channel kernels in ungrouped 1D and
2D convolutions. `neuron_pruning` removes
entire hidden units, but only where cattorch can confirm a
`Linear -> activation -> Linear` path, and hidden channels in
`StackedSwiGLUMoE` experts.

An `overrides` entry replaces the whole policy for the named module. A
`x @ weight` product with a stored weight uses the global policy. Products of
two runtime tensors, such as attention scores, stay exact.

A low-rank factorization puts two plain layers in place of the module, so
it applies only where the module does nothing but use its weight: a module
whose class has its own `forward` (a subclass of `nn.Linear` with extra
steps), one with forward hooks, and one inside a `torch.nn` module that may
read its parameters itself, such as `out_proj` of `nn.MultiheadAttention` or
the layers of `nn.TransformerEncoderLayer`, keep their weights, and the
export warns. Block pruning, which keeps the module, still applies to them.
Neuron pruning, which also puts new layers in place, skips a pair of layers
with forward hooks the same way.

A transform only changes the module it applies to. When the model also
reads that module's parameters directly, outside the module, such as
`x @ self.up.weight.T` (in any method a program exports), the module is left
untransformed, so those reads keep the original values, and the export warns
which modules were left. If `torch.fx` can't trace the model, that can't be
checked, and the export warns that a direct read of a block-pruned weight
would read the pruned values.

Transforms are applied to a copy; your model is not modified.

## Measuring the tradeoff

Measure the accuracy change with `verify`, using tolerances that suit your
application. Measure speed with a benchmark project run in Scratch; the Python
emulator's timings don't reflect browser performance. See
[verification and benchmarking](verification-and-benchmarking.md).

Related: [`FastConfig` and `FastLayerConfig` fields](api-reference.md#fastconfig).
