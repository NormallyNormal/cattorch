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

- Replaces tanh GELU with QuickGELU, `x * sigmoid(1.702 * x)`.
- Computes LayerNorm variance in one pass as `E[x²] - E[x]²`.
- Skips subtracting the maximum in softmax and reuses exponentials.

Fast softmax can overflow on large positive logits. One-pass LayerNorm can lose
precision when all values share a large offset. Sigmoid, tanh, SiLU, and ELU
stay exact.

Each approximation can be turned off separately:

```python
from cattorch import FastConfig

config = FastConfig(activations=True, layer_norm=False, softmax=False)
transpile(model, example, "model_fast", optimization="fast", fast_config=config)
```

## Weight transforms

Low-rank factorization and structured pruning are fast-mode options that are
off by default, because how much accuracy they cost depends on the model.

```python
from cattorch import FastConfig, FastLayerConfig, transpile

config = FastConfig(
    weights=FastLayerConfig(rank_ratio=0.5, pruning=0.25),
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

`pruning` removes the smallest-magnitude weights in whole groups so the
generated loops can skip them: groups of four input weights in linear layers,
and whole input-channel kernels in convolutions. `neuron_pruning` removes
entire hidden units, but only where cattorch can confirm a
`Linear -> activation -> Linear` path, and hidden channels in
`StackedSwiGLUMoE` experts.

An `overrides` entry replaces the whole policy for the named module. A
`x @ weight` product with a stored weight uses the global policy. Products of
two runtime tensors, such as attention scores, stay exact.

Transforms are applied to a copy; your model is not modified.

## Measuring the tradeoff

Measure the accuracy change with `verify`, using tolerances that suit your
application. Measure speed with a benchmark project run in Scratch; the Python
emulator's timings don't reflect browser performance. See
[verification and benchmarking](verification-and-benchmarking.md).

Related: [`FastConfig` and `FastLayerConfig` fields](api-reference.md#fastconfig).
