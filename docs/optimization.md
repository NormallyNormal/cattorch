# Optimization modes

cattorch has two public kernel modes. Both generate sequential Scratch code;
they optimize list access, loop structure, repeated work, and materialization
rather than relying on parallel hardware.

## Exact mode

`exact` is the default and preserves cattorch's supported PyTorch semantics
within Scratch's numeric behavior.

```python
artifact = transpile(model, example, "model", optimization="exact")
```

Exact optimizations include export-time weight layout, constant folding,
fused linear epilogues, reduced inner-loop index arithmetic, and four-output
activation reuse for sufficiently large dense linear layers. It also includes:

- a single traversal for `F.silu(gate) * value`;
- one traversal for eligible straight-line, same-shape arithmetic chains;
- evaluation Conv/Linear followed by BatchNorm folding when the graph proves
  the producer-consumer relationship is safe;
- causal-attention specializations that fold scale and masks, skip masked work,
  and avoid unnecessary QKV copies;
- shared token and position embedding traversal for recognized graphs.

Fallback kernels are used when a fusion cannot preserve broadcasting, aliasing,
or shared-intermediate behavior.

## Fast mode

`fast` permits numerical approximation. Operations without a useful proven
approximation continue to use exact kernels.

```python
artifact = transpile(model, example, "model_fast", optimization="fast")
```

Its default approximations are:

- tanh GELU becomes QuickGELU: `x * sigmoid(1.702 * x)`;
- LayerNorm variance uses one pass: `E[x²] - E[x]²`;
- softmax omits max subtraction and reuses exponentials.

Fast softmax can overflow on large positive logits. One-pass LayerNorm can lose
precision when values share a large offset. Sigmoid, tanh, SiLU, and ELU remain
exact because their tested approximations were not faster in the Scratch VM.

Approximation families can be disabled independently:

```python
from cattorch import FastConfig

config = FastConfig(activations=True, layer_norm=False, softmax=False)
transpile(model, example, "model_fast", optimization="fast", fast_config=config)
```

## Weight transforms

Low-rank factorization and structured pruning are explicit fast-mode options.
They are disabled by default because quality depends on the trained model.

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

`rank` selects a truncated-SVD rank; `rank_ratio` selects a fraction of the
matrix's full rank. Factorization is skipped unless its two factors reduce
multiply-accumulates. Linear pruning removes contiguous groups of four input
weights, while convolution pruning removes complete input-channel kernels, so
the generated loop can skip work. `neuron_pruning` removes complete hidden
units only from graph-proven `Linear -> activation -> Linear` MLP paths.

An `overrides` entry replaces the global policy for that named module. Explicit
`x @ static_parameter` operations use the global weight policy; dynamic
matmuls, including attention-score products, stay exact.

Weight transformations operate on an export-only model copy. They do not
mutate the supplied model.

## Measuring the tradeoff

Use `verify` with tolerances that reflect the application to measure numerical
change. Use a generated benchmark project in real Scratch to measure speed;
the Python emulator is a correctness tool and its timing is not a browser
performance proxy. See [verification and benchmarking](verification-and-benchmarking.md).
