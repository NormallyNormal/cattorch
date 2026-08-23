# MythicGPT 4 Pro Scratch Transformer Analysis

Analyzed artifact:

`benchmarks/artifacts/MythicGPT 4 Pro - Real AI in Scratch.sb3`

This is a stock Scratch 3 project, not a TurboWarp project. The SB3 archive is
1,454,529 bytes and its expanded `project.json` is 3,174,555 bytes. The model
fits because its weights are stored in compact strings and decoded once at
startup into ordinary Scratch lists.

## Architecture

- Vocabulary: 3,337
- Residual/embedding width: 128
- Learned context length: 192
- Active transformer layers: 4
- Attention: 2 query heads, head width 64, full multi-head K/V
- FFN width: 384 with ReLU
- Learned token and position embeddings
- Token embedding is reused as the output projection
- Affine LayerNorm, including biases in the dense projections
- Incremental single-token generation with per-layer K/V caches
- Greedy output selection, with the largest/second-largest margin retained

The active network contains 1,113,472 meaningful parameters:

```text
token embedding              3337 x 128 = 427,136
position embedding            192 x 128 =  24,576
four transformer layers    4 x 165,376 = 661,504
final LayerNorm                              256
                                           -------
                                         1,113,472
```

The project actually packs and initializes five transformer layers, for
1,278,848 meaningful parameters in total, but `fw` calls only `l0` through
`l3`. The `l4` procedure has no call site. After a diagnostic forward pass,
`K0` through `K3` and `V0` through `V3` each contained 128 values while
`K4`/`V4` remained empty. The dead fifth layer accounts for 165,376 parameters,
12.9% of the packed model, and 220,512 encoded characters.

## Weight codec

The Stage list `PK` contains 67 string items totaling 1,726,993 characters.
One item is the 21,801-character vocabulary; the others contain weights. The
longest weight strings are 259,928 characters, which is legal because Scratch's
list limit concerns item count, not the length of a string item.

The codec uses a 66-character JSON-safe alphabet. Every group of four base-66
characters is converted to an integer, then split into three base-255 digits.
Subtracting 127 gives three signed values in the range -127 through 127:

```text
4 encoded characters -> 3 signed int8-like weights
1 weight             -> 1.333 encoded characters
```

This is half the text size of cattorch's Base64 float16 encoding and only 33%
larger than Mochi's one-hex-character int4 encoding, while retaining much more
precision. The decoder runs once and clears `PK` afterward.

Crucially, MythicGPT does not dequantize every weight to a float. A decoded
list retains the small integer values. A dense kernel accumulates
`activation * integer_weight`, then applies the tensor's hard-coded scale once
per output and adds its correspondingly scaled bias. This makes quantization
primarily a storage technique without introducing parsing work in the MAC
loop. The per-tensor scale is less accurate than row-wise quantization would
be, so this remains a lossy fast/storage mode rather than an exact codec.

All decoded weight lists are kept below 200,000 items. The tied embedding/head
is split into `194,944 + 194,944 + 37,248` meaningful values, corresponding to
1,523, 1,523, and 291 vocabulary rows. Some decoded lists have one or two
padding values because the codec emits groups of three.

## Runtime structure

Every compute procedure is a warp custom block. `fw` embeds one token and
runs the four active transformer layers, but deliberately omits final
normalization and vocabulary projection. Prompt/history tokens therefore use
this hidden-only path. During generation, `pick` performs final LayerNorm,
the tied 3,337-way vocabulary projection, and greedy selection; only then does
`fw` advance the model with the chosen token.

This is the same important prefill split now used by cattorch: never evaluate
the vocabulary head for prompt tokens whose logits will be discarded.

The tokenizer is a custom word-level dictionary with special handling for
numbers, punctuation, and unknown slots. The surrounding application also has
confidence-margin checks, canned fallback responses, and special math
postprocessing. The transformer inference is real, but not every visible
response should be interpreted as unconstrained output from the network.

The attention softmax uses a 321-entry lookup table for `exp(x)`, covering
`[-16, 0]` at increments of 0.05. Values below the range become zero and the
maximum becomes one. This is an approximate fast-mode candidate, distinct
from caching exact Scratch exponent reporters.

Dense projections compute four output neurons together. One input list read
is stored in a variable and reused by four multiply-accumulates, with four
running weight indices and accumulators. This is more aggressive than the
two-output input-read-sharing form previously screened in cattorch and should
be benchmarked independently rather than assumed faster.

## Official Scratch VM diagnostic

`benchmarks/run_mythic_forward_vm.cjs` invokes the existing `ld`, `fw`, and
`pick` warp procedures directly. On one diagnostic run in turbo mode:

| Phase | Time |
| --- | ---: |
| compact weight initialization | 2.790 s |
| hidden-only forward, context 1 | 0.396 s |
| final norm, output head, and pick | 0.239 s |
| complete generated-token step | 0.635 s |

A second complete run measured 0.658 seconds, so the figures are screening
measurements rather than stable browser benchmarks. The output head consumes
roughly 38% of the context-one token step. The official VM is useful for
prioritization, but Scratch in the browser remains the final performance
authority.

For comparison, cattorch's approximately 998K-parameter cached-MQA CatGPT2
export measured about 0.618 seconds at context one in the same VM. MythicGPT's
1.113M active model is therefore in the same rough performance class despite
using full MHA, but the architectures and output behavior are not identical
enough to treat this as a controlled kernel comparison.

## Ideas worth testing in cattorch

1. Add a packed signed-int8 storage/fast mode using the four-base66-to-three-
   value codec. Keep decoded integers in lists and apply a scale after each
   output dot product. Start with row-wise scales for accuracy, then compare
   their extra list access against per-tensor scales.
2. Benchmark four-output-at-once linear kernels at cattorch's common widths.
   The candidate must be compared with the current row-base kernel in both the
   official VM and browser Scratch, with exact output hashes where applicable.
3. Benchmark the clipped 0.05-step exponential lookup for attention softmax in
   fast mode. Measure both latency across realistic context lengths and error
   against the exact softmax, since a list read may or may not beat Scratch's
   exponent reporter.
4. Retain hidden-only prefill, tied embedding/output weights, physical list
   sharding, and single-token KV caching; cattorch already implements these
   structural wins.
5. Do not copy full-width K/V, affine LayerNorm, ReLU, or per-tensor
   quantization merely because this project uses them. Cattorch's MQA and
   architecture-specific normalization/activation choices should be evaluated
   on model quality as well as Scratch runtime.

The clearest immediate experiments are the four-output linear kernel and the
lookup-table softmax. The int8 codec is the largest project-size opportunity,
but it needs an accuracy study before becoming a recommended fast export mode.

## Focused kernel screens

`benchmarks/build_mythic_kernel_suite.py` generates a single browser-ready
project containing 27 sequential timings. The VM values below summarize
several initial runs; browser Scratch remains the final authority.

| Screen | Representative VM result | Conclusion |
| --- | ---: | --- |
| current linear, 128 -> 384, 20 calls | 0.454 s | baseline |
| Mythic four-output linear | 0.413 s | about 9% faster |
| unrolled four-output linear | 0.409 s | about 10% faster |
| current linear, 384 -> 128, 20 calls | 0.447 s | baseline |
| Mythic four-output linear | 0.400 s | about 11% faster |
| unrolled four-output linear | 0.397 s | about 11% faster |
| current exact softmax, 128, 3,000 calls | 0.640 s | baseline |
| exact store-then-normalize softmax | 0.753 s | slower despite one exponent pass |
| Mythic lookup softmax | 1.072 s | about 68% slower and approximate |
| cattorch Base64 float16 decode, 49,152 values x5 | 1.681 s | higher-precision baseline |
| Mythic base66 int8 decode, 49,152 values x5 | 0.473 s | 3.55x faster, but lossy |

All four dense implementations produced bit-identical outputs at 128 -> 128,
128 -> 384, and 384 -> 128. The 128 -> 128 timing was noisy, ranging from
neutral to a useful improvement, so it needs the browser result before a
width policy is chosen. Integer-list weights followed by one output-scale
multiplication had only floating-operation-order differences from a list of
the equivalent scaled floats (maximum absolute difference `3.8e-15`).

The lookup softmax retained the same argmax on these screens, with maximum
probability error between 0.00139 and 0.00171 and total-variation distance
between 0.00243 and 0.00611. Since it was also substantially slower, there is
no reason to adopt it in either exact or fast mode based on this implementation.

The base66 result is normalized to the same number of decoded values, but it
is not a same-precision codec comparison: signed int8 needs fewer input
characters and avoids IEEE reconstruction. It confirms that int8 would make
startup markedly cheaper in addition to shrinking the project. Model-level
accuracy remains the deciding test.

### Browser Scratch results

The suite was run in turbo mode on `scratch.mit.edu` and saved as
`benchmarks/artifacts/mythic_kernel_suite_benched.sb3`. All 27 cases completed,
and every equivalence check passed.

| Screen | Browser time | Change from current append |
| --- | ---: | ---: |
| current append, 128 -> 128, 60 calls | 0.616 s | baseline |
| current replace, 128 -> 128 | 0.576 s | 6.5% faster |
| Mythic four-output | 0.505 s | 18.0% faster |
| unrolled four-output | 0.512 s | 16.9% faster |
| current append, 128 -> 384, 20 calls | 0.479 s | baseline |
| current replace, 128 -> 384 | 0.478 s | neutral |
| Mythic four-output | 0.434 s | 9.4% faster |
| unrolled four-output | 0.439 s | 8.4% faster |
| current append, 384 -> 128, 20 calls | 0.465 s | baseline |
| current replace, 384 -> 128 | 0.470 s | 1.1% slower |
| Mythic four-output | 0.450 s | 3.2% faster |
| unrolled four-output | 0.428 s | 8.0% faster |

The browser suggests a simple specialization: use Mythic's ordinary inner
repeat when the dot width is 128, and its four-way-unrolled form when the dot
width is 384. Relative to the matching current-replace control, activation
reuse itself saves 9.2% at 128 -> 384, 12.3% at 128 -> 128, and 8.9% for the
unrolled 384 -> 128 case. The result is therefore not merely an artifact of
preallocating the destination list. All corresponding output lists were
bit-identical.

| Softmax screen | Exact | Stored exact | Mythic lookup |
| --- | ---: | ---: | ---: |
| context 32 | 0.964 s | 1.069 s | 1.471 s |
| context 128 | 0.929 s | 1.070 s | 1.450 s |
| context 192 | 0.973 s | 1.027 s | 1.430 s |

The lookup version is 47.0% to 56.1% slower than cattorch exact softmax. The
exact store-then-normalize control is also slower, showing that repeated list
mutation/access costs more than recomputing Scratch's exponential reporter.
The lookup-table path should be rejected.

For 49,152 decoded values repeated five times, Base64 float16 initialization
took 1.818 seconds and Mythic base66 int8 took 0.544 seconds. Base66 int8 is
3.34x faster and decoded all 49,152 test values exactly. This supports the
codec as a lossy storage mode, but says nothing by itself about model accuracy.

The integer-weight current kernel took 0.483 seconds versus 0.462 seconds for
the equivalent scaled-float kernel, so small integer values do not inherently
make Scratch multiplication faster. The best integer four-output variant took
0.442 seconds; that improvement comes from the four-output loop, not integer
arithmetic.

### Append versus preallocated replace

The follow-up project `mythic_append_suite_benched.sb3` directly tested
cattorch-compatible clear-and-append output against Mythic's persistent
replace-in-place output:

| Shape and implementation | Browser time | Change from current append |
| --- | ---: | ---: |
| 128 -> 128 current append | 0.671 s | baseline |
| 128 -> 128 four-output append | 0.672 s | neutral |
| 128 -> 384 current append | 0.556 s | baseline |
| 128 -> 384 four-output append | 0.515 s | 7.4% faster |
| 128 -> 384 four-output replace | 0.558 s | 0.4% slower |
| 384 -> 128 current append | 0.546 s | baseline |
| 384 -> 128 four-output append | 0.522 s | 4.4% faster |
| 384 -> 128 unrolled four-output append | 0.532 s | 2.6% faster |
| 384 -> 128 unrolled four-output replace | 0.520 s | 4.8% faster |

All fourteen output lists were bit-identical. Four-output append can therefore
preserve cattorch's existing temporary-list lifecycle, avoiding initialization
and logical-length complications from persistent buffers. The focused result
also refines the policy: keep the current kernel for the smaller 128 -> 128
matrix, and use the faithful four-output append loop for the two 49,152-MAC
transformer matrices. A conservative generic eligibility threshold is a
multiple-of-four output width and at least roughly 49,152 MACs per input row;
whole-network testing should follow any production integration.

### Production integration

The four-output clear-and-append loop is now the production dense
`aten.linear` path when output width is divisible by four and the matrix has at
least 49,152 MACs per input row. Smaller, remainder, and sparse matrices retain
the preceding row-base/sparse kernels. Bias, multiple input rows, residual
addition, scalar arithmetic, and fused activations preserve their existing
order; the grouped correctness screen is bit-identical to the prior kernel.

The upload-safe whole-transformer VM screen measured 1.907 seconds for five
previous-exact forwards and 1.614 seconds grouped, a 15.4% time reduction with
identical 1,028-value output hashes. The real 998K CatGPT2 diagnostic measured
prefill32 at 8.169 versus 7.581 seconds and three cached decodes at 2.148 versus
1.768 seconds. The browser-ready artifact is
`benchmarks/artifacts/grouped_linear_whole_suite.sb3`.

The completed browser Scratch run measured 1.973 seconds previous exact versus
1.901 seconds grouped exact over five forwards, a 3.65% reduction. The full
1,028-item outputs were byte-for-byte identical. This measurement represents
one Firefox/Linux/browser-and-hardware combination. The official VM provides a
more controlled, reproducible comparison, while browser runs sample real
deployments. The difference between their measured gains means neither should
be treated as a universal speedup figure.
