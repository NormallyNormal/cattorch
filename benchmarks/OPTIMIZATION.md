# Scratch optimization audit

This is a dated log of kernel optimizations for contributors: what was kept,
what was rejected, and the measurements behind each decision. Candidates are
timed in the official Scratch VM and kept only if their output is unchanged.
The VM gives repeatable comparisons; browser runs show the effect of real
browsers, JITs, operating systems, and hardware. The Python emulator is used
for correctness and opcode counts, not timing.

Terms used below:

- **MAC**: one multiply-accumulate, `sum += weight * value`, the unit of work
  in a dot product.
- **Epilogue**: the operations applied right after a kernel's main loop, such
  as adding a bias, a residual, or an activation.
- **Unrolling**: repeating a loop body N times per iteration ("four-way"
  unrolling does four elements per pass).

## 2026-08-24 inference red-team follow-up

The semantic pairwise RoPE production lowering measured 0.160 seconds versus
1.675 seconds for the equivalent dense signed-permutation expression over 160
CatGPT-sized applications (96 rows by 14 features), a 10.5x speedup in the
official VM.

At the CatGPT3 expert geometry (8 experts, width 112, hidden width 192), the
grouped top-1 stacked-SwiGLU production kernel measured 0.544 seconds for 20
forwards. That is about 34% below the prior 0.0413-second-per-forward baseline;
exact and fast controls produced identical output hashes when approximate
pruning was disabled.

The compact top-k selector now compares each candidate with the retained kth
value before entering its insertion scan. At k=16 and 1,024 logits it reached
the VM timer floor (0.032 seconds for 20 selections), matching the much larger
unrolled selector while preserving identical value and ID hashes.

## Retained exact-mode strategies

- Linear and matrix multiply unroll dot products four-way. Contiguous dense
  dots index the right operand as a row offset plus the running left index,
  saving one variable write per MAC. Strided matrix products keep two running
  indices.
- Linear starts the accumulator at the bias and fuses simple scalar, residual,
  and activation epilogues.
- Dense Linear matrices whose output width is divisible by four and that have
  at least 49,152 MACs per input row use the grouped path: one activation read
  feeds four output accumulators. It still clears and appends its output and
  fuses the same epilogues. Its weights are interleaved at export time, so one
  running weight index replaces four. Oversized matrices are only split between
  complete output rows, or complete four-row groups on the grouped path, so
  shard selection stays outside every dot product.
- Elementwise kernels use one-based indices and eliminate modulo for
  equal-sized tensors. Reused exponentials and GELU intermediates are cached;
  cheap values are recomputed when Scratch variable mutation costs more.
  Lightweight sequential loops are partially unrolled eight ways. Heavier
  math remains four-way, except GELU where a longer VM A/B run showed that the
  larger block body lost performance. Safe one-based loops use Scratch's
  official palette-hidden `for each` primitive.
- Softmax skips the redundant first maximum comparison and appends directly
  for contiguous reduction groups. Contiguous groups carry a running start
  rather than consulting an index list. Caching every exponential was
  rejected: native exponentiation was cheaper than extra list writes.
- Mean and normalization kernels append sequentially, avoid output
  preallocation, and cache channel constants. BatchNorm folds
  `sqrt(variance + eps)` into static data.
- Embedding computes a weight-row base once per token. Transpose appends in
  output order using an export-time inverse map. Slice and concatenation use
  four-way partially unrolled copies.
- Convolution and pooling use compact export-time receptive-field maps shared
  across batches and channels. Convolution specializes full interior kernels
  with sequential weight indices and carries batch/weight starts between outer
  iterations. Uniform pooling windows stream through one compact offset map;
  padded and nonuniform adaptive windows retain their start/count maps. This
  avoids runtime coordinate arithmetic and padding branches without storing a
  full per-MAC schedule.
- Pure static tensor subgraphs are evaluated at export time. Shape-only flat
  data operations remain aliases and emit no Scratch work.
- Canonical same-input SwiGLU fuses both Linear projections, SiLU, and the
  multiply, sharing activation reads and avoiding both projection lists.
  Oversized projection weights use aligned hidden-row shards.
- Straight-line same-shape arithmetic expressions are evaluated in one loop,
  eliminating each intermediate list traversal without changing arithmetic.
- Standard causal attention folds QK scaling and the `-inf` mask, skips the
  upper-triangular score/softmax/value work, and consumes K before its final
  transpose. Longer sequences write combined QKV directly in head-major form.
- Token and positional embedding lookup/add is one traversal for batch-one
  decoder graphs, and singleton-dimension transposes are aliases.
- Optional generation exports emit K/V state directly from canonical QKV
  projections and compute only the new token. Transformer-sized QKV weights
  use the same four-output interleaving, split along Q/K/V row boundaries when
  necessary. Hidden prompt tokens stop after
  the final layer's K/V emission, skipping its remaining attention, output
  projection, MLP, final normalization, and vocabulary head. Exact cached
  attention with `max_context >= 64` stores stable exponentials once and
  consumes them directly in probability-times-V. Prefill, decode, reset,
  context guards, and cache length remain stock Scratch custom blocks.
- Fixed-small generation top-k maintains a sorted candidate list in one logits
  pass. At `k=5` this was 3.15x faster in browser Scratch than five destructive
  maximum scans, with exactly equal selected values and IDs.
- Eval Conv/Linear + BatchNorm pairs are folded on an export-only model copy.

## Rejected or neutral candidates

- Runtime quantization/unpacking: it adds parsing work and is useful only when
  project/list compression outweighs the one-time initialization cost.
- Full exponential caching in softmax: slower because of list mutation.
- Caching cheap values in square, ReLU, leaky-ReLU, ELU, LayerNorm variance,
  and RMSNorm squares: neutral or slower than direct list reads in longer VM
  runs. In Scratch, an extra variable assignment is not a free cache.
- Full convolution MAC schedules: faster indexing is possible, but duplicating
  indices or weights for every output would spend too much of Scratch's project
  and list-size budgets. Shared spatial maps keep most of the speedup compactly.
- Loop unroll factors above four for arithmetic-heavy dots: larger generated
  programs did not improve those matrix kernels consistently. A later vanilla
  Scratch test settled on eight-way unrolling for lightweight list loops.
- Unrolling the outer embedding-token loop: slower even though unrolling the
  inner contiguous copy remains a clear win.
- Static-RHS output-major matmul was about 1.8% slower than the existing
  strided kernel in the official VM. Sharing a left read across only two
  output accumulators was about 3.6% slower because of extra variable writes;
  four-output sharing amortizes that cost and is retained for large Linear
  layers.

## Browser Scratch checkpoint

`artifacts/full_suite_benched.sb3` was run in turbo mode on scratch.mit.edu on
2026-08-20/21. Each row contains 100 forwards, with initialization before the
timer reset. Representative results were:

| Case | Legacy | Exact | Speedup |
|---|---:|---:|---:|
| Conv2d | 20.764 s | 5.460 s | 3.80x |
| Conv1d | 1.947 s | 0.696 s | 2.80x |
| batched matmul | 3.320 s | 1.897 s | 1.75x |
| embedding | 0.663 s | 0.463 s | 1.43x |
| softmax | 0.298 s | 0.232 s | 1.28x |
| max pool 2d | 0.397 s | 0.133 s | 2.98x |
| average pool 2d | 0.364 s | 0.099 s | 3.68x |
| Quick GPT | 3.099 s | 1.678 s | 1.85x |
| CatGPT1 | 34.498 s | 22.176 s | 1.56x |

The browser timer advanced in roughly 0.033-second steps during this run.
Consequently, individual results below about 0.2 seconds are treated as
resolution-limited rather than evidence for small regressions or wins.

## Official VM tests after the browser run

Longer focused runs were used to test the final small changes. At 1,000
forwards, the uniform pooling kernels measured 0.208 s (max1d), 0.785 s
(max2d), 0.160 s (avg1d), 0.641 s (avg2d), and 0.528 s (adaptive avg2d),
roughly another 18–50% below the preceding exact pooling kernel. Carrying
outer convolution starts produced 0.529 s for Conv1d and 4.447 s for Conv2d
at 100 forwards, about another 7–8% reduction. These controlled VM results and
the saved browser sample above should both be retained when judging portability.

The same VM measured a 1,024-element, three-operation arithmetic chain at
0.192 s materialized versus 0.064 s fused over 100 forwards (3.0x). The case
is included in the regenerated comprehensive suite for browser confirmation.

The grouped-linear whole-transformer suite measured 1.907 s for five
previous-exact forwards and 1.614 s for grouped exact, a 15.4% reduction with
identical output hashes. An oversized diagnostic pair using the real 998K
CatGPT2 checkpoint measured hidden prefill32 at 8.169 s versus 7.581 s and
three cached decodes at 2.148 s versus 1.768 s. Its expanded JSON is 13.2 MB,
so the 3.44 MB upload-safe transformer pair remains the browser test candidate.
That Firefox/Linux browser run measured 1.973 s previous exact versus 1.901 s
grouped exact over five forwards, a 3.65% reduction. Both 1,028-item outputs
had the same numeric sum and serialized SHA-256
(`7d8d1c328991d8ab527ec5f0f783b640ea1cfceca29b417e02039038f992e002`).
The browser result confirms the direction and exactness on one real Scratch
deployment, but it is specific to that browser, operating system, hardware,
and run. The official VM is the more controlled and reproducible comparison;
the different improvement magnitudes show that neither result should be
assumed universal. The 0.072 s browser difference is only a little over two
observed Scratch timer ticks, so its exact percentage should also be treated as
approximate.

After interleaved grouped weights and the general loop lowering were promoted,
the controlled whole-transformer suite measured 1.825 s for the previous path
and 1.500 s for production exact over five forwards, a 17.8% reduction. This
newer production comparison still needs a scratch.mit.edu sample.

## Fast-mode tests

Fast mode retains only arithmetic substitutions that won focused official-VM
tests. At 1,000 forwards, QuickGELU was 2.22x faster than exact tanh GELU;
unstabilized softmax and one-pass LayerNorm were each about 1.19x faster.
Candidate approximations for sigmoid, tanh, SiLU, and ELU were slower and were
therefore rejected.

Structured weight transforms showed larger opportunities in representative
100-forward tests:

| Transform | Linear | Conv1d | Conv2d |
|---|---:|---:|---:|
| 50% structured pruning | 1.37x | 1.33x | 1.81x |
| `rank_ratio=0.25` | 2.53x | 1.66x | 2.74x |

Pruned Linear kernels skip four-wide input blocks; pruned convolution kernels
skip whole input-channel kernels. Low-rank layers are decomposed at export time
into two smaller dense layers. No compressed values are decoded at runtime.

The generated `artifacts/fast_suite.sb3` also completed end-to-end in the
official VM. With 100 forwards per arm, Quick GPT measured 1.359 s exact and
1.228 s fast (1.11x). CatGPT1 measured 18.070 s and 17.944 s respectively;
matrix multiplication dominates that larger model, so default arithmetic-only
fast mode had little effect. These remain diagnostic results pending a browser
Scratch run.

Static-right-hand-side generic matmul now uses the same structured weight
options as module layers. In focused official-VM tests, 50% pruning improved a
32x32-by-32x96 matmul by 1.35x and `rank_ratio=0.25` improved it by 2.64x.
Dynamic matmuls are intentionally unchanged. Exact SwiGLU fusion reduced a
representative 128x128 `silu(gate) * value` test from 1.614 s to 1.122 s
(1.44x) without changing its output.

The generation suite prefills 16 tokens and times 16 additional decode calls.
The diagnostic official-VM run measured 29.5x for Quick GPT and 32.2x for
CatGPT1. The authoritative scratch.mit.edu run saved as
`artifacts/generation_suite_benchmarked.sb3` measured 1.105 s stateless versus
0.033 s cached for Quick GPT (33.5x), and 3.282 s versus 0.133 s for CatGPT1
(24.7x). The cached Quick GPT result is at the browser timer's resolution,
but both models show an unambiguous structural win.
