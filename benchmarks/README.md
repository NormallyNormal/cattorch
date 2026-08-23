# cattorch Scratch benchmarks

These benchmarks measure inside the stock Scratch editor. Python and the
test emulator are not used as performance clocks.

Generate the suite:

```bash
python benchmarks/build_suite.py
```

This writes `benchmarks/artifacts/cattorch_suite.sb3`. The project contains
an exact/fast pair for every case and runs them sequentially. For each variant
it runs init outside the timed region, resets the timer, executes 100 forwards,
and appends a `name variant: seconds` entry to the visible results list. The
analyzer can still read older legacy/exact result projects.

For a single project containing both the 39 operator cases and the eight
matrix/LLM cases, generate the comprehensive suite instead:

```bash
python benchmarks/build_full_suite.py
```

This writes `benchmarks/artifacts/full_suite.sb3` with 94 sequential result
rows. `build_kernel_suite.py` remains useful when only operator-level results
are wanted.

To generate the focused `exact` and `fast` approximation suite, run:

```bash
python benchmarks/build_fast_suite.py
```

This writes `benchmarks/artifacts/fast_suite.sb3`. Its fast arm uses the
default arithmetic approximations without pruning or low-rank weight changes.
To isolate the two weight transformations in one project, generate:

```bash
python benchmarks/build_weight_suite.py
```

`benchmarks/artifacts/fast_weight_suite.sb3` contains pruning and low-rank
pairs for Linear, Conv1d, Conv2d, Quick GPT, and CatGPT1. Arithmetic fast paths
are disabled in this suite, so timing changes and output errors come only from
the weight transformation. The `prune50` cases remove 50% of Scratch-friendly
blocks; the `rank25` cases use 25% of each eligible layer's full rank.

The focused generic-matmul and exact SwiGLU-fusion suite is generated with:

```bash
python benchmarks/build_matmul_swiglu_suite.py
```

This creates `benchmarks/artifacts/matmul_swiglu_suite.sb3`. Static matmul is
tested with the same `prune50` and `rank25` settings. `swiglu_fused` computes
`silu(gate) * value` in one traversal, while `swiglu_materialized` deliberately
prevents fusion and provides the exact-output performance baseline.

Stateful decoding has its own stateless-forward versus KV-cache project:

```bash
python benchmarks/build_generation_suite.py
```

`benchmarks/artifacts/generation_suite.sb3` prefills each cached arm before
resetting the timer, runs all decode calls sequentially, and adds stateless and
cached totals to the usual results list.

The processor used by the
[published TinyStories project](https://scratch.mit.edu/projects/1374224416/)
is reproducible when a sibling `catgpt2` checkout and its checkpoint are
available:

```bash
python benchmarks/build_tinystories_complete.py
```

That build combines the cached model, SentencePiece tokenizer/detokenizer, and
streaming sampler, then writes a checksum and size manifest beside the ignored
generated artifacts. `add_tinystories_sampler.py`, `combine_sprites.py`, and
`catgpt2_approx_experiment.py` are supporting modules for this production
artifact and related model-level experiments.

Storage encoding has a focused suite as well:

```bash
python benchmarks/build_storage_suite.py
```

`benchmarks/artifacts/storage_suite.sb3` compares uncompressed float32 with
costume-Base85 float32, float16, groupwise int8, bit-packed int6, and packed
int4. It accumulates ten separately timed init calls (save preparation happens
outside each timed interval), then times ten ordinary forwards. A second
visible list records each standalone sprite size.

The completed scratch.mit.edu browser run on 2026-08-21 produced these totals
for the superseded Base64 implementation.
Each timing covers ten initialization or forward calls. This run predates the
small-matrix fallback and is retained in
`benchmarks/artifacts/storage_suite_benchmarked.sb3`.

| Case | Storage | Sprite bytes | Init | Forward |
| --- | --- | ---: | ---: | ---: |
| 256x256 matmul | Base64 f32 | 274,100 | 8.534 s | 0.333 s |
| 256x256 matmul | Base64 f16 | 143,858 | 4.913 s | 0.334 s |
| 256x256 matmul | Base64 int8 | 84,840 | 2.212 s | 0.335 s |
| 256x256 matmul | Base64 int4 | 53,425 | 1.669 s | 0.337 s |
| Quick GPT | Base64 f16 | 78,411 | 0.207 s | 0.170 s |
| Quick GPT | Base64 int8 | 87,260 | 0.135 s | 0.165 s |
| Quick GPT | Base64 int4 | 93,901 | 0.100 s | 0.165 s |
| CatGPT1 | Base64 f16 | 90,720 | 0.832 s | 1.890 s |
| CatGPT1 | Base64 int8 | 96,223 | 0.396 s | 1.896 s |
| CatGPT1 | Base64 int4 | 102,452 | 0.337 s | 1.902 s |

The matrix case shows the expected payload win and no meaningful forward
penalty from startup dequantization. On small multi-list models, the generated
decoder blocks outweighed the shorter payload. The production exporter now
keeps matrix shards below 16,384 values and one-dimensional parameters in
float16. Thus small models carry no integer decoder blocks, while substantial
weight matrices retain int8/int4 compression. Set `min_quantized_values=1` to
force the old all-matrix behavior for isolated format measurements.

The current TinyStories checkpoint has a reproducible model-level comparison:

```bash
PYTHONPATH=../catgpt2/src:. ../catgpt2/.venv/bin/python \
  benchmarks/tinystories_quant_experiment.py
```

It evaluates 256 held-out windows in PyTorch, builds four Scratch projects,
then evaluates eight of the same kind of windows in `@scratch/scratch-vm`
15.0.1. Results below are for checkpoint SHA-256
`c53b7329fbe8ebf5163a15339de7b9884f65f5f6948a5496324c561829ad0d44`.

| Storage | Sprite bytes | Expanded JSON | Python top-1 agreement | Mean abs logit error | VM top-1 agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| f32 | 2,791,813 | 6,391,959 | 100% | 0 | 100% |
| f16 | 1,447,303 | 4,408,161 | 99.22% | 0.00230 | 100% |
| int8 | 863,054 | 4,226,045 | 96.09% | 0.05088 | 100% |
| int4 | 525,264 | 4,099,935 | 53.91% | 0.95201 | 37.5% |

All four VM forwards remained near 3.0 seconds per 16-token window; storage
changes initialization and project size, not forward arithmetic. Int8 is the
strong compressed candidate for this checkpoint. Int4 is substantially
smaller but should be treated as an aggressive lossy option and judged on
long-form generation, not the eight-window VM sample alone. Reports and built
projects live under `benchmarks/artifacts/tinystories_quant/`; each report
records the checkpoint hash because that checkpoint is actively retrained.

Custom suites can select the same comparison, including an explicit weight
policy, from Python:

```python
from cattorch import FastConfig, FastLayerConfig, build_benchmark_suite

build_benchmark_suite(
    cases,
    "fast_weights.sb3",
    fast_config=FastConfig(weights=FastLayerConfig(rank_ratio=0.25)),
)
```

To run it:

1. Open <https://scratch.mit.edu/projects/editor/>.
2. Use **File → Load from your computer** and select the `.sb3`.
3. Enable Scratch's built-in Turbo Mode by shift-clicking the green flag.
4. Click the green flag normally and wait for the visible results list to
   contain every expected entry. List updates occur after each timed region.
5. Save the completed project to your computer.
6. Analyze it with `cattorch-benchmark downloaded-project.sb3`, or with
   `python -m cattorch.benchmark downloaded-project.sb3` from a checkout.

Both implementations use equivalent no-refresh custom blocks. The report
includes the raw named timings, speedup for each case, output error, block
counts, file size, and the project SHA-256.

Record the Scratch run date, browser version, operating system, and hardware
alongside saved benchmark reports. Use the official VM for controlled,
repeatable comparisons and scratch.mit.edu runs as samples of real browser
deployments. Neither is a universal conversion factor for all environments.

The browser timer advanced in roughly 0.033-second steps in the 2026-08-20/21
reference run. Treat short entries as resolution-limited and generate a
focused suite with a larger `iterations` value before comparing small changes.

## Next optimization screens

The benchmark-only candidates found in the vanilla Scratch VM audit are
bundled into one sequential project:

```bash
python benchmarks/build_next_optimization_suite.py
```

This writes `benchmarks/artifacts/next_optimization_suite.sb3`. Its 30 timed
regions cover hidden `for each`, literal repeat bounds, loop unrolling,
shard-local traversal, fused stable attention, interleaved grouped-linear
weights, RMSNorm-to-Linear fusion, paired-projection SwiGLU, and a synthetic
final-layer K/V-only prefill path. Every comparison retains output lists for
numerical checking.

Unlike the ordinary suites, this project times with deltas from Scratch's
`days since 2000` reporter. The official VM updates the ordinary timer only
once per scheduler step, whereas `days since 2000` reads the clock when the
reporter executes. The cases are still amplified to roughly one second where
small differences matter.

Open the project on scratch.mit.edu, enable Turbo Mode, and click the green
flag once. The visible list fills with all 30 results in order. Save the
completed project, then verify both its timings and outputs with:

```bash
node benchmarks/run_next_optimization_vm.cjs \
  benchmarks/artifacts/next_optimization_suite_benched.sb3
```

The controlled official-VM screen on 2026-08-21 took 27.3 wall-clock seconds.
It found the strongest exact improvements in `for each` (about 15%), direct
shard-local traversal (about 35%), interleaved grouped weights (about 13%),
and paired-projection SwiGLU (about 10%). Literal repeat bounds and partial
unrolling also helped. The tested RMSNorm fusion was slower, and fused stable
attention ranged from slightly slower to about 3% faster depending on context;
neither should move to production from this VM result alone.

MythicGPT-derived instruction and storage screens are generated with:

```bash
python benchmarks/build_mythic_kernel_suite.py
```

`benchmarks/artifacts/mythic_kernel_suite.sb3` compares cattorch's current
linear loop with Mythic's four-output loop at all three transformer matrix
shapes. It also screens integer-weight/post-dot scaling, exact and lookup-table
softmax, and equal-value-count cattorch costume-Base85-float16 versus Mythic-base66-int8 startup
decoding. The project runs 27 named timings sequentially and retains every
output list for numerical comparison.

The smaller follow-up destination-list screen is generated with:

```bash
python benchmarks/build_mythic_append_suite.py
```

`benchmarks/artifacts/mythic_append_suite.sb3` contains 14 timings comparing
four-output append and preallocated replace directly. It determines whether
the optimized linear loop can preserve cattorch's ordinary temporary-list
lifecycle without adding one-time buffer allocation.

After production integration, build the isolated whole-transformer comparison
with:

```bash
python benchmarks/build_grouped_linear_whole_suite.py
```

`benchmarks/artifacts/grouped_linear_whole_suite.sb3` is 0.92 MB compressed
and has a 3.44 MB expanded `project.json`. It compares the previous exact
linear kernel with production grouped exact over five forwards of a complete
width-128, FFN-384 causal transformer. Initialization and one warmup are both
outside the timer. The output head is deliberately ineligible, isolating the
QKV, FFN-up, and FFN-down changes. The project records two named results and
retains both output lists for an exact hash comparison.

The completed browser artifact is
`benchmarks/artifacts/grouped_linear_whole_suite_benched.sb3`. On the stock
Scratch VM it measured 1.973 seconds for the previous path and 1.901 seconds
for grouped exact over five forwards (3.65% less time), with identical
1,028-item outputs. This is a Firefox/Linux measurement on one machine, not a
browser-independent performance figure. Use the official VM result as the
controlled comparison and browser runs as deployment samples; retain both when
reporting performance.

The case-sensitive costume-codec screen is generated with:

```bash
python benchmarks/build_costume_codec_suite.py
```

`benchmarks/artifacts/costume_codec_suite.sb3` compares the former production
Base64 byte decoder with packed-loop Base64 and experimental Base85 decoders
that map both letter cases through 85 costume names. Unrolled and compact
Base85 forms are included. Each case decodes the same 49,152 bytes five times
and leaves an output list for exact comparison. Open it on
scratch.mit.edu, enable Turbo Mode, and click the green flag once. This case
must be decided in the browser: changing costume updates renderer state, which
the headless VM does not model. The VM remains useful as a renderer-free lower
bound and correctness check:

```bash
node benchmarks/run_scratch_vm.cjs \
  benchmarks/artifacts/costume_codec_suite.sb3 --inspect-lists
```

The JSON-size screen is generated with:

```bash
python benchmarks/build_json_size_suite.py
```

It exports the current TinyStories checkpoint as an ordinary stateless sprite,
an opt-in shared-layer sprite, and a shared-layer sprite with internal-name and
schema compaction. The current individual sprite measurements are 1,105,873,
948,739, and 924,996 expanded JSON bytes respectively; layer sharing reduces
the block count from 1,879 to 966. The suite is retained as a size/timing and
browser-compatibility screen. Shared-layer output agrees exactly in cattorch's
emulator, but the official headless VM currently diverges on both this model's
ordinary and factored full-network paths, so layer sharing remains off by
default until the vanilla browser comparison is authoritative.

`tinystories_quant_experiment.py` also records the rejected learned-codebook6
screen. On the current 1,024-window report, symmetric int6 retained 94.53% top-1
agreement while codebook6 retained 91.70%, with larger logit MAE and RMSE, so
codebook6 is intentionally not a public storage mode.

The standalone top-k size/speed screen is generated with:

```bash
python benchmarks/build_top_k_size_suite.py
```

For `k=16` and 1,024 logits, the loop-based selector is 31,878 expanded JSON
bytes / 69 blocks versus 135,411 bytes / 618 blocks for the unrolled ladder.
The official VM measured 0.294 versus 0.033 seconds over 20 selections. Both
produced identical value and token-ID hashes. This is why normal size-aware
exports use the compact form while `CodegenConfig(unrolling="speed")` preserves
the unrolled form.

`build_grouped_linear_network_suite.py` builds an additional diagnostic pair
from the real 998K CatGPT2 checkpoint. Its 13.2 MB expanded JSON may exceed
Scratch's upload parser limit, so it is intended for the official VM rather
than the browser.

## Local Scratch VM screening

The same suite can be run headlessly in the official Scratch VM for rapid
iteration:

```bash
cd benchmarks
npm install
npm run run-vm
```

The runner enables VM Turbo Mode, waits for the green-flag stack to finish,
and prints the same named timing entries as JSON. It does not attach a renderer,
so harmless missing-costume warnings may be printed. Use this to screen kernel
candidates, then periodically recalibrate against scratch.mit.edu—especially
for small differences or sub-second cases.

For the Mythic-specific suite, the paired numerical runner is:

```bash
node benchmarks/run_mythic_kernel_vm.cjs \
  benchmarks/artifacts/mythic_kernel_suite.sb3
```
