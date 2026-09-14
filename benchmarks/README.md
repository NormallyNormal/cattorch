# cattorch Scratch benchmarks

These scripts build Scratch projects that time cattorch kernels and exports
inside Scratch itself. The Python emulator is only used to check correctness,
never for timing. For the benchmark builders in the `cattorch` package, see the
[benchmark API reference](../docs/api-reference.md#benchmark-api).

Several suites need checkpoints from a sibling `catgpt2` checkout. Built
projects are written to `benchmarks/artifacts/`, which is not committed.

## Run a suite in Scratch

1. Build a suite, for example `python benchmarks/build_suite.py`.
2. Open <https://scratch.mit.edu/projects/editor/>.
3. Choose **File → Load from your computer** and select the `.sb3`.
4. Turn on Turbo Mode by shift-clicking the green flag.
5. Click the green flag and wait until the results list on the stage has every
   expected entry. Each entry is added after its timed section ends.
6. Save the finished project to your computer.
7. Run `cattorch-benchmark downloaded-project.sb3`, or
   `python -m cattorch.benchmark downloaded-project.sb3` from a checkout.

The report lists each named timing, the speedup for each case, output error,
block counts, file size, and the project's SHA-256. Save the report together
with the run date, browser version, operating system, and hardware.

In the browser, Scratch's timer advanced in steps of about 0.033 seconds during
the August 2026 runs. Treat short results as imprecise, and raise the
iteration count before comparing small differences.

### Headless Scratch VM

For faster iteration, run a suite in the official Scratch VM without a browser:

```bash
cd benchmarks
npm install
npm run run-vm
```

`run-vm` runs `artifacts/cattorch_suite.sb3` with Turbo Mode on and prints the
timings as JSON. To run another project, call the runner directly:

```bash
node benchmarks/run_scratch_vm.cjs benchmarks/artifacts/costume_codec_suite.sb3 --inspect-lists
```

The VM has no renderer, so it may print warnings about missing costumes; these
are harmless. VM timings are repeatable, which makes them good for comparing
candidates, while browser runs show what users actually get. Neither converts
to the other, so recheck small or sub-second differences in the browser, and
keep both kinds of result when reporting performance.

## Suites

Each script writes its project to `benchmarks/artifacts/`, named after the
script unless another name is shown.

| Script | Measures |
|---|---|
| `build_suite.py` | The main suite (`cattorch_suite.sb3`), with an exact and a fast variant of every case. Each variant runs `cattorch init` untimed, then 100 timed forwards, and adds `name variant: seconds` to the results list. The analyzer can also read results from older projects that compared the legacy backend with exact mode. |
| `build_kernel_suite.py` | Only the 39 operator-level cases. |
| `build_full_suite.py` | The 39 operator cases plus eight matrix and language-model cases in one project (`full_suite.sb3`, 94 results). |
| `build_fast_suite.py` | Fast mode's arithmetic approximations only, with no pruning or low-rank transforms (`fast_suite.sb3`). |
| `build_weight_suite.py` | Pruning and low-rank transforms only, with arithmetic approximations off, for Linear, Conv1d, Conv2d, Quick GPT, and CatGPT1 (`fast_weight_suite.sb3`). `prune50` removes 50% of weight groups; `rank25` keeps 25% of each eligible layer's rank. |
| `build_matmul_swiglu_suite.py` | Matrix multiply with the same `prune50` and `rank25` settings, and SwiGLU fused into one loop (`swiglu_fused`) versus computed in separate steps (`swiglu_materialized`). |
| `build_generation_suite.py` | Stateless forward versus KV-cached decoding. Cached cases prefill before the timer starts. |
| `build_storage_suite.py` | Startup and forward time for uncompressed float32 and Base92-encoded float32, float16, 8-bit, 6-bit, and 4-bit weights. Ten timed init calls, then ten forwards. A second list records each sprite's size. |
| `build_next_optimization_suite.py` | 30 kernel candidates from an audit of the Scratch VM. See [candidate kernels](#candidate-kernels-august-21). |
| `build_mythic_kernel_suite.py` | Techniques from MythicGPT, another transformer written in Scratch: its four-output linear loop at all three transformer matrix shapes, integer weights scaled after the dot product, exact and lookup-table softmax, and its base-66 int8 weight decoding versus cattorch's Base92 float16. 27 timings; every output list is kept. |
| `build_mythic_append_suite.py` | Whether MythicGPT's four-output linear loop can append to its output list instead of writing into a preallocated one. 14 timings. |
| `build_grouped_linear_whole_suite.py` | The previous exact linear kernel versus the grouped exact kernel on a complete transformer. |
| `build_grouped_linear_network_suite.py` | The same comparison on the 998K-parameter CatGPT2 checkpoint. Its 13.2 MB of JSON may be too large for the browser editor, so run it in the VM. |
| `build_costume_codec_suite.py` | Weight decoders that read data from costume names: the old Base64 decoder, a packed-loop Base64 decoder, and experimental Base85 decoders. Decide this one in the browser; see [costume decoders](#costume-decoders). |
| `build_json_size_suite.py` | JSON size and speed of layer sharing and compaction on the TinyStories checkpoint. |
| `build_cached_layer_sharing_suite.py` | Layer sharing with KV-cached generation on a 6M-parameter MoE model. |
| `build_top_k_size_suite.py` | Size and speed of the loop-based and unrolled top-k selectors. |
| `build_rope_redteam_suite.py` | The RoPE matrix expression versus more direct loops, at CatGPT size. |

Model-level builds:

| Script | Builds |
|---|---|
| `build_tinystories_complete.py` | The processor used by the [published TinyStories project](https://scratch.mit.edu/projects/1374224416/): cached model, SentencePiece tokenizer, and sampler, with a checksum and size manifest. Uses `add_tinystories_sampler.py` and `combine_sprites.py`. |
| `build_catgpt3_complete.py` | The same for the CatGPT3 MoE checkpoint. |
| `tinystories_quant_experiment.py` | Storage precision versus accuracy on the TinyStories checkpoint. See [storage accuracy](#storage-accuracy-on-tinystories). |
| `catgpt2_approx_experiment.py` | Accuracy and VM speed of low-rank and pruned CatGPT2 exports. |

### Custom suites

The same comparisons can be built from Python with your own models:

```python
from cattorch import FastConfig, FastLayerConfig, build_benchmark_suite

build_benchmark_suite(
    cases,
    "fast_weights.sb3",
    fast_config=FastConfig(weights=FastLayerConfig(rank_ratio=0.25)),
)
```

## Recorded results

Results below record one environment on one date. Rerun them before relying on
them for a decision.

### Storage decoding (browser, August 21)

This run on scratch.mit.edu used the earlier Base64 codec, before small
matrices were kept in float16. The project is saved as
`artifacts/storage_suite_benchmarked.sb3`. Each time covers ten calls.

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

For the large matrix, integer storage shrank the sprite and forward time didn't
change. For the small models, the extra decoder blocks outweighed the smaller
weights. As a result, matrices under 16,384 values and all one-dimensional
parameters now stay float16, so small models don't carry integer decoders. Set
`min_quantized_values=1` to quantize every matrix when measuring formats.

### Storage accuracy on TinyStories

```bash
PYTHONPATH=../catgpt2/src:. ../catgpt2/.venv/bin/python \
  benchmarks/tinystories_quant_experiment.py
```

This compares 256 held-out windows in PyTorch, then eight windows in
`@scratch/scratch-vm` 15.0.1. Checkpoint SHA-256:
`c53b7329fbe8ebf5163a15339de7b9884f65f5f6948a5496324c561829ad0d44`.

| Storage | Sprite bytes | Expanded JSON | Python top-1 agreement | Mean abs logit error | VM top-1 agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| f32 | 2,791,813 | 6,391,959 | 100% | 0 | 100% |
| f16 | 1,447,303 | 4,408,161 | 99.22% | 0.00230 | 100% |
| int8 | 863,054 | 4,226,045 | 96.09% | 0.05088 | 100% |
| int4 | 525,264 | 4,099,935 | 53.91% | 0.95201 | 37.5% |

Every variant took about 3.0 seconds per 16-token window in the VM: storage
affects startup time and size, not forward speed. Int8 is the best compressed
option for this checkpoint. Int4 is much smaller but loses a lot of accuracy;
judge it on long generated text, not these eight windows. Reports and projects
are in `artifacts/tinystories_quant/`, and each report records the checkpoint
hash because the checkpoint is still being retrained.

The same script also tested a learned 64-entry codebook for 6-bit weights. On a
1,024-window run, symmetric int6 kept 94.53% top-1 agreement and the codebook
kept 91.70%, with higher logit error, so the codebook was not added.

### Candidate kernels (August 21)

`build_next_optimization_suite.py` times each candidate with the `days since
2000` reporter instead of the timer, because the VM only updates the timer once
per scheduler step. Cases are sized to take about a second. To check both timings and
outputs of a saved project:

```bash
node benchmarks/run_next_optimization_vm.cjs \
  benchmarks/artifacts/next_optimization_suite_benched.sb3
```

In the VM, the whole suite took 27.3 seconds. The largest exact-mode gains were
`for each` loops (about 15%), reading shards directly (about 35%), interleaved
grouped weights (about 13%), and paired-projection SwiGLU (about 10%). Literal
repeat counts and partial unrolling also helped. RMSNorm fused into Linear was
slower, and fused stable attention ranged from slightly slower to about 3%
faster. Neither of those two should ship based on this result alone.

### MythicGPT kernels

To check outputs of a saved MythicGPT suite:

```bash
node benchmarks/run_mythic_kernel_vm.cjs \
  benchmarks/artifacts/mythic_kernel_suite.sb3
```

### Grouped linear on a whole transformer (browser)

The project is 0.92 MB compressed with 3.44 MB of JSON. It runs five forwards
of a width-128 transformer with a 384-wide feed-forward layer, with init and
one warmup untimed. The output layer doesn't qualify for the grouped kernel, so
only the attention and feed-forward projections change.

In Firefox on Linux, the previous kernel took 1.973 seconds and the grouped
kernel 1.901 seconds (3.65% faster), with identical 1,028-value outputs. The
saved project is `artifacts/grouped_linear_whole_suite_benched.sb3`.

### Costume decoders

Each case decodes the same 49,152 bytes five times. Base85 needs upper- and
lowercase costume names, and switching costumes updates the renderer, which the
headless VM skips. The VM result is therefore only a lower bound; decide with a
browser run.

### Layer sharing and JSON size

On the TinyStories checkpoint, the normal sprite, the layer-shared sprite, and
the layer-shared sprite with name and schema compaction have 1,105,873,
948,739, and 924,996 bytes of JSON. Layer sharing cuts blocks from 1,879 to
966. Its output matches exactly in cattorch's emulator, but the official VM's
output currently differs from the emulator for this model both with and without
sharing, so
layer sharing stays off by default until a browser comparison settles it.

### Layer sharing with cached generation (VM)

The unshared and shared 4-bit MoE processor sprites have 4,826,578 and
3,859,827 bytes of JSON. The combined project is larger than the online editor
accepts, so load it locally, click the green flag once, and read the six rows
of `cattorch benchmark results`.

| Variant | Init | Prefill | Two decodes |
| --- | ---: | ---: | ---: |
| Unshared | 22.206 s | 0.703 s | 1.505 s |
| Shared | 22.619 s | 1.192 s | 2.386 s |

Both produced identical top-k values and token IDs. Layer sharing with caching
saves space at the cost of speed.

### Top-k selector size

For `k=16` over 1,024 logits, the loop-based selector is 31,878 bytes of JSON
and 69 blocks; the unrolled selector is 135,411 bytes and 618 blocks. Over 20
selections, the VM measured 0.294 and 0.033 seconds, with identical outputs.
Exports sized with `unrolling="auto"` or `"compact"` use the loop, and
`unrolling="speed"` uses the unrolled form. The loop has since been sped up; see
[OPTIMIZATION.md](OPTIMIZATION.md).
