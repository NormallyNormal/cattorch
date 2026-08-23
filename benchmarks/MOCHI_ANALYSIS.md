# Mochi 20M Scratch/TurboWarp Transformer Analysis

Analyzed artifact:

`benchmarks/artifacts/Mochi AI Real Transformer LLM - 20M Parameters CoT.sb3`

The checked-in file is a post-initialization save. Its archive is 34,540,211
bytes and its expanded `project.json` is 231,421,761 bytes because both the
compact weights and decoded weights are present, along with populated KV
caches.

## Architecture

- Vocabulary: 4,096
- Residual/embedding width: 384
- Layers: 12
- Attention: 6 query heads, head width 64, full multi-head K/V
- FFN width: 1,536 with GELU
- Learned position table: 256 x 384
- RMSNorm and bias-free dense matrices
- Token embedding is reused as the output projection
- Incremental single-token generation with per-layer K/V caches
- Top-k sampling (`k=5` in the saved state)

There are 22,914,432 quantized weight values, including learned positions and
norm weights. The “20M” name is approximate.

## Weight codec

Each `w_*` list contains one long hexadecimal string. Each character encodes
one signed 4-bit weight. A companion `rs_*` list stores one decimal scale per
output row. Initialization performs:

```text
scale = row_scale / 1,000,000
weight = ((numeric value of "0x" + hex_character) - 8) * scale
```

The decoded integer range is -8 through 7. This is row-wise int4
quantization, but inference uses ordinary decoded Scratch numbers; it does not
pay quantization parsing cost during every matmul.

The compact form is one safe ASCII character per weight, plus 45,849 row
scales. It relies on ZIP DEFLATE to exploit the non-uniform/repeated hex data.
Clearing decoded `d_*` lists and K/V caches from the analyzed JSON gives these
estimates:

- Compact expanded `project.json`: 24,799,994 bytes
- DEFLATE-compressed compact JSON: 10,645,784 bytes

This explains the roughly 10 MB distribution size. It is not two weights per
character; every nibble is represented by its own ASCII hex character.

## Runtime observations

All compute procedures are custom blocks marked “run without screen refresh.”
The model specializes and unrolls blocks per layer, preallocates its working
lists, uses a KV cache, and evaluates only one new token at a time. It still
uses full multi-head K/V and evaluates the tied 4,096-way output projection on
every prompt token.

A saved/decompressed forward pass runs in the official Scratch VM despite the
oversized static lists, because deserialization does not apply the list
mutation cap. One measured forward at saved context length 36 took 11.944
seconds in turbo mode and produced 4,096 logits. The reusable runner is
`benchmarks/run_mochi_forward_vm.cjs`.

The compact initializer is not compatible with the official VM as written:
`add item to list` stops at 200,000 items, while several decoded lists contain
up to 1,572,864 items. TurboWarp removes that practical restriction. Cattorch
must retain physical list sharding when it decodes compact weights for the
official Scratch runtime.

## Useful ideas for cattorch

1. Add row-wise int4 as an explicitly lossy storage option for fast mode. It
   can be decoded once at initialization, preserves ordinary numeric matmuls,
   and should permit much larger models than float16/float32 storage.
2. Use a hex decoder fast path. Scratch can coerce a string such as `0xF`
   directly to a number, avoiding cattorch's Base64 alphabet search and IEEE
   reconstruction for int4 payloads.
3. Keep tied token embeddings/output weights, RMSNorm, bias-free projections,
   fixed working buffers, and cached single-token generation. Cattorch already
   has equivalents for most of these.
4. Keep cattorch's MQA and hidden-only prefill. Both improve on Mochi's runtime
   architecture: Mochi stores full-width K/V and computes the vocabulary head
   for every prompt token.
5. For generation-only exports, fuse top-k/argmax into the vocabulary scan so
   logits do not need a separate full-list pass. Mochi currently materializes
   all logits and then scans them repeatedly for top-k, so its implementation
   is evidence for the feature but not the optimal loop.

## Official-VM kernel screens

`benchmarks/build_mochi_kernel_suite.py` builds a focused Scratch project at
`benchmarks/artifacts/mochi_kernel_suite.sb3`. Each case runs inside a warp
custom block, in turbo mode, after one untimed warmup. The following values are
the medians of seven fresh official Scratch VM runs; the VM is useful for
screening, while the browser remains the final performance authority.

| Screen | Work per timed case | Median |
| --- | ---: | ---: |
| dual running indices + append, 128 x 384 | 20 calls | 0.454 s |
| row base + feature + append, 128 x 384 | 20 calls | 0.442 s |
| row base + feature + replace, 128 x 384 | 20 calls | 0.445 s |
| dual running indices + append, 384 x 384 | 8 calls | 0.550 s |
| row base + feature + append, 384 x 384 | 8 calls | 0.521 s |
| row base + feature + replace, 384 x 384 | 8 calls | 0.520 s |
| materialize 1,024 logits, then argmax | 8 calls | 0.473 s |
| argmax during the 1,024-output projection | 8 calls | 0.470 s |
| five repeated maximum scans of 4,096 logits | 30 calls | 0.240 s |
| one-pass maintained top-5 over 4,096 logits | 30 calls | 0.080 s |
| learned-position add at width 128 | 1,000 calls | 0.064 s |
| three layers of Q/MQA-K RoPE work | 1,000 calls | 0.432 s |

The row-base form replaces a second changing index in every MAC with one
`row base + feature` reporter. It was 2.7% faster at inner width 128 and 5.6%
faster at inner width 384. Preallocating and replacing the output was not a
reliable improvement over clearing and appending. All three linear variants
produced identical output-list hashes.

Fusing argmax into the output projection was within run-to-run noise. It can
still avoid retaining a logits list in an argmax-only interface, but it should
not be sold as a VM speed optimization based on this result.

The maintained top-5 loop was exactly 3.0x faster than five destructive scans,
with identical value and ID list hashes. This is the clear instruction-level
optimization to adopt when generation requests a small fixed `k`.

Learned positional addition took about one seventh of the representative
three-layer RoPE work. This supports learned positions for a Scratch-speed-
first architecture, but it is an architecture/quality trade rather than an
exact substitution for RoPE.

### Browser Scratch results

The same project was subsequently run in turbo mode on `scratch.mit.edu` and
saved as `artifacts/mochi_kernel_suite_benchmarked.sb3`:

| Screen | Browser time | Official-VM median |
| --- | ---: | ---: |
| dual indices + append, 128 x 384 | 0.506 s | 0.454 s |
| row base + append, 128 x 384 | 0.470 s | 0.442 s |
| row base + replace, 128 x 384 | 0.454 s | 0.445 s |
| dual indices + append, 384 x 384 | 0.573 s | 0.550 s |
| row base + append, 384 x 384 | 0.532 s | 0.521 s |
| row base + replace, 384 x 384 | 0.528 s | 0.520 s |
| materialize logits, then argmax | 0.484 s | 0.473 s |
| fused projection argmax | 0.483 s | 0.470 s |
| five repeated top-k scans | 0.343 s | 0.240 s |
| one-pass maintained top-5 | 0.109 s | 0.080 s |
| learned-position add | 0.071 s | 0.064 s |
| three-layer RoPE work | 0.501 s | 0.432 s |

Scratch confirms the actionable rankings. Row-base append is 7.7% faster than
dual indices at both widths. In this steady-state test, row-base replace is
11.5% faster at width 128 and 8.5% faster at width 384. The replace advantage
over row-base append itself is only 3.5% and 0.8%, so preallocation should be
validated in whole exported networks before becoming a general policy.

One-pass top-5 is 3.15x faster, fused argmax remains neutral, and learned
position addition is 7.06x cheaper than the representative RoPE workload. The
saved linear, head, top-k value, and top-k ID lists compare exactly equal
between equivalent implementations.

The official VM is directionally useful, but the browser/VM time ratio varies
by kernel (roughly 1.02x through 1.43x here). It should be used to reject weak
ideas and prioritize candidates, not as a universal conversion factor or a
complete replacement for final Scratch measurements.

## Project-limit conclusion

The artifact identifies its platform as TurboWarp. Its compact expanded JSON
is about 24.8 MB, well above Scratch's documented/currently reported 5 MB
`project.json` upload cap. Therefore the 10 MB compressed archive demonstrates
a TurboWarp/local-project strategy, not a confirmed bypass for publishing the
same model on `scratch.mit.edu`. Archive size and expanded JSON size must be
tracked separately.
