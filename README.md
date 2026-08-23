# cattorch

cattorch exports inference-ready PyTorch neural networks as Scratch 3 sprites.
It lowers a `torch.nn.Module` to ordinary Scratch blocks and lists, so the
result can be imported into a project without a Scratch extension or modified
runtime.

Try the [published TinyStories example](https://scratch.mit.edu/projects/1374224416/)
to see an exported transformer running in vanilla Scratch.

cattorch is an inference exporter, not a training framework. Train the model in
PyTorch, put it in evaluation mode, then export it with representative inputs.
The complete workflow is in the
[getting-started guide](https://github.com/NormallyNormal/cattorch/blob/main/docs/getting-started.md).

Install with:

```bash
pip install cattorch
```

cattorch requires Python 3.10 or newer and PyTorch 2.6 or newer.

## What it supports

- MLPs, CNNs, and decoder-style transformer graphs with fixed export shapes.
- Linear layers, convolution, pooling, matrix multiplication, embeddings,
  normalization, common activations, softmax, masking, arithmetic, reductions,
  slicing, concatenation, splitting, transposes, and shape operations.
- Exact Scratch-specific optimizations, including constant folding, fused
  epilogues, causal-attention specialization, and export-time weight layout.
- An opt-in `fast` mode with approximate GELU, LayerNorm, and softmax kernels,
  plus configurable structured pruning and low-rank weight transforms.
- Stateful single-token generation with KV caches for recognized causal
  decoder graphs.
- Lossless float32 plus lossy float16, groupwise int8/int6, and packed int4
  static-weight storage, decoded once at startup.
- Automatic sharding of logical tensors across Scratch's 200,000-item physical
  list limit.
- Character, raw-text BPE, and SentencePiece BPE tokenizer/detokenizer sprites.
- Structured export metadata, numerical verification in the included emulator,
  and benchmark projects designed to be timed in Scratch itself.

The generated model sprite exposes `cattorch init`, `cattorch forward`, and
`cattorch prepare for save` custom blocks. Model inputs use the `input`,
`input_1`, ... lists and the single tensor result is written to `output`.
Generated custom blocks run without screen refresh.

## Documentation

- [Getting started](https://github.com/NormallyNormal/cattorch/blob/main/docs/getting-started.md)
- [Supported models and operations](https://github.com/NormallyNormal/cattorch/blob/main/docs/supported-models.md)
- [Optimization modes](https://github.com/NormallyNormal/cattorch/blob/main/docs/optimization.md)
- [Storage and Scratch limits](https://github.com/NormallyNormal/cattorch/blob/main/docs/storage.md)
- [Code generation and JSON size](https://github.com/NormallyNormal/cattorch/blob/main/docs/code-generation.md)
- [KV-cached generation](https://github.com/NormallyNormal/cattorch/blob/main/docs/generation.md)
- [Tokenizers](https://github.com/NormallyNormal/cattorch/blob/main/docs/tokenizers.md)
- [Verification and benchmarking](https://github.com/NormallyNormal/cattorch/blob/main/docs/verification-and-benchmarking.md)
- [Troubleshooting export failures](https://github.com/NormallyNormal/cattorch/blob/main/docs/troubleshooting.md)
- [Kernel development](https://github.com/NormallyNormal/cattorch/blob/main/docs/kernels.md)

## License

MIT
