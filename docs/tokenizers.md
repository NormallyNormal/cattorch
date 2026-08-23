# Tokenizers

cattorch can export character-level, raw-text BPE, and SentencePiece BPE
tokenizers as separate Scratch sprites, allowing the text-to-token and
token-to-text steps to run in the project.

Install the optional tokenizer dependencies when using Hugging Face
tokenizers:

```bash
pip install "cattorch[tokenizers]"
```

```python
from cattorch import BPETokenizer, transpile_tokenizer

result = BPETokenizer(tokenizer).save("build/my_tokenizer")

# Or let cattorch select BPE versus character lookup from the backend.
result = transpile_tokenizer(tokenizer, "build/my_tokenizer")
```

The returned `TokenizerResult` includes the artifact path, Scratch sprite name,
archive and expanded-JSON sizes, block/list counts, warnings, tokenizer type,
and token count. Parent directories are created automatically and `.sprite3`
is appended when needed. Use `name=` to choose a Scratch sprite name that is
different from the filename.

The generated sprite provides two top-level stacks:

- Encode reads the `input` variable and writes zero-based IDs to `token_ids`.
- Decode reads `token_ids` and writes text to the `output` variable.

These interfaces are local to the tokenizer sprite. Scratch does not let one
sprite call another sprite's custom blocks or directly read its local data. To
connect a separately imported tokenizer and model, add project-level broadcast
and global-data bridge scripts, or combine their blocks into one processor
sprite before import.

`CharTokenizer` maps each character directly. `BPETokenizer` repeatedly applies
the backend's canonical merges over the full raw input string, including
spaces. Unknown characters use the tokenizer's unknown-token ID when it has
one. `CharTokenizer` requires such an ID. Raw Hugging Face BPE models without
an unknown token omit unmatched characters, and the Scratch export preserves
that backend behavior.

Vanilla Scratch compares text without ASCII case distinctions. Exports reject
case-sensitive vocabularies containing entries such as both `a` and `A`, since
they cannot preserve token IDs. They also reject non-BMP vocabulary characters
such as emoji: Scratch's string blocks expose their two UTF-16 surrogate halves
rather than one character.

## SentencePiece BPE

`SentencePieceBPETokenizer` uses DSL-generated Scratch blocks rather than a
static JSON template. Its sprite includes run-without-screen-refresh
`cattorch tokenize` and `cattorch detokenize` custom blocks, plus clickable
stacks for both. It preserves zero-based SentencePiece IDs, converts the
SentencePiece whitespace marker, skips control tokens while decoding, and
supports ASCII byte fallback.

```python
from cattorch import SentencePieceBPETokenizer

# `tokenizer` may be a SentencePieceProcessor or expose one as `.processor`.
SentencePieceBPETokenizer(
    tokenizer,
    scratch_casefold=True,
).save("build/sentencepiece_tokenizer")

# Auto-detection is also available:
transpile_tokenizer(
    tokenizer,
    "build/sentencepiece_tokenizer",
    scratch_casefold=True,
)
```

Set `scratch_casefold=True` only for a model trained with case-folding
normalization. Scratch's case-insensitive comparisons then normalize ASCII case
inside the tokenizer, with no controller-side lowercase pass.

SentencePiece export requires a raw-text-compatible model trained with
`add_dummy_prefix=False` and `remove_extra_whitespaces=False`. The supported
normalizers are `identity` and `nfkc_cf`; other normalization pipelines must be
performed by the surrounding project and are rejected when cattorch can
inspect the serialized model.

Vanilla Scratch has no Unicode-codepoint reporter. Supported BMP characters
represented directly in the SentencePiece vocabulary and all ASCII input
tokenize normally, but arbitrary out-of-vocabulary Unicode cannot be converted
to its UTF-8 byte fallback sequence. On decode, non-ASCII byte tokens remain visible as
`<0xNN>` markers rather than silently producing incorrect text.

## Compatibility

Large production tokenizers often depend on byte-level preprocessing, regular
expression splitting, normalization stages, or other behavior that these
Scratch templates do not implement. Their vocabularies can also make the model
embedding prohibitively large. Train a small tokenizer on the model's corpus
without a pre-tokenizer so BPE operates on the raw string.

```python
from tokenizers import Tokenizer, models, trainers
from transformers import PreTrainedTokenizerFast

from cattorch import BPETokenizer

corpus = ["the cat sat on the mat", "the dog sat on the log"]
backend = Tokenizer(models.BPE())
trainer = trainers.BpeTrainer(vocab_size=100, min_frequency=1, special_tokens=[])
backend.train_from_iterator(corpus, trainer=trainer)

tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
BPETokenizer(tokenizer).save("small_bpe")
```

cattorch exports existing tokenizers; it does not train them. The `tokenizers`
and `transformers` packages are installed by the `tokenizers` extra used above.
