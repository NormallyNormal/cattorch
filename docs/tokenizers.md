# Tokenizers

[Documentation home](index.md)

cattorch can export a tokenizer as its own Scratch sprite, so a project can
turn text into token IDs and back without leaving Scratch. Three kinds are
supported:

| Exporter | Use for | Main requirement |
|---|---|---|
| `CharTokenizer` | Very small character vocabularies | Requires a valid unknown-token ID. |
| `BPETokenizer` | Raw-text Hugging Face BPE | No pre-tokenizer, dropout, subword prefix, or suffix. |
| `SentencePieceBPETokenizer` | Raw SentencePiece BPE with control tokens or ASCII byte fallback | Requires compatible normalization and preprocessing. |

Hugging Face tokenizers need the optional dependencies:

```bash
pip install "cattorch[tokenizers]"
```

```python
from cattorch import BPETokenizer, transpile_tokenizer

result = BPETokenizer(tokenizer).save("build/my_tokenizer")

# Or let cattorch select BPE versus character lookup from the backend.
result = transpile_tokenizer(tokenizer, "build/my_tokenizer")
```

The returned `TokenizerResult` gives the saved path, size, and any warnings.
As with `transpile`, parent directories are created, `.sprite3` is added if
missing, and `name=` sets the sprite name.

Every tokenizer sprite has two custom blocks, which run without screen
refresh, and a clickable stack for each:

- `cattorch tokenize` reads the `input` variable and writes token IDs to the
  `token_ids` list.
- `cattorch detokenize` reads `token_ids` and writes the text to the `output`
  variable.

Token IDs are zero-based, as in Python, even though Scratch list positions
start at 1. The variables and lists belong to the tokenizer sprite; to use them
from another sprite, see the
[global-list bridge](getting-started.md#connect-another-sprite).

`CharTokenizer` maps each character to its ID. `BPETokenizer` applies the
tokenizer's merges, in the tokenizer's order, to the whole input string,
spaces included. A character not in the vocabulary becomes the unknown token
if the tokenizer has one. `CharTokenizer` requires an unknown token. A raw
Hugging Face BPE tokenizer without one drops unknown characters, and the
exported sprite does the same.

Scratch compares text without regard to ASCII case, so export fails if the
vocabulary contains entries that differ only by case, such as `a` and `A`.
Export also fails on characters outside the Basic Multilingual Plane, such as
emoji, because Scratch treats each one as two separate characters.

## SentencePiece BPE

The SentencePiece sprite keeps SentencePiece's token IDs, converts its `▁`
space marker, skips control tokens when decoding, and supports byte fallback
for ASCII.

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

Set `scratch_casefold=True` only if the SentencePiece model was trained with
case-folding normalization. The sprite then relies on Scratch's
case-insensitive comparisons, so you don't need to lowercase text first.

The SentencePiece model must be trained with `add_dummy_prefix=False` and
`remove_extra_whitespaces=False`, and use the `identity` or `nfkc_cf`
normalizer. When cattorch can read these settings from the model, it rejects
anything else. Any other normalization has to be done by your project before
tokenizing.

Scratch can't get a character's Unicode code point, so byte fallback only works
for ASCII. ASCII text and characters that appear in the vocabulary tokenize
normally. Other characters can't be encoded. When decoding, non-ASCII byte
tokens are shown as `<0xNN>` rather than turned into wrong text.

## Compatibility

Tokenizers from large pretrained models usually rely on byte-level
preprocessing, regex splitting, or normalization that these sprites don't
implement, and their large vocabularies make the model's embedding table too
big for Scratch. Instead, train a small BPE tokenizer on your model's data,
without a pre-tokenizer:

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

cattorch doesn't train tokenizers. The example uses the `tokenizers` and
`transformers` packages, which the `cattorch[tokenizers]` extra installs.

Related: [tokenizer API reference](api-reference.md#tokenizer-export) and
[KV-cached generation](generation.md).
