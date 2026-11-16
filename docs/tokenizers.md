# Tokenizers

[Documentation home](index.md)

cattorch can export a tokenizer as its own Scratch sprite, so a project can
turn text into token IDs and back without leaving Scratch. Three kinds are
supported:

| Exporter | Use for | Main requirement |
|---|---|---|
| `CharTokenizer` | Very small character vocabularies | Single-character tokens (added tokens may be longer), read one character at a time, and a valid unknown-token ID. |
| `BPETokenizer` | Raw-text Hugging Face BPE | No normalizer, pre-tokenizer, dropout, subword prefix or suffix, `fuse_unk`, or `byte_fallback`, and `ignore_merges` only if the merges build every vocabulary entry. |
| `SentencePieceBPETokenizer` | Raw SentencePiece BPE (`model_type="bpe"`) with control tokens or ASCII byte fallback | Requires compatible normalization and preprocessing. |

Hugging Face tokenizers need the optional dependencies:

```bash
pip install "cattorch[tokenizers]"
```

```python
from cattorch import BPETokenizer, transpile_tokenizer

result = BPETokenizer(tokenizer).save("build/my_tokenizer")

# Or let cattorch pick the exporter. A backend that isn't BPE is exported with
# character lookup only if every token is one character.
result = transpile_tokenizer(tokenizer, "build/my_tokenizer")
```

Other kinds of tokenizer are rejected with an error rather than exported with
different token IDs: Hugging Face WordPiece (BERT), Unigram (T5, ALBERT,
XLNet) and word-level tokenizers, and SentencePiece models trained with
`model_type` `unigram`, `word` or `char`, or with user-defined symbols. A
WordPiece, Unigram or word-level tokenizer whose tokens are all single
characters is exported only if it really encodes text one character at a
time; `CharTokenizer` checks this by encoding every vocabulary character, and
rejects a normalizer or a pre-tokenizer, such as `Whitespace`, that changes
the result.

The returned `TokenizerResult` gives the saved path, size, and any warnings.
As with `transpile`, parent directories are created, `.sprite3` is added if
missing, and `name=` sets the sprite name.

Every tokenizer sprite has two custom blocks, which run without screen
refresh, and a clickable stack for each (see the
[Scratch interface](scratch-interface.md#tokenizer-sprites)):

- `cattorch tokenize` reads the `cattorch text` variable and writes token IDs
  to the `cattorch token ids` list.
- `cattorch detokenize` reads `cattorch token ids` and writes the text to
  `cattorch text`. A token ID that isn't a whole number in the vocabulary
  sets `cattorch status` to `invalid item: cattorch token ids`, and the text
  is left as it was.

Like every sprite's blocks, each sets `cattorch status` when it ends and then
broadcasts `cattorch tokenize complete` or `cattorch detokenize complete`.

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

### Added and special tokens

Hugging Face tokenizers find their added tokens in the text before anything
else: tokens from `add_tokens` and `add_special_tokens`, and a trainer's
`special_tokens` such as `<unk>` or `</s>`. The character and BPE sprites do
the same, so `the cat</s>` ends with the ID of `</s>`, and a multi-character
added token such as `dog cat` is one ID wherever it appears. Matching follows
the `tokenizers` library: case-sensitive, the longest token at the leftmost
position first, tokens with `normalized=False` before those with
`normalized=True`, and `lstrip` and `rstrip` taking the whitespace beside a
match. The text between matches is tokenized on its own, so BPE merges never
cross an added token. `CharTokenizer` accepts multi-character added tokens
for this reason. The export checks the result against the tokenizer on texts
where added tokens meet, overlap and touch whitespace, and raises
`ValueError` if they differ. Added tokens with `single_word=True` are
rejected: whether a match stands alone depends on whether its neighbours are
letters or digits, which a sprite can't tell for every character.

With `split_special_tokens=True` (transformers) or `encode_special_tokens`
(a bare `tokenizers.Tokenizer`), the tokenizer reads its special tokens as
ordinary text, and so does the sprite: a special token the model's
vocabulary doesn't have becomes the unknown token. Exporting leaves the
tokenizer's settings as they were.

### Special tokens added around the text

A tokenizer's post-processor can add tokens around every text it encodes,
such as `<s>` before it: `tokenizer.encode("the cat")` then starts with the
ID of `<s>`. `cattorch tokenize` writes the same IDs as `encode(text)` with
its default `add_special_tokens=True`, so a model trained on such sequences
sees the prompts it was trained on. The sprite reproduces
`TemplateProcessing` with one `$A` in its single-sequence template (such as
`"<s> $A"` or `"<s> $A </s>"`), `BertProcessing`, `RobertaProcessing`,
`ByteLevel` (which changes no IDs) and a `Sequence` of these; any other
post-processor raises `ValueError`, and the export checks the result against
`encode`. `cattorch detokenize` writes these tokens' text too, as
`decode(..., skip_special_tokens=False)` does.

`cattorch detokenize` writes every token's text, added and special tokens
included, as `transformers`' `decode(..., skip_special_tokens=False)` keeps
them, so detokenizing what tokenize wrote gives the text back.

Characters are told apart by case, as the tokenizer does: `a` and `A` can
both be tokens, and with only lowercase entries `The` starts with the unknown
token. Scratch compares text without regard to case, so when the vocabulary
has letters, the sprite gets a costume named after each character and finds
a character by switching to its costume, since costume names match exactly.
Don't rename, reorder, or delete a tokenizer sprite's costumes. A vocabulary
without letters, such as digits and punctuation, is looked up with
`item # of` and has one costume.

Export fails on characters outside the Basic Multilingual Plane, such as
emoji, in a single-character token or an added token, because Scratch treats
each one as two separate characters. For the
same reason an emoji in the text reads as two unknown characters.

## SentencePiece BPE

The SentencePiece sprite keeps SentencePiece's token IDs, converts its `▁`
space marker, skips control tokens when decoding, decodes the unknown token
as SentencePiece does (its `unk_surface`, ` ⁇ ` by default), and supports
byte fallback for ASCII. Like SentencePiece, it doesn't read control tokens such as `<s>`
from text: `<s>` in the text is three ordinary characters. A processor made
with `add_bos=True` or `add_eos=True` adds the beginning or end token to
every text it encodes, and so does the sprite; one made with `reverse=True`
or `enable_sampling=True` raises `ValueError`.

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
case-folding normalization (`nfkc_cf`). The sprite then relies on Scratch's
case-insensitive comparisons for ASCII letters, so you don't need to
lowercase text first, and needs no character costumes. Any other model, with
the `identity` normalizer, tells characters apart by case like the other
tokenizers.

The SentencePiece model must be trained with `add_dummy_prefix=False` and
`remove_extra_whitespaces=False`, and use the `identity` or `nfkc_cf`
normalizer. When cattorch can read these settings from the model, it rejects
anything else. Any other normalization has to be done by your project before
tokenizing.

Scratch can't get a character's Unicode code point, so byte fallback only works
for ASCII. ASCII text, in either case, and characters that appear in the
vocabulary tokenize as SentencePiece does. A non-ASCII character outside the
vocabulary, which SentencePiece encodes as its UTF-8 bytes, becomes the
unknown token instead, and the export result warns about this. When
decoding, non-ASCII byte tokens are shown as `<0xNN>` rather than turned into
wrong text.

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
