"""
Tokenizer transpilation tests.

Each test transpiles a tokenizer, runs encode/decode through the emulator,
and compares against the HuggingFace tokenizer output.
"""

import json
import os
import zipfile

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from cattorch.tokenizer import transpile_tokenizer, CharTokenizer, BPETokenizer
from cattorch.util.scratch.emulator import ScratchEmulator

SPRITE_PATH = os.path.join(os.path.dirname(__file__), "_test_tokenizer")


def _make_char_tokenizer(chars: str) -> PreTrainedTokenizerFast:
    """Build a character-level HuggingFace tokenizer from a string of chars."""
    vocab = {c: i for i, c in enumerate(chars)}
    tok_model = models.WordLevel(vocab=vocab, unk_token=chars[0])
    tokenizer = Tokenizer(tok_model)
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer)


def _load_sprite() -> dict:
    """Transpile and load the sprite JSON, cleaning up the file."""
    sprite_path = SPRITE_PATH + ".sprite3"
    try:
        with zipfile.ZipFile(sprite_path, "r") as z:
            return json.loads(z.read("sprite.json"))
    finally:
        os.remove(sprite_path)


def _encode(sprite: dict, text: str) -> list:
    """Run the encode block stack and return token_ids."""
    emu = ScratchEmulator(sprite)
    emu.variables["input"] = text
    emu.run(root_index=0)
    return emu.lists["token_ids"]


def _decode(sprite: dict, token_ids: list) -> str:
    """Run the decode block stack and return the output string."""
    emu = ScratchEmulator(sprite)
    emu.lists["token_ids"] = list(token_ids)
    emu.run(root_index=1)
    return emu.variables["output"]


# ── Tests ────────────────────────────────────────────────────────────────────


class TestCharTokenizerEncode:

    def test_basic_encode(self):
        tok = _make_char_tokenizer("abcdefghijklmnopqrstuvwxyz .")
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        ids = _encode(sprite, "hello")
        expected = tok.encode("hello")
        assert [int(x) for x in ids] == expected

    def test_single_char(self):
        tok = _make_char_tokenizer("abcdefghijklmnopqrstuvwxyz .")
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        ids = _encode(sprite, "a")
        assert [int(x) for x in ids] == tok.encode("a")

    def test_spaces(self):
        tok = _make_char_tokenizer("abcdefghijklmnopqrstuvwxyz .")
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        ids = _encode(sprite, "hi there")
        assert [int(x) for x in ids] == tok.encode("hi there")

    def test_all_chars(self):
        chars = "abcdefghijklmnopqrstuvwxyz ."
        tok = _make_char_tokenizer(chars)
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        ids = _encode(sprite, chars)
        assert [int(x) for x in ids] == tok.encode(chars)


class TestCharTokenizerDecode:

    def test_basic_decode(self):
        tok = _make_char_tokenizer("abcdefghijklmnopqrstuvwxyz .")
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        result = _decode(sprite, [7, 4, 11, 11, 14])
        assert result == "hello"

    def test_single_token(self):
        tok = _make_char_tokenizer("abcdefghijklmnopqrstuvwxyz .")
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        result = _decode(sprite, [0])
        assert result == "a"

    def test_empty_ids(self):
        tok = _make_char_tokenizer("abcdefghijklmnopqrstuvwxyz .")
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        result = _decode(sprite, [])
        assert result == ""


class TestCharTokenizerRoundTrip:

    def test_roundtrip(self):
        tok = _make_char_tokenizer("abcdefghijklmnopqrstuvwxyz .")
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        text = "hello world."
        ids = _encode(sprite, text)
        result = _decode(sprite, ids)
        assert result == text

    def test_roundtrip_repeated_chars(self):
        tok = _make_char_tokenizer("abcdefghijklmnopqrstuvwxyz .")
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        text = "aaa bbb"
        ids = _encode(sprite, text)
        result = _decode(sprite, ids)
        assert result == text

    def test_small_vocab(self):
        tok = _make_char_tokenizer("abc")
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        text = "abcabc"
        ids = _encode(sprite, text)
        expected = tok.encode(text)
        assert [int(x) for x in ids] == expected

        result = _decode(sprite, ids)
        assert result == text


# ── BPE helpers ──────────────────────────────────────────────────────────────


def _make_bpe_tokenizer(corpus: list[str], vocab_size: int = 50) -> PreTrainedTokenizerFast:
    """Train a small BPE tokenizer on the given corpus.

    Uses no pre-tokenizer so BPE merges across the full input, matching
    the Scratch template's character-by-character split.
    """
    tok = Tokenizer(models.BPE())
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=1, special_tokens=[]
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tok)


# ── BPE Tests ────────────────────────────────────────────────────────────────


class TestBPETokenizerEncode:

    def test_basic_encode(self):
        tok = _make_bpe_tokenizer(["hello world", "hello there"])
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        ids = _encode(sprite, "hello")
        expected = tok.encode("hello")
        assert [int(x) for x in ids] == expected

    def test_single_char(self):
        tok = _make_bpe_tokenizer(["abcabc", "abcabc"])
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        ids = _encode(sprite, "a")
        assert [int(x) for x in ids] == tok.encode("a")

    def test_merges_applied(self):
        """Verify that BPE merges actually reduce the token count."""
        tok = _make_bpe_tokenizer(
            ["ab ab ab ab", "ab ab ab ab"], vocab_size=20
        )
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        text = "ab"
        ids = _encode(sprite, text)
        expected = tok.encode(text)
        assert [int(x) for x in ids] == expected
        # BPE should merge "a"+"b" → "ab", giving 1 token instead of 2
        assert len(ids) < len(text)

    def test_encode_with_spaces(self):
        tok = _make_bpe_tokenizer(["the cat sat", "the cat sat on the mat"])
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        ids = _encode(sprite, "the cat")
        expected = tok.encode("the cat")
        assert [int(x) for x in ids] == expected


class TestBPETokenizerDecode:

    def test_basic_decode(self):
        tok = _make_bpe_tokenizer(["hello world", "hello there"])
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        encoded = tok.encode("hello")
        result = _decode(sprite, encoded)
        assert result == "hello"

    def test_decode_single_token(self):
        tok = _make_bpe_tokenizer(["hello world", "hello there"])
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        # Decode a single base character token
        vocab = tok.get_vocab()
        char_id = vocab.get("h", 0)
        result = _decode(sprite, [char_id])
        assert result == "h"

    def test_decode_empty(self):
        tok = _make_bpe_tokenizer(["hello world"])
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        result = _decode(sprite, [])
        assert result == ""


class TestBPETokenizerRoundTrip:

    def test_roundtrip_simple(self):
        tok = _make_bpe_tokenizer(["hello world", "hello there", "the world"])
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        text = "hello"
        ids = _encode(sprite, text)
        result = _decode(sprite, ids)
        assert result == text

    def test_roundtrip_with_spaces(self):
        tok = _make_bpe_tokenizer(["hello world", "hello there", "the world"])
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        text = "hello world"
        ids = _encode(sprite, text)
        result = _decode(sprite, ids)
        assert result == text

    def test_roundtrip_repeated_pattern(self):
        tok = _make_bpe_tokenizer(["abab", "abab", "abab"], vocab_size=20)
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        text = "ababab"
        ids = _encode(sprite, text)
        result = _decode(sprite, ids)
        assert result == text

    def test_roundtrip_all_base_chars(self):
        """Ensure every base character round-trips correctly."""
        corpus = ["abcdefghij", "klmnopqrst", "uvwxyz"]
        tok = _make_bpe_tokenizer(corpus, vocab_size=30)
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        text = "abcxyz"
        ids = _encode(sprite, text)
        result = _decode(sprite, ids)
        assert result == text

    def test_roundtrip_longer_text(self):
        corpus = ["the cat sat on the mat", "the dog sat on the log"]
        tok = _make_bpe_tokenizer(corpus, vocab_size=60)
        transpile_tokenizer(tok, SPRITE_PATH)
        sprite = _load_sprite()

        text = "the cat sat"
        ids = _encode(sprite, text)
        result = _decode(sprite, ids)
        assert result == text


# ── Class API Tests ──────────────────────────────────────────────────────────


class TestClassAPI:

    def test_char_tokenizer_class(self):
        tok = _make_char_tokenizer("abcdefghijklmnopqrstuvwxyz .")
        CharTokenizer(tok).save(SPRITE_PATH)
        sprite = _load_sprite()

        text = "hello"
        ids = _encode(sprite, text)
        assert [int(x) for x in ids] == tok.encode(text)
        assert _decode(sprite, ids) == text

    def test_bpe_tokenizer_class(self):
        tok = _make_bpe_tokenizer(["hello world", "hello there", "the world"])
        BPETokenizer(tok).save(SPRITE_PATH)
        sprite = _load_sprite()

        text = "hello"
        ids = _encode(sprite, text)
        assert [int(x) for x in ids] == tok.encode(text)
        assert _decode(sprite, ids) == text
