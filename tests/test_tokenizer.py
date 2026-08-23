"""
Tokenizer transpilation tests.

Each test transpiles a tokenizer, runs encode/decode through the emulator,
and compares against the HuggingFace tokenizer output.
"""

import json
import os
import zipfile
from pathlib import Path

import pytest
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from cattorch.tokenizer import (
    BPETokenizer,
    CharTokenizer,
    SentencePieceBPETokenizer,
    transpile_tokenizer,
)
from cattorch.results import TokenizerResult
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

    def test_unknown_character_uses_tokenizer_unknown_id(self):
        backend = Tokenizer(models.WordLevel(vocab={"<unk>": 0, "a": 1}, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="<unk>",
        )
        CharTokenizer(tokenizer).save(SPRITE_PATH)
        sprite = _load_sprite()

        assert [int(value) for value in _encode(sprite, "z")] == [0]

    def test_character_tokenizer_requires_unknown_id(self):
        backend = Tokenizer(models.WordLevel(vocab={"a": 0}))
        backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)

        with pytest.raises(ValueError, match="unknown-token ID"):
            CharTokenizer(tokenizer).save(SPRITE_PATH)

    def test_rejects_case_collisions_and_non_bmp_characters(self):
        case_backend = Tokenizer(
            models.WordLevel(vocab={"<unk>": 0, "a": 1, "A": 2}, unk_token="<unk>")
        )
        case_backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
        case_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=case_backend, unk_token="<unk>",
        )
        with pytest.raises(ValueError, match="Scratch compares as equal"):
            CharTokenizer(case_tokenizer).save(SPRITE_PATH)

        emoji_backend = Tokenizer(
            models.WordLevel(vocab={"<unk>": 0, "😀": 1}, unk_token="<unk>")
        )
        emoji_backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
        emoji_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=emoji_backend, unk_token="<unk>",
        )
        with pytest.raises(ValueError, match="UTF-16 surrogate"):
            CharTokenizer(emoji_tokenizer).save(SPRITE_PATH)


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

    def test_uses_canonical_merge_rank_instead_of_token_id(self):
        backend = Tokenizer(models.BPE(
            vocab={"<unk>": 0, "a": 1, "b": 2, "c": 3, "ab": 4, "bc": 5},
            merges=[("a", "b"), ("b", "c")],
            unk_token="<unk>",
        ))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="<unk>",
        )
        BPETokenizer(tokenizer).save(SPRITE_PATH)
        sprite = _load_sprite()

        assert [int(value) for value in _encode(sprite, "abc")] == tokenizer.encode("abc")

    def test_unknown_character_uses_bpe_unknown_id(self):
        backend = Tokenizer(models.BPE(
            vocab={"<unk>": 0, "a": 1}, merges=[], unk_token="<unk>",
        ))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="<unk>",
        )
        BPETokenizer(tokenizer).save(SPRITE_PATH)
        sprite = _load_sprite()

        assert [int(value) for value in _encode(sprite, "z")] == [0]

    def test_bpe_without_unknown_token_omits_unmatched_characters(self):
        backend = Tokenizer(models.BPE(vocab={"a": 0}, merges=[]))
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
        BPETokenizer(tokenizer).save(SPRITE_PATH)
        sprite = _load_sprite()

        assert tokenizer.encode("aza") == [0, 0]
        assert [int(value) for value in _encode(sprite, "aza")] == [0, 0]

    def test_rejects_case_colliding_bpe_vocab(self):
        backend = Tokenizer(models.BPE(
            vocab={"<unk>": 0, "a": 1, "A": 2}, merges=[], unk_token="<unk>",
        ))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="<unk>",
        )
        with pytest.raises(ValueError, match="Scratch compares as equal"):
            BPETokenizer(tokenizer).save(SPRITE_PATH)


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


class _FakeSentencePieceProcessor:
    pieces = (
        "<|pad|>",
        "<|unk|>",
        "<0x5F>",
        "ab",
        "▁ab",
        "▁",
        "a",
        "b",
        "!",
        "<0xF0>",
    )

    def get_piece_size(self):
        return len(self.pieces)

    def id_to_piece(self, token_id):
        return self.pieces[token_id]

    def is_byte(self, token_id):
        return token_id in {2, 9}

    def is_control(self, token_id):
        return token_id == 0

    def get_score(self, token_id):
        return -float(token_id)

    def unk_id(self):
        return 1


class _ManySplitSentencePieceProcessor:
    pieces = ("<|pad|>", "<|unk|>", *("a" * size for size in range(1, 9)))

    def get_piece_size(self):
        return len(self.pieces)

    def id_to_piece(self, token_id):
        return self.pieces[token_id]

    def is_byte(self, token_id):
        return False

    def is_control(self, token_id):
        return token_id == 0

    def get_score(self, token_id):
        return -float(token_id)

    def unk_id(self):
        return 1


class TestSentencePieceBPE:

    def test_rejects_default_dummy_prefix_sentencepiece(self, tmp_path):
        sentencepiece = pytest.importorskip("sentencepiece")
        corpus = tmp_path / "corpus.txt"
        corpus.write_text(("hello world\nthe cat sat\n") * 20)
        prefix = tmp_path / "default"
        sentencepiece.SentencePieceTrainer.train(
            input=str(corpus),
            model_prefix=str(prefix),
            vocab_size=20,
            model_type="bpe",
            hard_vocab_limit=False,
            pad_id=0,
            unk_id=1,
            bos_id=-1,
            eos_id=-1,
        )
        processor = sentencepiece.SentencePieceProcessor(
            model_file=str(prefix.with_suffix(".model")),
        )
        with pytest.raises(ValueError, match="add_dummy_prefix"):
            SentencePieceBPETokenizer(processor)

    def test_transpile_auto_detects_sentencepiece(self, tmp_path):
        result = transpile_tokenizer(
            _FakeSentencePieceProcessor(),
            tmp_path / "sentencepiece",
            scratch_casefold=True,
        )
        assert result.tokenizer_type == "SentencePieceBPETokenizer"

    def test_encode_casefold_ascii_fallback_and_merges(self, tmp_path):
        result = SentencePieceBPETokenizer(
            _FakeSentencePieceProcessor(),
            scratch_casefold=True,
        ).save(tmp_path / "sentencepiece")
        with zipfile.ZipFile(result.path) as archive:
            sprite = json.loads(archive.read("sprite.json"))

        emulator = ScratchEmulator(sprite)
        emulator.variables["input"] = "AB_ab!"
        emulator.run_procedure("cattorch tokenize")
        assert [int(value) for value in emulator.lists["token_ids"]] == [3, 2, 3, 8]

    def test_encode_considers_merges_beyond_vocabulary_length(self, tmp_path):
        result = SentencePieceBPETokenizer(
            _ManySplitSentencePieceProcessor(),
        ).save(tmp_path / "many_split_sentencepiece")
        with zipfile.ZipFile(result.path) as archive:
            sprite = json.loads(archive.read("sprite.json"))

        emulator = ScratchEmulator(sprite)
        emulator.variables["input"] = "aaaaaaaa"
        emulator.run_procedure("cattorch tokenize")
        assert [int(value) for value in emulator.lists["token_ids"]] == [9]

    def test_decode_skips_specials_and_converts_space_marker(self, tmp_path):
        result = SentencePieceBPETokenizer(
            _FakeSentencePieceProcessor(),
            scratch_casefold=True,
        ).save(tmp_path / "sentencepiece")
        with zipfile.ZipFile(result.path) as archive:
            sprite = json.loads(archive.read("sprite.json"))

        emulator = ScratchEmulator(sprite)
        emulator.lists["token_ids"] = [0, 1, 3, 2, 4, 8]
        emulator.run_procedure("cattorch detokenize")
        assert emulator.variables["output"] == "ab_ ab!"

    def test_exports_clickable_stacks_warp_procedures_and_warning(self, tmp_path):
        result = SentencePieceBPETokenizer(
            _FakeSentencePieceProcessor(),
            scratch_casefold=True,
        ).save(tmp_path / "sentencepiece")
        with zipfile.ZipFile(result.path) as archive:
            sprite = json.loads(archive.read("sprite.json"))

        emulator = ScratchEmulator(sprite)
        assert set(emulator._procedures) == {
            "cattorch tokenize",
            "cattorch detokenize",
        }
        assert len(emulator._find_roots()) == 2
        assert any("non-ASCII byte tokens" in warning for warning in result.warnings)
        prototypes = [
            block
            for block in sprite["blocks"].values()
            if block["opcode"] == "procedures_prototype"
        ]
        assert all(block["mutation"]["warp"] == "true" for block in prototypes)

def test_tokenizer_export_returns_metadata_and_separates_name(tmp_path):
    tok = _make_char_tokenizer("abc ")
    result = transpile_tokenizer(
        tok,
        tmp_path / "nested" / "tokenizer.sprite3",
        name="Friendly tokenizer",
    )
    assert isinstance(result, TokenizerResult)
    assert result.path == tmp_path / "nested" / "tokenizer.sprite3"
    assert result.sprite_name == "Friendly tokenizer"
    assert result.tokenizer_type == "CharTokenizer"
    assert result.token_count == 4
    assert result.archive_bytes == result.path.stat().st_size
    with zipfile.ZipFile(result.path) as archive:
        sprite = json.loads(archive.read("sprite.json"))
    assert sprite["name"] == "Friendly tokenizer"
