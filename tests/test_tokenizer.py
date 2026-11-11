"""
Tokenizer transpilation tests.

Each test transpiles a tokenizer, runs encode/decode through the emulator,
and compares against the HuggingFace tokenizer output.
"""

import json
import os
import tempfile
import zipfile

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

# One directory per process, so parallel test workers never share the file.
SPRITE_PATH = os.path.join(tempfile.mkdtemp(prefix="cattorch-tokenizer-"), "tokenizer")


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
    """Run the encode block stack and return the token IDs."""
    emu = ScratchEmulator(sprite)
    emu.variables["cattorch text"] = text
    emu.run(root_index=0)
    return emu.lists["cattorch token ids"]


def _decode(sprite: dict, token_ids: list) -> str:
    """Run the decode block stack and return the output string."""
    emu = ScratchEmulator(sprite)
    emu.lists["cattorch token ids"] = list(token_ids)
    emu.run(root_index=1)
    return emu.variables["cattorch text"]


_MIXED_CASE_CORPUS = [
    "The quick brown Fox jumps over the lazy dog.", "the cat sat on the mat",
    "A Cat and a DOG sat together, 12 times!", "Hello World: hello world?",
    "Numbers 0 1 2 3 4 5 6 7 8 9 and 42.", "tabs\tand  double  spaces",
] * 10


def _mixed_case_strings(count: int) -> list[str]:
    """Corpus fragments with flipped case, and random printable ASCII."""
    import random

    rng = random.Random(0)
    alphabet = "".join(map(chr, range(32, 127))) + "\t\n☃"
    texts = ["The Cat", "THE CAT", "Bb bB", "Q", "q", "1 01 10", "  x  "]
    while len(texts) < count:
        if rng.random() < 0.5:
            line = rng.choice(_MIXED_CASE_CORPUS)
            start = rng.randrange(len(line))
            texts.append("".join(
                c.swapcase() if rng.random() < 0.2 else c
                for c in line[start:start + rng.randint(1, 20)]
            ))
        else:
            texts.append("".join(rng.choice(alphabet) for _ in range(rng.randint(1, 20))))
    return texts


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

    def test_tells_characters_apart_by_case(self):
        # Scratch compares "a" and "A" as equal; costume names don't.
        backend = Tokenizer(models.WordLevel(
            vocab={"<unk>": 0, "a": 1, "A": 2, "b": 3, "7": 4, " ": 5}, unk_token="<unk>",
        ))
        backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
        CharTokenizer(tokenizer).save(SPRITE_PATH)
        sprite = _load_sprite()

        # "B" has no token of its own, and an unknown digit or whitespace
        # must not select a costume by number.
        for text in ("aAbB", "Aa", "B", "1 7 9", "\t a"):
            assert [int(value) for value in _encode(sprite, text)] == tokenizer.encode(text)

    def test_rejects_non_bmp_characters(self):
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

    def test_digit_lookup_ignores_numerically_equal_tokens(self):
        # Scratch's item # of treats "1", " 1", and "01" as equal numbers.
        backend = Tokenizer(models.BPE(
            vocab={"<unk>": 0, " 1": 1, "01": 2, " ": 3, "0": 4, "1": 5, "a": 6},
            merges=[(" ", "1"), ("0", "1")],
            unk_token="<unk>",
        ))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="<unk>",
        )
        BPETokenizer(tokenizer).save(SPRITE_PATH)
        sprite = _load_sprite()

        for text in ("a1 1", "1", "01 0"):
            assert [int(value) for value in _encode(sprite, text)] == tokenizer.encode(text)

    def test_real_mixed_case_bpe_matches_hugging_face(self):
        tokenizer = _make_bpe_tokenizer(_MIXED_CASE_CORPUS, vocab_size=120)
        BPETokenizer(tokenizer).save(SPRITE_PATH)
        sprite = _load_sprite()

        for text in _mixed_case_strings(200):
            assert [int(value) for value in _encode(sprite, text)] == tokenizer.encode(text), text


class TestBPESettingsTheSpriteCantReproduce:

    def test_rejects_normalizers_and_pre_tokenizers(self):
        from tokenizers import normalizers

        for stage, value in (
            ("normalizer", normalizers.NFD()),
            ("pre_tokenizer", pre_tokenizers.Whitespace()),
        ):
            tok = _make_bpe_tokenizer(["hello world", "hello there"])
            setattr(tok.backend_tokenizer, stage, value)
            with pytest.raises(ValueError, match="doesn't support a"):
                BPETokenizer(tok).save(SPRITE_PATH)

    def test_rejects_word_vocabularies(self):
        vocab = {"[UNK]": 0, "the": 1, "cat": 2}
        backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
        with pytest.raises(ValueError, match="multi-character tokens"):
            transpile_tokenizer(tok, SPRITE_PATH)


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


class _CasedBytesSentencePieceProcessor(_FakeSentencePieceProcessor):
    pieces = ("<|pad|>", "<|unk|>", "<0x51>", "<0x71>", "a")

    def is_byte(self, token_id):
        return token_id in {2, 3}


def _tokenize(result, text):
    with zipfile.ZipFile(result.path) as archive:
        emulator = ScratchEmulator(json.loads(archive.read("sprite.json")))
    emulator.variables["cattorch text"] = text
    emulator.run_procedure("cattorch tokenize")
    return [int(value) for value in emulator.lists["cattorch token ids"]]


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
        emulator.variables["cattorch text"] = "AB_ab!"
        emulator.run_procedure("cattorch tokenize")
        assert [int(value) for value in emulator.lists["cattorch token ids"]] == [3, 2, 3, 8]

    def test_casefolded_byte_fallback_uses_the_lowercase_byte(self, tmp_path):
        # A case-folding model reads "Q" and "q" as "q", byte 0x71.
        result = SentencePieceBPETokenizer(
            _CasedBytesSentencePieceProcessor(), scratch_casefold=True,
        ).save(tmp_path / "bytes")
        assert _tokenize(result, "qQa") == [3, 3, 4]

    def test_a_run_of_unknown_characters_is_one_unknown_token(self, tmp_path):
        result = SentencePieceBPETokenizer(
            _FakeSentencePieceProcessor(), scratch_casefold=True,
        ).save(tmp_path / "unknown")
        assert _tokenize(result, "%%a%") == [1, 6, 1]

    def test_encode_considers_merges_beyond_vocabulary_length(self, tmp_path):
        result = SentencePieceBPETokenizer(
            _ManySplitSentencePieceProcessor(),
        ).save(tmp_path / "many_split_sentencepiece")
        with zipfile.ZipFile(result.path) as archive:
            sprite = json.loads(archive.read("sprite.json"))

        emulator = ScratchEmulator(sprite)
        emulator.variables["cattorch text"] = "aaaaaaaa"
        emulator.run_procedure("cattorch tokenize")
        assert [int(value) for value in emulator.lists["cattorch token ids"]] == [9]

    def test_decode_skips_specials_and_converts_space_marker(self, tmp_path):
        result = SentencePieceBPETokenizer(
            _FakeSentencePieceProcessor(),
            scratch_casefold=True,
        ).save(tmp_path / "sentencepiece")
        with zipfile.ZipFile(result.path) as archive:
            sprite = json.loads(archive.read("sprite.json"))

        emulator = ScratchEmulator(sprite)
        emulator.lists["cattorch token ids"] = [0, 1, 3, 2, 4, 8]
        emulator.run_procedure("cattorch detokenize")
        assert emulator.variables["cattorch text"] == "ab_ ab!"

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


# ── Tokenizer kinds the sprite can't reproduce ───────────────────────────────

_CORPUS = ["the cat sat on the mat", "the dog sat on the log", "a big cat"] * 10


def _train_sentencepiece(model_type="bpe", corpus=_CORPUS, **options):
    import io

    sentencepiece = pytest.importorskip("sentencepiece")
    settings = dict(
        vocab_size=30, add_dummy_prefix=False, remove_extra_whitespaces=False,
        normalization_rule_name="identity", minloglevel=3, hard_vocab_limit=False,
    )
    settings.update(options)
    model = io.BytesIO()
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(corpus), model_writer=model, model_type=model_type, **settings,
    )
    return sentencepiece.SentencePieceProcessor(model_proto=model.getvalue())


@pytest.mark.parametrize("options", [
    # Byte pieces for "B" and "b" used to be rejected as equal in Scratch.
    dict(vocab_size=400, byte_fallback=True, character_coverage=0.98),
    dict(vocab_size=120),
], ids=["byte-fallback", "no-byte-fallback"])
def test_real_mixed_case_sentencepiece_matches_sentencepiece(tmp_path, options):
    processor = _train_sentencepiece(corpus=_MIXED_CASE_CORPUS, **options)
    result = SentencePieceBPETokenizer(processor).save(tmp_path / "sentencepiece")
    texts = _mixed_case_strings(300)
    if options.get("byte_fallback"):
        # SentencePiece encodes "☃" as UTF-8 bytes, which vanilla
        # Scratch can't find; the sprite gives the unknown token and warns.
        assert "☃" not in "".join(map(processor.id_to_piece, range(len(processor))))
        assert _tokenize(result, "☃") == [processor.unk_id()]
        assert any("encodes as the unknown token" in warning for warning in result.warnings)
        texts = [text for text in texts if "☃" not in text]
    for text in texts:
        assert _tokenize(result, text) == processor.encode(text), text


def test_objects_that_are_not_tokenizers_are_rejected_with_a_type_error(tmp_path):
    for export in (
        lambda: transpile_tokenizer(object(), tmp_path / "auto"),
        lambda: CharTokenizer(object()),
        lambda: BPETokenizer(object()),
    ):
        with pytest.raises(TypeError, match="needs a Hugging Face tokenizer.*not object"):
            export()


@pytest.mark.parametrize("model_type", ["unigram", "word", "char"])
def test_sentencepiece_models_other_than_bpe_are_rejected(tmp_path, model_type):
    processor = _train_sentencepiece(model_type)
    for export in (
        lambda: transpile_tokenizer(processor, tmp_path / "auto"),
        lambda: SentencePieceBPETokenizer(processor).save(tmp_path / "bpe"),
    ):
        with pytest.raises(ValueError, match=f"is a {model_type} model.*model_type='bpe'"):
            export()


def test_sentencepiece_user_defined_symbols_are_rejected():
    processor = _train_sentencepiece(user_defined_symbols=["<cat>"])
    with pytest.raises(ValueError, match="user-defined symbols such as '<cat>'"):
        SentencePieceBPETokenizer(processor)


def _word_tokenizers():
    from tokenizers import normalizers

    wordpiece = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    wordpiece.normalizer = normalizers.BertNormalizer(lowercase=True)
    wordpiece.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    wordpiece.train_from_iterator(
        _CORPUS, trainers.WordPieceTrainer(vocab_size=60, special_tokens=["[UNK]"]),
    )
    unigram = Tokenizer(models.Unigram())
    unigram.train_from_iterator(
        _CORPUS,
        trainers.UnigramTrainer(vocab_size=30, special_tokens=["<unk>"], unk_token="<unk>"),
    )
    words = Tokenizer(models.WordLevel({"[UNK]": 0, "the": 1, "cat": 2}, unk_token="[UNK]"))
    words.pre_tokenizer = pre_tokenizers.Whitespace()
    # Single-character vocabularies whose models still don't read one
    # character at a time.
    spaced = Tokenizer(models.WordLevel(
        {character: index for index, character in enumerate("?thecasdoglm ")}, unk_token="?",
    ))
    spaced.pre_tokenizer = pre_tokenizers.Whitespace()
    pieces = Tokenizer(models.WordPiece(
        {character: index for index, character in enumerate("?thecasdoglm ")}, unk_token="?",
    ))
    return {
        "wordpiece": (wordpiece, "[UNK]", "WordPiece tokenizers"),
        "unigram": (unigram, "<unk>", "Unigram tokenizers"),
        "wordlevel": (words, "[UNK]", "WordLevel tokenizers"),
        "wordlevel-characters": (spaced, "?", "one character at a"),
        "wordpiece-characters": (pieces, "?", "one character at a"),
    }


@pytest.mark.parametrize("kind", list(_word_tokenizers()))
def test_hugging_face_models_other_than_bpe_and_characters_are_rejected(kind):
    backend, unknown, message = _word_tokenizers()[kind]
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token=unknown)
    with pytest.raises(ValueError, match=message):
        transpile_tokenizer(tokenizer, SPRITE_PATH)
    with pytest.raises(ValueError, match="single characters|one character at a"):
        CharTokenizer(tokenizer).save(SPRITE_PATH)
