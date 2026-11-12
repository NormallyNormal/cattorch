"""
tokenizer.py
------------
Transpile a HuggingFace tokenizer into a Scratch sprite.

Supports character-level, raw-text BPE, and SentencePiece BPE tokenizers.
Every other kind (WordPiece, Unigram, word-level, and SentencePiece Unigram,
word and char models) is rejected with an error rather than exported wrong.

Usage
-----
    from transformers import AutoTokenizer
    from cattorch import CharTokenizer, BPETokenizer

    tokenizer = AutoTokenizer.from_pretrained("my-model")
    BPETokenizer(tokenizer).save("my_tokenizer")
"""

import itertools
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from cattorch import interface
from cattorch.codegen import CodegenConfig
from cattorch.errors import collect_warnings, with_warnings
from cattorch.results import TokenizerResult
from cattorch.sprite import (
    _EPOCH,
    _add_epoch_procedure,
    _add_warp_procedure,
    _epoch_current,
    _merge_variables_by_name,
    _procedure_mutation,
)
from cattorch.util.scratch.dsl import (
    Expr,
    Program,
    Statement,
    add,
    append,
    call,
    change_var,
    clear,
    costume_number,
    delete,
    eq,
    gt,
    if_,
    if_else,
    index_of,
    item,
    join,
    for_each,
    length,
    letter,
    lt,
    mul,
    not_,
    repeat,
    repeat_until,
    replace,
    set_var,
    string_length,
    sub,
    switch_costume,
    var,
)
from cattorch.util.scratch.finalize_scratch import finalize_sprite
from cattorch.util.scratch.interface import rename_lists, rename_procedure, rename_variables
from cattorch.util.scratch.ids import uniquify_data_ids

# The Scratch interface; see ``docs/scratch-interface.md``.
TEXT_VARIABLE = interface.TEXT
TOKEN_IDS_LIST = interface.TOKEN_IDS
# What tokenize and detokenize write, before they copy it to the public names.
_TOKENIZED = "_cattorch tokenized"
_DETOKENIZED = "_cattorch detokenized"

_SENTENCEPIECE_METHODS = (
    "get_piece_size",
    "id_to_piece",
    "is_byte",
    "is_control",
    "get_score",
    "unk_id",
)
# ``TrainerSpec.ModelType`` in sentencepiece_model.proto.
_SENTENCEPIECE_MODEL_TYPES = {1: "unigram", 2: "bpe", 3: "word", 4: "char"}
# ``SentencePiece.Type``.
_SENTENCEPIECE_USER_DEFINED = 4
# The costume shown between lookups; never the name of a single character.
_LOOKUP_COSTUME = "cat"


class _CharacterLookup:
    """Find each input character in a table of single characters.

    Scratch's ``item # of`` compares text case-insensitively, so it can look
    characters up only when none of them has another case (or when, as for a
    case-folding SentencePiece model, that is the intended meaning).
    Otherwise the sprite gets a costume named after each character and
    switches to the character: ``switch costume to`` matches costume names
    exactly, so ``A`` and ``a`` stay apart, and the costume number gives the
    character's position. A character with no costume leaves the lookup
    costume on, which marks it unknown. Every digit has a costume, since
    Scratch reads an unmatched digit as a costume number.
    """

    def __init__(self, characters: Sequence[str], *, case_insensitive: bool = False):
        self.characters = list(characters)
        self.exact = not case_insensitive and any(
            _scratch_cased(character) for character in self.characters
        )
        present = set(self.characters)
        unknown_digits = [digit for digit in "0123456789" if digit not in present]
        self.offset = 1 + len(unknown_digits)
        self.costume_names: tuple[str, ...] | None = (
            (_LOOKUP_COSTUME, *unknown_digits, *self.characters) if self.exact else None
        )

    def lists(self, characters_list: str) -> tuple[str, ...]:
        """The character list ``find`` searches, if any."""
        return () if self.exact else (characters_list,)

    def find(self, characters_list: str, character: Expr, result: str) -> tuple[Statement, ...]:
        """Set ``result`` to the character's 1-based position in the table."""
        if not self.exact:
            return (set_var(result, index_of(characters_list, character)),)
        return (
            switch_costume(_LOOKUP_COSTUME),
            switch_costume(character),
            set_var(result, sub(costume_number(), self.offset)),
        )

    def missing(self, result: str) -> Expr:
        """Whether ``find`` found no character."""
        return lt(var(result), 1) if self.exact else eq(var(result), 0)

    def finish(self) -> tuple[Statement, ...]:
        """Show the lookup costume again after encoding."""
        return (switch_costume(_LOOKUP_COSTUME),) if self.exact else ()


_LOWERCASE_TARGETS: frozenset[str] | None = None


def _scratch_cased(character: str) -> bool:
    """Whether Scratch's case-insensitive comparison equates ``character`` with another."""
    global _LOWERCASE_TARGETS
    if character.lower() != character.upper():
        return True
    if _LOWERCASE_TARGETS is None:
        # Characters such as the Kelvin sign lower-case to another one.
        _LOWERCASE_TARGETS = frozenset(
            lowered
            for character in map(chr, range(0x10000))
            for lowered in (character.lower(),)
            if lowered != character and len(lowered) == 1
        )
    return character in _LOWERCASE_TARGETS


_SUPPORTED_TYPES = (
    "a Hugging Face tokenizer (transformers' PreTrainedTokenizerFast or "
    "PreTrainedTokenizer, or a tokenizers.Tokenizer) or a "
    "sentencepiece.SentencePieceProcessor"
)


def _require_vocabulary(tokenizer, exporter: str) -> None:
    if not callable(getattr(tokenizer, "get_vocab", None)):
        raise TypeError(
            f"{exporter} needs {_SUPPORTED_TYPES}, not {type(tokenizer).__name__}"
        )


def _is_sentencepiece(processor) -> bool:
    return all(hasattr(processor, method) for method in _SENTENCEPIECE_METHODS)


def _backend(tokenizer):
    """The ``tokenizers.Tokenizer`` behind a Hugging Face tokenizer, if any.

    A ``tokenizers.Tokenizer`` is its own backend.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None and hasattr(tokenizer, "model"):
        backend = tokenizer
    if backend is None or not callable(getattr(backend, "to_str", None)):
        return None
    return backend


def _backend_settings(backend) -> dict:
    try:
        settings = json.loads(backend.to_str())
    except (TypeError, ValueError) as exc:
        raise TypeError("unable to read the Hugging Face tokenizer's settings") from exc
    if not isinstance(settings, dict) or not isinstance(settings.get("model"), dict):
        raise TypeError("unable to read the Hugging Face tokenizer's model")
    return settings


def _special_tokens(tokenizer) -> set[str]:
    """Tokens the tokenizer never produces from ordinary text."""
    special = getattr(tokenizer, "all_special_tokens", None)
    if special is not None:
        return set(special)
    added = getattr(tokenizer, "get_added_tokens_decoder", None)
    if callable(added):
        # A tokenizers.Tokenizer marks its special added tokens.
        return {
            token.content for token in added().values() if getattr(token, "special", False)
        }
    return set()


def _protobuf_fields(data: bytes):
    """Yield ``(field number, value)`` for each field of a protobuf message.

    Enough of the wire format to read a SentencePiece model's type without
    depending on protobuf, which sentencepiece doesn't require.
    """
    position = 0

    def varint() -> int:
        nonlocal position
        result = shift = 0
        while True:
            if position >= len(data):
                raise ValueError("truncated protobuf message")
            byte = data[position]
            position += 1
            result |= (byte & 0x7F) << shift
            if byte < 0x80:
                return result
            shift += 7

    while position < len(data):
        key = varint()
        field, wire_type = key >> 3, key & 7
        value: int | bytes
        if wire_type == 0:
            value = varint()
        elif wire_type == 1:
            value, position = data[position:position + 8], position + 8
        elif wire_type == 2:
            size = varint()
            value, position = data[position:position + size], position + size
        elif wire_type == 5:
            value, position = data[position:position + 4], position + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")
        if position > len(data):
            raise ValueError("truncated protobuf message")
        yield field, value


def _sentencepiece_model_summary(serialized: bytes) -> tuple[str, list[str]]:
    """The model's type and its user-defined pieces."""
    model_type = 1  # The proto2 default for TrainerSpec.model_type.
    user_defined: list[str] = []
    for field, value in _protobuf_fields(serialized):
        if field == 1 and isinstance(value, bytes):  # ModelProto.pieces
            piece, piece_type = "", 1
            for piece_field, piece_value in _protobuf_fields(value):
                if piece_field == 1 and isinstance(piece_value, bytes):
                    piece = piece_value.decode("utf-8", "replace")
                elif piece_field == 3 and isinstance(piece_value, int):
                    piece_type = piece_value
            if piece_type == _SENTENCEPIECE_USER_DEFINED:
                user_defined.append(piece)
        elif field == 2 and isinstance(value, bytes):  # ModelProto.trainer_spec
            for spec_field, spec_value in _protobuf_fields(value):
                if spec_field == 3 and isinstance(spec_value, int):
                    model_type = spec_value
    return _SENTENCEPIECE_MODEL_TYPES.get(model_type, f"type {model_type}"), user_defined


class _TokenizerBase(ABC):
    """Base class for tokenizer transpilers.

    Subclasses construct their Scratch program with the typed DSL. ``kind``
    is the ``TokenizerResult.tokenizer_type`` of the sprites they write.
    """

    kind: str
    def __init__(self, tokenizer) -> None:
        """
        Parameters
        ----------
        tokenizer : PreTrainedTokenizerBase
            A HuggingFace tokenizer. Must expose a ``.get_vocab()`` method.
        """
        if not isinstance(self, SentencePieceBPETokenizer):
            _require_vocabulary(tokenizer, type(self).__name__)
        self.tokenizer = tokenizer
        # Set by ``_build_sprite`` when the sprite looks characters up by costume.
        self._costume_names: tuple[str, ...] | None = None

    def _build_tokens_list(self) -> list[str]:
        vocab = self.tokenizer.get_vocab()
        max_id = max(vocab.values())
        tokens = [""] * (max_id + 1)
        for token_str, token_id in vocab.items():
            tokens[token_id] = token_str
        return tokens

    def _unknown_id(self) -> int | None:
        """Return the tokenizer's valid unknown ID, when it defines one."""
        value = getattr(self.tokenizer, "unk_token_id", None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value

        unknown = getattr(self.tokenizer, "unk_token", None)
        vocab = self.tokenizer.get_vocab()
        if isinstance(unknown, str) and unknown in vocab:
            return int(vocab[unknown])

        backend = _backend(self.tokenizer)
        if backend is not None:
            try:
                model = json.loads(backend.to_str()).get("model", {})
            except (TypeError, ValueError, json.JSONDecodeError):
                model = {}
            unknown = model.get("unk_token")
            if isinstance(unknown, str) and unknown in vocab:
                return int(vocab[unknown])
        return None

    @abstractmethod
    def _build_sprite(self, tokens: list[str]) -> dict:
        """Build the tokenizer sprite before common finalization."""

    def save(
        self,
        output_path: str | Path,
        *,
        name: str | None = None,
        codegen: CodegenConfig | None = None,
    ) -> TokenizerResult:
        """Write the tokenizer as a Scratch .sprite3 file.

        The generated sprite has two top-level block stacks:

        - **Encode**: ``cattorch tokenize`` reads the ``cattorch text``
          variable (a string) and writes token IDs to the
          ``cattorch token ids`` list.
        - **Decode**: ``cattorch detokenize`` reads ``cattorch token ids``
          and writes the decoded string to ``cattorch text``.

        Token IDs are 0-based, matching PyTorch embedding conventions.

        Parameters
        ----------
        output_path : str or Path
            Destination path. ``.sprite3`` is appended when absent.
        name : str, optional
            Display name inside Scratch. Defaults to the output path stem.
        codegen : CodegenConfig, optional
            JSON-size, loop-unrolling, and compact-ID controls.

        Its warnings are also emitted as ``CattorchWarning``.
        """
        with collect_warnings() as collected:
            result = self._save(output_path, name=name, codegen=codegen)
        return with_warnings(result, collected)

    def _save(
        self, output_path: str | Path, *, name: str | None, codegen: CodegenConfig | None,
    ) -> TokenizerResult:
        tokens = self._build_tokens_list()

        sprite = self._build_sprite(tokens)
        # The public interface: one text variable for both directions.
        rename_variables(sprite, {"input": TEXT_VARIABLE, "output": _DETOKENIZED})
        rename_lists(sprite, {"token_ids": TOKEN_IDS_LIST, "tokens": "_cattorch tokens"})
        sprite = _add_tokenizer_interface(sprite, len(tokens))

        uniquify_data_ids(sprite)
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".sprite3":
            output_path = Path(f"{output_path}.sprite3")
        sprite_name = output_path.stem if name is None else name
        if not isinstance(sprite_name, str) or not sprite_name.strip():
            raise ValueError("name must be a non-empty string")
        finalized = finalize_sprite(
            sprite, output_path, sprite_name=sprite_name, codegen=codegen,
            costume_names=self._costume_names,
        )
        return TokenizerResult(
            path=finalized.path,
            sprite_name=sprite_name,
            archive_bytes=finalized.archive_bytes,
            expanded_json_bytes=finalized.expanded_json_bytes,
            block_count=len(sprite.get("blocks", {})),
            list_count=len(sprite.get("lists", {})),
            sharded_lists=(),
            warnings=finalized.warnings,
            tokenizer_type=self.kind,
            token_count=len(tokens),
        )


def _add_tokenizer_interface(sprite: dict, vocabulary: int) -> dict:
    """Give the tokenize and detokenize blocks the interface every sprite's blocks have.

    Each sets ``cattorch status`` and sends its completion broadcast when it
    ends. Detokenize first checks that every token ID is a whole number in
    the vocabulary. Each writes into an internal list or variable, and
    copies it to the public one at its end only if it succeeded and no
    other call started meanwhile, so a call that fails writes nothing.
    """
    status, ok = interface.STATUS_VARIABLE, interface.OK
    current = _epoch_current()
    copy = "_cattorch commit index"
    for public, inner in (
        (interface.TOKENIZE, "_cattorch tokenize text"),
        (interface.DETOKENIZE, "_cattorch detokenize ids"),
    ):
        rename_procedure(sprite, public, inner)
        for block in sprite["blocks"].values():
            if block.get("topLevel") and block.get("opcode") == "procedures_call" and (
                block["mutation"].get("proccode") == inner
            ):
                block["mutation"] = _procedure_mutation(public)
    # A call it resumed during may have read what it wrote meanwhile, and
    # fails too.
    overtaken = (if_(not_(current), (
        change_var(_EPOCH, 1), set_var(status, interface.OVERTAKEN),
    )),)
    tokenize = Program(
        "tokenize_in_epoch",
        variables=(status, _EPOCH, copy),
        lists=(_TOKENIZED, TOKEN_IDS_LIST),
        body=(
            set_var(status, ok),
            call("_cattorch tokenize text"),
            if_(current, (
                clear(TOKEN_IDS_LIST),
                for_each(copy, length(_TOKENIZED), (
                    append(TOKEN_IDS_LIST, item(_TOKENIZED, var(copy))),
                )),
            )),
            *overtaken,
        ),
    )
    detokenize = Program(
        "detokenize_in_epoch",
        variables=(status, _EPOCH, _DETOKENIZED, TEXT_VARIABLE, "_cattorch check index"),
        lists=(TOKEN_IDS_LIST,),
        body=(
            set_var(status, ok),
            *interface.item_checks(TOKEN_IDS_LIST, interface.WHOLE_NUMBERS, vocabulary),
            if_(eq(var(status), ok), (
                call("_cattorch detokenize ids"),
                if_(current, (set_var(TEXT_VARIABLE, var(_DETOKENIZED)),)),
            )),
            *overtaken,
        ),
    )
    blocks = ((interface.TOKENIZE, tokenize), (interface.DETOKENIZE, detokenize))
    for y, (public, body) in enumerate(blocks):
        sprite = _add_epoch_procedure(
            sprite, public, body, x=640, y=360 * y, variables=(status,),
            completion=interface.completion(public),
        )
    _merge_variables_by_name(sprite, {TEXT_VARIABLE, status})
    return sprite


def _reject_non_bmp(tokens: Sequence[str]) -> None:
    """Reject vocab entries vanilla Scratch can't read one character at a time.

    Characters which Scratch compares as equal, such as ``a`` and ``A``, are
    told apart by ``_CharacterLookup``.
    """
    for token in tokens:
        if any(ord(character) > 0xFFFF for character in token):
            raise ValueError(
                "tokenizer vocabulary contains a non-BMP character which "
                "vanilla Scratch splits into UTF-16 surrogate halves"
            )


def _check_encodes_characters(
    tokenizer, tokens: Sequence[str], special: set[str], unknown_id: int,
) -> None:
    """Check that ``tokenizer`` maps each character to its own token.

    A word-level or WordPiece tokenizer whose vocabulary happens to be single
    characters, or one with a pre-tokenizer or a normalizer, encodes text
    differently from a character table. Encode a probe of every character,
    their other cases and unknown characters, and compare.
    """
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return
    ids: dict[str, int] = {}
    for token_id, token in enumerate(tokens):
        if len(token) == 1 and token not in special:
            ids.setdefault(token, token_id)
    # A character outside the vocabulary; any character not in it will do,
    # so the search ends within len(ids) + 1 code points.
    unknown = next(
        character for character in map(chr, itertools.count(0x2603)) if character not in ids
    )
    other_cases = [
        swapped for character in ids
        for swapped in (character.swapcase(),)
        if len(swapped) == 1 and swapped not in ids
    ]
    probe = "".join(ids) + unknown * 2 + "".join(other_cases) + "".join(ids)[:1]
    try:
        encoded = encode(probe, add_special_tokens=False)
    except TypeError:
        encoded = encode(probe)
    actual = [int(value) for value in getattr(encoded, "ids", encoded)]
    expected = [ids.get(character, unknown_id) for character in probe]
    if actual != expected:
        at = next(
            (index for index, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]),
            min(len(actual), len(expected)),
        )
        window = slice(max(0, at - 2), at + 3)
        raise ValueError(
            "CharTokenizer needs a tokenizer that encodes text one character at a "
            f"time, but this one doesn't: {probe[window]!r} gives token IDs "
            f"{actual[window]} rather than {expected[window]}. A pre-tokenizer, "
            "normalizer, or word-level model makes the difference"
        )


class CharTokenizer(_TokenizerBase):
    """Transpile a character-level tokenizer into a Scratch sprite.

    Each character in the input is mapped to a token ID via a vocab lookup.
    No merging is performed. The tokenizer's tokens must be single characters
    (special tokens aside), and it must encode text one character at a time,
    which is checked by encoding the vocabulary when it has ``encode``.

    Example
    -------
    ::

        from cattorch import CharTokenizer

        CharTokenizer(tokenizer).save("my_char_tok")
    """

    kind = "char"
    def _build_sprite(self, tokens: list[str]) -> dict:
        unknown_id = self._unknown_id()
        if unknown_id is None:
            raise ValueError(
                "CharTokenizer requires a tokenizer with a valid unknown-token ID"
            )
        # The unknown token is never read from text, whatever its length.
        special = _special_tokens(self.tokenizer) | {tokens[unknown_id]}
        words = [token for token in tokens if len(token) > 1 and token not in special]
        if words:
            raise ValueError(
                "CharTokenizer needs a tokenizer whose tokens are single characters, "
                f"but this vocabulary has multi-character tokens such as {words[0]!r}"
            )
        _reject_non_bmp([token for token in tokens if len(token) == 1])
        backend = _backend(self.tokenizer)
        if backend is not None:
            normalizer = _backend_settings(backend).get("normalizer")
            if normalizer is not None:
                raise ValueError(
                    "CharTokenizer doesn't support a normalizer "
                    f"({normalizer.get('type', 'unknown')}); the sprite looks up the "
                    "raw text, so tokens would differ"
                )
        _check_encodes_characters(self.tokenizer, tokens, special, unknown_id)
        sprite, self._costume_names = _build_character_tokenizer_sprite(
            tokens, bpe=False, unknown_id=unknown_id,
        )
        return sprite


class BPETokenizer(_TokenizerBase):
    """Transpile a BPE tokenizer into a Scratch sprite.

    The encoder splits the input into individual characters, then
    iteratively merges the highest-priority adjacent pair until no more
    merges are possible.  The tokenizer should be trained without a
    pre-tokenizer (or with character-level splitting) so that BPE
    operates on the full input string including spaces.

    Example
    -------
    ::

        from cattorch import BPETokenizer

        BPETokenizer(tokenizer).save("my_bpe_tok")
    """

    kind = "bpe"

    def _build_sprite(self, tokens: list[str]) -> dict:
        # Merges combine token IDs, so tokens are never compared as text and
        # may differ only in case.
        _reject_non_bmp(tokens)
        backend = _backend(self.tokenizer)
        if backend is None:
            raise TypeError(
                "BPETokenizer requires a Hugging Face fast tokenizer backend "
                "whose canonical merge order can be exported"
            )
        settings = _backend_settings(backend)
        model = settings["model"]
        if model.get("type") != "BPE":
            raise TypeError(
                f"BPETokenizer requires a BPE model, not {model.get('type', 'unknown')}"
            )
        if model.get("dropout") is not None:
            raise ValueError("BPE dropout is nondeterministic and cannot be exported")
        if model.get("continuing_subword_prefix") or model.get("end_of_word_suffix"):
            raise ValueError(
                "raw-text BPETokenizer does not support subword prefixes or suffixes"
            )
        for stage in ("normalizer", "pre_tokenizer"):
            if settings.get(stage) is not None:
                kind = settings[stage].get("type", "unknown")
                raise ValueError(
                    f"BPETokenizer doesn't support a {stage.replace('_', '-')} ({kind}); "
                    "the sprite merges the raw text, so tokens would differ"
                )
        for option in ("fuse_unk", "byte_fallback"):
            if model.get(option):
                raise ValueError(f"BPETokenizer doesn't support {option}=True")

        vocab = self.tokenizer.get_vocab()
        merges: list[tuple[int, int, int]] = []
        for raw_pair in model.get("merges", []):
            if isinstance(raw_pair, list) and len(raw_pair) == 2:
                left, right = raw_pair
            elif isinstance(raw_pair, str):
                parts = raw_pair.split(" ", 1)
                if len(parts) != 2:
                    raise ValueError(f"malformed BPE merge entry: {raw_pair!r}")
                left, right = parts
            else:
                raise ValueError(f"malformed BPE merge entry: {raw_pair!r}")
            merged = f"{left}{right}"
            if left not in vocab or right not in vocab or merged not in vocab:
                raise ValueError(
                    f"BPE merge {left!r} + {right!r} has no matching vocabulary entry"
                )
            merges.append((int(vocab[left]), int(vocab[right]), int(vocab[merged])))
        sprite, self._costume_names = _build_character_tokenizer_sprite(
            tokens,
            bpe=True,
            unknown_id=self._unknown_id(),
            merges=merges,
        )
        return sprite


def _build_character_tokenizer_sprite(
    tokens: list[str],
    *,
    bpe: bool,
    unknown_id: int | None,
    merges: Sequence[tuple[int, int, int]] = (),
) -> tuple[dict, tuple[str, ...] | None]:
    """Build the character/raw-BPE contract with the typed DSL.

    Returns the sprite and, when it looks characters up by costume, the
    costume names.
    """
    variables = (
        (
            "input", "output", "idx", "lookup", "candidate",
            "best_rank", "best_idx", "merging",
        )
        if bpe else ("input", "output", "idx", "lookup")
    )
    # Look characters up among single-character tokens only. Scratch's
    # item # of compares numbers by value, so searching every token would let
    # "1" match an earlier " 1" or "01".
    characters: list[str] = []
    character_ids: list[int] = []
    for token_id, token in enumerate(tokens):
        if len(token) == 1:
            characters.append(token)
            character_ids.append(token_id)
    lookup = _CharacterLookup(characters)
    lists = (
        ("tokens", "token_ids", _TOKENIZED, *lookup.lists("_characters"), "_character ids")
        + (("splitlist", "merge_keys", "merge_ids") if bpe else ())
    )
    values: dict[str, Sequence[int | float | str]] = {
        "tokens": tokens, "token_ids": [], _TOKENIZED: [],
        "_characters": characters, "_character ids": character_ids,
    }
    if lookup.exact:
        del values["_characters"]
    if bpe:
        values["splitlist"] = []
        vocab_size = len(tokens)
        values["merge_keys"] = [left * vocab_size + right for left, right, _ in merges]
        values["merge_ids"] = [merged for _, _, merged in merges]

    def append_lookup(destination: str) -> tuple[Statement, ...]:
        missing = () if unknown_id is None else (append(destination, unknown_id),)
        return (
            *lookup.find("_characters", letter(var("input"), var("idx")), "lookup"),
            if_else(
                lookup.missing("lookup"),
                missing,
                (append(destination, item("_character ids", var("lookup"))),),
            ),
        )

    split_input = (
        clear("splitlist"),
        set_var("idx", 0),
        repeat(
            string_length(var("input")),
            (
                change_var("idx", 1),
                *append_lookup("splitlist"),
            ),
        ),
    )
    merge_body: tuple[Statement, ...] = ()
    if bpe:
        scan = (
            change_var("idx", 1),
            set_var(
                "candidate",
                index_of(
                    "merge_keys",
                    add(
                        item("splitlist", add(var("idx"), 1)),
                        mul(item("splitlist", var("idx")), len(tokens)),
                    ),
                ),
            ),
            if_(
                gt(var("candidate"), 0),
                (
                    if_(
                        lt(var("candidate"), var("best_rank")),
                        (
                            set_var("best_rank", var("candidate")),
                            set_var("best_idx", var("idx")),
                        ),
                    ),
                ),
            ),
        )
        merge_once = (
            set_var("best_rank", len(merges) + 1),
            set_var("best_idx", 0),
            set_var("idx", 0),
            repeat(sub(length("splitlist"), 1), scan),
            if_else(
                eq(var("best_idx"), 0),
                (set_var("merging", 0),),
                (
                    replace(
                        "splitlist",
                        var("best_idx"),
                        item("merge_ids", var("best_rank")),
                    ),
                    delete("splitlist", add(var("best_idx"), 1)),
                ),
            ),
        )
        merge_body = (
            set_var("merging", 1),
            repeat_until(eq(var("merging"), 0), merge_once),
        )

    if bpe:
        encode_body: tuple[Statement, ...] = (
            *split_input,
            *merge_body,
            set_var("idx", 0),
            repeat(
                length("splitlist"),
                (
                    change_var("idx", 1),
                    append(_TOKENIZED, item("splitlist", var("idx"))),
                ),
            ),
            *lookup.finish(),
        )
    else:
        encode_body = (
            set_var("idx", 0),
            repeat(
                string_length(var("input")),
                (
                    change_var("idx", 1),
                    *append_lookup(_TOKENIZED),
                ),
            ),
            *lookup.finish(),
        )
    tokenize = Program(
        "BPE encode" if bpe else "character encode",
        variables=variables,
        lists=lists,
        list_values=values,
        body=(
            clear(_TOKENIZED),
            *encode_body,
        ),
    )
    detokenize = Program(
        "BPE decode" if bpe else "character decode",
        variables=variables,
        lists=lists,
        list_values=values,
        body=(
            set_var("output", ""),
            set_var("idx", 0),
            repeat(
                length("token_ids"),
                (
                    change_var("idx", 1),
                    set_var(
                        "output",
                        join(
                            var("output"),
                            item("tokens", add(item("token_ids", var("idx")), 1)),
                        ),
                    ),
                ),
            ),
        ),
    )
    sprite: dict = {
        "isStage": False,
        "name": "BPE tokenizer" if bpe else "character tokenizer",
        "variables": {}, "lists": {}, "broadcasts": {}, "blocks": {},
    }
    _add_warp_procedure(sprite, "cattorch tokenize", tokenize, x=320, y=0)
    _add_warp_procedure(sprite, "cattorch detokenize", detokenize, x=320, y=320)
    _add_top_level_call(sprite, "cattorch tokenize", x=0, y=0)
    _add_top_level_call(sprite, "cattorch detokenize", x=0, y=100)
    return sprite, lookup.costume_names


def _add_top_level_call(sprite: dict, procedure: str, *, x: int, y: int) -> None:
    """Add a clickable invocation stack for a tokenizer custom block."""
    block_id = f"cattorch_tokenizer_call_{len(sprite['blocks'])}"
    sprite["blocks"][block_id] = {
        "opcode": "procedures_call",
        "next": None,
        "parent": None,
        "inputs": {},
        "fields": {},
        "shadow": False,
        "topLevel": True,
        "x": x,
        "y": y,
        "mutation": _procedure_mutation(procedure),
    }


class SentencePieceBPETokenizer(_TokenizerBase):
    """Export a SentencePiece BPE tokenizer with Scratch-native procedures.

    The generated sprite exposes ``cattorch tokenize`` and
    ``cattorch detokenize`` as run-without-screen-refresh custom blocks, as
    well as one clickable stack for each procedure. SentencePiece's whitespace
    marker and control tokens are handled by the sprite.

    Characters are told apart by case, as SentencePiece does, using one
    costume per character (see ``_CharacterLookup``). Scratch's own string
    comparisons are case-insensitive, so ``scratch_casefold=True`` supports
    models trained with an NFKC case-folding normalizer for ASCII input
    without a separate lowercase pass or the costumes. Byte fallback works
    for ASCII; arbitrary out-of-vocabulary Unicode byte fallback cannot be
    reproduced by vanilla Scratch, which has no character-code reporter, so
    such characters encode as the unknown token.

    Parameters
    ----------
    tokenizer:
        A ``SentencePieceProcessor`` or an object exposing one as
        ``.processor``.
    scratch_casefold:
        Whether the model was trained with case-folding normalization.
    """

    kind = "sentencepiece-bpe"

    def __init__(self, tokenizer, *, scratch_casefold: bool = False):
        super().__init__(tokenizer)
        self.processor = getattr(tokenizer, "processor", tokenizer)
        missing = [
            name for name in _SENTENCEPIECE_METHODS if not hasattr(self.processor, name)
        ]
        if missing:
            raise TypeError(
                "SentencePieceBPETokenizer requires a SentencePiece processor; "
                f"missing: {', '.join(missing)}"
            )
        self.scratch_casefold = bool(scratch_casefold)
        self._validate_processor_configuration()
        self._special_ids = {
            token_id
            for token_id in range(self.processor.get_piece_size())
            if self.processor.is_control(token_id)
        }
        unknown_id = int(self.processor.unk_id())
        if unknown_id < 0:
            raise ValueError("SentencePiece model must define an unknown token")
        self.unknown_id = unknown_id
        self._special_ids.add(unknown_id)
        self._decode_has_non_ascii_bytes = False

    def _validate_processor_configuration(self) -> None:
        """Reject SentencePiece preprocessing Scratch cannot reproduce."""
        serialized = getattr(self.processor, "serialized_model_proto", None)
        if callable(serialized):
            try:
                model_type, user_defined = _sentencepiece_model_summary(serialized())
            except ValueError as exc:
                raise ValueError("unable to inspect the SentencePiece model") from exc
            if model_type != "bpe":
                raise ValueError(
                    f"this SentencePiece model is a {model_type} model, which "
                    "SentencePieceBPETokenizer can't reproduce; cattorch exports "
                    "SentencePiece models trained with model_type='bpe'"
                )
            if user_defined:
                raise ValueError(
                    "SentencePiece user-defined symbols such as "
                    f"{user_defined[0]!r} are matched before BPE, which the Scratch "
                    "tokenizer doesn't reproduce; train the model without "
                    "user_defined_symbols"
                )
            try:
                from sentencepiece import sentencepiece_model_pb2
            except ImportError:
                sentencepiece_model_pb2 = None
            if sentencepiece_model_pb2 is not None:
                model = sentencepiece_model_pb2.ModelProto()
                try:
                    model.ParseFromString(serialized())
                except Exception as exc:
                    raise ValueError("unable to inspect the SentencePiece model") from exc
                normalizer = model.normalizer_spec
                if normalizer.add_dummy_prefix:
                    raise ValueError(
                        "SentencePiece models with add_dummy_prefix=True cannot be "
                        "reproduced by the Scratch tokenizer"
                    )
                if normalizer.remove_extra_whitespaces:
                    raise ValueError(
                        "SentencePiece models with remove_extra_whitespaces=True cannot "
                        "be reproduced by the Scratch tokenizer"
                    )
                if normalizer.name not in {"identity", "nfkc_cf"}:
                    raise ValueError(
                        "SentencePiece normalization must be 'identity' or 'nfkc_cf' "
                        "for Scratch export"
                    )

        # sentencepiece does not depend on protobuf, so retain a behavioral
        # check when its generated ModelProto helper is unavailable.
        normalize = getattr(self.processor, "normalize", None)
        if callable(normalize):
            if str(normalize("cattorch")).startswith("▁"):
                raise ValueError(
                    "SentencePiece models with add_dummy_prefix=True cannot be "
                    "reproduced by the Scratch tokenizer"
                )
            whitespace_probe = "  a  b "
            normalized_probe = str(normalize(whitespace_probe)).replace("▁", " ")
            if normalized_probe != whitespace_probe:
                raise ValueError(
                    "SentencePiece normalization removes or changes whitespace "
                    "which the Scratch tokenizer preserves"
                )
            expected_case = "a" if self.scratch_casefold else "A"
            normalized_case = str(normalize("A")).replace("▁", " ")
            if normalized_case != expected_case:
                raise ValueError(
                    "scratch_casefold does not match the SentencePiece normalizer"
                )

    def _build_tokens_list(self) -> list[str]:
        raw_pieces = [
            self.processor.id_to_piece(token_id)
            for token_id in range(self.processor.get_piece_size())
        ]
        # A case-folding model is looked up case-insensitively, as Scratch
        # compares; any other model's characters are told apart exactly.
        fold = str.casefold if self.scratch_casefold else str
        normal_pieces = {
            fold(piece.replace("▁", " "))
            for token_id, piece in enumerate(raw_pieces)
            if not self.processor.is_byte(token_id)
            and not self.processor.is_control(token_id)
        }

        pieces: list[str] = []
        for token_id, raw_piece in enumerate(raw_pieces):
            if any(ord(character) > 0xFFFF for character in raw_piece):
                raise ValueError(
                    "SentencePiece vocabulary contains a non-BMP character which "
                    "vanilla Scratch splits into UTF-16 surrogate halves"
                )
            piece = raw_piece.replace("▁", " ")
            if self.processor.is_byte(token_id):
                try:
                    byte_value = int(raw_piece[3:5], 16)
                except (ValueError, IndexError) as exc:
                    raise ValueError(
                        f"malformed SentencePiece byte token: {raw_piece!r}"
                    ) from exc
                literal = chr(byte_value)
                # A byte stands for its ASCII character when no normal
                # token does (under Scratch's case-insensitive comparison,
                # for a case-folding model).
                if byte_value < 128 and fold(literal) not in normal_pieces:
                    piece = literal
            pieces.append(piece)
        return pieces

    def _tables(
        self,
        pieces: list[str],
    ) -> tuple[list[str], list[int], list[str], list[int], list[int]]:
        characters: list[str] = []
        character_ids: list[int] = []
        seen_characters: set[str] = set()
        decode_pieces: list[str] = []

        for token_id, piece in enumerate(pieces):
            if token_id in self._special_ids:
                decoded = ""
            elif self.processor.is_byte(token_id):
                raw_piece = self.processor.id_to_piece(token_id)
                byte_value = int(raw_piece[3:5], 16)
                if byte_value < 128:
                    decoded = chr(byte_value)
                else:
                    decoded = raw_piece
                    self._decode_has_non_ascii_bytes = True
            else:
                decoded = piece
            decode_pieces.append(decoded)

            if len(piece) != 1 or token_id in self._special_ids:
                continue
            if not self.scratch_casefold:
                # ``_CharacterLookup`` tells characters of different case apart.
                characters.append(piece)
                character_ids.append(token_id)
                continue
            comparison_key = piece.casefold()
            if comparison_key in seen_characters:
                if piece == comparison_key:
                    # The model case-folds its input, so the folded form,
                    # such as the byte for "q" rather than "Q", is the match.
                    earlier = [character.casefold() for character in characters].index(
                        comparison_key,
                    )
                    characters[earlier], character_ids[earlier] = piece, token_id
                continue
            seen_characters.add(comparison_key)
            characters.append(piece)
            character_ids.append(token_id)

        # SentencePiece BPE chooses the highest-scoring token available for an
        # adjacent pair. Encode a pair of token IDs as one exact Scratch number
        # so the hot scan uses numeric lookup instead of joining strings and
        # searching the entire piece vocabulary.
        # A case-folding model's pieces are matched as Scratch compares them.
        # Otherwise they match exactly, and byte tokens, which SentencePiece
        # only emits for characters outside the vocabulary, never merge.
        fold = str.casefold if self.scratch_casefold else str
        piece_ids: dict[str, int] = {}
        for token_id, piece in enumerate(pieces):
            if self.scratch_casefold or not self.processor.is_byte(token_id):
                piece_ids.setdefault(fold(piece), token_id)
        vocab_size = len(pieces)
        merge_entries: list[tuple[float, int, int]] = []
        for merged_id, piece in enumerate(pieces):
            if merged_id in self._special_ids or self.processor.is_byte(merged_id):
                continue
            for split_at in range(1, len(piece)):
                left_id = piece_ids.get(fold(piece[:split_at]))
                right_id = piece_ids.get(fold(piece[split_at:]))
                if left_id is None or right_id is None:
                    continue
                if left_id in self._special_ids or right_id in self._special_ids:
                    continue
                pair_key = left_id * vocab_size + right_id
                merge_entries.append(
                    (float(self.processor.get_score(merged_id)), pair_key, merged_id)
                )
        merge_entries.sort(key=lambda entry: (-entry[0], entry[2], entry[1]))
        merge_keys = [entry[1] for entry in merge_entries]
        merge_ids = [entry[2] for entry in merge_entries]
        if len(merge_keys) != len(set(merge_keys)):
            raise ValueError("SentencePiece model gives one token pair multiple merges")

        return characters, character_ids, decode_pieces, merge_keys, merge_ids

    def _build_sprite(self, pieces: list[str]) -> dict:
        (
            characters,
            character_ids,
            decode_pieces,
            merge_keys,
            merge_ids,
        ) = self._tables(pieces)
        lookup = _CharacterLookup(characters, case_insensitive=self.scratch_casefold)
        self._costume_names = lookup.costume_names
        variables = (
            "input",
            "output",
            "_sp i",
            "_sp lookup",
            "_sp candidate",
            "_sp best rank",
            "_sp best index",
            "_sp merging",
            "_sp token id",
        )
        lists = (
            "tokens",
            "token_ids",
            _TOKENIZED,
            "_sp split",
            *lookup.lists("_sp characters"),
            "_sp character ids",
            "_sp decoded tokens",
            "_sp merge keys",
            "_sp merge ids",
        )
        list_values = {
            "tokens": pieces,
            "token_ids": [],
            _TOKENIZED: [],
            "_sp split": [],
            "_sp characters": characters,
            "_sp character ids": character_ids,
            "_sp decoded tokens": decode_pieces,
            "_sp merge keys": merge_keys,
            "_sp merge ids": merge_ids,
        }
        if lookup.exact:
            del list_values["_sp characters"]

        scan_pair = (
            change_var("_sp i", 1),
            set_var(
                "_sp candidate",
                index_of(
                    "_sp merge keys",
                    add(
                        item("_sp split", add(var("_sp i"), 1)),
                        # Numeric pair encoding stays below 2^53 for any
                        # Scratch-sized tokenizer vocabulary.
                        mul(
                            item("_sp split", var("_sp i")),
                            len(pieces),
                        ),
                    ),
                ),
            ),
            if_(
                gt(var("_sp candidate"), 0),
                (
                    if_(
                        lt(var("_sp candidate"), var("_sp best rank")),
                        (
                            set_var("_sp best rank", var("_sp candidate")),
                            set_var("_sp best index", var("_sp i")),
                        ),
                    ),
                ),
            ),
        )
        merge_once = (
            # A SentencePiece token can have several valid left/right
            # decompositions, so the merge table can be longer than the
            # vocabulary. Use the table length as the sentinel; otherwise
            # valid lower-priority merges past ``len(pieces)`` are ignored.
            set_var("_sp best rank", len(merge_keys) + 1),
            set_var("_sp best index", 0),
            set_var("_sp i", 0),
            repeat(sub(length("_sp split"), 1), scan_pair),
            if_else(
                eq(var("_sp best index"), 0),
                (set_var("_sp merging", 0),),
                (
                    replace(
                        "_sp split",
                        var("_sp best index"),
                        item("_sp merge ids", var("_sp best rank")),
                    ),
                    delete("_sp split", add(var("_sp best index"), 1)),
                ),
            ),
        )
        tokenize = Program(
            "SentencePiece BPE encode",
            variables=variables,
            lists=lists,
            list_values=list_values,
            body=(
                clear(_TOKENIZED),
                clear("_sp split"),
                set_var("_sp i", 0),
                repeat(
                    string_length(var("input")),
                    (
                        change_var("_sp i", 1),
                        *lookup.find(
                            "_sp characters",
                            letter(var("input"), var("_sp i")),
                            "_sp lookup",
                        ),
                        if_else(
                            lookup.missing("_sp lookup"),
                            # SentencePiece reads a run of unknown characters
                            # as one unknown token.
                            (
                                if_else(
                                    eq(item("_sp split", length("_sp split")), self.unknown_id),
                                    (),
                                    (append("_sp split", self.unknown_id),),
                                ),
                            ),
                            (
                                set_var(
                                    "_sp token id",
                                    item("_sp character ids", var("_sp lookup")),
                                ),
                                append(
                                    "_sp split",
                                    var("_sp token id"),
                                ),
                            ),
                        ),
                    ),
                ),
                set_var("_sp merging", 1),
                repeat_until(eq(var("_sp merging"), 0), merge_once),
                set_var("_sp i", 0),
                repeat(
                    length("_sp split"),
                    (
                        change_var("_sp i", 1),
                        append(
                            _TOKENIZED,
                            item("_sp split", var("_sp i")),
                        ),
                    ),
                ),
                *lookup.finish(),
            ),
        )
        detokenize = Program(
            "SentencePiece BPE decode",
            variables=variables,
            lists=lists,
            list_values=list_values,
            body=(
                set_var("output", ""),
                set_var("_sp i", 0),
                repeat(
                    length("token_ids"),
                    (
                        change_var("_sp i", 1),
                        set_var(
                            "_sp token id",
                            item("token_ids", var("_sp i")),
                        ),
                        set_var(
                            "output",
                            join(
                                var("output"),
                                item(
                                    "_sp decoded tokens",
                                    add(var("_sp token id"), 1),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

        sprite = {
            "isStage": False,
            "name": "SentencePiece tokenizer",
            "variables": {},
            "lists": {},
            "broadcasts": {},
            "blocks": {},
        }
        _add_warp_procedure(sprite, "cattorch tokenize", tokenize, x=320, y=0)
        _add_warp_procedure(sprite, "cattorch detokenize", detokenize, x=320, y=360)
        _add_top_level_call(sprite, "cattorch tokenize", x=0, y=0)
        _add_top_level_call(sprite, "cattorch detokenize", x=0, y=100)
        return sprite

    def _save(
        self, output_path: str | Path, *, name: str | None, codegen: CodegenConfig | None,
    ) -> TokenizerResult:
        result = super()._save(output_path, name=name, codegen=codegen)
        if not self._decode_has_non_ascii_bytes:
            return result
        warning = (
            "Vanilla Scratch cannot convert between characters and arbitrary "
            "UTF-8 byte-fallback sequences. ASCII bytes encode and decode "
            "exactly; a non-ASCII character outside the vocabulary encodes as "
            "the unknown token rather than its UTF-8 bytes, and sampled "
            "non-ASCII byte tokens remain visible as <0xNN> markers.",
        )
        return TokenizerResult(
            path=result.path,
            sprite_name=result.sprite_name,
            archive_bytes=result.archive_bytes,
            expanded_json_bytes=result.expanded_json_bytes,
            block_count=result.block_count,
            list_count=result.list_count,
            sharded_lists=result.sharded_lists,
            warnings=result.warnings + warning,
            tokenizer_type=result.tokenizer_type,
            token_count=result.token_count,
        )


def transpile_tokenizer(
    tokenizer,
    output_path: str | Path,
    *,
    name: str | None = None,
    scratch_casefold: bool = False,
    codegen: CodegenConfig | None = None,
) -> TokenizerResult:
    """Transpile a tokenizer into a Scratch .sprite3 file.

    Detects a SentencePiece BPE model, a Hugging Face BPE tokenizer, or a
    character-level tokenizer, and exports it with ``SentencePieceBPETokenizer``,
    ``BPETokenizer`` or ``CharTokenizer``; use one of those to choose the
    kind yourself. Every other kind, such as WordPiece, Unigram, or a
    SentencePiece Unigram model, raises ``ValueError``.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        A HuggingFace tokenizer.
    output_path : str or Path
        Destination path. ``.sprite3`` is appended when absent.
    name : str, optional
        Display name inside Scratch. Defaults to the output path stem.
    scratch_casefold : bool, optional
        Set for a SentencePiece model trained with case-folding normalization,
        as ``SentencePieceBPETokenizer(scratch_casefold=True)``. Other kinds
        of tokenizer raise ``ValueError`` when it is set.
    codegen : CodegenConfig, optional
        JSON-size, loop-unrolling, and compact-ID controls.

    Returns
    -------
    TokenizerResult
        Its ``tokenizer_type`` says which kind was exported.
    """
    processor = getattr(tokenizer, "processor", tokenizer)
    if scratch_casefold and not _is_sentencepiece(processor):
        raise ValueError(
            "scratch_casefold applies to a SentencePiece model trained with case "
            "folding, and this tokenizer isn't a SentencePiece model"
        )
    if _is_sentencepiece(processor):
        # The constructor rejects SentencePiece models other than BPE.
        return SentencePieceBPETokenizer(
            tokenizer,
            scratch_casefold=scratch_casefold,
        ).save(output_path, name=name, codegen=codegen)

    _require_vocabulary(tokenizer, "transpile_tokenizer")
    backend = _backend(tokenizer)
    if backend is not None:
        model_type = _backend_settings(backend)["model"].get("type", "unknown")
        if model_type == "BPE":
            return BPETokenizer(tokenizer).save(output_path, name=name, codegen=codegen)
    else:
        model_type = type(tokenizer).__name__
    # Anything else must be a character tokenizer: a WordPiece, WordLevel or
    # Unigram vocabulary of words would be encoded one character at a time.
    # CharTokenizer also checks that the tokenizer encodes characters alone.
    special = _special_tokens(tokenizer)
    unknown = getattr(tokenizer, "unk_token", None)
    if isinstance(unknown, str):
        special.add(unknown)
    if backend is not None:
        backend_unknown = _backend_settings(backend)["model"].get("unk_token")
        if isinstance(backend_unknown, str):
            special.add(backend_unknown)
    if any(len(token) > 1 for token in tokenizer.get_vocab() if token not in special):
        raise ValueError(
            f"{model_type} tokenizers with multi-character tokens can't be exported; "
            "transpile_tokenizer supports BPE, SentencePiece BPE, and tokenizers "
            "whose tokens are single characters"
        )
    return CharTokenizer(tokenizer).save(output_path, name=name, codegen=codegen)
