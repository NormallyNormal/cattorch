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
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
from typing import Iterator, Sequence

from cattorch import interface
from cattorch.codegen import CodegenConfig
from cattorch.errors import collect_warnings, with_warnings
from cattorch.results import TokenizerResult
from cattorch.guards import _EPOCH, _add_epoch_procedure, _epoch_current, _overtaken
from cattorch.procedures import _add_warp_procedure, _merge_variables_by_name, _procedure_mutation
from cattorch.util.paths import output_file
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
# SentencePiece's ``kDefaultUnknownSymbol``: what it decodes the unknown token to.
_SENTENCEPIECE_UNKNOWN_SURFACE = " \u2047 "
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


_HUGGING_FACE_TYPES = (
    "a Hugging Face tokenizer (transformers' PreTrainedTokenizerFast or "
    "PreTrainedTokenizer, or a tokenizers.Tokenizer)"
)
_SUPPORTED_TYPES = f"{_HUGGING_FACE_TYPES} or a sentencepiece.SentencePieceProcessor"


def _require_vocabulary(tokenizer, exporter: str) -> None:
    """``tokenizer`` must be one ``exporter`` takes: any supported kind for
    ``transpile_tokenizer``, a Hugging Face tokenizer for the others."""
    if callable(getattr(tokenizer, "get_vocab", None)):
        return
    if exporter == "transpile_tokenizer":
        raise TypeError(f"{exporter} needs {_SUPPORTED_TYPES}, not {type(tokenizer).__name__}")
    hint = (
        "; export a SentencePiece model with SentencePieceBPETokenizer"
        if _is_sentencepiece(getattr(tokenizer, "processor", tokenizer)) else ""
    )
    raise TypeError(
        f"{exporter} needs {_HUGGING_FACE_TYPES}, not {type(tokenizer).__name__}{hint}"
    )


def _check_scratch_casefold(value) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"scratch_casefold must be True or False, not {value!r}")


def _check_codegen(name, codegen) -> None:
    """Check a tokenizer export's ``name`` and ``codegen`` as ``transpile`` does.

    A tokenizer has no layers or loops to generate, so ``CodegenConfig``'s
    ``layer_sharing`` and ``unrolling`` can't apply and raise when set.
    """
    from cattorch.export_options import validate_export_options

    validate_export_options(name=name, codegen=codegen)
    if codegen is not None:
        for field, default in (("layer_sharing", False), ("unrolling", "auto")):
            if getattr(codegen, field) != default:
                raise ValueError(
                    f"CodegenConfig.{field} applies to model exports; a tokenizer has no "
                    "layers or loops for it to change"
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
        # Set by ``_build_sprite``: the token IDs encode puts before and
        # after the text's, as the tokenizer's post-processor does.
        self._wrap: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())

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
            JSON-size and compact-ID controls; ``layer_sharing`` and
            ``unrolling`` apply only to models, and raise ``ValueError``.

        Its warnings are also emitted as ``CattorchWarning``.
        """
        _check_codegen(name, codegen)
        with collect_warnings() as collected, _unchanged(self.tokenizer):
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
        sprite = _add_tokenizer_interface(sprite, len(tokens), *self._wrap)
        # The text starts empty, so tokenize before any text is set reads
        # no text rather than "0".
        for entry in sprite["variables"].values():
            if entry[0] in (TEXT_VARIABLE, _DETOKENIZED):
                entry[1] = ""

        uniquify_data_ids(sprite)
        output_path = Path(output_file(output_path, ".sprite3"))
        sprite_name = interface.check_export(
            output_path.stem if name is None else name, (interface.TOKENIZE, interface.DETOKENIZE),
        )
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
            broadcasts=tuple(
                interface.message(sprite_name, block)
                for block in (interface.TOKENIZE, interface.DETOKENIZE)
            ),
        )


def _add_tokenizer_interface(
    sprite: dict, vocabulary: int, prefix: Sequence[int] = (), suffix: Sequence[int] = (),
) -> dict:
    """Give the tokenize and detokenize blocks the interface every sprite's blocks have.

    Each sets ``cattorch status`` and sends its completion broadcast when it
    ends. Detokenize first checks that every token ID is a whole number in
    the vocabulary. Each writes into an internal list or variable, and
    copies it to the public one at its end only if it succeeded and no
    other call started meanwhile, so a call that fails writes nothing.
    Tokenize writes ``prefix`` and ``suffix`` around the text's token IDs.
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
    overtaken = (if_(not_(current), _overtaken()),)
    tokenize = Program(
        "tokenize_in_epoch",
        variables=(status, _EPOCH, copy),
        lists=(_TOKENIZED, TOKEN_IDS_LIST),
        body=(
            set_var(status, ok),
            call("_cattorch tokenize text"),
            if_(current, (
                clear(TOKEN_IDS_LIST),
                *(append(TOKEN_IDS_LIST, token_id) for token_id in prefix),
                for_each(copy, length(_TOKENIZED), (
                    append(TOKEN_IDS_LIST, item(_TOKENIZED, var(copy))),
                )),
                *(append(TOKEN_IDS_LIST, token_id) for token_id in suffix),
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


@contextmanager
def _unchanged(tokenizer) -> Iterator[None]:
    """Leave a Hugging Face tokenizer's backend settings as they were.

    transformers' ``encode``, which the export's checks call, sets the
    backend's ``encode_special_tokens`` and turns off its truncation and
    padding.
    """
    backend = _backend(tokenizer)
    if backend is None:
        yield
        return
    saved: dict = {}
    for setting in ("encode_special_tokens", "truncation", "padding"):
        try:
            saved[setting] = getattr(backend, setting)
        except Exception:  # noqa: BLE001 - a backend without the setting
            pass
    try:
        yield
    finally:
        if "encode_special_tokens" in saved:
            backend.encode_special_tokens = saved["encode_special_tokens"]
        if "truncation" in saved:
            if saved["truncation"] is None:
                backend.no_truncation()
            else:
                backend.enable_truncation(**saved["truncation"])
        if "padding" in saved:
            if saved["padding"] is None:
                backend.no_padding()
            else:
                backend.enable_padding(**saved["padding"])


def _post_processor_wrap(processor: dict | None, exporter: str) -> tuple[list[int], list[int]]:
    """The token IDs a ``tokenizers`` post-processor adds before and after
    one sequence's, from its settings.

    ``TemplateProcessing`` (with one ``$A`` in its single template),
    ``BertProcessing`` and ``RobertaProcessing`` add fixed tokens;
    ``ByteLevel`` only moves offsets; a ``Sequence`` applies each in turn.
    Others raise ``ValueError``.
    """
    if processor is None:
        return [], []
    kind = processor.get("type")
    if kind == "ByteLevel":
        return [], []
    if kind in ("BertProcessing", "RobertaProcessing"):
        return [int(processor["cls"][1])], [int(processor["sep"][1])]
    if kind == "Sequence":
        prefix: list[int] = []
        suffix: list[int] = []
        for step in processor.get("processors", []):
            step_prefix, step_suffix = _post_processor_wrap(step, exporter)
            prefix, suffix = step_prefix + prefix, suffix + step_suffix
        return prefix, suffix
    if kind == "TemplateProcessing":
        pieces = processor.get("single", [])
        sequences = [index for index, piece in enumerate(pieces) if "Sequence" in piece]
        if len(sequences) == 1:
            special = processor.get("special_tokens", {})
            before: list[int] = []
            after: list[int] = []
            for index, piece in enumerate(pieces):
                if "SpecialToken" in piece:
                    ids = special[piece["SpecialToken"]["id"]]["ids"]
                    (before if index < sequences[0] else after).extend(map(int, ids))
            return before, after
    raise ValueError(
        f"{exporter} can't reproduce this tokenizer's {kind or 'unknown'} post-processor, "
        "which changes the token IDs encode gives; it reproduces TemplateProcessing with one "
        "sequence, BertProcessing, RobertaProcessing and ByteLevel"
    )


def _special_token_wrap(
    tokenizer, exporter: str, tokens: Sequence[str], texts: Sequence[str],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The token IDs ``encode(text)`` adds before and after the text's.

    A post-processor such as ``TemplateProcessing("<s> $A")`` adds them when
    encoding with its default ``add_special_tokens=True``, and the sprite
    writes them too. They are read from the post-processor's settings, or,
    without a ``tokenizers`` backend, from what ``encode`` adds to a probe;
    either way the export checks them against ``encode`` on ``texts``.
    """
    if not callable(getattr(tokenizer, "encode", None)):
        return (), ()
    backend = _backend(tokenizer)
    if backend is not None:
        prefix, suffix = _post_processor_wrap(
            _backend_settings(backend).get("post_processor"), exporter,
        )
    else:
        probe = "a"
        bare = _encode_ids(tokenizer, probe)
        full = _encode_ids(tokenizer, probe, special_tokens=True)
        at = next(
            (start for start in range(len(full) - len(bare) + 1)
             if full[start:start + len(bare)] == bare), None,
        )
        if at is None:
            raise ValueError(
                f"{exporter} can't reproduce the special tokens this tokenizer adds: "
                f"encode({probe!r}) gives {full}, which doesn't contain the text's {bare}"
            )
        prefix, suffix = full[:at], full[at + len(bare):]
    for token_id in (*prefix, *suffix):
        if not 0 <= token_id < len(tokens):
            raise ValueError(
                f"{exporter}: the tokenizer's post-processor adds token ID {token_id}, which "
                f"isn't in its vocabulary of {len(tokens)}"
            )
    for text in ("", "a", *texts[:64]):
        bare = _encode_ids(tokenizer, text)
        full = _encode_ids(tokenizer, text, special_tokens=True)
        if full != [*prefix, *bare, *suffix]:
            raise ValueError(
                f"{exporter} can't reproduce the special tokens this tokenizer adds: "
                f"encode({text!r}) gives {full}, but the sprite would give "
                f"{[*prefix, *bare, *suffix]}"
            )
    return tuple(prefix), tuple(suffix)


def _reject_non_bmp(tokens: Sequence[str]) -> None:
    """Reject vocab entries vanilla Scratch can't read one character at a time.

    Characters which Scratch compares as equal, such as ``a`` and ``A``, are
    told apart by ``_CharacterLookup``.
    """
    for token in tokens:
        if any(ord(character) > 0xFFFF for character in token):
            raise ValueError(
                f"tokenizer vocabulary contains a non-BMP character, in {token!r}, which "
                "vanilla Scratch splits into UTF-16 surrogate halves"
            )


def _encode_ids(tokenizer, text: str, *, special_tokens: bool = False) -> list[int]:
    """The token IDs ``tokenizer`` gives ``text``, with or without the special
    tokens its post-processor adds."""
    encode = tokenizer.encode
    try:
        encoded = encode(text, add_special_tokens=special_tokens)
    except TypeError:
        encoded = encode(text)
    return [int(value) for value in getattr(encoded, "ids", encoded)]


def _check_encodes_characters(
    tokenizer, matching: "_Matching", unknown_id: int,
) -> None:
    """Check that ``tokenizer`` maps each character to its own token.

    A word-level or WordPiece tokenizer whose vocabulary happens to be single
    characters, or one with a pre-tokenizer or a normalizer, encodes text
    differently from a character table. Encode a probe of every character in
    the sprite's table, their other cases and unknown characters, and compare
    with what the sprite does.
    """
    if not callable(getattr(tokenizer, "encode", None)):
        return
    ids = dict.fromkeys(matching.characters)
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
    actual = _encode_ids(tokenizer, probe)
    expected = _reference_ids(probe, matching, unknown_id, None)
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


# ── Added tokens ────────────────────────────────────────────────────────────
#
# Hugging Face tokenizers match their added tokens (``add_tokens``,
# ``add_special_tokens`` and a trainer's special tokens) in the raw text
# before the model runs, and the model tokenizes only the text between them.
# ``tokenizers`` does it in two passes: tokens with ``normalized=False`` on the
# whole text, then tokens with ``normalized=True`` on each piece left over
# (there is no normalizer here to apply in between). Each pass takes the
# leftmost-longest matches, left to right; ``lstrip`` and ``rstrip`` widen a
# match over the whitespace beside it, which no token then gets.

# Rust's ``char::is_whitespace`` (Unicode White_Space), which ``lstrip`` and
# ``rstrip`` strip.
_UNICODE_WHITE_SPACE = (
    "\t\n\v\f\r \x85\xa0          "
    "      　"
)


@dataclass(frozen=True)
class _AddedToken:
    content: str
    token_id: int
    normalized: bool
    lstrip: bool
    rstrip: bool


def _split_special_tokens(tokenizer) -> bool:
    """Whether ``tokenizer.encode`` reads special tokens as ordinary text.

    transformers' ``split_special_tokens`` decides, and transformers sets the
    backend's ``encode_special_tokens`` from it on every call, so the backend
    flag only records the last call. A bare ``tokenizers.Tokenizer`` has only
    the flag.
    """
    split = getattr(tokenizer, "split_special_tokens", None)
    if isinstance(split, bool):
        return split
    backend = _backend(tokenizer)
    return backend is not None and bool(getattr(backend, "encode_special_tokens", False))


def _added_token_split(tokenizer, exporter: str) -> tuple[list[_AddedToken], list[_AddedToken]]:
    """The added tokens the tokenizer matches in text, and those it reads as
    ordinary text, each in ID order.

    With ``split_special_tokens``, a fast tokenizer reads its special tokens
    as text, and a Python (slow) one all its added tokens. ``single_word``
    tokens are rejected: whether a match stands alone depends on whether its
    neighbours are Unicode letters or digits, which a sprite can't tell for
    characters outside its tables.
    """
    backend = _backend(tokenizer)
    split = _split_special_tokens(tokenizer)
    if backend is not None and callable(getattr(backend, "get_added_tokens_decoder", None)):
        decoder = backend.get_added_tokens_decoder()
        skip_all = False
    else:
        decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
        skip_all = split
    added: list[_AddedToken] = []
    skipped: list[_AddedToken] = []
    for token_id, token in sorted(decoder.items()):
        content = getattr(token, "content", token)
        if not isinstance(content, str) or not content:
            continue
        special = bool(getattr(token, "special", False))
        entry = _AddedToken(
            content, int(token_id), bool(getattr(token, "normalized", not special)),
            bool(getattr(token, "lstrip", False)), bool(getattr(token, "rstrip", False)),
        )
        if skip_all or (special and split):
            skipped.append(entry)
            continue
        if getattr(token, "single_word", False):
            raise ValueError(
                f"{exporter} doesn't support added tokens with single_word=True, such as "
                f"{content!r}: whether a match stands alone depends on its neighbours "
                "being letters or digits, which a sprite can't tell for every character"
            )
        added.append(entry)
    return added, skipped


@dataclass(frozen=True)
class _Matching:
    """What a character or raw BPE sprite matches text with.

    ``characters`` and ``character_ids`` are its table of single characters,
    in ID order (the first of equal characters wins); ``added`` the added
    tokens it matches before looking characters up. ``probes`` are the
    contents of every added token, including those left out of ``added``
    and those the tokenizer reads as text, for the export's checks.
    """
    characters: tuple[str, ...]
    character_ids: tuple[int, ...]
    added: tuple[_AddedToken, ...]
    probes: tuple[str, ...]


def _sprite_matching(
    tokenizer, exporter: str, tokens: Sequence[str], unknown_id: int | None,
    merges: Sequence[tuple[int, int, int]] = (),
) -> _Matching:
    """Decide the sprite's character table and added tokens.

    The sprite, ``_reference_ids`` and the export's checks all use what this
    returns, so the checks test exactly what the sprite does.
    """
    added, skipped = _added_token_split(tokenizer, exporter)
    # A token the tokenizer reads as text is the model's to encode: when the
    # model doesn't know it, it isn't in the character table.
    left_out: set[int] = set()
    for token in skipped:
        if len(token.content) != 1 or not callable(getattr(tokenizer, "encode", None)):
            continue
        ids = _encode_ids(tokenizer, token.content)
        if ids == [token.token_id]:
            continue
        if ids in ([], [unknown_id]):
            left_out.add(token.token_id)
            continue
        raise ValueError(
            f"{exporter} can't reproduce how this tokenizer encodes {token.content!r}, a "
            f"special token it reads as text (split_special_tokens): it gives token IDs {ids}"
        )
    characters: list[str] = []
    character_ids: list[int] = []
    for token_id, text in enumerate(tokens):
        if len(text) == 1 and token_id not in left_out:
            characters.append(text)
            character_ids.append(token_id)
    # A one-character added token matches the same with or without the
    # added-token pass when the character lookup already gives its ID, no
    # merge combines it and no other added token contains it, unless it is
    # whitespace and a token strips whitespace: then matching it ends the
    # whitespace lstrip takes, and tokenizers emits it inside what rstrip took.
    merged_ids = {token_id for left, right, _ in merges for token_id in (left, right)}
    first_ids = dict(zip(reversed(characters), reversed(character_ids)))
    strips = any(token.lstrip or token.rstrip for token in added)
    matched = [
        token for token in added
        if not (
            len(token.content) == 1 and not token.lstrip and not token.rstrip
            and first_ids.get(token.content) == token.token_id
            and token.token_id not in merged_ids
            and not (strips and token.content in _UNICODE_WHITE_SPACE)
            and not any(
                token.content in other.content for other in added if other is not token
            )
        )
    ]
    # The sprite reads text in UTF-16 units, so it can't match a character
    # outside the BMP, alone or in an added token.
    _reject_non_bmp([*characters, *(token.content for token in matched)])
    # The checks probe the first tokens most: put those that change how
    # others match (one character, or stripping whitespace) first.
    probes = sorted(
        (*added, *skipped),
        key=lambda token: not (len(token.content) == 1 or token.lstrip or token.rstrip),
    )
    return _Matching(
        tuple(characters), tuple(character_ids), tuple(matched),
        tuple(token.content for token in probes),
    )


def _match_added(text: str, tokens: Sequence[_AddedToken]) -> list[tuple[int | None, str]]:
    """One pass of ``tokenizers``' added-token matching: ``(token ID, text)``
    for each match and ``(None, text)`` for the text between matches."""
    if not text:
        return []
    candidates = sorted(tokens, key=lambda token: -len(token.content))
    pieces: list[tuple[int | None, str]] = []
    start_offset = position = 0
    while position < len(text):
        match = next(
            (token for token in candidates if text.startswith(token.content, position)), None,
        )
        if match is None:
            position += 1
            continue
        start, stop = position, position + len(match.content)
        if match.lstrip:
            while start > 0 and text[start - 1] in _UNICODE_WHITE_SPACE:
                start -= 1
            start = max(start, start_offset)
        if match.rstrip:
            while stop < len(text) and text[stop] in _UNICODE_WHITE_SPACE:
                stop += 1
        if start_offset < start:
            pieces.append((None, text[start_offset:start]))
        pieces.append((match.token_id, text[start:stop]))
        start_offset = stop
        position += len(match.content)
    if start_offset < len(text):
        pieces.append((None, text[start_offset:]))
    return pieces


def _split_added(text: str, added: Sequence[_AddedToken]) -> list[tuple[int | None, str]]:
    """Both passes of added-token matching over ``text``."""
    first = _match_added(text, [token for token in added if not token.normalized])
    second = [token for token in added if token.normalized]
    pieces: list[tuple[int | None, str]] = []
    for token_id, piece in first:
        if token_id is None:
            pieces += _match_added(piece, second)
        else:
            pieces.append((token_id, piece))
    return pieces


def _reference_ids(
    text: str, matching: _Matching, unknown_id: int | None,
    merges: Sequence[tuple[int, int, int]] | None,
) -> list[int]:
    """The token IDs a character or raw BPE sprite writes for ``text``."""
    characters = dict(zip(reversed(matching.characters), reversed(matching.character_ids)))
    ranks = {(left, right): (rank, merged) for rank, (left, right, merged) in enumerate(merges or ())}
    ids: list[int] = []
    for added_id, piece in _split_added(text, matching.added):
        if added_id is not None:
            ids.append(added_id)
            continue
        found = [characters.get(character, unknown_id) for character in piece]
        units = [unit for unit in found if unit is not None]
        while merges is not None and len(units) > 1:
            best = min(
                ((ranks[pair][0], index) for index, pair in enumerate(zip(units, units[1:]))
                 if pair in ranks),
                default=None,
            )
            if best is None:
                break
            rank, index = best
            units[index:index + 2] = [merges[rank][2]]
        ids += units
    return ids


def _added_token_texts(matching: _Matching) -> list[str]:
    """Texts that make added tokens meet, overlap and touch whitespace."""
    contents = list(matching.probes[:24])
    texts = []
    for content in contents:
        texts += [content, f"x{content}y", content * 2, f" {content}  ", f"\t{content}\n",
                  content.swapcase(), content[:-1], f"{content[:1]} {content}"]
    for left, right in itertools.product(contents[:12], repeat=2):
        texts += [left + right, left[:-1] + right, left + right[1:], f"{left} {right}",
                  f"{left}  {right} "]
    return texts


def _check_added_tokens(
    tokenizer, exporter: str, matching: _Matching, unknown_id: int | None,
    merges: Sequence[tuple[int, int, int]] | None,
) -> None:
    """Check that the tokenizer matches its added tokens as the sprite will.

    Encodes texts built to make added tokens meet, overlap and touch
    whitespace, and compares each with ``_reference_ids``, the sprite's
    algorithm in Python, run on the same tables as the sprite.
    """
    if not matching.probes or not callable(getattr(tokenizer, "encode", None)):
        return
    for text in _added_token_texts(matching):
        actual = _encode_ids(tokenizer, text)
        expected = _reference_ids(text, matching, unknown_id, merges)
        if actual != expected:
            raise ValueError(
                f"{exporter} can't reproduce how this tokenizer matches its added tokens: "
                f"{text!r} gives token IDs {actual}, but the sprite would give {expected}"
            )


def _added_token_tables(
    added: Sequence[_AddedToken], characters: list[str], character_ids: list[int],
) -> dict[str, list]:
    """The lists the sprite matches added tokens with.

    Each text character is read as its code, its 1-based position in
    ``characters``; the characters of added tokens that aren't tokens
    themselves are appended there with the ID -1. Tokens are sorted by
    first code, longest first, so the first match at a position is the
    longest. ``_added first`` and ``_added count`` give, per code, the
    tokens that start with it.
    """
    for token in added:
        for character in token.content:
            if character not in characters:
                characters.append(character)
                character_ids.append(-1)
    code = {character: index + 1 for index, character in reversed(list(enumerate(characters)))}
    ordered = sorted(added, key=lambda token: (code[token.content[0]], -len(token.content)))
    tables: dict[str, list] = {
        "_added first": [0] * len(characters), "_added count": [0] * len(characters),
        "_added length": [], "_added offset": [], "_added id": [], "_added pass": [],
        "_added lstrip": [], "_added rstrip": [], "_added codes": [],
    }
    for position, token in enumerate(ordered, start=1):
        first = code[token.content[0]] - 1
        if not tables["_added first"][first]:
            tables["_added first"][first] = position
        tables["_added count"][first] += 1
        tables["_added length"].append(len(token.content))
        tables["_added offset"].append(len(tables["_added codes"]) + 1)
        tables["_added id"].append(token.token_id)
        tables["_added pass"].append(2 if token.normalized else 1)
        tables["_added lstrip"].append(int(token.lstrip))
        tables["_added rstrip"].append(int(token.rstrip))
        tables["_added codes"] += [code[character] for character in token.content]
    return tables


def _added_match(group: int, pieces: tuple[str, str, str], *, strips: bool) -> tuple:
    """One pass of added-token matching over codes ``lo`` to ``hi``.

    Appends each match, and the text between matches, to ``pieces`` (token
    ID, first and last position, with ID -1 for text), as ``_match_added``
    does in Python.
    """
    ids, starts, ends = pieces
    p, k, j, best, same = "_added p", "_added k", "_added j", "_added best", "_added same"
    start, stop, offset, code, walking = (
        "_added start", "_added stop", "_added from", "_added code", "_added walking",
    )

    def piece(token_id, first, last) -> tuple:
        return (append(ids, token_id), append(starts, first), append(ends, last))

    length_of = item("_added length", var(best))
    try_token = if_(eq(var(best), 0), (
        if_(eq(item("_added pass", var(k)), group), (
            if_(not_(gt(add(var(p), item("_added length", var(k))), add(var("hi"), 1))), (
                set_var(same, 1),
                set_var(j, 0),
                repeat(item("_added length", var(k)), (
                    if_(eq(var(same), 1), (
                        if_(not_(eq(
                            item("_codes", add(var(p), var(j))),
                            item("_added codes", add(item("_added offset", var(k)), var(j))),
                        )), (set_var(same, 0),)),
                    )),
                    change_var(j, 1),
                )),
                if_(eq(var(same), 1), (set_var(best, var(k)),)),
            )),
        )),
    ))
    widen: tuple = ()
    if strips:
        widen = (
            if_(eq(item("_added lstrip", var(best)), 1), (
                set_var(walking, 1),
                repeat_until(eq(var(walking), 0), (
                    if_else(gt(var(start), var("lo")), (
                        if_else(eq(item("_spaces", sub(var(start), 1)), 1),
                                (change_var(start, -1),), (set_var(walking, 0),)),
                    ), (set_var(walking, 0),)),
                )),
                if_(lt(var(start), var(offset)), (set_var(start, var(offset)),)),
            )),
            if_(eq(item("_added rstrip", var(best)), 1), (
                set_var(walking, 1),
                repeat_until(eq(var(walking), 0), (
                    if_else(gt(var(stop), var("hi")), (set_var(walking, 0),), (
                        if_else(eq(item("_spaces", var(stop)), 1),
                                (change_var(stop, 1),), (set_var(walking, 0),)),
                    )),
                )),
            )),
        )
    return (
        set_var(offset, var("lo")),
        set_var(p, var("lo")),
        repeat_until(gt(var(p), var("hi")), (
            set_var(best, 0),
            set_var(code, item("_codes", var(p))),
            if_(gt(var(code), 0), (
                set_var(k, item("_added first", var(code))),
                repeat(item("_added count", var(code)), (try_token, change_var(k, 1))),
            )),
            if_else(gt(var(best), 0), (
                set_var(start, var(p)),
                set_var(stop, add(var(p), length_of)),
                *widen,
                if_(lt(var(offset), var(start)), piece(-1, var(offset), sub(var(start), 1))),
                *piece(item("_added id", var(best)), var(start), sub(var(stop), 1)),
                set_var(offset, var(stop)),
                change_var(p, length_of),
            ), (change_var(p, 1),)),
        )),
        if_(not_(gt(var(offset), var("hi"))), piece(-1, var(offset), var("hi"))),
    )


def _added_token_encode(
    lookup: "_CharacterLookup", added: Sequence[_AddedToken], unknown_id: int | None,
    merge_body: tuple, *, bpe: bool,
) -> tuple:
    """The encode body of a character or raw BPE sprite with added tokens.

    Reads every character's code, matches the added tokens in two passes,
    then writes each match's ID and runs the model on the text between them,
    one piece at a time (BPE merges never cross a match).
    """
    strips = any(token.lstrip or token.rstrip for token in added)
    first_pass = ("_pieces 1 id", "_pieces 1 start", "_pieces 1 end")
    second_pass = ("_pieces id", "_pieces start", "_pieces end")
    two_passes = any(token.normalized for token in added) and not all(
        token.normalized for token in added
    )
    single_group = 2 if all(token.normalized for token in added) else 1
    codes = (
        clear("_codes"),
        *((clear("_spaces"),) if strips else ()),
        set_var("idx", 0),
        repeat(string_length(var("input")), (
            change_var("idx", 1),
            *lookup.find("_characters", letter(var("input"), var("idx")), "lookup"),
            if_else(lookup.missing("lookup"), (append("_codes", 0),),
                    (append("_codes", var("lookup")),)),
            *((if_else(
                gt(index_of("_white space", letter(var("input"), var("idx"))), 0),
                (append("_spaces", 1),), (append("_spaces", 0),),
            ),) if strips else ()),
        )),
        *lookup.finish(),
    )
    target = first_pass if two_passes else second_pass
    matching = (
        *(clear(name) for name in (*first_pass, *second_pass)),
        set_var("lo", 1),
        set_var("hi", length("_codes")),
        *_added_match(1 if two_passes else single_group, target, strips=strips),
    )
    if two_passes:
        matching += (
            set_var("_added i", 0),
            repeat(length(first_pass[0]), (
                change_var("_added i", 1),
                if_else(gt(item(first_pass[0], var("_added i")), -1), (
                    append(second_pass[0], item(first_pass[0], var("_added i"))),
                    append(second_pass[1], item(first_pass[1], var("_added i"))),
                    append(second_pass[2], item(first_pass[2], var("_added i"))),
                ), (
                    set_var("lo", item(first_pass[1], var("_added i"))),
                    set_var("hi", item(first_pass[2], var("_added i"))),
                    *_added_match(2, second_pass, strips=strips),
                )),
            )),
        )
    destination = "splitlist" if bpe else _TOKENIZED
    missing = () if unknown_id is None else (append(destination, unknown_id),)
    unit = "_added unit"
    model = (
        *((clear("splitlist"),) if bpe else ()),
        set_var("_added q", sub(item(second_pass[1], var("_added i")), 1)),
        repeat(add(sub(item(second_pass[2], var("_added i")),
                       item(second_pass[1], var("_added i"))), 1), (
            change_var("_added q", 1),
            set_var(unit, -1),
            if_(gt(item("_codes", var("_added q")), 0), (
                set_var(unit, item("_character ids", item("_codes", var("_added q")))),
            )),
            if_else(gt(var(unit), -1), (append(destination, var(unit)),), missing),
        )),
        *(merge_body if bpe else ()),
        *((
            set_var("_added q", 0),
            repeat(length("splitlist"), (
                change_var("_added q", 1),
                append(_TOKENIZED, item("splitlist", var("_added q"))),
            )),
        ) if bpe else ()),
    )
    emit = (
        set_var("_added i", 0),
        repeat(length(second_pass[0]), (
            change_var("_added i", 1),
            if_else(gt(item(second_pass[0], var("_added i")), -1),
                    (append(_TOKENIZED, item(second_pass[0], var("_added i"))),), model),
        )),
    )
    return (*codes, *matching, *emit)


_ADDED_VARIABLES = (
    "lo", "hi", "_added p", "_added k", "_added j", "_added best", "_added same",
    "_added start", "_added stop", "_added from", "_added code", "_added walking", "_added i",
    "_added q", "_added unit",
)
_ADDED_LISTS = (
    "_codes", "_spaces", "_white space", "_pieces 1 id", "_pieces 1 start", "_pieces 1 end",
    "_pieces id", "_pieces start", "_pieces end",
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
        added, skipped = _added_token_split(self.tokenizer, "CharTokenizer")
        # The unknown token is never read from text, whatever its length, and
        # added tokens are matched whole before the characters are looked up.
        special = (
            _special_tokens(self.tokenizer) | {tokens[unknown_id]}
            | {token.content for token in (*added, *skipped)}
        )
        words = [token for token in tokens if len(token) > 1 and token not in special]
        if words:
            raise ValueError(
                "CharTokenizer needs a tokenizer whose tokens are single characters, "
                f"but this vocabulary has multi-character tokens such as {words[0]!r}"
            )
        backend = _backend(self.tokenizer)
        if backend is not None:
            normalizer = _backend_settings(backend).get("normalizer")
            if normalizer is not None:
                raise ValueError(
                    "CharTokenizer doesn't support a normalizer "
                    f"({normalizer.get('type', 'unknown')}); the sprite looks up the "
                    "raw text, so tokens would differ"
                )
        matching = _sprite_matching(self.tokenizer, "CharTokenizer", tokens, unknown_id)
        _check_encodes_characters(self.tokenizer, matching, unknown_id)
        _check_added_tokens(self.tokenizer, "CharTokenizer", matching, unknown_id, None)
        self._wrap = _special_token_wrap(
            self.tokenizer, "CharTokenizer", tokens, _added_token_texts(matching),
        )
        sprite, self._costume_names = _build_character_tokenizer_sprite(
            tokens, bpe=False, unknown_id=unknown_id, matching=matching,
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
            raise ValueError("BPE dropout is nondeterministic and can't be exported")
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
        if model.get("ignore_merges") not in (None, False, True):
            raise ValueError(f"unexpected ignore_merges {model.get('ignore_merges')!r}")

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
        unknown_id = self._unknown_id()
        matching = _sprite_matching(self.tokenizer, "BPETokenizer", tokens, unknown_id, merges)
        if model.get("ignore_merges"):
            _check_merges_build_vocabulary(model, matching, unknown_id, merges)
        _check_added_tokens(self.tokenizer, "BPETokenizer", matching, unknown_id, merges)
        self._wrap = _special_token_wrap(
            self.tokenizer, "BPETokenizer", tokens, _added_token_texts(matching),
        )
        sprite, self._costume_names = _build_character_tokenizer_sprite(
            tokens,
            bpe=True,
            unknown_id=unknown_id,
            merges=merges,
            matching=matching,
        )
        return sprite


def _check_merges_build_vocabulary(
    model: dict, matching: _Matching, unknown_id: int | None,
    merges: Sequence[tuple[int, int, int]],
) -> None:
    """Accept ``ignore_merges=True`` only where it changes nothing.

    With it, the model gives a text that is itself a vocabulary entry that
    entry's ID without merging. The sprite always merges, which gives the
    same ID exactly when the merges build every entry from its characters.
    """
    vocab = model.get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError("BPETokenizer can't read the vocabulary of an ignore_merges model")
    bare = _Matching(matching.characters, matching.character_ids, (), ())
    # The sprite matches its added tokens before merging.
    matched = {token.content for token in matching.added}
    for entry, token_id in vocab.items():
        if len(entry) > 1 and entry not in matched and (
            _reference_ids(entry, bare, unknown_id, merges) != [token_id]
        ):
            raise ValueError(
                "BPETokenizer doesn't support ignore_merges=True when the merges don't "
                f"build every vocabulary entry: the model gives the text {entry!r} its "
                "entry's ID, where merging, as the sprite does, gives "
                f"{_reference_ids(entry, bare, unknown_id, merges)}"
            )


def _build_character_tokenizer_sprite(
    tokens: list[str],
    *,
    bpe: bool,
    unknown_id: int | None,
    merges: Sequence[tuple[int, int, int]] = (),
    matching: _Matching,
) -> tuple[dict, tuple[str, ...] | None]:
    """Build the character/raw-BPE contract with the typed DSL.

    ``matching`` gives the character table and the added tokens, matched in
    the text before the model runs. Returns the sprite and, when it looks
    characters up by costume, the costume names.
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
    # "1" match an earlier " 1" or "01". The added-token tables extend these.
    characters = list(matching.characters)
    character_ids = list(matching.character_ids)
    added = list(matching.added)
    added_tables = _added_token_tables(added, characters, character_ids) if added else {}
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

    encode_lists, encode_variables, encode_values = lists, variables, values
    if added:
        encode_lists = (*lists, *added_tables, *_ADDED_LISTS)
        encode_variables = (*variables, *_ADDED_VARIABLES)
        encode_values = {
            **values, **added_tables, "_white space": list(_UNICODE_WHITE_SPACE),
            **{name: [] for name in _ADDED_LISTS if name != "_white space"},
        }
        encode_body: tuple[Statement, ...] = (
            *_added_token_encode(lookup, added, unknown_id, merge_body, bpe=bpe),
        )
    elif bpe:
        encode_body = (
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
        variables=encode_variables,
        lists=encode_lists,
        list_values=encode_values,
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
        _check_scratch_casefold(scratch_casefold)
        self.scratch_casefold = scratch_casefold
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
        """Reject SentencePiece preprocessing Scratch can't reproduce."""
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
                        "SentencePiece models with add_dummy_prefix=True can't be "
                        "reproduced by the Scratch tokenizer"
                    )
                if normalizer.remove_extra_whitespaces:
                    raise ValueError(
                        "SentencePiece models with remove_extra_whitespaces=True can't "
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
                    "SentencePiece models with add_dummy_prefix=True can't be "
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

    def _bos_and_eos(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The IDs ``processor.encode`` adds around the text's, as a processor
        made with ``add_bos=True`` or ``add_eos=True`` does."""
        processor = self.processor
        for option in ("reverse", "enable_sampling"):
            if getattr(processor, f"_{option}", False):
                raise ValueError(
                    f"SentencePieceBPETokenizer can't reproduce a processor made with "
                    f"{option}=True"
                )
        prefix = (int(processor.bos_id()),) if getattr(processor, "_add_bos", False) else ()
        suffix = (int(processor.eos_id()),) if getattr(processor, "_add_eos", False) else ()
        if any(token_id < 0 for token_id in (*prefix, *suffix)):
            raise ValueError(
                "this SentencePiece processor adds a beginning or end token its model "
                "doesn't define"
            )
        encode = getattr(processor, "encode", None)
        if callable(encode):
            for text in ("", "a"):
                try:
                    bare = list(encode(text, add_bos=False, add_eos=False))
                except TypeError:
                    break
                full = list(encode(text))
                if full != [*prefix, *bare, *suffix]:
                    raise ValueError(
                        "SentencePieceBPETokenizer can't reproduce the tokens this "
                        f"processor adds: encode({text!r}) gives {full}, but the sprite "
                        f"would give {[*prefix, *bare, *suffix]}"
                    )
        return prefix, suffix

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
            if token_id == self.unknown_id:
                # What SentencePiece decodes the unknown token to: its
                # unk_surface, " \u2047 " by default.
                decode = getattr(self.processor, "decode", None)
                decoded = (
                    decode([token_id]) if callable(decode) else _SENTENCEPIECE_UNKNOWN_SURFACE
                )
            elif token_id in self._special_ids:
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
        self._wrap = self._bos_and_eos()
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
            "Vanilla Scratch can't convert between characters and arbitrary "
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
            broadcasts=result.broadcasts,
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
        JSON-size and compact-ID controls; ``layer_sharing`` and
        ``unrolling`` apply only to models, and raise ``ValueError``.

    Returns
    -------
    TokenizerResult
        Its ``tokenizer_type`` says which kind was exported.
    """
    _check_scratch_casefold(scratch_casefold)
    _check_codegen(name, codegen)
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
    added, skipped = _added_token_split(tokenizer, "transpile_tokenizer")
    special |= {token.content for token in (*added, *skipped)}
    if any(len(token) > 1 for token in tokenizer.get_vocab() if token not in special):
        raise ValueError(
            f"{model_type} tokenizers with multi-character tokens can't be exported; "
            "transpile_tokenizer supports BPE, SentencePiece BPE, and tokenizers "
            "whose tokens are single characters"
        )
    return CharTokenizer(tokenizer).save(output_path, name=name, codegen=codegen)
