"""
tokenizer.py
------------
Transpile a HuggingFace tokenizer into a Scratch sprite.

Supports character-level, raw-text BPE, and SentencePiece BPE tokenizers.

Usage
-----
    from transformers import AutoTokenizer
    from cattorch import CharTokenizer, BPETokenizer

    tokenizer = AutoTokenizer.from_pretrained("my-model")
    BPETokenizer(tokenizer).save("my_tokenizer")
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from cattorch.codegen import CodegenConfig
from cattorch.results import TokenizerResult
from cattorch.sprite import (
    _add_warp_procedure,
    _merge_lists_by_name,
    _merge_variables_by_name,
    _procedure_mutation,
)
from cattorch.util.scratch.dsl import (
    Program,
    Statement,
    add,
    append,
    change_var,
    clear,
    delete,
    eq,
    gt,
    if_,
    if_else,
    index_of,
    item,
    join,
    length,
    letter,
    lt,
    mul,
    repeat,
    repeat_until,
    replace,
    set_var,
    string_length,
    sub,
    var,
)
from cattorch.util.scratch.finalize_scratch import finalize_sprite
from cattorch.util.scratch.ids import uniquify_data_ids


class _TokenizerBase(ABC):
    """Base class for tokenizer transpilers.

    Subclasses construct their Scratch program with the typed DSL.
    """
    def __init__(self, tokenizer):
        """
        Parameters
        ----------
        tokenizer : PreTrainedTokenizerBase
            A HuggingFace tokenizer. Must expose a ``.get_vocab()`` method.
        """
        self.tokenizer = tokenizer

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

        backend = getattr(self.tokenizer, "backend_tokenizer", None)
        if backend is not None and hasattr(backend, "to_str"):
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

        - **Encode**: reads the ``input`` variable (a string), writes
          token IDs to the ``token_ids`` list.
        - **Decode**: reads the ``token_ids`` list, writes the decoded
          string to the ``output`` variable.

        Token IDs are 0-based, matching PyTorch embedding conventions.

        Parameters
        ----------
        output_path : str or Path
            Destination path. ``.sprite3`` is appended when absent.
        name : str, optional
            Display name inside Scratch. Defaults to the output path stem.
        codegen : CodegenConfig, optional
            JSON-size, loop-unrolling, and compact-ID controls.
        """
        tokens = self._build_tokens_list()

        sprite = self._build_sprite(tokens)

        uniquify_data_ids(sprite)
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".sprite3":
            output_path = Path(f"{output_path}.sprite3")
        sprite_name = output_path.stem if name is None else name
        if not isinstance(sprite_name, str) or not sprite_name.strip():
            raise ValueError("name must be a non-empty string")
        finalized = finalize_sprite(
            sprite, output_path, sprite_name=sprite_name, codegen=codegen,
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
            tokenizer_type=self.__class__.__name__,
            token_count=len(tokens),
        )


def _reject_scratch_text_ambiguity(
    tokens: Sequence[str], *, characters_only: bool,
) -> None:
    """Reject vocab entries Scratch cannot distinguish or iterate safely."""
    seen: dict[str, str] = {}
    for token in tokens:
        if characters_only and len(token) != 1:
            continue
        if any(ord(character) > 0xFFFF for character in token):
            raise ValueError(
                "tokenizer vocabulary contains a non-BMP character which "
                "vanilla Scratch splits into UTF-16 surrogate halves"
            )
        key = token.lower()
        previous = seen.get(key)
        if previous is not None and previous != token:
            raise ValueError(
                "tokenizer vocabulary contains text which Scratch compares "
                f"as equal: {previous!r} and {token!r}"
            )
        seen[key] = token


class CharTokenizer(_TokenizerBase):
    """Transpile a character-level tokenizer into a Scratch sprite.

    Each character in the input is mapped to a token ID via a vocab lookup.
    No merging is performed.

    Example
    -------
    ::

        from cattorch import CharTokenizer

        CharTokenizer(tokenizer).save("my_char_tok")
    """
    def _build_sprite(self, tokens: list[str]) -> dict:
        _reject_scratch_text_ambiguity(tokens, characters_only=True)
        unknown_id = self._unknown_id()
        if unknown_id is None:
            raise ValueError(
                "CharTokenizer requires a tokenizer with a valid unknown-token ID"
            )
        return _build_character_tokenizer_sprite(
            tokens, bpe=False, unknown_id=unknown_id,
        )


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
    def _build_sprite(self, tokens: list[str]) -> dict:
        _reject_scratch_text_ambiguity(tokens, characters_only=False)
        backend = getattr(self.tokenizer, "backend_tokenizer", None)
        if backend is None or not hasattr(backend, "to_str"):
            raise TypeError(
                "BPETokenizer requires a Hugging Face fast tokenizer backend "
                "whose canonical merge order can be exported"
            )
        try:
            model = json.loads(backend.to_str())["model"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TypeError("unable to read the tokenizer's BPE model") from exc
        if model.get("type") != "BPE":
            raise TypeError("BPETokenizer requires a BPE backend model")
        if model.get("dropout") is not None:
            raise ValueError("BPE dropout is nondeterministic and cannot be exported")
        if model.get("continuing_subword_prefix") or model.get("end_of_word_suffix"):
            raise ValueError(
                "raw-text BPETokenizer does not support subword prefixes or suffixes"
            )

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
        return _build_character_tokenizer_sprite(
            tokens,
            bpe=True,
            unknown_id=self._unknown_id(),
            merges=merges,
        )


def _build_character_tokenizer_sprite(
    tokens: list[str],
    *,
    bpe: bool,
    unknown_id: int | None,
    merges: Sequence[tuple[int, int, int]] = (),
) -> dict:
    """Build the character/raw-BPE contract with the typed DSL."""
    variables = (
        (
            "input", "output", "idx", "lookup", "candidate",
            "best_rank", "best_idx", "merging",
        )
        if bpe else ("input", "output", "idx", "lookup")
    )
    lists = (
        ("tokens", "token_ids", "_characters", "_character ids")
        + (("splitlist", "merge_keys", "merge_ids") if bpe else ())
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
    values: dict[str, Sequence[int | float | str]] = {
        "tokens": tokens, "token_ids": [],
        "_characters": characters, "_character ids": character_ids,
    }
    if bpe:
        values["splitlist"] = []
        vocab_size = len(tokens)
        values["merge_keys"] = [left * vocab_size + right for left, right, _ in merges]
        values["merge_ids"] = [merged for _, _, merged in merges]

    def append_lookup(destination: str) -> tuple[Statement, ...]:
        missing = () if unknown_id is None else (append(destination, unknown_id),)
        return (
            set_var("lookup", index_of("_characters", letter(var("input"), var("idx")))),
            if_else(
                eq(var("lookup"), 0),
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
                    append("token_ids", item("splitlist", var("idx"))),
                ),
            ),
        )
    else:
        encode_body = (
            set_var("idx", 0),
            repeat(
                string_length(var("input")),
                (
                    change_var("idx", 1),
                    *append_lookup("token_ids"),
                ),
            ),
        )
    tokenize = Program(
        "BPE encode" if bpe else "character encode",
        variables=variables,
        lists=lists,
        list_values=values,
        body=(
            clear("token_ids"),
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
    _merge_variables_by_name(sprite, set(variables))
    _merge_lists_by_name(sprite, set(lists))
    _add_top_level_call(sprite, "cattorch tokenize", x=0, y=0)
    _add_top_level_call(sprite, "cattorch detokenize", x=0, y=100)
    return sprite


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

    Scratch string comparisons are case-insensitive. ``scratch_casefold=True``
    therefore supports SentencePiece models trained with an NFKC case-folding
    normalizer for ASCII input without requiring a separate lowercase pass.
    Arbitrary out-of-vocabulary Unicode byte fallback cannot be reproduced by
    vanilla Scratch, which has no character-code reporter; characters present
    directly in the vocabulary and all ASCII input are supported.

    Parameters
    ----------
    tokenizer:
        A ``SentencePieceProcessor`` or an object exposing one as
        ``.processor``.
    scratch_casefold:
        Whether the model was trained with case-folding normalization.
    """

    def __init__(self, tokenizer, *, scratch_casefold: bool = False):
        super().__init__(tokenizer)
        self.processor = getattr(tokenizer, "processor", tokenizer)
        required = (
            "get_piece_size",
            "id_to_piece",
            "is_byte",
            "is_control",
            "get_score",
            "unk_id",
        )
        missing = [name for name in required if not hasattr(self.processor, name)]
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
        normal_pieces = {
            piece.replace("▁", " ").casefold()
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
                # Do not expose byte forms which would shadow a normal token
                # under Scratch's case-insensitive list comparison.
                if byte_value < 128 and literal.casefold() not in normal_pieces:
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
            comparison_key = piece.casefold()
            if comparison_key in seen_characters:
                if not self.scratch_casefold:
                    raise ValueError(
                        "SentencePiece vocabulary contains characters which "
                        "Scratch compares as equal; use a case-folded model"
                    )
                continue
            seen_characters.add(comparison_key)
            characters.append(piece)
            character_ids.append(token_id)

        # SentencePiece BPE chooses the highest-scoring token available for an
        # adjacent pair. Encode a pair of token IDs as one exact Scratch number
        # so the hot scan uses numeric lookup instead of joining strings and
        # searching the entire piece vocabulary.
        piece_ids: dict[str, int] = {}
        for token_id, piece in enumerate(pieces):
            piece_ids.setdefault(piece.casefold(), token_id)
        vocab_size = len(pieces)
        merge_entries: list[tuple[float, int, int]] = []
        for merged_id, piece in enumerate(pieces):
            if merged_id in self._special_ids or self.processor.is_byte(merged_id):
                continue
            for split_at in range(1, len(piece)):
                left_id = piece_ids.get(piece[:split_at].casefold())
                right_id = piece_ids.get(piece[split_at:].casefold())
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
            "_sp split",
            "_sp characters",
            "_sp character ids",
            "_sp decoded tokens",
            "_sp merge keys",
            "_sp merge ids",
        )
        list_values = {
            "tokens": pieces,
            "token_ids": [],
            "_sp split": [],
            "_sp characters": characters,
            "_sp character ids": character_ids,
            "_sp decoded tokens": decode_pieces,
            "_sp merge keys": merge_keys,
            "_sp merge ids": merge_ids,
        }

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
                clear("token_ids"),
                clear("_sp split"),
                set_var("_sp i", 0),
                repeat(
                    string_length(var("input")),
                    (
                        change_var("_sp i", 1),
                        set_var(
                            "_sp lookup",
                            index_of(
                                "_sp characters",
                                letter(var("input"), var("_sp i")),
                            ),
                        ),
                        if_else(
                            eq(var("_sp lookup"), 0),
                            (
                                append(
                                    "_sp split",
                                    self.unknown_id,
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
                            "token_ids",
                            item("_sp split", var("_sp i")),
                        ),
                    ),
                ),
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
        _merge_variables_by_name(sprite, set(variables))
        _merge_lists_by_name(sprite, set(lists))
        _add_top_level_call(sprite, "cattorch tokenize", x=0, y=0)
        _add_top_level_call(sprite, "cattorch detokenize", x=0, y=100)
        return sprite

    def save(
        self,
        output_path: str | Path,
        *,
        name: str | None = None,
        codegen: CodegenConfig | None = None,
    ) -> TokenizerResult:
        result = super().save(output_path, name=name, codegen=codegen)
        if not self._decode_has_non_ascii_bytes:
            return result
        warning = (
            "Vanilla Scratch cannot convert arbitrary UTF-8 byte-fallback "
            "sequences to characters. ASCII bytes decode exactly; sampled "
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
    """Transpile a HuggingFace tokenizer into a Scratch .sprite3 file.

    Auto-detects BPE vs character-level from the tokenizer backend.
    For explicit control, use ``CharTokenizer`` or ``BPETokenizer`` directly.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        A HuggingFace tokenizer.
    output_path : str or Path
        Destination path. ``.sprite3`` is appended when absent.
    name : str, optional
        Display name inside Scratch. Defaults to the output path stem.
    scratch_casefold : bool, optional
        Set for a SentencePiece model trained with case-folding normalization.
    codegen : CodegenConfig, optional
        JSON-size, loop-unrolling, and compact-ID controls.
    """
    processor = getattr(tokenizer, "processor", tokenizer)
    sentencepiece_methods = (
        "get_piece_size",
        "id_to_piece",
        "is_byte",
        "is_control",
        "get_score",
        "unk_id",
    )
    if all(hasattr(processor, method) for method in sentencepiece_methods):
        return SentencePieceBPETokenizer(
            tokenizer,
            scratch_casefold=scratch_casefold,
        ).save(output_path, name=name, codegen=codegen)

    try:
        from tokenizers import models
        is_bpe = isinstance(tokenizer.backend_tokenizer.model, models.BPE)
    except ImportError as exc:
        if hasattr(tokenizer, "backend_tokenizer"):
            raise ImportError(
                "Hugging Face tokenizer auto-detection requires the "
                "'cattorch[tokenizers]' optional dependencies"
            ) from exc
        is_bpe = False
    except AttributeError:
        is_bpe = False

    cls = BPETokenizer if is_bpe else CharTokenizer
    return cls(tokenizer).save(output_path, name=name, codegen=codegen)
