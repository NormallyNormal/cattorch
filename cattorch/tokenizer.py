"""
tokenizer.py
------------
Transpile a HuggingFace tokenizer into a Scratch sprite.

Supports character-level and BPE tokenizers via the ``CharTokenizer``
and ``BPETokenizer`` classes.

Usage
-----
    from transformers import AutoTokenizer
    from cattorch import CharTokenizer, BPETokenizer

    tokenizer = AutoTokenizer.from_pretrained("my-model")
    BPETokenizer(tokenizer).save("my_tokenizer")
"""

import json
import uuid
from abc import ABC, abstractmethod

from cattorch.templates.template import TEMPLATE_DIR
from cattorch.util.scratch.finalize_scratch import finalize_sprite
from cattorch.util.scratch.remap import remap_ids


def _uniquify_ids(sprite):
    """Add a UUID suffix to all list and variable IDs to prevent conflicts."""
    suffix = uuid.uuid4().hex[:12]
    mapping = {}

    for section in ("lists", "variables"):
        slots = sprite.get(section, {})
        for sid in list(slots):
            new_id = f"{sid}_{suffix}"
            mapping[sid] = new_id
            slots[new_id] = slots.pop(sid)

    if mapping:
        sprite["blocks"] = remap_ids(sprite["blocks"], mapping)


class _TokenizerBase(ABC):
    """Base class for tokenizer transpilers.

    Subclasses set ``template_name`` to select which Scratch template to use.
    """
    template_name: str

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

    def save(self, sprite_name: str):
        """Write the tokenizer as a Scratch .sprite3 file.

        The generated sprite has two top-level block stacks:

        - **Encode**: reads the ``input`` variable (a string), writes
          token IDs to the ``token_ids`` list.
        - **Decode**: reads the ``token_ids`` list, writes the decoded
          string to the ``output`` variable.

        Token IDs are 0-based, matching PyTorch embedding conventions.

        Parameters
        ----------
        sprite_name : str
            Name for the output sprite.  The file is written as
            ``{sprite_name}.sprite3``.
        """
        tokens = self._build_tokens_list()

        template_path = TEMPLATE_DIR / self.template_name / "template.json"
        with open(template_path) as f:
            sprite = json.load(f)

        for entry in sprite["lists"].values():
            if entry[0] == "tokens":
                entry[1] = tokens
                break

        _uniquify_ids(sprite)
        finalize_sprite(sprite, f"{sprite_name}.sprite3", sprite_name=sprite_name)


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
    template_name = "char_tokenizer"


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
    template_name = "bpe_tokenizer"


def transpile_tokenizer(tokenizer, sprite_name: str):
    """Transpile a HuggingFace tokenizer into a Scratch .sprite3 file.

    Auto-detects BPE vs character-level from the tokenizer backend.
    For explicit control, use ``CharTokenizer`` or ``BPETokenizer`` directly.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        A HuggingFace tokenizer.
    sprite_name : str
        Name for the output sprite.
    """
    try:
        from tokenizers import models
        is_bpe = isinstance(tokenizer.backend_tokenizer.model, models.BPE)
    except (AttributeError, ImportError):
        is_bpe = False

    cls = BPETokenizer if is_bpe else CharTokenizer
    cls(tokenizer).save(sprite_name)
