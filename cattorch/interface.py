"""The Scratch-side interface of exported sprites, versioned for 1.x.

Every name that starts with ``cattorch `` is public: procedures, lists,
variables and broadcasts that Scratch projects use. Within an interface
version none is renamed or removed, and their meaning doesn't change; new
ones may be added. Everything else, including names that start with
``_cattorch``, is internal and may change in any release.

``docs/scratch-interface.md`` describes each name. Each sprite records the
version it implements in the ``cattorch interface version`` variable.
"""

from __future__ import annotations

import re

INTERFACE_VERSION = 1
VERSION_VARIABLE = "cattorch interface version"
PUBLIC_PREFIX = "cattorch "

# Names every kind of export may have. Model inputs and outputs (``input``,
# ``output``, ``input_1``, ...) and their ``shard N`` parts are public too,
# but don't start with the prefix; a program's entry point, state and list
# names are added per export.
PUBLIC_PROCEDURES = frozenset({
    # Every model.
    "cattorch init", "cattorch forward", "cattorch prepare for save",
    # Generation and programs.
    "cattorch reset", "cattorch prefill", "cattorch decode",
    # Tokenizers.
    "cattorch tokenize", "cattorch detokenize",
})
PUBLIC_DATA = frozenset({
    VERSION_VARIABLE,
    "cattorch status", "cattorch initialized",
    # Generation.
    "cattorch tokens", "cattorch logits", "cattorch cache length",
    "cattorch max context", "cattorch top k values", "cattorch top k ids",
    # Tokenizers.
    "cattorch text", "cattorch token ids",
    # How the weights are stored, for display.
    "cattorch storage compression", "cattorch storage precision",
    "cattorch storage scale precision", "cattorch quantization method",
    "cattorch quantization bits",
})
PUBLIC_BROADCASTS = frozenset({
    "cattorch init complete", "cattorch forward complete", "cattorch reset complete",
    "cattorch prefill complete", "cattorch decode complete",
})


_SHARD_VARIABLES = (" shard count", " logical length")


def is_public_data(name: str, extra: frozenset[str] = frozenset()) -> bool:
    """Whether a list or variable is part of the interface.

    A split public list's shards (``NAME shard 2``) and its
    ``cattorch NAME shard count`` and ``cattorch NAME logical length``
    variables are public too.
    """
    for suffix in _SHARD_VARIABLES:
        if name.startswith(PUBLIC_PREFIX) and name.endswith(suffix):
            listed = name[len(PUBLIC_PREFIX):-len(suffix)]
            return is_public_data(listed, extra) or is_public_data(
                PUBLIC_PREFIX + listed, extra,
            )
    base = name.split(" shard ", 1)[0]
    return (
        base in PUBLIC_DATA or base in extra
        or re.fullmatch(r"(input|output)(_\d+)?", base) is not None
    )


def internal_name(name: str) -> str:
    """The name an internal ``cattorch ...`` helper gets: ``_cattorch ...``."""
    return f"_{name}" if name.startswith(PUBLIC_PREFIX) else name


def final_data_name(name: str, extra: frozenset[str] = frozenset()) -> str:
    """The name a list or variable has in the finished sprite."""
    if name.startswith(PUBLIC_PREFIX) and not is_public_data(name, extra):
        return internal_name(name)
    return name
