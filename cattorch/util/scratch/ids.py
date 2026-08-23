"""Scratch identifier utilities shared by model and tokenizer exporters."""

from __future__ import annotations

import uuid

from cattorch.util.scratch.remap import remap_ids


ID_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
CATTORCH_ID_PREFIX = "ct"

# scratch-gui assigns these fixed IDs to blocks and shadows in the standard
# toolbox. The sensing ``of`` menu, among others, asks the VM for its source by
# ID and checks the editing target before the flyout workspace. Reusing one of
# these IDs in an imported sprite can therefore make the palette inspect an
# unrelated generated block and crash. Random Scratch block IDs almost never
# collide; compact sequential IDs eventually do, so omit the fixed set.
SCRATCH_TOOLBOX_IDS = frozenset({
    "answer",
    "askandwait",
    "backdropnumbername",
    "current",
    "forever",
    "glidex",
    "glidey",
    "loudness",
    "motion_glideto",
    "movex",
    "movey",
    "of",
    "online",
    "operator_contains",
    "repeat_until",
    "sensing_of_object_menu",
    "sensing_setdragmode",
    "setx",
    "sety",
    "timer",
    "wait_until",
})


def _base64url_uint(value: int, *, width: int = 1) -> str:
    """Encode a non-negative integer without JSON-escaped characters."""
    if value < 0:
        raise ValueError("value must be non-negative")
    encoded = ID_ALPHABET[value & 63]
    value >>= 6
    while value:
        encoded = ID_ALPHABET[value & 63] + encoded
        value >>= 6
    return encoded.rjust(width, ID_ALPHABET[0])


def _default_namespace() -> str:
    # Eighteen random bits are enough for the overwhelmingly common case of a
    # single cattorch sprite, while retaining collision protection when a few
    # independently exported sprites are combined.
    return _base64url_uint(uuid.uuid4().int & ((1 << 18) - 1), width=3)


def uniquify_data_ids(sprite: dict) -> None:
    """Suffix list and variable IDs to avoid cross-sprite project conflicts."""
    suffix = uuid.uuid4().hex[:12]
    mapping = {}
    for section in ("lists", "variables"):
        slots = sprite.get(section, {})
        for identifier in list(slots):
            new_identifier = f"{identifier}_{suffix}"
            mapping[identifier] = new_identifier
            slots[new_identifier] = slots.pop(identifier)
    if mapping:
        sprite["blocks"] = remap_ids(sprite["blocks"], mapping)


def compact_sprite_ids(sprite: dict, *, namespace: str | None = None) -> dict:
    """Return a sprite whose internal IDs use one short unique namespace.

    Generated IDs are repeated throughout Scratch JSON as dictionary keys and
    block references. Their descriptive construction-time names are useful
    while assembling a sprite but have no meaning to Scratch or its users.
    Compacting only at final serialization preserves debuggability during
    compilation while reducing expanded project JSON without changing display
    names, values, or block semantics.
    """
    namespace = _default_namespace() if namespace is None else namespace
    if namespace and len(namespace) != 3:
        raise ValueError(
            "namespace must contain exactly three Base64url characters, "
            "or be empty for a single-sprite export"
        )
    if any(character not in ID_ALPHABET for character in namespace):
        raise ValueError("namespace may contain only Base64url characters")
    prefix = f"{CATTORCH_ID_PREFIX}{namespace}"
    mapping = {}
    # Scratch's official VM internally assumes IDs are unique across block and
    # data sections in several lookup paths, even though their JSON references
    # are typed. Keep one global sequence; separate sequences can load yet
    # silently execute against the wrong object.
    counter = 0
    for section in ("blocks", "variables", "lists", "broadcasts", "comments"):
        for identifier in sprite.get(section, {}):
            compact = f"{prefix}{_base64url_uint(counter)}"
            counter += 1
            while compact in SCRATCH_TOOLBOX_IDS:
                compact = f"{prefix}{_base64url_uint(counter)}"
                counter += 1
            mapping[identifier] = compact
    return remap_ids(sprite, mapping)


__all__ = [
    "CATTORCH_ID_PREFIX", "ID_ALPHABET", "SCRATCH_TOOLBOX_IDS",
    "compact_sprite_ids", "uniquify_data_ids",
]
