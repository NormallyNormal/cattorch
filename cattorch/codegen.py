"""Configuration shared by cattorch's Scratch code generators."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


_ID_NAMESPACE_RE = re.compile(r"[A-Za-z0-9_-]*\Z")


@dataclass(frozen=True)
class CodegenConfig:
    """Control generated-code size and Scratch identifier names.

    ``target_json_bytes`` is a soft budget used by automatic code generation.
    It is not a hard Scratch limit: the ordinary online editor currently has
    an approximately 5 MiB expanded-JSON limit, while archive import paths can
    accept larger projects.

    Compact IDs always begin with ``ct`` so they cannot collide with Scratch's
    fixed toolbox IDs. ``id_namespace`` follows that marker and defaults to a
    random three-character namespace. Set it to exactly three Base64url
    characters for reproducible exports, or to ``""`` for the smallest safe
    IDs when the sprite will not be combined with another generated sprite.

    ``compact_internal_names`` shortens private variable, list, and procedure
    display names. Public model/tokenizer interfaces keep their documented
    names so other sprites can continue to call and inspect them.

    ``compact_schema`` drops redundant false-valued block fields. It remains
    opt-in until browser import/edit/save compatibility has broader coverage.

    ``layer_sharing="auto"`` factors compatible stateless ``blocks.N`` stacks
    into one procedure and banks their weights. It falls back when caches or
    incompatible layouts make the transform unsafe, and is opt-in for now.
    """

    target_json_bytes: int | None = 4 * 1024 * 1024
    unrolling: Literal["auto", "compact", "speed"] = "auto"
    id_namespace: str | None = None
    compact_internal_names: bool = False
    compact_schema: bool = False
    layer_sharing: Literal["auto", "off"] = "off"

    def __post_init__(self) -> None:
        if self.target_json_bytes is not None and (
            isinstance(self.target_json_bytes, bool)
            or not isinstance(self.target_json_bytes, int)
            or self.target_json_bytes <= 0
        ):
            raise ValueError("target_json_bytes must be a positive integer or None")
        if self.unrolling not in {"auto", "compact", "speed"}:
            raise ValueError(
                "unrolling must be 'auto', 'compact', or 'speed', "
                f"got {self.unrolling!r}"
            )
        if self.id_namespace is not None:
            if not isinstance(self.id_namespace, str):
                raise TypeError("id_namespace must be a string or None")
            if _ID_NAMESPACE_RE.fullmatch(self.id_namespace) is None:
                raise ValueError(
                    "id_namespace may contain only Base64url characters "
                    "(A-Z, a-z, 0-9, '-' and '_')"
                )
            if self.id_namespace and len(self.id_namespace) != 3:
                raise ValueError(
                    "id_namespace must contain exactly three Base64url "
                    "characters, or be empty for a single-sprite export"
                )
        if not isinstance(self.compact_internal_names, bool):
            raise TypeError("compact_internal_names must be a bool")
        if not isinstance(self.compact_schema, bool):
            raise TypeError("compact_schema must be a bool")
        if self.layer_sharing not in {"auto", "off"}:
            raise ValueError("layer_sharing must be 'auto' or 'off'")


__all__ = ["CodegenConfig"]
