#!/usr/bin/env python3
"""Combine generated cattorch sprites into one importable Scratch sprite."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from cattorch.util.scratch.finalize_scratch import FinalizedSprite, finalize_sprite
from cattorch.util.scratch.remap import remap_ids


def _read_sprite(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("sprite.json"))


def combine_sprites(
    primary_path: Path,
    attached_path: Path,
    output_path: Path,
    *,
    name: str,
) -> FinalizedSprite:
    """Attach every script and data slot from one sprite to another.

    IDs in the attached sprite are namespaced before merging. Display names and
    data remain unchanged, which is useful when the scripts intentionally form
    an interface (for example ``token_ids`` feeding a model's ``input`` list).
    """
    primary = _read_sprite(primary_path)
    attached = _read_sprite(attached_path)
    mapping = {
        identifier: f"cattorch_attached_{section}_{identifier}"
        for section in ("blocks", "variables", "lists", "broadcasts")
        for identifier in attached.get(section, {})
    }
    attached = remap_ids(attached, mapping)

    for section in ("blocks", "variables", "lists", "broadcasts"):
        primary.setdefault(section, {}).update(attached.get(section, {}))
    return finalize_sprite(primary, output_path, sprite_name=name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path)
    parser.add_argument("attached", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    print(
        combine_sprites(
            args.primary,
            args.attached,
            args.output,
            name=args.name,
        )
    )


if __name__ == "__main__":
    main()
