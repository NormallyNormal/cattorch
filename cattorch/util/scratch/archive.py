"""Read and write Scratch archives: ``.sprite3`` and ``.sb3`` zip files.

A ``.sprite3`` holds ``sprite.json`` and a ``.sb3`` holds ``project.json``,
each next to the costume and sound files it names.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

SPRITE_JSON = "sprite.json"
PROJECT_JSON = "project.json"


def read_archive(path: str | Path, document: str = SPRITE_JSON) -> tuple[dict, dict[str, bytes]]:
    """The archive's JSON document, and its other files by name."""
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read(document))
        assets = {name: archive.read(name) for name in archive.namelist() if name != document}
    return data, assets


def read_json(path: str | Path, document: str = SPRITE_JSON) -> dict:
    """Only the archive's JSON document."""
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read(document))


def write_archive(
    path: str | Path, data: dict, assets: dict[str, bytes], document: str = SPRITE_JSON,
) -> str:
    """Write ``data`` as the archive's compact JSON document, with ``assets``.

    Returns the JSON text, whose size Scratch limits.
    """
    text = json.dumps(data, separators=(",", ":"))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(document, text)
        for name, content in assets.items():
            archive.writestr(name, content)
    return text


__all__ = ["PROJECT_JSON", "SPRITE_JSON", "read_archive", "read_json", "write_archive"]
