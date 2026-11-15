"""The one rule for the file an export, project or benchmark writes."""

from __future__ import annotations

import os
from pathlib import Path


def output_file(path, suffix: str, *, parameter: str = "output_path") -> Path:
    """The file ``path`` names, with ``suffix`` appended unless it already ends in it.

    The suffix is compared without regard to case, and other dots stay in the
    name (``net.v2`` → ``net.v2.sprite3``). A path that names no file raises
    ``ValueError`` naming ``parameter``: an empty path, one ending in a
    separator, ``.`` or ``..``, or an existing directory.
    """
    if isinstance(path, bytes) or not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"{parameter} must be a str or path, not {type(path).__name__}")
    text = os.fspath(path)
    if not isinstance(text, str):
        raise TypeError(f"{parameter} must be a str or path, not {type(text).__name__}")
    separators = tuple(filter(None, (os.sep, os.altsep)))
    last = text
    for separator in separators:
        last = last.rsplit(separator, 1)[-1]
    if last in ("", ".", ".."):
        raise ValueError(f"{parameter} {text!r} doesn't name a file")
    if Path(text).is_dir():
        raise ValueError(f"{parameter} {text!r} is a directory; name a file in it")
    result = Path(text) if Path(text).suffix.lower() == suffix else Path(f"{text}{suffix}")
    if result.is_dir():
        raise ValueError(f"{parameter} {str(result)!r} is a directory")
    return result
