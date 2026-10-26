"""Shared test fixtures."""

from __future__ import annotations

import sys

import pytest

import cattorch  # noqa: F401 - loads every module that holds the list limit


@pytest.fixture
def small_list_limit(monkeypatch):
    """Lower Scratch's 200,000-item list limit to ``limit`` for one test.

    Sharding, banking and storage behave the same at any limit, so a test can
    cross list boundaries with a small model instead of 200,000 values. Every
    loaded cattorch module that imported the limit is patched.
    """

    def apply(limit: int) -> int:
        for name, module in list(sys.modules.items()):
            if not name.startswith("cattorch") or module is None:
                continue
            for attribute in ("SCRATCH_LIST_LIMIT", "SCRATCH_MAX_LIST_LENGTH"):
                if hasattr(module, attribute):
                    monkeypatch.setattr(module, attribute, limit)
        return limit

    return apply
