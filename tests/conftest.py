"""Shared test fixtures."""

from __future__ import annotations

import os
import sys

import pytest
import torch

import cattorch  # noqa: F401 - loads every module that holds the list limit

if os.environ.get("PYTEST_XDIST_WORKER"):
    # Parallel workers already use every core; PyTorch's own threads on top
    # of them oversubscribe the machine and make the run slower.
    torch.set_num_threads(1)


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
            if hasattr(module, "SCRATCH_LIST_LIMIT"):
                monkeypatch.setattr(module, "SCRATCH_LIST_LIMIT", limit)
        return limit

    return apply
