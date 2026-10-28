"""PyTorch OpInfo samples that export and match eager mode must keep passing.

The suite checks the first passing sample of each operator so it stays fast;
``python tests/opinfo_conformance.py --check`` runs every recorded case.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("expecttest", reason="OpInfo samples need expecttest")

sys.path.insert(0, str(Path(__file__).parent))
import opinfo_conformance as conformance  # noqa: E402


def _first_per_operator() -> list[str]:
    chosen: dict[str, str] = {}
    for key in conformance.recorded()[1]:
        chosen.setdefault(key.rsplit("#", 1)[0], key)
    return sorted(chosen.values())


@pytest.mark.parametrize("key", _first_per_operator())
def test_opinfo_sample_still_passes(tmp_path, key):
    if (reason := conformance.version_mismatch()) is not None:
        pytest.skip(reason)
    op, sample = conformance.sample(key)
    outcome, detail = conformance.run_case(op, sample, tmp_path)
    assert outcome == "pass", f"{key}: {outcome}: {detail}"


def test_coverage_page_lists_the_recorded_cases():
    """The generated page and the pass list come from the same run."""
    page = conformance.COVERAGE.read_text()
    version, passing = conformance.recorded()
    assert f"{len(passing)} pass" in page
    assert f"PyTorch {version}" in page
