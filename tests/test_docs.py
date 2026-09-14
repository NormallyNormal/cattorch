"""Lightweight checks that keep the Markdown documentation maintainable."""

from __future__ import annotations

import re
from pathlib import Path

import cattorch
import pytest
import torch


ROOT = Path(__file__).parents[1]
MARKDOWN_FILES = (ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")))
LINK_RE = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
PYTHON_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def _heading_anchors(text: str) -> set[str]:
    """Approximate GitHub's stable heading anchors used by local docs links."""
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*$", text, re.MULTILINE):
        plain = re.sub(r"[`*_~]", "", heading).lower()
        anchor = re.sub(r"[^a-z0-9 _-]", "", plain)
        anchor = re.sub(r"[ ]+", "-", anchor).strip("-")
        count = counts.get(anchor, 0)
        counts[anchor] = count + 1
        anchors.add(anchor if count == 0 else f"{anchor}-{count}")
    return anchors


@pytest.mark.parametrize("path", MARKDOWN_FILES, ids=lambda path: path.name)
def test_markdown_python_fences_are_valid_python(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for index, source in enumerate(PYTHON_FENCE_RE.findall(text), start=1):
        try:
            compile(source, f"{path}:python-fence-{index}", "exec")
        except SyntaxError as error:
            pytest.fail(str(error), pytrace=False)


@pytest.mark.parametrize("path", MARKDOWN_FILES, ids=lambda path: path.name)
def test_local_markdown_links_resolve(path: Path) -> None:
    for raw_target in LINK_RE.findall(path.read_text(encoding="utf-8")):
        target = raw_target.strip().strip("<>")
        if "://" in target or target.startswith(("mailto:", "#")):
            continue
        relative, _, fragment = target.partition("#")
        destination = (path.parent / relative).resolve()
        assert destination.exists(), f"{path.relative_to(ROOT)} links to missing {target}"
        if fragment and destination.suffix == ".md":
            anchors = _heading_anchors(destination.read_text(encoding="utf-8"))
            assert fragment in anchors, (
                f"{path.relative_to(ROOT)} links to missing heading {target}"
            )


def test_api_reference_mentions_every_public_export() -> None:
    reference = (ROOT / "docs" / "api-reference.md").read_text(encoding="utf-8")
    missing = [name for name in cattorch.__all__ if f"`{name}`" not in reference]
    assert not missing, f"Public cattorch exports missing from API reference: {missing}"


def test_generation_benchmark_documentation_uses_prompt_case() -> None:
    guide = (ROOT / "docs" / "verification-and-benchmarking.md").read_text(
        encoding="utf-8"
    )
    assert '("decoder", decoder, stateless_input, prompt)' in guide


@pytest.mark.parametrize(
    ("relative_path", "fence_indices"),
    [
        ("README.md", (0,)),
        ("docs/getting-started.md", (0, 1)),
        ("docs/programs-and-moe.md", (0, 1)),
        ("docs/programs-and-moe.md", (3,)),
    ],
    ids=("quick-start", "getting-started", "stateful-program", "sparse-moe"),
)
def test_documented_export_examples_run(
    relative_path: str,
    fence_indices: tuple[int, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run complete examples, including their PyTorch/emulator comparisons."""
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    fences = PYTHON_FENCE_RE.findall(source)
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__main__"}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        for index in fence_indices:
            code = compile(fences[index], f"{relative_path}:python-fence-{index + 1}", "exec")
            exec(code, namespace)
    assert list(tmp_path.rglob("*.sprite3")), "Example did not produce a sprite"
