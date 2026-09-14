import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from add_tinystories_sampler import (  # noqa: E402
    build_initializer_program,
    build_one_token_program,
    build_sampler_program,
)


def test_sampler_runs_until_eos_or_total_context_is_full():
    program = build_sampler_program()
    pseudocode = program.pseudocode()

    assert "max new tokens" not in pseudocode
    assert "cattorch cache length" in pseudocode
    assert "cattorch max context" in pseudocode
    assert "sampler finished" in pseudocode


def test_sampler_sizes_frequency_table_from_exported_vocabulary():
    pseudocode = build_initializer_program().pseudocode()

    assert "repeat length(cattorch logits):" in pseudocode
    assert "sampler finished = (length(cattorch logits) == 0)" in pseudocode
    assert "tinystories max new tokens" not in pseudocode


def test_sampler_rejects_below_kth_before_scanning_and_defers_temperature():
    pseudocode = build_one_token_program().pseudocode()

    candidate = "(cattorch logits[sampler index] - (0.3 * tinystories generated token counts[sampler index]))"
    assert f"sampler candidate = {candidate}" in pseudocode
    assert (
        "if (sampler candidate > tinystories sampler top values[20]):"
    ) in pseudocode
    assert f"{candidate} / 0.6" not in pseudocode
    assert (
        "e ^(((tinystories sampler top values[sampler index] - "
        "sampler maximum) / 0.6))"
    ) in pseudocode
