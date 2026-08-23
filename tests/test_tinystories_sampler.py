import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from add_tinystories_sampler import (  # noqa: E402
    build_initializer_program,
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

    assert "repeat length(output):" in pseudocode
    assert "sampler finished = (length(output) == 0)" in pseudocode
    assert "tinystories max new tokens" not in pseudocode
