"""Structured results returned by cattorch's public export tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TensorSpec:
    """One flattened Scratch list at a model interface boundary."""

    list_name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int


@dataclass(frozen=True)
class ArtifactResult:
    """Common metadata for a generated Scratch artifact."""

    path: Path
    sprite_name: str
    archive_bytes: int
    expanded_json_bytes: int
    block_count: int
    list_count: int
    sharded_lists: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class TranspileResult(ArtifactResult):
    """Metadata for a completed model sprite export."""

    inputs: tuple[TensorSpec, ...]
    output: TensorSpec
    procedures: tuple[str, ...]


@dataclass(frozen=True)
class TokenizerResult(ArtifactResult):
    """Metadata for a completed tokenizer sprite export."""

    tokenizer_type: str
    token_count: int


@dataclass(frozen=True)
class VerifyResult:
    """Numerical comparison between PyTorch and a generated Scratch sprite."""

    passed: bool
    values_compared: int
    expected_shape: tuple[int, ...]
    actual_values: int
    max_abs_error: float
    max_rel_error: float
    mean_abs_error: float
    worst_index: int | None
