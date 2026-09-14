"""Structured results returned by cattorch's public export tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cattorch.quantization import QuantizationReport


@dataclass(frozen=True)
class BoundedDimension:
    """One runtime tensor extent with a finite exported capacity."""

    name: str
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("bounded dimension name must be non-empty")
        if (
            isinstance(self.minimum, bool) or not isinstance(self.minimum, int)
            or isinstance(self.maximum, bool) or not isinstance(self.maximum, int)
            or self.minimum < 0 or self.maximum < self.minimum
        ):
            raise ValueError("bounded dimension requires 0 <= minimum <= maximum")


@dataclass(frozen=True)
class TensorSpec:
    """One flattened Scratch list at a model interface boundary.

    ``list_name`` is the generated display name; ``shape`` and ``dtype``
    describe the original tensor; ``numel`` is its flattened item count.
    """

    list_name: str
    shape: tuple[int | BoundedDimension, ...]
    dtype: str
    numel: int | None
    max_numel: int | None = None

    def __post_init__(self) -> None:
        maximum = 1
        dynamic = False
        for extent in self.shape:
            if isinstance(extent, BoundedDimension):
                dynamic = True
                maximum *= extent.maximum
            else:
                maximum *= extent
        if dynamic:
            if self.numel is not None:
                raise ValueError("dynamic TensorSpec.numel must be None")
            if self.max_numel is None:
                object.__setattr__(self, "max_numel", maximum)
            elif self.max_numel != maximum:
                raise ValueError("dynamic TensorSpec.max_numel does not match its shape")
        elif self.max_numel is None:
            object.__setattr__(self, "max_numel", self.numel)
        elif self.max_numel != self.numel:
            raise ValueError("fixed TensorSpec.max_numel must equal numel")
        if not dynamic and self.numel != maximum:
            raise ValueError("fixed TensorSpec.numel does not match its shape")


@dataclass(frozen=True)
class ArtifactResult:
    """Common metadata for a generated Scratch artifact.

    Size fields are bytes: ``archive_bytes`` measures the compressed artifact
    and ``expanded_json_bytes`` its uncompressed Scratch JSON. ``warnings`` are
    non-fatal compatibility, size, or fallback messages that callers should
    inspect before distribution.
    """

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
    """Model artifact metadata, tensor interface, and public procedures.

    ``quantization`` is set when ``QuantizationConfig`` was used, and records
    whether each parameter used GPTQ, symmetric quantization, or float16.
    """

    inputs: tuple[TensorSpec, ...]
    output: TensorSpec
    procedures: tuple[str, ...]
    quantization: QuantizationReport | None = None
    additional_outputs: tuple[TensorSpec, ...] = ()

    @property
    def outputs(self) -> tuple[TensorSpec, ...]:
        """All tensor output leaves in their original pytree order."""
        return (self.output, *self.additional_outputs)


@dataclass(frozen=True)
class EntryPointResult:
    """Generated interface metadata for one program entrypoint."""

    name: str
    procedure: str
    completion_broadcast: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    state_inputs: tuple[str, ...]
    state_updates: tuple[str, ...]


@dataclass(frozen=True)
class StateResult:
    """Generated persistent Scratch list and logical tensor contract."""

    name: str
    list_name: str
    mode: str
    shape: tuple[int, ...]
    dtype: str
    capacity: int | None


@dataclass(frozen=True)
class ProgramResult(ArtifactResult):
    """Artifact metadata for an experimental multi-entrypoint program."""

    entrypoints: tuple[EntryPointResult, ...]
    states: tuple[StateResult, ...]
    procedures: tuple[str, ...]
    quantization: QuantizationReport | None = None


@dataclass(frozen=True)
class TokenizerResult(ArtifactResult):
    """Tokenizer artifact metadata, selected backend type, and vocabulary size."""

    tokenizer_type: str
    token_count: int


@dataclass(frozen=True)
class VerifyResult:
    """Numerical comparison between PyTorch and a generated Scratch sprite.

    Error fields compare flattened values. ``worst_index`` is zero-based and
    is ``None`` for an empty result. Output-length mismatches return a failed
    result with infinite aggregate errors.
    """

    passed: bool
    values_compared: int
    expected_shape: tuple[int, ...]
    actual_values: int
    max_abs_error: float
    max_rel_error: float
    mean_abs_error: float
    worst_index: int | None


@dataclass(frozen=True)
class MultiOutputVerifyResult:
    """Per-tensor comparisons in the exported pytree leaf order.

    ``passed`` is true only when every output comparison passes.
    """

    passed: bool
    outputs: tuple[VerifyResult, ...]


@dataclass(frozen=True)
class ProgramCallVerifyResult:
    """Numerical result for one stateful program invocation."""

    entrypoint: str
    passed: bool
    outputs: tuple[VerifyResult, ...]
    states: tuple[tuple[str, VerifyResult], ...]
    status: str = "ok"


@dataclass(frozen=True)
class ProgramVerifyResult:
    """Ordered stateful verification results for an exported program."""

    passed: bool
    calls: tuple[ProgramCallVerifyResult, ...]


def _tensor_spec(list_name, shape, dtype, dynamic=None):
    if dynamic is None:
        return TensorSpec(
            list_name=list_name,
            shape=tuple(shape),
            dtype=str(dtype).removeprefix("torch."),
            numel=math.prod(shape),
        )
    axis, (dimension_name, minimum, maximum) = dynamic
    public_shape = list(shape)
    public_shape[axis] = BoundedDimension(
        dimension_name, minimum, maximum,
    )
    max_numel = math.prod(
        maximum if index == axis else extent
        for index, extent in enumerate(shape)
    )
    return TensorSpec(
        list_name=list_name,
        shape=tuple(public_shape),
        dtype=str(dtype).removeprefix("torch."),
        numel=None,
        max_numel=max_numel,
    )
