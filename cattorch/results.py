"""Structured results returned by cattorch's public export tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cattorch.quantization import QuantizationReport


@dataclass(frozen=True)
class BoundedDimension:
    """One tensor dimension whose length varies at run time, within bounds.

    ``name`` names the dimension (``tokens``, or the input it comes from),
    and ``minimum`` and ``maximum`` bound its length.
    """

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

    ``list_name`` is the list's Scratch name; ``shape`` and ``dtype``
    describe the tensor it holds, flattened, with a ``BoundedDimension`` for
    a dimension whose length varies. ``numel`` is the number of items, or
    ``None`` when it varies, and ``max_numel`` the most items the list holds.
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
    """What every export writes: the file and its size.

    ``path`` is the file written and ``sprite_name`` the sprite's name in
    Scratch. Size fields are bytes: ``archive_bytes`` measures the
    compressed file and ``expanded_json_bytes`` its uncompressed Scratch
    JSON. ``block_count`` and ``list_count`` count the sprite's blocks and
    lists, and ``sharded_lists`` names the public lists split into shards
    because they hold more than a Scratch list can. ``warnings`` are the
    export's warnings (size, fallbacks, anything that changed what was
    exported), each also emitted as a ``CattorchWarning``.
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
    """A model sprite from tensor inputs: its interface and public procedures.

    ``inputs`` and ``outputs`` describe the ``input``, ``input_1``, ... and
    ``output``, ``output_1``, ... lists, in the model's argument and output
    order; ``output`` is the first output. ``procedures`` are the public
    blocks. ``quantization`` is set when ``QuantizationConfig`` was used, and
    records whether each parameter used GPTQ, symmetric quantization, or
    float16. ``fallback_operations`` lists operations without a kernel of
    their own, which run as PyTorch's decomposition into simpler operations,
    with how many calls of each the sprite runs. ``approximate_operations``
    lists the operations that use an approximate (fast-mode) kernel, the
    same way.
    """

    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    procedures: tuple[str, ...]
    quantization: QuantizationReport | None = None
    fallback_operations: tuple[tuple[str, int], ...] = ()
    approximate_operations: tuple[tuple[str, int], ...] = ()

    @property
    def output(self) -> TensorSpec:
        """The first output, ``outputs[0]``: the ``output`` list."""
        return self.outputs[0]


@dataclass(frozen=True)
class GenerationResult(ArtifactResult):
    """A text generator sprite, from a ``GenerationProgram`` or ``RecurrentGenerationProgram``.

    Both kinds have the same interface. ``inputs`` describes the
    ``cattorch token ids`` list: a prompt for ``cattorch prefill`` of 1 to
    ``max_context`` tokens (to the most a Scratch list holds when there is
    no limit), or the one token ``cattorch decode`` reads. ``outputs``
    describes ``cattorch logits``, one value per token ID, then, with
    ``top_k``, ``cattorch top k values`` and ``cattorch top k ids``.
    ``max_context`` is the most tokens the sprite reads between resets
    (``None``: no limit, for a recurrent model), ``vocabulary_size`` the
    number of token IDs it accepts, and ``top_k`` the number of largest
    logits it lists (``None``: none). ``procedures``, ``quantization``,
    ``fallback_operations`` and ``approximate_operations`` are as for
    ``TranspileResult``.
    """

    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    max_context: int | None
    vocabulary_size: int
    top_k: int | None
    procedures: tuple[str, ...]
    quantization: QuantizationReport | None = None
    fallback_operations: tuple[tuple[str, int], ...] = ()
    approximate_operations: tuple[tuple[str, int], ...] = ()


def generation_result(
    result: TranspileResult | ProgramResult,
    *,
    max_context: int | None,
    top_k: int | None,
    vocabulary_size: int,
    logits: int | None = None,
) -> GenerationResult:
    """The ``GenerationResult`` of a generation export assembled as ``result``.

    ``logits`` is the number of logits, when ``result`` doesn't describe
    them as a ``TranspileResult`` does.
    """
    from cattorch import interface
    from cattorch.util.scratch.sharding import SCRATCH_LIST_LIMIT

    if logits is None:
        assert isinstance(result, TranspileResult)
        logits = math.prod(
            extent for extent in result.outputs[0].shape if isinstance(extent, int)
        )
    capacity = min(max_context or SCRATCH_LIST_LIMIT, SCRATCH_LIST_LIMIT)
    outputs = [TensorSpec(interface.LOGITS, (logits,), "float32", logits)]
    if top_k is not None:
        outputs += [
            TensorSpec(interface.TOP_K_VALUES, (top_k,), "float32", top_k),
            TensorSpec(interface.TOP_K_IDS, (top_k,), "int64", top_k),
        ]
    return GenerationResult(
        path=result.path,
        sprite_name=result.sprite_name,
        archive_bytes=result.archive_bytes,
        expanded_json_bytes=result.expanded_json_bytes,
        block_count=result.block_count,
        list_count=result.list_count,
        sharded_lists=result.sharded_lists,
        warnings=result.warnings,
        inputs=(TensorSpec(
            interface.TOKEN_IDS, (1, BoundedDimension("tokens", 1, capacity)), "int64", None,
        ),),
        outputs=tuple(outputs),
        max_context=max_context,
        vocabulary_size=vocabulary_size,
        top_k=top_k,
        procedures=(
            interface.INIT, interface.RESET, interface.PREFILL, interface.DECODE,
            interface.PREPARE_FOR_SAVE,
        ),
        quantization=result.quantization,
        fallback_operations=result.fallback_operations,
        approximate_operations=result.approximate_operations,
    )


def fallback_warning(fallbacks: tuple[tuple[str, int], ...]) -> str | None:
    """One warning naming the operations that ran as decompositions."""
    if not fallbacks:
        return None
    listed = ", ".join(
        f"{name} ({count} call{'s' if count != 1 else ''})" for name, count in fallbacks
    )
    return (
        f"{len(fallbacks)} operation{'s' if len(fallbacks) != 1 else ''} without "
        f"a kernel of their own ran as PyTorch decompositions, which can be "
        f"slower than a dedicated kernel: {listed}"
    )


@dataclass(frozen=True)
class EntryPointResult:
    """The Scratch interface of one program entry point.

    ``name`` is the entry point's name, ``procedure`` its block
    (``cattorch ENTRY``) and ``completion_broadcast`` the broadcast the block
    sends when it ends (``cattorch ENTRY complete``). ``inputs`` and
    ``outputs`` describe its ``cattorch ENTRY INPUT`` and
    ``cattorch ENTRY OUTPUT`` lists, in declaration order. ``state_inputs``
    names the states it reads and ``state_updates`` the states it writes.
    """

    name: str
    procedure: str
    completion_broadcast: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    state_inputs: tuple[str, ...]
    state_updates: tuple[str, ...]


@dataclass(frozen=True)
class StateResult:
    """One piece of a program's persistent state and its Scratch list.

    ``name`` is the state's name and ``list_name`` its list
    (``cattorch state STATE``). ``mode`` is ``"replace"`` or ``"append"``,
    and ``shape`` and ``dtype`` describe its initial tensor. ``capacity`` is
    the most rows an append state holds, ``None`` for a replace state.
    """

    name: str
    list_name: str
    mode: str
    shape: tuple[int, ...]
    dtype: str
    capacity: int | None


@dataclass(frozen=True)
class ProgramResult(ArtifactResult):
    """A program sprite, from an ``ExportProgram`` or ``@entrypoint`` markers.

    ``entrypoints`` and ``states`` describe its entry points and states, and
    ``procedures`` are its public blocks: one per entry point, then
    ``cattorch init``, ``cattorch reset`` and ``cattorch prepare for save``.
    ``quantization``, ``fallback_operations`` and ``approximate_operations``
    are as for ``TranspileResult``, summed over the entry points.
    """

    entrypoints: tuple[EntryPointResult, ...]
    states: tuple[StateResult, ...]
    procedures: tuple[str, ...]
    quantization: QuantizationReport | None = None
    fallback_operations: tuple[tuple[str, int], ...] = ()
    approximate_operations: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class TokenizerResult(ArtifactResult):
    """A tokenizer sprite.

    ``tokenizer_type`` is ``"char"``, ``"bpe"`` or ``"sentencepiece-bpe"``,
    the kind of tokenizer exported, and ``token_count`` its vocabulary size.
    """

    tokenizer_type: str
    token_count: int


@dataclass(frozen=True)
class ProjectResult:
    """A project written by :func:`build_project`.

    ``path`` is the ``.sb3`` written and ``sprites`` its sprites' names.
    ``broadcasts`` are its public broadcasts, each bridge's message followed
    by the bridge's completion broadcast (``NAME run``, ``NAME run complete``,
    ...), and ``lists`` and ``variables`` the global data they read and
    write, all by Scratch name. ``warnings`` are as for ``ArtifactResult``.
    """

    path: Path
    sprites: tuple[str, ...]
    broadcasts: tuple[str, ...]
    lists: tuple[str, ...]
    variables: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class VerifyResult:
    """Numerical comparison between PyTorch and one list of a generated sprite.

    ``passed`` is whether every value is within the tolerance.
    ``values_compared`` counts the values compared, ``expected_shape`` is
    PyTorch's tensor shape and ``actual_values`` the number of values the
    sprite wrote. Error fields compare flattened values. ``worst_index`` is
    zero-based and is ``None`` for an empty result. Output-length mismatches
    return a failed result with infinite aggregate errors.
    ``approximate_operations`` repeats the export's approximate kernels when
    ``verify`` was given its result, since they explain small differences.
    """

    passed: bool
    values_compared: int
    expected_shape: tuple[int, ...]
    actual_values: int
    max_abs_error: float
    max_rel_error: float
    mean_abs_error: float
    worst_index: int | None
    approximate_operations: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class MultiOutputVerifyResult:
    """Per-output comparisons of a model with several outputs.

    ``outputs`` holds one ``VerifyResult`` per output list, in the exported
    order. ``passed`` is true only when every comparison passes.
    ``approximate_operations`` is as for ``VerifyResult``.
    """

    passed: bool
    outputs: tuple[VerifyResult, ...]
    approximate_operations: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class ProgramCallVerifyResult:
    """The result of one block call in a stateful verification.

    ``entrypoint`` names the block called: a program's entry point, or
    ``prefill``, ``decode`` or ``reset`` of a generation sprite. ``status``
    is the ``cattorch status`` it ended with and ``expected_status`` the one
    it should end with: ``ok``, or the failure the call's input should give.
    ``outputs`` compares its output lists with PyTorch when both are
    ``ok`` (for generation: ``cattorch logits``, then the top-k values, then
    the logits at the top-k IDs against the top-k values), and ``states``
    compares each state list, by state name. ``passed`` is whether the
    status is the expected one and every comparison passed.
    """

    entrypoint: str
    passed: bool
    outputs: tuple[VerifyResult, ...]
    states: Mapping[str, VerifyResult]
    status: str = "ok"
    expected_status: str = "ok"


@dataclass(frozen=True)
class ProgramVerifyResult:
    """The results of a sequence of block calls, in order.

    ``calls`` holds one ``ProgramCallVerifyResult`` per call, and ``passed``
    is whether every call passed.
    """

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
