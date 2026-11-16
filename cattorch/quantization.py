"""Offline weight quantization used by Scratch static storage."""

from __future__ import annotations

import itertools
import math
import weakref
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Literal, Mapping, cast

import numpy as np
import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils import _pytree as pytree
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.weak import WeakIdKeyDictionary

from cattorch.storage import _quantize_arrays, group_scales, integer_storage_bytes
from cattorch.operator_registry import OperatorRegistry, default_registry


@dataclass(frozen=True)
class QuantizationOverride:
    """Quantization settings for the weights one ``QuantizationConfig.overrides``
    key selects.

    ``bits=None`` keeps the selected weights in float16, unquantized.
    ``method`` and ``group_size`` default to the config's own; ``method``
    can be ``"symmetric"`` within a GPTQ config, but ``"gptq"`` needs a
    GPTQ config, whose calibration data it uses.
    """

    bits: Literal[4, 6, 8] | None
    method: Literal["symmetric", "gptq"] | None = None
    group_size: int | None = None

    def __post_init__(self) -> None:
        if self.bits is not None and (isinstance(self.bits, bool) or self.bits not in {4, 6, 8}):
            raise ValueError("override bits must be 4, 6, 8, or None for float16")
        if self.method is not None and self.method not in {"symmetric", "gptq"}:
            raise ValueError("override method must be 'symmetric', 'gptq', or None")
        if self.group_size is not None and (
            isinstance(self.group_size, bool)
            or not isinstance(self.group_size, int)
            or self.group_size < 1
        ):
            raise ValueError("override group_size must be a positive integer or None")
        if self.bits is None and (self.method is not None or self.group_size is not None):
            raise ValueError("an override with bits=None keeps float16; it takes no method or group_size")


@dataclass(frozen=True)
class QuantizationConfig:
    """Configure lossy matrix-weight quantization during export.

    Quantized values are reconstructed during ``cattorch init``. The generated
    forward kernels continue to operate on ordinary Scratch numeric lists.
    ``group_size`` is a maximum: matrix groups are shortened to the largest
    divisor of the input width which does not exceed it.

    ``min_quantized_values`` leaves smaller eligible matrices in float16.
    GPTQ additionally uses ``damp_percent`` for positive Hessian diagonal
    damping, ``block_size`` for sequential compensation blocks, and
    ``min_calibration_rows`` before accepting captured statistics. An
    eligible matrix without compatible statistics falls back to symmetric
    quantization.

    ``overrides`` maps a module path or parameter name, as
    ``model.named_parameters()`` spells it, to other settings for the
    weights under it: bits (4, 6 or 8), ``None`` to keep them float16, or a
    ``QuantizationOverride``. The longest matching key wins. A weight read
    under several names, such as an embedding tied to the output head,
    takes the highest precision any of its names selects, and a weight
    computed from parameters the highest precision of its parameters.

    ``head_bits`` sets the output head of a generation export
    (``GenerationProgram``, ``RecurrentGenerationProgram``): the weights of
    the last matrix products that compute the logits, found by following
    the model's computation. ``"auto"`` stores it in int8 when ``bits`` is
    4 or 6, and in ``bits`` otherwise; 4, 6 or 8 store it in that many
    bits, and ``None`` keeps it float16. An override that selects the head
    under any of its names takes precedence. Other exports have no output
    head: they take the default ``"auto"`` and raise for any other value.
    """

    bits: Literal[4, 6, 8]
    method: Literal["symmetric", "gptq"] = "symmetric"
    group_size: int = 64
    scale_precision: Literal["float32", "float16"] = "float16"
    min_quantized_values: int = 16_384
    damp_percent: float = 0.01
    block_size: int = 128
    min_calibration_rows: int = 128
    overrides: Mapping[str, int | QuantizationOverride | None] = field(default_factory=dict)
    head_bits: Literal["auto", 4, 6, 8] | None = "auto"

    def __post_init__(self) -> None:
        if isinstance(self.bits, bool) or self.bits not in {4, 6, 8}:
            raise ValueError("bits must be 4, 6, or 8")
        if self.method not in {"symmetric", "gptq"}:
            raise ValueError("method must be 'symmetric' or 'gptq'")
        if (
            isinstance(self.group_size, bool)
            or not isinstance(self.group_size, int)
            or self.group_size < 1
        ):
            raise ValueError("group_size must be a positive integer")
        if self.scale_precision not in {"float32", "float16"}:
            raise ValueError("scale_precision must be 'float32' or 'float16'")
        for name in ("min_quantized_values", "block_size", "min_calibration_rows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.damp_percent, bool)
            or not isinstance(self.damp_percent, (int, float))
            or not math.isfinite(float(self.damp_percent))
            or self.damp_percent <= 0
        ):
            raise ValueError("damp_percent must be a positive finite number")
        if self.method != "gptq":
            for name, default in (
                ("damp_percent", 0.01), ("block_size", 128), ("min_calibration_rows", 128),
            ):
                if getattr(self, name) != default:
                    raise ValueError(
                        f"{name} applies to method='gptq'; a {self.method} config doesn't use it"
                    )
        if self.head_bits is not None and (
            isinstance(self.head_bits, bool) or self.head_bits not in {"auto", 4, 6, 8}
        ):
            raise ValueError("head_bits must be 'auto', 4, 6, 8, or None for float16")
        if not isinstance(self.overrides, Mapping):
            raise TypeError("overrides must be a mapping of names to settings")
        normalized: dict[str, QuantizationOverride] = {}
        for key, value in self.overrides.items():
            if not isinstance(key, str) or not key or key != key.strip(". "):
                raise ValueError(
                    f"quantization override key {key!r} must be a module path or "
                    "parameter name"
                )
            if value is None or (isinstance(value, int) and not isinstance(value, bool)):
                value = QuantizationOverride(bits=cast(Literal[4, 6, 8] | None, value))
            elif not isinstance(value, QuantizationOverride):
                raise TypeError(
                    f"quantization override {key!r} must be bits (4, 6 or 8), None, "
                    "or a QuantizationOverride"
                )
            if value.method == "gptq" and self.method != "gptq":
                raise ValueError(
                    f"quantization override {key!r} uses method='gptq', which needs "
                    "a QuantizationConfig with method='gptq' and its calibration data"
                )
            normalized[key] = value
        object.__setattr__(self, "overrides", normalized)

    def __hash__(self) -> int:
        # The overrides are a dict; a config stays hashable, as before them.
        return hash(tuple(
            frozenset(cast(dict, value).items()) if name == "overrides" else value
            for name, value in vars(self).items()
        ))

    @property
    def precision(self) -> Literal["int4", "int6", "int8"]:
        return cast(
            Literal["int4", "int6", "int8"],
            {4: "int4", 6: "int6", 8: "int8"}[self.bits],
        )

    def _integer_formats(self) -> tuple[tuple[str, int], ...]:
        """Every (precision, maximum group size) the config may store."""
        formats: list[tuple[str, int]] = [(self.precision, self.group_size)]
        for override in cast(dict, self.overrides).values():
            if override.bits is not None:
                item = (f"int{override.bits}", override.group_size or self.group_size)
                if item not in formats:
                    formats.append(item)
        return tuple(formats)

    def _head_settings(
        self, names: Iterable[str], *, every_name: bool = True,
    ) -> _WeightSettings:
        """The settings for the output head, read under ``names``.

        An override that selects any of its names decides, as for any
        weight; otherwise ``head_bits`` does.
        """
        names = list(names)
        if any(
            name == key or name.startswith(f"{key}.")
            for name in names for key in cast(dict, self.overrides)
        ):
            return self._settings(names, every_name=every_name)
        bits = self.head_bits
        if bits == "auto":
            bits = 8 if self.bits < 8 else self.bits
        if bits == self.bits:
            return _WeightSettings(self.bits, self.method, self.group_size, None)
        return _WeightSettings(bits, self.method, self.group_size, HEAD_BITS)

    def _settings(self, names: Iterable[str], *, every_name: bool = True) -> _WeightSettings:
        """The settings for a weight read under ``names``.

        Each name takes its longest matching override key, or the config's
        own settings; the weight takes the highest precision among them.
        With ``every_name`` false, as for a weight computed from parameters
        and selected by any of their names, only names an override selects
        count, unless none is.
        """
        found: list[_WeightSettings] = []
        for name in names:
            keys = [
                key for key in cast(dict, self.overrides)
                if name == key or name.startswith(f"{key}.")
            ]
            if not keys:
                if every_name:
                    found.append(_WeightSettings(self.bits, self.method, self.group_size, None))
                continue
            key = max(keys, key=len)
            override = cast(dict, self.overrides)[key]
            found.append(_WeightSettings(
                override.bits,
                override.method or self.method,
                override.group_size or self.group_size,
                key,
            ))
        if not found:
            return _WeightSettings(self.bits, self.method, self.group_size, None)
        return max(
            found,
            key=lambda item: (
                16 if item.bits is None else item.bits, -item.group_size,
                item.method == "gptq", item.key is not None,
            ),
        )


# The ``TensorQuantizationReport.override`` of an output head whose settings
# ``QuantizationConfig.head_bits`` chose.
HEAD_BITS = "head_bits"


@dataclass(frozen=True)
class _WeightSettings:
    """How one weight is quantized: ``bits`` None keeps it float16; ``key``
    is the override that chose these settings, if one did."""

    bits: int | None
    method: str
    group_size: int
    key: str | None

    def float16_reason(self) -> str:
        if self.key == HEAD_BITS:
            return "head_bits=None keeps the output head in float16"
        return f"override {self.key!r} keeps it in float16"

    def config(self, base: QuantizationConfig) -> QuantizationConfig:
        """``base`` with these settings, for one weight's quantization."""
        if (self.bits, self.method, self.group_size) == (base.bits, base.method, base.group_size):
            return base
        assert self.bits is not None
        return replace(
            base, bits=cast(Literal[4, 6, 8], self.bits),
            method=cast(Literal["symmetric", "gptq"], self.method),
            group_size=self.group_size, overrides={},
        )


@dataclass(frozen=True)
class TensorQuantizationReport:
    """Quantization decision for one parameter or parameter view.

    A weight computed from parameters names them in ``sources``; ``override``
    is the ``QuantizationConfig.overrides`` key whose settings it used, or
    ``"head_bits"`` for an output head whose settings ``head_bits`` chose.
    """

    name: str
    method: Literal["gptq", "symmetric", "float16"]
    bits: int | None
    values: int
    calibration_rows: int
    effective_group_size: int | None
    fallback_reason: str | None = None
    sources: tuple[str, ...] = ()
    override: str | None = None


@dataclass(frozen=True)
class QuantizationReport:
    """Structured summary of quantization performed for an export."""

    method: Literal["gptq", "symmetric"]
    bits: int
    calibration_batches: int
    calibration_rows: int
    quantized_values: int
    gptq_values: int
    symmetric_values: int
    tensors: tuple[TensorQuantizationReport, ...]
    # Where each quantized entry's values sit in its parameter, for export to
    # say how storage stored them. Neither they nor the rest below are part
    # of the public report.
    _regions: tuple[_Region, ...] = field(default=(), repr=False, compare=False)
    # What storage stored of the parameters in the exports that produced
    # this report (``_StorageFindings``), and the entries quantization made,
    # which ``_regions`` number, before storage's findings changed them.
    _stored: Any = field(default=None, repr=False, compare=False)
    _entries: tuple[TensorQuantizationReport, ...] | None = field(
        default=None, repr=False, compare=False,
    )
    # Weights computed from parameters, by storage, which regions may sit in.
    _computed: dict = field(default_factory=dict, repr=False, compare=False)
    # Matrices quantization left as they are because export computes with
    # them before any input arrives, by storage, with their names.
    _unquantized: dict = field(default_factory=dict, repr=False, compare=False)
    # Copies of parameters, by storage: (the parameter, the element of its
    # storage at each of theirs). See ``_storage_sources``.
    _copies: dict = field(default_factory=dict, repr=False, compare=False)
    # Every (precision, maximum group size) the quantized weights use.
    _integer_formats: tuple = field(default=(), repr=False, compare=False)


@dataclass(frozen=True)
class _StoredFloat16:
    """How many of one entry's values an export stored in float16, and why.

    ``only`` values are stored in float16 alone; ``copies`` are also stored
    in integers elsewhere, as when two uses read one region in different
    layouts.
    """

    only: int
    copies: int
    reason: str


@dataclass
class _StorageFindings:
    """What storage stored of the parameters, over every export so far.

    The entry points of one program are compiled one at a time, and each
    stores its own lists: a region one stores in integers and another as a
    float16 copy is stored both ways. Masks are of parameter storage
    elements, by parameter storage; ``contributions`` are ``(masks, values
    held, why storage kept it in float)`` of each float16 list (see
    ``StoredList.reason``); ``derived`` are float16 weights computed from
    parameters at export, by name.
    """

    float16: dict[int, torch.Tensor] = field(default_factory=dict)
    integer: dict[int, torch.Tensor] = field(default_factory=dict)
    contributions: list[tuple[dict[int, torch.Tensor], int, str | None]] = field(
        default_factory=list,
    )
    derived: dict[str, TensorQuantizationReport] = field(default_factory=dict)
    derived_warnings: dict[str, str] = field(default_factory=dict)
    # Values of unquantized matrices stored as they are, by name.
    unquantized: dict[str, int] = field(default_factory=dict)

    def merged(self, other: _StorageFindings) -> _StorageFindings:
        def union(first, second):
            result = dict(first)
            for pointer, mask in second.items():
                previous = result.get(pointer)
                result[pointer] = mask if previous is None else previous | mask
            return result

        return _StorageFindings(
            float16=union(self.float16, other.float16),
            integer=union(self.integer, other.integer),
            contributions=[*self.contributions, *other.contributions],
            derived={**other.derived, **self.derived},
            derived_warnings={**other.derived_warnings, **self.derived_warnings},
            unquantized={
                name: max(self.unquantized.get(name, 0), other.unquantized.get(name, 0))
                for name in {**self.unquantized, **other.unquantized}
            },
        )


# Why integer storage keeps a floating list in float (``StoredList.reason``,
# decided by ``static_storage._static_storage_decision``).
STORED_AS_VECTOR = "vector"
STORED_BELOW_MINIMUM = "below minimum"
STORED_OFF_GRID = "off grid"
# What that means for quantized values the list holds.
_FLOAT_STORAGE_REASONS: dict[str | None, str] = {
    STORED_OFF_GRID: (
        "storage reads the values in groups along a different axis than they were "
        "quantized in"
    ),
    STORED_BELOW_MINIMUM: (
        "storage reads them from a list of fewer than min_quantized_values={minimum} "
        "values, which it keeps in float16"
    ),
    STORED_AS_VECTOR: "storage reads them as a vector, which it keeps in float16",
    None: "storage keeps the list that holds them in float16",
}


def _stored_quantization_report(
    report: QuantizationReport,
    findings: _StorageFindings | None = None,
    *,
    storage_precision: str = "",
    min_quantized_values: int = 0,
) -> tuple[tuple[str, ...], QuantizationReport]:
    """Warnings, and ``report`` saying what storage stored.

    ``findings`` are this export's; they add to those of the exports that
    produced ``report``, as the entry points of one program do. Every entry
    quantization made then says how its values are stored, from the
    findings together, and weights computed from parameters at export that
    storage keeps in float16 are added.
    """
    base = report.tensors if report._entries is None else report._entries
    previous = report._stored if isinstance(report._stored, _StorageFindings) else None
    if findings is None:
        accumulated = previous
    else:
        accumulated = findings if previous is None else previous.merged(findings)
    found: dict[int, _StoredFloat16] = {}
    warnings_found: list[str] = []
    if accumulated is not None and accumulated.float16:
        regions = report._regions
        masks = _region_masks(
            region for region in regions if region.pointer in accumulated.float16
        )
        for region in regions:
            if region.entry not in masks or base[region.entry].method == "float16":
                continue
            mask = masks[region.entry]
            float16 = accumulated.float16[region.pointer] & mask
            if not bool(float16.any()):
                continue
            integer = accumulated.integer.get(region.pointer)
            copies = int((float16 & integer).sum()) if integer is not None else 0
            only = int(float16.sum()) - copies
            # A list holds no more values than it has: a float16 shard of a
            # weight stores that many of the weight's values.
            capacity = 0.0
            why: dict[str | None, int] = {}
            for held, count, stored_reason in accumulated.contributions:
                if region.pointer not in held:
                    continue
                overlap = int((held[region.pointer] & mask).sum())
                if overlap:
                    share = sum(int(value.sum()) for value in held.values())
                    capacity += count * overlap / share
                    why[stored_reason] = why.get(stored_reason, 0) + overlap
            if only + copies > capacity:
                scale = capacity / (only + copies)
                only, copies = round(only * scale), round(copies * scale)
            if not only + copies:
                continue
            # Storage's own reason for most of the values.
            reason = _FLOAT_STORAGE_REASONS.get(
                max(why, key=lambda key: why[key]) if why else None,
                _FLOAT_STORAGE_REASONS[None],
            ).format(minimum=min_quantized_values)
            found[region.entry] = _StoredFloat16(only=only, copies=copies, reason=reason)
            item = base[region.entry]
            if only >= item.values:
                warnings_found.append(
                    f"quantized weight {item.name!r} is stored as float16, not "
                    f"{storage_precision}: {reason}"
                )
            else:
                warnings_found.append(
                    f"quantized weight {item.name!r}: {only + copies} of its values are "
                    f"stored as float16, not {storage_precision}"
                    + (" (a second copy)" if copies else "") + f": {reason}"
                )
    tensors: list[TensorQuantizationReport] = []
    for entry, item in enumerate(base):
        stored: _StoredFloat16 | None = found.get(entry)
        if stored is None:
            tensors.append(item)
            continue
        stored_as = (
            f"{item.method} int{item.bits} values are stored as float16: {stored.reason}"
        )
        if stored.only >= item.values:
            tensors.append(replace(
                item, method="float16", bits=None, effective_group_size=None,
                fallback_reason=stored_as,
            ))
            continue
        tensors.append(replace(item, values=item.values - stored.only))
        tensors.append(TensorQuantizationReport(
            name=f"{item.name}[float16]",
            method="float16",
            bits=None,
            values=stored.only + stored.copies,
            calibration_rows=0,
            effective_group_size=None,
            fallback_reason=(
                f"{stored.copies} of these values are also stored in integers; {stored_as}"
                if stored.copies else stored_as
            ),
        ))
    if accumulated is not None:
        tensors.extend(
            TensorQuantizationReport(
                name=name, method="float16", bits=None, values=values, calibration_rows=0,
                effective_group_size=None, fallback_reason=_READ_AT_CONSTANT_POSITIONS,
            )
            for name, values in accumulated.unquantized.items()
        )
        tensors.extend(accumulated.derived.values())
        warnings_found.extend(accumulated.derived_warnings.values())
    gptq_values = sum(item.values for item in tensors if item.method == "gptq")
    symmetric_values = sum(item.values for item in tensors if item.method == "symmetric")
    return tuple(warnings_found), replace(
        report,
        quantized_values=gptq_values + symmetric_values,
        gptq_values=gptq_values,
        symmetric_values=symmetric_values,
        tensors=tuple(tensors),
        _stored=accumulated,
        _entries=base,
    )


def effective_group_size(width: int, maximum: int) -> int:
    """Return the largest divisor of ``width`` no greater than ``maximum``."""
    if width < 1:
        raise ValueError("matrix width must be positive")
    for candidate in range(min(width, maximum), 0, -1):
        if width % candidate == 0:
            return candidate
    return 1


Orientation = Literal["linear", "rhs_matmul", "conv", "conv_transpose"]


@dataclass
class _ActivationStats:
    name: str
    weight: torch.Tensor
    orientation: Orientation
    hessian: torch.Tensor
    rows: int = 0


def _matrix_width(weight: torch.Tensor, orientation: str) -> int | None:
    """The input width of ``weight`` used in ``orientation``, if it can be."""
    if orientation in {"linear", "rhs_matmul"}:
        if weight.ndim != 2:
            return None
        return weight.shape[1] if orientation == "linear" else weight.shape[0]
    if weight.ndim < 3:
        return None
    kernel = math.prod(weight.shape[2:])
    return (weight.shape[1] if orientation == "conv" else weight.shape[0]) * kernel


def _logical_matrix(weight: torch.Tensor, orientation: str) -> torch.Tensor:
    """``weight`` as an output-major matrix over its inputs.

    A group of convolution kernels has a row per output channel, over its
    input channels and kernel positions in storage order.
    """
    if orientation == "linear":
        return weight
    if orientation == "rhs_matmul":
        return weight.T
    if orientation == "conv":
        return weight.reshape(weight.shape[0], -1)
    return weight.transpose(0, 1).reshape(weight.shape[1], -1)


def _physical_weight(
    matrix: torch.Tensor, weight: torch.Tensor, orientation: str,
) -> torch.Tensor:
    """Undo ``_logical_matrix`` for a matrix of ``weight``'s values."""
    if orientation == "linear":
        return matrix
    if orientation == "rhs_matmul":
        return matrix.T
    if orientation == "conv":
        return matrix.reshape(weight.shape)
    swapped = (weight.shape[1], weight.shape[0], *weight.shape[2:])
    return matrix.reshape(swapped).transpose(0, 1)


def _storage_group_width(weight: torch.Tensor, orientation: str) -> int | None:
    """The run of a logical row which storage groups together, if not all.

    Storage groups consecutive values of the kernels the compiler stores.
    A 1D or 2D convolution stores its kernels as they are: an output
    channel's fan-in, every input channel and kernel position, is
    consecutive, so groups run over it as a linear layer's do. A 3D
    convolution is lowered to a 2D one per kernel depth, and a transposed
    convolution to a plain one with its kernels flipped, so their groups
    stay within the last two kernel dimensions (the last one, in 1D), which
    every such slice keeps consecutive and a flip only reverses.
    """
    dimensions = weight.ndim - 2
    if orientation == "conv" and dimensions <= 2:
        return None
    if orientation in {"conv", "conv_transpose"}:
        return math.prod(int(size) for size in weight.shape[-min(2, dimensions):])
    return None


def _storage_key(value: torch.Tensor) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    return (
        value.untyped_storage().data_ptr(), int(value.storage_offset()),
        tuple(value.shape), tuple(value.stride()),
    )


class _GPTQCaptureMode(TorchDispatchMode):
    """Collect linear input covariances without depending on module structure."""

    def __init__(
        self,
        parameters: dict[int, str],
        registry: OperatorRegistry,
        *,
        collect_hessian: bool = True,
        static_storages: Iterable[int] = (),
        follow_products: bool = False,
    ):
        super().__init__()
        # Parameters, and weights computed from them (see
        # ``_ComputedWeightCapture``), by storage.
        self.parameters = parameters
        self.registry = registry
        # Symmetric quantization only needs to know how each matrix is used.
        self.collect_hessian = collect_hessian
        self.stats: dict[tuple, _ActivationStats] = {}
        # A storage known before any input arrives -> (a weak reference to
        # it, or None for the model's own tensors, the parameters whose
        # values it holds, whether it holds them unchanged). This covers
        # parameters, buffers, and what operations make from them alone.
        self.static: dict[int, tuple[StorageWeakRef | None, frozenset[int], bool]] = {
            pointer: (None, frozenset({pointer}), True) for pointer in parameters
        }
        for pointer in static_storages:
            self.static.setdefault(pointer, (None, frozenset(), True))
        # Parameters read as they are by an operation that also reads an
        # input, and parameters combined by arithmetic into another tensor
        # known at export, such as ``weight_norm``'s weight.
        self.direct_reads: set[int] = set()
        self.derivations: set[int] = set()
        # With ``follow_products``: each tensor computed at run time -> the
        # storages of the weights whose matrix products last computed it.
        # A product with a weight replaces its inputs' weights with its own;
        # every other operation passes on all its inputs' weights.
        self.products: WeakIdKeyDictionary | None = (
            WeakIdKeyDictionary() if follow_products else None
        )

    def _static_entry(
        self, tensor: torch.Tensor,
    ) -> tuple[StorageWeakRef | None, frozenset[int], bool] | None:
        pointer = tensor.untyped_storage().data_ptr()
        entry = self.static.get(pointer)
        if entry is not None and entry[0] is not None and entry[0].expired():
            del self.static[pointer]
            return None
        return entry

    def _track_static(self, func, target: str, args, kwargs, result) -> bool:
        """Follow which tensors are known at export, and how parameters are read.

        Export folds every operation whose inputs are all known into a
        constant, so a weight computed from parameters, as ``weight_norm``
        and ``F.linear(x, w * g)`` compute theirs, is stored as computed.
        Views, copies and concatenations keep the values they rearrange;
        any other operation computes new ones. Returns whether every input
        was known.
        """
        inputs = [
            item for item in pytree.tree_leaves((args, kwargs))
            if isinstance(item, torch.Tensor)
        ]
        outputs = [
            item for item in pytree.tree_leaves(result) if isinstance(item, torch.Tensor)
        ]
        if target in _LITERALS:
            # ``torch.tensor(0.5)``: a literal, as constant as a factory's.
            inputs = []
        entries = [self._static_entry(item) for item in inputs]
        input_storages = {item.untyped_storage().data_ptr() for item in inputs}
        mutable = bool(func._schema.is_mutable)
        known = [entry for entry in entries if entry is not None]
        if len(known) == len(entries) and (entries or not _random_function(target)):
            parameters = frozenset().union(*(entry[1] for entry in known))
            aliasing = any(
                item.untyped_storage().data_ptr() in input_storages for item in outputs
            )
            unchanged = (
                bool(known) and (target in _REARRANGEMENTS or aliasing)
                and all(entry[2] for entry in known)
            )
            if not unchanged:
                for entry in known:
                    if entry[2]:
                        self.derivations.update(entry[1])
            for item in outputs:
                pointer = item.untyped_storage().data_ptr()
                if pointer in self.parameters or (pointer in input_storages and not mutable):
                    continue
                self.static[pointer] = (
                    StorageWeakRef(item.untyped_storage()), parameters, unchanged,
                )
            return True
        for entry in known:
            if entry[2]:
                self.direct_reads.update(entry[1])
        if mutable:
            # A known tensor written with input values is no longer known.
            for item in inputs:
                pointer = item.untyped_storage().data_ptr()
                if pointer not in self.parameters:
                    self.static.pop(pointer, None)
        return False

    def _parameter_name(self, weight: torch.Tensor) -> str | None:
        return self.parameters.get(weight.untyped_storage().data_ptr())

    def add_computed(self, tensor: torch.Tensor, name: str, sources: Iterable[int]) -> None:
        """Follow a weight computed from the parameters at ``sources`` as one
        of its own: they are its ingredients, and reading it reads only it."""
        pointer = tensor.untyped_storage().data_ptr()
        self.parameters[pointer] = name
        self.static[pointer] = (
            StorageWeakRef(tensor.untyped_storage()), frozenset({pointer}), True,
        )
        self.derivations.update(sources)

    def remove_computed(self, tensor: torch.Tensor) -> None:
        self.parameters.pop(tensor.untyped_storage().data_ptr(), None)

    def _parameter_view(self, tensor: torch.Tensor) -> torch.Tensor | None:
        """``tensor`` if it is a view of a parameter or computed weight.

        A copy the model makes of a weight before using it, as in
        ``F.linear(x, weight.t().contiguous())``, is a computed weight too.
        """
        return tensor if tensor.untyped_storage().data_ptr() in self.parameters else None

    def _record(self, value: torch.Tensor, weight: torch.Tensor, orientation: str) -> None:
        if not value.is_floating_point() or not weight.is_floating_point():
            return
        name = self._parameter_name(weight)
        width = _matrix_width(weight, orientation)
        if name is None or not width:
            return
        # A Hessian is width x width doubles: past a bound, the matrix
        # falls back to symmetric quantization rather than build it.
        collect = self.collect_hessian and width <= _GPTQ_MAX_WIDTH
        slices: Iterable[torch.Tensor] = ()
        if orientation in {"conv", "conv_transpose"}:
            # Windows ``[batch, *positions, channels, *kernel]``: only GPTQ
            # reads them. They are copied a few rows of positions at a time,
            # so the copies stay far smaller than every window at once.
            trailing = weight.ndim - 1
            if value.ndim <= trailing or math.prod(value.shape[-trailing:]) != width:
                return
            windows = value.detach()
            leading = windows.shape[:-trailing]
            rows_seen = math.prod(leading)
            if collect:
                if len(leading) == 1:
                    windows, leading = windows.unsqueeze(1), (1, *leading)
                per_position = math.prod(leading[2:]) * width
                step = max(1, _WINDOW_CHUNK_VALUES // max(1, per_position))
                slices = (
                    windows[batch, start:start + step].reshape(-1, width)
                    for batch in range(leading[0])
                    for start in range(0, leading[1], step)
                )
        elif value.shape[-1] != width:
            return
        else:
            if collect:
                slices = (value.detach().reshape(-1, width),)
            rows_seen = int(value.numel()) // width
        if rows_seen == 0:
            return
        key = (*_storage_key(weight), orientation)
        stat = self.stats.get(key)
        if stat is None:
            suffix = ""
            if weight.storage_offset() or weight.numel() != weight.untyped_storage().nbytes() // weight.element_size():
                suffix = f"[offset={weight.storage_offset()},shape={tuple(weight.shape)}]"
            stat = _ActivationStats(
                name=f"{name}{suffix}",
                weight=weight.detach(),
                orientation=orientation,  # type: ignore[arg-type]
                hessian=(
                    torch.zeros((width, width), dtype=torch.float64)
                    if collect else torch.empty(0)
                ),
            )
            self.stats[key] = stat
        if collect:
            for rows in slices:
                rows = rows.to(device="cpu", dtype=torch.float64)
                if not bool(torch.isfinite(rows).all()):
                    raise ValueError(
                        f"GPTQ calibration for {name!r} contains nonfinite activations"
                    )
                stat.hessian.addmm_(rows.T, rows)
        stat.rows += rows_seen

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        target = str(func)
        # Observer formulas are eager analysis, not additional model
        # operations. Disable dispatch so semantic observers can reconstruct
        # selected-expert activations without recursively observing themselves.
        with torch._C._DisableTorchDispatch():
            observations = self.registry.observe(target, func, args, kwargs, result)
            # An operation on known tensors alone is folded at export: it
            # computes another weight, and reads no matrix at run time.
            folded = self._track_static(func, target, args, kwargs, result)
            views = [
                (value, self._parameter_view(weight), orientation)
                for value, weight, orientation in observations
                if isinstance(value, torch.Tensor) and isinstance(weight, torch.Tensor)
                and not folded
            ]
        for value, view, orientation in views:
            if view is not None:
                self._record(value, view, orientation)
        if self.products is not None:
            self._follow_products(args, kwargs, result, views)
        return result

    def _follow_products(self, args, kwargs, result, views) -> None:
        products = self.products
        assert products is not None
        weights = frozenset(
            view.untyped_storage().data_ptr() for _, view, _ in views if view is not None
        )
        if not weights:
            weights = frozenset().union(*(
                products.get(item, frozenset())
                for item in pytree.tree_leaves((args, kwargs))
                if isinstance(item, torch.Tensor)
            ))
        if not weights:
            return
        for item in pytree.tree_leaves(result):
            if isinstance(item, torch.Tensor):
                products[item] = weights

    def last_products(self, output) -> frozenset[int]:
        """The storages of the weights whose matrix products last computed
        ``output``: of the output head, when ``output`` is the logits."""
        assert self.products is not None
        if not isinstance(output, torch.Tensor):
            return frozenset()
        return self.products.get(output, frozenset())


# Operations, besides views, whose results only rearrange, repeat or convert
# their inputs' values: a parameter read through them is read as it is.
_REARRANGEMENTS = frozenset({
    "aten.clone.default", "aten.contiguous.default", "aten._to_copy.default",
    "aten.to.dtype", "aten.to.dtype_layout", "aten.to.device", "aten.to.other",
    "aten.cat.default", "aten.stack.default", "aten.index_select.default",
    "aten.index.Tensor", "aten.flip.default", "aten.roll.default",
    "aten.repeat.default", "aten.repeat_interleave.self_Tensor",
    "aten.repeat_interleave.self_int", "aten.constant_pad_nd.default",
    "aten.lift_fresh_copy.default", "aten.copy.default", "aten.copy_.default",
    "aten.permute_copy.default", "aten.t_copy.default", "aten.transpose_copy.int",
    "aten.view_copy.default", "aten.slice_copy.Tensor", "aten.select_copy.int",
    "aten.expand_copy.default", "aten.unsqueeze_copy.default",
    "aten.squeeze_copy.dim", "aten.split_with_sizes_copy.default",
    "aten.unbind_copy.int", "aten.alias_copy.default", "aten.narrow_copy.default",
    "aten._unsafe_view.default",
})


def _normalize_batch(batch) -> tuple[torch.Tensor, ...]:
    if isinstance(batch, torch.Tensor):
        return (batch,)
    if isinstance(batch, tuple) and batch and all(isinstance(item, torch.Tensor) for item in batch):
        return batch
    raise TypeError(
        "each calibration item must be a Tensor or a nonempty tuple of Tensors"
    )


def _symmetric_matrix(
    matrix: torch.Tensor, config: QuantizationConfig, group_width: int | None = None,
) -> tuple[torch.Tensor, int, np.ndarray]:
    """Quantize an output-major matrix in groups along its rows.

    ``group_width`` is a run of each row that groups must not cross, when
    storage groups less than a whole row together. Returns the
    reconstructed matrix, the group size, and how often each code occurs
    (index 0 is the most negative), which decides what storage would cost.
    """
    width = matrix.shape[1] if group_width is None else group_width
    group = effective_group_size(width, config.group_size)
    source = matrix.detach().cpu()
    # Rows hold whole groups, so a few rows at a time quantize exactly as the
    # whole matrix does, without float64 temporaries the size of a large
    # embedding. Check finiteness first, as one pass over everything would.
    if not bool(torch.isfinite(source).all()):
        raise ValueError("integer weight quantization requires finite values")
    maximum_code = (1 << (config.bits - 1)) - 1
    counts = np.zeros(2 * maximum_code + 1, dtype=np.int64)
    reconstructed = torch.empty(source.shape, dtype=matrix.dtype)
    row_values = max(1, source[:1].numel())
    step = max(1, _SYMMETRIC_CHUNK_VALUES // row_values)
    for start in range(0, source.shape[0], step):
        rows = source[start:start + step]
        codes, scales = _quantize_arrays(
            rows.double().reshape(-1).numpy(),
            config.precision,
            group,
            config.scale_precision,
        )
        counts += np.bincount(codes + maximum_code, minlength=counts.size)
        values = torch.from_numpy(codes * np.repeat(scales, group)[:codes.size])
        reconstructed[start:start + step] = values.to(matrix.dtype).reshape(rows.shape)
    return reconstructed.to(device=matrix.device), group, counts


@dataclass(frozen=True)
class _Quantized:
    """One matrix region's quantization: ``reconstructed`` is ``None`` when
    it stays float16 (``method`` ``"float16"``), and ``reason`` says why
    GPTQ wasn't used or why integers weren't."""

    reconstructed: torch.Tensor | None
    group: int | None
    method: Literal["gptq", "symmetric", "float16"]
    reason: str | None = None


def _float16_reason(
    config: QuantizationConfig, group: int, values: int, counts: np.ndarray | None = None,
    groups: int | None = None,
) -> str | None:
    """Why storing ``values`` in integers would not be smaller than float16.

    The cost is the storage codec's own (see ``integer_storage_bytes``):
    packed codes and a scale per group, ``groups`` of them if given. For
    int4 the Huffman stream depends on the codes, ``counts``; without them
    a lower bound, one bit per value, is used. ``None`` means integers are
    smaller.
    """
    if counts is None:
        if config.precision == "int4":
            counts = np.zeros(15, dtype=np.int64)
            counts[7] = values
        else:
            counts = np.array([values])
    stored = integer_storage_bytes(
        counts, config.precision, group, config.scale_precision, groups=groups,
    )
    if stored < 2 * values:
        return None
    scale_bits = 32 if config.scale_precision == "float32" else 16
    return (
        f"int{config.bits} storage would take {stored * 8 / values:.3g} bits per value "
        f"(groups of {group}, each with a {scale_bits}-bit scale), no fewer than float16's 16"
    )


def _quantize_matrix(
    logical: torch.Tensor,
    config: QuantizationConfig,
    group_width: int | None = None,
    stat: _ActivationStats | None = None,
) -> _Quantized:
    """Quantize an output-major matrix, with GPTQ when ``stat`` is given.

    A matrix whose integer storage would not be smaller than float16, such
    as one with a scale for every value, is left as it is.
    """
    width = logical.shape[1]
    group = effective_group_size(width if group_width is None else group_width, config.group_size)
    values = logical.numel()
    reason = _float16_reason(config, group, values)
    if reason is not None:
        return _Quantized(None, group, "float16", reason)
    method: Literal["gptq", "symmetric"] = "symmetric"
    reconstructed = None
    if stat is not None:
        method = "gptq"
        if stat.hessian.numel() == 0:
            method = "symmetric"
            reason = (
                f"input width {width} is above {_GPTQ_MAX_WIDTH}; its GPTQ "
                f"Hessian would take {width * width * 8 / 2**30:.1f} GiB"
            )
        elif stat.rows < config.min_calibration_rows:
            method = "symmetric"
            reason = (
                f"only {stat.rows} calibration rows; "
                f"requires {config.min_calibration_rows}"
            )
        if method == "gptq":
            try:
                reconstructed, group = _gptq_matrix(logical, stat.hessian, config, group_width)
                # GPTQ fixes each group's scale before error feedback
                # changes the group's later columns, so a group can end
                # up without a largest code. Rounding through the
                # storage codec, whose scale rule is a fixed point,
                # puts every group on the grid storage reproduces.
                reconstructed, group, counts = _symmetric_matrix(
                    reconstructed, config, group_width,
                )
            except ValueError as error:
                method = "symmetric"
                reason = str(error)
                reconstructed = None
    if reconstructed is None:
        reconstructed, group, counts = _symmetric_matrix(logical, config, group_width)
    late = _float16_reason(config, group, values, counts)
    if late is not None:
        return _Quantized(None, group, "float16", late)
    return _Quantized(reconstructed, group, method, reason)


# The widest input GPTQ builds a Hessian for: 8192 x 8192 doubles are 512 MiB.
_GPTQ_MAX_WIDTH = 8192
# Values quantized per step by _symmetric_matrix.
_SYMMETRIC_CHUNK_VALUES = 1 << 21
# Convolution window values copied per step by GPTQ calibration.
_WINDOW_CHUNK_VALUES = 1 << 20


def _gptq_matrix(
    matrix: torch.Tensor,
    hessian: torch.Tensor,
    config: QuantizationConfig,
    group_width: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Quantize one output-major matrix with blocked GPTQ error feedback.

    ``group_width`` is as for ``_symmetric_matrix``.
    """
    original_device = matrix.device
    original_dtype = matrix.dtype
    work = matrix.detach().to(device="cpu", dtype=torch.float64).clone()
    if not bool(torch.isfinite(work).all()):
        raise ValueError("GPTQ requires finite matrix weights")
    width = work.shape[1]
    group = effective_group_size(
        width if group_width is None else group_width, config.group_size,
    )
    hessian = hessian.clone()
    diagonal = torch.diag(hessian)
    damp = float(config.damp_percent) * float(diagonal.mean())
    if not math.isfinite(damp) or damp <= 0:
        raise ValueError("GPTQ calibration Hessian has no usable diagonal")
    indices = torch.arange(width)
    hessian[indices, indices] += damp
    # On one thread: torch's thread pool makes these factorizations ten
    # times slower on a busy machine, and gains little on an idle one.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        chol = torch.linalg.cholesky(hessian)
        inverse = torch.cholesky_inverse(chol)
        inverse_factor = torch.linalg.cholesky(inverse, upper=True)
    except torch.linalg.LinAlgError as error:
        raise ValueError("GPTQ calibration Hessian is not positive definite") from error
    finally:
        torch.set_num_threads(threads)

    qmax = (1 << (config.bits - 1)) - 1
    # NumPy rather than torch for the many small per-column steps: torch's
    # thread pool makes each one far slower on a busy machine.
    values = work.numpy()
    factor = inverse_factor.numpy()
    result = np.empty_like(values)
    # Lazy batch updates (GPTQ, Frantar et al. 2022, section 3): error
    # feedback within a block of columns is applied column by column, and
    # to the columns after the block once, as one matrix product. Blocks
    # hold whole groups, so each group's scale is chosen from its columns
    # with every earlier column's feedback applied, as without blocking.
    # Rows are quantized independently, so a few at a time keep each
    # block's working set in cache, as for an output head of 50,000 rows.
    block = max(group, config.block_size - config.block_size % group)
    row_step = max(1, _GPTQ_CHUNK_VALUES // block)
    for row_start in range(0, values.shape[0], row_step):
        rows = slice(row_start, row_start + row_step)
        for block_start in range(0, width, block):
            block_end = min(width, block_start + block)
            count = block_end - block_start
            columns = values[rows, block_start:block_end].copy()
            errors = np.empty_like(columns)
            local = factor[block_start:block_end, block_start:block_end]
            scales = np.ones(columns.shape[0])
            for index in range(count):
                if index % group == 0:
                    # Storage's own scale rule: a group whose largest code
                    # stays the largest after error feedback is stored
                    # exactly as chosen.
                    scales = group_scales(
                        np.abs(columns[:, index:index + group]).max(axis=1), qmax,
                        config.scale_precision,
                    )
                    if not np.isfinite(scales).all():
                        raise ValueError(
                            "GPTQ scale is outside the selected scale precision"
                        )
                column = columns[:, index]
                quantized = np.clip(np.rint(column / scales), -qmax, qmax) * scales
                result[rows, block_start + index] = quantized
                feedback = (column - quantized) / local[index, index]
                errors[:, index] = feedback
                if index + 1 < count:
                    columns[:, index + 1:] -= np.outer(feedback, local[index, index + 1:])
            if block_end < width:
                values[rows, block_end:] -= errors @ factor[block_start:block_end, block_end:]
    return (
        torch.from_numpy(result).to(device=original_device, dtype=original_dtype), group,
    )


# Values of one block GPTQ works on at a time: 2 MiB of doubles.
_GPTQ_CHUNK_VALUES = 1 << 18


def expert_bank_effective_ndims(model: torch.nn.Module) -> dict[int, int]:
    """Map each ExpertFamily bank's storage to the rank of one expert's tensor.

    A bank of expert biases is stored as ``[experts, hidden]`` but is a vector
    per expert, so storage treats it like any other bias.
    """
    effective_ndims: dict[int, int] = {}
    from cattorch.experts import ExpertFamily
    for module in model.modules():
        if not isinstance(module, ExpertFamily):
            continue
        template_state = {}
        template_state.update(dict(
            module.template.named_parameters(remove_duplicate=False),
        ))
        template_state.update(dict(
            module.template.named_buffers(remove_duplicate=False),
        ))
        seen_locations = set()
        for name in module._state_order:
            location = module._bank_names[name]
            if location in seen_locations:
                continue
            seen_locations.add(location)
            bank = module._bank(name)
            effective_ndims[bank.untyped_storage().data_ptr()] = template_state[name].ndim
    return effective_ndims


def _is_weight_matrix(value: torch.Tensor, effective_ndim: int) -> bool:
    """Whether a parameter is a matrix, rather than a vector such as a bias.

    A parameter is a matrix when at least two of its dimensions (for an
    expert bank, of one expert's tensor) are larger than one. Biases,
    normalization weights, ``nn.MultiheadAttention``'s ``(1, 1, E)``
    ``bias_k`` and ``(N, 1)`` gains are vectors.
    """
    shape = value.shape[value.ndim - effective_ndim:] if effective_ndim else ()
    return sum(1 for size in shape if size > 1) >= 2


def _canonical_spec(value: torch.Tensor) -> tuple:
    """The storage elements ``value`` reads, as offset, shape and strides."""
    return (
        int(value.storage_offset()), tuple(value.shape),
        tuple(0 if size == 1 else int(stride) for size, stride in zip(value.shape, value.stride())),
    )


def _storage_numel(value: torch.Tensor) -> int:
    return value.untyped_storage().nbytes() // value.element_size()


def _view_mask(
    numel: int, views: Iterable[tuple[int, tuple[int, ...], tuple[int, ...]]],
) -> torch.Tensor:
    """A storage-sized mask of the elements the ``views`` read."""
    mask = torch.zeros(numel, dtype=torch.bool)
    for offset, shape, stride in views:
        mask.as_strided(shape, stride, offset).fill_(True)
    return mask


@dataclass(frozen=True)
class _Region:
    """Where one report entry's values sit in its parameter's storage.

    ``views`` are ``(offset, shape, strides)`` of the elements the entry
    covers; with ``remaining``, it covers the elements of those views that
    no other region of the parameter covers.
    """

    entry: int
    pointer: int
    numel: int
    views: tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]
    remaining: bool = False


def _region_masks(regions: Iterable[_Region]) -> dict[int, torch.Tensor]:
    """Each region's storage mask, by report entry."""
    masks: dict[int, torch.Tensor] = {}
    taken: dict[int, torch.Tensor] = {}
    ordered = sorted(regions, key=lambda region: region.remaining)
    for region in ordered:
        mask = _view_mask(region.numel, region.views)
        if region.remaining:
            mask &= ~taken.get(region.pointer, torch.zeros_like(mask))
        else:
            previous = taken.get(region.pointer)
            taken[region.pointer] = mask.clone() if previous is None else previous | mask
        masks[region.entry] = mask
    return masks


@dataclass
class _Use:
    """One matrix region of a parameter, with every use that reads it so."""

    stat: _ActivationStats
    logical: torch.Tensor
    group_width: int | None
    mask: torch.Tensor


def _quantize_remaining(
    parameter: torch.Tensor, mask: torch.Tensor, config: QuantizationConfig,
) -> _Quantized:
    """Quantize the masked elements of ``parameter`` along its last dimension.

    Groups run along rows as storage reads a parameter. A group only part of
    which is masked is quantized over its masked elements alone: the rest
    already belong to a region quantized in another layout. Elements whose
    integer storage would not be smaller than float16 are left as they are.
    """
    width = int(parameter.shape[-1])
    group = effective_group_size(width, config.group_size)
    rows = parameter.detach().reshape(-1, width)
    selected = mask.as_strided(
        parameter.shape, parameter.stride(), parameter.storage_offset(),
    ).reshape(-1, width)
    grouped = rows.reshape(-1, group)
    chosen = selected.reshape(-1, group)
    full = chosen.all(dim=1)
    partial = chosen.any(dim=1) & ~full
    if not bool(torch.isfinite(rows[selected]).all()):
        raise ValueError("integer weight quantization requires finite values")
    maximum_code = (1 << (config.bits - 1)) - 1
    counts = np.zeros(2 * maximum_code + 1, dtype=np.int64)
    result = grouped.clone()
    if bool(full.any()):
        values = grouped[full]
        codes, scales = _quantize_arrays(
            values.double().reshape(-1).numpy(), config.precision, group,
            config.scale_precision,
        )
        counts += np.bincount(codes + maximum_code, minlength=counts.size)
        result[full] = torch.from_numpy(
            codes * np.repeat(scales, group),
        ).to(parameter.dtype).reshape(values.shape)
    for index in torch.nonzero(partial).flatten().tolist():
        keep = chosen[index]
        values = grouped[index][keep]
        codes, scales = _quantize_arrays(
            values.double().numpy(), config.precision, values.numel(), config.scale_precision,
        )
        counts += np.bincount(codes + maximum_code, minlength=counts.size)
        result[index, keep] = torch.from_numpy(codes * scales[0]).to(parameter.dtype)
    group_count = int(full.sum()) + int(partial.sum())
    reason = _float16_reason(config, group, int(counts.sum()), counts, groups=group_count)
    if reason is not None:
        return _Quantized(None, group, "float16", reason)
    parameter.copy_(result.reshape(parameter.shape))
    return _Quantized(parameter, None if bool(partial.any()) else group, "symmetric")


def _unwrapped(model: torch.nn.Module) -> torch.nn.Module:
    """The caller's model inside an export's method wrapper, for its names."""
    if type(model).__name__ in {"MethodModule", "MethodLeaves"}:
        return getattr(model, "model", model)
    return model


def _resolve_override_keys(
    config: QuantizationConfig,
    all_names: dict[int, list[str]],
    replaced: dict[str, tuple[str, str | None]],
) -> tuple[QuantizationConfig, dict[str, str]]:
    """``config`` with override keys that name the caller's own parameters
    resolved against the transformed model export quantizes, and the
    caller's key of each resolved one.

    ``replaced`` maps each module a transform replaced to how: fast mode's
    low-rank factorization (``("factorized", None)``), whose factors keep
    the module's path and take its overrides, or BatchNorm folding
    (``("folded", layer)``), which merges the BatchNorm into its layer.
    """
    names = {name for found in all_names.values() for name in found}
    overrides = dict(cast(dict, config.overrides))
    user_keys: dict[str, str] = {}
    for key, override in cast(dict, config.overrides).items():
        if any(name == key or name.startswith(f"{key}.") for name in names):
            continue
        for path, (how, layer) in replaced.items():
            if key != path and not key.startswith(f"{path}."):
                continue
            if how == "folded":
                raise ValueError(
                    f"quantization override key {key!r} names a BatchNorm that export "
                    f"folds into the layer before it, {layer!r}; override {layer!r} instead"
                )
            # A factorized parameter's factors: the parameters under the
            # module with its last name, as `a.weight` selects `a.0.weight`
            # and `a.1.weight`.
            attribute = key[len(path) + 1:].rsplit(".", 1)[-1]
            factors = sorted(
                name for name in names
                if name.startswith(f"{path}.") and name.rsplit(".", 1)[-1] == attribute
            )
            if factors:
                del overrides[key]
                for name in factors:
                    overrides.setdefault(name, override)
                    user_keys[name] = key
            break
    if not user_keys:
        return config, {}
    return replace(config, overrides=overrides), user_keys


def _check_override_keys(
    config: QuantizationConfig, all_names: dict[int, list[str]], aliases: Iterable[str] = (),
) -> None:
    """Reject override keys that select no parameter."""
    names = {name for found in all_names.values() for name in found} | set(aliases)
    unknown = sorted(
        key for key in config.overrides
        if not any(name == key or name.startswith(f"{key}.") for name in names)
    )
    if unknown:
        raise ValueError(
            f"unknown quantization override key(s) {', '.join(map(repr, unknown))}: "
            "each must be a module path or parameter name of the model, as "
            "named_parameters() spells it"
        )


def _named(name: str, quantize):
    """Call ``quantize``, naming the parameter in its value errors."""
    try:
        return quantize()
    except ValueError as error:
        raise ValueError(f"{error} (parameter {name!r})") from error


# ── Weights computed from parameters ─────────────────────────────────────────
#
# Export folds every operation on known tensors alone into a constant, so a
# weight computed from parameters, such as ``weight_norm``'s or
# ``F.linear(x, w * g)``'s, reaches the matrix product as a computed tensor.
# Quantization quantizes that tensor as the product reads it, and the model
# then computes with the quantized tensor in place of the computation: a
# torch-function mode on the module that computes it returns the stored
# tensor for the operation that made it. Export traces the returned tensor
# as a constant, and every PyTorch reference computes with the same values.
#
# Operations are recognised by a recipe: the function and its arguments,
# each tensor argument by the parameter or buffer it is, or by its own
# recipe. Recipes don't depend on values, so a weight is recognised however
# its parameters have been rounded since.


def _argument_recipe(value, recipe_of):
    """A hashable stand-in for one argument, or ``_UNKNOWN``."""
    if isinstance(value, torch.Tensor):
        recipe = recipe_of(value)
        return _UNKNOWN if recipe is None else recipe
    if isinstance(value, (list, tuple)):
        items = tuple(_argument_recipe(item, recipe_of) for item in value)
        return _UNKNOWN if any(item is _UNKNOWN for item in items) else (type(value).__name__, items)
    if isinstance(value, slice):
        return ("slice", value.start, value.stop, value.step)
    if value is None or isinstance(value, (bool, int, float, complex, str, torch.dtype,
                                           torch.device, torch.memory_format, torch.layout)):
        return value
    if value is Ellipsis:
        return "..."
    return ("object", type(value).__qualname__, repr(value))


_UNKNOWN = object()


class _RecipeTracker:
    """Recipes of the tensors known before any input arrives.

    ``leaves`` maps the model's own tensors, by object, to their names.
    Every result of a function whose tensor arguments are all known is known
    too, and so is the result of a function without tensor arguments that
    isn't random (``_random_function``), such as ``torch.ones``. A function that writes to a known tensor with an unknown
    one makes it unknown. A known tensor is computed when some function on
    the way changed values rather than rearranged or copied them.
    """

    def __init__(self, leaves: Iterable[tuple[str, torch.Tensor]]):
        self.recipes: dict[int, tuple[weakref.ref, object, bool]] = {}
        self._leaves = []
        for name, tensor in leaves:
            self._leaves.append(tensor)
            self.recipes.setdefault(id(tensor), (weakref.ref(tensor), ("leaf", name), False))

    def _entry(self, tensor: torch.Tensor):
        entry = self.recipes.get(id(tensor))
        if entry is None or entry[0]() is not tensor:
            return None
        return entry

    def recipe_of(self, tensor: torch.Tensor):
        entry = self._entry(tensor)
        return None if entry is None else entry[1]

    def remember(self, tensor: torch.Tensor, recipe, computed: bool) -> None:
        self.recipes[id(tensor)] = (weakref.ref(tensor), recipe, computed)
        weakref.finalize(tensor, self._forget, id(tensor))

    def _forget(self, identity: int) -> None:
        entry = self.recipes.get(identity)
        if entry is not None and entry[0]() is None:
            del self.recipes[identity]

    def call_recipe(self, func, args, kwargs, result) -> tuple[object, bool] | None:
        """The recipe of ``func(*args, **kwargs)``, and whether it computes
        new values, if all its tensor arguments are known."""
        tensors = [
            item for item in pytree.tree_leaves((args, kwargs)) if isinstance(item, torch.Tensor)
        ]
        entries = [self._entry(item) for item in tensors]
        name = getattr(func, "__name__", "")
        writes = _writes(func, kwargs)
        if not tensors and not writes and not _random_function(name):
            # A factory of constants, such as ``torch.ones`` or ``torch.eye``,
            # is known by its arguments, as export folds it.
            arguments = _argument_recipe(args, self.recipe_of)
            keywords = _argument_recipe(tuple(sorted(kwargs.items())), self.recipe_of)
            if arguments is _UNKNOWN or keywords is _UNKNOWN:
                return None
            return (func, arguments, keywords), True
        if not tensors or any(entry is None for entry in entries):
            if writes:
                # A known tensor written with unknown values is no longer known.
                for item in tensors:
                    self.recipes.pop(id(item), None)
            return None
        if writes:
            return None
        arguments = _argument_recipe(args, self.recipe_of)
        keywords = _argument_recipe(tuple(sorted(kwargs.items())), self.recipe_of)
        if arguments is _UNKNOWN or keywords is _UNKNOWN:
            return None
        outputs = [
            item for item in pytree.tree_leaves(result) if isinstance(item, torch.Tensor)
        ]
        computed = (
            any(entry is not None and entry[2] for entry in entries)
            or name not in _RECIPE_REARRANGEMENTS
            or any(item.dtype not in {tensor.dtype for tensor in tensors} for item in outputs)
        )
        return (func, arguments, keywords), computed

    def remember_result(self, result, recipe, computed: bool) -> None:
        # A function that returns its argument itself, as ``flatten`` of a
        # tensor already flat does, leaves the argument's recipe.
        if isinstance(result, torch.Tensor):
            if self.recipe_of(result) is None:
                self.remember(result, recipe, computed)
        elif isinstance(result, (list, tuple)):
            for index, item in enumerate(result):
                if isinstance(item, torch.Tensor) and self.recipe_of(item) is None:
                    self.remember(item, (recipe, index), computed)


def _writes(func, kwargs) -> bool:
    """Whether a torch function writes to a tensor argument."""
    name = getattr(func, "__name__", "")
    return (
        (name.endswith("_") and not name.endswith("__"))
        or name in _IN_PLACE_OPERATORS or "out" in kwargs
    )


_IN_PLACE_OPERATORS = frozenset({
    "__setitem__", "__iadd__", "__isub__", "__imul__", "__itruediv__", "__ifloordiv__",
    "__imod__", "__ipow__", "__imatmul__", "__iand__", "__ior__", "__ixor__",
    "__ilshift__", "__irshift__",
})


# Functions, by name, that only rearrange, select or copy their arguments'
# values. A weight made by them alone is the parameter read in another
# layout, which quantization follows as such (``_GPTQCaptureMode``).
_RECIPE_REARRANGEMENTS = frozenset({
    "clone", "contiguous", "detach", "t", "T", "mT", "transpose", "swapaxes", "swapdims",
    "permute", "movedim", "moveaxis", "reshape", "reshape_as", "view", "view_as", "flatten",
    "unflatten", "squeeze", "unsqueeze", "expand", "expand_as", "narrow", "select",
    "__getitem__", "chunk", "split", "tensor_split", "unbind", "cat", "concat",
    "concatenate", "stack", "flip", "fliplr", "flipud", "roll", "repeat", "tile",
    "index_select", "as_strided", "to", "type", "float", "double", "half", "bfloat16",
    "__get__",
})


def _recipe_leaves(recipe) -> set[str]:
    """The names of the model's tensors a recipe reads."""
    found: set[str] = set()
    pending = [recipe]
    while pending:
        item = pending.pop()
        if isinstance(item, tuple):
            if len(item) == 2 and item[0] == "leaf" and isinstance(item[1], str):
                found.add(item[1])
            else:
                pending.extend(item)
    return found


def _relative_recipe(recipe, prefix: str):
    """``recipe`` with leaf names relative to the module at ``prefix``."""
    if not prefix:
        return recipe
    if isinstance(recipe, tuple):
        if len(recipe) == 2 and recipe[0] == "leaf" and isinstance(recipe[1], str):
            return ("leaf", recipe[1].removeprefix(f"{prefix}."))
        return tuple(_relative_recipe(item, prefix) for item in recipe)
    return recipe


def _module_leaves(module: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    """The tensors a module holds, by name: its parameters, its buffers, and
    tensors held as plain attributes, which export takes as constants too."""
    return [
        *module.named_parameters(remove_duplicate=False),
        *module.named_buffers(remove_duplicate=False),
        *_plain_tensors(module),
    ]


def _plain_tensors(module: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Tensors held as plain attributes, not as parameters or buffers, such
    as ``self.mask = torch.tril(...)``, by name."""
    found = []
    for path, child in module.named_modules(remove_duplicate=False):
        for attribute, value in vars(child).items():
            if isinstance(value, torch.Tensor) and not isinstance(value, torch.nn.Parameter):
                found.append((f"{path}.{attribute}" if path else attribute, value))
    return found


# Operations that make a tensor of a literal written in the model's code.
_LITERALS = frozenset({"aten.lift_fresh.default", "aten.lift_fresh_copy.default"})


def _random_function(name: str) -> bool:
    """Whether a function of this name (a torch function's or an ATen
    operation's) makes random values: without tensor arguments, every other
    function makes the same tensor each call, which export takes as a
    constant."""
    base = name.removeprefix("aten.").split(".")[0]
    return base.removesuffix("_") in _RANDOM_FUNCTIONS


_RANDOM_FUNCTIONS = frozenset({
    "rand", "randn", "randint", "randperm", "rand_like", "randn_like", "randint_like",
    "normal", "bernoulli", "multinomial", "poisson", "uniform", "exponential",
    "geometric", "log_normal", "cauchy", "random", "empty", "empty_like",
    "empty_strided", "native_dropout", "dropout",
})


@dataclass
class _DerivedWeight:
    """A weight computed from parameters, as the first call computed it."""

    tensor: torch.Tensor
    recipe: object
    owner: torch.nn.Module | None
    owner_path: str
    sources: frozenset[int]
    # Made by rearranging or copying, not computing new values.
    copy: bool = False


class _ComputedWeightCapture(TorchFunctionMode):
    """Give every weight computed from parameters one tensor across calls.

    While quantization observes the model, the first computation of a tensor
    from known tensors alone is kept, and later computations with the same
    recipe return it, so every calibration call reads one tensor, as it
    reads a parameter. The innermost module of ``owners`` computing it that
    holds every tensor it reads is where its quantized value is later put
    back; ``owners`` is the caller's model, not a wrapper export replaces.
    """

    def __init__(self, model: torch.nn.Module, owners: torch.nn.Module, on_new, on_written):
        super().__init__()
        self.tracker = _RecipeTracker(_module_leaves(model))
        self.owners = {id(module) for module in owners.modules()}
        self.paths: dict[int, str] = {}
        for path, module in model.named_modules(remove_duplicate=False):
            self.paths.setdefault(id(module), path)
        self.stack: list[torch.nn.Module] = []
        self.computed: dict[object, _DerivedWeight] = {}
        self.leaf_storages = {
            name: tensor.untyped_storage().data_ptr() for name, tensor in _module_leaves(model)
        }
        self.on_new = on_new
        self.on_written = on_written
        self._hooks: list = []

    def __enter__(self):
        from torch.nn.modules.module import (
            register_module_forward_hook, register_module_forward_pre_hook,
        )
        self._hooks = [
            register_module_forward_pre_hook(lambda module, _args: self.stack.append(module)),
            register_module_forward_hook(
                lambda module, _args, _output: self._leave(module), always_call=True,
            ),
        ]
        return super().__enter__()

    def __exit__(self, *exc):
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self.stack.clear()
        return super().__exit__(*exc)

    def _leave(self, module) -> None:
        if self.stack and self.stack[-1] is module:
            self.stack.pop()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self.computed and _writes(func, kwargs):
            # A weight the model writes to is not one value across calls:
            # each call computes its own again, and it isn't quantized.
            written = {
                item.untyped_storage().data_ptr()
                for item in pytree.tree_leaves((args, kwargs)) if isinstance(item, torch.Tensor)
            }
            for recipe, weight in list(self.computed.items()):
                if weight.tensor.untyped_storage().data_ptr() in written:
                    del self.computed[recipe]
                    self.on_written(weight)
        result = func(*args, **kwargs)
        found = self.tracker.call_recipe(func, args, kwargs, result)
        if found is None:
            return result
        recipe, computed_values = found
        if (
            isinstance(result, torch.Tensor) and result.is_floating_point()
            and _recipe_leaves(recipe)
        ):
            computed = self.computed.get(recipe)
            if computed is not None:
                result = computed.tensor
            elif not any(
                result.untyped_storage().data_ptr() == item.untyped_storage().data_ptr()
                for item in pytree.tree_leaves((args, kwargs)) if isinstance(item, torch.Tensor)
            ):
                # A tensor of its own, not a view. Kept outside inference
                # mode, so quantization can write it.
                with torch.inference_mode(False):
                    result = result.clone()
                leaves = _recipe_leaves(recipe)
                owner, path = None, ""
                for module in reversed(self.stack):
                    candidate = self.paths.get(id(module))
                    if id(module) in self.owners and candidate is not None and all(
                        not candidate or leaf.startswith(f"{candidate}.") for leaf in leaves
                    ):
                        owner, path = module, candidate
                        break
                computed = _DerivedWeight(
                    tensor=result, recipe=recipe, owner=owner, owner_path=path,
                    sources=frozenset(
                        self.leaf_storages[leaf] for leaf in leaves if leaf in self.leaf_storages
                    ),
                    copy=not computed_values,
                )
                self.computed[recipe] = computed
                self.on_new(computed)
        self.tracker.remember_result(result, recipe, computed_values)
        return result


class _ComputedWeights(TorchFunctionMode):
    """Compute a module's weights from parameters as the stored tensors."""

    def __init__(self, module: torch.nn.Module, table: dict):
        super().__init__()
        self.table = table
        self.tracker = _RecipeTracker(_module_leaves(module))

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        found = self.tracker.call_recipe(func, args, kwargs, result)
        if found is None:
            return result
        recipe, computed = found
        stored = self.table.get(recipe)
        if stored is not None and isinstance(result, torch.Tensor):
            result = stored
        self.tracker.remember_result(result, recipe, computed)
        return result


class _ComputedWeightHooks:
    """Forward hooks that put a module's stored computed weights in place.

    The table maps each weight's recipe, its tensor names relative to the
    module, to the tensor the module computes with instead. Copies of the
    module copy the hooks and their table.
    """

    def __init__(self, table: dict):
        self.table = table
        self.modes: list[_ComputedWeights] = []

    def enter(self, module, _args) -> None:
        mode = _ComputedWeights(module, self.table)
        mode.__enter__()
        self.modes.append(mode)

    def leave(self, module, _args, _output) -> None:
        if self.modes:
            self.modes.pop().__exit__(None, None, None)


def _install_computed_weights(weights: Iterable[_DerivedWeight]) -> None:
    """Make each weight's module compute it as its tensor."""
    tables: dict[int, tuple[torch.nn.Module, str, dict]] = {}
    for weight in weights:
        if weight.owner is None:
            continue
        entry = tables.setdefault(id(weight.owner), (weight.owner, weight.owner_path, {}))
        entry[2][_relative_recipe(weight.recipe, weight.owner_path)] = weight.tensor
    for module, _, table in tables.values():
        hooks = _ComputedWeightHooks(table)
        module.register_forward_pre_hook(hooks.enter, prepend=True)
        module.register_forward_hook(hooks.leave, always_call=True)


def _computed_weight_name(
    model: torch.nn.Module, weight: _DerivedWeight, source_names: list[str],
) -> str:
    """The name a weight computed from ``source_names`` is reported under.

    A copy of one parameter is named after it; a parametrized attribute, or
    one ``torch.nn.utils.weight_norm`` or ``spectral_norm`` computes, as the
    attribute; any other weight after the parameters it is computed from.
    """
    import re
    if weight.copy and len(source_names) == 1:
        return source_names[0]
    parametrized = {
        match.groups() for match in (
            re.fullmatch(r"(?:(.*)\.)?parametrizations\.([^.]+)\.original\d*", name)
            for name in source_names
        ) if match
    }
    if len(parametrized) == 1 and all("parametrizations." in name for name in source_names):
        prefix, attribute = next(iter(parametrized))
        return f"{prefix}.{attribute}" if prefix else attribute
    for prefix, attribute in _hooked_attributes(model):
        sources = {f"{prefix}{attribute}{suffix}" for suffix in ("_g", "_v", "_orig")}
        if source_names and set(source_names) <= sources:
            return f"{prefix}{attribute}"
    return f"weight computed from {', '.join(sorted(source_names))}"


def _hooked_attributes(model: torch.nn.Module) -> list[tuple[str, str]]:
    """``(module prefix, attribute)`` of each ``weight_norm``/``spectral_norm`` hook."""
    from torch.nn.utils.spectral_norm import SpectralNorm
    from torch.nn.utils.weight_norm import WeightNorm
    found = []
    for path, module in model.named_modules():
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, (WeightNorm, SpectralNorm)):
                found.append((f"{path}." if path else "", hook.name))
    return found


def _parameter_aliases(model: torch.nn.Module) -> list[str]:
    """Names of computed attributes that overrides may select, as
    ``fc.weight`` for a parametrized or ``weight_norm`` module."""
    from torch.nn.utils import parametrize
    aliases = [
        f"{prefix}{attribute}" for prefix, attribute in _hooked_attributes(model)
    ]
    for path, module in model.named_modules():
        if parametrize.is_parametrized(module):
            prefix = f"{path}." if path else ""
            aliases.extend(f"{prefix}{attribute}" for attribute in module.parametrizations)
    return aliases


# Conversions a copy may make; positions don't convert.
_RECIPE_CONVERSIONS = frozenset({"to", "type", "float", "double", "half", "bfloat16"})


def _replay_recipe(recipe, leaves: dict[str, torch.Tensor]):
    """Evaluate ``recipe`` with ``leaves`` for the model's tensors."""
    if isinstance(recipe, tuple) and len(recipe) == 2 and recipe[0] == "leaf":
        return leaves[recipe[1]]
    if isinstance(recipe, tuple) and len(recipe) == 3 and callable(recipe[0]):
        func, arguments, keywords = recipe
        args = _replay_argument(arguments, leaves)
        kwargs = dict(_replay_argument(keywords, leaves))
        if getattr(func, "__name__", "") in _RECIPE_CONVERSIONS:
            return args[0]
        return func(*args, **kwargs)
    if (
        isinstance(recipe, tuple) and len(recipe) == 2 and isinstance(recipe[0], tuple)
        and isinstance(recipe[1], int)
    ):
        return _replay_recipe(recipe[0], leaves)[recipe[1]]
    raise ValueError("not a recipe")


def _replay_argument(value, leaves: dict[str, torch.Tensor]):
    if isinstance(value, tuple) and value:
        tag = value[0]
        if tag == "leaf" or callable(tag) or (
            len(value) == 2 and isinstance(tag, tuple) and isinstance(value[1], int)
        ):
            return _replay_recipe(value, leaves)
        if tag in {"tuple", "list"} and len(value) == 2 and isinstance(value[1], tuple):
            items = [_replay_argument(item, leaves) for item in value[1]]
            return tuple(items) if tag == "tuple" else items
        if tag == "slice":
            return slice(*value[1:])
        if tag == "object":
            raise ValueError("an argument recipes can't rebuild")
    if isinstance(value, str) and value == "...":
        return Ellipsis
    return value


def _copy_positions(
    weight: _DerivedWeight, leaves: dict[str, torch.Tensor], parameters: dict[int, tuple],
) -> tuple[int, torch.Tensor] | None:
    """``(parameter storage, the parameter element at each storage element
    of the copy)`` of a weight copied from one parameter, as
    ``w.t().contiguous()`` or ``w.clone()`` copies it, or ``None``."""
    if not weight.copy or len(weight.sources) != 1:
        return None
    (source,) = weight.sources
    if source not in parameters:
        return None
    positions: dict[str, torch.Tensor] = {}
    for name in _recipe_leaves(weight.recipe):
        tensor = leaves.get(name)
        if tensor is None or tensor.untyped_storage().data_ptr() != source:
            return None
        base = torch.arange(_storage_numel(tensor), dtype=torch.float64)
        positions[name] = base.as_strided(tensor.shape, tensor.stride(), tensor.storage_offset())
    copy = weight.tensor
    try:
        ids = _replay_recipe(weight.recipe, positions)
        if not isinstance(ids, torch.Tensor) or ids.shape != copy.shape:
            return None
        found = torch.full((_storage_numel(copy),), -1, dtype=torch.long)
        found.as_strided(copy.shape, copy.stride(), copy.storage_offset()).copy_(ids.long())
    except (RuntimeError, TypeError, ValueError, IndexError, KeyError):
        return None
    return source, found


def _strided_view(ids: torch.Tensor) -> tuple[int, tuple[int, ...], tuple[int, ...]] | None:
    """``(offset, shape, strides)`` of the storage view whose elements are
    ``ids``, if one is."""
    if not ids.numel() or bool((ids < 0).any()):
        return None
    flat = ids.reshape(-1)
    offset = int(flat[0])
    strides = []
    for dim, size in enumerate(ids.shape):
        if size > 1:
            index = [0] * ids.ndim
            index[dim] = 1
            strides.append(int(ids[tuple(index)]) - offset)
        else:
            strides.append(0)
    numel = int(ids.max()) + 1
    try:
        rebuilt = torch.arange(numel, dtype=torch.long).as_strided(
            tuple(ids.shape), tuple(strides), offset,
        )
    except RuntimeError:
        return None
    if not torch.equal(rebuilt, ids):
        return None
    return offset, tuple(int(size) for size in ids.shape), tuple(strides)


def quantize_model(
    model: torch.nn.Module,
    config: QuantizationConfig,
    calibration_inputs: Iterable | None,
    *,
    orientation_inputs: Iterable | None = None,
    call_kwargs: dict | None = None,
    _registry: OperatorRegistry | None = None,
    _calibration_name: str = "calibration",
    _names_from: torch.nn.Module | None = None,
    _logits_of: Callable[[Any], torch.Tensor] | None = None,
) -> QuantizationReport:
    """Quantize eligible weights in-place and return a structured report.

    This is an internal export step. ``transpile`` deep-copies the model before
    calling it, so caller-owned parameters are never modified.

    Symmetric quantization runs the model once on ``orientation_inputs`` to see
    how each matrix is read: as a linear weight, on the right of a matmul,
    or as convolution kernels. Each region of a parameter that a use reads
    is quantized once, grouped over the inputs of that use, the layout
    Scratch storage uses for it. Elements no such use reads are grouped
    along the parameter's last dimension. Without observation, matrices
    keep their stored orientation.
    ``call_kwargs`` are passed to every observed model call.
    ``_calibration_name`` is the caller's name for ``calibration_inputs``,
    for its errors. ``_names_from`` is the module whose parameter names the
    report uses, when ``model`` wraps it.

    A generation export passes ``_logits_of``, which picks the logits from
    the model's output. The weights of the last matrix products that
    compute them on every observed call form the output head, which takes
    ``config.head_bits``.
    """
    effective_ndims = expert_bank_effective_ndims(model)

    def effective_ndim(value: torch.Tensor) -> int:
        return effective_ndims.get(value.untyped_storage().data_ptr(), value.ndim)

    # Every name of every parameter, as the caller's model spells it: tied
    # weights have several, and overrides may select any of them.
    names_module = _names_from if _names_from is not None else _unwrapped(model)
    all_names: dict[int, list[str]] = {}
    for name, parameter in names_module.named_parameters(remove_duplicate=False):
        all_names.setdefault(parameter.untyped_storage().data_ptr(), []).append(name)
    parameters_by_storage: dict[int, str] = {}
    base_parameters: dict[int, tuple[str, torch.Tensor]] = {}
    for name, parameter in model.named_parameters():
        pointer = parameter.untyped_storage().data_ptr()
        names = all_names.setdefault(pointer, [name])
        parameters_by_storage.setdefault(pointer, names[0])
        base_parameters.setdefault(pointer, (names[0], parameter))
    config, user_keys = _resolve_override_keys(
        config, all_names, getattr(names_module, "_cattorch_replaced_modules", {}),
    )
    _check_override_keys(config, all_names, _parameter_aliases(names_module))

    registry = _registry or default_registry()
    observing = config.method == "gptq" or orientation_inputs is not None
    if observing:
        from cattorch.frontend import capture_model
        for family in registry.expert_families.values():
            object.__setattr__(
                family, "_gptq_captured_template",
                capture_model(
                    family.template, (family.example_input,),
                ),
            )
    capture = _GPTQCaptureMode(
        parameters_by_storage, registry,
        collect_hessian=config.method == "gptq",
        static_storages=[
            tensor.untyped_storage().data_ptr()
            for tensor in (*model.buffers(), *(value for _, value in _plain_tensors(model)))
        ],
        follow_products=_logits_of is not None and observing,
    )
    # The storages of the output head's weights.
    head: set[int] = set()
    # Weights computed from parameters, by storage, with their name and the
    # names of the parameters they are computed from.
    computed: dict[int, tuple[_DerivedWeight, str, tuple[str, ...]]] = {}

    def computed_weight(weight: _DerivedWeight) -> None:
        pointer = weight.tensor.untyped_storage().data_ptr()
        sources = tuple(sorted({
            all_names[source][0] for source in weight.sources if source in all_names
        }))
        name = _computed_weight_name(names_module, weight, list(sources))
        # A copy of a parameter is reported as the parameter.
        computed[pointer] = (weight, name, () if name in sources else sources)
        capture.add_computed(weight.tensor, name, weight.sources)

    def written_weight(weight: _DerivedWeight) -> None:
        pointer = weight.tensor.untyped_storage().data_ptr()
        computed.pop(pointer, None)
        capture.remove_computed(weight.tensor)

    computed_capture = _ComputedWeightCapture(
        model, names_module, computed_weight, written_weight,
    )
    batches = 0
    if config.method == "gptq":
        if calibration_inputs is None:
            raise ValueError(f"GPTQ requires nonempty {_calibration_name}")
        observed_inputs = calibration_inputs
    elif calibration_inputs is not None:
        raise ValueError(f"{_calibration_name}= is only used by GPTQ quantization")
    else:
        observed_inputs = orientation_inputs or ()
    if observing:
        from cattorch.experts import _expert_family_context
        with (
            torch.inference_mode(), _expert_family_context(registry.expert_families),
            capture, computed_capture,
        ):
            for raw_batch in observed_inputs:
                batch = _normalize_batch(raw_batch)
                try:
                    output = model(*batch, **(call_kwargs or {}))
                except RuntimeError as error:
                    if config.method != "gptq":
                        raise
                    shapes = ", ".join(
                        f"{tuple(item.shape)} {str(item.dtype).removeprefix('torch.')}"
                        for item in batch
                    )
                    raise ValueError(
                        f"the model failed on {_calibration_name}[{batches}] "
                        f"(inputs {shapes}): {error}"
                    ) from error
                if capture.products is not None:
                    assert _logits_of is not None
                    head |= capture.last_products(_logits_of(output))
                batches += 1
        if config.method == "gptq" and batches == 0:
            raise ValueError(f"GPTQ requires nonempty {_calibration_name}")
        if config.method == "symmetric":
            batches = 0

    reports: list[TensorQuantizationReport] = []
    regions: list[_Region] = []
    gptq = config.method == "gptq"
    total_rows = sum(stat.rows for stat in capture.stats.values()) if gptq else 0
    settings_by_storage = {
        pointer: (config._head_settings if pointer in head else config._settings)(
            all_names[pointer],
        )
        for pointer in base_parameters
    }
    # A copy of one parameter, as ``F.linear(x, w.t().contiguous())`` or
    # ``x @ w.clone()`` reads, is the parameter read in another layout: its
    # uses are uses of the parameter, so the parameter is quantized once for
    # all of them, and the copy then holds the parameter's quantized values.
    leaves = dict(_module_leaves(model))
    copies: dict[int, tuple[_DerivedWeight, int, torch.Tensor]] = {}
    for pointer, (weight, _, _) in list(computed.items()):
        found = _copy_positions(weight, leaves, base_parameters)
        if found is not None:
            copies[pointer] = (weight, *found)
            del computed[pointer]
    if copies:
        stats: dict[tuple, _ActivationStats] = {}
        for key, stat in capture.stats.items():
            copied = copies.get(stat.weight.untyped_storage().data_ptr())
            view = None if copied is None else _strided_view(copied[2].as_strided(
                stat.weight.shape, stat.weight.stride(), stat.weight.storage_offset(),
            ))
            if copied is not None and view is not None:
                name, parameter = base_parameters[copied[1]]
                offset, shape, strides = view
                weight = parameter.detach().as_strided(shape, strides, offset)
                suffix = ""
                if offset or weight.numel() != _storage_numel(parameter):
                    suffix = f"[offset={offset},shape={shape}]"
                stat = replace(stat, name=f"{name}{suffix}", weight=weight)
                key = (*_storage_key(weight), stat.orientation)
                previous = stats.get(key)
                if previous is not None:
                    if previous.hessian.numel():
                        previous.hessian = previous.hessian + stat.hessian
                    previous.rows += stat.rows
                    continue
            stats[key] = stat
        capture.stats = stats
    # A computed weight read as a matrix is quantized like a parameter, and
    # takes the settings of the parameters it is computed from.
    read = {stat.weight.untyped_storage().data_ptr() for stat in capture.stats.values()}
    computed = {
        pointer: entry for pointer, entry in computed.items()
        if pointer in read and entry[0].owner is not None
    }
    sources_by_storage: dict[int, tuple[str, ...]] = {}
    for pointer, (weight, name, sources) in computed.items():
        base_parameters[pointer] = (name, weight.tensor)
        sources_by_storage[pointer] = sources
        settings_by_storage[pointer] = (
            config._head_settings if pointer in head else config._settings
        )([
            name, *(alias for source in weight.sources for alias in all_names.get(source, ())),
        ], every_name=False)

    def report(item: TensorQuantizationReport, pointer: int | None = None, views=(), remaining=False):
        if pointer is not None:
            regions.append(_Region(
                entry=len(reports), pointer=pointer,
                numel=_storage_numel(base_parameters[pointer][1]),
                views=tuple(views), remaining=remaining,
            ))
        reports.append(item)

    def kept_float16(name: str, values: int, reason: str, rows: int = 0) -> TensorQuantizationReport:
        return TensorQuantizationReport(
            name=name, method="float16", bits=None, values=values,
            calibration_rows=rows, effective_group_size=None, fallback_reason=reason,
        )

    with torch.no_grad():
        # Every region of a parameter that a use reads as a matrix. Uses that
        # read the same elements the same way share one region; regions that
        # share elements but group them differently can't all be stored in
        # integers, so the largest, then the first used, is quantized.
        uses_by_storage: dict[int, list[_Use]] = {}
        for stat in capture.stats.values():
            pointer = stat.weight.untyped_storage().data_ptr()
            base = base_parameters.get(pointer)
            if base is None or base[1].numel() < config.min_quantized_values:
                continue
            logical = _logical_matrix(stat.weight, stat.orientation)
            group_width = _storage_group_width(stat.weight, stat.orientation)
            numel = _storage_numel(base[1])
            uses = uses_by_storage.setdefault(pointer, [])
            same = next(
                (
                    use for use in uses
                    if use.group_width == group_width
                    and logical.untyped_storage().data_ptr() == pointer
                    and use.logical.untyped_storage().data_ptr() == pointer
                    and _canonical_spec(use.logical) == _canonical_spec(logical)
                ),
                None,
            )
            if same is not None:
                if gptq and same.stat.hessian.numel():
                    same.stat.hessian += stat.hessian
                same.stat.rows += stat.rows
                continue
            uses.append(_Use(
                stat=_ActivationStats(
                    name=stat.name, weight=stat.weight, orientation=stat.orientation,
                    hessian=stat.hessian.clone() if gptq else stat.hessian, rows=stat.rows,
                ),
                logical=logical, group_width=group_width,
                mask=_view_mask(numel, (_canonical_spec(stat.weight),)),
            ))

        quantized_storage: set[int] = set()
        unquantized: dict[int, str] = {}
        for pointer, uses in uses_by_storage.items():
            taken = torch.zeros_like(uses[0].mask)
            settings = settings_by_storage[pointer]
            for use in sorted(uses, key=lambda use: -int(use.mask.sum())):
                if bool((taken & use.mask).any()):
                    continue
                taken |= use.mask
                stat = use.stat
                name = base_parameters[pointer][0]
                quantized_storage.add(pointer)
                views = (_canonical_spec(stat.weight),)
                rows = stat.rows if gptq else 0
                sources = sources_by_storage.get(pointer, ())
                if settings.bits is None:
                    report(replace(kept_float16(
                        stat.name, stat.weight.numel(),
                        settings.float16_reason(), rows,
                    ), sources=sources, override=settings.key), pointer, views)
                    continue
                weight_config = settings.config(config)
                outcome = _named(name, lambda: _quantize_matrix(
                    use.logical, weight_config, use.group_width,
                    stat if weight_config.method == "gptq" else None,
                ))
                if outcome.reconstructed is None:
                    report(replace(kept_float16(
                        stat.name, stat.weight.numel(), str(outcome.reason), rows,
                    ), sources=sources, override=settings.key), pointer, views)
                    continue
                stat.weight.copy_(_physical_weight(
                    outcome.reconstructed, stat.weight, stat.orientation,
                ))
                report(TensorQuantizationReport(
                    name=stat.name,
                    method=outcome.method,
                    bits=weight_config.bits,
                    values=stat.weight.numel(),
                    calibration_rows=rows,
                    effective_group_size=outcome.group,
                    fallback_reason=outcome.reason,
                    sources=sources,
                    override=settings.key,
                ), pointer, views)

        # Elements no use reads as a matrix, such as an untied embedding, are
        # grouped along the parameter's last dimension with the documented
        # symmetric fallback. Parameters only combined into other weights,
        # such as ``weight_norm``'s, are left as they are: the weight export
        # computes from them is what storage stores.
        for name, parameter in model.named_parameters():
            pointer = parameter.untyped_storage().data_ptr()
            name = base_parameters[pointer][0]
            if parameter.numel() < config.min_quantized_values:
                continue
            if pointer not in quantized_storage and (
                not _is_weight_matrix(parameter, effective_ndim(parameter))
                or (
                    observing and pointer in capture.derivations
                    and pointer not in capture.direct_reads
                )
            ):
                if _is_weight_matrix(parameter, effective_ndim(parameter)):
                    # Export computes with it before any input arrives. If
                    # storage holds some of its values as they are, such as
                    # the rows of a position table read at constant
                    # positions, the report says so (see
                    # ``_storage_quantization_report``).
                    unquantized[pointer] = name
                continue
            if pointer in quantized_storage and pointer not in uses_by_storage:
                continue
            quantized_storage.add(pointer)
            settings = settings_by_storage[pointer]
            numel = _storage_numel(parameter)
            taken = _view_mask(numel, (
                view for region in regions
                if region.pointer == pointer and not region.remaining
                for view in region.views
            ))
            if bool(taken.all()):
                continue  # every element belongs to a region of its uses
            matrices = (
                [((), parameter)] if parameter.ndim == 2 else [
                    (index, parameter[index])
                    for index in itertools.product(*map(range, parameter.shape[:-2]))
                ]
            )
            for index, matrix in matrices:
                spec = _canonical_spec(matrix)
                rest = _view_mask(numel, (spec,)) & ~taken
                count = int(rest.sum())
                if not count:
                    continue
                matrix_name = name if not index else (
                    f"{name}[offset={matrix.storage_offset()},shape={tuple(matrix.shape)}]"
                )
                remaining = count != matrix.numel()
                if remaining:
                    # Other elements belong to regions quantized for their uses.
                    matrix_name = f"{matrix_name}[remaining]"
                if settings.bits is None:
                    report(replace(kept_float16(
                        matrix_name, count, settings.float16_reason(),
                    ), override=settings.key), pointer, (spec,), remaining=remaining)
                    continue
                weight_config = settings.config(config)
                if remaining:
                    outcome = _named(
                        name, lambda: _quantize_remaining(matrix, rest, weight_config),
                    )
                else:
                    outcome = _named(name, lambda: _quantize_matrix(matrix, weight_config))
                    if outcome.reconstructed is not None:
                        matrix.copy_(outcome.reconstructed)
                if outcome.method == "float16":
                    report(replace(
                        kept_float16(matrix_name, count, str(outcome.reason)),
                        override=settings.key,
                    ), pointer, (spec,), remaining=remaining)
                    continue
                report(TensorQuantizationReport(
                    name=matrix_name,
                    method="symmetric",
                    bits=weight_config.bits,
                    values=count,
                    calibration_rows=0,
                    effective_group_size=outcome.group,
                    fallback_reason=(
                        "no compatible linear activation statistics"
                        if weight_config.method == "gptq" else None
                    ),
                    override=settings.key,
                ), pointer, (spec,), remaining=remaining)

        # The model now computes each computed weight as its tensor, quantized
        # where a use reads it, and so does every copy of the model.
        for weight, source, positions in copies.values():
            parameter = base_parameters[source][1]
            flat = parameter.detach().as_strided((_storage_numel(parameter),), (1,), 0)
            copy = weight.tensor
            copy.copy_(flat[positions.as_strided(
                copy.shape, copy.stride(), copy.storage_offset(),
            )])
        _install_computed_weights([
            *(weight for weight, _, _ in computed.values()),
            *(weight for weight, _, _ in copies.values()),
        ])

        reported_names = {item.name.split("[", 1)[0] for item in reports}
        used_storage = {
            stat.weight.untyped_storage().data_ptr() for stat in capture.stats.values()
        }
        for name, parameter in model.named_parameters():
            pointer = parameter.untyped_storage().data_ptr()
            name = base_parameters[pointer][0]
            if (
                parameter.numel() < config.min_quantized_values
                and name not in reported_names
                and (
                    pointer in used_storage
                    or _is_weight_matrix(parameter, effective_ndim(parameter))
                )
            ):
                reported_names.add(name)
                reports.append(kept_float16(
                    name, parameter.numel(),
                    f"below min_quantized_values={config.min_quantized_values}",
                ))

    formats = list(config._integer_formats())
    for settings in settings_by_storage.values():
        if settings.key == HEAD_BITS and settings.bits is not None:
            item = (f"int{settings.bits}", settings.group_size)
            if item not in formats:
                formats.append(item)
    if user_keys:
        # Entries name the override as the caller wrote it.
        reports = [
            replace(item, override=user_keys.get(item.override, item.override))
            if item.override is not None else item
            for item in reports
        ]
    gptq_values = sum(item.values for item in reports if item.method == "gptq")
    symmetric_values = sum(item.values for item in reports if item.method == "symmetric")
    return QuantizationReport(
        method=config.method,
        bits=config.bits,
        calibration_batches=batches,
        calibration_rows=total_rows,
        quantized_values=gptq_values + symmetric_values,
        gptq_values=gptq_values,
        symmetric_values=symmetric_values,
        tensors=tuple(reports),
        _regions=tuple(regions),
        _unquantized=unquantized,
        _copies={
            pointer: (base_parameters[source][1], positions)
            for pointer, (_, source, positions) in copies.items()
        },
        _computed={pointer: weight.tensor for pointer, (weight, _, _) in computed.items()},
        _integer_formats=tuple(formats),
    )


class _SourceRecordingLists(dict):
    """A compiler's stored lists, remembering the graph values behind each."""

    def __init__(self, initial, recorder: _StaticSourceRecorder):
        super().__init__(initial)
        self.recorder = recorder

    def __setitem__(self, key, value):
        recorder = self.recorder
        if key not in recorder.sources:
            recorder.sources[key] = frozenset(
                recorder.resolved.get(recorder.compiler.active_node, ()),
            )
        super().__setitem__(key, value)


class _StaticSourceRecorder:
    """Record which graph values each list a compiler stores is made from.

    A list named after a graph value holds that value. Any other list, such
    as a weight's row shards or a fused projection, holds values of those
    the compiler read for the node it was compiling when it made the list.
    """

    def __init__(self, compiler):
        self.compiler = compiler
        self.graph = compiler.graph
        self.resolve = compiler.graph.resolve_weight
        self.resolved: dict[str | None, set[str]] = {}
        self.sources: dict[str, frozenset[str]] = {}

        def resolve_weight(name):
            value = self.resolve(name)
            if isinstance(value, torch.Tensor):
                self.resolved.setdefault(compiler.active_node, set()).add(name)
            return value

        self.graph.resolve_weight = resolve_weight
        compiler.static_lists = _SourceRecordingLists(compiler.static_lists, self)

    def finish(self) -> None:
        self.graph.resolve_weight = self.resolve

    def graph_values(self, key: str, tensor: torch.Tensor) -> tuple[str, ...]:
        """The graph values whose elements list ``key`` holds."""
        if isinstance(self.resolve(key), torch.Tensor):
            return (key,)
        names = sorted(self.sources.get(key, ()))
        values = {name: self.resolve(name) for name in names}
        # A list holds no more values than the values it is made from, so a
        # weight's shard is not made from the bias read with it. A list made
        # from several values, as a fused projection is, is larger than each.
        large = tuple(name for name in names if values[name].numel() >= tensor.numel())
        return large or tuple(names)


# Graph operations that only rearrange their inputs' values.
_GRAPH_REARRANGEMENTS = _REARRANGEMENTS | frozenset({
    "aten.t.default", "aten.transpose.int", "aten.permute.default", "aten.view.default",
    "aten.reshape.default", "aten.expand.default", "aten.slice.Tensor",
    "aten.select.int", "aten.unsqueeze.default", "aten.squeeze.default",
    "aten.squeeze.dim", "aten.squeeze.dims", "aten.split.Tensor",
    "aten.split_with_sizes.default", "aten.chunk.default", "aten.unbind.int",
    "aten.narrow.default", "aten.alias.default", "aten.detach.default",
    "aten.detach_.default", "aten.flatten.using_ints", "aten.unflatten.int",
    "aten.movedim.int", "aten.swapaxes.default", "aten.numpy_T.default",
    "aten.as_strided.default", "aten.tensor_split.indices",
    "aten.tensor_split.sections", "<built-in function getitem>",
    "aten.embedding.default",
})
# Conversions keep a rearranged value's identity; positions don't convert.
_CONVERSION_KEYWORDS = frozenset({"dtype", "layout", "device", "pin_memory", "memory_format"})
_POSITION_BITS = 40


class _GraphProvenance:
    """Which parameter elements each graph value known at export holds.

    A value rearranged from parameters holds known elements of them; one
    computed from them by arithmetic is a weight derived from them as a
    whole. Each parameter element is numbered, and the graph's operations
    are replayed on the numbers.
    """

    def __init__(self, graph, parameters: dict[int, torch.Tensor], resolve, aliases=None):
        self.nodes = {node.name: node for node in graph.nodes}
        self.resolve = resolve
        self.parameters = parameters
        self.indices = {pointer: index for index, pointer in enumerate(parameters)}
        self.pointers = list(parameters)
        self.traced: dict[str, object] = {}
        # Tensors of their own that hold parameter elements, by storage:
        # (the parameter's storage, its element at each of theirs).
        self.aliases: dict[int, tuple[int, torch.Tensor]] = {
            pointer: found for pointer, found in (aliases or {}).items()
            if found[0] in self.indices
        }

    def parameter_of(self, value: torch.Tensor) -> int | None:
        pointer = value.untyped_storage().data_ptr()
        return pointer if pointer in self.parameters or pointer in self.aliases else None

    def _positions(self, value: torch.Tensor, pointer: int) -> torch.Tensor:
        alias = self.aliases.get(pointer)
        if alias is not None and pointer not in self.parameters:
            pointer, base = alias[0], alias[1].reshape(-1).clone()
        else:
            base = torch.arange(_storage_numel(value), dtype=torch.long)
        base += self.indices[pointer] << _POSITION_BITS
        return base.as_strided(value.shape, value.stride(), value.storage_offset())

    def _trace(self, name: str):
        """``("ids", numbers)``, ``("derived", parameters)`` or ``None``."""
        if name in self.traced:
            return self.traced[name]
        self.traced[name] = None
        node = self.nodes.get(name)
        value = self.resolve(name)
        result: object = None
        if isinstance(value, torch.Tensor) and (
            node is None or node.op != "call_function"
            or str(node.target) in {"aten.detach_.default"}
        ):
            pointer = self.parameter_of(value)
            if pointer is not None:
                result = ("ids", self._positions(value, pointer))
        elif node is not None and node.op == "call_function":
            result = self._trace_call(node)
        self.traced[name] = result
        return result

    def _trace_call(self, node):
        traced: dict[str, object] = {}
        missing = object()

        def argument(item):
            if hasattr(item, "name") and hasattr(item, "op"):
                found = self._trace(item.name)
                traced[item.name] = found
                if found is None:
                    value = self.resolve(item.name)
                    if (
                        isinstance(value, torch.Tensor) and not value.is_floating_point()
                        and not value.is_complex() and value.dtype != torch.bool
                    ):
                        # Positions known at export, such as a position
                        # table's indices, select elements as they are.
                        return value
                    shape = (
                        value.shape if isinstance(value, torch.Tensor)
                        else getattr(item.meta.get("val"), "shape", None)
                    )
                    if shape is None:
                        return missing
                    return torch.full(tuple(shape), -1, dtype=torch.long)
                if found[0] == "ids":
                    return found[1]
                return missing
            if isinstance(item, (list, tuple)):
                values = [argument(element) for element in item]
                if any(element is missing for element in values):
                    return missing
                return type(item)(values)
            return item

        args = argument(node.args)
        kwargs = {
            key: argument(value) for key, value in node.kwargs.items()
            if key not in _CONVERSION_KEYWORDS
        }
        sources: set[int] = set()
        for found in traced.values():
            if found is None:
                continue
            if found[0] == "ids":
                ids = found[1]
                if isinstance(ids, torch.Tensor):
                    known = ids[ids >= 0] >> _POSITION_BITS
                    sources.update(self.pointers[index] for index in known.unique().tolist())
                else:
                    for part in ids:
                        known = part[part >= 0] >> _POSITION_BITS
                        sources.update(
                            self.pointers[index] for index in known.unique().tolist()
                        )
            else:
                sources.update(found[1])
        if not sources:
            return None
        if (
            str(node.target) in _GRAPH_REARRANGEMENTS
            and args is not missing and not any(value is missing for value in kwargs.values())
        ):
            try:
                if str(node.target) in {"aten._to_copy.default", "aten.to.dtype"}:
                    result = args[0]
                else:
                    result = node.target(*args, **kwargs)
            except (RuntimeError, TypeError, ValueError, IndexError):
                result = None
            if isinstance(result, torch.Tensor) and result.dtype == torch.long:
                return ("ids", result)
            if isinstance(result, (list, tuple)) and all(
                isinstance(part, torch.Tensor) and part.dtype == torch.long for part in result
            ):
                return ("ids", tuple(result))
        return ("derived", frozenset(sources))

    def elements(self, names: Iterable[str]):
        """``(masks by parameter, derived from parameters)`` for graph values."""
        masks: dict[int, torch.Tensor] = {}
        derived: set[int] = set()
        for name in names:
            found = self._trace(name)
            if found is None:
                continue
            if found[0] == "derived" or not isinstance(found[1], torch.Tensor):
                derived.update(found[1] if found[0] == "derived" else ())
                continue
            ids = found[1].reshape(-1)
            ids = ids[ids >= 0]
            for index in (ids >> _POSITION_BITS).unique().tolist():
                pointer = self.pointers[index]
                mask = masks.get(pointer)
                if mask is None:
                    mask = masks[pointer] = torch.zeros(
                        _storage_numel(self.parameters[pointer]), dtype=torch.bool,
                    )
                positions = ids[(ids >> _POSITION_BITS) == index] & ((1 << _POSITION_BITS) - 1)
                mask[positions] = True
        return masks, frozenset(derived)


# Why a matrix export reads only at positions fixed before any input arrives
# is stored in float16.
_READ_AT_CONSTANT_POSITIONS = (
    "read only at positions fixed at export, as a position table read with "
    "torch.arange is: export stores the values read as they are, in float16"
)
# Why a weight computed from parameters stays float16 when quantization
# couldn't quantize it as computed.
_UNFOLLOWED_COMPUTATION = (
    "export computes it from parameters where quantization can't compute it from "
    "quantized values instead: outside the forward of any of the model's modules "
    "(as in an entry point method), or within an operation that also reads an input"
)


def _storage_sources(model: torch.nn.Module) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Tensors a module made from parameter elements before export, such as
    a generation step's position tables, by storage: the parameter and the
    element of its storage at each of their storage elements.

    A module lists them in ``_cattorch_storage_sources``, mapping a tensor
    attribute's name to ``(parameter, element positions)``.
    """
    found: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for module in model.modules():
        for name, (parameter, positions) in getattr(
            module, "_cattorch_storage_sources", {},
        ).items():
            tensor = getattr(module, name, None)
            if isinstance(tensor, torch.Tensor) and tensor.numel() == positions.numel():
                found[tensor.untyped_storage().data_ptr()] = (parameter, positions)
    return found


def _parameter_elements(
    graph_nodes, values: dict, name: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(parameter, the element of its storage at each position)`` of the
    graph value ``name``, when it only rearranges one parameter's elements.

    ``values`` holds the values of the graph's static nodes, parameters
    included.
    """
    nodes = list(graph_nodes)
    parameters = {
        value.untyped_storage().data_ptr(): value for value in values.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        and value.untyped_storage().data_ptr() != 0
    }

    class _Graph:
        pass

    graph = _Graph()
    graph.nodes = nodes  # type: ignore[attr-defined]
    found = _GraphProvenance(graph, parameters, values.get)._trace(name)
    if found is None or found[0] != "ids" or not isinstance(found[1], torch.Tensor):
        return None
    ids = found[1]
    if bool((ids < 0).any()) or not ids.numel():
        return None
    indices = (ids >> _POSITION_BITS).unique().tolist()
    if len(indices) != 1:
        return None
    parameter = list(parameters.values())[indices[0]]
    if values.get(name) is parameter:
        return None
    return parameter, (ids & ((1 << _POSITION_BITS) - 1)).contiguous()


def _storage_quantization_report(
    report: QuantizationReport,
    *,
    stored: dict,
    storage,
    effective_ndims: dict[str, int],
    recorder: _StaticSourceRecorder,
    model: torch.nn.Module,
) -> tuple[tuple[str, ...], QuantizationReport]:
    """Warnings and the report for what storage stored of quantized weights.

    ``stored`` is storage's own record of every list it stored
    (``StoredList``, by key), made as it stored them: the report says what
    it holds, not what storage would be expected to choose. Storage keeps a
    list in integers only when it is on the integer grid along its own last
    dimension, so quantized values it reads grouped another way, or a weight
    computed from parameters at export, are kept in float16. Each stored
    list is traced back through the graph to the parameter elements it
    holds, and every report entry then says how its values are stored.
    """
    if storage.precision not in {"int8", "int6", "int4"}:
        return _stored_quantization_report(report)
    names_module = _unwrapped(model)
    parameter_names: dict[int, str] = {}
    parameters: dict[int, torch.Tensor] = {}
    for name, parameter in names_module.named_parameters():
        pointer = parameter.untyped_storage().data_ptr()
        parameter_names.setdefault(pointer, name)
        parameters.setdefault(pointer, parameter)
    # A computed weight is stored as the tensor quantization made of it.
    for pointer, tensor in report._computed.items():
        parameters.setdefault(pointer, tensor)
    sources_of = {**report._copies, **_storage_sources(model)}
    for parameter, _ in sources_of.values():
        parameters.setdefault(parameter.untyped_storage().data_ptr(), parameter)
    provenance = _GraphProvenance(
        recorder.graph, parameters, recorder.resolve,
        {
            pointer: (parameter.untyped_storage().data_ptr(), positions)
            for pointer, (parameter, positions) in sources_of.items()
        },
    )
    region_pointers = {region.pointer for region in report._regions}

    findings = _StorageFindings()
    derived: dict[tuple, list] = {}
    for key, item in stored.items():
        if item.values == 0:
            continue
        shape = torch.empty(item.shape, device="meta")
        sources = recorder.graph_values(key, shape)
        masks, from_parameters = provenance.elements(sources)
        # Traced values are of this list alone.
        provenance.traced.clear()
        integer = item.precision in ("int8", "int6", "int4")
        unquantized = {
            pointer: int(mask.sum()) for pointer, mask in masks.items()
            if pointer in report._unquantized
        }
        if unquantized and not integer and item.shared_with is None:
            total = sum(unquantized.values())
            for pointer, count in unquantized.items():
                name = report._unquantized[pointer]
                findings.unquantized[name] = (
                    findings.unquantized.get(name, 0) + round(item.values * count / total)
                )
        masks = {pointer: mask for pointer, mask in masks.items() if pointer in region_pointers}
        target = findings.integer if integer else findings.float16
        for pointer, mask in masks.items():
            previous = target.get(pointer)
            target[pointer] = mask.clone() if previous is None else previous | mask
        if integer:
            continue
        if masks:
            findings.contributions.append((masks, item.values, item.reason))
        ndim = effective_ndims.get(key, len(item.shape))
        if (
            from_parameters and not masks
            and item.values >= storage.min_quantized_values
            and _is_weight_matrix(shape, min(ndim, len(item.shape)))
        ):
            names = tuple(sorted(parameter_names[pointer] for pointer in from_parameters))
            entry = derived.setdefault((names, sources), [0, item.shape])
            entry[0] += item.values

    for (names, _), (values, shape) in derived.items():
        description = f"weight computed from {', '.join(names)}"
        previous_item = findings.derived.get(description)
        findings.derived[description] = TensorQuantizationReport(
            name=description,
            method="float16",
            bits=None,
            values=values + (previous_item.values if previous_item is not None else 0),
            calibration_rows=0,
            effective_group_size=None,
            fallback_reason=_UNFOLLOWED_COMPUTATION,
            sources=names,
        )
        findings.derived_warnings[description] = (
            f"{description} {shape} is stored as float16, not {storage.precision}: "
            f"{_UNFOLLOWED_COMPUTATION}"
        )
    return _stored_quantization_report(
        report, findings,
        storage_precision=storage.precision,
        min_quantized_values=storage.min_quantized_values,
    )


__all__ = [
    "QuantizationConfig", "QuantizationOverride", "QuantizationReport",
    "TensorQuantizationReport",
]
