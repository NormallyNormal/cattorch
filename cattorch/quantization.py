"""Offline weight quantization used by Scratch static storage."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Literal, cast

import numpy as np
import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from cattorch.storage import _quantize_arrays, group_scales
from cattorch.operator_registry import OperatorRegistry, default_registry


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
    ``min_calibration_rows`` before accepting captured statistics. The only
    current fallback for an eligible matrix without compatible statistics is
    symmetric quantization.
    """

    bits: Literal[4, 6, 8]
    method: Literal["symmetric", "gptq"] = "symmetric"
    group_size: int = 64
    scale_precision: Literal["float32", "float16"] = "float16"
    min_quantized_values: int = 16_384
    damp_percent: float = 0.01
    block_size: int = 128
    min_calibration_rows: int = 128
    fallback: Literal["symmetric"] = "symmetric"

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
        if self.fallback != "symmetric":
            raise ValueError("fallback must be 'symmetric'")

    @property
    def precision(self) -> Literal["int4", "int6", "int8"]:
        return cast(
            Literal["int4", "int6", "int8"],
            {4: "int4", 6: "int6", 8: "int8"}[self.bits],
        )


@dataclass(frozen=True)
class TensorQuantizationReport:
    """Quantization decision for one parameter or parameter view."""

    name: str
    method: Literal["gptq", "symmetric", "float16"]
    bits: int | None
    values: int
    calibration_rows: int
    effective_group_size: int | None
    fallback_reason: str | None = None


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
    # say how storage stored them, and what the exports that share this
    # report found so far. Neither is part of the public report.
    _regions: tuple[_Region, ...] = field(default=(), repr=False, compare=False)
    _stored: dict = field(default_factory=dict, repr=False, compare=False)


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


def _stored_quantization_report(
    report: QuantizationReport,
    stored: dict[int, _StoredFloat16],
    derived: Iterable[TensorQuantizationReport] = (),
) -> QuantizationReport:
    """``report`` saying what storage stored, not only the rounding applied.

    ``stored`` holds, by entry, the values storage keeps in float16;
    ``derived`` describes weights computed from parameters at export. They
    add to what earlier exports sharing ``report`` found, as the entry points
    of one program do, and the result has no storage details of its own.
    """
    accumulated = report._stored
    for entry, finding in stored.items():
        previous = accumulated.get(entry)
        if previous is None or (finding.only, finding.copies) > (previous.only, previous.copies):
            accumulated[entry] = finding
    for item in derived:
        accumulated.setdefault(("derived", item.name), item)
    tensors: list[TensorQuantizationReport] = []
    for entry, item in enumerate(report.tensors):
        found: _StoredFloat16 | None = accumulated.get(entry)
        if found is None or item.method == "float16" or not (found.only or found.copies):
            tensors.append(item)
            continue
        stored_as = (
            f"{item.method} int{item.bits} values are stored as float16: {found.reason}"
        )
        if found.only >= item.values:
            tensors.append(replace(
                item, method="float16", bits=None, effective_group_size=None,
                fallback_reason=stored_as,
            ))
            continue
        tensors.append(replace(item, values=item.values - found.only))
        tensors.append(TensorQuantizationReport(
            name=f"{item.name}[float16]",
            method="float16",
            bits=None,
            values=found.only + found.copies,
            calibration_rows=0,
            effective_group_size=None,
            fallback_reason=(
                f"{found.copies} of these values are also stored in integers; {stored_as}"
                if found.copies else stored_as
            ),
        ))
    tensors.extend(
        item for key, item in accumulated.items()
        if isinstance(key, tuple) and key[0] == "derived"
    )
    gptq_values = sum(item.values for item in tensors if item.method == "gptq")
    symmetric_values = sum(item.values for item in tensors if item.method == "symmetric")
    return QuantizationReport(
        method=report.method,
        bits=report.bits,
        calibration_batches=report.calibration_batches,
        calibration_rows=report.calibration_rows,
        quantized_values=gptq_values + symmetric_values,
        gptq_values=gptq_values,
        symmetric_values=symmetric_values,
        tensors=tuple(tensors),
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

    Storage groups a convolution kernel along its last dimension, which is
    also the last in each logical row.
    """
    return int(weight.shape[-1]) if orientation in {"conv", "conv_transpose"} else None


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
        bases: dict[int, torch.Tensor] | None = None,
        static_storages: Iterable[int] = (),
    ):
        super().__init__()
        self.parameters = parameters
        self.registry = registry
        # Symmetric quantization only needs to know how each matrix is used.
        self.collect_hessian = collect_hessian
        self.stats: dict[tuple, _ActivationStats] = {}
        self.bases = dict(bases or {})
        # A copy's storage -> (a weak reference to it, the copy's layout in
        # it, the parameter view it copied), for weights a model copies
        # before using them. Views made in inference mode don't keep the
        # copy itself alive, only its storage. An operation that writes to a
        # copy forgets it.
        self.copies: dict[int, tuple[StorageWeakRef, tuple, torch.Tensor]] = {}
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
        entries = [self._static_entry(item) for item in inputs]
        input_storages = {item.untyped_storage().data_ptr() for item in inputs}
        mutable = bool(func._schema.is_mutable)
        known = [entry for entry in entries if entry is not None]
        if len(known) == len(entries):
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

    def _track_copy(self, source: torch.Tensor, copy: torch.Tensor) -> None:
        """Remember ``copy`` as the parameter values ``source`` holds."""
        pointer = copy.untyped_storage().data_ptr()
        if (
            copy.shape != source.shape or copy.dtype != source.dtype
            or not copy.is_floating_point()
            or pointer == source.untyped_storage().data_ptr()
            or pointer in self.parameters
        ):
            return
        view = self._parameter_view(source)
        if view is None:
            return
        self.copies = {
            key: entry for key, entry in self.copies.items() if not entry[0].expired()
        }
        self.copies[pointer] = (
            StorageWeakRef(copy.untyped_storage()), _storage_key(copy), view,
        )

    def _parameter_view(self, tensor: torch.Tensor) -> torch.Tensor | None:
        """``tensor`` as a view of a parameter, through any copies of one.

        A model can copy a weight into another layout before using it, as in
        ``F.linear(x, weight.t().contiguous())``. Quantizing the parameter
        through the view the copy was made from groups it as it is used, and
        the copy the model makes from it then holds the quantized values.
        """
        pointer = tensor.untyped_storage().data_ptr()
        if pointer in self.parameters:
            return tensor
        entry = self.copies.get(pointer)
        if entry is None:
            return None
        storage, layout, view = entry
        if storage.expired():
            return None
        if _storage_key(tensor) == layout:
            return view
        _, copy_offset, copy_shape, copy_stride = layout
        # A view of the copy reads the parameter positions the copy holds at
        # the places it reads.
        base = self.bases.get(view.untyped_storage().data_ptr())
        if base is None:
            return None
        values = base.untyped_storage().nbytes() // base.element_size()
        positions = torch.arange(values).as_strided(
            view.shape, view.stride(), view.storage_offset(),
        )
        held = torch.full(
            (tensor.untyped_storage().nbytes() // tensor.element_size(),), -1,
            dtype=torch.long,
        )
        held.as_strided(copy_shape, copy_stride, copy_offset).copy_(positions)
        read = held.as_strided(tensor.shape, tensor.stride(), tensor.storage_offset())
        if read.numel() == 0 or bool((read < 0).any()):
            return None
        origin = int(read[(0,) * read.ndim])
        strides = []
        for dimension, size in enumerate(read.shape):
            step = [0] * read.ndim
            step[dimension] = min(1, size - 1)
            strides.append(int(read[tuple(step)]) - origin if size > 1 else 1)
        last = origin + sum((size - 1) * stride for size, stride in zip(read.shape, strides))
        if min(strides, default=1) < 0 or last >= values or not torch.equal(
            torch.arange(values).as_strided(read.shape, strides, origin), read,
        ):
            return None
        return base.as_strided(read.shape, strides, origin)

    def _record(self, value: torch.Tensor, weight: torch.Tensor, orientation: str) -> None:
        if not value.is_floating_point() or not weight.is_floating_point():
            return
        name = self._parameter_name(weight)
        width = _matrix_width(weight, orientation)
        if name is None or not width:
            return
        if orientation in {"conv", "conv_transpose"}:
            # Windows ``[batch, *positions, channels, *kernel]``: only GPTQ
            # reads them. They are copied a few rows of positions at a time,
            # so the copies stay far smaller than every window at once.
            trailing = weight.ndim - 1
            if (
                not self.collect_hessian
                or value.ndim <= trailing
                or math.prod(value.shape[-trailing:]) != width
            ):
                return
            windows = value.detach()
            leading = windows.shape[:-trailing]
            rows_seen = math.prod(leading)
            if len(leading) == 1:
                windows, leading = windows.unsqueeze(1), (1, *leading)
            per_position = math.prod(leading[2:]) * width
            step = max(1, _WINDOW_CHUNK_VALUES // max(1, per_position))
            slices: Iterable[torch.Tensor] = (
                windows[batch, start:start + step].reshape(-1, width)
                for batch in range(leading[0])
                for start in range(0, leading[1], step)
            )
        elif value.shape[-1] != width:
            return
        else:
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
                    if self.collect_hessian else torch.empty(0)
                ),
            )
            self.stats[key] = stat
        if self.collect_hessian:
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
            if self.copies and func._schema.is_mutable:
                for argument in (*args, *kwargs.values()):
                    if isinstance(argument, torch.Tensor):
                        self.copies.pop(argument.untyped_storage().data_ptr(), None)
            if (
                target in _COPY_TARGETS and args
                and isinstance(args[0], torch.Tensor) and isinstance(result, torch.Tensor)
            ):
                self._track_copy(args[0], result)
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
        return result


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


# Operations whose result is a copy of their first argument's values, when it
# has the same shape and dtype.
_COPY_TARGETS = frozenset({
    "aten.clone.default", "aten.contiguous.default", "aten._to_copy.default",
    "aten.to.dtype", "aten.to.dtype_layout", "aten.to.device", "aten.to.other",
})


def _normalize_batch(batch) -> tuple[torch.Tensor, ...]:
    if isinstance(batch, torch.Tensor):
        return (batch,)
    if isinstance(batch, tuple) and batch and all(isinstance(item, torch.Tensor) for item in batch):
        return batch
    raise TypeError(
        "each calibration_inputs item must be a Tensor or a nonempty tuple of Tensors"
    )


def _symmetric_matrix(
    matrix: torch.Tensor, config: QuantizationConfig, group_width: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Quantize an output-major matrix in groups along its rows.

    ``group_width`` is a run of each row that groups must not cross, when
    storage groups less than a whole row together.
    """
    width = matrix.shape[1] if group_width is None else group_width
    group = effective_group_size(width, config.group_size)
    source = matrix.detach().cpu()
    # Rows hold whole groups, so a few rows at a time quantize exactly as the
    # whole matrix does, without float64 temporaries the size of a large
    # embedding. Check finiteness first, as one pass over everything would.
    if not bool(torch.isfinite(source).all()):
        raise ValueError("integer weight quantization requires finite values")
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
        values = torch.from_numpy(codes * np.repeat(scales, group)[:codes.size])
        reconstructed[start:start + step] = values.to(matrix.dtype).reshape(rows.shape)
    return reconstructed.to(device=matrix.device), group


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
    try:
        chol = torch.linalg.cholesky(hessian)
        inverse = torch.cholesky_inverse(chol)
        inverse_factor = torch.linalg.cholesky(inverse, upper=True)
    except torch.linalg.LinAlgError as error:
        raise ValueError("GPTQ calibration Hessian is not positive definite") from error

    result = torch.empty_like(work)
    qmax = (1 << (config.bits - 1)) - 1
    scales_by_group: dict[int, torch.Tensor] = {}
    for block_start in range(0, width, config.block_size):
        block_end = min(width, block_start + config.block_size)
        for column in range(block_start, block_end):
            group_start = column - column % group
            group_end = min(width, group_start + group)
            scales = scales_by_group.get(group_start)
            if scales is None:
                maximum = work[:, group_start:group_end].abs().amax(dim=1)
                # Storage's own scale rule: a group whose largest code stays
                # the largest after error feedback is stored exactly as chosen.
                scales = torch.from_numpy(group_scales(
                    maximum.numpy(), qmax, config.scale_precision,
                ))
                if not bool(torch.isfinite(scales).all()):
                    raise ValueError(
                        "GPTQ scale is outside the selected scale precision"
                    )
                scales_by_group[group_start] = scales
            quantized = torch.round(work[:, column] / scales).clamp(-qmax, qmax)
            quantized = quantized * scales
            result[:, column] = quantized
            divisor = inverse_factor[column, column]
            quant_error = (work[:, column] - quantized) / divisor
            if column + 1 < width:
                work[:, column + 1:] -= quant_error.unsqueeze(1) * inverse_factor[
                    column, column + 1:
                ].unsqueeze(0)
    return result.to(device=original_device, dtype=original_dtype), group


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
) -> tuple[int, int | None]:
    """Quantize the masked elements of ``parameter`` along its last dimension.

    Groups run along rows as storage reads a parameter. A group only part of
    which is masked is quantized over its masked elements alone: the rest
    already belong to a region quantized in another layout.
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
    result = grouped.clone()
    if bool(full.any()):
        values = grouped[full]
        codes, scales = _quantize_arrays(
            values.double().reshape(-1).numpy(), config.precision, group,
            config.scale_precision,
        )
        result[full] = torch.from_numpy(
            codes * np.repeat(scales, group),
        ).to(parameter.dtype).reshape(values.shape)
    for index in torch.nonzero(partial).flatten().tolist():
        keep = chosen[index]
        values = grouped[index][keep]
        codes, scales = _quantize_arrays(
            values.double().numpy(), config.precision, values.numel(), config.scale_precision,
        )
        result[index, keep] = torch.from_numpy(codes * scales[0]).to(parameter.dtype)
    parameter.copy_(result.reshape(parameter.shape))
    return int(selected.sum()), None if bool(partial.any()) else group


def _named(name: str, quantize):
    """Call ``quantize``, naming the parameter in its value errors."""
    try:
        return quantize()
    except ValueError as error:
        raise ValueError(f"{error} (parameter {name!r})") from error


def quantize_model(
    model: torch.nn.Module,
    config: QuantizationConfig,
    calibration_inputs: Iterable | None,
    *,
    orientation_inputs: Iterable | None = None,
    call_kwargs: dict | None = None,
    _registry: OperatorRegistry | None = None,
    _calibration_name: str = "calibration_inputs",
    _names_from: torch.nn.Module | None = None,
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
    """
    effective_ndims = expert_bank_effective_ndims(model)

    def effective_ndim(value: torch.Tensor) -> int:
        return effective_ndims.get(value.untyped_storage().data_ptr(), value.ndim)

    display_names: dict[int, str] = {}
    if _names_from is not None:
        for name, parameter in _names_from.named_parameters():
            display_names.setdefault(parameter.untyped_storage().data_ptr(), name)
    parameters_by_storage: dict[int, str] = {}
    base_parameters: dict[int, tuple[str, torch.Tensor]] = {}
    for name, parameter in model.named_parameters():
        pointer = parameter.untyped_storage().data_ptr()
        name = display_names.get(pointer, name)
        parameters_by_storage.setdefault(pointer, name)
        base_parameters.setdefault(pointer, (name, parameter))

    registry = _registry or default_registry()
    observing = config.method == "gptq" or orientation_inputs is not None
    if observing:
        from cattorch.frontend import capture_model
        for family in registry.expert_families.values():
            object.__setattr__(
                family, "_gptq_captured_template",
                capture_model(
                    family.template, (family.example_input,), frontend="fx",
                ),
            )
    capture = _GPTQCaptureMode(
        parameters_by_storage, registry,
        collect_hessian=config.method == "gptq",
        bases={pointer: parameter for pointer, (_, parameter) in base_parameters.items()},
        static_storages=[
            buffer.untyped_storage().data_ptr() for buffer in model.buffers()
        ],
    )
    batches = 0
    if config.method == "gptq":
        if calibration_inputs is None:
            raise ValueError(f"GPTQ requires nonempty {_calibration_name}")
        observed_inputs = calibration_inputs
    elif calibration_inputs is not None:
        raise ValueError("calibration_inputs can only be used with method='gptq'")
    else:
        observed_inputs = orientation_inputs or ()
    if observing:
        from cattorch.experts import _expert_family_context
        with torch.inference_mode(), _expert_family_context(registry.expert_families), capture:
            for raw_batch in observed_inputs:
                batch = _normalize_batch(raw_batch)
                try:
                    model(*batch, **(call_kwargs or {}))
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
                batches += 1
        if config.method == "gptq" and batches == 0:
            raise ValueError(f"GPTQ requires nonempty {_calibration_name}")
        if config.method == "symmetric":
            batches = 0

    reports: list[TensorQuantizationReport] = []
    regions: list[_Region] = []
    gptq = config.method == "gptq"
    total_rows = sum(stat.rows for stat in capture.stats.values()) if gptq else 0

    def report(item: TensorQuantizationReport, pointer: int | None = None, views=(), remaining=False):
        if pointer is not None:
            regions.append(_Region(
                entry=len(reports), pointer=pointer,
                numel=_storage_numel(base_parameters[pointer][1]),
                views=tuple(views), remaining=remaining,
            ))
        reports.append(item)

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
                if gptq:
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
        for pointer, uses in uses_by_storage.items():
            taken = torch.zeros_like(uses[0].mask)
            for use in sorted(uses, key=lambda use: -int(use.mask.sum())):
                if bool((taken & use.mask).any()):
                    continue
                taken |= use.mask
                stat = use.stat
                name = base_parameters[pointer][0]
                method: Literal["gptq", "symmetric"] = "gptq"
                fallback_reason = None
                if not gptq:
                    method = "symmetric"
                elif stat.rows < config.min_calibration_rows:
                    method = "symmetric"
                    fallback_reason = (
                        f"only {stat.rows} calibration rows; "
                        f"requires {config.min_calibration_rows}"
                    )
                if method == "gptq":
                    try:
                        reconstructed, group = _gptq_matrix(
                            use.logical, stat.hessian, config, use.group_width,
                        )
                        # GPTQ fixes each group's scale before error feedback
                        # changes the group's later columns, so a group can end
                        # up without a largest code. Rounding through the
                        # storage codec, whose scale rule is a fixed point,
                        # puts every group on the grid storage reproduces.
                        reconstructed, group = _symmetric_matrix(
                            reconstructed, config, use.group_width,
                        )
                    except ValueError as error:
                        method = "symmetric"
                        fallback_reason = str(error)
                if method == "symmetric":
                    reconstructed, group = _named(name, lambda: _symmetric_matrix(
                        use.logical, config, use.group_width,
                    ))
                stat.weight.copy_(_physical_weight(reconstructed, stat.weight, stat.orientation))
                quantized_storage.add(pointer)
                report(TensorQuantizationReport(
                    name=stat.name,
                    method=method,
                    bits=config.bits,
                    values=stat.weight.numel(),
                    calibration_rows=stat.rows if gptq else 0,
                    effective_group_size=group,
                    fallback_reason=fallback_reason,
                ), pointer, (_canonical_spec(stat.weight),))

        # Elements no use reads as a matrix, such as an untied embedding, are
        # grouped along the parameter's last dimension with the documented
        # symmetric fallback. Parameters only combined into other weights,
        # such as ``weight_norm``'s, are left as they are: the weight export
        # computes from them is what storage stores.
        fallback_reason = "no compatible linear activation statistics" if gptq else None
        for name, parameter in model.named_parameters():
            pointer = parameter.untyped_storage().data_ptr()
            name = display_names.get(pointer, name)
            if parameter.numel() < config.min_quantized_values:
                continue
            if pointer not in quantized_storage and (
                not _is_weight_matrix(parameter, effective_ndim(parameter))
                or (
                    observing and pointer in capture.derivations
                    and pointer not in capture.direct_reads
                )
            ):
                continue
            if pointer in quantized_storage and pointer not in uses_by_storage:
                continue
            quantized_storage.add(pointer)
            numel = _storage_numel(parameter)
            taken = _view_mask(numel, (
                view for region in regions
                if region.pointer == pointer and not region.remaining
                for view in region.views
            ))
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
                if count == matrix.numel():
                    reconstructed, group = _named(
                        name, lambda: _symmetric_matrix(matrix, config),
                    )
                    matrix.copy_(reconstructed)
                else:
                    # Other elements belong to regions quantized for their uses.
                    matrix_name = f"{matrix_name}[remaining]"
                    count, group = _named(
                        name, lambda: _quantize_remaining(matrix, rest, config),
                    )
                report(TensorQuantizationReport(
                    name=matrix_name,
                    method="symmetric",
                    bits=config.bits,
                    values=count,
                    calibration_rows=0,
                    effective_group_size=group,
                    fallback_reason=fallback_reason,
                ), pointer, (spec,), remaining=count != matrix.numel())


        reported_names = {item.name.split("[", 1)[0] for item in reports}
        used_storage = {
            stat.weight.untyped_storage().data_ptr() for stat in capture.stats.values()
        }
        for name, parameter in model.named_parameters():
            pointer = parameter.untyped_storage().data_ptr()
            name = display_names.get(pointer, name)
            if (
                parameter.numel() < config.min_quantized_values
                and name not in reported_names
                and (
                    pointer in used_storage
                    or _is_weight_matrix(parameter, effective_ndim(parameter))
                )
            ):
                reported_names.add(name)
                reports.append(TensorQuantizationReport(
                    name=name,
                    method="float16",
                    bits=None,
                    values=parameter.numel(),
                    calibration_rows=0,
                    effective_group_size=None,
                    fallback_reason=(
                        f"below min_quantized_values={config.min_quantized_values}"
                    ),
                ))

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

    def __init__(self, graph, parameters: dict[int, torch.Tensor], resolve):
        self.nodes = {node.name: node for node in graph.nodes}
        self.resolve = resolve
        self.parameters = parameters
        self.indices = {pointer: index for index, pointer in enumerate(parameters)}
        self.pointers = list(parameters)
        self.traced: dict[str, object] = {}

    def parameter_of(self, value: torch.Tensor) -> int | None:
        pointer = value.untyped_storage().data_ptr()
        return pointer if pointer in self.parameters else None

    def _positions(self, value: torch.Tensor, pointer: int) -> torch.Tensor:
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


def _stored_precisions(
    static_tensors, storage, *, sprite, layouts, effective_ndims, interleaved_layouts,
):
    """``(float16 values, all values, tensor-level precision)`` by stored list."""
    from cattorch.sprite import _static_storage_precision

    stored = {}
    for key, tensor in static_tensors.items():
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            continue
        precision = _static_storage_precision(
            tensor, storage, effective_ndims.get(key), interleaved_layouts.get(key, ()),
        )
        float16 = tensor.numel() if not precision.startswith("int") else 0
        layout = layouts.get(f"W_{key}")
        if precision.startswith("int") and layout is not None and sprite is not None:
            # Storage keeps a shard too small to be worth decoding in float16.
            for shard in layout.shards:
                size = len(sprite["lists"][shard.identifier][1])
                if 0 < size < storage.min_quantized_values:
                    float16 += size
        stored[key] = (float16, tensor.numel(), precision)
    return stored


def _storage_quantization_report(
    report: QuantizationReport,
    *,
    static_tensors: dict,
    deduplicated: dict,
    storage,
    sprite: dict,
    layouts: dict,
    effective_ndims: dict[str, int],
    interleaved_layouts: dict,
    recorder: _StaticSourceRecorder,
    model: torch.nn.Module,
) -> tuple[tuple[str, ...], QuantizationReport]:
    """Warnings and the report for what storage stores of quantized weights.

    Storage keeps a list in integers only when it is on the integer grid
    along its own last dimension, so quantized values it reads grouped
    another way, or a weight computed from parameters at export, are kept in
    float16. Each stored list is traced back through the graph to the
    parameter elements it holds, and every report entry then says how its
    values are stored. ``deduplicated`` are lists storage shares with an
    identical one; they are stored as that one is.
    """
    if storage.precision not in {"int8", "int6", "int4"} or storage.grouping != "row":
        return (), _stored_quantization_report(report, {})
    stored = _stored_precisions(
        static_tensors, storage, sprite=sprite, layouts=layouts,
        effective_ndims=effective_ndims, interleaved_layouts=interleaved_layouts,
    )
    stored.update(_stored_precisions(
        deduplicated, storage, sprite=None, layouts={},
        effective_ndims=effective_ndims, interleaved_layouts=interleaved_layouts,
    ))
    tensors = {**deduplicated, **static_tensors}
    names_module = getattr(model, "model", model) if type(model).__name__ == "MethodModule" else model
    parameter_names: dict[int, str] = {}
    parameters: dict[int, torch.Tensor] = {}
    for name, parameter in names_module.named_parameters():
        pointer = parameter.untyped_storage().data_ptr()
        parameter_names.setdefault(pointer, name)
        parameters.setdefault(pointer, parameter)
    provenance = _GraphProvenance(recorder.graph, parameters, recorder.resolve)
    regions = report._regions
    region_pointers = {region.pointer for region in regions}

    float16_masks: dict[int, torch.Tensor] = {}
    contributions: list[tuple[str, dict[int, torch.Tensor], int]] = []
    derived: dict[tuple, list] = {}
    for key, (float16, total, precision) in stored.items():
        if not float16:
            continue
        tensor = tensors[key]
        sources = recorder.graph_values(key, tensor)
        masks, from_parameters = provenance.elements(sources)
        masks = {pointer: mask for pointer, mask in masks.items() if pointer in region_pointers}
        for pointer, mask in masks.items():
            previous = float16_masks.get(pointer)
            float16_masks[pointer] = mask.clone() if previous is None else previous | mask
        if masks:
            contributions.append((key, masks, float16))
        ndim = effective_ndims.get(key, tensor.ndim)
        if (
            from_parameters and not masks and float16 == total
            and total >= storage.min_quantized_values
            and _is_weight_matrix(tensor, min(ndim, tensor.ndim))
        ):
            names = tuple(sorted(parameter_names[pointer] for pointer in from_parameters))
            entry = derived.setdefault((names, sources), [0, tuple(tensor.shape)])
            entry[0] += total

    integer_masks: dict[int, torch.Tensor] = {}
    if float16_masks:
        for key, (float16, total, precision) in stored.items():
            if float16 == total:
                continue
            masks, _ = provenance.elements(recorder.graph_values(key, tensors[key]))
            for pointer, mask in masks.items():
                if pointer in float16_masks:
                    previous = integer_masks.get(pointer)
                    integer_masks[pointer] = mask.clone() if previous is None else previous | mask

    warnings_found: list[str] = []
    found: dict[int, _StoredFloat16] = {}
    if float16_masks:
        masks = _region_masks(region for region in regions if region.pointer in float16_masks)
        for region in regions:
            if region.entry not in masks:
                continue
            mask = masks[region.entry]
            float16 = float16_masks[region.pointer] & mask
            if not bool(float16.any()):
                continue
            integer = integer_masks.get(region.pointer)
            copies = int((float16 & integer).sum()) if integer is not None else 0
            only = int(float16.sum()) - copies
            # A list holds no more values than it has: a float16 shard of a
            # weight stores that many of the weight's values.
            capacity = 0.0
            whole_list = False
            for key, held, count in contributions:
                if region.pointer not in held:
                    continue
                overlap = int((held[region.pointer] & mask).sum())
                if overlap:
                    share = sum(int(value.sum()) for value in held.values())
                    capacity += count * overlap / share
                    whole_list = whole_list or stored[key][2] == "float16"
            if only + copies > capacity:
                scale = capacity / (only + copies)
                only, copies = round(only * scale), round(copies * scale)
            if not only + copies:
                continue
            reason = (
                "storage reads the values in groups along a different axis than they were "
                "quantized in" if whole_list else
                f"a shard of fewer than min_quantized_values={storage.min_quantized_values} "
                "values is stored as float16"
            )
            found[region.entry] = _StoredFloat16(only=only, copies=copies, reason=reason)
            item = report.tensors[region.entry]
            if only >= item.values:
                warnings_found.append(
                    f"quantized weight {item.name!r} is stored as float16, not "
                    f"{storage.precision}: {reason}"
                )
            else:
                warnings_found.append(
                    f"quantized weight {item.name!r}: {only + copies} of its values are "
                    f"stored as float16, not {storage.precision}"
                    + (" (a second copy)" if copies else "") + f": {reason}"
                )
    derived_items = []
    for (names, _), (values, shape) in derived.items():
        description = f"weight computed from {', '.join(names)}"
        derived_items.append(TensorQuantizationReport(
            name=description,
            method="float16",
            bits=None,
            values=values,
            calibration_rows=0,
            effective_group_size=None,
            fallback_reason=(
                "computed from parameters during export; quantization only rounds "
                "parameters the model reads directly"
            ),
        ))
        warnings_found.append(
            f"{description} {shape} is stored as float16, not {storage.precision}: "
            "export computes it from parameters, and quantization only rounds "
            "parameters the model reads directly"
        )
    return tuple(warnings_found), _stored_quantization_report(report, found, derived_items)


__all__ = [
    "QuantizationConfig", "QuantizationReport", "TensorQuantizationReport",
]
