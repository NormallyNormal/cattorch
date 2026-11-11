"""Offline weight quantization used by Scratch static storage."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Literal, cast

import numpy as np
import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode

from cattorch.storage import _quantize_arrays
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


def _report_float16_storage(
    report: QuantizationReport, names: Iterable[str],
) -> QuantizationReport:
    """``report`` with the named parameters reported as stored in float16.

    Storage keeps a quantized weight in float16 when it reads it grouped
    along another axis than quantization grouped it. The report then says
    what is stored, and why, rather than the rounding applied to it.
    """
    names = frozenset(names)
    if not names:
        return report
    tensors = tuple(
        replace(
            item,
            method="float16",
            bits=None,
            effective_group_size=None,
            fallback_reason=(
                f"{item.method} int{item.bits} values are stored as float16: storage "
                "reads them in groups along a different axis than they were quantized in"
            ),
        )
        if item.method != "float16" and item.name.split("[", 1)[0] in names else item
        for item in report.tensors
    )
    gptq_values = sum(item.values for item in tensors if item.method == "gptq")
    symmetric_values = sum(item.values for item in tensors if item.method == "symmetric")
    return replace(
        report,
        quantized_values=gptq_values + symmetric_values,
        gptq_values=gptq_values,
        symmetric_values=symmetric_values,
        tensors=tensors,
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


def _storage_interval(value: torch.Tensor) -> tuple[int, int]:
    start = int(value.storage_offset())
    end = start + sum(
        (int(size) - 1) * int(stride)
        for size, stride in zip(value.shape, value.stride())
    )
    return start, end + 1


class _GPTQCaptureMode(TorchDispatchMode):
    """Collect linear input covariances without depending on module structure."""

    def __init__(
        self,
        parameters: dict[int, str],
        registry: OperatorRegistry,
        *,
        collect_hessian: bool = True,
        bases: dict[int, torch.Tensor] | None = None,
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
            views = [
                (value, self._parameter_view(weight), orientation)
                for value, weight, orientation in observations
                if isinstance(value, torch.Tensor) and isinstance(weight, torch.Tensor)
            ]
        for value, view, orientation in views:
            if view is not None:
                self._record(value, view, orientation)
        return result


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
    group_scales: dict[int, torch.Tensor] = {}
    for block_start in range(0, width, config.block_size):
        block_end = min(width, block_start + config.block_size)
        for column in range(block_start, block_end):
            group_start = column - column % group
            group_end = min(width, group_start + group)
            scales = group_scales.get(group_start)
            if scales is None:
                maximum = work[:, group_start:group_end].abs().amax(dim=1)
                scales = maximum / qmax
                scales = torch.where(maximum > 0, scales, torch.ones_like(scales))
                # Match the exact scale values which Scratch will decode.
                scale_dtype = (
                    torch.float16
                    if config.scale_precision == "float16" else torch.float32
                )
                scales = scales.to(scale_dtype).to(torch.float64)
                if bool(torch.any((maximum > 0) & (scales == 0))):
                    minimum = math.ldexp(
                        1.0,
                        -24 if config.scale_precision == "float16" else -149,
                    )
                    scales = torch.where(
                        (maximum > 0) & (scales == 0), minimum, scales,
                    )
                if not bool(torch.isfinite(scales).all()):
                    raise ValueError(
                        "GPTQ scale is outside the selected scale precision"
                    )
                group_scales[group_start] = scales
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


def quantize_model(
    model: torch.nn.Module,
    config: QuantizationConfig,
    calibration_inputs: Iterable | None,
    *,
    orientation_inputs: Iterable | None = None,
    call_kwargs: dict | None = None,
    _registry: OperatorRegistry | None = None,
    _calibration_name: str = "calibration_inputs",
) -> QuantizationReport:
    """Quantize eligible weights in-place and return a structured report.

    This is an internal export step. ``transpile`` deep-copies the model before
    calling it, so caller-owned parameters are never modified.

    Symmetric quantization runs the model once on ``orientation_inputs`` to see
    whether each matrix is used as a linear weight or on the right of a
    matmul. Both are then grouped over the matrix's input width, the layout
    Scratch storage uses. Without it, matrices keep their stored orientation.
    ``call_kwargs`` are passed to every observed model call.
    ``_calibration_name`` is the caller's name for ``calibration_inputs``,
    for its errors.
    """
    effective_ndims = expert_bank_effective_ndims(model)

    def effective_ndim(value: torch.Tensor) -> int:
        return effective_ndims.get(value.untyped_storage().data_ptr(), value.ndim)

    parameters_by_storage: dict[int, str] = {}
    base_parameters: dict[int, tuple[str, torch.Tensor]] = {}
    for name, parameter in model.named_parameters():
        pointer = parameter.untyped_storage().data_ptr()
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
                model(*_normalize_batch(raw_batch), **(call_kwargs or {}))
                batches += 1
        if config.method == "gptq" and batches == 0:
            raise ValueError(f"GPTQ requires nonempty {_calibration_name}")
        if config.method == "symmetric":
            batches = 0

    reports: list[TensorQuantizationReport] = []
    handled: set[tuple] = set()
    fully_handled_storage: set[int] = set()
    covered: dict[int, int] = {}
    gptq = config.method == "gptq"
    total_rows = sum(stat.rows for stat in capture.stats.values()) if gptq else 0
    with torch.no_grad():
        views_by_storage: dict[int, list[_ActivationStats]] = {}
        for stat in capture.stats.values():
            views_by_storage.setdefault(
                stat.weight.untyped_storage().data_ptr(), [],
            ).append(stat)
        incompatible_storage = set()
        for pointer, stats in views_by_storage.items():
            for index, left in enumerate(stats):
                left_start, left_end = _storage_interval(left.weight)
                for right in stats[index + 1:]:
                    right_start, right_end = _storage_interval(right.weight)
                    if max(left_start, right_start) < min(left_end, right_end):
                        incompatible_storage.add(pointer)
                        break
                if pointer in incompatible_storage:
                    break

        for pointer in incompatible_storage:
            name, parameter = base_parameters[pointer]
            if parameter.numel() < config.min_quantized_values:
                continue
            matrices = (
                (parameter,) if parameter.ndim == 2 else parameter.reshape(
                    -1, parameter.shape[-2], parameter.shape[-1],
                )
            )
            values = 0
            groups = []
            for matrix in matrices:
                reconstructed, group = _symmetric_matrix(matrix, config)
                matrix.copy_(reconstructed)
                values += matrix.numel()
                groups.append(group)
            reports.append(TensorQuantizationReport(
                name=name,
                method="symmetric",
                bits=config.bits,
                values=values,
                calibration_rows=sum(
                    stat.rows for stat in views_by_storage[pointer]
                ) if gptq else 0,
                effective_group_size=(
                    groups[0] if len(set(groups)) == 1 else None
                ),
                fallback_reason=(
                    "parameter is used through overlapping matrix views"
                    if gptq else None
                ),
            ))
            fully_handled_storage.add(pointer)

        for stat in capture.stats.values():
            pointer = stat.weight.untyped_storage().data_ptr()
            if pointer in incompatible_storage:
                continue
            key = _storage_key(stat.weight)
            base_parameter = base_parameters.get(pointer)
            eligible_values = (
                stat.weight.numel()
                if base_parameter is None else base_parameter[1].numel()
            )
            if key in handled or eligible_values < config.min_quantized_values:
                continue
            handled.add(key)
            if base_parameter is not None and stat.weight.numel() == base_parameter[1].numel():
                # A full view such as a tied ``weight.T`` covers the parameter.
                fully_handled_storage.add(pointer)
            elif base_parameter is not None and stat.orientation in {"conv", "conv_transpose"}:
                # So do a grouped convolution's groups together. Views of one
                # parameter that overlap were excluded above.
                covered[pointer] = covered.get(pointer, 0) + stat.weight.numel()
                if covered[pointer] == base_parameter[1].numel():
                    fully_handled_storage.add(pointer)
            logical = _logical_matrix(stat.weight, stat.orientation)
            group_width = _storage_group_width(stat.weight, stat.orientation)
            method: Literal["gptq", "symmetric"] = "gptq"
            fallback_reason = None
            if not gptq:
                method = "symmetric"
                reconstructed, group = _symmetric_matrix(logical, config, group_width)
            elif stat.rows < config.min_calibration_rows:
                method = "symmetric"
                fallback_reason = (
                    f"only {stat.rows} calibration rows; "
                    f"requires {config.min_calibration_rows}"
                )
                reconstructed, group = _symmetric_matrix(logical, config, group_width)
            else:
                try:
                    reconstructed, group = _gptq_matrix(
                        logical, stat.hessian, config, group_width,
                    )
                    # Canonicalize through the physical symmetric codec. This
                    # makes the reconstructed tensor an exact fixed point of
                    # the existing Scratch storage encoder, including its one
                    # deliberately-unused signed code.
                    reconstructed, group = _symmetric_matrix(
                        reconstructed, config, group_width,
                    )
                except ValueError as error:
                    method = "symmetric"
                    fallback_reason = str(error)
                    reconstructed, group = _symmetric_matrix(logical, config, group_width)
            stat.weight.copy_(_physical_weight(reconstructed, stat.weight, stat.orientation))
            reports.append(TensorQuantizationReport(
                name=stat.name,
                method=method,
                bits=config.bits,
                values=stat.weight.numel(),
                calibration_rows=stat.rows if gptq else 0,
                effective_group_size=group,
                fallback_reason=fallback_reason,
            ))

        # Large floating matrices not observed as linear weights use the
        # documented symmetric fallback. This includes untied embeddings.
        for name, parameter in model.named_parameters():
            if effective_ndim(parameter) < 2 or parameter.numel() < config.min_quantized_values:
                continue
            if parameter.untyped_storage().data_ptr() in fully_handled_storage:
                continue
            base_key = _storage_key(parameter)
            if parameter.ndim == 2:
                if base_key in handled:
                    continue
                reconstructed, group = _symmetric_matrix(parameter, config)
                parameter.copy_(reconstructed)
                values = parameter.numel()
                handled.add(base_key)
                reports.append(TensorQuantizationReport(
                    name=name,
                    method="symmetric",
                    bits=config.bits,
                    values=values,
                    calibration_rows=0,
                    effective_group_size=group,
                    fallback_reason=(
                        "no compatible linear activation statistics"
                        if config.method == "gptq" else None
                    ),
                ))
                continue

            for matrix in parameter.reshape(
                -1, parameter.shape[-2], parameter.shape[-1],
            ):
                matrix_key = _storage_key(matrix)
                if matrix_key in handled:
                    continue
                reconstructed, group = _symmetric_matrix(matrix, config)
                matrix.copy_(reconstructed)
                handled.add(matrix_key)
                reports.append(TensorQuantizationReport(
                    name=(
                        f"{name}[offset={matrix.storage_offset()},"
                        f"shape={tuple(matrix.shape)}]"
                    ),
                    method="symmetric",
                    bits=config.bits,
                    values=matrix.numel(),
                    calibration_rows=0,
                    effective_group_size=group,
                    fallback_reason=(
                        "no compatible linear activation statistics"
                        if config.method == "gptq" else None
                    ),
                ))

        reported_names = {item.name.split("[", 1)[0] for item in reports}
        for name, parameter in model.named_parameters():
            if (
                effective_ndim(parameter) >= 2
                and parameter.numel() < config.min_quantized_values
                and name not in reported_names
            ):
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
    )


__all__ = [
    "QuantizationConfig", "QuantizationReport", "TensorQuantizationReport",
]
