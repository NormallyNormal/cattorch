"""How a sprite stores its static tensors: weights, buffers and constants.

Each static tensor is kept at one precision (``_static_storage_precision``):
float32, float16, or an integer encoding that reproduces the quantized
weight exactly. Bit-identical tensors and a tied embedding's shards are
shared first; the tensors are then cut into physical shards that each
decode within Scratch's list limit, and compressed storage replaces their
contents with payloads that ``cattorch init`` decodes
(``_apply_static_storage``). Grouped kernels keep some matrices in a
four-row interleaved order, which storage undoes before encoding and redoes
after decoding. The encodings themselves are in ``cattorch.storage``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import weakref
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import torch

from cattorch.procedures import (
    _add_warp_procedure, _insert_init_call, _merge_lists_by_name, _merge_variables_by_name,
)
from cattorch.quantization import (
    STORED_AS_VECTOR, STORED_BELOW_MINIMUM, STORED_OFF_GRID, effective_group_size,
)
from cattorch.storage import (
    EncodedList,
    StorageConfig,
    build_decoder_programs,
    build_shared_unpack_program,
    group_scales,
    rounded_values,
    _float_bytes,
    _quantize_arrays,
    _quantized_bytes,
)
from cattorch.util.scratch.dsl import (
    Program, add, append, change_var, clear, for_each, item, length, repeat, set_var, var,
)
from cattorch.util.scratch.remap import remap_ids
from cattorch.util.scratch.sharding import SCRATCH_LIST_LIMIT, ListLayout
from cattorch.util.scratch.tensor_adder import StaticValues, static_list_contents

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredList:
    """How storage stored one logical static list, recorded as it stored it.

    ``precision`` and ``group_size`` (of an integer encoding) are what its
    payloads hold; ``values`` and ``shape`` are the list's. A list storage
    shares with an identical one names that one as ``shared_with``, and is
    stored as it is.
    """

    precision: str
    group_size: int | None
    values: int
    shape: tuple[int, ...]
    # Why integer storage kept it in float (see ``_static_storage_decision``).
    reason: str | None = None
    shared_with: str | None = None


def _deduplicate_static_tensors(
    sprite: dict,
    static_tensors: dict[str, torch.Tensor | None],
    *,
    effective_ndims: dict[str, int] | None = None,
    shared: dict[str, str] | None = None,
) -> dict[str, torch.Tensor | None]:
    """Share bit-identical flattened tensors before storage and sharding.

    Tensor shape is intentionally not part of the identity: Scratch kernels
    access the physical lists as flat arrays and have already compiled their
    shape-specific index arithmetic. Hash matches are confirmed byte-for-byte
    so this transform remains exact even for NaNs and signed zeroes.
    ``shared`` receives each dropped tensor's key with the key of the
    tensor that stands in for it.
    """
    winners: dict[tuple[str, int, int, bytes], tuple[str, torch.Tensor]] = {}
    aliases: dict[str, str] = {}
    kept: dict[str, torch.Tensor | None] = {}
    # A tensor that no block reads, such as one a fast-mode kernel replaced,
    # can't stand in for the others: its list isn't in the sprite.
    in_sprite = {entry[0] for entry in sprite.get("lists", {}).values()}
    for key, tensor in static_tensors.items():
        if not isinstance(tensor, torch.Tensor) or f"W_{key}" not in in_sprite:
            kept[key] = tensor
            continue
        contiguous = tensor.detach().cpu().contiguous()
        # The bytes in place, not a copy of every weight held until the end.
        raw = contiguous.reshape(-1).view(torch.uint8)
        identity = (
            str(contiguous.dtype), contiguous.numel(),
            (effective_ndims or {}).get(key, tensor.ndim),
            hashlib.sha256(raw.numpy()).digest(),
        )
        previous = winners.get(identity)
        if previous is not None and torch.equal(previous[1], raw):
            aliases[f"W_{key}"] = f"W_{previous[0]}"
            if shared is not None:
                shared[key] = previous[0]
            log.info("Sharing identical static tensor W_%s -> W_%s", key, previous[0])
        else:
            winners[identity] = (key, raw)
            kept[key] = tensor
    if not aliases:
        return kept

    lists = sprite.get("lists", {})
    ids_by_name: dict[str, list[str]] = {}
    for identifier, entry in lists.items():
        ids_by_name.setdefault(entry[0], []).append(identifier)
    id_remap: dict[str, str] = {}
    name_by_winner_id: dict[str, str] = {}
    for duplicate_name, winner_name in aliases.items():
        winner_ids = ids_by_name.get(winner_name, [])
        if not winner_ids:
            continue
        winner_id = winner_ids[0]
        name_by_winner_id[winner_id] = winner_name
        for duplicate_id in ids_by_name.get(duplicate_name, []):
            id_remap[duplicate_id] = winner_id
            lists.pop(duplicate_id, None)
    if id_remap:
        sprite["blocks"] = remap_ids(sprite["blocks"], id_remap, share_unchanged=True)

        # Scratch fields redundantly store a display name next to the ID.
        stack = [sprite["blocks"]]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, list):
                if (
                    len(value) == 2
                    and isinstance(value[1], str)
                    and value[1] in name_by_winner_id
                ):
                    value[0] = name_by_winner_id[value[1]]
                stack.extend(value)
    return kept


def _apply_tied_static_aliases(
    sprite: dict,
    layouts: dict[str, ListLayout],
    static_tensors: dict[str, torch.Tensor | None],
    aliases: dict[str, tuple[str, ...]],
) -> dict[str, str]:
    """Point aligned embedding shards at the output head's row shards.

    Returns the tables that keep their own copy, with the reason.
    """
    lists = sprite.get("lists", {})
    remap: dict[str, str] = {}
    display_names: dict[str, str] = {}
    copies: dict[str, str] = {}
    for source_name, destination_names in aliases.items():
        source = layouts.get(source_name)
        if source is None:
            continue  # only the head reads the table
        destinations = [layouts.get(name) for name in destination_names]
        if (
            any(layout is None or len(layout.shards) != 1 for layout in destinations)
            or [shard.capacity for shard in source.shards]
            != [layout.shards[0].capacity for layout in destinations if layout is not None]
        ):
            # The compiler cuts the head's shards to the table's physical
            # lists, so this is not expected. Both copies remain correct.
            copies[source_name] = "its shards do not line up with the output head's"
            log.warning("Tied embedding/head sharing skipped: %s", copies[source_name])
            continue
        for source_shard, destination_layout in zip(source.shards, destinations):
            assert destination_layout is not None
            destination_shard = destination_layout.shards[0]
            if lists[source_shard.identifier][1] != lists[destination_shard.identifier][1]:
                raise ValueError("tied embedding/head shard contents do not match")
            remap[source_shard.identifier] = destination_shard.identifier
            display_names[destination_shard.identifier] = destination_shard.name
            del lists[source_shard.identifier]
        del layouts[source_name]
        static_tensors.pop(source_name.removeprefix("W_"), None)

    if not remap:
        return copies
    sprite["blocks"] = remap_ids(sprite["blocks"], remap, share_unchanged=True)
    stack = [sprite["blocks"]]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            if (
                len(value) == 2
                and isinstance(value[1], str)
                and value[1] in display_names
            ):
                value[0] = display_names[value[1]]
            stack.extend(value)
    return copies


_StoragePrecision = Literal["float32", "float16", "int8", "int6", "int4"]


def _static_storage_precision(
    tensor: torch.Tensor | None,
    storage: StorageConfig,
    effective_ndim: int | None = None,
    interleave_segments: tuple[tuple[int, int, int], ...] = (),
    source_numel: int | None = None,
) -> _StoragePrecision:
    """How storage keeps ``tensor``: decided once for the whole tensor.

    A row shard the compiler cut from a weight gives that weight's size as
    ``source_numel``, so it is kept like the rest of the weight.
    """
    return _static_storage_decision(
        tensor, storage, effective_ndim, interleave_segments, source_numel,
    )[0]


def _static_storage_decision(
    tensor: torch.Tensor | None,
    storage: StorageConfig,
    effective_ndim: int | None = None,
    interleave_segments: tuple[tuple[int, int, int], ...] = (),
    source_numel: int | None = None,
) -> tuple[_StoragePrecision, str | None]:
    """``_static_storage_precision``, and why integer storage keeps a
    floating tensor in float: ``STORED_AS_VECTOR``, ``STORED_BELOW_MINIMUM``
    or ``STORED_OFF_GRID``, when no integer encoding reproduces it."""
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        return "float32", None
    if storage.precision not in {"int8", "int6", "int4"}:
        return storage.precision, None
    ndim = tensor.ndim if effective_ndim is None else effective_ndim
    if ndim < 2:
        return "float16", STORED_AS_VECTOR
    # Quantization leaves matrices below the threshold unquantized.
    reason = STORED_BELOW_MINIMUM
    if (source_numel or tensor.numel()) >= storage.min_quantized_values:
        encoding = _integer_codec_encoding(tensor, storage, interleave_segments)
        if encoding is not None:
            return cast(_StoragePrecision, encoding[0]), None
        reason = STORED_OFF_GRID
    # QuantizationConfig decides which weights are lossy. Storage must not
    # quantize anything it did not reconstruct, such as buffers (RoPE
    # tables, masks) or values derived from weights by other arithmetic.
    finite = tensor.detach()[torch.isfinite(tensor.detach())]
    if finite.numel() and float(finite.abs().max()) > 65504:
        return "float32", reason
    return "float16", reason


def _integer_encodings(
    tensor: torch.Tensor, storage: StorageConfig,
) -> tuple[tuple[str, int], ...]:
    """The integer encodings storage tries for ``tensor``, smallest first.

    Each is a precision quantization may use (``storage.integer_formats``)
    with groups of consecutive values within one row of the tensor's
    trailing dimensions: a convolution's kernels are grouped over their
    whole fan-in, a transposed convolution's within each kernel, and a
    matrix's along its rows. Quantization chose one of them; only that one
    reproduces the values, unless several do exactly, as for zeros.
    """
    formats = storage.integer_formats or ((storage.precision, storage.group_size),)
    shape = [int(size) for size in tensor.shape] or [1]
    extents = {math.prod(shape[start:]) for start in range(1, len(shape))} or {shape[-1]}
    scale_bits = 32 if storage.scale_precision == "float32" else 16
    candidates = {
        (precision, effective_group_size(extent, maximum))
        for precision, maximum in formats for extent in extents if extent >= 1
    }
    return tuple(sorted(
        candidates,
        key=lambda item: (_INTEGER_BITS[item[0]] + scale_bits / item[1], item[0], -item[1]),
    ))


_INTEGER_BITS = {"int8": 8, "int6": 6, "int4": 4}


def _integer_codec_encoding(
    tensor: torch.Tensor,
    storage: StorageConfig,
    interleave_segments: tuple[tuple[int, int, int], ...] = (),
) -> tuple[str, int] | None:
    """The smallest integer encoding that reproduces ``tensor`` exactly.

    Grouped kernels keep static tensors in a four-row physical order. Storage
    quantizes the logical order, so undo that permutation first.
    """
    # Compilation, sharding and storage each ask about the same tensors.
    key = (
        storage.precision, storage.integer_formats, storage.group_size,
        storage.scale_precision, tuple(interleave_segments), tensor._version,
    )
    cached = _CODEC_GRID_CACHE.get(id(tensor))
    if cached is not None and cached[0]() is tensor and cached[1] == key:
        return cached[2]
    result = next(
        (
            (precision, group_size)
            for precision, group_size in _integer_encodings(tensor, storage)
            if _check_integer_codec_grid(
                tensor, storage, group_size, interleave_segments, precision=precision,
            )
        ),
        None,
    )
    if cached is None or cached[0]() is not tensor:
        weakref.finalize(tensor, _CODEC_GRID_CACHE.pop, id(tensor), None)
    _CODEC_GRID_CACHE[id(tensor)] = (weakref.ref(tensor), key, result)
    return result


def _on_integer_codec_grid(
    tensor: torch.Tensor,
    storage: StorageConfig,
    interleave_segments: tuple[tuple[int, int, int], ...] = (),
) -> bool:
    """Return whether some integer encoding reproduces ``tensor`` exactly."""
    return _integer_codec_encoding(tensor, storage, interleave_segments) is not None


# id(tensor) -> (weak reference, the question's key, answer).
_CODEC_GRID_CACHE: dict[int, tuple[weakref.ref, tuple, tuple[str, int] | None]] = {}
# Groups checked per step, to bound the float64 temporaries for large tables.
_CODEC_GRID_CHUNK_GROUPS = 1 << 13


def _check_integer_codec_grid(
    tensor: torch.Tensor,
    storage: StorageConfig,
    group_size: int,
    interleave_segments: tuple[tuple[int, int, int], ...],
    *,
    precision: str | None = None,
) -> bool:
    # NumPy rather than torch: these are many small passes, and torch's
    # thread pool makes each one much slower on a busy machine.
    flat = tensor.detach().cpu().reshape(-1)
    native = flat.dtype in (torch.float16, torch.float32, torch.float64)
    # A format NumPy lacks, such as bfloat16, widens exactly to float64 and
    # rounds back through torch.
    values = flat.numpy(force=True) if native else flat.to(torch.float64).numpy()
    for start, rows, inner in interleave_segments:
        if rows % 4 or start < 0 or start + rows * inner > values.size:
            return False
    if values.size == 0 or values.size % group_size:
        return False
    maximum_code = {"int8": 127, "int6": 31, "int4": 7}[precision or storage.precision]
    # Every group is checked on its own, so whole groups at a time, in
    # logical order, give the same answer as all of them at once.
    for groups in _logical_group_chunks(values, interleave_segments, group_size):
        work = groups.astype(np.float64)
        if not np.isfinite(work).all():
            return False
        # The encoder's own scale rule, so a tensor passes exactly when the
        # encoder reproduces it.
        scale = group_scales(
            np.abs(work).max(axis=1), maximum_code, storage.scale_precision,
        )[:, None]
        if not np.isfinite(scale).all():
            return False
        reconstructed = np.clip(np.round(work / scale), -maximum_code, maximum_code) * scale
        if native:
            rounded = reconstructed.astype(values.dtype)
        else:
            rounded = torch.from_numpy(reconstructed).to(flat.dtype).double().numpy()
        if not np.array_equal(rounded, groups):
            return False
    return True


def _logical_group_chunks(
    values: np.ndarray,
    segments: tuple[tuple[int, int, int], ...],
    group_size: int,
):
    """Yield ``values`` in logical order as ``(groups, group_size)`` blocks.

    Grouped kernels store four-row blocks column-interleaved; each segment
    is undone on its own as it is reached, instead of through one gather of
    the whole tensor.
    """
    chunk_values = _CODEC_GRID_CHUNK_GROUPS * group_size
    carry = values[:0]
    for piece in _logical_pieces(values, segments, chunk_values):
        if carry.size:
            piece = np.concatenate((carry, piece))
        usable = piece.size - piece.size % group_size
        if usable:
            yield piece[:usable].reshape(-1, group_size)
        carry = piece[usable:]


def _logical_pieces(
    values: np.ndarray,
    segments: tuple[tuple[int, int, int], ...],
    chunk_values: int,
):
    """Yield ``values`` in logical order, about ``chunk_values`` at a time."""
    ordered = sorted(segments)
    if any(
        ordered[index][0] + ordered[index][1] * ordered[index][2] > ordered[index + 1][0]
        for index in range(len(ordered) - 1)
    ):
        # Overlapping segments compose; gather through every permutation.
        order = np.arange(values.size)
        for start, rows, inner in segments:
            end = start + rows * inner
            order[start:end] = order[start:end].reshape(rows // 4, inner, 4).transpose(0, 2, 1).reshape(-1)
        values = values[order]
        ordered = []
    position = 0
    for start, rows, inner in [*ordered, (values.size, 0, 1)]:
        for offset in range(position, start, chunk_values):
            yield values[offset:min(start, offset + chunk_values)]
        blocks = values[start:start + rows * inner].reshape(rows // 4, inner, 4)
        step = max(1, chunk_values // (4 * inner))
        for block in range(0, rows // 4, step):
            yield blocks[block:block + step].transpose(0, 2, 1).reshape(-1)
        position = start + rows * inner


def _static_storage_group_size(
    tensor: torch.Tensor | None,
    storage: StorageConfig,
    interleave_segments: tuple[tuple[int, int, int], ...] = (),
) -> int:
    """The quantization group width storage encodes ``tensor`` with.

    It is the group of the integer encoding that reproduces the tensor (see
    ``_integer_encodings``); otherwise a matrix's groups run along its rows,
    never crossing one.
    """
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2 or tensor.shape[-1] < 1:
        return storage.group_size
    if storage.precision in {"int8", "int6", "int4"} and tensor.is_floating_point():
        encoding = _integer_codec_encoding(tensor, storage, interleave_segments)
        if encoding is not None:
            return encoding[1]
    return effective_group_size(int(tensor.shape[-1]), storage.group_size)


def _static_storage_shard_limits(
    static_tensors: dict[str, torch.Tensor | None],
    storage: StorageConfig,
    *,
    effective_ndims: dict[str, int] | None = None,
    interleaved_layouts: dict[str, tuple[tuple[int, int, int], ...]] | None = None,
    source_numels: dict[str, int] | None = None,
) -> dict[str, int]:
    """Cap static shards so every temporary decode stream fits in Scratch."""
    if not storage.compression:
        return {}
    limits: dict[str, int] = {}
    for key, tensor in static_tensors.items():
        precision = _static_storage_precision(
            tensor, storage, (effective_ndims or {}).get(key),
            (interleaved_layouts or {}).get(key, ()),
            (source_numels or {}).get(key),
        )
        group_size = _static_storage_group_size(
            tensor, storage, (interleaved_layouts or {}).get(key, ()),
        )
        if precision == "float32":
            limit = SCRATCH_LIST_LIMIT // 4
        elif precision == "float16":
            limit = SCRATCH_LIST_LIMIT // 2
        else:
            scale_bytes = 4 if storage.scale_precision == "float32" else 2
            scale_groups = SCRATCH_LIST_LIMIT // scale_bytes
            limit = min(SCRATCH_LIST_LIMIT, scale_groups * group_size)
            # Each physical payload is quantized independently.  Ending a
            # shard mid-group would restart scale selection in the next shard
            # and change weights compared with the logical matrix (notably
            # shared MoE banks whose effective row group is 48 values).
            limit -= limit % group_size
            if limit < group_size:
                raise ValueError("quantized static shard cannot fit one group")
        limits[f"W_{key}"] = limit
    return limits


def _validate_nonfloating_static_tensors(
    static_tensors: dict[str, torch.Tensor | None],
    *,
    compressed: bool,
) -> None:
    """Reject integer constants which Scratch's selected number path changes."""
    scratch_dtype = torch.float32 if compressed else torch.float64
    label = "float32 compressed storage" if compressed else "Scratch numbers"
    for key, tensor in static_tensors.items():
        if not isinstance(tensor, torch.Tensor) or tensor.is_floating_point():
            continue
        if tensor.is_complex():
            raise ValueError(f"static tensor {key!r} has unsupported complex values")
        original = tensor.detach().cpu()
        try:
            round_trip = original.to(scratch_dtype).to(original.dtype)
        except (RuntimeError, TypeError) as error:
            raise ValueError(
                f"static tensor {key!r} cannot be represented by {label}"
            ) from error
        if not torch.equal(original, round_trip):
            suggestion = (
                "; use compression=False for integers exactly representable "
                "by Scratch doubles"
                if compressed else ""
            )
            raise ValueError(
                f"static tensor {key!r} contains values not exactly "
                f"representable by {label}{suggestion}"
            )


def _deinterleave_static_values(
    values: list,
    segments: tuple[tuple[int, int, int], ...],
) -> list:
    """Convert four-row physical groups back to logical output-major rows."""
    if isinstance(values, StaticValues):
        array = values.array.copy()
        for start, rows, inner in segments:
            if rows % 4 or start < 0 or start + rows * inner > len(values):
                raise ValueError("malformed grouped static-weight layout")
            end = start + rows * inner
            array[start:end] = (
                values.array[start:end].reshape(rows // 4, inner, 4).transpose(0, 2, 1).reshape(-1)
            )
        return StaticValues(array)
    result = list(values)
    for start, rows, inner in segments:
        if rows % 4 or start < 0 or start + rows * inner > len(values):
            raise ValueError("malformed grouped static-weight layout")
        source = values[start:start + rows * inner]
        logical: list = []
        for group_start in range(0, len(source), 4 * inner):
            group = source[group_start:group_start + 4 * inner]
            for lane in range(4):
                logical.extend(group[column * 4 + lane] for column in range(inner))
        result[start:start + rows * inner] = logical
    return result


def _interleave_static_values(
    values: list,
    segments: tuple[tuple[int, int, int], ...],
) -> list:
    """Apply the grouped kernel's four-row physical storage permutation."""
    result = list(values)
    for start, rows, inner in segments:
        source = values[start:start + rows * inner]
        physical: list = []
        for group_start in range(0, len(source), 4 * inner):
            group = source[group_start:group_start + 4 * inner]
            for column in range(inner):
                physical.extend(
                    group[lane * inner + column] for lane in range(4)
                )
        result[start:start + rows * inner] = physical
    return result


def _physical_interleave_segments(
    layout: ListLayout,
    segments: tuple[tuple[int, int, int], ...],
) -> dict[str, tuple[tuple[int, int, int], ...]]:
    """Map logical-list segment offsets onto its physical Scratch shards."""
    result = {}
    shard_start = 0
    for shard in layout.shards:
        shard_end = shard_start + shard.capacity
        local = []
        for start, rows, inner in segments:
            end = start + rows * inner
            if end <= shard_start or start >= shard_end:
                continue
            overlap_start = max(start, shard_start)
            overlap_end = min(end, shard_end)
            group_values = 4 * inner
            if (
                (overlap_start - start) % group_values
                or (overlap_end - overlap_start) % group_values
            ):
                raise ValueError(
                    "grouped weight sharding must preserve complete four-row groups"
                )
            local.append((
                overlap_start - shard_start,
                (overlap_end - overlap_start) // inner,
                inner,
            ))
        if local:
            result[shard.name] = tuple(local)
        shard_start = shard_end
    return result


def _build_interleave_program(
    specs: list[tuple[str, tuple[tuple[int, int, int], ...]]],
):
    """Build one init-only row-major to four-row storage permutation."""
    temporary = "cattorch interleave buffer"
    body = []
    for name, segments in specs:
        body.append(clear(temporary))
        for start, rows, inner in segments:
            body.extend((
                set_var("interleave group start", start + 1),
                repeat(rows // 4, (
                    set_var("interleave column", 0),
                    repeat(inner, (
                        *tuple(
                            append(
                                temporary,
                                item(
                                    name,
                                    add(
                                        var("interleave group start"),
                                        add(lane * inner, var("interleave column")),
                                    ),
                                ),
                            )
                            for lane in range(4)
                        ),
                        change_var("interleave column", 1),
                    )),
                    change_var("interleave group start", 4 * inner),
                )),
            ))
        body.extend((
            clear(name),
            for_each(
                "interleave copy index",
                length(temporary),
                (append(
                    name,
                    item(temporary, var("interleave copy index")),
                ),),
            ),
        ))
    body.append(clear(temporary))
    return Program(
        "interleave_grouped_weights",
        variables=(
            "interleave group start", "interleave column",
            "interleave copy index",
        ),
        lists=(temporary, *(name for name, _ in specs)),
        body=tuple(body),
    )


def _apply_static_storage(
    sprite: dict,
    layouts: dict[str, ListLayout],
    static_tensors: dict[str, torch.Tensor | None],
    storage: StorageConfig,
    *,
    interleaved_layouts: dict[
        str, tuple[tuple[int, int, int], ...]
    ] | None = None,
    effective_ndims: dict[str, int] | None = None,
    source_numels: dict[str, int] | None = None,
    record: dict[str, StoredList] | None = None,
) -> set[str]:
    """Encode static physical shards and install the one-time unpacker.

    Each floating tensor's precision is decided once, here, and ``record``
    receives that decision for every list stored, by key: what the
    quantization report says was stored is read from it.
    """
    source_numels = source_numels or {}
    static_names = {f"W_{key}" for key in static_tensors}
    interleaved_layouts = interleaved_layouts or {}
    physical_interleaves = {}
    for key, segments in interleaved_layouts.items():
        logical_name = f"W_{key}"
        layout = layouts.get(logical_name)
        if layout is not None:
            physical_interleaves.update(
                _physical_interleave_segments(layout, segments)
            )
    _validate_nonfloating_static_tensors(
        static_tensors, compressed=storage.compression,
    )
    decisions: dict[str, tuple[_StoragePrecision, int]] = {}
    for logical_name in sorted(static_names):
        key = logical_name.removeprefix("W_")
        tensor = static_tensors.get(key)
        if logical_name not in layouts or not isinstance(tensor, torch.Tensor):
            continue
        precision, reason = _static_storage_decision(
            tensor, storage, (effective_ndims or {}).get(key),
            interleaved_layouts.get(key, ()), source_numels.get(key),
        )
        group_size = _static_storage_group_size(tensor, storage, interleaved_layouts.get(key, ()))
        decisions[key] = (precision, group_size)
        if record is not None and tensor.is_floating_point():
            record[key] = StoredList(
                precision=precision,
                group_size=group_size if precision in ("int8", "int6", "int4") else None,
                values=tensor.numel(),
                shape=tuple(int(size) for size in tensor.shape),
                reason=reason,
            )
    if not storage.compression:
        if storage.precision in {"float16", "int8", "int6", "int4"}:
            for logical_name in static_names:
                tensor = static_tensors.get(logical_name.removeprefix("W_"))
                if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                    continue
                if logical_name.removeprefix("W_") not in decisions:
                    continue
                precision, group_size = decisions[logical_name.removeprefix("W_")]
                for shard in layouts[logical_name].shards:
                    contents = sprite["lists"][shard.identifier][1]
                    segments = physical_interleaves.get(shard.name, ())
                    logical_contents = (
                        _deinterleave_static_values(contents, segments)
                        if segments else contents
                    )
                    rounded = rounded_values(
                        logical_contents,
                        precision,
                        group_size,
                        storage.scale_precision,
                    )
                    sprite["lists"][shard.identifier][1] = (
                        _interleave_static_values(rounded, segments)
                        if segments else rounded
                    )
        _materialize_static_values(sprite)
        return static_names

    specs: list[EncodedList] = []
    runtime_interleaves = []
    for logical_name in sorted(static_names):
        layout = layouts.get(logical_name)
        if layout is None:
            continue
        precision, group_size = decisions.get(
            logical_name.removeprefix("W_"), ("float32", storage.group_size),
        )
        for position, shard in enumerate(layout.shards, 1):
            contents = sprite["lists"][shard.identifier][1]
            if not contents:
                continue
            payload_name = f"cattorch payload {len(specs) + 1}"
            segments = physical_interleaves.get(shard.name, ())
            encoded_contents = (
                _deinterleave_static_values(contents, segments)
                if segments else contents
            )
            if precision in ("int8", "int6", "int4"):
                codes, scales = _quantize_arrays(
                    encoded_contents,
                    precision,
                    group_size,
                    storage.scale_precision,
                )
                raw_payload = _quantized_bytes(codes, precision)
                scale_number = len(specs) + 1
                # With payload_bytes on every spec the unpacker encodes whole
                # banks itself, so the per-list Base92 payloads stay empty.
                specs.append(EncodedList(
                    shard.name,
                    payload_name,
                    "",
                    precision,
                    value_count=len(contents),
                    group_size=group_size,
                    scale_list_name=f"cattorch quant scales {scale_number}",
                    scale_payload_name=f"cattorch scale payload {scale_number}",
                    scale_precision=storage.scale_precision,
                    payload_bytes=raw_payload,
                    scale_values=tuple(scales.tolist()),
                ))
            else:
                raw_payload = _float_bytes(
                    encoded_contents, cast(Literal["float32", "float16"], precision),
                )
                specs.append(EncodedList(
                    shard.name,
                    payload_name,
                    "",
                    precision,
                    payload_bytes=raw_payload,
                ))
            if segments:
                runtime_interleaves.append((shard.name, segments))
            sprite["lists"][shard.identifier][1] = []
        # Its shards hold the payload now; let a derived tensor go.
        static_tensors[logical_name.removeprefix("W_")] = None

    if specs:
        for index, (decoder_name, program) in enumerate(
            build_decoder_programs(specs).items()
        ):
            _add_warp_procedure(
                sprite,
                decoder_name,
                program,
                x=960,
                y=480 + index * 120,
            )
        sprite = _add_warp_procedure(
            sprite,
            "cattorch unpack weights",
            build_shared_unpack_program(specs),
            x=640,
            y=480,
        )
        _merge_lists_by_name(
            sprite,
            {spec.name for spec in specs}
            | {spec.scale_list_name for spec in specs if spec.scale_list_name}
            | {
                "cattorch f32 powers", "cattorch f16 powers", "cattorch decoded bytes",
                "cattorch decoded values", "cattorch decoded scales",
                "cattorch huffman table", "cattorch huffman powers",
            },
        )
        _merge_variables_by_name(
            sprite,
            {
                entry[0]
                for entry in sprite.get("variables", {}).values()
                if entry[0].startswith("decode ")
                or entry[0] == "cattorch active payload"
            },
        )
        if runtime_interleaves:
            _add_warp_procedure(
                sprite,
                "cattorch interleave grouped weights",
                _build_interleave_program(runtime_interleaves),
                x=640,
                y=600,
            )
            _merge_lists_by_name(
                sprite,
                {name for name, _ in runtime_interleaves}
                | {"cattorch interleave buffer"},
            )
            _merge_variables_by_name(
                sprite,
                {
                    "interleave group start", "interleave column",
                    "interleave copy index",
                },
            )
            # Calls are prepended to init, so add the permutation first and
            # unpack second to execute them in unpack -> interleave order.
            _insert_init_call(sprite, "cattorch interleave grouped weights")
        _insert_init_call(sprite, "cattorch unpack weights")
    _materialize_static_values(sprite)
    return static_names


def _materialize_static_values(sprite: dict) -> None:
    """Turn the static values storage left in place into the lists JSON needs."""
    for entry in sprite.get("lists", {}).values():
        entry[1] = static_list_contents(entry[1])
