"""Static tensor storage configuration and Scratch-safe byte codecs."""

from __future__ import annotations

import math
import struct
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, cast

import torch

from cattorch.util.scratch.dsl import (
    Program,
    Statement,
    add,
    append,
    change_var,
    clear,
    call,
    costume_number,
    delete,
    div,
    eq,
    gt,
    if_,
    if_else,
    item,
    length,
    letter,
    mathop,
    lt,
    mod,
    mul,
    for_each,
    repeat,
    set_var,
    string_length,
    sub,
    switch_costume,
    var,
)
from cattorch.util.scratch.sharding import SCRATCH_LIST_LIMIT


# Costume-name lookup in vanilla Scratch is case-sensitive even though its
# ordinary string and list comparisons are not. All 92 printable non-space
# ASCII characters other than JSON's quote and backslash are one-byte and need
# no escaping. Thirteen bytes fit in sixteen Base92 digits when represented as
# two exact 52-bit integers; the costumes continue to share one physical SVG.
BASE92_ALPHABET = "".join(
    chr(code) for code in range(33, 127) if chr(code) not in {'"', "\\"}
)
assert len(BASE92_ALPHABET) == 92 and len(set(BASE92_ALPHABET)) == 92

# Retained as a source-level compatibility alias. New payloads and generated
# sprites use the complete Base92 alphabet.
BASE85_ALPHABET = BASE92_ALPHABET[:85]


@dataclass(frozen=True)
class StorageConfig:
    """Control serialized numeric tensor storage.

    Compressed storage is the default.  ``float16`` and integer precisions are
    explicitly lossy and are permitted with either exact or fast kernels.
    Integer formats use symmetric quantization over fixed-size flat groups and
    are dequantized once during ``cattorch init``.

    The integer ``precision`` values and their grouping fields are deprecated.
    Use ``QuantizationConfig`` instead, which uses row-aligned groups and
    float16 scales by default.
    """

    compression: bool = True
    precision: Literal["float32", "float16", "int8", "int6", "int4"] = "float32"
    group_size: int = 64
    min_quantized_values: int = 16384
    scale_precision: Literal["float32", "float16"] = "float32"
    grouping: Literal["flat", "row"] = "flat"

    def __post_init__(self):
        if not isinstance(self.compression, bool):
            raise TypeError("compression must be a bool")
        if self.precision not in {"float32", "float16", "int8", "int6", "int4"}:
            raise ValueError(
                "precision must be 'float32', 'float16', 'int8', 'int6', or 'int4'"
            )
        if self.scale_precision not in {"float32", "float16"}:
            raise ValueError("scale_precision must be 'float32' or 'float16'")
        if self.grouping not in {"flat", "row"}:
            raise ValueError("grouping must be 'flat' or 'row'")
        if (
            isinstance(self.group_size, bool)
            or not isinstance(self.group_size, int)
            or self.group_size < 1
        ):
            raise ValueError("group_size must be a positive integer")
        if (
            isinstance(self.min_quantized_values, bool)
            or not isinstance(self.min_quantized_values, int)
            or self.min_quantized_values < 1
        ):
            raise ValueError("min_quantized_values must be a positive integer")


def encode_values(values, precision: Literal["float32", "float16"]) -> str:
    """Encode numeric values with the Scratch costume-name Base92 codec."""
    code = ">f" if precision == "float32" else ">e"
    return _encode_bytes(_float_bytes(values, precision))


def _float_bytes(values, precision: Literal["float32", "float16"]) -> bytes:
    code = ">f" if precision == "float32" else ">e"
    return b"".join(_pack_value(code, value) for value in values)


def _encode_bytes(raw: bytes) -> str:
    """Encode bytes as one pad digit followed by 13-byte Base92 groups."""
    padding = (-len(raw)) % 13
    padded = raw + bytes(padding)
    encoded: list[str] = [BASE92_ALPHABET[padding]]
    mask = 2**52 - 1
    for offset in range(0, len(padded), 13):
        packed = int.from_bytes(padded[offset:offset + 13], "big")
        for half in (packed >> 52, packed & mask):
            digits = [0] * 8
            for index in range(7, -1, -1):
                half, digits[index] = divmod(half, 92)
            encoded.extend(BASE92_ALPHABET[digit] for digit in digits)
    return "".join(encoded)


def quantize_values(
    values,
    precision: Literal["int8", "int6", "int4"],
    group_size: int = 64,
    scale_precision: Literal["float32", "float16"] = "float32",
) -> tuple[list[int], list[float]]:
    """Return signed integer codes and float32 reconstruction scales."""
    if precision not in {"int8", "int6", "int4"}:
        raise ValueError("precision must be 'int8', 'int6', or 'int4'")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise ValueError("group_size must be a positive integer")
    if scale_precision not in {"float32", "float16"}:
        raise ValueError("scale_precision must be 'float32' or 'float16'")

    numbers = [float(value) for value in values]
    if any(not math.isfinite(number) for number in numbers):
        raise ValueError("integer weight quantization requires finite values")
    maximum_code = {"int8": 127, "int6": 31, "int4": 7}[precision]
    codes: list[int] = []
    scales: list[float] = []
    for start in range(0, len(numbers), group_size):
        group = numbers[start:start + group_size]
        maximum = max((abs(number) for number in group), default=0.0)
        scale = maximum / maximum_code if maximum else 1.0
        scale_code = ">f" if scale_precision == "float32" else ">e"
        scale_guidance = (
            "use scale_precision='float32'"
            if scale_precision == "float16"
            else "rescale the tensor before export"
        )
        try:
            scale = struct.unpack(scale_code, struct.pack(scale_code, scale))[0]
        except OverflowError as error:
            raise ValueError(
                f"quantization scale {scale!r} is outside the finite "
                f"{scale_precision} range; {scale_guidance}"
            ) from error
        if not math.isfinite(scale):
            raise ValueError(
                f"quantization scale is outside the finite {scale_precision} "
                f"range; {scale_guidance}"
            )
        if maximum and scale == 0:
            # Preserve a finite reconstruction for groups whose ideal scale
            # falls below the selected IEEE format's smallest subnormal.
            scale = math.ldexp(1.0, -149 if scale_precision == "float32" else -24)
        scales.append(scale)
        for number in group:
            code = round(number / scale) if maximum else 0
            codes.append(max(-maximum_code, min(maximum_code, code)))
    return codes, scales


def quantize_codebook_values(
    values, *, entries: int = 64, iterations: int = 32,
) -> tuple[list[int], list[float]]:
    """Deterministically fit a scalar k-means codebook to one logical tensor."""
    if entries != 64:
        raise ValueError("codebook storage currently requires exactly 64 entries")
    numbers = torch.as_tensor(list(values), dtype=torch.float64)
    if numbers.numel() == 0:
        return [], [0.0] * entries
    if not bool(torch.isfinite(numbers).all()):
        raise ValueError("codebook quantization requires finite values")

    sorted_values = numbers.sort().values
    positions = torch.linspace(
        0, sorted_values.numel() - 1, entries, dtype=torch.float64,
    ).round().to(torch.int64)
    centers = sorted_values[positions].clone()
    for _ in range(iterations):
        boundaries = (centers[:-1] + centers[1:]) / 2
        assignments = torch.bucketize(numbers, boundaries)
        sums = torch.zeros(entries, dtype=torch.float64)
        counts = torch.zeros(entries, dtype=torch.float64)
        sums.scatter_add_(0, assignments, numbers)
        counts.scatter_add_(0, assignments, torch.ones_like(numbers))
        updated = torch.where(counts > 0, sums / counts.clamp_min(1), centers)
        updated = updated.sort().values
        if torch.equal(updated, centers):
            break
        centers = updated
    boundaries = (centers[:-1] + centers[1:]) / 2
    assignments = torch.bucketize(numbers, boundaries)
    return assignments.tolist(), centers.tolist()


def rounded_codebook_values(values) -> list[float]:
    """Return values reconstructed by the deterministic 64-entry codebook."""
    codes, centers = quantize_codebook_values(values)
    return [centers[code] for code in codes]


def encode_quantized_values(
    values,
    precision: Literal["int8", "int6", "int4"],
    group_size: int = 64,
    scale_precision: Literal["float32", "float16"] = "float32",
) -> tuple[str, str]:
    """Encode centered quantized values and their group scales."""
    codes, scales = quantize_values(
        values, precision, group_size, scale_precision,
    )
    payload = _encode_bytes(_quantized_bytes(codes, precision))
    return payload, encode_values(scales, scale_precision)


def _quantized_bytes(
    codes: list[int], precision: Literal["int8", "int6", "int4"],
) -> bytes:
    """Pack centered integer codes before their common Base92 encoding."""
    if precision == "int8":
        # Code 255 is deliberately unused. Centering removes a sign branch
        # from the Scratch startup decoder.
        return bytes(code + 127 for code in codes)
    if precision == "int6":
        # Four centered six-bit codes occupy three bytes. Continuous Base92
        # packing approaches 12 characters per 13 bytes while
        # sharing the faster costume decoder with every format.
        packed: list[int] = []
        for index in range(0, len(codes), 4):
            group = [code + 31 for code in codes[index:index + 4]]
            group.extend([31] * (4 - len(group)))
            packed.extend((
                group[0] * 4 + group[1] // 16,
                (group[1] % 16) * 16 + group[2] // 4,
                (group[2] % 4) * 64 + group[3],
            ))
        raw_size = math.ceil(len(codes) * 6 / 8)
        return bytes(packed[:raw_size])
    if precision == "int4":
        packed = []
        for index in range(0, len(codes), 2):
            high = codes[index] + 7
            low = codes[index + 1] + 7 if index + 1 < len(codes) else 7
            packed.append(high * 16 + low)
        return bytes(packed)
    raise ValueError("precision must be 'int8', 'int6', or 'int4'")


def _int4_symbols(spec: EncodedList):
    """Yield the centered unsigned q4 symbols stored by one encoded list."""
    if spec.precision != "int4" or spec.payload_bytes is None:
        raise ValueError("Huffman coding requires an int4 byte payload")
    remaining = spec.value_count
    if remaining is None:
        remaining = len(spec.payload_bytes) * 2
    for byte in spec.payload_bytes:
        if remaining <= 0:
            break
        yield byte // 16
        remaining -= 1
        if remaining:
            yield byte % 16
            remaining -= 1


def _length_limited_huffman_lengths(
    frequencies: Counter[int], *, maximum_bits: int = 8,
) -> dict[int, int]:
    """Return optimal prefix lengths for this tiny alphabet.

    A dynamic program chooses how many leaves to place at each depth. The q4
    alphabet has only fifteen live symbols, so this is both simpler and more
    auditable than a general package-merge implementation.
    """
    ordered = sorted(
        (symbol for symbol, count in frequencies.items() if count > 0),
        key=lambda symbol: (-frequencies[symbol], symbol),
    )
    if not ordered:
        return {}
    if len(ordered) == 1:
        return {ordered[0]: 1}
    if len(ordered) > 2**maximum_bits:
        raise ValueError("Huffman alphabet exceeds its maximum code length")
    weights = tuple(frequencies[symbol] for symbol in ordered)
    prefix = [0]
    for weight in weights:
        prefix.append(prefix[-1] + weight)

    @lru_cache(maxsize=None)
    def solve(depth: int, assigned: int, slots: int):
        remaining = len(ordered) - assigned
        if depth == maximum_bits:
            if remaining <= slots:
                return (depth * (prefix[-1] - prefix[assigned]), (remaining,))
            return None
        best = None
        for leaves in range(min(slots, remaining) + 1):
            after = remaining - leaves
            local = depth * (prefix[assigned + leaves] - prefix[assigned])
            if after == 0:
                candidate = (local, (leaves,))
            else:
                next_slots = (slots - leaves) * 2
                maximum_leaves = next_slots * 2 ** (maximum_bits - depth - 1)
                if next_slots == 0 or after > maximum_leaves:
                    continue
                suffix = solve(depth + 1, assigned + leaves, next_slots)
                if suffix is None:
                    continue
                candidate = (local + suffix[0], (leaves, *suffix[1]))
            if best is None or candidate < best:
                best = candidate
        return best

    solution = solve(1, 0, 2)
    if solution is None:
        raise ValueError("could not construct a length-limited Huffman code")
    lengths = {}
    cursor = 0
    for depth, leaves in enumerate(solution[1], 1):
        for symbol in ordered[cursor:cursor + leaves]:
            lengths[symbol] = depth
        cursor += leaves
    if cursor != len(ordered):
        raise AssertionError("Huffman construction did not assign every symbol")
    return lengths


def _canonical_huffman_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    codes = {}
    code = 0
    previous_length = 0
    for symbol, bit_length in sorted(lengths.items(), key=lambda item: (item[1], item[0])):
        code <<= bit_length - previous_length
        codes[symbol] = (code, bit_length)
        code += 1
        previous_length = bit_length
    return codes


def _huffman_prefix_table(
    codes: dict[int, tuple[int, int]], *, lookup_bits: int = 8,
) -> bytes:
    table = bytearray(2**lookup_bits)
    for symbol, (code, bit_length) in codes.items():
        if bit_length > lookup_bits:
            raise ValueError("Huffman code exceeds the prefix table width")
        start = code << (lookup_bits - bit_length)
        stop = (code + 1) << (lookup_bits - bit_length)
        table[start:stop] = bytes([symbol * 16 + bit_length]) * (stop - start)
    return bytes(table)


def _encode_huffman_symbols(
    symbols, codes: dict[int, tuple[int, int]],
) -> bytes:
    encoded = bytearray()
    bits = 0
    bit_count = 0
    for symbol in symbols:
        code, length_ = codes[symbol]
        bits = (bits << length_) | code
        bit_count += length_
        while bit_count >= 8:
            bit_count -= 8
            encoded.append((bits >> bit_count) & 255)
            bits &= (1 << bit_count) - 1
    if bit_count:
        encoded.append(bits << (8 - bit_count))
    return bytes(encoded)


def rounded_values(
    values,
    precision: Literal["float32", "float16", "int8", "int6", "int4"],
    group_size: int = 64,
    scale_precision: Literal["float32", "float16"] = "float32",
):
    """Round values to the exact numbers represented by the selected format."""
    if precision in {"int8", "int6", "int4"}:
        codes, scales = quantize_values(
            values, precision, group_size, scale_precision,
        )
        return [
            code * scales[index // group_size]
            for index, code in enumerate(codes)
        ]
    code = ">f" if precision == "float32" else ">e"
    return [struct.unpack(code, _pack_value(code, value))[0] for value in values]


def _pack_value(code: str, value) -> bytes:
    number = float(value)
    try:
        return struct.pack(code, number)
    except OverflowError:
        return struct.pack(code, math.copysign(math.inf, number))


def power_table(precision: Literal["float32", "float16"]) -> list[float]:
    """Return exact multipliers for reconstructing normal IEEE values."""
    if precision == "float32":
        mantissa_bits, bias, maximum = 23, 127, 254
    else:
        mantissa_bits, bias, maximum = 10, 15, 30
    return [
        math.ldexp(1.0, exponent - bias - mantissa_bits)
        for exponent in range(1, maximum + 1)
    ]


@dataclass(frozen=True)
class EncodedList:
    """One physical Scratch list and its startup payload."""

    name: str
    payload_name: str
    payload: str
    precision: Literal["float32", "float16", "int8", "int6", "int4"]
    value_count: int | None = None
    group_size: int | None = None
    scale_list_name: str | None = None
    scale_payload_name: str | None = None
    scale_payload: str | None = None
    scale_precision: Literal["float32", "float16"] = "float32"
    payload_bytes: bytes | None = None
    scale_values: tuple[float, ...] = ()


def _decoded_value(precision):
    if precision == "float32":
        sign_divisor = 2**31
        exponent_divisor = 2**23
        exponent_modulus = 256
        maximum_exponent = 255
        hidden_bit = 2**23
        subnormal_scale = math.ldexp(1.0, -149)
        powers = "cattorch f32 powers"
    else:
        sign_divisor = 2**15
        exponent_divisor = 2**10
        exponent_modulus = 32
        maximum_exponent = 31
        hidden_bit = 2**10
        subnormal_scale = math.ldexp(1.0, -24)
        powers = "cattorch f16 powers"

    classify = (
        set_var("decode sign", mathop("floor", div(var("decode bits"), sign_divisor))),
        set_var(
            "decode exponent",
            mod(mathop("floor", div(var("decode bits"), exponent_divisor)), exponent_modulus),
        ),
        set_var("decode mantissa", mod(var("decode bits"), hidden_bit)),
        if_else(
            eq(var("decode exponent"), 0),
            (set_var("decode value", mul(var("decode mantissa"), subnormal_scale)),),
            (
                if_else(
                    eq(var("decode exponent"), maximum_exponent),
                    (
                        if_else(
                            eq(var("decode mantissa"), 0),
                            (set_var("decode value", div(1, 0)),),
                            (set_var("decode value", div(0, 0)),),
                        ),
                    ),
                    (
                        set_var(
                            "decode value",
                            mul(
                                add(hidden_bit, var("decode mantissa")),
                                item(powers, var("decode exponent")),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        if_else(
            eq(var("decode sign"), 1),
            (set_var("decode value", mul(-1, var("decode value"))),),
            (set_var("decode value", var("decode value")),),
        ),
    )
    return classify


def _emit_quantized_value(code, spec: EncodedList, *, guard: bool = False):
    assert spec.scale_list_name is not None
    center = 127 if spec.precision == "int8" else 7
    statements = (
        set_var("decode value", sub(code, center)),
        append(spec.name, mul(var("decode value"), var("decode scale"))),
        change_var("decode value count", 1),
        change_var("decode group offset", 1),
        if_(
            eq(var("decode group offset"), spec.group_size),
            (
                set_var("decode group offset", 0),
                change_var("decode scale index", 1),
                set_var(
                    "decode scale",
                    item(spec.scale_list_name, var("decode scale index")),
                ),
            ),
        ),
    )
    if guard:
        return (if_(lt(var("decode value count"), spec.value_count), statements),)
    return statements


def _emit_byte(byte, spec: EncodedList):
    if spec.precision == "int8":
        return _emit_quantized_value(byte, spec)
    if spec.precision == "int4":
        return (
            *_emit_quantized_value(mathop("floor", div(byte, 16)), spec),
            *_emit_quantized_value(
                mod(byte, 16),
                spec,
                guard=bool(spec.value_count and spec.value_count % 2),
            ),
        )

    byte_count = 4 if spec.precision == "float32" else 2
    return (
        set_var("decode bits", add(mul(var("decode bits"), 256), byte)),
        change_var("decode byte count", 1),
        if_else(
            eq(var("decode byte count"), byte_count),
            (
                *_decoded_value(spec.precision),
                append(spec.name, var("decode value")),
                set_var("decode bits", 0),
                set_var("decode byte count", 0),
            ),
            (set_var("decode value", var("decode value")),),
        ),
    )


def _consume_base92_digit(payload_name: str, offset: int, accumulator: str):
    return (
        switch_costume(
            letter(var(payload_name), add(var("decode cursor"), offset)),
        ),
        set_var("decode digit", sub(costume_number(), 1)),
        set_var(
            accumulator,
            add(mul(var(accumulator), 92), var("decode digit")),
        ),
    )


def _base92_decoded_bytes():
    return (
        mathop("floor", div(var("decode high"), 2**44)),
        mod(mathop("floor", div(var("decode high"), 2**36)), 256),
        mod(mathop("floor", div(var("decode high"), 2**28)), 256),
        mod(mathop("floor", div(var("decode high"), 2**20)), 256),
        mod(mathop("floor", div(var("decode high"), 2**12)), 256),
        mod(mathop("floor", div(var("decode high"), 2**4)), 256),
        add(
            mul(mod(var("decode high"), 16), 16),
            mathop("floor", div(var("decode low"), 2**48)),
        ),
        mod(mathop("floor", div(var("decode low"), 2**40)), 256),
        mod(mathop("floor", div(var("decode low"), 2**32)), 256),
        mod(mathop("floor", div(var("decode low"), 2**24)), 256),
        mod(mathop("floor", div(var("decode low"), 2**16)), 256),
        mod(mathop("floor", div(var("decode low"), 2**8)), 256),
        mod(var("decode low"), 256),
    )


def _decode_payload(spec: EncodedList):
    if not spec.payload:
        raise ValueError("Base92 payload is missing its padding header")
    padding = BASE92_ALPHABET.index(spec.payload[0])
    data_characters = len(spec.payload) - 1
    if data_characters % 16:
        raise ValueError("invalid Base92 payload length")
    groups = data_characters // 16
    decoded_bytes = _base92_decoded_bytes()
    parser: list[Statement] = [
        set_var("decode high", 0), set_var("decode low", 0),
    ]
    for offset in range(8):
        parser.extend(_consume_base92_digit(
            spec.payload_name, offset, "decode high",
        ))
    for offset in range(8, 16):
        parser.extend(_consume_base92_digit(
            spec.payload_name, offset, "decode low",
        ))
    parser.append(change_var("decode cursor", 16))
    group_body = list(parser)
    for byte in decoded_bytes:
        group_body.extend(_emit_byte(byte, spec))
    body = [
        set_var("decode cursor", 2),
        set_var("decode bits", 0),
        set_var("decode byte count", 0),
    ]
    if groups > 1:
        body.append(repeat(groups - 1, tuple(group_body)))
    if groups:
        body.extend(parser)
        for byte in decoded_bytes[:13 - padding]:
            body.extend(_emit_byte(byte, spec))
    elif padding:
        raise ValueError("empty Base92 payload has nonzero padding")
    return tuple(body)


def _decode_list(spec: EncodedList):
    if spec.precision not in {"int8", "int4"}:
        return _decode_payload(spec)
    if (
        spec.value_count is None
        or spec.group_size is None
        or spec.scale_list_name is None
        or spec.scale_payload_name is None
        or spec.scale_payload is None
    ):
        raise ValueError("quantized encoded lists require count, group, and scale metadata")
    scale_spec = EncodedList(
        spec.scale_list_name,
        spec.scale_payload_name,
        spec.scale_payload,
        spec.scale_precision,
    )
    return (
        *_decode_payload(scale_spec),
        set_var("decode value count", 0),
        set_var("decode group offset", 0),
        set_var("decode scale index", 1),
        set_var("decode scale", item(spec.scale_list_name, 1)),
        *_decode_payload(spec),
        clear(spec.scale_list_name),
    )


_ACTIVE_PAYLOAD = "cattorch active payload"
_DECODED_BYTES = "cattorch decoded bytes"
_DECODED_VALUES = "cattorch decoded values"
_DECODED_SCALES = "cattorch decoded scales"
_HUFFMAN_TABLE = "cattorch huffman table"
_HUFFMAN_POWERS = "cattorch huffman powers"
_BYTE_DECODER = "cattorch decode costume base92 bytes"


def _shared_base92_decoder() -> Program:
    """Decode the active costume-name Base92 payload into reusable bytes."""
    decoded_bytes = _base92_decoded_bytes()
    group: list[Statement] = [
        set_var("decode high", 0), set_var("decode low", 0),
    ]
    for offset in range(8):
        group.extend(_consume_base92_digit(
            _ACTIVE_PAYLOAD, offset, "decode high",
        ))
    for offset in range(8, 16):
        group.extend(_consume_base92_digit(
            _ACTIVE_PAYLOAD, offset, "decode low",
        ))
    group.extend(append(_DECODED_BYTES, byte) for byte in decoded_bytes)
    group.append(change_var("decode cursor", 16))

    return Program(
        "shared_base92_decoder",
        variables=(
            _ACTIVE_PAYLOAD, "decode cursor", "decode padding",
            "decode high", "decode low", "decode digit", "decode raw length",
        ),
        lists=(_DECODED_BYTES,),
        body=(
            clear(_DECODED_BYTES),
            switch_costume(letter(var(_ACTIVE_PAYLOAD), 1)),
            set_var("decode padding", sub(costume_number(), 1)),
            set_var("decode cursor", 2),
            repeat(
                div(sub(string_length(var(_ACTIVE_PAYLOAD)), 1), 16),
                tuple(group),
            ),
            set_var(
                "decode raw length",
                sub(
                    mul(
                        13,
                        div(sub(string_length(var(_ACTIVE_PAYLOAD)), 1), 16),
                    ),
                    var("decode padding"),
                ),
            ),
            repeat(
                sub(length(_DECODED_BYTES), var("decode raw length")),
                (delete(_DECODED_BYTES, length(_DECODED_BYTES)),),
            ),
        ),
    )


def _shared_float_decoder(precision: Literal["float32", "float16"]) -> Program:
    byte_count = 4 if precision == "float32" else 2
    assemble = []
    for offset in range(byte_count):
        assemble.append(set_var(
            "decode bits",
            add(
                mul(var("decode bits"), 256),
                item(_DECODED_BYTES, add(var("decode cursor"), offset)),
            ),
        ))
    return Program(
        f"shared_{precision}_decoder",
        variables=(
            "decode cursor", "decode bits", "decode sign", "decode exponent",
            "decode mantissa", "decode value", "decode byte start",
            "decode byte length",
        ),
        lists=(
            _DECODED_BYTES, _DECODED_VALUES,
            "cattorch f32 powers", "cattorch f16 powers",
        ),
        list_values={
            "cattorch f32 powers": power_table("float32"),
            "cattorch f16 powers": power_table("float16"),
        },
        body=(
            clear(_DECODED_VALUES),
            set_var("decode cursor", var("decode byte start")),
            repeat(
                mathop("floor", div(var("decode byte length"), byte_count)),
                (
                    set_var("decode bits", 0),
                    *assemble,
                    *_decoded_value(precision),
                    append(_DECODED_VALUES, var("decode value")),
                    change_var("decode cursor", byte_count),
                ),
            ),
        ),
    )


def _quant_append(code):
    return (
        set_var("decode value", code),
        append(_DECODED_VALUES, mul(var("decode value"), var("decode scale"))),
        change_var("decode value count", 1),
        change_var("decode group offset", 1),
        if_(
            eq(var("decode group offset"), var("decode group size")),
            (
                set_var("decode group offset", 0),
                change_var("decode scale index", 1),
                set_var("decode scale", item(_DECODED_SCALES, var("decode scale index"))),
            ),
        ),
    )


def _shared_quant_decoder(
    precision: Literal["int8", "int6", "int4"],
) -> Program:
    setup = (
        clear(_DECODED_VALUES),
        set_var("decode cursor", var("decode byte start")),
        set_var("decode value count", 0),
        set_var("decode group offset", 0),
        set_var("decode scale index", var("decode scale start")),
        set_var("decode scale", item(_DECODED_SCALES, var("decode scale start"))),
    )
    if precision == "int8":
        loop = repeat(
            var("decode total values"),
            (
                *_quant_append(sub(item(_DECODED_BYTES, var("decode cursor")), 127)),
                change_var("decode cursor", 1),
            ),
        )
        lists = (_DECODED_BYTES, _DECODED_VALUES, _DECODED_SCALES)
    elif precision == "int6":
        first = item(_DECODED_BYTES, var("decode cursor"))
        second = item(_DECODED_BYTES, add(var("decode cursor"), 1))
        third = item(_DECODED_BYTES, add(var("decode cursor"), 2))
        codes = (
            mathop("floor", div(first, 4)),
            add(mul(mod(first, 4), 16), mathop("floor", div(second, 16))),
            add(mul(mod(second, 16), 4), mathop("floor", div(third, 64))),
            mod(third, 64),
        )
        group: list[Statement] = []
        for code in codes:
            group.append(if_(
                lt(var("decode value count"), var("decode total values")),
                _quant_append(sub(code, 31)),
            ))
        group.append(change_var("decode cursor", 3))
        loop = repeat(
            mathop("ceiling", div(var("decode total values"), 4)),
            tuple(group),
        )
        lists = (_DECODED_BYTES, _DECODED_VALUES, _DECODED_SCALES)
    else:
        first = _quant_append(sub(mathop(
            "floor", div(item(_DECODED_BYTES, var("decode cursor")), 16),
        ), 7))
        second = _quant_append(sub(
            mod(item(_DECODED_BYTES, var("decode cursor")), 16), 7,
        ))
        loop = repeat(
            mathop("ceiling", div(var("decode total values"), 2)),
            (
                *first,
                if_(lt(var("decode value count"), var("decode total values")), second),
                change_var("decode cursor", 1),
            ),
        )
        lists = (_DECODED_BYTES, _DECODED_VALUES, _DECODED_SCALES)
    return Program(
        f"shared_{precision}_decoder",
        variables=(
            _ACTIVE_PAYLOAD, "decode cursor", "decode value", "decode value count",
            "decode total values", "decode group offset", "decode group size",
            "decode scale index", "decode scale", "decode scale start",
            "decode byte start",
        ),
        lists=lists,
        body=(*setup, loop),
    )


def _shared_huffman_int4_decoder() -> Program:
    """Decode byte-aligned canonical q4 Huffman streams into float weights."""
    load_byte = (
        set_var(
            "huffman bits",
            add(
                mul(var("huffman bits"), 256),
                item(_DECODED_BYTES, var("decode cursor")),
            ),
        ),
        change_var("decode cursor", 1),
        change_var("huffman bit count", 8),
    )
    return Program(
        "shared_huffman_int4_decoder",
        variables=(
            "decode cursor", "decode byte start", "decode value",
            "decode value count", "decode total values", "decode group offset",
            "decode group size", "decode scale index", "decode scale",
            "decode scale start", "huffman bits", "huffman bit count",
            "huffman prefix", "huffman entry", "huffman length",
        ),
        lists=(
            _DECODED_BYTES, _DECODED_VALUES, _DECODED_SCALES,
            _HUFFMAN_TABLE, _HUFFMAN_POWERS,
        ),
        list_values={_HUFFMAN_POWERS: [2**index for index in range(16)]},
        body=(
            clear(_DECODED_VALUES),
            set_var("decode cursor", var("decode byte start")),
            set_var("decode value count", 0),
            set_var("decode group offset", 0),
            set_var("decode scale index", var("decode scale start")),
            set_var("decode scale", item(_DECODED_SCALES, var("decode scale start"))),
            set_var("huffman bits", 0),
            set_var("huffman bit count", 0),
            repeat(
                var("decode total values"),
                (
                    if_(lt(var("huffman bit count"), 8), load_byte),
                    set_var(
                        "huffman prefix",
                        mathop("floor", div(
                            var("huffman bits"),
                            item(
                                _HUFFMAN_POWERS,
                                sub(var("huffman bit count"), 7),
                            ),
                        )),
                    ),
                    set_var(
                        "huffman entry",
                        item(_HUFFMAN_TABLE, add(var("huffman prefix"), 1)),
                    ),
                    set_var("huffman length", mod(var("huffman entry"), 16)),
                    *_quant_append(sub(
                        mathop("floor", div(var("huffman entry"), 16)), 7,
                    )),
                    change_var("huffman bit count", mul(-1, var("huffman length"))),
                    set_var(
                        "huffman bits",
                        mod(
                            var("huffman bits"),
                            item(
                                _HUFFMAN_POWERS,
                                add(var("huffman bit count"), 1),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def build_decoder_programs(specs: list[EncodedList]) -> dict[str, Program]:
    """Return the shared startup procedures required by ``specs``."""
    precisions = {spec.precision for spec in specs}
    scale_precisions = {
        spec.scale_precision for spec in specs if spec.precision.startswith("int")
    }
    programs = {_BYTE_DECODER: _shared_base92_decoder()}
    for precision in sorted((precisions | scale_precisions) & {"float32", "float16"}):
        float_precision = cast(Literal["float32", "float16"], precision)
        programs[f"cattorch decode {precision}"] = _shared_float_decoder(float_precision)
    for precision in sorted(precisions & {"int8", "int6", "int4"}):
        integer_precision = cast(Literal["int8", "int6", "int4"], precision)
        programs[f"cattorch decode {precision}"] = _shared_quant_decoder(integer_precision)
    if "int4" in precisions:
        programs["cattorch decode int4 huffman"] = _shared_huffman_int4_decoder()
    return programs


def _copy_decoded(destination: str):
    return (
        clear(destination),
        for_each(
            "decode copy index",
            length(_DECODED_VALUES),
            (append(destination, item(_DECODED_VALUES, var("decode copy index"))),),
        ),
    )


def build_shared_unpack_program(specs: list[EncodedList]) -> Program:
    """Build compact wrappers around shared per-precision decoders."""
    if specs and all(spec.payload_bytes is not None for spec in specs):
        return _build_banked_unpack_program(specs)

    body: list[Statement] = []
    for spec in specs:
        if spec.precision in {"int8", "int6", "int4"}:
            if not spec.scale_payload_name or spec.scale_payload is None:
                raise ValueError("quantized encoded lists require scale metadata")
            body.extend((
                set_var(_ACTIVE_PAYLOAD, var(spec.scale_payload_name)),
                call(_BYTE_DECODER),
                set_var("decode byte start", 1),
                set_var("decode byte length", length(_DECODED_BYTES)),
                call(f"cattorch decode {spec.scale_precision}"),
                clear(_DECODED_SCALES),
                for_each(
                    "decode copy index", length(_DECODED_VALUES),
                    (append(
                        _DECODED_SCALES,
                        item(_DECODED_VALUES, var("decode copy index")),
                    ),),
                ),
                set_var(_ACTIVE_PAYLOAD, var(spec.payload_name)),
            ))
            body.append(call(_BYTE_DECODER))
            body.extend((
                set_var("decode byte start", 1),
                set_var("decode byte length", length(_DECODED_BYTES)),
                set_var("decode scale start", 1),
                set_var("decode total values", spec.value_count or 0),
                set_var("decode group size", spec.group_size or 1),
                call(f"cattorch decode {spec.precision}"),
            ))
        else:
            body.extend((
                set_var(_ACTIVE_PAYLOAD, var(spec.payload_name)),
                call(_BYTE_DECODER),
                set_var("decode byte start", 1),
                set_var("decode byte length", length(_DECODED_BYTES)),
                call(f"cattorch decode {spec.precision}"),
            ))
        body.extend(_copy_decoded(spec.name))
    body.extend((clear(_DECODED_BYTES), clear(_DECODED_VALUES), clear(_DECODED_SCALES)))
    variables = (
        _ACTIVE_PAYLOAD, "decode copy index", "decode total values", "decode group size",
        "decode byte start", "decode byte length", "decode scale start",
        *(spec.payload_name for spec in specs),
        *(spec.scale_payload_name for spec in specs if spec.scale_payload_name),
    )
    variable_values = {spec.payload_name: spec.payload for spec in specs}
    variable_values.update({
        spec.scale_payload_name: spec.scale_payload
        for spec in specs
        if spec.scale_payload_name and spec.scale_payload is not None
    })
    return Program(
        "unpack_weights",
        variables=variables,
        variable_values=variable_values,
        lists=(
            _DECODED_BYTES, _DECODED_VALUES, _DECODED_SCALES,
            *(spec.name for spec in specs),
        ),
        body=tuple(body),
    )


def _huffman_bank(
    bank: list[EncodedList],
) -> tuple[dict[int, tuple[int, int]], list[bytes]]:
    """Return one bank's canonical codes and each tensor's byte-aligned stream."""
    frequencies: Counter[int] = Counter()
    for spec in bank:
        frequencies.update(_int4_symbols(spec))
    codes = _canonical_huffman_codes(_length_limited_huffman_lengths(frequencies))
    return codes, [_encode_huffman_symbols(_int4_symbols(spec), codes) for spec in bank]


def _fit_huffman_banks(
    banks: list[list[EncodedList]],
    encoded: dict[int, tuple[dict[int, tuple[int, int]], list[bytes]]],
) -> list[list[EncodedList]]:
    """Split int4 banks whose Huffman stream would overflow a Scratch list.

    Banks are sized by their 4-bit payloads. Byte-aligning each tensor's
    stream can make the Huffman form slightly longer, and Scratch silently
    drops appends past 200,000 items. A single tensor always fits, because a
    Huffman code built for it alone is never longer than fixed 4-bit codes.
    """
    fitted: list[list[EncodedList]] = []
    pending = list(reversed(banks))
    while pending:
        bank = pending.pop()
        codes, chunks = _huffman_bank(bank)
        if len(bank) > 1 and sum(len(chunk) for chunk in chunks) > SCRATCH_LIST_LIMIT:
            middle = len(bank) // 2
            pending.extend((bank[middle:], bank[:middle]))
            continue
        encoded[id(bank)] = (codes, chunks)
        fitted.append(bank)
    return fitted


def _build_banked_unpack_program(specs: list[EncodedList]) -> Program:
    """Decode jointly capped weight and quantization-scale banks.

    The Base92 decoder materializes bytes in a Scratch list. Vanilla Scratch
    silently ignores appends after 200,000 items, so neither a weight stream
    nor its scale stream may exceed that limit. Scales use bank-local offsets
    and are discarded before the next bank. Int4 banks use one canonical,
    length-limited Huffman table and byte-align each physical tensor so it can
    decode directly into its destination without a second oversized code list.
    """
    integer_precisions = {"int8", "int6", "int4"}
    body: list[Statement] = []
    payload_values: dict[str, str] = {}
    for precision in ("float32", "float16", "int8", "int6", "int4"):
        selected = [spec for spec in specs if spec.precision == precision]
        if not selected:
            continue
        banks: list[list[EncodedList]] = []
        current: list[EncodedList] = []
        current_weight_bytes = 0
        current_scale_bytes = {"float32": 0, "float16": 0}
        for spec in selected:
            assert spec.payload_bytes is not None
            weight_bytes = len(spec.payload_bytes)
            scale_bytes = 0
            if spec.precision in integer_precisions:
                if spec.value_count is None or spec.group_size is None:
                    raise ValueError("quantized encoded lists require count and group metadata")
                expected_scales = math.ceil(spec.value_count / spec.group_size)
                if len(spec.scale_values) != expected_scales:
                    raise ValueError(
                        f"encoded list {spec.name!r} has {len(spec.scale_values)} "
                        f"scales, expected {expected_scales}"
                    )
                scale_bytes = len(spec.scale_values) * (
                    4 if spec.scale_precision == "float32" else 2
                )
            if weight_bytes > SCRATCH_LIST_LIMIT:
                raise ValueError(
                    f"encoded list {spec.name!r} needs {weight_bytes} temporary "
                    f"weight bytes, exceeding Scratch's {SCRATCH_LIST_LIMIT}-item limit"
                )
            if scale_bytes > SCRATCH_LIST_LIMIT:
                raise ValueError(
                    f"encoded list {spec.name!r} needs {scale_bytes} temporary "
                    f"scale bytes, exceeding Scratch's {SCRATCH_LIST_LIMIT}-item limit"
                )
            exceeds_weight_bank = current_weight_bytes + weight_bytes > SCRATCH_LIST_LIMIT
            exceeds_scale_bank = (
                spec.precision in integer_precisions
                and current_scale_bytes[spec.scale_precision] + scale_bytes
                > SCRATCH_LIST_LIMIT
            )
            if current and (exceeds_weight_bank or exceeds_scale_bank):
                banks.append(current)
                current = []
                current_weight_bytes = 0
                current_scale_bytes = {"float32": 0, "float16": 0}
            current.append(spec)
            current_weight_bytes += weight_bytes
            if spec.precision in integer_precisions:
                current_scale_bytes[spec.scale_precision] += scale_bytes
        if current:
            banks.append(current)
        huffman_banks: dict[int, tuple[dict[int, tuple[int, int]], list[bytes]]] = {}
        if precision == "int4":
            banks = _fit_huffman_banks(banks, huffman_banks)

        for bank_index, bank in enumerate(banks, 1):
            suffix = f" {bank_index}" if len(banks) > 1 else ""
            scale_offsets: dict[int, int] = {}
            if precision in integer_precisions:
                body.append(clear(_DECODED_SCALES))
                scale_cursor = 1
                for scale_precision in ("float32", "float16"):
                    scale_specs = [
                        spec for spec in bank
                        if spec.scale_precision == scale_precision
                    ]
                    if not scale_specs:
                        continue
                    scale_values: list[float] = []
                    for spec in scale_specs:
                        scale_offsets[id(spec)] = scale_cursor + len(scale_values)
                        scale_values.extend(spec.scale_values)
                    scale_name = (
                        f"cattorch {precision} {scale_precision} "
                        f"scale bank{suffix} payload"
                    )
                    payload_values[scale_name] = encode_values(
                        scale_values, scale_precision,
                    )
                    body.extend((
                        set_var(_ACTIVE_PAYLOAD, var(scale_name)),
                        call(_BYTE_DECODER),
                        set_var("decode byte start", 1),
                        set_var("decode byte length", length(_DECODED_BYTES)),
                        call(f"cattorch decode {scale_precision}"),
                        for_each(
                            "decode copy index", length(_DECODED_VALUES),
                            (append(
                                _DECODED_SCALES,
                                item(_DECODED_VALUES, var("decode copy index")),
                            ),),
                        ),
                    ))
                    scale_cursor += len(scale_values)

            huffman_offsets: dict[int, int] = {}
            if precision == "int4":
                codes, chunks = huffman_banks[id(bank)]
                table_name = f"cattorch int4 huffman bank{suffix} table"
                payload_values[table_name] = _encode_bytes(
                    _huffman_prefix_table(codes),
                )
                body.extend((
                    set_var(_ACTIVE_PAYLOAD, var(table_name)),
                    call(_BYTE_DECODER),
                    clear(_HUFFMAN_TABLE),
                    for_each(
                        "decode copy index", 256,
                        (append(
                            _HUFFMAN_TABLE,
                            item(_DECODED_BYTES, var("decode copy index")),
                        ),),
                    ),
                ))
                compressed_cursor = 1
                for spec, chunk in zip(bank, chunks):
                    huffman_offsets[id(spec)] = compressed_cursor
                    compressed_cursor += len(chunk)
                raw = b"".join(chunks)
            else:
                raw = b"".join(spec.payload_bytes or b"" for spec in bank)

            payload_name = f"cattorch {precision} weight bank{suffix} payload"
            payload_values[payload_name] = _encode_bytes(raw)
            body.extend((
                set_var(_ACTIVE_PAYLOAD, var(payload_name)),
                call(_BYTE_DECODER),
            ))
            byte_offset = 1
            for spec in bank:
                assert spec.payload_bytes is not None
                body.extend((
                    set_var(
                        "decode byte start",
                        huffman_offsets.get(id(spec), byte_offset),
                    ),
                    set_var("decode byte length", len(spec.payload_bytes)),
                ))
                if precision in integer_precisions:
                    body.extend((
                        set_var("decode scale start", scale_offsets[id(spec)]),
                        set_var("decode total values", spec.value_count or 0),
                        set_var("decode group size", spec.group_size or 1),
                        call(
                            "cattorch decode int4 huffman"
                            if precision == "int4"
                            else f"cattorch decode {precision}"
                        ),
                    ))
                else:
                    body.append(call(f"cattorch decode {precision}"))
                body.extend(_copy_decoded(spec.name))
                byte_offset += len(spec.payload_bytes)

    body.extend((
        clear(_DECODED_BYTES), clear(_DECODED_VALUES), clear(_DECODED_SCALES),
        clear(_HUFFMAN_TABLE),
    ))
    return Program(
        "unpack_banked_weights",
        variables=(
            _ACTIVE_PAYLOAD, "decode copy index", "decode total values",
            "decode group size", "decode byte start", "decode byte length",
            "decode scale start", *payload_values,
        ),
        variable_values=payload_values,
        lists=(
            _DECODED_BYTES, _DECODED_VALUES, _DECODED_SCALES, _HUFFMAN_TABLE,
            *(spec.name for spec in specs),
        ),
        body=tuple(body),
    )


def build_unpack_program(specs: list[EncodedList]) -> Program:
    """Build a self-contained decoder, retained for direct DSL consumers.

    Production exports use :func:`build_shared_unpack_program` plus the shared
    decoder procedures. Keeping this form makes a decoder program independently
    executable, which is useful for unit tests and small custom sprites.
    """
    if any(spec.precision == "int6" for spec in specs):
        raise ValueError("int6 requires the shared production decoder")
    variables = (
        "decode cursor", "decode bits", "decode byte count",
        "decode high", "decode low", "decode digit",
        "decode sign", "decode exponent", "decode mantissa", "decode value",
        "decode value count", "decode group offset", "decode scale index",
        "decode scale",
        *(spec.payload_name for spec in specs),
        *(spec.scale_payload_name for spec in specs if spec.scale_payload_name),
    )
    lists = (
        "cattorch f32 powers", "cattorch f16 powers",
        *(spec.name for spec in specs),
        *(spec.scale_list_name for spec in specs if spec.scale_list_name),
    )
    variable_values = {spec.payload_name: spec.payload for spec in specs}
    variable_values.update({
        spec.scale_payload_name: spec.scale_payload
        for spec in specs
        if spec.scale_payload_name and spec.scale_payload is not None
    })
    return Program(
        "unpack_weights",
        variables=variables,
        variable_values=variable_values,
        lists=lists,
        list_values={
            "cattorch f32 powers": power_table("float32"),
            "cattorch f16 powers": power_table("float16"),
        },
        body=tuple(statement for spec in specs for statement in _decode_list(spec)),
    )
