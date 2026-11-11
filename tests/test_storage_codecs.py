"""The vectorized storage codecs against the scalar implementations they replaced.

The reference functions below are the previous per-value implementations,
kept here so every change to the codecs is checked bit for bit against them.
"""

import math
import random
import struct

import numpy as np
import pytest
import torch

import cattorch.storage as storage_module
from cattorch import QuantizationConfig
from cattorch.quantization import _symmetric_matrix, effective_group_size
from cattorch.storage import (
    BASE92_ALPHABET,
    _encode_bytes,
    _float_bytes,
    _quantized_bytes,
    encode_values,
    quantize_values,
    rounded_values,
)

PRECISIONS = ("int8", "int6", "int4")
MAXIMUM_CODES = {"int8": 127, "int6": 31, "int4": 7}


def _reference_pack(code, value):
    number = float(value)
    try:
        return struct.pack(code, number)
    except OverflowError:
        return struct.pack(code, math.copysign(math.inf, number))


def _reference_float_bytes(values, precision):
    code = ">f" if precision == "float32" else ">e"
    return b"".join(_reference_pack(code, value) for value in values)


def _reference_encode_bytes(raw):
    padding = (-len(raw)) % 13
    padded = raw + bytes(padding)
    encoded = [BASE92_ALPHABET[padding]]
    mask = 2**52 - 1
    for offset in range(0, len(padded), 13):
        packed = int.from_bytes(padded[offset:offset + 13], "big")
        for half in (packed >> 52, packed & mask):
            digits = [0] * 8
            for index in range(7, -1, -1):
                half, digits[index] = divmod(half, 92)
            encoded.extend(BASE92_ALPHABET[digit] for digit in digits)
    return "".join(encoded)


def _reference_quantize_values(values, precision, group_size=64, scale_precision="float32"):
    numbers = [float(value) for value in values]
    if any(not math.isfinite(number) for number in numbers):
        raise ValueError("integer weight quantization requires finite values")
    maximum_code = MAXIMUM_CODES[precision]
    codes, scales = [], []
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
        smallest = math.ldexp(1.0, -149 if scale_precision == "float32" else -24)
        if maximum and scale == 0:
            scale = smallest
        if maximum:
            # Storage re-derives the scale from the largest reconstructed
            # value; a scale it wouldn't get back is replaced by the largest
            # representable one below maximum / maximum_code.
            top = min(round(maximum / scale), maximum_code)
            recovered = struct.unpack(
                scale_code, struct.pack(scale_code, top * scale / maximum_code),
            )[0] if top else scale
            if top and recovered == 0:
                recovered = smallest
            if recovered != scale:
                ideal = maximum / maximum_code
                below = struct.unpack(scale_code, struct.pack(scale_code, ideal))[0]
                if below > ideal:
                    integer = ">I" if scale_precision == "float32" else ">H"
                    bits = struct.unpack(integer, struct.pack(scale_code, below))[0]
                    below = struct.unpack(scale_code, struct.pack(integer, bits - 1))[0]
                scale = max(below, smallest)
        scales.append(scale)
        for number in group:
            code = round(number / scale) if maximum else 0
            codes.append(max(-maximum_code, min(maximum_code, code)))
    return codes, scales


def _reference_quantized_bytes(codes, precision):
    if precision == "int8":
        return bytes(code + 127 for code in codes)
    if precision == "int6":
        packed = []
        for index in range(0, len(codes), 4):
            group = [code + 31 for code in codes[index:index + 4]]
            group.extend([31] * (4 - len(group)))
            packed.extend((
                group[0] * 4 + group[1] // 16,
                (group[1] % 16) * 16 + group[2] // 4,
                (group[2] % 4) * 64 + group[3],
            ))
        return bytes(packed[:math.ceil(len(codes) * 6 / 8)])
    packed = []
    for index in range(0, len(codes), 2):
        high = codes[index] + 7
        low = codes[index + 1] + 7 if index + 1 < len(codes) else 7
        packed.append(high * 16 + low)
    return bytes(packed)


def _reference_rounded_values(values, precision, group_size=64, scale_precision="float32"):
    if precision in PRECISIONS:
        codes, scales = _reference_quantize_values(values, precision, group_size, scale_precision)
        return [code * scales[index // group_size] for index, code in enumerate(codes)]
    code = ">f" if precision == "float32" else ">e"
    return [struct.unpack(code, _reference_pack(code, value))[0] for value in values]


def _bits(values):
    """Exact float identities, so -0.0 and 0.0 differ."""
    return [struct.pack(">d", value) for value in values]


def _from_bits(pattern, code):
    return struct.unpack(code, pattern.to_bytes(struct.calcsize(code), "big"))[0]


def _float16_boundaries():
    """Every finite float16, the midpoints between neighbours, and the doubles
    just either side of each midpoint."""
    values = np.arange(0x7C00, dtype=np.uint16).view(np.float16).astype(np.float64)
    midpoints = (values[:-1] + values[1:]) / 2
    above = np.nextafter(midpoints, math.inf)
    below = np.nextafter(midpoints, -math.inf)
    edges = np.concatenate((values, midpoints, above, below, [65504.0, 65519.99, 65520.0, 1e6]))
    return np.concatenate((edges, -edges)).tolist()


def _float32_boundaries(rng, count=20_000):
    # Finite float32 patterns below the largest, whose upper neighbour is finite.
    patterns = [pattern for pattern in (rng.getrandbits(31) for _ in range(count)) if pattern < 0x7F7FFFFF]
    patterns += [0, 1, 2, 0x007FFFFF, 0x00800000, 0x00800001, 0x7F7FFFFE]
    values = np.array(patterns, dtype=np.uint32).view(np.float32)
    following = np.nextafter(values, np.float32(math.inf)).astype(np.float64)
    values = values.astype(np.float64)
    midpoints = (values + following) / 2
    edges = np.concatenate((
        values, midpoints, np.nextafter(midpoints, math.inf), np.nextafter(midpoints, -math.inf),
        [3.4028234663852886e38, 3.4028235677973366e38, 3.402823669209385e38, 1e39, 1e300, 5e-324, 2.0**-150, 2.0**-151],
    ))
    return np.concatenate((edges, -edges)).tolist()


@pytest.mark.parametrize("precision", ["float32", "float16"])
def test_float_bytes_match_struct_at_every_rounding_boundary(precision):
    rng = random.Random(0)
    values = _float16_boundaries() if precision == "float16" else _float32_boundaries(rng)
    values += [0.0, -0.0, math.inf, -math.inf, 1, -3, 2**70]
    values += [rng.gauss(0, 10 ** rng.randint(-40, 40)) for _ in range(5_000)]
    assert _float_bytes(values, precision) == _reference_float_bytes(values, precision)
    assert _bits(rounded_values(values, precision)) == _bits(_reference_rounded_values(values, precision))
    tensor = torch.tensor(values[:1000], dtype=torch.float64)
    assert _float_bytes(tensor, precision) == _reference_float_bytes(values[:1000], precision)


DOUBLE_NANS = (
    0x7FF8000000000000, 0xFFF8000000000000, 0x7FF4000000000000, 0x7FF0000000000001,
    0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x7FF0040000000000, 0xFFF0000020000000,
)


@pytest.mark.parametrize("pattern", DOUBLE_NANS)
def test_nan_storage_keeps_its_sign_and_is_quiet(pattern):
    # float16 NaNs are always the quiet NaN, as struct packs them. float32
    # NaNs keep the top of the payload and become quiet, as the hardware
    # conversion struct uses does.
    value = _from_bits(pattern, ">d")
    sign = pattern >> 63
    assert _float_bytes([value], "float16") == (0xFE00 if sign else 0x7E00).to_bytes(2, "big")
    mantissa = pattern & (2**52 - 1)
    expected = (sign << 31) | 0x7FC00000 | (mantissa >> 29)
    assert _float_bytes([value], "float32") == expected.to_bytes(4, "big")
    for precision in ("float32", "float16"):
        (rounded,) = rounded_values([value], precision)
        assert math.isnan(rounded) and math.copysign(1.0, rounded) == (-1.0 if sign else 1.0)
        assert _bits(rounded_values([value], precision)) == _bits(_reference_rounded_values([value], precision))
    assert _float_bytes([value], "float16") == _reference_float_bytes([value], "float16")
    assert _float_bytes([value], "float32") == _reference_float_bytes([value], "float32")


def _quantization_cases(rng):
    """Groups covering ties, zeros, signed zeros, subnormal scales and widths
    that do and don't divide the group size."""
    cases = []
    for _ in range(40):
        size = rng.randint(0, 300)
        magnitude = 10.0 ** rng.randint(-46, 30)
        cases.append([rng.gauss(0, magnitude) for _ in range(size)])
    for maximum_code in MAXIMUM_CODES.values():
        # A power-of-two scale is exact in every format, so k + 0.5 multiples
        # are exact ties for round-half-even.
        scale = 2.0**-3
        group = [maximum_code * scale] + [(k + 0.5) * scale for k in range(-maximum_code, maximum_code)]
        cases.append(group)
    cases.append([0.0] * 70)
    cases.append([-0.0, 0.0, -0.0])
    cases.append([0.0] * 64 + [1e-3] * 64)
    cases.append([5e-324, -5e-324, 1e-310])
    cases.append([2.0**-150, 2.0**-140, -(2.0**-130)])
    cases.append([1e-8, -1e-9, 3e-8])
    cases.append([65504.0 * 127, 1.0, -2.5])
    cases.append([1, 2, -3, True])
    return cases


@pytest.mark.parametrize("scale_precision", ["float32", "float16"])
@pytest.mark.parametrize("precision", PRECISIONS)
def test_quantization_matches_the_scalar_implementation(precision, scale_precision):
    rng = random.Random(f"{precision} {scale_precision}")
    for values in _quantization_cases(rng):
        for group_size in (1, 3, 64, 100):
            try:
                expected = _reference_quantize_values(values, precision, group_size, scale_precision)
            except ValueError as error:
                with pytest.raises(ValueError) as raised:
                    quantize_values(values, precision, group_size, scale_precision)
                assert str(raised.value) == str(error)
                with pytest.raises(ValueError):
                    rounded_values(values, precision, group_size, scale_precision)
                continue
            codes, scales = quantize_values(values, precision, group_size, scale_precision)
            assert (codes, scales) == expected
            assert all(type(code) is int for code in codes)
            assert _bits(scales) == _bits(expected[1])
            assert _quantized_bytes(codes, precision) == _reference_quantized_bytes(codes, precision)
            assert _bits(rounded_values(values, precision, group_size, scale_precision)) == _bits(
                _reference_rounded_values(values, precision, group_size, scale_precision)
            )


@pytest.mark.parametrize("precision", PRECISIONS)
def test_quantization_rejects_non_finite_values(precision):
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="requires finite values"):
            quantize_values([1.0, value], precision)


@pytest.mark.parametrize("scale_precision", ["float32", "float16"])
def test_quantization_overflow_message_names_the_first_overflowing_scale(scale_precision):
    values = [1.0, 1e300, 1e301] if scale_precision == "float32" else [1.0, 1e8, 1e9]
    with pytest.raises(ValueError) as expected:
        _reference_quantize_values(values, "int8", 1, scale_precision)
    with pytest.raises(ValueError) as raised:
        quantize_values(values, "int8", 1, scale_precision)
    assert str(raised.value) == str(expected.value)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_packed_codes_match_for_every_code_and_length(precision):
    maximum_code = MAXIMUM_CODES[precision]
    every_code = list(range(-maximum_code, maximum_code + 1))
    rng = random.Random(precision)
    for length in range(0, 13):
        codes = [rng.choice(every_code) for _ in range(length)]
        assert _quantized_bytes(codes, precision) == _reference_quantized_bytes(codes, precision)
    assert _quantized_bytes(every_code, precision) == _reference_quantized_bytes(every_code, precision)
    assert _quantized_bytes(np.array(every_code), precision) == _reference_quantized_bytes(every_code, precision)
    with pytest.raises(ValueError, match="codes must lie in"):
        _quantized_bytes([maximum_code + 1], precision)


def test_base92_matches_the_scalar_encoder(monkeypatch):
    rng = random.Random(0)
    payloads = [bytes(rng.getrandbits(8) for _ in range(length)) for length in range(0, 60)]
    payloads += [b"\xff" * 26, b"\x00" * 13, bytes(range(256)) * 7]
    for raw in payloads:
        assert _encode_bytes(raw) == _reference_encode_bytes(raw)
    # Long payloads are encoded in chunks of groups.
    monkeypatch.setattr(storage_module, "_BASE92_CHUNK_GROUPS", 3)
    for raw in payloads:
        assert _encode_bytes(raw) == _reference_encode_bytes(raw)
    values = [rng.gauss(0, 1) for _ in range(500)]
    for precision in ("float32", "float16"):
        assert encode_values(values, precision) == _reference_encode_bytes(
            _reference_float_bytes(values, precision),
        )


def _reference_symmetric_matrix(matrix, config):
    group = effective_group_size(matrix.shape[1], config.group_size)
    codes, scales = _reference_quantize_values(
        matrix.detach().reshape(-1).tolist(), config.precision, group, config.scale_precision,
    )
    return torch.tensor(
        [code * scales[index // group] for index, code in enumerate(codes)],
        device=matrix.device, dtype=matrix.dtype,
    ).reshape_as(matrix), group


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bits", [8, 4])
def test_symmetric_matrix_matches_the_scalar_implementation(dtype, bits):
    torch.manual_seed(bits)
    config = QuantizationConfig(bits=bits, group_size=48)
    for shape in ((7, 96), (5, 30), (1, 1)):
        matrix = (torch.randn(shape) * 3).to(dtype)
        matrix[0, 0] = 0
        reconstructed, group = _symmetric_matrix(matrix, config)
        expected, expected_group = _reference_symmetric_matrix(matrix, config)
        assert group == expected_group
        assert reconstructed.dtype == dtype
        bits_view = {2: torch.int16, 4: torch.int32, 8: torch.int64}[dtype.itemsize]
        assert torch.equal(reconstructed.view(bits_view), expected.view(bits_view))
