"""
jsmath.py
---------
JavaScript's ``Math`` functions as scratch-vm computes them, bit for bit.

V8 computes ``Math.sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``,
``log``, ``exp`` and ``pow`` with its port of fdlibm (``src/base/ieee754.cc``),
which rounds differently from the C library Python's ``math`` uses: about one
result in thirty differs in its last bit. These are line-for-line ports of
V8's code, so the emulator gets the doubles Scratch gets. ``round`` is
``Math.round`` and ``to_fixed`` is ``parseFloat(x.toFixed(digits))``, which
scratch-vm uses to round trigonometry.

C's 32-bit integer operations on the halves of a double are spelled out with
``_high``, ``_low`` and ``_words``.
"""

from __future__ import annotations

import math
import struct
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

_DOUBLE = struct.Struct("<d")
_WORD64 = struct.Struct("<Q")


def _bits(x: float) -> int:
    return _WORD64.unpack(_DOUBLE.pack(x))[0]


def _high(x: float) -> int:
    """The high 32 bits of ``x``, as a signed int32 (C's ``GET_HIGH_WORD``)."""
    high = _bits(x) >> 32
    return high - 0x100000000 if high & 0x80000000 else high


def _low(x: float) -> int:
    """The low 32 bits of ``x``, unsigned."""
    return _bits(x) & 0xFFFFFFFF


def _words(high: int, low: int) -> float:
    """The double with these halves (C's ``INSERT_WORDS``)."""
    return _DOUBLE.unpack(_WORD64.pack(((high & 0xFFFFFFFF) << 32) | (low & 0xFFFFFFFF)))[0]


def _clear_low(x: float) -> float:
    """``x`` with its low word zeroed (C's ``SET_LOW_WORD(x, 0)``)."""
    return _DOUBLE.unpack(_WORD64.pack(_bits(x) & 0xFFFFFFFF00000000))[0]


def _int32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def _trunc(x: float) -> int:
    """C's ``(int32_t) x`` for an ``x`` that fits."""
    return int(x)


_NAN = math.nan
_INF = math.inf


# ── Math.round and toFixed ──────────────────────────────────────────────────


def round(x):  # noqa: A001 - the name of the JavaScript function
    """``Math.round``: the nearest integer, halves up, ``-0`` kept.

    ``Math.floor(x + 0.5)`` would round twice: ``0.49999999999999994 + 0.5``
    is 1. An int in is an int out.
    """
    if x.__class__ is int:
        return x
    if not math.isfinite(x):
        return x
    floor = math.floor(x)
    result = floor + 1 if x - floor >= 0.5 else floor
    if result == 0 and (x < 0 or math.copysign(1.0, x) < 0):
        return -0.0
    return result


def floor(x):
    """``Math.floor``, with ``-0`` kept."""
    if x.__class__ is int or not math.isfinite(x):
        return x
    if x == 0:
        return x
    return math.floor(x)


def ceil(x):
    """``Math.ceil``: ``-0.5`` gives ``-0``, as it does in JavaScript."""
    if x.__class__ is int or not math.isfinite(x):
        return x
    result = math.ceil(x)
    if result == 0 and (x < 0 or math.copysign(1.0, x) < 0):
        return -0.0
    return result


def to_fixed(x: float, digits: int) -> float:
    """``parseFloat(x.toFixed(digits))``.

    ``toFixed`` rounds the exact value of ``|x|`` to ``digits`` decimals,
    halves up, and keeps the sign only when ``x < 0`` (so ``-0`` gives 0, and
    a negative number that rounds to zero gives ``-0``).
    """
    if not math.isfinite(x):
        return x
    if abs(x) >= 1e21:
        return x
    scaled = (Decimal(abs(x)) * (10 ** digits)).to_integral_value(rounding=ROUND_HALF_UP)
    result = float(Fraction(int(scaled), 10 ** digits))
    return -result if x < 0 else result


# ── exp, log and pow ────────────────────────────────────────────────────────

_HALF = (0.5, -0.5)
_O_THRESHOLD = 7.09782712893383973096e02
_U_THRESHOLD = -7.45133219101941108420e02
_LN2_HI = (6.93147180369123816490e-01, -6.93147180369123816490e-01)
_LN2_LO = (1.90821492927058770002e-10, -1.90821492927058770002e-10)
_INVLN2 = 1.44269504088896338700e00
_P1 = 1.66666666666666019037e-01
_P2 = -2.77777777770155933842e-03
_P3 = 6.61375632143793436117e-05
_P4 = -1.65339022054652515390e-06
_P5 = 4.13813679705723846039e-08
_E = 2.718281828459045
_TWOM1000 = 9.33263618503218878990e-302
_TWO1023 = 8.988465674311579539e307


def exp(x: float) -> float:
    """V8's ``ieee754::exp`` (fdlibm's ``e_exp.c``)."""
    hx = _high(x)
    xsb = (hx >> 31) & 1
    hx &= 0x7FFFFFFF
    hi = lo = 0.0
    k = 0
    if hx >= 0x40862E42:  # |x| >= 709.78...
        if hx >= 0x7FF00000:
            if ((hx & 0xFFFFF) | _low(x)) != 0:
                return x + x  # NaN
            return x if xsb == 0 else 0.0
        if x > _O_THRESHOLD:
            return _INF
        if x < _U_THRESHOLD:
            return 0.0
    if hx > 0x3FD62E42:  # |x| > 0.5 ln2
        if hx < 0x3FF0A2B2:  # and |x| < 1.5 ln2
            if x == 1.0:
                # V8 special-cases exp(1), which the reduction gets wrong.
                return _E
            hi = x - _LN2_HI[xsb]
            lo = _LN2_LO[xsb]
            k = 1 - xsb - xsb
        else:
            k = _trunc(_INVLN2 * x + _HALF[xsb])
            t = float(k)
            hi = x - t * _LN2_HI[0]
            lo = t * _LN2_LO[0]
        x = hi - lo
    elif hx < 0x3E300000:  # |x| < 2**-28
        return 1.0 + x
    else:
        k = 0
    t = x * x
    if k >= -1021:
        twopk = _words(0x3FF00000 + (k << 20), 0)
    else:
        twopk = _words(0x3FF00000 + ((k + 1000) << 20), 0)
    c = x - t * (_P1 + t * (_P2 + t * (_P3 + t * (_P4 + t * _P5))))
    if k == 0:
        return 1.0 - ((x * c) / (c - 2.0) - x)
    y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi)
    if k >= -1021:
        if k == 1024:
            return y * 2.0 * _TWO1023
        return y * twopk
    return y * twopk * _TWOM1000


_LG1 = 6.666666666666735130e-01
_LG2 = 3.999999999940941908e-01
_LG3 = 2.857142874366239149e-01
_LG4 = 2.222219843214978396e-01
_LG5 = 1.818357216161805012e-01
_LG6 = 1.531383769920937332e-01
_LG7 = 1.479819860511658591e-01
_LN2_HI_1 = 6.93147180369123816490e-01
_LN2_LO_1 = 1.90821492927058770002e-10
_TWO54 = 1.80143985094819840000e16


def log(x: float) -> float:
    """V8's ``ieee754::log`` (fdlibm's ``e_log.c``)."""
    hx = _high(x)
    lx = _low(x)
    k = 0
    if hx < 0x00100000:  # x < 2**-1022
        if ((hx & 0x7FFFFFFF) | lx) == 0:
            return -_INF
        if hx < 0:
            return _NAN
        k -= 54
        x *= _TWO54
        hx = _high(x)
    if hx >= 0x7FF00000:
        return x + x
    k += (hx >> 20) - 1023
    hx &= 0x000FFFFF
    i = (hx + 0x95F64) & 0x100000
    x = _words(hx | (i ^ 0x3FF00000), _low(x))
    k += i >> 20
    f = x - 1.0
    if (0x000FFFFF & (2 + hx)) < 3:  # -2**-20 <= f < 2**-20
        if f == 0.0:
            if k == 0:
                return 0.0
            dk = float(k)
            return dk * _LN2_HI_1 + dk * _LN2_LO_1
        r = f * f * (0.5 - 0.33333333333333333 * f)
        if k == 0:
            return f - r
        dk = float(k)
        return dk * _LN2_HI_1 - ((r - dk * _LN2_LO_1) - f)
    s = f / (2.0 + f)
    dk = float(k)
    z = s * s
    i = hx - 0x6147A
    w = z * z
    j = 0x6B851 - hx
    t1 = w * (_LG2 + w * (_LG4 + w * _LG6))
    t2 = z * (_LG1 + w * (_LG3 + w * (_LG5 + w * _LG7)))
    i |= j
    r = t2 + t1
    if i > 0:
        hfsq = 0.5 * f * f
        if k == 0:
            return f - (hfsq - s * (hfsq + r))
        return dk * _LN2_HI_1 - ((hfsq - (s * (hfsq + r) + dk * _LN2_LO_1)) - f)
    if k == 0:
        return f - s * (f - r)
    return dk * _LN2_HI_1 - ((s * (f - r) - dk * _LN2_LO_1) - f)


_BP = (1.0, 1.5)
_DP_H = (0.0, 5.84962487220764160156e-01)
_DP_L = (0.0, 1.35003920212974897128e-08)
_TWO53 = 9007199254740992.0
_L1 = 5.99999999999994648725e-01
_L2 = 4.28571428578550184252e-01
_L3 = 3.33333329818377432918e-01
_L4 = 2.72728123808534006489e-01
_L5 = 2.30660745775561754067e-01
_L6 = 2.06975017800338417784e-01
_LG2_FULL = 6.93147180559945286227e-01
_LG2_H = 6.93147182464599609375e-01
_LG2_L = -1.90465429995776804525e-09
_OVT = 8.0085662595372944372e-17
_CP = 9.61796693925975554329e-01
_CP_H = 9.61796700954437255859e-01
_CP_L = -7.02846165095275826516e-09
_IVLN2 = 1.44269504088896338700e00
_IVLN2_H = 1.44269502162933349609e00
_IVLN2_L = 1.92596299112661746887e-08


def pow(x: float, y: float) -> float:  # noqa: A001 - the name of the JavaScript function
    """``Math.pow``: V8's ``ieee754::pow``, fdlibm's ``e_pow.c`` with one change."""
    if y != y:
        return _NAN  # JavaScript: anything ** NaN is NaN, even 1 ** NaN.
    if abs(y) == _INF and abs(x) == 1.0:
        return _NAN  # JavaScript: (+-1) ** (+-Infinity) is NaN.
    x, y = float(x), float(y)
    hx, lx = _high(x), _low(x)
    hy, ly = _high(y), _low(y)
    ix = hx & 0x7FFFFFFF
    iy = hy & 0x7FFFFFFF

    if (iy | ly) == 0:
        return 1.0
    if ix > 0x7FF00000 or (ix == 0x7FF00000 and lx != 0) or iy > 0x7FF00000 or (
        iy == 0x7FF00000 and ly != 0
    ):
        return x + y

    # yisint: 0 if y isn't an integer, 1 if it's odd, 2 if even (for x < 0).
    yisint = 0
    if hx < 0:
        if iy >= 0x43400000:
            yisint = 2
        elif iy >= 0x3FF00000:
            k = (iy >> 20) - 0x3FF
            if k > 20:
                j = ly >> (52 - k)
                if ((j << (52 - k)) & 0xFFFFFFFF) == ly:
                    yisint = 2 - (j & 1)
            elif ly == 0:
                j = iy >> (20 - k)
                if (j << (20 - k)) == iy:
                    yisint = 2 - (j & 1)

    if ly == 0:
        if iy == 0x7FF00000:  # y is +-inf
            if ((ix - 0x3FF00000) | lx) == 0:
                return 1.0
            if ix >= 0x3FF00000:
                return y if hy >= 0 else 0.0
            return -y if hy < 0 else 0.0
        if iy == 0x3FF00000:  # y is +-1
            if hy >= 0:
                return x
            return 1.0 / x if x != 0 else math.copysign(_INF, x)
        if hy == 0x40000000:  # y is 2
            return x * x
        if hy == 0x3FE00000 and hx >= 0:  # y is 0.5
            return math.sqrt(x)

    ax = abs(x)
    if lx == 0 and (ix == 0x7FF00000 or ix == 0 or ix == 0x3FF00000):
        z = ax
        if hy < 0:
            z = 1.0 / z if z != 0 else _INF
        if hx < 0:
            if ((ix - 0x3FF00000) | yisint) == 0:
                z = _NAN
            elif yisint == 1:
                z = -z
        return z

    n = ((hx & 0xFFFFFFFF) >> 31) - 1
    if (n | yisint) == 0:
        return _NAN

    s = 1.0
    if (n | (yisint - 1)) == 0:
        s = -1.0

    if iy > 0x41E00000:  # |y| > 2**31
        if iy > 0x43F00000:  # |y| > 2**64
            if ix <= 0x3FEFFFFF:
                return _INF if hy < 0 else 0.0
            if ix >= 0x3FF00000:
                return _INF if hy > 0 else 0.0
        if ix < 0x3FEFFFFF:
            return s * _INF if hy < 0 else s * 0.0
        if ix > 0x3FF00000:
            return s * _INF if hy > 0 else s * 0.0
        t = ax - 1.0
        w = (t * t) * (0.5 - t * (3.3333333333333331e-01 - t * 0.25))
        u = _IVLN2_H * t
        v = t * _IVLN2_L - w * _IVLN2
        t1 = _clear_low(u + v)
        t2 = v - (t1 - u)
    else:
        n = 0
        if ix < 0x00100000:
            ax *= _TWO53
            n -= 53
            ix = _high(ax)
        n += (ix >> 20) - 0x3FF
        j = ix & 0x000FFFFF
        ix = j | 0x3FF00000
        if j <= 0x3988E:
            k = 0
        elif j < 0xBB67A:
            k = 1
        else:
            k = 0
            n += 1
            ix -= 0x00100000
        ax = _words(ix, _low(ax))

        u = ax - _BP[k]
        v = 1.0 / (ax + _BP[k])
        ss = u * v
        s_h = _clear_low(ss)
        t_h = _words(((ix >> 1) | 0x20000000) + 0x00080000 + (k << 18), 0)
        t_l = ax - (t_h - _BP[k])
        s_l = v * ((u - s_h * t_h) - s_h * t_l)
        s2 = ss * ss
        r = s2 * s2 * (_L1 + s2 * (_L2 + s2 * (_L3 + s2 * (_L4 + s2 * (_L5 + s2 * _L6)))))
        r += s_l * (s_h + ss)
        s2 = s_h * s_h
        t_h = _clear_low(3.0 + s2 + r)
        t_l = r - ((t_h - 3.0) - s2)
        u = s_h * t_h
        v = s_l * t_h + t_l * ss
        p_h = _clear_low(u + v)
        p_l = v - (p_h - u)
        z_h = _CP_H * p_h
        z_l = _CP_L * p_h + p_l * _CP + _DP_L[k]
        t = float(n)
        t1 = _clear_low(((z_h + z_l) + _DP_H[k]) + t)
        t2 = z_l - (((t1 - t) - _DP_H[k]) - z_h)

    y1 = _clear_low(y)
    p_l = (y - y1) * t1 + y * t2
    p_h = y1 * t1
    z = p_l + p_h
    j, i = _high(z), _low(z)
    if j >= 0x40900000:  # z >= 1024
        if ((j - 0x40900000) | i) != 0:
            return s * _INF
        if p_l + _OVT > z - p_h:
            return s * _INF
    elif (j & 0x7FFFFFFF) >= 0x4090CC00:  # z <= -1075
        if ((_int32(j - 0xC090CC00)) | i) != 0:
            return s * 0.0
        if p_l <= z - p_h:
            return s * 0.0

    i = j & 0x7FFFFFFF
    k = (i >> 20) - 0x3FF
    n = 0
    if i > 0x3FE00000:  # |z| > 0.5: n = [z + 0.5]
        n = _int32(j + (0x00100000 >> (k + 1)))
        k = ((n & 0x7FFFFFFF) >> 20) - 0x3FF
        t = _words(n & ~(0x000FFFFF >> k), 0)
        n = ((n & 0x000FFFFF) | 0x00100000) >> (20 - k)
        if j < 0:
            n = -n
        p_h -= t
    t = _clear_low(p_l + p_h)
    u = t * _LG2_H
    v = (p_l - (t - p_h)) * _LG2_FULL + t * _LG2_L
    z = u + v
    w = v - (z - u)
    t = z * z
    t1 = z - t * (_P1 + t * (_P2 + t * (_P3 + t * (_P4 + t * _P5))))
    # V8 divides by ``(t1 - 2) - (w + z*w)`` where fdlibm subtracts ``w + z*w``
    # from the quotient, which is why ``Math.pow(10, -4)`` is
    # 0.00009999999999999999.
    r = (z * t1) / ((t1 - 2.0) - (w + z * w))
    z = 1.0 - (r - z)
    j = _high(z) + (n << 20)
    if (j >> 20) <= 0:
        z = math.ldexp(z, n)  # a subnormal result
    else:
        z = _words(j, _low(z))
    return s * z


# ── Trigonometry ────────────────────────────────────────────────────────────

_S1 = -1.66666666666666324348e-01
_S2 = 8.33333333332248946124e-03
_S3 = -1.98412698298579493134e-04
_S4 = 2.75573137070700676789e-06
_S5 = -2.50507602534068634195e-08
_S6 = 1.58969099521155010221e-10


def _kernel_sin(x: float, y: float, iy: int) -> float:
    ix = _high(x) & 0x7FFFFFFF
    if ix < 0x3E400000 and _trunc(x) == 0:  # |x| < 2**-27
        return x
    z = x * x
    v = z * x
    r = _S2 + z * (_S3 + z * (_S4 + z * (_S5 + z * _S6)))
    if iy == 0:
        return x + v * (_S1 + z * r)
    return x - ((z * (0.5 * y - v * r) - y) - v * _S1)


_C1 = 4.16666666666666019037e-02
_C2 = -1.38888888888741095749e-03
_C3 = 2.48015872894767294178e-05
_C4 = -2.75573143513906633035e-07
_C5 = 2.08757232129817482790e-09
_C6 = -1.13596475577881948265e-11


def _kernel_cos(x: float, y: float) -> float:
    ix = _high(x) & 0x7FFFFFFF
    if ix < 0x3E400000 and _trunc(x) == 0:  # |x| < 2**-27
        return 1.0
    z = x * x
    r = z * (_C1 + z * (_C2 + z * (_C3 + z * (_C4 + z * (_C5 + z * _C6)))))
    if ix < 0x3FD33333:  # |x| < 0.3
        return 1.0 - (0.5 * z - (z * r - x * y))
    if ix > 0x3FE90000:  # |x| > 0.78125
        qx = 0.28125
    else:
        qx = _words(ix - 0x00200000, 0)  # x / 4
    iz = 0.5 * z - qx
    a = 1.0 - qx
    return a - (iz - (z * r - x * y))


_T = (
    3.33333333333334091986e-01,
    1.33333333333201242699e-01,
    5.39682539762260521377e-02,
    2.18694882948595424599e-02,
    8.86323982359930005737e-03,
    3.59207910759131235356e-03,
    1.45620945432529025516e-03,
    5.88041240820264096874e-04,
    2.46463134818469906812e-04,
    7.81794442939557092300e-05,
    7.14072491382608190305e-05,
    -1.85586374855275456654e-05,
    2.59073051863633712884e-05,
)
_PIO4 = 7.85398163397448278999e-01
_PIO4LO = 3.06161699786838301793e-17


def _kernel_tan(x: float, y: float, iy: int) -> float:
    hx = _high(x)
    ix = hx & 0x7FFFFFFF
    if ix < 0x3E300000 and _trunc(x) == 0:  # |x| < 2**-28
        if ((ix | _low(x)) | (iy + 1)) == 0:
            return _INF  # 1 / |x| for x = 0
        if iy == 1:
            return x
        w = x + y
        z = _clear_low(w)
        v = y - (z - x)
        a = -1.0 / w
        t = _clear_low(a)
        s = 1.0 + t * z
        return t + a * (s + t * v)
    if ix >= 0x3FE59428:  # |x| >= 0.6744
        if hx < 0:
            x = -x
            y = -y
        z = _PIO4 - x
        w = _PIO4LO - y
        x = z + w
        y = 0.0
    z = x * x
    w = z * z
    r = _T[1] + w * (_T[3] + w * (_T[5] + w * (_T[7] + w * (_T[9] + w * _T[11]))))
    v = z * (_T[2] + w * (_T[4] + w * (_T[6] + w * (_T[8] + w * (_T[10] + w * _T[12])))))
    s = z * x
    r = y + z * (s * (r + v) + y)
    r += _T[0] * s
    w = x + r
    if ix >= 0x3FE59428:
        v = float(iy)
        return float(1 - ((hx >> 30) & 2)) * (v - 2.0 * (x - (w * w / (w + v) - r)))
    if iy == 1:
        return w
    z = _clear_low(w)
    v = r - (z - x)
    a = -1.0 / w
    t = _clear_low(a)
    s = 1.0 + t * z
    return t + a * (s + t * v)


# The bits of 2/pi, 24 at a time, and pi/2 cut into 24-bit doubles.
_TWO_OVER_PI = (
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C, 0x439041,
    0xFE5163, 0xABDEBB, 0xC561B7, 0x246E3A, 0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C,
    0xFE1DEB, 0x1CB129, 0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41,
    0x3991D6, 0x398353, 0x39F49C, 0x845F8B, 0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F,
    0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF, 0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D,
    0x7527BA, 0xC7EBE5, 0xF17B3D, 0x0739F7, 0x8A5292, 0xEA6BFB, 0x5FB11F, 0x8D5D08,
    0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3, 0x91615E, 0xE61B08,
    0x659985, 0x5F14A0, 0x68408D, 0xFFD880, 0x4D7327, 0x310606, 0x1556CA, 0x73A8C9,
    0x60E27B, 0xC08C6B,
)
_PIO2 = (
    1.57079625129699707031e00,
    7.54978941586159635335e-08,
    5.39030252995776476554e-15,
    3.28200341580791294123e-22,
    1.27065575308067607349e-29,
    1.22933308981111328932e-36,
    2.73370053816464559624e-44,
    2.16741683877804819444e-51,
)
_TWO24 = 1.67772160000000000000e07
_TWON24 = 5.96046447753906250000e-08


def _kernel_rem_pio2(x: list[float], e0: int, nx: int) -> tuple[int, float, float]:
    """fdlibm's ``__kernel_rem_pio2`` for ``prec`` 2: ``x`` (24-bit pieces
    scaled by ``2**e0``) mod pi/2, as ``(n mod 8, y0, y1)``."""
    jk = jp = 4
    jx = nx - 1
    jv = max((e0 - 3) // 24, 0)
    q0 = e0 - 24 * (jv + 1)
    f = [0.0] * 20
    q = [0.0] * 20
    iq = [0] * 20
    fq = [0.0] * 20
    j = jv - jx
    m = jx + jk
    for i in range(m + 1):
        f[i] = 0.0 if j < 0 else float(_TWO_OVER_PI[j])
        j += 1
    for i in range(jk + 1):
        fw = 0.0
        for j in range(jx + 1):
            fw += x[j] * f[jx + i - j]
        q[i] = fw
    jz = jk
    while True:
        i, j, z = 0, jz, q[jz]
        while j > 0:
            fw = float(_trunc(_TWON24 * z))
            iq[i] = _trunc(z - _TWO24 * fw)
            z = q[j - 1] + fw
            i += 1
            j -= 1
        z = math.ldexp(z, q0)
        z -= 8.0 * math.floor(z * 0.125)
        n = _trunc(z)
        z -= float(n)
        ih = 0
        if q0 > 0:
            i = iq[jz - 1] >> (24 - q0)
            n += i
            iq[jz - 1] -= i << (24 - q0)
            ih = iq[jz - 1] >> (23 - q0)
        elif q0 == 0:
            ih = iq[jz - 1] >> 23
        elif z >= 0.5:
            ih = 2
        if ih > 0:
            n += 1
            carry = 0
            for i in range(jz):
                j = iq[i]
                if carry == 0:
                    if j != 0:
                        carry = 1
                        iq[i] = 0x1000000 - j
                else:
                    iq[i] = 0xFFFFFF - j
            if q0 == 1:
                iq[jz - 1] &= 0x7FFFFF
            elif q0 == 2:
                iq[jz - 1] &= 0x3FFFFF
            if ih == 2:
                z = 1.0 - z
                if carry != 0:
                    z -= math.ldexp(1.0, q0)
        if z == 0.0:
            j = 0
            for i in range(jz - 1, jk - 1, -1):
                j |= iq[i]
            if j == 0:  # recompute with more terms
                k = 1
                while iq[jk - k] == 0:
                    k += 1
                for i in range(jz + 1, jz + k + 1):
                    f[jx + i] = float(_TWO_OVER_PI[jv + i])
                    fw = 0.0
                    for j in range(jx + 1):
                        fw += x[j] * f[jx + i - j]
                    q[i] = fw
                jz += k
                continue
        break
    if z == 0.0:
        jz -= 1
        q0 -= 24
        while iq[jz] == 0:
            jz -= 1
            q0 -= 24
    else:
        z = math.ldexp(z, -q0)
        if z >= _TWO24:
            fw = float(_trunc(_TWON24 * z))
            iq[jz] = _trunc(z - _TWO24 * fw)
            jz += 1
            q0 += 24
            iq[jz] = _trunc(fw)
        else:
            iq[jz] = _trunc(z)
    fw = math.ldexp(1.0, q0)
    for i in range(jz, -1, -1):
        q[i] = fw * float(iq[i])
        fw *= _TWON24
    for i in range(jz, -1, -1):
        fw = 0.0
        k = 0
        while k <= jp and k <= jz - i:
            fw += _PIO2[k] * q[i + k]
            k += 1
        fq[jz - i] = fw
    fw = 0.0
    for i in range(jz, -1, -1):
        fw += fq[i]
    y0 = fw if ih == 0 else -fw
    fw = fq[0] - fw
    for i in range(1, jz + 1):
        fw += fq[i]
    y1 = fw if ih == 0 else -fw
    return n & 7, y0, y1


_NPIO2_HW = (
    0x3FF921FB, 0x400921FB, 0x4012D97C, 0x401921FB, 0x401F6A7A, 0x4022D97C, 0x4025FDBB,
    0x402921FB, 0x402C463A, 0x402F6A7A, 0x4031475C, 0x4032D97C, 0x40346B9C, 0x4035FDBB,
    0x40378FDB, 0x403921FB, 0x403AB41B, 0x403C463A, 0x403DD85A, 0x403F6A7A, 0x40407E4C,
    0x4041475C, 0x4042106C, 0x4042D97C, 0x4043A28C, 0x40446B9C, 0x404534AC, 0x4045FDBB,
    0x4046C6CB, 0x40478FDB, 0x404858EB, 0x404921FB,
)
_INVPIO2 = 6.36619772367581382433e-01
_PIO2_1 = 1.57079632673412561417e00
_PIO2_1T = 6.07710050650619224932e-11
_PIO2_2 = 6.07710050630396597660e-11
_PIO2_2T = 2.02226624879595063154e-21
_PIO2_3 = 2.02226624871116645580e-21
_PIO2_3T = 8.47842766036889956997e-32


def _rem_pio2(x: float) -> tuple[int, float, float]:
    """fdlibm's ``__ieee754_rem_pio2``: ``x`` mod pi/2 as ``(n, y0, y1)``."""
    hx = _high(x)
    ix = hx & 0x7FFFFFFF
    if ix <= 0x3FE921FB:
        return 0, x, 0.0
    if ix < 0x4002D97C:  # |x| < 3pi/4
        if hx > 0:
            z = x - _PIO2_1
            if ix != 0x3FF921FB:
                y0 = z - _PIO2_1T
                return 1, y0, (z - y0) - _PIO2_1T
            z -= _PIO2_2
            y0 = z - _PIO2_2T
            return 1, y0, (z - y0) - _PIO2_2T
        z = x + _PIO2_1
        if ix != 0x3FF921FB:
            y0 = z + _PIO2_1T
            return -1, y0, (z - y0) + _PIO2_1T
        z += _PIO2_2
        y0 = z + _PIO2_2T
        return -1, y0, (z - y0) + _PIO2_2T
    if ix <= 0x413921FB:  # |x| <= 2**19 * pi/2
        t = abs(x)
        n = _trunc(t * _INVPIO2 + 0.5)
        fn = float(n)
        r = t - fn * _PIO2_1
        w = fn * _PIO2_1T
        if n < 32 and ix != _NPIO2_HW[n - 1]:
            y0 = r - w
        else:
            j = ix >> 20
            y0 = r - w
            i = j - ((_high(y0) >> 20) & 0x7FF)
            if i > 16:
                t = r
                w = fn * _PIO2_2
                r = t - w
                w = fn * _PIO2_2T - ((t - r) - w)
                y0 = r - w
                i = j - ((_high(y0) >> 20) & 0x7FF)
                if i > 49:
                    t = r
                    w = fn * _PIO2_3
                    r = t - w
                    w = fn * _PIO2_3T - ((t - r) - w)
                    y0 = r - w
        y1 = (r - y0) - w
        if hx < 0:
            return -n, -y0, -y1
        return n, y0, y1
    if ix >= 0x7FF00000:
        return 0, x - x, x - x
    e0 = (ix >> 20) - 1046
    z = _words(ix - (e0 << 20), _low(x))
    tx = [0.0, 0.0, 0.0]
    for i in range(2):
        tx[i] = float(_trunc(z))
        z = (z - tx[i]) * _TWO24
    tx[2] = z
    nx = 3
    while tx[nx - 1] == 0.0:
        nx -= 1
    n, y0, y1 = _kernel_rem_pio2(tx, e0, nx)
    if hx < 0:
        return -n, -y0, -y1
    return n, y0, y1


def sin(x: float) -> float:
    """V8's ``ieee754::sin``."""
    ix = _high(x) & 0x7FFFFFFF
    if ix <= 0x3FE921FB:
        return _kernel_sin(x, 0.0, 0)
    if ix >= 0x7FF00000:
        return _NAN
    n, y0, y1 = _rem_pio2(x)
    n &= 3
    if n == 0:
        return _kernel_sin(y0, y1, 1)
    if n == 1:
        return _kernel_cos(y0, y1)
    if n == 2:
        return -_kernel_sin(y0, y1, 1)
    return -_kernel_cos(y0, y1)


def cos(x: float) -> float:
    """V8's ``ieee754::cos``."""
    ix = _high(x) & 0x7FFFFFFF
    if ix <= 0x3FE921FB:
        return _kernel_cos(x, 0.0)
    if ix >= 0x7FF00000:
        return _NAN
    n, y0, y1 = _rem_pio2(x)
    n &= 3
    if n == 0:
        return _kernel_cos(y0, y1)
    if n == 1:
        return -_kernel_sin(y0, y1, 1)
    if n == 2:
        return -_kernel_cos(y0, y1)
    return _kernel_sin(y0, y1, 1)


def tan(x: float) -> float:
    """V8's ``ieee754::tan``."""
    ix = _high(x) & 0x7FFFFFFF
    if ix <= 0x3FE921FB:
        return _kernel_tan(x, 0.0, 1)
    if ix >= 0x7FF00000:
        return _NAN
    n, y0, y1 = _rem_pio2(x)
    return _kernel_tan(y0, y1, 1 - ((n & 1) << 1))


_ATANHI = (
    4.63647609000806093515e-01,
    7.85398163397448278999e-01,
    9.82793723247329054082e-01,
    1.57079632679489655800e00,
)
_ATANLO = (
    2.26987774529616870924e-17,
    3.06161699786838301793e-17,
    1.39033110312309984516e-17,
    6.12323399573676603587e-17,
)
_AT = (
    3.33333333333329318027e-01,
    -1.99999999998764832476e-01,
    1.42857142725034663711e-01,
    -1.11111104054623557880e-01,
    9.09088713343650656196e-02,
    -7.69187620504482999495e-02,
    6.66107313738753120669e-02,
    -5.83357013379057348645e-02,
    4.97687799461593236017e-02,
    -3.65315727442169155270e-02,
    1.62858201153657823623e-02,
)


def atan(x: float) -> float:
    """V8's ``ieee754::atan``."""
    hx = _high(x)
    ix = hx & 0x7FFFFFFF
    if ix >= 0x44100000:  # |x| >= 2**66
        if ix > 0x7FF00000 or (ix == 0x7FF00000 and _low(x) != 0):
            return x + x
        if hx > 0:
            return _ATANHI[3] + _ATANLO[3]
        return -_ATANHI[3] - _ATANLO[3]
    if ix < 0x3FDC0000:  # |x| < 0.4375
        if ix < 0x3E400000:  # |x| < 2**-27
            return x
        ident = -1
    else:
        x = abs(x)
        if ix < 0x3FF30000:  # |x| < 1.1875
            if ix < 0x3FE60000:  # 7/16 <= |x| < 11/16
                ident = 0
                x = (2.0 * x - 1.0) / (2.0 + x)
            else:  # 11/16 <= |x| < 19/16
                ident = 1
                x = (x - 1.0) / (x + 1.0)
        elif ix < 0x40038000:  # |x| < 2.4375
            ident = 2
            x = (x - 1.5) / (1.0 + 1.5 * x)
        else:  # 2.4375 <= |x| < 2**66
            ident = 3
            x = -1.0 / x
    z = x * x
    w = z * z
    s1 = z * (_AT[0] + w * (_AT[2] + w * (_AT[4] + w * (_AT[6] + w * (_AT[8] + w * _AT[10])))))
    s2 = w * (_AT[1] + w * (_AT[3] + w * (_AT[5] + w * (_AT[7] + w * _AT[9]))))
    if ident < 0:
        return x - x * (s1 + s2)
    z = _ATANHI[ident] - ((x * (s1 + s2) - _ATANLO[ident]) - x)
    return -z if hx < 0 else z


_PI = 3.14159265358979311600e00
_PIO2_HI = 1.57079632679489655800e00
_PIO2_LO = 6.12323399573676603587e-17
_PIO4_HI = 7.85398163397448278999e-01
_PS0 = 1.66666666666666657415e-01
_PS1 = -3.25565818622400915405e-01
_PS2 = 2.01212532134862925881e-01
_PS3 = -4.00555345006794114027e-02
_PS4 = 7.91534994289814532176e-04
_PS5 = 3.47933107596021167570e-05
_QS1 = -2.40339491173441421878e00
_QS2 = 2.02094576023350569471e00
_QS3 = -6.88283971605453293030e-01
_QS4 = 7.70381505559019352791e-02


def _ratio(t: float) -> tuple[float, float]:
    p = t * (_PS0 + t * (_PS1 + t * (_PS2 + t * (_PS3 + t * (_PS4 + t * _PS5)))))
    q = 1.0 + t * (_QS1 + t * (_QS2 + t * (_QS3 + t * _QS4)))
    return p, q


def asin(x: float) -> float:
    """V8's ``ieee754::asin``."""
    hx = _high(x)
    ix = hx & 0x7FFFFFFF
    if ix >= 0x3FF00000:  # |x| >= 1
        if ((ix - 0x3FF00000) | _low(x)) == 0:
            return x * _PIO2_HI + x * _PIO2_LO
        return _NAN
    if ix < 0x3FE00000:  # |x| < 0.5
        if ix < 0x3E400000:  # |x| < 2**-27
            return x
        t = x * x
        p, q = _ratio(t)
        return x + x * (p / q)
    w = 1.0 - abs(x)
    t = w * 0.5
    p, q = _ratio(t)
    s = math.sqrt(t)
    if ix >= 0x3FEF3333:  # |x| > 0.975
        w = p / q
        t = _PIO2_HI - (2.0 * (s + s * w) - _PIO2_LO)
    else:
        w = _clear_low(s)
        c = (t - w * w) / (s + w)
        r = p / q
        p = 2.0 * s * r - (_PIO2_LO - 2.0 * c)
        q = _PIO4_HI - 2.0 * w
        t = _PIO4_HI - (p - q)
    return t if hx > 0 else -t


def acos(x: float) -> float:
    """V8's ``ieee754::acos``."""
    hx = _high(x)
    ix = hx & 0x7FFFFFFF
    if ix >= 0x3FF00000:  # |x| >= 1
        if ((ix - 0x3FF00000) | _low(x)) == 0:
            return 0.0 if hx > 0 else _PI + 2.0 * _PIO2_LO
        return _NAN
    if ix < 0x3FE00000:  # |x| < 0.5
        if ix <= 0x3C600000:  # |x| < 2**-57
            return _PIO2_HI + _PIO2_LO
        z = x * x
        p, q = _ratio(z)
        r = p / q
        return _PIO2_HI - (x - (_PIO2_LO - x * r))
    if hx < 0:  # x < -0.5
        z = (1.0 + x) * 0.5
        p, q = _ratio(z)
        s = math.sqrt(z)
        r = p / q
        w = r * s - _PIO2_LO
        return _PI - 2.0 * (s + w)
    z = (1.0 - x) * 0.5  # x > 0.5
    s = math.sqrt(z)
    df = _clear_low(s)
    c = (z - df * df) / (s + df)
    p, q = _ratio(z)
    r = p / q
    w = r * s + c
    return 2.0 * (df + w)
