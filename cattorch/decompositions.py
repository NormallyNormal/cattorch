"""Decompositions of operations into ones cattorch has kernels for.

PyTorch's own decompositions of these end in primitives Scratch can't run, or
don't exist. Each function is traced in place of its operation, so it may only
call ATen operations cattorch lowers (or other operations in this table).
"""

from __future__ import annotations

import math

import torch

_aten = torch.ops.aten


def traced_from_input(value) -> bool | None:
    """Whether a tensor being traced depends on a model input.

    ``None`` when the tracer doesn't know the tensor at all.
    """
    import torch.fx.experimental.proxy_tensor as proxy_tensor

    mode = proxy_tensor.get_proxy_mode()
    if mode is None:
        return None
    slot = proxy_tensor.get_proxy_slot(value, mode.tracer, None)
    proxy = getattr(slot, "proxy", slot)
    if not isinstance(proxy, torch.fx.Proxy):
        return None
    seen, stack = set(), [proxy.node]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if node.op == "placeholder":
            return True
        stack.extend(node.all_input_nodes)
    return False


def _sign(value):
    positive = _aten.gt.Scalar(value, 0)
    negative = _aten.lt.Scalar(value, 0)
    result = _aten.where.ScalarSelf(positive, 1, _aten.where.ScalarSelf(
        negative, -1, _aten.zeros_like.default(value),
    ))
    return _aten.to.dtype(result, value.dtype)


def _trunc(value):
    if not value.is_floating_point():
        return value
    return _aten.where.self(
        _aten.lt.Scalar(value, 0), _aten.ceil.default(value), _aten.floor.default(value),
    )


def _round(value):
    """Round half to even, as PyTorch does; Scratch's round goes half up."""
    if not value.is_floating_point():
        return value
    floor = _aten.floor.default(value)
    fraction = _aten.sub.Tensor(value, floor)
    # 1 when ``floor`` is odd, so ``floor + odd`` is the even neighbour.
    odd = _aten.sub.Tensor(floor, _aten.mul.Tensor(
        _aten.floor.default(_aten.mul.Tensor(floor, 0.5)), 2,
    ))
    up = _aten.add.Tensor(floor, 1)
    return _aten.where.self(
        _aten.gt.Scalar(fraction, 0.5), up,
        _aten.where.self(
            _aten.lt.Scalar(fraction, 0.5), floor, _aten.add.Tensor(floor, odd),
        ),
    )


def _divide(left, right, rounding_mode=None):
    quotient = _aten.div.Tensor(left, right)
    if rounding_mode == "floor":
        quotient = _aten.floor.default(quotient)
    elif rounding_mode == "trunc":
        quotient = _trunc(quotient)
    elif rounding_mode is not None:
        return NotImplemented
    dtype = torch.result_type(left, right)
    if rounding_mode is not None and not dtype.is_floating_point:
        quotient = _aten.to.dtype(quotient, dtype)
    return quotient


def _remainder(left, right):
    """Python's ``%``: the result takes the divisor's sign."""
    quotient = _divide(left, right, "floor")
    return _aten.sub.Tensor(left, _aten.mul.Tensor(quotient, right))


def _fmod(left, right):
    """C's ``fmod``: the result takes the dividend's sign."""
    quotient = _divide(left, right, "trunc")
    return _aten.sub.Tensor(left, _aten.mul.Tensor(quotient, right))


def _boolean(operation):
    """``&``, ``|`` and ``^`` on boolean masks; integer bitwise stays unsupported."""
    def decomposition(left, right):
        boolean = (
            isinstance(left, torch.Tensor) and left.dtype == torch.bool
            and (not isinstance(right, torch.Tensor) or right.dtype == torch.bool)
            and (isinstance(right, torch.Tensor) or isinstance(right, bool))
        )
        if not boolean:
            return NotImplemented
        if not isinstance(right, torch.Tensor):
            right = _aten.full_like.default(left, right)
        if operation == "and":
            return _aten.logical_and.default(left, right)
        if operation == "or":
            return _aten.logical_or.default(left, right)
        return _aten.ne.Tensor(left, right)
    return decomposition


def _logical(operation):
    def decomposition(left, right):
        left = _aten.ne.Scalar(left, 0)
        right = _aten.ne.Scalar(right, 0)
        if operation == "and":
            return _aten.where.self(left, right, _aten.zeros_like.default(left))
        if operation == "or":
            return _aten.where.self(left, _aten.ones_like.default(left), right)
        return _aten.ne.Tensor(left, right)
    return decomposition


def _full_like(value, fill, *, dtype=None, layout=None, device=None,
               pin_memory=None, memory_format=None):
    return _aten.full.default(
        list(value.shape), fill, dtype=dtype or value.dtype, device=value.device,
    )


def _isnan(value):
    return _aten.ne.Tensor(value, value)


def _nan_like(value):
    return _aten.full_like.default(value, math.nan)


def _float(value):
    return value if value.is_floating_point() else _aten.to.dtype(value, torch.float32)


def _sinh(value):
    value = _float(value)
    return _aten.mul.Tensor(_aten.sub.Tensor(
        _aten.exp.default(value), _aten.exp.default(_aten.neg.default(value)),
    ), 0.5)


def _cosh(value):
    value = _float(value)
    return _aten.mul.Tensor(_aten.add.Tensor(
        _aten.exp.default(value), _aten.exp.default(_aten.neg.default(value)),
    ), 0.5)


def _asinh(value):
    """``sign(x) ln(|x| + sqrt(x² + 1))``, which keeps precision for x < 0."""
    value = _float(value)
    magnitude = _aten.abs.default(value)
    root = _aten.sqrt.default(_aten.add.Tensor(_aten.mul.Tensor(value, value), 1))
    result = _aten.log.default(_aten.add.Tensor(magnitude, root))
    return _aten.where.self(_aten.lt.Scalar(value, 0), _aten.neg.default(result), result)


def _acosh(value):
    value = _float(value)
    root = _aten.sqrt.default(_aten.sub.Tensor(_aten.mul.Tensor(value, value), 1))
    return _aten.log.default(_aten.add.Tensor(value, root))


def _atanh(value):
    value = _float(value)
    ratio = _aten.div.Tensor(_aten.add.Tensor(value, 1), _aten.rsub.Scalar(value, 1))
    return _aten.mul.Tensor(_aten.log.default(ratio), 0.5)


def _atan2(y, x):
    y, x = _float(y), _float(x)
    # x = 0 gives ±infinity, whose arctangent is ±pi/2.
    base = _aten.atan.default(_aten.div.Tensor(y, x))
    shift = _aten.where.ScalarSelf(_aten.lt.Scalar(y, 0), -math.pi, _aten.full_like.default(base, math.pi))
    result = _aten.where.self(_aten.lt.Scalar(x, 0), _aten.add.Tensor(base, shift), base)
    origin = _aten.logical_and.default(_aten.eq.Scalar(x, 0), _aten.eq.Scalar(y, 0))
    return _aten.where.ScalarSelf(origin, 0.0, result)


def _hypot(left, right):
    return _aten.sqrt.default(_aten.add.Tensor(
        _aten.mul.Tensor(left, left), _aten.mul.Tensor(right, right),
    ))


def _nan_ignoring(extreme):
    """``fmax``/``fmin``: like ``maximum``, but a NaN loses to a number."""
    def decomposition(left, right):
        result = extreme(left, right)
        result = _aten.where.self(_isnan(left), right, result)
        return _aten.where.self(_isnan(right), left, result)
    return decomposition


def _isfinite(value):
    if not value.is_floating_point():
        return _aten.ones_like.default(value, dtype=torch.bool)
    return _aten.logical_and.default(
        _aten.eq.Tensor(value, value),
        _aten.ne.Scalar(_aten.abs.default(value), math.inf),
    )


def _copysign(left, right):
    magnitude = _aten.abs.default(_float(left))
    if not isinstance(right, torch.Tensor):
        return magnitude if math.copysign(1, right) > 0 else _aten.neg.default(magnitude)
    return _aten.where.self(_aten.lt.Scalar(right, 0), _aten.neg.default(magnitude), magnitude)


def _logaddexp(left, right):
    """``max + log(1 + exp(-|a - b|))``; equal infinities give that infinity."""
    largest = _aten.maximum.default(left, right)
    gap = _aten.abs.default(_aten.sub.Tensor(left, right))
    result = _aten.add.Tensor(largest, _aten.log.default(
        _aten.add.Tensor(_aten.exp.default(_aten.neg.default(gap)), 1),
    ))
    return _aten.where.self(_aten.eq.Tensor(left, right), _aten.add.Tensor(
        largest, math.log(2),
    ), result)


def _logaddexp2(left, right):
    scaled = _logaddexp(_aten.mul.Tensor(left, math.log(2)), _aten.mul.Tensor(right, math.log(2)))
    return _aten.div.Tensor(scaled, math.log(2))


def _logsumexp(value, dims, keepdim=False):
    dims = list(dims) if dims else list(range(value.ndim))
    value = _float(value)
    largest = _aten.amax.default(value, dims, True)
    # An all-infinite slice would give inf - inf.
    largest = _aten.where.ScalarSelf(_aten.eq.Scalar(_aten.abs.default(largest), math.inf), 0.0, largest)
    total = _aten.sum.dim_IntList(_aten.exp.default(_aten.sub.Tensor(value, largest)), dims, True)
    result = _aten.add.Tensor(_aten.log.default(total), largest)
    if not keepdim:
        for dim in sorted((dim % max(value.ndim, 1) for dim in dims), reverse=True):
            result = _aten.squeeze.dim(result, dim)
    return result


def _logcumsumexp(value, dim):
    value = _float(value)
    largest = _aten.amax.default(value, [dim], True)
    largest = _aten.where.ScalarSelf(_aten.eq.Scalar(_aten.abs.default(largest), math.inf), 0.0, largest)
    total = _aten.cumsum.default(_aten.exp.default(_aten.sub.Tensor(value, largest)), dim)
    return _aten.add.Tensor(_aten.log.default(total), largest)


def _power(base, exponent):
    """``base ** exponent`` with a tensor exponent, through ``exp(y ln |x|)``."""
    base, exponent = _float(base), _float(exponent)
    magnitude = _aten.exp.default(_aten.mul.Tensor(
        exponent, _aten.log.default(_aten.abs.default(base)),
    ))
    integral = _aten.eq.Tensor(exponent, _aten.floor.default(exponent))
    odd = _aten.eq.Scalar(_remainder(exponent, 2), 1)
    negative = _aten.where.self(
        integral,
        _aten.where.self(odd, _aten.neg.default(magnitude), magnitude),
        _nan_like(magnitude),
    )
    result = _aten.where.self(_aten.lt.Scalar(base, 0), negative, magnitude)
    # 0 ** 0 is 1, where the logarithm gives NaN.
    return _aten.where.ScalarSelf(_aten.eq.Scalar(exponent, 0), 1.0, result)


def _scalar_power(base, exponent):
    """``number ** tensor``."""
    if base > 0:
        return _aten.exp.default(_aten.mul.Tensor(_float(exponent), math.log(base)))
    return _power(_aten.full_like.default(_float(exponent), base), exponent)


def _round_decimals(value, *, decimals=0):
    scale = 10.0 ** decimals
    return _aten.div.Tensor(_round(_aten.mul.Tensor(value, scale)), scale)


def _identity(value):
    return _aten.alias.default(value)


def _angle(value):
    """A real number's angle is pi when negative and 0 otherwise."""
    value = _float(value)
    result = _aten.where.ScalarSelf(_aten.lt.Scalar(value, 0), math.pi, _aten.zeros_like.default(value))
    return _aten.where.self(_isnan(value), value, result)


def _addbmm(bias, left, right, *, beta=1, alpha=1):
    product = _aten.sum.dim_IntList(_aten.bmm.default(left, right), [0])
    if alpha != 1:
        product = _aten.mul.Tensor(product, alpha)
    if beta == 0:
        return _aten.add.Tensor(product, _aten.mul.Tensor(_aten.zeros_like.default(bias), 0))
    return _aten.add.Tensor(product, bias if beta == 1 else _aten.mul.Tensor(bias, beta))


def _distance(differences, p):
    """The ``p``-norm over the last dimension."""
    magnitude = _aten.abs.default(differences)
    if p == 0:
        return _aten.sum.dim_IntList(_aten.to.dtype(_aten.ne.Scalar(magnitude, 0), magnitude.dtype), [-1])
    if p == math.inf:
        return _aten.amax.default(magnitude, [-1])
    if p == 1:
        return _aten.sum.dim_IntList(magnitude, [-1])
    if p == 2:
        return _aten.sqrt.default(_aten.sum.dim_IntList(_aten.mul.Tensor(magnitude, magnitude), [-1]))
    total = _aten.sum.dim_IntList(_aten.pow.Tensor_Scalar(magnitude, p), [-1])
    return _aten.pow.Tensor_Scalar(total, 1 / p)


def _cdist(left, right, p, compute_mode=None):
    differences = _aten.sub.Tensor(_aten.unsqueeze.default(left, -2), _aten.unsqueeze.default(right, -3))
    return _distance(differences, p)


def _pdist(value, p=2):
    rows = value.shape[0]
    first, second = zip(*((i, j) for i in range(rows) for j in range(i + 1, rows))) if rows > 1 else ((), ())
    if not first:
        return _aten.sum.dim_IntList(value[:0], [-1])
    first = torch.tensor(first, dtype=torch.long, device=value.device)
    second = torch.tensor(second, dtype=torch.long, device=value.device)
    differences = _aten.sub.Tensor(
        _aten.index_select.default(value, 0, first), _aten.index_select.default(value, 0, second),
    )
    return _distance(differences, p)


def _bilinear(left, right, weight, bias=None):
    """``y[..., o] = sum_ij left[..., i] weight[o, i, j] right[..., j] + bias[o]``."""
    outputs, inner, last = weight.shape
    rows = _aten.reshape.default(left, [-1, inner])
    folded = _aten.reshape.default(_aten.permute.default(weight, [1, 0, 2]), [inner, outputs * last])
    mixed = _aten.reshape.default(_aten.mm.default(rows, folded), [-1, outputs, last])
    columns = _aten.unsqueeze.default(_aten.reshape.default(right, [-1, last]), 1)
    result = _aten.sum.dim_IntList(_aten.mul.Tensor(mixed, columns), [-1])
    if bias is not None:
        result = _aten.add.Tensor(result, bias)
    return _aten.reshape.default(result, [*left.shape[:-1], outputs])


def _fill(value, fill):
    return _aten.full_like.default(value, fill)


def _per_dimension(value, dimensions):
    value = list(value) if isinstance(value, (list, tuple)) else [value]
    return value * dimensions if len(value) == 1 else value


def _convolution(value, weight, bias, stride, padding, dilation, transposed,
                 output_padding, groups):
    """Route every convolution to the 1D and 2D kernels."""
    dimensions = weight.ndim - 2
    stride, padding, dilation, output_padding = (
        _per_dimension(item, dimensions)
        for item in (stride, padding, dilation, output_padding)
    )
    if transposed:
        return _transposed_convolution(
            value, weight, bias, stride, padding, dilation, output_padding, groups,
        )
    if dimensions == 1:
        return _aten.conv1d.default(value, weight, bias, stride, padding, dilation, groups)
    if dimensions == 2:
        return _aten.conv2d.default(value, weight, bias, stride, padding, dilation, groups)
    if dimensions == 3:
        return _convolution3d(value, weight, bias, stride, padding, dilation, groups)
    return NotImplemented


def _convolution3d(value, weight, bias, stride, padding, dilation, groups):
    """One 2D convolution per kernel depth, with output depths as the batch."""
    unbatched = value.ndim == 4
    if unbatched:
        value = _aten.unsqueeze.default(value, 0)
    if padding[0]:
        value = _aten.constant_pad_nd.default(value, [0, 0, 0, 0, padding[0], padding[0]])
    batches, channels, depth, height, width = value.shape
    kernel_depth = weight.shape[2]
    output_depth = (depth - dilation[0] * (kernel_depth - 1) - 1) // stride[0] + 1
    total = None
    for offset in range(kernel_depth):
        rows = torch.tensor(
            [row * stride[0] + offset * dilation[0] for row in range(output_depth)],
            dtype=torch.long, device=value.device,
        )
        slab = _aten.index_select.default(value, 2, rows)
        slab = _aten.reshape.default(
            _aten.permute.default(slab, [0, 2, 1, 3, 4]),
            [batches * output_depth, channels, height, width],
        )
        part = _aten.conv2d.default(
            slab, _aten.select.int(weight, 2, offset), None,
            stride[1:], padding[1:], dilation[1:], groups,
        )
        total = part if total is None else _aten.add.Tensor(total, part)
    outputs, rows_out, columns_out = total.shape[1:]
    result = _aten.permute.default(
        _aten.reshape.default(total, [batches, output_depth, outputs, rows_out, columns_out]),
        [0, 2, 1, 3, 4],
    )
    if bias is not None:
        result = _aten.add.Tensor(result, _aten.reshape.default(bias, [1, -1, 1, 1, 1]))
    return _aten.squeeze.dim(result, 0) if unbatched else result


def _transposed_convolution(value, weight, bias, stride, padding, dilation,
                            output_padding, groups):
    """Spread the input by ``stride``, pad, and convolve with the flipped kernel."""
    dimensions = weight.ndim - 2
    unbatched = value.ndim == dimensions + 1
    if unbatched:
        value = _aten.unsqueeze.default(value, 0)
    for index in range(dimensions):
        axis = 2 + index
        if stride[index] > 1:
            # Zeros after every element, then drop the trailing ones.
            length = value.shape[axis]
            spread = _aten.unsqueeze.default(value, axis + 1)
            shape = list(spread.shape)
            shape[axis + 1] = stride[index] - 1
            zeros = _aten.zeros.default(shape, dtype=value.dtype, device=value.device)
            spread = _aten.cat.default([spread, zeros], axis + 1)
            merged = list(value.shape)
            merged[axis] = length * stride[index]
            value = _aten.slice.Tensor(
                _aten.reshape.default(spread, merged), axis, 0,
                (length - 1) * stride[index] + 1,
            )
        kernel = weight.shape[2 + index]
        before = dilation[index] * (kernel - 1) - padding[index]
        after = before + output_padding[index]
        if before < 0 or after < 0:
            # More padding than the kernel reaches: crop instead.
            start, stop = max(-before, 0), value.shape[axis] - max(-after, 0)
            value = _aten.slice.Tensor(value, axis, start, stop)
            before, after = max(before, 0), max(after, 0)
        if before or after:
            pad = [0, 0] * (dimensions - 1 - index) + [before, after]
            value = _aten.constant_pad_nd.default(value, pad)
    # (in, out / groups, k...) to (out, in / groups, k...), flipped.
    inputs, outputs_per_group = weight.shape[:2]
    spatial = list(weight.shape[2:])
    grouped = _aten.reshape.default(
        weight, [groups, inputs // groups, outputs_per_group, *spatial],
    )
    swapped = _aten.reshape.default(
        _aten.transpose.int(grouped, 1, 2),
        [groups * outputs_per_group, inputs // groups, *spatial],
    )
    flipped = _aten.flip.default(swapped, list(range(2, 2 + dimensions)))
    result = _convolution(
        value, flipped, bias, [1] * dimensions, [0] * dimensions, dilation,
        False, [0] * dimensions, groups,
    )
    return _aten.squeeze.dim(result, 0) if unbatched else result


def _string_padding(dimensions):
    """``padding="same"`` or ``"valid"`` made explicit."""
    def decomposition(value, weight, bias=None, stride=1, padding="valid", dilation=1, groups=1):
        stride, dilation = _per_dimension(stride, dimensions), _per_dimension(dilation, dimensions)
        if padding == "valid":
            explicit = [0] * dimensions
        else:
            explicit = []
            extra = []
            for index in range(dimensions):
                total = dilation[index] * (weight.shape[2 + index] - 1)
                explicit.append(total // 2)
                extra.append(total - 2 * (total // 2))
            if any(extra):
                # An odd total puts the extra column on the right, like PyTorch.
                pad = []
                for index in reversed(range(dimensions)):
                    pad += [0, extra[index]]
                value = _aten.constant_pad_nd.default(value, pad)
        return _convolution(
            value, weight, bias, stride, explicit, dilation, False,
            [0] * dimensions, groups,
        )
    return decomposition


def _window_extremes(value, dimensions, windows):
    """Max over each window, and the flat input index where it was found.

    ``windows`` holds one list of ``(positions, extent)`` per spatial
    dimension. The windows are gathered through an export-time table; reads of
    padding hit an appended ``-inf``, so they never win.
    """
    import itertools

    spatial = list(value.shape[-dimensions:])
    leading = list(value.shape[:-dimensions])
    size = math.prod(spatial)
    strides = [math.prod(spatial[index + 1:]) for index in range(dimensions)]
    rows = []
    for window in itertools.product(*windows):
        row = []
        for position in itertools.product(*(positions for positions, _ in window)):
            inside = all(0 <= at < extent for at, extent in zip(position, spatial))
            row.append(sum(a * b for a, b in zip(position, strides)) if inside else size)
        rows.append(row)
    width = max(len(row) for row in rows)
    table = torch.tensor(
        [row + [size] * (width - len(row)) for row in rows], dtype=torch.long, device=value.device,
    )
    flat = _aten.reshape.default(value, [*leading, size])
    padding = _aten.full.default([*leading, 1], -math.inf, dtype=value.dtype, device=value.device)
    flat = _aten.cat.default([flat, padding], -1)
    gathered = _aten.reshape.default(
        _aten.index_select.default(flat, -1, _aten.reshape.default(table, [-1])),
        [*leading, len(rows), width],
    )
    values = _aten.amax.default(gathered, [-1])
    choice = _aten.argmax.default(gathered, -1, True)
    positions = _aten.expand.default(table, [*leading, len(rows), width])
    indices = _aten.squeeze.dim(_aten.gather.default(positions, -1, choice), -1)
    shape = [*leading, *(len(axis) for axis in windows)]
    return _aten.reshape.default(values, shape), _aten.reshape.default(indices, shape)


_MAX_POOL = {1: _aten.max_pool1d.default, 2: _aten.max_pool2d.default}


def _max_pool_with_indices(dimensions):
    def decomposition(value, kernel_size, stride=(), padding=0, dilation=1, ceil_mode=False):
        from cattorch.util.instruction.optimized_convolution import pool_windows

        kernel = _per_dimension(kernel_size, dimensions)
        stride = _per_dimension(stride, dimensions) if stride else kernel
        padding = _per_dimension(padding, dimensions)
        dilation = _per_dimension(dilation, dimensions)
        spatial = value.shape[-dimensions:]
        windows = [
            pool_windows(spatial[index], kernel[index], stride[index], padding[index],
                         dilation[index], ceil_mode)
            for index in range(dimensions)
        ]
        values, indices = _window_extremes(value, dimensions, windows)
        if dimensions in _MAX_POOL:
            # The pooling kernel is far cheaper than gathering the windows;
            # the gather stays only if the indices are used.
            values = _MAX_POOL[dimensions](value, kernel, stride, padding, dilation, ceil_mode)
        return values, indices
    return decomposition


def _adaptive_max_pool(dimensions):
    def decomposition(value, output_size):
        from cattorch.util.instruction.optimized_convolution import adaptive_windows

        output_size = _per_dimension(output_size, dimensions)
        spatial = value.shape[-dimensions:]
        windows = [
            adaptive_windows(spatial[index], output_size[index]) for index in range(dimensions)
        ]
        values, indices = _window_extremes(value, dimensions, windows)
        if all(size == 1 for size in output_size):
            # Global pooling is one reduction.
            values = _aten.amax.default(value, list(range(-dimensions, 0)), True)
        elif dimensions in _MAX_POOL and all(
            length % size == 0 for length, size in zip(spatial, output_size)
        ):
            # Even windows: ordinary max pooling, as for global pooling.
            kernel = [length // size for length, size in zip(spatial, output_size)]
            values = _MAX_POOL[dimensions](
                value, kernel, kernel, [0] * dimensions, [1] * dimensions, False,
            )
        return values, indices
    return decomposition


def _pool_depth_then_planes(value, depth_pool, plane_pool):
    """Pool a 5D tensor over depth, then height and width.

    Average pooling's divisor is a product of per-dimension extents or
    counts, so pooling one dimension at a time gives the same result.
    """
    unbatched = value.ndim == 4
    if unbatched:
        value = _aten.unsqueeze.default(value, 0)
    batches, channels, depth, height, width = value.shape
    along_depth = _aten.reshape.default(
        _aten.permute.default(value, [0, 1, 3, 4, 2]), [batches, channels * height * width, depth],
    )
    pooled = depth_pool(along_depth)
    depth = pooled.shape[-1]
    planes = _aten.reshape.default(
        _aten.permute.default(
            _aten.reshape.default(pooled, [batches, channels, height, width, depth]), [0, 1, 4, 2, 3],
        ),
        [batches, channels * depth, height, width],
    )
    pooled = plane_pool(planes)
    result = _aten.reshape.default(pooled, [batches, channels, depth, *pooled.shape[-2:]])
    return _aten.squeeze.dim(result, 0) if unbatched else result


def _avg_pool3d(value, kernel_size, stride=(), padding=0, ceil_mode=False,
                count_include_pad=True, divisor_override=None):
    kernel = _per_dimension(kernel_size, 3)
    stride = _per_dimension(stride, 3) if stride else kernel
    padding = _per_dimension(padding, 3)
    # A fixed divisor divides the window's sum, so pool sums (divisor 1).
    summed = 1 if divisor_override is not None else None

    def along_depth(part):
        rows = _aten.unsqueeze.default(part, -2)
        pooled = _aten.avg_pool2d.default(
            rows, [1, kernel[0]], [1, stride[0]], [0, padding[0]],
            ceil_mode, count_include_pad, summed,
        )
        return _aten.squeeze.dim(pooled, -2)

    result = _pool_depth_then_planes(
        value, along_depth,
        lambda part: _aten.avg_pool2d.default(
            part, kernel[1:], stride[1:], padding[1:], ceil_mode, count_include_pad, summed,
        ),
    )
    if divisor_override is not None:
        result = _aten.div.Tensor(result, divisor_override)
    return result


def _adaptive_avg_pool3d(value, output_size):
    output_size = _per_dimension(output_size, 3)
    return _pool_depth_then_planes(
        value,
        lambda part: _aten.adaptive_avg_pool1d.default(part, output_size[:1]),
        lambda part: _aten.adaptive_avg_pool2d.default(part, output_size[1:]),
    )


def _positive_index(index, size):
    """Negative indices count from the end."""
    return _aten.where.self(_aten.lt.Scalar(index, 0), _aten.add.Tensor(index, size), index)


def _along(index, dim, shape):
    """A 1D ``index`` laid along ``dim`` and expanded to ``shape``."""
    view = [1] * len(shape)
    view[dim] = index.numel()
    return _aten.expand.default(_aten.reshape.default(index, view), list(shape))


def _replaced(shape, dim, size):
    shape = list(shape)
    shape[dim] = size
    return shape


def _as_vector(decomposition):
    """Run a decomposition on a 0-d tensor as a 1-element vector."""
    def wrapped(value, dim, *args, **kwargs):
        if value.ndim != 0:
            return decomposition(value, dim, *args, **kwargs)
        vectors = [
            _aten.reshape.default(arg, [1])
            if isinstance(arg, torch.Tensor) and arg.ndim == 0 and arg.is_floating_point()
            else arg
            for arg in args
        ]
        result = decomposition(_aten.reshape.default(value, [1]), 0, *vectors, **kwargs)
        if result is NotImplemented:
            return result
        return _aten.reshape.default(result, [])
    return wrapped


def _index_select(value, dim, index):
    """``index_select`` with input-dependent indices: a runtime ``gather``."""
    if not traced_from_input(index):
        return NotImplemented
    dim %= value.ndim
    index = _positive_index(_aten.reshape.default(index, [-1]), value.shape[dim])
    return _aten.gather.default(
        value, dim, _along(index, dim, _replaced(value.shape, dim, index.numel())),
    )


def _take(value, index):
    flat = _aten.reshape.default(value, [-1])
    positions = _positive_index(_aten.reshape.default(index, [-1]), flat.numel())
    return _aten.reshape.default(_aten.gather.default(flat, 0, positions), list(index.shape))


def _scatter_along(update):
    """``index_add``, ``index_copy`` and ``index_reduce`` as scatters."""
    def decomposition(value, dim, index, source, *args, **kwargs):
        dim %= value.ndim
        index = _positive_index(_aten.reshape.default(index, [-1]), value.shape[dim])
        return update(value, dim, _along(index, dim, source.shape), source, *args, **kwargs)
    return _as_vector(decomposition)


def _index_add(value, dim, index, source, *, alpha=1):
    if alpha != 1:
        source = _aten.mul.Tensor(source, alpha)
    return _aten.scatter_add.default(value, dim, index, source)


def _index_reduce(value, dim, index, source, reduce, *, include_self=True):
    return _aten.scatter_reduce.two(value, dim, index, source, reduce, include_self=include_self)


@_as_vector
def _index_fill(value, dim, index, fill):
    dim %= value.ndim
    index = _positive_index(_aten.reshape.default(index, [-1]), value.shape[dim])
    expanded = _along(index, dim, _replaced(value.shape, dim, index.numel()))
    if isinstance(fill, torch.Tensor):
        source = _aten.expand.default(_aten.to.dtype(fill, value.dtype), list(expanded.shape))
        return _aten.scatter.src(value, dim, expanded, source)
    return _aten.scatter.value(value, dim, expanded, fill)


def _advanced_index(value, indices):
    """Lay out integer advanced indexing as a gather from a 2D view.

    The indexed dimensions are moved first and flattened, so one linear index
    per element picks a row. Returns the view, the linear index, the
    indexed result's broadcast shape, the remaining dimensions' sizes, and
    where the broadcast dimensions go in the result.
    """
    indices = list(indices) + [None] * (value.ndim - len(indices))
    indexed = [dim for dim, index in enumerate(indices) if index is not None]
    rest = [dim for dim in range(value.ndim) if dim not in indexed]
    broadcast = list(torch.broadcast_shapes(*(indices[dim].shape for dim in indexed)))
    linear, stride = None, 1
    for dim in reversed(indexed):
        term = _positive_index(indices[dim], value.shape[dim])
        if stride != 1:
            term = _aten.mul.Tensor(term, stride)
        linear = term if linear is None else _aten.add.Tensor(linear, term)
        stride *= value.shape[dim]
    linear = _aten.expand.default(linear, broadcast)
    rest_shape = [value.shape[dim] for dim in rest]
    rows = _aten.reshape.default(_aten.permute.default(value, indexed + rest), [stride, *rest_shape])
    # PyTorch puts the broadcast dimensions where the indexed ones were if
    # they are adjacent, else first.
    adjacent = indexed == list(range(indexed[0], indexed[-1] + 1))
    front = indexed[0] if adjacent else 0
    return rows, linear, broadcast, rest_shape, front, indexed + rest


def _rows_index(linear, rest_shape):
    count = linear.numel()
    view = _aten.reshape.default(linear, [count, *([1] * len(rest_shape))])
    return _aten.expand.default(view, [count, *rest_shape])


def _result_order(broadcast, rest_shape, front):
    """Permutation from ``[*broadcast, *rest]`` to the indexing result's order."""
    count, rests = len(broadcast), len(rest_shape)
    return (
        [count + position for position in range(front)]
        + list(range(count))
        + [count + position for position in range(front, rests)]
    )


def _integer_indices(indices):
    present = [index for index in indices if index is not None]
    return present and all(
        not index.is_floating_point() and index.dtype != torch.bool for index in present
    )


def _index(value, indices):
    """Tensor indexing with input-dependent integer indices."""
    if not _integer_indices(indices) or not any(
        traced_from_input(index) for index in indices if index is not None
    ):
        return NotImplemented
    rows, linear, broadcast, rest_shape, front, _ = _advanced_index(value, indices)
    picked = _aten.gather.default(rows, 0, _rows_index(linear, rest_shape))
    picked = _aten.reshape.default(picked, [*broadcast, *rest_shape])
    return _aten.permute.default(picked, _result_order(broadcast, rest_shape, front))


def _index_put(value, indices, values, accumulate=False):
    present = [index for index in indices if index is not None]
    if len(present) == 1 and present[0].dtype == torch.bool and indices[0] is not None:
        # ``x[mask] = v`` with the same values for every selected element: a
        # select, whatever the mask holds.
        mask = present[0]
        trailing = list(value.shape[mask.ndim:])
        if tuple(mask.shape) != tuple(value.shape[:mask.ndim]) or values.ndim > len(trailing):
            return NotImplemented
        try:
            torch.broadcast_shapes(tuple(values.shape), tuple(trailing))
        except RuntimeError:
            return NotImplemented
        mask = _aten.expand.default(
            _aten.reshape.default(mask, [*mask.shape, *([1] * len(trailing))]),
            list(value.shape),
        )
        fill = _aten.expand.default(_aten.to.dtype(values, value.dtype), list(value.shape))
        if accumulate:
            return _aten.add.Tensor(value, _aten.where.self(mask, fill, _aten.zeros_like.default(value)))
        return _aten.where.self(mask, fill, value)
    if not _integer_indices(indices):
        return NotImplemented
    rows, linear, broadcast, rest_shape, front, order = _advanced_index(value, indices)
    result_shape = [*rest_shape[:front], *broadcast, *rest_shape[front:]]
    values = _aten.expand.default(_aten.to.dtype(values, value.dtype), result_shape)
    # Back from the result's order to ``[*broadcast, *rest]``.
    forward = _result_order(broadcast, rest_shape, front)
    inverse = [forward.index(position) for position in range(len(forward))]
    values = _aten.reshape.default(
        _aten.permute.default(values, inverse), [linear.numel(), *rest_shape],
    )
    positions = _rows_index(linear, rest_shape)
    if accumulate:
        rows = _aten.scatter_add.default(rows, 0, positions, values)
    else:
        rows = _aten.scatter.src(rows, 0, positions, values)
    permuted = [value.shape[dim] for dim in order]
    restore = [order.index(dim) for dim in range(value.ndim)]
    return _aten.permute.default(_aten.reshape.default(rows, permuted), restore)


def _put(value, index, source, accumulate=False):
    flat = _aten.reshape.default(value, [-1])
    updated = _index_put(
        flat, [_aten.reshape.default(index, [-1])], _aten.reshape.default(source, [-1]), accumulate,
    )
    return _aten.reshape.default(updated, list(value.shape))


def _sort(value, dim=-1, descending=False, *, stable=False):
    """A full-length ``topk``, whose ties keep their order like a stable sort."""
    if value.ndim == 0:
        return value, _aten.zeros_like.default(value, dtype=torch.long)
    dim %= value.ndim
    count = value.shape[dim]
    return (
        torch.ops.cattorch.topk_values(value, count, dim, descending),
        torch.ops.cattorch.topk_indices(value, count, dim, descending),
    )


def _stable_sort(value, *, stable=None, dim=-1, descending=False):
    return _sort(value, dim, descending)


def _argsort(value, dim=-1, descending=False, *, stable=False):
    return _sort(value, dim, descending)[1]


def _kthvalue(value, k, dim=-1, keepdim=False):
    if value.ndim == 0:
        return value, _aten.zeros_like.default(value, dtype=torch.long)
    values, indices = _sort(value, dim)
    dim %= value.ndim
    values = _aten.slice.Tensor(values, dim, k - 1, k)
    indices = _aten.slice.Tensor(indices, dim, k - 1, k)
    if not keepdim:
        values, indices = _aten.squeeze.dim(values, dim), _aten.squeeze.dim(indices, dim)
    return values, indices


def _median_dim(value, dim=-1, keepdim=False):
    """The lower median, as in PyTorch."""
    count = value.shape[dim % value.ndim] if value.ndim else 1
    return _kthvalue(value, (count + 1) // 2, dim, keepdim)


def _median(value):
    flat = _aten.reshape.default(value, [-1])
    return _median_dim(flat, 0)[0]


def _cumulative_extreme(largest):
    """``cummax``/``cummin`` through a triangular mask; PyTorch reports the
    latest position of the running extreme."""
    def decomposition(value, dim):
        if value.ndim == 0:
            return value, _aten.zeros_like.default(value, dtype=torch.long)
        dim %= value.ndim
        count = value.shape[dim]
        moved = _aten.movedim.int(value, dim, -1)
        earlier = torch.tensor(
            [[column <= row for column in range(count)] for row in range(count)],
            dtype=torch.bool, device=value.device,
        )
        rows = _aten.unsqueeze.default(moved, -2)
        blocked = -math.inf if largest else math.inf
        masked = _aten.where.self(earlier, rows, _aten.full_like.default(rows, blocked))
        extreme = (_aten.amax if largest else _aten.amin).default(masked, [-1], True)
        positions = torch.arange(count, device=value.device)
        found = _aten.logical_and.default(earlier, _aten.eq.Tensor(masked, extreme))
        latest = _aten.amax.default(_aten.where.self(
            found, _aten.expand.default(positions, list(found.shape)),
            _aten.full_like.default(found, -1, dtype=torch.long),
        ), [-1])
        return (
            _aten.movedim.int(_aten.squeeze.dim(extreme, -1), -1, dim),
            _aten.movedim.int(latest, -1, dim),
        )
    return decomposition


def _isin(elements, test_elements, *, assume_unique=False, invert=False):
    if not isinstance(elements, torch.Tensor):
        return NotImplemented
    tests = test_elements if isinstance(test_elements, torch.Tensor) else torch.tensor(
        test_elements, device=elements.device,
    )
    found = _aten.any.dim(
        _aten.eq.Tensor(_aten.unsqueeze.default(elements, -1), _aten.reshape.default(tests, [-1])),
        -1,
    )
    return _aten.logical_not.default(found) if invert else found


POINTWISE_DECOMPOSITIONS = {
    _aten.sign.default: _sign,
    _aten.sgn.default: _sign,
    _aten.signbit.default: lambda value: _aten.lt.Scalar(value, 0),
    _aten.trunc.default: _trunc,
    _aten.fix.default: _trunc,
    _aten.round.default: _round,
    _aten.round.decimals: _round_decimals,
    _aten.div.Tensor_mode: _divide,
    _aten.div.Scalar_mode: _divide,
    _aten.floor_divide.default: lambda left, right: _divide(left, right, "floor"),
    _aten.floor_divide.Scalar: lambda left, right: _divide(left, right, "floor"),
    _aten.remainder.Tensor: _remainder,
    _aten.remainder.Scalar: _remainder,
    _aten.fmod.Tensor: _fmod,
    _aten.fmod.Scalar: _fmod,
    _aten.bitwise_and.Tensor: _boolean("and"),
    _aten.bitwise_and.Scalar: _boolean("and"),
    _aten.bitwise_or.Tensor: _boolean("or"),
    _aten.bitwise_or.Scalar: _boolean("or"),
    _aten.bitwise_xor.Tensor: _boolean("xor"),
    _aten.bitwise_xor.Scalar: _boolean("xor"),
    _aten.logical_and.default: _logical("and"),
    _aten.logical_or.default: _logical("or"),
    _aten.logical_xor.default: _logical("xor"),
    _aten.full_like.default: _full_like,
    _aten.fill.Scalar: _fill,
    _aten.sinh.default: _sinh,
    _aten.cosh.default: _cosh,
    _aten.asinh.default: _asinh,
    _aten.acosh.default: _acosh,
    _aten.atanh.default: _atanh,
    _aten.atan2.default: _atan2,
    _aten.hypot.default: _hypot,
    _aten.fmax.default: _nan_ignoring(_aten.maximum.default),
    _aten.fmin.default: _nan_ignoring(_aten.minimum.default),
    _aten.isfinite.default: _isfinite,
    _aten.copysign.Tensor: _copysign,
    _aten.copysign.Scalar: _copysign,
    _aten.logaddexp.default: _logaddexp,
    _aten.logaddexp2.default: _logaddexp2,
    _aten.logsumexp.default: _logsumexp,
    _aten.logcumsumexp.default: _logcumsumexp,
    _aten.pow.Tensor_Tensor: _power,
    _aten.pow.Scalar: _scalar_power,
    _aten.float_power.Tensor_Tensor: _power,
    _aten.real.default: _identity,
    _aten.conj_physical.default: _identity,
    _aten.angle.default: _angle,
    _aten.addbmm.default: _addbmm,
    _aten._cdist_forward.default: _cdist,
    _aten.cdist.default: _cdist,
    _aten._pdist_forward.default: _pdist,
    _aten.pdist.default: _pdist,
    _aten.bilinear.default: _bilinear,
    _aten.convolution.default: _convolution,
    _aten.conv1d.padding: _string_padding(1),
    _aten.conv2d.padding: _string_padding(2),
    _aten.conv3d.padding: _string_padding(3),
    _aten.max_pool2d_with_indices.default: _max_pool_with_indices(2),
    _aten.max_pool3d_with_indices.default: _max_pool_with_indices(3),
    _aten.adaptive_max_pool1d.default: _adaptive_max_pool(1),
    _aten.adaptive_max_pool2d.default: _adaptive_max_pool(2),
    _aten.adaptive_max_pool3d.default: _adaptive_max_pool(3),
    _aten.avg_pool3d.default: _avg_pool3d,
    _aten._adaptive_avg_pool3d.default: _adaptive_avg_pool3d,
    _aten.adaptive_avg_pool3d.default: _adaptive_avg_pool3d,
    _aten.index_select.default: _as_vector(_index_select),
    _aten.index.Tensor: _index,
    _aten.take.default: _take,
    _aten.index_add.default: _scatter_along(_index_add),
    _aten.index_copy.default: _scatter_along(_aten.scatter.src),
    _aten.index_reduce.default: _scatter_along(_index_reduce),
    _aten.index_fill.int_Scalar: _index_fill,
    _aten.index_fill.int_Tensor: _index_fill,
    _aten.index_put.default: _index_put,
    _aten.put.default: _put,
    _aten.sort.default: _sort,
    _aten.sort.stable: _stable_sort,
    _aten.argsort.default: _argsort,
    _aten.argsort.stable: lambda value, *, stable, dim=-1, descending=False: _sort(
        value, dim, descending,
    )[1],
    _aten.msort.default: lambda value: _sort(value, 0)[0],
    _aten.kthvalue.default: _kthvalue,
    _aten.median.dim: _median_dim,
    _aten.median.default: _median,
    _aten.cummax.default: _cumulative_extreme(True),
    _aten.cummin.default: _cumulative_extreme(False),
    _aten.isin.Tensor_Tensor: _isin,
    _aten.isin.Tensor_Scalar: _isin,
}
