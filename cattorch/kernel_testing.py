"""Check a kernel against PyTorch in the emulator (experimental)."""

from __future__ import annotations

import dataclasses
import tempfile
import warnings
from pathlib import Path

import torch
from torch.utils import _pytree as pytree

from cattorch.kernel_api import Kernel


class _Call(torch.nn.Module):
    """Call ``operation`` with a sample's tensors supplied as model inputs."""

    def __init__(self, operation, spec, constants, positions):
        super().__init__()
        self.operation = operation
        self.spec = spec
        self.constants = constants
        self.positions = positions

    def forward(self, *tensors):
        leaves = list(self.constants)
        for position, tensor in zip(self.positions, tensors):
            leaves[position] = tensor
        args, kwargs = pytree.tree_unflatten(leaves, self.spec)
        return self.operation(*args, **kwargs)


def check_kernel(
    kernel: Kernel,
    samples,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    static: tuple[str, ...] = (),
):
    """Export each sample with ``kernel`` and compare it with PyTorch.

    ``samples`` is an iterable of argument tuples for the kernel's operation,
    or ``(args, kwargs)`` pairs, such as ``(torch.randn(2, 3),)`` or
    ``((torch.randn(4), 2), {"dim": 0})``. Tensor arguments become Scratch
    inputs, except the ones named in ``static`` (schema argument names, such
    as ``"weight"``), which are stored in the sprite as weights. An
    approximate kernel is checked in fast mode.

    Raises ``AssertionError`` naming the first sample that doesn't match, or
    whose export didn't run the kernel. Returns each sample's ``verify``
    result.
    """
    from cattorch.transpiler import transpile
    from cattorch.verify import verify

    operator = kernel.target
    if not isinstance(operator, torch._ops.OpOverload):
        from cattorch.kernel_api import resolve_operator

        operator = resolve_operator(operator)
        if operator is None:
            raise ValueError(f"cannot call kernel target {kernel.target_name!r}")
    names = [argument.name for argument in operator._schema.arguments]
    optimization = "fast" if kernel.approximate else "exact"
    emitted = []

    def emit(context):
        emitted.append(context)
        return kernel.emit(context)

    # The same kernel, recording that the export used it.
    probe = dataclasses.replace(kernel, emit=emit)
    results = []
    with tempfile.TemporaryDirectory() as directory:
        for index, sample in enumerate(samples):
            args, kwargs = _split(sample)
            model, inputs = _module(operator, args, kwargs, names, static)
            path = Path(directory) / f"sample_{index}"
            emitted.clear()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                result = transpile(
                    model, inputs, path, optimization=optimization, kernels=(probe,),
                )
            if not emitted:
                raise AssertionError(
                    f"sample {index}: the export did not use the kernel for "
                    f"{kernel.target_name}; check that the sample calls it"
                )
            report = verify(model, inputs, result, atol=atol, rtol=rtol)
            if not report.passed:
                raise AssertionError(f"sample {index} doesn't match PyTorch: {report}")
            results.append(report)
    if not results:
        raise ValueError("check_kernel needs at least one sample")
    return results


def _split(sample):
    if (
        isinstance(sample, tuple) and len(sample) == 2
        and isinstance(sample[0], (tuple, list)) and isinstance(sample[1], dict)
    ):
        return tuple(sample[0]), dict(sample[1])
    if isinstance(sample, torch.Tensor):
        return (sample,), {}
    return tuple(sample), {}


def _module(operator, args, kwargs, names, static):
    unknown = set(static) - set(names)
    if unknown:
        raise ValueError(f"static names unknown arguments: {sorted(unknown)}")
    named = dict(zip(names, args))
    named.update(kwargs)
    leaves, spec = pytree.tree_flatten((args, kwargs))
    static_tensors = {id(named[name]) for name in static if isinstance(named.get(name), torch.Tensor)}
    positions = [
        index for index, leaf in enumerate(leaves)
        if isinstance(leaf, torch.Tensor) and id(leaf) not in static_tensors
    ]
    constants = [None if index in positions else leaf for index, leaf in enumerate(leaves)]
    model = _Call(operator, spec, constants, positions).eval()
    inputs = tuple(leaves[index].detach().clone() for index in positions)
    if not inputs:
        raise ValueError("a sample needs at least one tensor input that isn't static")
    return model, inputs
