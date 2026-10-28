"""Declare new operations for export (experimental).

``op`` turns an eager function into a PyTorch custom operation. During export
the operation appears in the graph whole, so a ``Kernel`` can lower it; without
one it is traced through its eager function and exported from the operations
that function uses, like any decomposed operation.
"""

from __future__ import annotations

import inspect
import re
import types
import typing
from typing import Callable

import torch

from cattorch.kernel_api import Kernel

# Declared operations by overload, for the capture frontend.
DECLARED: dict[torch._ops.OpOverload, "Operation"] = {}


class Operation:
    """An operation declared with ``op``.

    Call it like the eager function. ``name`` is its qualified
    ``namespace::name``, ``operator`` the ``OpOverload`` that appears in
    exported graphs, and ``eager`` the function itself, which ``verify``
    compares the sprite against. Attach a lowering with :meth:`kernel`.
    """

    def __init__(self, name: str, eager: Callable, definition) -> None:
        self.name = name
        self.eager = eager
        self._definition = definition
        namespace, operation = name.split("::")
        self.operator = getattr(getattr(torch.ops, namespace), operation).default
        self.kernels: list[Kernel] = []
        self.__doc__ = eager.__doc__
        self.__name__ = getattr(eager, "__name__", operation)

    def __call__(self, *args, **kwargs):
        return self._definition(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<cattorch operation {self.name}>"

    def kernel(
        self,
        emit: Callable | None = None,
        *,
        name: str | None = None,
        validator: Callable | None = None,
        approximate: bool = False,
    ):
        """Attach a Scratch lowering; usable as ``@operation.kernel``.

        An attached kernel is used whenever the operation is exported, so
        ``transpile`` needs no ``kernels=`` argument for it. An approximate
        one applies only in fast mode; exact exports then trace the eager
        function instead. Returns the ``Kernel``.
        """
        def attach(emit: Callable) -> Kernel:
            from cattorch.operator_registry import default_registry

            kernel = Kernel(
                self.operator, emit, name=name, validator=validator, approximate=approximate,
            )
            self.kernels.append(kernel)
            default_registry().register_kernel(kernel)
            return kernel

        return attach if emit is None else attach(emit)


def _default_name(function: Callable) -> str:
    module = re.sub(r"\W", "_", getattr(function, "__module__", "") or "main")
    return f"cattorch_user::{module}_{function.__name__}".replace("__", "_")


def _fresh(result, args):
    """A custom operation may not return one of its inputs or a view of one."""
    inputs = [arg for arg in args if isinstance(arg, torch.Tensor)]

    def fresh(value):
        if isinstance(value, torch.Tensor) and any(
            torch._C._is_alias_of(value, arg) for arg in inputs
        ):
            return value.clone()
        return value

    if isinstance(result, tuple):
        return tuple(fresh(value) for value in result)
    return fresh(result)


def _typing_form(hint):
    """``hint`` with ``list[float]`` and ``X | None`` in their ``typing`` forms.

    Older PyTorch infers schemas only from ``typing.List`` and
    ``typing.Optional``, not from builtin generics or ``|`` unions.
    """
    origin, arguments = typing.get_origin(hint), typing.get_args(hint)
    if origin is None or not arguments:
        return hint
    arguments = tuple(_typing_form(argument) for argument in arguments)
    if origin in {list, typing.List}:
        return typing.List[arguments[0]]
    if origin in {tuple, typing.Tuple}:
        return typing.Tuple[arguments]
    if origin in {types.UnionType, typing.Union}:
        return typing.Union[arguments]
    return hint


def op(function: Callable | None = None, *, name: str | None = None, fake: Callable | None = None):
    """Declare ``function`` as a new operation; usable as ``@op`` or ``@op(...)``.

    ``function`` needs type annotations, which become the operation's schema:
    ``torch.Tensor`` arguments, numbers, booleans, and lists of them, returning
    a tensor or a tuple of tensors. It must not modify its inputs.

    ``name`` is the qualified ``namespace::name``; the default is derived
    from the function's module and name. ``fake`` computes output shapes for
    tracing; the default runs ``function`` itself on shape-only tensors, which
    works when its shapes don't depend on tensor values.
    """
    def declare(function: Callable) -> Operation:
        qualified = name or _default_name(function)
        if "::" not in qualified:
            raise ValueError(f"operation name must be namespace::name, got {qualified!r}")

        def eager(*args, **kwargs):
            return _fresh(function(*args, **kwargs), args)

        # The schema comes from the annotations, resolved in the function's
        # own module (they may be strings).
        try:
            hints = {
                key: _typing_form(hint) for key, hint in typing.get_type_hints(function).items()
            }
        except NameError as error:
            raise TypeError(f"cannot resolve the annotations of {qualified}: {error}") from error
        signature = inspect.signature(function)
        eager.__signature__ = signature.replace(  # type: ignore[attr-defined]
            parameters=[
                parameter.replace(annotation=hints.get(parameter.name, parameter.annotation))
                for parameter in signature.parameters.values()
            ],
            return_annotation=hints.get("return", signature.return_annotation),
        )
        eager.__name__ = function.__name__
        eager.__annotations__ = hints
        try:
            definition = torch.library.custom_op(qualified, mutates_args=())(eager)
        except (ValueError, TypeError) as error:
            raise TypeError(
                f"cannot declare {qualified}: {error}; annotate every argument and "
                "the return type"
            ) from error
        shape = fake or eager
        definition.register_fake(lambda *args, **kwargs: _fresh(shape(*args, **kwargs), args))
        operation = Operation(qualified, function, definition)
        DECLARED[operation.operator] = operation
        return operation

    return declare if function is None else declare(function)


def declared_without_kernel(fast_kernels: frozenset[str]):
    """Declared operations to trace through their eager function.

    These are the ones without an exact kernel in the default registry and
    without a kernel active for this export (``fast_kernels``).
    """
    from cattorch.operator_registry import default_registry

    registry = default_registry()
    for operator, operation in DECLARED.items():
        rule = registry.operation(str(operator))
        if (rule is None or rule.exact_kernel is None) and str(operator) not in fast_kernels:
            yield operator, operation
