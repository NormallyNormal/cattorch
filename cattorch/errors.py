"""Public exception types raised while exporting models to Scratch."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


class CattorchError(Exception):
    """Base class for cattorch-specific failures."""


class UnsupportedModelError(CattorchError, ValueError):
    """The model can't be exported as it is, or doesn't do what its declaration says.

    Raised for what cattorch finds by running or tracing the model: an
    unsupported structure or interface, a requirement of the export kind the
    model doesn't meet (such as a ``GenerationProgram``'s causal attention),
    or a marker or program that doesn't match the model. Invalid arguments
    raise ``ValueError`` or ``TypeError`` instead, before the model runs.
    """


class UnsupportedOperationError(CattorchError, NotImplementedError):
    """An operation in the traced model has no Scratch lowering.

    ``operation`` names it, ``module_path`` is the submodule it came from
    (``None`` at the top level), and ``detail`` says why, when known.
    """

    def __init__(
        self,
        operation: str,
        *,
        node_name: str | None = None,
        module_path: str | None = None,
        detail: str | None = None,
    ):
        self.operation = operation
        self.node_name = node_name
        self.module_path = module_path
        self.detail = detail
        where = f" in module {module_path!r}" if module_path else ""
        parts = [f"unsupported operation {operation}{where}"]
        if detail:
            parts.append(detail)
        parts.append(
            "supported operations: "
            "https://github.com/NormallyNormal/cattorch/blob/main/"
            "docs/supported-models.md#operations"
        )
        super().__init__("; ".join(parts))


class CattorchWarning(UserWarning):
    """Base class of the warnings cattorch emits.

    An export also returns its warnings in its result's ``warnings``, so
    ``warnings.simplefilter("ignore", CattorchWarning)`` loses nothing.
    """


# Messages warned during the export running in this context, which the
# public export functions add to their result's ``warnings``.
_COLLECTED: ContextVar[list[str] | None] = ContextVar("cattorch_warnings", default=None)


def warn(message: str, category: type[Warning] = CattorchWarning, *, stacklevel: int = 2) -> None:
    """Emit ``message`` as a warning, and record it for the running export's result."""
    warnings.warn(message, category, stacklevel=stacklevel + 1)
    collected = _COLLECTED.get()
    if collected is not None and message not in collected:
        collected.append(message)


@contextmanager
def collect_warnings() -> Iterator[list[str]]:
    """Record the messages ``warn`` emits inside the block, in order.

    Nested blocks share the outermost record, so a check that runs another
    export reports to the export that runs it.
    """
    outer = _COLLECTED.get()
    if outer is not None:
        yield outer
        return
    collected: list[str] = []
    token = _COLLECTED.set(collected)
    try:
        yield collected
    finally:
        _COLLECTED.reset(token)


def with_warnings(result, collected: list[str], *, stacklevel: int = 2):
    """``result`` with every export warning in ``result.warnings``, each emitted once.

    Messages already in ``result.warnings`` that ``warn`` didn't emit are
    emitted now, as ``CattorchWarning``; messages ``warn`` emitted are
    added to the result.
    """
    for message in result.warnings:
        if message not in collected:
            warnings.warn(message, CattorchWarning, stacklevel=stacklevel + 1)
    merged = tuple(dict.fromkeys((*result.warnings, *collected)))
    if merged == tuple(result.warnings):
        return result
    return dataclasses.replace(result, warnings=merged)
