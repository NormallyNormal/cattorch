"""Public exception types raised while exporting models to Scratch."""

from __future__ import annotations


class CattorchError(Exception):
    """Base class for cattorch-specific failures."""


class UnsupportedModelError(CattorchError, ValueError):
    """The exported model has an unsupported interface or semantic pattern."""


class UnsupportedOperationError(CattorchError, NotImplementedError):
    """An exported PyTorch operation has no safe Scratch lowering."""

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
        parts = [f"Unsupported operation: {operation}"]
        if module_path:
            parts.append(f"module: {module_path}")
        if node_name:
            parts.append(f"exported node: {node_name}")
        if detail:
            parts.append(detail)
        parts.append(
            "supported operations: "
            "https://github.com/NormallyNormal/cattorch/blob/main/"
            "docs/supported-models.md#operations"
        )
        super().__init__("; ".join(parts))
