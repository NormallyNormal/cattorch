"""Scoped public module adaptation for the experimental FX frontend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class AdapterContext:
    """Read-only context supplied while adapting an export-owned model copy."""

    module_path: str
    state_names: frozenset[str] = frozenset()


AdapterFactory = Callable[[torch.nn.Module, AdapterContext], torch.nn.Module]


@dataclass(frozen=True)
class ModuleAdapter:
    """Replace one third-party module type before FX capture.

    ``factory(module, context)`` receives a copy of each matching module and an
    ``AdapterContext`` with its path, and must return an ``nn.Module``. The
    caller's model is never modified. A module used at several paths is
    replaced once, and every path gets the same replacement.
    """

    module_type: type[torch.nn.Module]
    factory: AdapterFactory

    def __post_init__(self) -> None:
        if not isinstance(self.module_type, type) or not issubclass(
            self.module_type, torch.nn.Module,
        ):
            raise TypeError("module_type must be an nn.Module subclass")
        if not callable(self.factory):
            raise TypeError("factory must be callable")


__all__ = ["AdapterContext", "AdapterFactory", "ModuleAdapter"]
