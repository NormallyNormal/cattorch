"""Experimental multi-entrypoint and persistent-state export contracts.

The objects in this module describe the public Scratch interface without
coupling it to FX placeholder names.  They are intentionally small immutable
values so an export can validate the complete interface before tracing or
writing an artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

import torch


def _name(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class Input:
    """One tensor argument supplied through a public Scratch list."""

    name: str
    example: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "input name"))
        if not isinstance(self.example, torch.Tensor):
            raise TypeError("Input.example must be a torch.Tensor")


@dataclass(frozen=True)
class State:
    """Persistent tensor storage shared by program entrypoints.

    Append state grows only along its leading dimension.  ``capacity`` counts
    leading-dimension items, not flattened Scratch list values.
    """

    name: str
    initial: torch.Tensor
    mode: Literal["replace", "append"] = "replace"
    capacity: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "state name"))
        if not isinstance(self.initial, torch.Tensor):
            raise TypeError("State.initial must be a torch.Tensor")
        if self.mode not in {"replace", "append"}:
            raise ValueError("State.mode must be 'replace' or 'append'")
        if self.mode == "append":
            if self.initial.ndim < 1:
                raise ValueError("append state must have a leading dimension")
            if (
                isinstance(self.capacity, bool)
                or not isinstance(self.capacity, int)
                or self.capacity < self.initial.shape[0]
            ):
                raise ValueError(
                    "append state capacity must be an integer no smaller than "
                    "its initial leading dimension"
                )
        elif self.capacity is not None:
            raise ValueError("capacity is only valid for append state")


@dataclass(frozen=True)
class StateInput:
    """Reference a declared state tensor as an entrypoint argument."""

    name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "state input name"))


@dataclass(frozen=True)
class Output:
    """Publish one flattened tensor return leaf through a Scratch list."""

    name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "output name"))


@dataclass(frozen=True)
class StateUpdate:
    """Commit one tensor return leaf to persistent state after computation."""

    name: str
    mode: Literal["replace", "append"] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "state update name"))
        if self.mode not in {None, "replace", "append"}:
            raise ValueError("StateUpdate.mode must be 'replace', 'append', or None")


ArgumentBinding = Input | StateInput
ReturnBinding = Output | StateUpdate


@dataclass(frozen=True)
class EntryPoint:
    """One callable model method and its explicit Scratch interface."""

    name: str
    method: str = "forward"
    arguments: tuple[ArgumentBinding, ...] = ()
    returns: tuple[ReturnBinding, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "entrypoint name"))
        if self.name in {"init", "reset", "prepare for save"}:
            raise ValueError(f"entrypoint name {self.name!r} is reserved for lifecycle procedures")
        if re.search(r"%[snb]", self.name):
            raise ValueError("entrypoint names cannot contain Scratch argument placeholders")
        object.__setattr__(self, "method", _name(self.method, "entrypoint method"))
        object.__setattr__(self, "arguments", tuple(self.arguments))
        object.__setattr__(self, "returns", tuple(self.returns))
        if not self.arguments:
            raise ValueError("an entrypoint must declare at least one argument")
        if not self.returns:
            raise ValueError("an entrypoint must declare at least one return binding")
        if any(not isinstance(value, (Input, StateInput)) for value in self.arguments):
            raise TypeError("entrypoint arguments must be Input or StateInput values")
        if any(not isinstance(value, (Output, StateUpdate)) for value in self.returns):
            raise TypeError("entrypoint returns must be Output or StateUpdate values")
        input_names = [value.name for value in self.arguments if isinstance(value, Input)]
        output_names = [value.name for value in self.returns if isinstance(value, Output)]
        if len(input_names) != len(set(input_names)):
            raise ValueError(f"entrypoint {self.name!r} has duplicate input names")
        if len(output_names) != len(set(output_names)):
            raise ValueError(f"entrypoint {self.name!r} has duplicate output names")
        updates = [value.name for value in self.returns if isinstance(value, StateUpdate)]
        if len(updates) != len(set(updates)):
            raise ValueError(f"entrypoint {self.name!r} has duplicate state updates")


@dataclass(frozen=True)
class ExportProgram:
    """Complete callable and persistent-state interface for one sprite."""

    entrypoints: tuple[EntryPoint, ...]
    states: tuple[State, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entrypoints", tuple(self.entrypoints))
        object.__setattr__(self, "states", tuple(self.states))
        if not self.entrypoints:
            raise ValueError("ExportProgram requires at least one entrypoint")
        if any(not isinstance(value, EntryPoint) for value in self.entrypoints):
            raise TypeError("ExportProgram.entrypoints must contain EntryPoint values")
        if any(not isinstance(value, State) for value in self.states):
            raise TypeError("ExportProgram.states must contain State values")
        entry_names = [value.name for value in self.entrypoints]
        state_names = [value.name for value in self.states]
        if len(entry_names) != len(set(entry_names)):
            raise ValueError("entrypoint names must be unique")
        if len(state_names) != len(set(state_names)):
            raise ValueError("state names must be unique")
        states = {value.name: value for value in self.states}
        public_names = {f"cattorch state {state.name}": ("state", state.name) for state in self.states}
        for entrypoint in self.entrypoints:
            for binding in (*entrypoint.arguments, *entrypoint.returns):
                if not isinstance(binding, (Input, Output)):
                    continue
                name = f"cattorch {entrypoint.name} {binding.name}"
                owner = ("entrypoint", entrypoint.name)
                if name in public_names and public_names[name] != owner:
                    raise ValueError(f"public list name collision: {name!r}")
                public_names[name] = owner
            for argument in entrypoint.arguments:
                if isinstance(argument, StateInput) and argument.name not in states:
                    raise ValueError(
                        f"entrypoint {entrypoint.name!r} references unknown state "
                        f"{argument.name!r}"
                    )
            for returned in entrypoint.returns:
                if not isinstance(returned, StateUpdate):
                    continue
                state = states.get(returned.name)
                if state is None:
                    raise ValueError(
                        f"entrypoint {entrypoint.name!r} updates unknown state "
                        f"{returned.name!r}"
                    )
                if returned.mode is not None and returned.mode != state.mode:
                    raise ValueError(
                        f"state update mode for {returned.name!r} does not match "
                        f"the declared {state.mode!r} state mode"
                    )
        for name in public_names:
            if re.search(r" shard [2-9][0-9]*$| shard 1[0-9]+$", name):
                raise ValueError(f"public list name uses a reserved shard suffix: {name!r}")


@dataclass(frozen=True)
class GenerationProgram:
    """A complete autoregressive decoder interface for one Scratch sprite.

    The named model method must accept ``example_token`` and return next-token
    logits. cattorch recognizes its causal attention, adds a KV cache of
    ``max_context`` tokens, and creates init, reset, prefill, and decode
    blocks.
    """

    method: str
    example_token: torch.Tensor
    max_context: int
    hidden_prefill: bool = True
    top_k: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", _name(self.method, "generation method"))
        if not isinstance(self.example_token, torch.Tensor):
            raise TypeError("GenerationProgram.example_token must be a torch.Tensor")
        if self.example_token.numel() != 1:
            raise ValueError("GenerationProgram.example_token must contain one token")
        if isinstance(self.max_context, bool) or not isinstance(self.max_context, int):
            raise TypeError("GenerationProgram.max_context must be an integer")
        if self.max_context < 1:
            raise ValueError("GenerationProgram.max_context must be positive")
        if not isinstance(self.hidden_prefill, bool):
            raise TypeError("GenerationProgram.hidden_prefill must be a boolean")
        if self.top_k is not None:
            if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
                raise TypeError("GenerationProgram.top_k must be an integer or None")
            if not 1 <= self.top_k <= 64:
                raise ValueError("GenerationProgram.top_k must be between 1 and 64")


@dataclass(frozen=True)
class ProgramCall:
    """One named invocation used by calibration and stateful verification."""

    entrypoint: str
    inputs: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "entrypoint", _name(self.entrypoint, "call entrypoint"))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if any(not isinstance(value, torch.Tensor) for value in self.inputs):
            raise TypeError("ProgramCall.inputs must contain only tensors")


__all__ = [
    "ArgumentBinding", "EntryPoint", "ExportProgram", "GenerationProgram", "Input", "Output",
    "ProgramCall", "ReturnBinding", "State", "StateInput", "StateUpdate",
]
