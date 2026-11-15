"""Shared entrypoint contracts and eager state replay for export tools."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils._pytree import tree_flatten

from cattorch.errors import UnsupportedModelError
from cattorch.program import ExportProgram, Input, Output, ProgramCall, StateUpdate


class MethodModule(nn.Module):
    def __init__(self, model: nn.Module, method: str):
        super().__init__()
        if not callable(getattr(model, method, None)):
            raise ValueError(f"model has no callable method {method!r}")
        self.model = model
        self.method = method
        self.training = model.training

    def forward(self, *args):
        return getattr(self.model, self.method)(*args)


def entry_examples(program, entrypoint):
    """Capture one append row while retaining its declared runtime bounds."""
    states = {state.name: state for state in program.states}
    args = []
    dynamic_inputs = {}
    for binding in entrypoint.arguments:
        if isinstance(binding, Input):
            if binding.dynamic is not None:
                # Traced with one row, like append state.
                ((axis, (minimum, maximum)),) = binding.dynamic.items()
                dynamic_inputs[len(args)] = (binding.name, minimum, maximum, axis)
                args.append(binding.example.narrow(axis, 0, 1))
            else:
                args.append(binding.example)
        else:
            state = states[binding.name]
            if state.mode == "append":
                dynamic_inputs[len(args)] = (
                    state.name,
                    int(state.initial.shape[0]),
                    int(state.capacity),
                    0,
                )
                args.append(state.initial.new_zeros((1, *state.initial.shape[1:])))
            else:
                args.append(state.initial)
    return tuple(args), dynamic_inputs


def default_program_calls(program: ExportProgram) -> tuple[ProgramCall, ...]:
    """Call every entrypoint once with its example inputs."""
    return tuple(
        ProgramCall(
            entry.name,
            tuple(binding.example for binding in entry.arguments if isinstance(binding, Input)),
        )
        for entry in program.entrypoints
    )


def validate_entry_outputs(program, entrypoint, specs):
    """Validate the same return bindings in analysis and sprite assembly."""
    if len(specs) != len(entrypoint.returns):
        raise UnsupportedModelError(
            f"entry point {entrypoint.name!r} returns {len(specs)} tensor leaves "
            f"but declares {len(entrypoint.returns)} bindings"
        )
    states = {state.name: state for state in program.states}
    inputs = {
        binding.name: binding for binding in entrypoint.arguments
        if isinstance(binding, Input)
    }
    for binding, spec in zip(entrypoint.returns, specs):
        shared_input = inputs.get(binding.name) if isinstance(binding, Output) else None
        if shared_input is not None and tuple(spec.shape) != tuple(shared_input.example.shape):
            # An input and output with one name share a Scratch list, which
            # only works as an in-place update of the same shape.
            raise UnsupportedModelError(
                f"entry point {entrypoint.name!r} output {binding.name!r} has shape "
                f"{tuple(spec.shape)} but shares its list with an input of shape "
                f"{tuple(shared_input.example.shape)}; give the output another name"
            )
        if not isinstance(binding, StateUpdate):
            continue
        state = states[binding.name]
        if state.mode == "replace" and tuple(spec.shape) != tuple(state.initial.shape):
            raise UnsupportedModelError(
                f"entry point {entrypoint.name!r} returns shape {spec.shape} for "
                f"replace state {state.name!r}, expected {tuple(state.initial.shape)}"
            )
        if state.mode == "append" and (
            not spec.shape or tuple(spec.shape[1:]) != tuple(state.initial.shape[1:])
        ):
            raise UnsupportedModelError(
                f"entry point {entrypoint.name!r} append update for {state.name!r} "
                "must preserve the state's leading axis and trailing dimensions"
            )


class ProgramReplay:
    """Replay named calls, committing all declared state updates together."""

    def __init__(self, model: nn.Module, program: ExportProgram):
        self.model = model
        self.program = program
        self.entries = {entry.name: entry for entry in program.entrypoints}
        self.declarations = {state.name: state for state in program.states}
        self.reset()

    def reset(self) -> None:
        """Restore every state to its initial value, as ``cattorch reset`` does."""
        self.states = {
            state.name: state.initial.detach().clone() for state in self.program.states
        }

    def entry(self, invocation):
        entry = self.entries.get(invocation.entrypoint)
        if entry is None:
            raise ValueError(f"unknown entry point {invocation.entrypoint!r}")
        expected = sum(isinstance(value, Input) for value in entry.arguments)
        if len(invocation.inputs) != expected:
            raise ValueError(
                f"entry point {entry.name!r} expects {expected} public inputs, "
                f"got {len(invocation.inputs)}"
            )
        return entry

    def checked_inputs(self, entry, inputs) -> list[torch.Tensor]:
        """Copies of a call's public inputs, each checked against its contract."""
        checked = []
        for binding, value in zip(
            (binding for binding in entry.arguments if isinstance(binding, Input)), inputs,
        ):
            expected = binding.example.shape
            if binding.dynamic is not None:
                ((axis, (minimum, maximum)),) = binding.dynamic.items()
                matches = (
                    value.ndim == len(expected)
                    and all(
                        size == wanted for dimension, (size, wanted)
                        in enumerate(zip(value.shape, expected)) if dimension != axis
                    )
                    and minimum <= value.shape[axis] <= maximum
                )
            else:
                matches = value.shape == expected
            if not matches or value.dtype != binding.example.dtype:
                raise ValueError(
                    f"input {binding.name!r} does not match its shape/dtype contract"
                )
            checked.append(value.detach().clone())
        return checked

    def run(self, invocation):
        entry = self.entry(invocation)
        public = iter(self.checked_inputs(entry, invocation.inputs))
        args = []
        for binding in entry.arguments:
            if isinstance(binding, Input):
                value = next(public)
            else:
                value = self.states[binding.name]
            args.append(value.detach().clone())
        with torch.no_grad():
            returned = getattr(self.model, entry.method)(*args)
        leaves, _ = tree_flatten(returned)
        if any(not isinstance(value, torch.Tensor) for value in leaves):
            raise UnsupportedModelError(f"entry point {entry.name!r} must return only tensor leaves")
        validate_entry_outputs(self.program, entry, leaves)
        pending = {}
        for binding, value in zip(entry.returns, leaves):
            if not isinstance(binding, StateUpdate):
                continue
            state = self.declarations[binding.name]
            updated = value.detach().clone()
            if state.mode == "append":
                updated = torch.cat((self.states[binding.name], updated), dim=0)
                if updated.shape[0] > state.capacity:
                    raise ValueError(f"state capacity exceeded: {state.name}")
            pending[binding.name] = updated
        self.states.update(pending)
        return leaves


MARKED_PREFIX = "_cattorch_entry_"


def is_marked(model: nn.Module, program: ExportProgram) -> bool:
    """Whether every entry point calls a method ``marked_program`` generated."""
    return all(
        entry.method.startswith(MARKED_PREFIX)
        and callable(getattr(model, entry.method.removeprefix(MARKED_PREFIX), None))
        for entry in program.entrypoints
    )


class MarkedReplay(ProgramReplay):
    """Replay calls of a marked program through the user's own methods.

    ``marked_program`` turns each ``@entrypoint`` method into a generated one
    that takes the states as arguments. This replay doesn't use those: it
    calls the user's method on the model, with the states living in its
    buffers, as in PyTorch, so a defect in the generated methods shows as a
    difference from the sprite.
    """

    def __init__(self, model: nn.Module, program: ExportProgram):
        canonical: dict[int, str] = {}
        self.aliases: dict[str, list[str]] = {}
        for name, buffer in model.named_buffers(remove_duplicate=False):
            self.aliases.setdefault(canonical.setdefault(id(buffer), name), []).append(name)
        super().__init__(model, program)

    @property  # type: ignore[override]
    def states(self) -> dict[str, torch.Tensor]:
        from cattorch.markers import _get_buffer

        return {
            state.name: _get_buffer(self.model, state.name) for state in self.program.states
        }

    @states.setter
    def states(self, values: dict[str, torch.Tensor]) -> None:
        from cattorch.markers import _set_buffer

        for name, value in values.items():
            value = value.detach().clone()
            for alias in self.aliases.get(name, [name]):
                _set_buffer(self.model, alias, value)

    def run(self, invocation):
        from cattorch.markers import _tensor_leaves

        entry = self.entry(invocation)
        inputs = self.checked_inputs(entry, invocation.inputs)
        before = {name: value.detach().clone() for name, value in self.states.items()}
        with torch.no_grad():
            returned = getattr(self.model, entry.method.removeprefix(MARKED_PREFIX))(*inputs)
        for name, value in self.states.items():
            state = self.declarations[name]
            if state.mode == "append" and value.shape[0] > (state.capacity or 0):
                self.states = before
                raise ValueError(f"state capacity exceeded: {state.name}")
        return _tensor_leaves(returned)


def program_replay(model: nn.Module, program: ExportProgram) -> ProgramReplay:
    """The replay of ``program``'s calls on ``model``, which a sprite should match."""
    return MarkedReplay(model, program) if is_marked(model, program) else ProgramReplay(model, program)
