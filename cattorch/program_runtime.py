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
            args.append(binding.example)
        else:
            state = states[binding.name]
            if state.mode == "append":
                dynamic_inputs[len(args)] = (
                    state.name,
                    int(state.initial.shape[0]),
                    int(state.capacity),
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
            f"entrypoint {entrypoint.name!r} returns {len(specs)} tensor leaves "
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
                f"entrypoint {entrypoint.name!r} output {binding.name!r} has shape "
                f"{tuple(spec.shape)} but shares its list with an input of shape "
                f"{tuple(shared_input.example.shape)}; give the output another name"
            )
        if not isinstance(binding, StateUpdate):
            continue
        state = states[binding.name]
        if state.mode == "replace" and tuple(spec.shape) != tuple(state.initial.shape):
            raise UnsupportedModelError(
                f"entrypoint {entrypoint.name!r} returns shape {spec.shape} for "
                f"replace state {state.name!r}, expected {tuple(state.initial.shape)}"
            )
        if state.mode == "append" and (
            not spec.shape or tuple(spec.shape[1:]) != tuple(state.initial.shape[1:])
        ):
            raise UnsupportedModelError(
                f"entrypoint {entrypoint.name!r} append update for {state.name!r} "
                "must preserve the state's leading axis and trailing dimensions"
            )


class ProgramReplay:
    """Replay named calls, committing all declared state updates together."""

    def __init__(self, model: nn.Module, program: ExportProgram):
        self.model = model
        self.program = program
        self.entries = {entry.name: entry for entry in program.entrypoints}
        self.declarations = {state.name: state for state in program.states}
        self.states = {state.name: state.initial.detach().clone() for state in program.states}

    def entry(self, invocation):
        entry = self.entries.get(invocation.entrypoint)
        if entry is None:
            raise ValueError(f"unknown entrypoint {invocation.entrypoint!r}")
        expected = sum(isinstance(value, Input) for value in entry.arguments)
        if len(invocation.inputs) != expected:
            raise ValueError(
                f"entrypoint {entry.name!r} expects {expected} public inputs, "
                f"got {len(invocation.inputs)}"
            )
        return entry

    def run(self, invocation):
        entry = self.entry(invocation)
        public = iter(invocation.inputs)
        args = []
        for binding in entry.arguments:
            if isinstance(binding, Input):
                value = next(public)
                if value.shape != binding.example.shape or value.dtype != binding.example.dtype:
                    raise ValueError(
                        f"input {binding.name!r} does not match its shape/dtype contract"
                    )
            else:
                value = self.states[binding.name]
            args.append(value.detach().clone())
        with torch.no_grad():
            returned = getattr(self.model, entry.method)(*args)
        leaves, _ = tree_flatten(returned)
        if any(not isinstance(value, torch.Tensor) for value in leaves):
            raise UnsupportedModelError(f"entrypoint {entry.name!r} must return only tensor leaves")
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
