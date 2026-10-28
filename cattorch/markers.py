"""Declarative markers on a model's own classes and methods (experimental).

A marker only attaches metadata: a ``__cattorch__`` dictionary on the class or
function. The model runs, trains, and saves exactly as without it. At export,
cattorch reads the markers on its copy of the model and builds the objects it
would otherwise need declared by hand (an ``ExportProgram``, a ``SparseMoE``, a
``ModuleAdapter``). A marker that doesn't match what the model does is an
export error.
"""

from __future__ import annotations

import copy
import inspect
import types
from typing import Callable

import torch
from torch import nn

from cattorch.adapters import AdapterContext, ModuleAdapter
from cattorch.errors import UnsupportedModelError

MARKER = "__cattorch__"


def _mark(target, kind: str, options) -> None:
    # Copy an inherited dictionary, so marking a subclass leaves its base alone.
    markers = dict(target.__dict__.get(MARKER) or getattr(target, MARKER, None) or {})
    markers[kind] = options
    setattr(target, MARKER, markers)


def marker(target, kind: str):
    """The options of ``target``'s ``kind`` marker, or ``None``."""
    markers = getattr(target, MARKER, None)
    return markers.get(kind) if isinstance(markers, dict) else None


def marked_methods(module_type: type, kind: str) -> list[tuple[str, Callable, dict]]:
    """Methods of a class carrying ``kind``, as the class resolves them."""
    found = []
    for name in dir(module_type):
        value = inspect.getattr_static(module_type, name, None)
        if isinstance(value, (staticmethod, classmethod)) or not callable(value):
            continue
        options = marker(value, kind)
        if options is not None:
            found.append((name, value, options))
    return found


# ── Method markers ──────────────────────────────────────────────────────────

def entrypoint(
    method: Callable | None = None,
    *,
    name: str | None = None,
    inputs: tuple[str, ...] | None = None,
    outputs: tuple[str, ...] | None = None,
    capacity: dict[str, int] | None = None,
    example=None,
    dynamic: dict[str, tuple[int, int]] | None = None,
):
    """Export this method as a Scratch block; usable bare or with options.

    ``name`` names the block (``cattorch <name>``), the method name by
    default. ``inputs`` and ``outputs`` name its Scratch lists; inputs default
    to the method's parameter names and outputs to ``output``, ``output_1``,
    and so on. ``example`` gives example arguments: a tuple, a tensor, or a
    function returning one. ``forward`` can take them from ``@example``
    instead.

    A buffer the method updates in place (``self.total += value``) or
    reassigns becomes persistent state, starting from the buffer's value. A
    buffer that grows along its first dimension, such as
    ``self.history = torch.cat([self.history, row])``, needs its largest
    length in ``capacity={"history": 64}``.

    ``dynamic={"rows": (1, 64)}`` lets an input's first dimension vary
    between those bounds at run time; keys are input names.
    """
    options = {
        "name": name, "inputs": None if inputs is None else tuple(inputs),
        "outputs": None if outputs is None else tuple(outputs),
        "capacity": dict(capacity or {}), "example": example,
        "dynamic": dict(dynamic or {}),
    }

    def apply(method: Callable) -> Callable:
        _mark(method, "entrypoint", options)
        return method

    return apply if method is None else apply(method)


def generate(method: Callable) -> Callable:
    """Use this method, not ``forward``, for a ``GenerationProgram`` export."""
    _mark(method, "generate", {})
    return method


def export_forward(method: Callable) -> Callable:
    """Trace this method instead of ``forward`` when exporting the class.

    Use it for an export-only version of a block, without training branches
    or control flow that depends on tensor values. It takes ``forward``'s
    arguments, and applies to every instance of the class and its
    subclasses, wherever they sit in the model.
    """
    _mark(method, "export_forward", {})
    return method


# ── Class markers ───────────────────────────────────────────────────────────

def example(inputs):
    """Give a model class example inputs for ``forward``.

    ``inputs`` is a tuple, a tensor, or a function returning one, such as
    ``@example(lambda: (torch.zeros(1, 1, 28, 28),))``. With it,
    ``transpile(model, None, path)`` needs no example of its own.
    """
    def apply(module_type: type) -> type:
        _mark(module_type, "example", {"inputs": inputs})
        return module_type
    return apply


def moe(
    *,
    router: str,
    experts: str,
    top_k: int,
    normalize: bool = False,
    scoring: str = "softmax",
    shared: str | None = None,
    scale: float = 1.0,
):
    """Export this class as a sparse mixture of experts.

    ``router`` names the submodule giving one score per expert, ``experts``
    the ``ModuleList`` of identical experts, and ``top_k`` how many run for
    each token. ``scoring`` turns router outputs into weights (``"softmax"``,
    ``"sigmoid"``, or ``"raw"``), and ``normalize`` rescales the chosen
    weights to sum to 1. ``shared`` names an expert every token runs, added to
    the routed output, and ``scale`` multiplies the routed output.

    The class's own ``forward`` is still used in PyTorch. At export it is
    replaced by an equivalent ``SparseMoE``, checked against ``forward`` on
    probe inputs, so wrong options fail the export.
    """
    if scoring not in {"softmax", "sigmoid", "raw"}:
        raise ValueError("scoring must be 'softmax', 'sigmoid', or 'raw'")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    options = {
        "router": router, "experts": experts, "top_k": top_k, "normalize": bool(normalize),
        "scoring": scoring, "shared": shared, "scale": float(scale),
    }

    def apply(module_type: type) -> type:
        _mark(module_type, "moe", options)
        return module_type
    return apply


def adapter(module_type: type[nn.Module]):
    """Build a ``ModuleAdapter`` from a function, for classes you don't own.

    ``@adapter(ThirdPartyBlock)`` on ``def replace(module, context)`` returns
    the adapter; pass it to ``transpile(adapters=[...])``. It isn't
    registered globally.
    """
    def apply(factory: Callable[[nn.Module, AdapterContext], nn.Module]) -> ModuleAdapter:
        return ModuleAdapter(module_type, factory)
    return apply


# ── Reading markers at export ───────────────────────────────────────────────

def example_inputs(model: nn.Module) -> tuple[torch.Tensor, ...] | None:
    """``forward``'s example from an ``@example`` marker, or ``None``."""
    options = marker(type(model), "example")
    return None if options is None else _arguments(options["inputs"])


def _arguments(value) -> tuple:
    if callable(value) and not isinstance(value, torch.Tensor):
        value = value()
    if isinstance(value, torch.Tensor):
        return (value,)
    return tuple(value)


def generation_method(model: nn.Module) -> str | None:
    methods = marked_methods(type(model), "generate")
    if len(methods) > 1:
        raise UnsupportedModelError(
            f"{type(model).__name__} marks several generation methods: "
            + ", ".join(name for name, _, _ in methods)
        )
    return methods[0][0] if methods else None


def apply_export_forward(module: nn.Module) -> None:
    """Point an export copy's ``forward`` at its ``@export_forward`` method."""
    methods = marked_methods(type(module), "export_forward")
    if len(methods) > 1:
        raise UnsupportedModelError(
            f"{type(module).__name__} marks several export forwards: "
            + ", ".join(name for name, _, _ in methods)
        )
    if methods:
        module.forward = types.MethodType(methods[0][1], module)


class _SharedExpertMoE(nn.Module):
    """A routed mixture plus an expert every token runs."""

    def __init__(self, routed: nn.Module, shared: nn.Module | None, scale: float):
        super().__init__()
        self.routed = routed
        self.shared = shared
        self.scale = scale

    def forward(self, value):
        result = self.routed(value)
        if self.scale != 1.0:
            result = result * self.scale
        if self.shared is not None:
            result = result + self.shared(value)
        return result


def marked_moe(module: nn.Module, path: str):
    """``(SparseMoE, wrapper or None)`` for a module with ``@moe``, or ``None``.

    The wrapper adds a shared expert or scale around the sparse part.
    """
    options = marker(type(module), "moe")
    if options is None:
        return None
    from cattorch.experts import ExpertFamily, SparseMoE
    from cattorch.moe_detection import _probe_outputs, _router

    where = f"{type(module).__name__} at {path or '<root>'}"

    def part(name):
        value = getattr(module, name, None)
        if not isinstance(value, nn.Module):
            raise UnsupportedModelError(f"the @moe marker on {where} names a missing submodule {name!r}")
        return value

    router, expert_list = part(options["router"]), part(options["experts"])
    shared = part(options["shared"]) if options["shared"] else None
    if not isinstance(expert_list, nn.ModuleList) or not len(expert_list):
        raise UnsupportedModelError(f"{options['experts']!r} on {where} is not a ModuleList of experts")
    if options["top_k"] > len(expert_list):
        raise UnsupportedModelError(f"the @moe marker on {where} has top_k above its expert count")
    width = getattr(router, "in_features", None) or next(
        (layer.in_features for layer in expert_list[0].modules() if isinstance(layer, nn.Linear)),
        None,
    )
    if width is None:
        raise UnsupportedModelError(f"cannot tell the input width of {where}")
    dtype = next(module.parameters()).dtype
    family = ExpertFamily.from_modules(
        list(expert_list), example_input=torch.zeros(1, width, dtype=dtype),
    )
    sparse = SparseMoE(
        _router(router, options["scoring"]), family,
        top_k=options["top_k"], normalize_selected=options["normalize"],
    ).train(module.training)
    wrapper = None
    if shared is not None or options["scale"] != 1.0:
        wrapper = _SharedExpertMoE(sparse, shared, options["scale"])
    replacement = wrapper or sparse
    probes = _probe_outputs(module, width, dtype)
    if probes is None:
        raise UnsupportedModelError(f"{where} doesn't accept (tokens, {width}) inputs to check its @moe marker")
    with torch.no_grad():
        differences = [
            float((replacement(value) - output).abs().max()) for value, output in zip(*probes)
        ]
    if max(differences) > 1e-4:
        raise UnsupportedModelError(
            f"the @moe marker on {where} doesn't reproduce its forward on probe inputs "
            f"(largest difference {max(differences):.3g}); check top_k, scoring, "
            "normalize, shared, and scale"
        )
    return sparse, wrapper


# ── Entry points ────────────────────────────────────────────────────────────

def _buffer_owner(module: nn.Module, name: str):
    prefix, _, leaf = name.rpartition(".")
    return (module.get_submodule(prefix) if prefix else module), leaf


def _set_buffer(module: nn.Module, name: str, value: torch.Tensor) -> None:
    owner, leaf = _buffer_owner(module, name)
    owner._buffers[leaf] = value


def _get_buffer(module: nn.Module, name: str) -> torch.Tensor:
    owner, leaf = _buffer_owner(module, name)
    return owner._buffers[leaf]


def _tensor_leaves(value) -> list[torch.Tensor]:
    from torch.utils import _pytree as pytree
    return [leaf for leaf in pytree.tree_leaves(value) if isinstance(leaf, torch.Tensor)]


def _changed_buffers(model: nn.Module, method_name: str, arguments: tuple):
    """Run a method on a copy and return the buffers it changed, with outputs."""
    probe = copy.deepcopy(model).eval()
    before = {
        name: (buffer, buffer._version, buffer.detach().clone())
        for name, buffer in probe.named_buffers()
    }
    with torch.no_grad():
        result = getattr(probe, method_name)(*copy.deepcopy(arguments))
    after = dict(probe.named_buffers())
    changed = {}
    for name, (buffer, version, values) in before.items():
        current = after.get(name)
        if current is None:
            continue
        if (
            current is not buffer or current._version != version
            or current.shape != values.shape or not torch.equal(current, values)
        ):
            changed[name] = (values, current.detach().clone())
    return changed, result


def has_entrypoints(model: nn.Module) -> bool:
    return bool(marked_methods(type(model), "entrypoint"))


def marked_program(model: nn.Module):
    """Build ``(export model, ExportProgram)`` from ``@entrypoint`` markers.

    The export model is a copy whose class gains one generated method per
    entry point. Each takes the entry point's inputs and every state's value,
    installs the values as buffers, calls the marked method, and returns its
    outputs and the updated states, which is how ``ExportProgram`` threads
    state.
    """
    from cattorch.program import EntryPoint, ExportProgram, Input, Output, State, StateInput, StateUpdate

    methods = marked_methods(type(model), "entrypoint")
    if not methods:
        raise UnsupportedModelError(f"{type(model).__name__} has no @entrypoint methods")
    buffers = dict(model.named_buffers())
    plans = []
    modes: dict[str, str] = {}
    capacities: dict[str, int] = {}
    for method_name, function, options in methods:
        source = options["example"]
        if source is None and method_name == "forward":
            source = marker(type(model), "example") and marker(type(model), "example")["inputs"]
        if source is None:
            raise UnsupportedModelError(
                f"@entrypoint {method_name} needs example=, the arguments to export it with"
            )
        arguments = _arguments(source)
        if any(not isinstance(value, torch.Tensor) for value in arguments):
            raise UnsupportedModelError(f"@entrypoint {method_name}'s example arguments must be tensors")
        parameters = [
            parameter.name for parameter in inspect.signature(function).parameters.values()
        ][1:]
        input_names = options["inputs"] or tuple(parameters[:len(arguments)])
        if len(input_names) != len(arguments):
            raise UnsupportedModelError(
                f"@entrypoint {method_name} names {len(input_names)} inputs for "
                f"{len(arguments)} example arguments"
            )
        changed, result = _changed_buffers(model, method_name, arguments)
        for name in options["capacity"]:
            if name not in buffers:
                raise UnsupportedModelError(
                    f"@entrypoint {method_name} gives a capacity for {name!r}, which isn't a buffer"
                )
            if name not in changed or changed[name][1].shape[1:] != changed[name][0].shape[1:] or (
                changed[name][1].shape[0] <= changed[name][0].shape[0]
            ):
                raise UnsupportedModelError(
                    f"@entrypoint {method_name} gives a capacity for {name!r}, but doesn't "
                    "grow it along its first dimension"
                )
        for name, (old, new) in changed.items():
            if name not in buffers:
                continue
            if name in options["capacity"]:
                mode = "append"
                capacities[name] = max(capacities.get(name, 0), options["capacity"][name])
            elif new.shape == old.shape:
                mode = "replace"
            else:
                raise UnsupportedModelError(
                    f"@entrypoint {method_name} changes the shape of buffer {name!r}; a "
                    f"buffer that grows needs capacity={{{name!r}: <largest length>}}"
                )
            if modes.setdefault(name, mode) != mode:
                raise UnsupportedModelError(f"entry points update buffer {name!r} in different ways")
        outputs = _tensor_leaves(result)
        output_names = options["outputs"] or tuple(
            "output" if index == 0 else f"output_{index}" for index in range(len(outputs))
        )
        if len(output_names) != len(outputs):
            raise UnsupportedModelError(
                f"@entrypoint {method_name} names {len(output_names)} outputs, but returns "
                f"{len(outputs)} tensors"
            )
        unknown = set(options["dynamic"]) - set(input_names)
        if unknown:
            raise UnsupportedModelError(
                f"@entrypoint {method_name} makes unknown inputs dynamic: {sorted(unknown)}"
            )
        plans.append((method_name, options["name"] or method_name, input_names, arguments,
                      output_names, set(changed) & set(buffers), options["dynamic"]))

    states = tuple(
        State(name, buffers[name].detach().clone(), mode="append", capacity=capacities[name])
        if mode == "append" else State(name, buffers[name].detach().clone())
        for name, mode in sorted(modes.items())
    )
    state_order = [(state.name, state.mode) for state in states]
    generated = {}
    entrypoints = []
    for method_name, entry_name, input_names, arguments, output_names, changed, dynamic in plans:
        wrapper_name = f"_cattorch_entry_{method_name}"
        generated[wrapper_name] = _entry_method(method_name, len(arguments), state_order, changed)
        entrypoints.append(EntryPoint(
            entry_name, method=wrapper_name,
            arguments=(
                *(
                    Input(name, value, dynamic=dynamic.get(name))
                    for name, value in zip(input_names, arguments)
                ),
                *(StateInput(name) for name, _ in state_order),
            ),
            returns=(
                *(Output(name) for name in output_names),
                *(StateUpdate(name) for name, _ in state_order if name in changed),
            ),
        ))
    exported = copy.deepcopy(model)
    exported.__class__ = type(type(model).__name__, (type(model),), generated)
    return exported, ExportProgram(entrypoints=tuple(entrypoints), states=states)


def _entry_method(method_name: str, inputs: int, states, changed):
    def entry(self, *values):
        arguments, values = values[:inputs], values[inputs:]
        for (name, _), value in zip(states, values):
            # A copy, so in-place updates are on a tensor the method created.
            _set_buffer(self, name, value.clone())
        result = getattr(self, method_name)(*arguments)
        updates = []
        for (name, mode), value in zip(states, values):
            if name not in changed:
                continue
            updated = _get_buffer(self, name)
            # Append state commits only the new rows.
            updates.append(updated[value.shape[0]:] if mode == "append" else updated)
        return (*_tensor_leaves(result), *updates)
    entry.__name__ = f"_cattorch_entry_{method_name}"
    return entry
