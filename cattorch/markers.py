"""Declarative markers on a model's own classes and methods.

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
from typing import TYPE_CHECKING, Callable, NamedTuple

import torch
from torch import nn

from cattorch.adapters import AdapterContext, ModuleAdapter
from cattorch.errors import UnsupportedModelError, warn
from cattorch.interface import output_name

if TYPE_CHECKING:
    from cattorch.program import ExportProgram

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
    input_names: tuple[str, ...] | None = None,
    output_names: tuple[str, ...] | None = None,
    capacity: dict[str, int] | None = None,
    example_inputs=None,
    dynamic: dict[str, tuple[int, int] | dict[int, tuple[int, int]]] | None = None,
) -> Callable:
    """Export this method as a Scratch block; usable bare or with options.

    ``name`` names the block (``cattorch <name>``), the method name by
    default. ``input_names`` and ``output_names`` name its Scratch lists;
    inputs default to the method's parameter names and outputs to
    ``output``, ``output_1``, and so on. ``example_inputs`` gives the
    arguments to export it with: a tuple, a tensor, or a function returning
    one. ``forward`` can take them from ``@example_inputs`` instead.

    A buffer the method updates in place (``self.total += value``) or
    reassigns becomes persistent state, starting from the buffer's value. A
    buffer that grows along its first dimension, such as
    ``self.history = torch.cat([self.history, row])``, needs its largest
    length in ``capacity={"history": 64}``.

    ``dynamic={"rows": {0: (1, 64)}}`` lets an input's dimension vary
    between those bounds at run time, as ``transpile(dynamic=...)`` does;
    keys are input names, and ``(1, 64)`` is short for the first dimension.
    """
    from cattorch.program import _name

    if name is not None:
        name = _name(name, "@entrypoint name")
    for label, names in (("input_names", input_names), ("output_names", output_names)):
        if names is not None and (
            isinstance(names, str) or not isinstance(names, (tuple, list))
            or any(not isinstance(item, str) for item in names)
        ):
            raise TypeError(f"@entrypoint {label} must be a tuple of strings")
    for label, mapping in (("capacity", capacity), ("dynamic", dynamic)):
        if mapping is not None and (
            not isinstance(mapping, dict) or any(not isinstance(key, str) for key in mapping)
        ):
            raise TypeError(f"@entrypoint {label} must be a dict keyed by name")
    for buffer, length in (capacity or {}).items():
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise ValueError(
                f"@entrypoint capacity for {buffer!r} must be a positive integer, not {length!r}"
            )
    options = {
        "name": name, "input_names": None if input_names is None else tuple(input_names),
        "output_names": None if output_names is None else tuple(output_names),
        "capacity": dict(capacity or {}), "example_inputs": example_inputs,
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

def example_inputs(inputs) -> Callable[[type], type]:
    """Give a model class example inputs for ``forward``.

    ``inputs`` is a tuple, a tensor, or a function returning one, such as
    ``@example_inputs(lambda: (torch.zeros(1, 1, 28, 28),))``. With it,
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
    if not isinstance(normalize, bool):
        raise TypeError(f"normalize must be True or False, not {normalize!r}")
    if scoring not in {"softmax", "sigmoid", "raw"}:
        raise ValueError("scoring must be 'softmax', 'sigmoid', or 'raw'")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    options = {
        "router": router, "experts": experts, "top_k": top_k, "normalize": normalize,
        "scoring": scoring, "shared": shared, "scale": float(scale),
    }

    def apply(module_type: type) -> type:
        _mark(module_type, "moe", options)
        return module_type
    return apply


def adapter(
    module_type: type[nn.Module],
) -> Callable[[Callable[[nn.Module, AdapterContext], nn.Module]], ModuleAdapter]:
    """Build a ``ModuleAdapter`` from a function, for classes you don't own.

    ``@adapter(ThirdPartyBlock)`` on ``def replace(module, context)`` returns
    the adapter; pass it to ``transpile(adapters=[...])``. It isn't
    registered globally.
    """
    def apply(factory: Callable[[nn.Module, AdapterContext], nn.Module]) -> ModuleAdapter:
        return ModuleAdapter(module_type, factory)
    return apply


# ── Reading markers at export ───────────────────────────────────────────────

def marked_example_inputs(model: nn.Module) -> tuple[torch.Tensor, ...] | None:
    """``forward``'s example inputs from an ``@example_inputs`` marker, or ``None``."""
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

    where = f"{type(module).__name__} at {path!r}" if path else f"the top-level {type(module).__name__}"

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
        raise UnsupportedModelError(f"can't tell the input width of {where}")
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


class _ReadBuffers(dict):
    """A module's ``_buffers`` that records which buffers are read."""

    def __init__(self, values, names, reads):
        super().__init__(values)
        self._names, self._reads = names, reads

    def _read(self, key) -> None:
        self._reads.add(self._names.get(key, key))

    def __getitem__(self, key):
        self._read(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self:
            self._read(key)
        return super().get(key, default)

    def _read_all(self) -> None:
        for key in super().__iter__():
            self._read(key)

    def __iter__(self):
        self._read_all()
        return super().__iter__()

    def keys(self):
        self._read_all()
        return super().keys()

    def values(self):
        self._read_all()
        return super().values()

    def items(self):
        self._read_all()
        return super().items()


def _canonical_buffer_names(model: nn.Module) -> dict[tuple[int, str], str]:
    """``{(id(owner module), attribute): buffer name}``, aliases mapped to one name."""
    canonical = {id(buffer): name for name, buffer in model.named_buffers()}
    found = {}
    for name, buffer in model.named_buffers(remove_duplicate=False):
        owner, leaf = _buffer_owner(model, name)
        found[(id(owner), leaf)] = canonical[id(buffer)]
    return found


# Attributes every module has, which an entry point doesn't change as state.
_MODULE_INTERNALS = frozenset(nn.Module().__dict__) | {"forward"}
_CHEAP = (bool, int, float, complex, str, bytes, type(None))


def _snapshot(value, depth: int = 0):
    """A comparable copy of an attribute's value, or ``None`` if it isn't tracked.

    Tensors, plain Python values, and lists, tuples, dictionaries and sets of
    them are tracked; other objects aren't.
    """
    if isinstance(value, torch.Tensor):
        return ("tensor", value.detach().clone())
    if isinstance(value, _CHEAP):
        return ("value", value)
    if depth >= 3:
        return None
    if isinstance(value, (list, tuple)) and len(value) <= 256:
        items = [_snapshot(item, depth + 1) for item in value]
        return None if None in items else (type(value).__name__, items)
    if isinstance(value, dict) and len(value) <= 256 and all(isinstance(key, _CHEAP) for key in value):
        entries = {key: _snapshot(item, depth + 1) for key, item in value.items()}
        return None if None in entries.values() else ("dict", entries)
    if isinstance(value, (set, frozenset)) and len(value) <= 256 and all(
        isinstance(item, _CHEAP) for item in value
    ):
        return ("value", frozenset(value))
    return None


def _same(before, after) -> bool:
    if before is None or after is None or before[0] != after[0]:
        return before is None and after is None
    kind, old = before
    new = after[1]
    if kind in {"tensor", "parameter"}:
        if old.shape != new.shape or old.dtype != new.dtype:
            return False
        if old.is_floating_point() or old.is_complex():
            return torch.allclose(old, new, rtol=0, atol=0, equal_nan=True)
        return torch.equal(old, new)
    if kind in {"list", "tuple"}:
        return len(old) == len(new) and all(_same(a, b) for a, b in zip(old, new))
    if kind == "dict":
        return old.keys() == new.keys() and all(_same(old[key], new[key]) for key in old)
    return type(old) is type(new) and (old == new or (old != old and new != new))


def _attributes(model: nn.Module) -> dict[str, object]:
    """Snapshots of every module's own attributes and parameters, by path."""
    found: dict[str, object] = {}
    for prefix, module in model.named_modules():
        where = f"{prefix}." if prefix else ""
        for key, value in vars(module).items():
            if key not in _MODULE_INTERNALS:
                found[f"{where}{key}"] = _snapshot(value)
        for key, value in module._parameters.items():
            if value is not None:
                found[f"{where}{key}"] = ("parameter", value.detach().clone())
    return found


def _describe(snapshot) -> str:
    if snapshot is None:
        return "nothing"
    kind, value = snapshot
    if kind in {"tensor", "parameter"}:
        return f"a tensor of shape {tuple(value.shape)}"
    if kind == "value":
        return repr(value) if len(repr(value)) <= 40 else f"a {type(value).__name__}"
    return f"a {kind}"


def _check_attributes(probe, method_name, arguments, before) -> None:
    """Reject tensors and Python values an entry point changes outside its buffers.

    The trace bakes them in as constants, so the sprite would keep the values
    they had when it was traced. A tensor attribute or parameter that changes
    is rejected; a Python value that changes again on a second call is too.
    One that changes on the first call only, such as a cache filled once, is
    reported with a warning, since the sprite always runs as that first call.
    """
    after = _attributes(probe)
    changed = [
        name for name in {**before, **after}
        if not _same(before.get(name), after.get(name))
    ]
    for name in changed:
        old = before.get(name)
        if old is not None and old[0] == "parameter":
            raise UnsupportedModelError(
                f"@entrypoint {method_name} changes the parameter self.{name}, which a "
                "sprite keeps as a constant. Only buffers become state: register the "
                f"tensor with self.register_buffer({name.rpartition('.')[2]!r}, ...) instead"
            )
        if old is not None and old[0] == "tensor":
            raise UnsupportedModelError(
                f"@entrypoint {method_name} changes self.{name}, a tensor attribute that "
                "isn't a buffer, so the sprite would keep its first value. Only buffers "
                "become state: register it with "
                f"self.register_buffer({name.rpartition('.')[2]!r}, ...) in __init__"
            )
    if not changed:
        return
    second = _attributes(probe)
    try:
        with torch.no_grad():
            getattr(probe, method_name)(*copy.deepcopy(arguments))
    except Exception:  # noqa: BLE001 - a second call that fails tells nothing more
        again = changed
    else:
        third = _attributes(probe)
        again = [name for name in changed if not _same(second.get(name), third.get(name))]
    for name in again:
        raise UnsupportedModelError(
            f"@entrypoint {method_name} changes self.{name} on every call (from "
            f"{_describe(before.get(name))} to {_describe(after.get(name))} on the "
            "first), but a Python value is fixed when cattorch traces the method, so "
            "the sprite would keep the value it had then. Keep a value that changes "
            "between calls in a tensor buffer, such as "
            f"self.register_buffer({name.rpartition('.')[2].split('[')[0]!r}, torch.tensor(0))"
        )
    warn(
        f"@entrypoint {method_name} changes "
        + ", ".join(f"self.{name}" for name in sorted(changed))
        + " on its first call only. The sprite runs every call as that first call "
        "traced from the initial values; if later calls take another path because "
        "of the change, it won't follow them",
        stacklevel=4,
    )


def _call_method(probe, method_name: str, arguments: tuple, shown: tuple, *, again=False):
    """Call an entry point's method, reporting its failure as the model's."""
    from cattorch.frontend import model_failure

    try:
        with torch.no_grad():
            return getattr(probe, method_name)(*arguments)
    except Exception as error:  # noqa: BLE001 - the user's method failed; reported
        caller = f"@entrypoint {method_name}" + (" called a second time" if again else "")
        raise UnsupportedModelError(model_failure(caller, error, shown)) from error


def _changed_buffers(model: nn.Module, method_name: str, arguments: tuple):
    """Run a method on a copy: the buffers it changed and read, and its outputs."""
    probe = copy.deepcopy(model).eval()
    attributes = _attributes(probe)
    before = {
        name: (buffer, buffer._version, buffer.detach().clone())
        for name, buffer in probe.named_buffers()
    }
    names = _canonical_buffer_names(probe)
    canonical = {id(buffer): name for name, buffer in probe.named_buffers()}
    aliases: dict[str, list[str]] = {}
    for name, buffer in probe.named_buffers(remove_duplicate=False):
        aliases.setdefault(canonical[id(buffer)], []).append(name)
    reads: set[str] = set()
    for module in probe.modules():
        own = {leaf: name for (owner, leaf), name in names.items() if owner == id(module)}
        module.__dict__["_buffers"] = _ReadBuffers(module._buffers, own, reads)
    try:
        result = _call_method(probe, method_name, copy.deepcopy(arguments), arguments)
    finally:
        for module in probe.modules():
            module.__dict__["_buffers"] = dict(dict.items(module.__dict__["_buffers"]))
    for name, registered in aliases.items():
        if len({id(_get_buffer(probe, alias)) for alias in registered}) > 1:
            raise UnsupportedModelError(
                f"@entrypoint {method_name} reassigns buffer {name!r} under only some "
                f"of the names it is registered as ({', '.join(map(repr, registered))}), "
                "so they stop being one tensor. Register it once, or update it in "
                "place (self.mem.mul_(0.5).add_(x)), which keeps the names together"
            )
    after = dict(probe.named_buffers())
    # Buffers are compared below; a second call (for Python values) runs on
    # a copy, so it doesn't change what is compared.
    _check_attributes(copy.deepcopy(probe), method_name, arguments, attributes)
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
    return changed, reads, result


def _rewritten_rows(model: nn.Module, method_name: str, arguments: tuple, names) -> list[str]:
    """Growing buffers whose existing rows a second call changes.

    The sprite commits only the rows a call adds. One call from the initial
    state, often empty, can't show whether the method also rewrites or
    shifts the old rows, as ``cat([decay * old, new])`` or prepending does.
    """
    from cattorch.dynamic import varied_at_length

    probe = copy.deepcopy(model).eval()
    # Other values the second time, so the rows it adds look different.
    others = tuple(
        varied_at_length(value, 0, value.shape[0], seed=1) if value.ndim else value + 1
        for value in arguments
    )
    _call_method(probe, method_name, copy.deepcopy(arguments), arguments)
    first = {name: _get_buffer(probe, name).detach().clone() for name in names}
    _call_method(probe, method_name, others, others, again=True)
    rewritten = []
    for name, rows in first.items():
        grown = _get_buffer(probe, name)
        if grown.shape[0] < rows.shape[0] or not torch.equal(grown[:rows.shape[0]], rows):
            rewritten.append(name)
    return rewritten


def _memory_span(value: torch.Tensor) -> tuple[int, int, int] | None:
    """``(storage address, first byte, end byte)`` a tensor's elements occupy."""
    if value.numel() == 0 or value.device.type == "meta":
        return None
    size = value.element_size()
    start = value.storage_offset() * size
    extent = 1 + sum((length - 1) * abs(step) for length, step in zip(value.shape, value.stride()))
    return value.untyped_storage().data_ptr(), start, start + extent * size


def _reject_shared_memory(model: nn.Module, state_names) -> None:
    """Reject a state whose memory another buffer or parameter also uses.

    Each state is a list of its own in the sprite, so a write through one
    view (``self.k = self.cache[0]``) wouldn't reach the other, as it does
    in PyTorch.
    """
    tensors = [
        (kind, name, value)
        for kind, named in (("buffer", model.named_buffers()), ("parameter", model.named_parameters()))
        for name, value in named
    ]
    spans = {name: _memory_span(value) for _, name, value in tensors}
    for state in state_names:
        mine = spans.get(state)
        if mine is None:
            continue
        for kind, name, _ in tensors:
            other = spans[name]
            if name == state or other is None or other[0] != mine[0]:
                continue
            if other[1] < mine[2] and mine[1] < other[2]:
                inside = other[1] <= mine[1] and mine[2] <= other[2]
                inner, outer = (state, name) if inside else (name, state)
                raise UnsupportedModelError(
                    f"buffer {state!r} shares memory with {kind} {name!r} ({inner!r} is a "
                    f"view of {outer!r}, as self.k = self.cache[0] makes one), so a write "
                    "to one changes the other in PyTorch. cattorch keeps each state in a "
                    f"list of its own and can't follow that. Register only {outer!r} as a "
                    "buffer and take the view inside the method, as in "
                    f"k = self.{outer.rpartition('.')[2]}[0]"
                )


def has_entrypoints(model: nn.Module) -> bool:
    return bool(marked_methods(type(model), "entrypoint"))


class MarkedProgram(NamedTuple):
    """What :func:`marked_program` builds from a model's ``@entrypoint`` markers.

    ``model`` is a copy of the model whose class gains one generated method
    per entry point, and ``program`` the ``ExportProgram`` that calls those
    methods. Pass both on together, as ``transpile(*marked, path)``, or
    unpack them: ``model, program = marked``.
    """

    model: nn.Module
    program: ExportProgram


def marked_program(model: nn.Module) -> MarkedProgram:
    """Build the export model and ``ExportProgram`` from ``@entrypoint`` markers.

    The export model is a copy whose class gains one generated method per
    entry point. Each takes the entry point's inputs and the values of the
    states the method reads or writes, installs them as buffers, calls the
    marked method, and returns its outputs and the updated states, which is how ``ExportProgram`` threads
    state. ``transpile``, ``verify`` and ``analyze`` do this themselves when
    given the model and ``None``; call it to inspect the program.
    """
    from cattorch.program import EntryPoint, ExportProgram, Input, Output, State, StateInput, StateUpdate

    methods = marked_methods(type(model), "entrypoint")
    if not methods:
        raise UnsupportedModelError(f"{type(model).__name__} has no @entrypoint methods")
    buffers = dict(model.named_buffers())
    # A tensor registered under several names is one state, set under each.
    names_of: dict[int, list[str]] = {}
    for name, buffer in model.named_buffers(remove_duplicate=False):
        names_of.setdefault(id(buffer), []).append(name)
    aliases = {name: names_of[id(buffer)] for name, buffer in buffers.items()}
    plans = []
    modes: dict[str, str] = {}
    capacities: dict[str, int] = {}
    for method_name, function, options in methods:
        source = options["example_inputs"]
        if source is None and method_name == "forward":
            source = marker(type(model), "example") and marker(type(model), "example")["inputs"]
        if source is None:
            raise UnsupportedModelError(
                f"@entrypoint {method_name} needs example_inputs=, the arguments to "
                "export it with"
            )
        arguments = _arguments(source)
        if any(not isinstance(value, torch.Tensor) for value in arguments):
            raise UnsupportedModelError(f"@entrypoint {method_name}'s example arguments must be tensors")
        parameters = [
            parameter.name for parameter in inspect.signature(function).parameters.values()
        ][1:]
        input_names = options["input_names"] or tuple(parameters[:len(arguments)])
        if len(input_names) != len(arguments):
            raise UnsupportedModelError(
                f"@entrypoint {method_name} has {len(input_names)} input_names for "
                f"{len(arguments)} example inputs"
            )
        changed, reads, result = _changed_buffers(model, method_name, arguments)
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
        for name in _rewritten_rows(model, method_name, arguments, options["capacity"]):
            raise UnsupportedModelError(
                f"@entrypoint {method_name} changes rows of {name!r} that were already "
                "there; a buffer with a capacity may only grow by adding rows at its end"
            )
        for name, (old, new) in changed.items():
            if name not in buffers:
                continue
            if name in options["capacity"]:
                mode = "append"
                capacity = options["capacity"][name]
                if capacities.setdefault(name, capacity) != capacity:
                    raise UnsupportedModelError(
                        f"entry points give {name!r} different capacities "
                        f"({capacities[name]} and {capacity} in @entrypoint {method_name}); "
                        "a buffer has one largest length, so give the same capacity= "
                        "on every entry point that grows it"
                    )
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
        if result is not None:
            from torch.utils import _pytree as pytree

            from cattorch.frontend import reject_tensor_containers

            others = [
                leaf for leaf in pytree.tree_leaves(result) if not isinstance(leaf, torch.Tensor)
            ]
            reject_tensor_containers(
                (f"@entrypoint {method_name}'s result", leaf) for leaf in others
            )
            dropped = [type(leaf).__name__ for leaf in others]
            if dropped:
                # As transpile does for forward; declared Output bindings
                # stay strict.
                warn(
                    f"@entrypoint {method_name}: ignoring non-tensor results ("
                    + ", ".join(dropped) + ")",
                    stacklevel=3,
                )
        output_names = options["output_names"] or tuple(
            output_name(index) for index in range(len(outputs))
        )
        if len(output_names) != len(outputs):
            raise UnsupportedModelError(
                f"@entrypoint {method_name} has {len(output_names)} output_names, but "
                f"returns {len(outputs)} tensors"
            )
        unknown = set(options["dynamic"]) - set(input_names)
        if unknown:
            raise UnsupportedModelError(
                f"@entrypoint {method_name}'s dynamic= names inputs it doesn't have: "
                f"{sorted(unknown)}; its inputs are {list(input_names)}"
            )
        plans.append((method_name, options["name"] or method_name, input_names, arguments,
                      output_names, set(changed) & set(buffers), reads, options["dynamic"]))

    _reject_shared_memory(model, modes)
    states = tuple(
        State(name, buffers[name].detach().clone(), mode="append", capacity=capacities[name])
        if mode == "append" else State(name, buffers[name].detach().clone())
        for name, mode in sorted(modes.items())
    )
    state_order = [(state.name, state.mode) for state in states]
    generated = {}
    entrypoints = []
    for method_name, entry_name, input_names, arguments, output_names, changed, reads, dynamic in plans:
        wrapper_name = f"_cattorch_entry_{method_name}"
        # Only the states the method reads or writes are its arguments; it
        # neither checks nor rewrites the others.
        used = [(name, mode) for name, mode in state_order if name in changed or name in reads]
        unused = {name: aliases[name] for name, _ in state_order if name not in changed | reads}
        generated[wrapper_name] = _entry_method(
            method_name, len(arguments), used, changed, aliases, unused,
        )
        entrypoints.append(EntryPoint(
            entry_name, method=wrapper_name,
            arguments=(
                *(
                    Input(name, value, dynamic=dynamic.get(name))
                    for name, value in zip(input_names, arguments)
                ),
                *(StateInput(name) for name, _ in used),
            ),
            returns=(
                *(Output(name) for name in output_names),
                *(StateUpdate(name) for name, _ in used if name in changed),
            ),
        ))
    exported = copy.deepcopy(model)
    exported.__class__ = type(type(model).__name__, (type(model),), generated)
    return MarkedProgram(exported, ExportProgram(entrypoints=tuple(entrypoints), states=states))


class _GuardedBuffers(dict):
    """A module's ``_buffers`` during a generated entry method.

    Reading a state that the method didn't read when cattorch ran it, so
    that isn't among its arguments, would bake that state's initial value
    into the export; it raises instead.
    """

    def __init__(self, values, forbidden, method_name):
        super().__init__(values)
        self._forbidden, self._method = forbidden, method_name

    def _refuse(self, key) -> None:
        if key in self._forbidden:
            raise UnsupportedModelError(
                f"@entrypoint {self._method} read buffer {self._forbidden[key]!r} when "
                "cattorch traced it, but not when cattorch ran it on its example "
                "inputs, so the buffer isn't one of its states. Read the same buffers "
                "on every call, without a Python branch on a buffer's shape or values"
            )

    def __getitem__(self, key):
        self._refuse(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._refuse(key)
        return super().get(key, default)

    def _refuse_all(self) -> None:
        for key in super().__iter__():
            self._refuse(key)

    def __iter__(self):
        self._refuse_all()
        return super().__iter__()

    def keys(self):
        self._refuse_all()
        return super().keys()

    def values(self):
        self._refuse_all()
        return super().values()

    def items(self):
        self._refuse_all()
        return super().items()


def _entry_method(method_name: str, inputs: int, states, changed, aliases, unused):
    def entry(self, *values):
        arguments, values = values[:inputs], values[inputs:]
        for (name, _), value in zip(states, values):
            # A copy, so in-place updates are on a tensor the method created.
            value = value.clone()
            for alias in aliases.get(name, (name,)):
                _set_buffer(self, alias, value)
        guarded = {}
        for name, names in unused.items():
            for alias in names:
                owner, leaf = _buffer_owner(self, alias)
                guarded.setdefault(id(owner), (owner, {}))[1][leaf] = name
        for owner, forbidden in guarded.values():
            owner.__dict__["_buffers"] = _GuardedBuffers(owner._buffers, forbidden, method_name)
        try:
            result = getattr(self, method_name)(*arguments)
        finally:
            for owner, _ in guarded.values():
                owner.__dict__["_buffers"] = dict(dict.items(owner.__dict__["_buffers"]))
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
