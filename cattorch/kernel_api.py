"""Context-style kernel interface (experimental).

Public through ``cattorch.experimental``: ``Kernel``, ``KernelContext``, and
``TensorInfo``, with the statements in ``cattorch.scratch``. Built-in kernels
in ``cattorch/builtin_kernels.py`` use the same interface. It may change
before 1.0.

A ``Kernel`` pairs an operator with an ``emit`` function. ``emit`` receives a
``KernelContext`` describing the operation by name (shapes, dtypes, static
flags, scalar arguments) and returns a body of ``cattorch.scratch``
statements. The context also allocates the lists and variables the body uses.
``_KernelInstruction`` binds a kernel to one call's arguments for the
compiler, which renames its ``T`` slots to the call's lists.

Rules for kernel authors:

* Inputs are keyed by the operator's schema argument names, with schema
  defaults filled in. Reference a tensor through ``TensorInfo.list`` (its
  ``T`` slot), never through a hard-coded slot number.
* ``temp_list`` and ``static_list`` return a private list (prefixed ``_``)
  that the block combiner never merges with another instruction's list.
  ``temp_var`` returns the plain variable name; variables are shared by name
  across the whole sprite, so a kernel must not rely on a value surviving.
* Lists appear in a program in the order they are requested, after the input
  slots. The output slot is requested the first time ``ctx.output`` is read,
  so read it before or after scratch lists to place it accordingly. The order
  is observable in the generated sprite, which is why it is controllable.
* ``ctx.options`` carries lowering facts the compiler decides after the
  operator is known, such as the row stride and base variable of a banked
  generation cache. It is empty for validators. An option must be identical
  for every repeated layer.
* A tensor input the compiler binds to a persistent named list (a generation
  K/V cache) is read like any other input through ``TensorInfo.list``;
  persistent state is just an input whose list the compiler names.
* Layer sharing factors repeated layers into one procedure by comparing
  generated structure. A kernel name and body must not include per-layer
  data such as a layer index, or identical layers stop being identical.
"""

from __future__ import annotations

import math
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Callable

import torch

from cattorch.fast import FastConfig
from cattorch.util.argument import Argument
from cattorch.util.scratch.dsl import Program, Statement


@dataclass(frozen=True)
class TensorInfo:
    """One tensor an operation reads or writes."""

    name: str
    list: str
    shape: tuple[int, ...]
    dtype: torch.dtype | None = None
    static: bool = False
    #: The axis holding a runtime number of items (see ``transpile(dynamic=)``),
    #: or ``None``. ``shape`` is then the traced shape, and the list's length
    #: gives the actual one.
    dynamic_axis: int | None = None
    #: A stored weight's values, known at export, or ``None``. Read them to
    #: specialize the code, for example to skip zero blocks; don't change them.
    value: torch.Tensor | None = field(default=None, compare=False, repr=False)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def strides(self) -> tuple[int, ...]:
        """Row-major strides of the contiguous list layout."""
        return tuple(math.prod(self.shape[dim + 1:]) for dim in range(len(self.shape)))


def _private(name: str) -> str:
    return name if name.startswith("_") else f"_{name}"


def resolve_operator(target: object):
    """Return the ``OpOverload`` for a target, or ``None`` if unavailable."""
    if isinstance(target, torch._ops.OpOverload):
        return target
    parts = str(target).split(".")
    if len(parts) != 3:
        return None
    namespace, name, overload = parts
    try:
        return getattr(getattr(getattr(torch.ops, namespace), name), overload)
    except (AttributeError, RuntimeError):
        return None


def _schema_names(operator, count: int) -> tuple[list[str], dict[str, object]]:
    """Return positional argument names and defaults for ``count`` arguments."""
    names: list[str] = []
    defaults: dict[str, object] = {}
    schema_arguments = () if operator is None else operator._schema.arguments
    for position in range(max(count, len(schema_arguments))):
        if position < len(schema_arguments):
            argument = schema_arguments[position]
            names.append(argument.name)
            if argument.has_default_value():
                defaults[argument.name] = argument.default_value
        else:
            names.append(f"arg{position}")
    return names, defaults


class KernelContext:
    """What a kernel needs to know about one operation, and where it allocates."""

    def __init__(
        self,
        target: str,
        inputs: dict[str, TensorInfo],
        scalars: dict[str, object],
        output: TensorInfo,
        input_lists: tuple[str, ...],
    ) -> None:
        self.target = target
        self.inputs = inputs
        self.scalars = scalars
        self._output = output
        self._lists: list[str] = list(input_lists)
        self._output_declared = False
        self._variables: list[str] = []
        self.list_values: dict[str, list] = {}
        self.temporary_sizes: dict[str, int] = {}
        self.options: dict[str, object] = {}
        # Each tensor input's list name in the compiler, to find its values.
        self.sources: dict[str, str] = {}
        #: The names of the arguments the call passed, in order.
        self.argument_names: tuple[str, ...] = ()

    @classmethod
    def from_arguments(
        cls, target: object, output: Argument | str, args: tuple[Argument, ...],
        operator=None, names: tuple[str, ...] | None = None,
    ) -> KernelContext:
        operator = resolve_operator(target) if operator is None else operator
        argument_names: list[str]
        defaults: dict[str, object]
        if names:
            # A kernel the compiler builds itself names its arguments.
            argument_names = [*names, *(f"arg{i}" for i in range(len(names), len(args)))]
            defaults = {}
        else:
            argument_names, defaults = _schema_names(operator, len(args))
        slots, output_slot = _tensor_slots(args)
        inputs: dict[str, TensorInfo] = {}
        scalars: dict[str, object] = {
            name: value for name, value in defaults.items()
        }
        sources: dict[str, str] = {}
        for position, argument in enumerate(args):
            name = argument_names[position]
            if position in slots and argument.name != "_none":
                sources[name] = str(argument.name)
                inputs[name] = TensorInfo(
                    name,
                    slots[position],
                    tuple(argument.shape),
                    argument.dtype,
                    argument.static,
                    getattr(argument, "dynamic_axis", None),
                )
                scalars.pop(name, None)
            else:
                scalars[name] = argument.value
        if isinstance(output, Argument):
            out = TensorInfo(
                "output", output_slot, tuple(output.shape), output.dtype, False,
            )
        else:
            out = TensorInfo("output", output_slot, ())
        context = cls(
            str(target), inputs, scalars, out, tuple(slots.values()),
        )
        context.sources = sources
        context.argument_names = tuple(argument_names[:len(args)])
        return context

    @classmethod
    def from_node(
        cls, node: torch.fx.Node, operator=None, *, dynamic_axes=None, resolve_weight=None,
        names: tuple[str, ...] | None = None,
    ) -> KernelContext:
        """Build a context from an FX node, for validators before lowering.

        ``dynamic_axes`` maps node names to their variable-length axis, and
        ``resolve_weight`` returns a node's stored tensor, so the context
        matches the one ``emit`` gets.
        """
        dynamic_axes = dynamic_axes or {}
        arguments = []
        values = {}
        for arg in node.args:
            if arg is None:
                arguments.append(Argument("_none", torch.Size([])))
            elif hasattr(arg, "name"):
                value = arg.meta.get("val") if hasattr(arg, "meta") else None
                stored = resolve_weight(arg.name) if resolve_weight is not None else None
                stored = stored if isinstance(stored, torch.Tensor) else None
                if stored is not None:
                    values[arg.name] = stored
                arguments.append(Argument(
                    arg.name,
                    getattr(value, "shape", torch.Size([])),
                    dtype=getattr(value, "dtype", None),
                    static=arg.op == "get_attr" or stored is not None,
                    dynamic_axis=dynamic_axes.get(arg.name),
                ))
            else:
                arguments.append(Argument(f"C_{arg}", torch.Size([]), value=arg))
        value = node.meta.get("val")
        output = Argument(
            node.name,
            getattr(value, "shape", torch.Size([])),
            dtype=getattr(value, "dtype", None),
        )
        context = cls.from_arguments(node.target, output, tuple(arguments), operator, names=names)
        for name, info in list(context.inputs.items()):
            stored = values.get(context.sources.get(name, ""))
            if stored is not None:
                context.inputs[name] = replace(info, value=stored)
        return context

    @property
    def output(self) -> TensorInfo:
        """The output tensor. Reading it places its list in the program."""
        if not self._output_declared:
            self._lists.append(self._output.list)
            self._output_declared = True
        return self._output

    @property
    def lists(self) -> tuple[str, ...]:
        if self._output_declared:
            return tuple(self._lists)
        return (*self._lists, self._output.list)

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self._variables)

    def _declare(self, name: str) -> str:
        if name in self._lists:
            raise ValueError(f"list {name!r} is already declared by this kernel")
        self._lists.append(name)
        return name

    def temp_list(self, name: str, length: int) -> str:
        """Declare a private scratch list of at most ``length`` items."""
        if length < 0:
            raise ValueError("temp_list length must be non-negative")
        name = self._declare(_private(name))
        self.temporary_sizes[name] = int(length)
        return name

    def temp_var(self, name: str) -> str:
        """Declare a variable used by the body and return its name."""
        if name not in self._variables:
            self._variables.append(name)
        return name

    def static_list(self, name: str, values) -> str:
        """Declare a private list initialised with export-time ``values``."""
        name = self._declare(_private(name))
        self.list_values[name] = list(values)
        return name


def _single_overload(packet: torch._ops.OpOverloadPacket) -> torch._ops.OpOverload:
    """A packet's ``default`` overload, if it takes tensors.

    ``torch.ops.aten.add`` has no tensor ``default``: its ``default`` adds two
    Python numbers, so the model's ``add.Tensor`` would never match.
    """
    def takes_tensors(name):
        schema = getattr(packet, name)._schema
        return not schema.is_mutable and any(
            "Tensor" in str(argument.type) for argument in schema.arguments
        )

    if "default" in packet.overloads() and takes_tensors("default"):
        return packet.default
    overloads = [
        name for name in packet.overloads() if "out" not in name and takes_tensors(name)
    ]
    choices = " or ".join(f"torch.ops.{packet}.{name}" for name in overloads[:3])
    raise ValueError(
        f"{packet} doesn't name one tensor operation; pass an overload such as {choices}"
    )


# The kernels an export's instructions ran, while ``warn_unused_kernels`` records.
_KERNELS_USED: ContextVar[set[int] | None] = ContextVar("cattorch_kernels_used", default=None)


@contextmanager
def warn_unused_kernels(kernels, *, fast: bool):
    """Warn, after the export succeeds, about each of ``kernels`` it never ran."""
    used: set[int] = set()
    token = _KERNELS_USED.set(used)
    try:
        yield
    finally:
        _KERNELS_USED.reset(token)
    for kernel in kernels:
        if id(kernel) in used:
            continue
        if kernel.approximate and not fast:
            reason = "an approximate kernel runs only with optimization='fast'"
        else:
            reason = f"the model has no {kernel.target_name} call for it to lower"
        warnings.warn(
            f"the kernel passed for {kernel.target_name} was not used: {reason}",
            UserWarning, stacklevel=4,
        )


@dataclass(frozen=True)
class Kernel:
    """A Scratch lowering for one operation.

    ``target`` is the operation: an ``OpOverload`` such as
    ``torch.ops.aten.gelu.default`` (a bare ``torch.ops.aten.gelu`` means its
    ``default`` overload), its name ``"aten.gelu.default"``, or an operation
    declared with ``cattorch.experimental.op``. ``emit`` receives a ``KernelContext`` and
    returns the body, a tuple of ``cattorch.scratch`` statements.

    ``name`` titles the generated program; it must not include per-layer data.
    ``validator`` receives the same context and returns ``None`` to accept
    the call or a reason, which is raised as ``UnsupportedOperationError``.
    An ``approximate`` kernel is used only with ``optimization="fast"``; an
    exact one replaces the exact kernel. ``fast_flag`` (internal) also ties an
    approximate kernel to a ``FastConfig`` flag.
    """

    target: object
    emit: Callable[[KernelContext], tuple[Statement, ...]]
    name: str | None = None
    validator: Callable[[KernelContext], str | None] | None = None
    fast_flag: str | None = None
    approximate: bool = False
    #: Names of the positional arguments, for an internal target with no
    #: PyTorch schema to name them.
    arguments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        target = getattr(self.target, "operator", self.target)
        if isinstance(target, torch._ops.OpOverloadPacket):
            target = _single_overload(target)
        if not isinstance(target, (torch._ops.OpOverload, str)):
            raise TypeError(
                "kernel target must be an operation such as torch.ops.aten.gelu.default, "
                f"its name \"aten.gelu.default\", or a cattorch.op; got {target!r}"
            )
        object.__setattr__(self, "target", target)
        if not callable(self.emit):
            raise TypeError("kernel emit must be callable")
        if self.validator is not None and not callable(self.validator):
            raise TypeError("kernel validator must be callable or None")
        if not self.target_name:
            raise ValueError("kernel target must be non-empty")
        if self.fast_flag is not None and not hasattr(FastConfig(), self.fast_flag):
            raise ValueError(f"unknown FastConfig flag {self.fast_flag!r}")
        if self.fast_flag is not None:
            object.__setattr__(self, "approximate", True)

    @property
    def target_name(self) -> str:
        return str(self.target)

    @property
    def program_name(self) -> str:
        return self.name or self.target_name.replace(".", "_")

    def instruction_class(self) -> type[_KernelInstruction]:
        """The registry-facing type that runs this kernel on one call."""
        return type(
            f"{self.program_name}_instruction",
            (_KernelInstruction,),
            {"kernel": self},
        )


def _tensor_slots(args) -> tuple[dict[int, str], str]:
    """Map argument positions to ``T`` list slots, and return the output slot.

    The transpiler binds every non-constant argument, including explicit
    ``_none`` placeholders, to consecutive ``T`` lists before the output.
    """
    slots: dict[int, str] = {}
    for position, argument in enumerate(args):
        if not str(argument.name).startswith("C_"):
            slots[position] = f"T{len(slots) + 1}"
    return slots, f"T{len(slots) + 1}"


class _KernelInstruction:
    """A ``Kernel`` bound to one call's arguments.

    ``args`` are positional ``Argument`` values; constants are named ``C_``.
    ``options`` may be set before ``finalize`` to fill ``KernelContext.options``.
    """

    kernel: Kernel
    # The transpiler passes the output's shape and dtype as an ``Argument``.
    wants_output_info = True

    def __init__(self, torch_name, output: Argument | str, *args: Argument):
        self.torch_name = torch_name
        self.output = output
        self.args = args
        self.options: dict[str, object] = {}
        self.context = KernelContext.from_arguments(
            self.kernel.target_name, output, args, self._operator(),
            names=self.kernel.arguments,
        )
        used = _KERNELS_USED.get()
        if used is not None:
            used.add(id(self.kernel))
        self.temporary_sizes: dict[str, int] = {}

    @classmethod
    def _operator(cls):
        # Resolved late: a custom op may be defined after the kernel is.
        return resolve_operator(cls.kernel.target)

    def transform_weights(self, static_lists):
        """Give each stored-weight input its values."""
        context = self.context
        for name, info in list(context.inputs.items()):
            source = context.sources.get(name, "")
            if info.static and source.startswith("W_"):
                value = static_lists.get(source.removeprefix("W_"))
                if isinstance(value, torch.Tensor):
                    context.inputs[name] = replace(info, value=value)

    def finalize(self):
        context = self.context
        context.options = dict(self.options or {})
        body = tuple(self.kernel.emit(context))
        self.temporary_sizes = dict(context.temporary_sizes)
        return Program(
            self.kernel.program_name,
            variables=context.variables,
            lists=context.lists,
            list_values=context.list_values,
            body=body,
        ).compile()

    @classmethod
    def validate_node(
        cls, node: torch.fx.Node, *, dynamic_axes=None, resolve_weight=None,
    ) -> str | None:
        if cls.kernel.validator is None:
            return None
        return cls.kernel.validator(KernelContext.from_node(
            node, cls._operator(), dynamic_axes=dynamic_axes, resolve_weight=resolve_weight,
            names=cls.kernel.arguments,
        ))


def active_kernel_targets(kernels, *, fast: bool, config: FastConfig) -> frozenset[str]:
    """The operations ``kernels``, or kernels attached to declared operations,
    lower in this mode; tracing keeps them whole."""
    from cattorch.custom_ops import DECLARED

    attached = [kernel for operation in DECLARED.values() for kernel in operation.kernels]
    return frozenset(
        kernel.target_name for kernel in (*kernels, *attached)
        if not kernel.approximate or (
            fast and (kernel.fast_flag is None or bool(getattr(config, kernel.fast_flag)))
        )
    )


def validate_kernels(kernels) -> tuple[Kernel, ...]:
    kernels = tuple(kernels)
    if any(not isinstance(kernel, Kernel) for kernel in kernels):
        raise TypeError("kernels must contain cattorch.experimental.Kernel values")
    return kernels

