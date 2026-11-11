"""Private registry for frontend-neutral operation and module lowering rules."""

from __future__ import annotations

import copy
import itertools
import math
import warnings
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Literal, cast

import torch

from cattorch.fast import FastConfig
from cattorch.huggingface import (
    convert_huggingface_experts, is_huggingface_experts, prepare_huggingface_models,
)
from cattorch.builtin_kernels import BUILTIN_KERNELS
from cattorch.kernel_api import Kernel, _KernelInstruction
from cattorch.moe_detection import detect_custom_moe
from cattorch.adapters import AdapterContext, ModuleAdapter as PublicModuleAdapter
from cattorch.util.instruction.optimized_convolution import CONVOLUTION_KERNELS, POOLING_KERNELS
from cattorch.util.instruction.optimized_elementwise import FAST_GELU_KERNEL, elementwise_kernel
from cattorch.util.instruction.optimized_linear import LINEAR_KERNEL, matmul_kernel
from cattorch.util.instruction.optimized_normalization import (
    BATCH_NORM_KERNEL, FAST_LAYER_NORM_KERNEL, FAST_SOFTMAX_KERNEL, LAYER_NORM_KERNEL,
    LOG_SOFTMAX_KERNELS, MEAN_KERNEL, RMS_NORM_KERNEL, SOFTMAX_KERNELS,
)
from cattorch.util.instruction.optimized_moe import (
    EXPERT_FAMILY_MOE_KERNEL, ROUTED_EXPERTS_KERNEL, STACKED_SWIGLU_MOE_KERNEL,
)
from cattorch.util.instruction.optimized_tensor import (
    EMBEDDING_KERNEL, MASKED_FILL_KERNEL, TRANSPOSE_KERNELS,
)
from cattorch.util.instruction.generic import (
    INDEX_MAP_KERNEL, REDUCTION_KERNELS, REDUCTIONS,
)


# A kernel's instruction class, or a function that picks one from the arguments.
KernelFactory = Callable[..., _KernelInstruction]
Validator = Callable[[torch.fx.Node, FastConfig, bool], str | None]
Observation = tuple[torch.Tensor, torch.Tensor, str]
Observer = Callable[[object, tuple, dict, object], Iterable[Observation]]
PrivateModuleAdapter = Callable[[torch.nn.Module, AdapterContext], torch.nn.Module]


@dataclass(frozen=True)
class OperationRule:
    """Internal contract for one captured operation.

    ``lowering_target`` lets a semantic/custom operator intentionally reuse a
    proven kernel implementation without teaching that kernel another PyTorch
    spelling. It is deliberately private while the extension API settles.
    """

    target: str
    exact_kernel: KernelFactory | None = None
    fast_kernel: KernelFactory | None = None
    fast_flag: Literal["activations", "layer_norm", "softmax"] | None = None
    validator: Validator | None = None
    observer: Observer | None = None
    lowering_target: object | None = None

    def kernel(self, *, fast: bool, config: FastConfig) -> KernelFactory | None:
        if fast and self.fast_kernel is not None:
            enabled = self.fast_flag is None or bool(getattr(config, self.fast_flag))
            if enabled:
                return self.fast_kernel
        return self.exact_kernel


def eval_mode_copy_warning() -> None:
    warnings.warn(
        "the model has modules in training mode; it was exported in evaluation "
        "mode (dropout disabled, batch norm using running statistics)",
        UserWarning,
        stacklevel=3,
    )


class OperatorRegistry:
    """Cloneable internal registry used by graph analysis and compilation."""

    def __init__(self) -> None:
        self._operations: dict[str, OperationRule] = {}
        self._module_adapters: dict[type[torch.nn.Module], PrivateModuleAdapter] = {}
        self._expert_families: dict[int, object] = {}
        self._next_expert_family = 0
        # Targets given an approximate kernel through ``register_kernel``.
        self._approximate_targets: set[str] = set()

    def for_export(self, *, adapters=(), kernels=()) -> "OperatorRegistry":
        """A copy with an export's adapters and kernels registered."""
        registry = self.clone()
        for adapter in adapters:
            registry.register_adapter(adapter)
        for kernel in kernels:
            registry.register_kernel(kernel)
        return registry

    def clone(self) -> OperatorRegistry:
        result = OperatorRegistry()
        result._operations = self._operations.copy()
        result._module_adapters = self._module_adapters.copy()
        result._approximate_targets = set(self._approximate_targets)
        return result

    @property
    def expert_families(self):
        return self._expert_families

    def expert_family(self, handle: int):
        try:
            return self._expert_families[handle]
        except KeyError as error:
            raise ValueError(f"unknown ExpertFamily handle {handle}") from error

    def _register_expert_family(self, family) -> int:
        for handle, registered in self._expert_families.items():
            if registered is family:
                return handle
        handle = self._next_expert_family
        self._next_expert_family += 1
        self._expert_families[handle] = family
        return handle

    def register_operation(self, rule: OperationRule) -> None:
        if not rule.target:
            raise ValueError("operation target must be non-empty")
        self._operations[rule.target] = rule

    def register_kernel(self, kernel: Kernel) -> None:
        """Register a context ``Kernel``, merging into any existing rule.

        An exact kernel replaces the rule's exact lowering; an approximate
        one replaces its fast lowering and flag. The observer,
        ``lowering_target``, registry validator and the other variant are
        kept, unlike ``register_operation`` which replaces the whole rule.
        """
        rule = self._operations.get(kernel.target_name)
        if rule is None:
            rule = OperationRule(target=kernel.target_name)
        instruction = kernel.instruction_class()
        if not kernel.approximate:
            # An exact kernel also replaces the built-in approximate one, so
            # fast mode runs it too, unless an approximate kernel was passed.
            fast_kernel = (
                rule.fast_kernel if kernel.target_name in self._approximate_targets else None
            )
            rule = replace(rule, exact_kernel=instruction, fast_kernel=fast_kernel,
                           fast_flag=rule.fast_flag if fast_kernel else None)
        else:
            self._approximate_targets.add(kernel.target_name)
            rule = replace(
                rule, fast_kernel=instruction,
                fast_flag=cast(Literal["activations", "layer_norm", "softmax"] | None,
                               kernel.fast_flag),
            )
        self._operations[kernel.target_name] = rule

    def operation(self, target: str) -> OperationRule | None:
        return self._operations.get(target)

    def replaced_targets(self, *, fast: bool, config: FastConfig) -> frozenset[str]:
        """Operations whose kernel here isn't the library's own, such as a
        kernel passed to an export."""
        default = _DEFAULT_REGISTRY
        return frozenset(
            target for target in self._operations
            if self.kernel(target, fast=fast, config=config)
            is not default.kernel(target, fast=fast, config=config)
        )

    def kernel(
        self, target: str, *, fast: bool, config: FastConfig,
    ) -> KernelFactory | None:
        rule = self.operation(target)
        return None if rule is None else rule.kernel(fast=fast, config=config)

    def validate(
        self, node: torch.fx.Node, *, fast: bool, config: FastConfig,
        dynamic_axes=None, resolve_weight=None,
    ) -> str | None:
        """A reason ``node`` can't be lowered, or ``None``.

        ``dynamic_axes`` and ``resolve_weight`` give a kernel's validator the
        same facts its ``emit`` gets: variable-length axes and stored values.
        """
        rule = self.operation(str(node.target))
        if rule is None:
            return None
        detail = None if rule.validator is None else rule.validator(node, config, fast)
        if detail is not None:
            return detail
        # A context kernel validates only when it is the one that will run.
        selected = rule.kernel(fast=fast, config=config)
        validate_node = getattr(selected, "validate_node", None)
        return None if validate_node is None else validate_node(
            node, dynamic_axes=dynamic_axes, resolve_weight=resolve_weight,
        )

    def observe(
        self, target: str, func: object, args: tuple, kwargs: dict, result: object,
    ):
        rule = self.operation(target)
        observer = (
            _DECOMPOSED_OBSERVERS.get(target)
            if rule is None or rule.observer is None else rule.observer
        )
        return () if observer is None else tuple(
            observer(func, args, kwargs, result)
        )

    def register_module_adapter(
        self, module_type: type[torch.nn.Module], adapter: Callable[[torch.nn.Module], torch.nn.Module],
    ) -> None:
        """Register a private one-argument adapter compatibility hook."""
        self._module_adapters[module_type] = lambda module, _context: adapter(module)

    def register_adapter(self, adapter: PublicModuleAdapter) -> None:
        """Register one scoped public adapter on this export-owned registry."""
        if not isinstance(adapter, PublicModuleAdapter):
            raise TypeError("adapter must be a cattorch.ModuleAdapter")
        self._module_adapters[adapter.module_type] = adapter.factory

    def adapt_model(
        self,
        model: torch.nn.Module,
        *,
        state_names: frozenset[str] = frozenset(),
        _copy: bool = True,
    ) -> torch.nn.Module:
        """Apply registered adapters to an export-owned model copy.

        ``_copy=False`` adapts ``model`` in place, for a caller that already
        made the copy and doesn't use it otherwise.
        """
        if _copy:
            model = copy.deepcopy(model)
        if any(module.training for module in model.modules()):
            model.eval()
            eval_mode_copy_warning()
        prepare_huggingface_models(model)
        replacements: dict[int, torch.nn.Module] = {}

        def adapt(module: torch.nn.Module, path: str) -> torch.nn.Module:
            identity = id(module)
            if identity in replacements:
                return replacements[identity]
            replacements[identity] = module
            # named_children() suppresses aliases. Visit every registered
            # reference, adapting a shared object once at its first path.
            for name, child in tuple(module._modules.items()):
                if child is None:
                    continue
                child_path = f"{path}.{name}" if path else name
                setattr(module, name, adapt(child, child_path))
            candidates = [
                (module.__class__.mro().index(module_type), module_type, adapter)
                for module_type, adapter in self._module_adapters.items()
                if isinstance(module, module_type) and module_type in module.__class__.mro()
            ]
            if candidates:
                _, _, adapter = min(candidates, key=lambda item: item[0])
                result = adapter(module, AdapterContext(path, state_names))
                if not isinstance(result, torch.nn.Module):
                    raise TypeError(
                        f"module adapter for {path or '<root>'} returned "
                        f"{type(result).__name__}, not nn.Module"
                    )
                replacements[identity] = result
                return result
            # Generic sparse experts are a built-in semantic boundary rather
            # than a user adapter. Register their canonical program only on
            # this export-owned registry so concurrent exports cannot leak
            # template handles into one another.
            from cattorch.markers import apply_export_forward, marked_moe
            from cattorch.experts import (
                ExpertFamily, SparseMoE, _BankedSparseMoE, _specialize_sparse_swiglu,
            )

            def banked(sparse):
                specialized = _specialize_sparse_swiglu(sparse)
                if specialized is not None:
                    return specialized
                return _BankedSparseMoE(sparse, self._register_expert_family(sparse.expert_family))

            apply_export_forward(module)
            marked = marked_moe(module, path)
            if marked is not None:
                sparse, wrapper = marked
                if wrapper is not None:
                    wrapper.routed = banked(sparse)
                    result = wrapper
                else:
                    result = banked(sparse)
                replacements[identity] = result
                return result
            if isinstance(module, ExpertFamily):
                # ``routed_experts`` reads the handle from the export copy.
                module._cattorch_handle = self._register_expert_family(module)
                return module
            if is_huggingface_experts(module):
                experts = convert_huggingface_experts(module, path)
                experts.experts._cattorch_handle = self._register_expert_family(
                    experts.experts,
                )
                replacements[identity] = experts
                return experts
            sparse = module if isinstance(module, SparseMoE) else detect_custom_moe(module, path)
            if sparse is not None:
                result = banked(sparse)
                replacements[identity] = result
                return result
            return module

        return adapt(model, "")


ELEMENTWISE_OPS = frozenset({
    "aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor", "aten.div.Tensor",
    "aten.neg.default", "aten.relu.default", "aten.sigmoid.default",
    "aten.tanh.default", "aten.gelu.default", "aten.silu.default",
    "aten.leaky_relu.default", "aten.elu.default", "aten.pow.Tensor_Scalar",
    "aten.rsqrt.default",
    "aten.exp.default", "aten.log.default", "aten.log10.default",
    "aten.sqrt.default", "aten.abs.default", "aten.floor.default",
    "aten.ceil.default", "aten.reciprocal.default", "aten.erf.default",
    "aten.logical_not.default", "aten.clamp.default", "aten.clamp_min.default",
    "aten.clamp_max.default", "aten.hardtanh.default", "aten.relu6.default",
    "aten.bitwise_not.default", "aten.to.dtype",
    "aten.eq.Tensor", "aten.eq.Scalar", "aten.ne.Tensor", "aten.ne.Scalar",
    "aten.gt.Tensor", "aten.gt.Scalar", "aten.lt.Tensor", "aten.lt.Scalar",
    "aten.ge.Tensor", "aten.ge.Scalar", "aten.le.Tensor", "aten.le.Scalar",
    "aten.maximum.default", "aten.minimum.default",
    "aten.where.self", "aten.where.ScalarSelf", "aten.where.ScalarOther",
    "aten.where.Scalar", "aten.sin.default", "aten.cos.default",
    "aten.tan.default", "aten.asin.default", "aten.acos.default", "aten.atan.default",
    "aten.exp2.default", "aten.log2.default", "aten.erfc.default", "aten.lgamma.default",
})

# Pure data-movement operations. The transpiler describes each call as a
# static index table (``generic.index_map``); only the first argument may vary
# at runtime.
INDEX_MAP_OPS = frozenset({
    "aten.expand.default", "aten.repeat.default", "aten.flip.default",
    "aten.roll.default", "aten.constant_pad_nd.default",
    "aten.reflection_pad1d.default", "aten.reflection_pad2d.default",
    "aten.replication_pad1d.default", "aten.replication_pad2d.default",
    "aten.gather.default", "aten.index_select.default", "aten.index.Tensor",
    "aten._unsafe_index.Tensor", "aten.tril.default", "aten.triu.default",
    "aten.unfold.default", "aten.diagonal.default", "aten.narrow.default",
    "aten.as_strided.default", "aten.upsample_nearest1d.vec",
    "aten.upsample_nearest2d.vec", "aten.upsample_nearest1d.default",
    "aten.upsample_nearest2d.default",
})

MATMUL_OPS = frozenset({
    "aten.matmul.default", "aten.mm.default", "aten.bmm.default",
})


def _shape(value) -> torch.Size:
    return value.meta["val"].shape if hasattr(value, "meta") and "val" in value.meta else torch.Size([])


def _validate_add_sub(node, _config, _fast) -> str | None:
    alpha = node.kwargs.get("alpha", 1)
    return None if alpha == 1 else f"The alpha={alpha!r} variant is not supported"


_REDUCTION_INSTRUCTIONS = {
    kernel.target_name: kernel.instruction_class() for kernel in REDUCTION_KERNELS
}


_MEAN_INSTRUCTION = MEAN_KERNEL.instruction_class()


def _mean_kernel(torch_name, output, *args):
    """Use the tuned mean kernel for consecutive dimensions, else the generic one."""
    dims = args[1].value
    if isinstance(dims, int):
        dims = (dims,)
    ndim = len(args[0].shape)
    if ndim == 0:
        return _REDUCTION_INSTRUCTIONS[str(torch_name)](torch_name, output, *args)
    normalized = sorted({dim % ndim for dim in dims or range(ndim)})
    if normalized and normalized == list(range(normalized[0], normalized[-1] + 1)):
        return _MEAN_INSTRUCTION(torch_name, output, *args)
    return _REDUCTION_INSTRUCTIONS[str(torch_name)](torch_name, output, *args)


def _validate_bitwise_not(node, _config, _fast) -> str | None:
    value = node.args[0].meta.get("val") if hasattr(node.args[0], "meta") else None
    if value is not None and value.dtype == torch.bool:
        return None
    return "Bitwise not is only supported on bool tensors"


def _validate_clamp(node, _config, _fast) -> str | None:
    if any(hasattr(bound, "name") for bound in node.args[1:]):
        return "Clamp bounds must be numbers, not tensors"
    return None


def _validate_convolution(node, _config, _fast) -> str | None:
    dimensions = 1 if str(node.target) == "aten.conv1d.default" else 2
    # Unbatched inputs have the same layout as a batch of one.
    if len(_shape(node.args[0])) not in {dimensions + 1, dimensions + 2}:
        return f"Convolution needs a {dimensions + 1}D or {dimensions + 2}D input"
    padding = node.args[4] if len(node.args) > 4 else [0]
    if isinstance(padding, str):
        return f"padding={padding!r} should have been made explicit"
    return None


def _validate_pool_rank(node, dimensions) -> str | None:
    # Unbatched inputs have the same layout as a batch of one.
    if len(_shape(node.args[0])) not in {dimensions + 1, dimensions + 2}:
        return f"Pooling needs a {dimensions + 1}D or {dimensions + 2}D input"
    return None


def _validate_max_pool(node, _config, _fast) -> str | None:
    dimensions = 1 if str(node.target) == "aten.max_pool1d.default" else 2
    return _validate_pool_rank(node, dimensions)


def _validate_adaptive_pool(node, _config, _fast) -> str | None:
    return _validate_pool_rank(node, 2)


def _validate_avg_pool(node, _config, _fast) -> str | None:
    dimensions = 1 if str(node.target) == "aten.avg_pool1d.default" else 2
    return _validate_pool_rank(node, dimensions)


def _validate_batch_norm(node, _config, _fast) -> str | None:
    training = bool(node.args[5]) if len(node.args) > 5 else False
    if training or node.args[3] is None or node.args[4] is None:
        return "BatchNorm requires evaluation-mode running statistics"
    return None


def _validate_matmul(node, _config, _fast) -> str | None:
    left_shape = _shape(node.args[0])
    right_shape = _shape(node.args[1])
    if len(right_shape) <= 2:
        return None
    left_batch = left_shape[:-2] if len(left_shape) >= 2 else torch.Size([])
    right_batch = right_shape[:-2]
    if len(left_shape) >= 2 and left_batch == right_batch:
        return None
    return (
        "Batched matmul currently requires identical batch dimensions; "
        "broadcasted batches and vector-by-batched matmul are not supported"
    )


def _observe_linear(_func, args, _kwargs, _result):
    if len(args) < 2:
        return ()
    return ((args[0], args[1], "linear"),)


def _observe_addmm(_func, args, _kwargs, _result):
    """``bias + left @ right``, as in GPT-2's ``Conv1D``: ``right`` is the weight."""
    if len(args) < 3:
        return ()
    return ((args[1], args[2], "rhs_matmul"),)


def _matrix_products(left, right) -> list[Observation]:
    """Either side of ``left @ right`` as a weight, one matrix at a time.

    A matrix on the right maps ``left``'s rows; one on the left maps
    ``right``'s columns, as a linear weight would. Batched matrices are
    observed one per batch, with every row their batch broadcasts to them.
    The capture keeps only parameters.
    """
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return []
    if left.ndim == 0 or right.ndim == 0:
        return []
    rows = left if left.ndim > 1 else left.unsqueeze(0)
    columns = right if right.ndim > 1 else right.unsqueeze(-1)
    try:
        batch = torch.broadcast_shapes(rows.shape[:-2], columns.shape[:-2])
    except RuntimeError:
        return []
    observations: list[Observation] = []
    if right.ndim > 1:
        observations.extend(_batched_matrices(rows, right, batch, "rhs_matmul"))
    if left.ndim > 1:
        observations.extend(
            _batched_matrices(columns.transpose(-1, -2), left, batch, "linear")
        )
    return observations


def _batched_matrices(value, weight, batch, orientation) -> list[Observation]:
    """``weight``'s matrices, each with the rows of ``value`` it maps."""
    if weight.ndim == 2:
        return [(value, weight, orientation)]
    leading = (1,) * (len(batch) - (weight.ndim - 2)) + tuple(weight.shape[:-2])
    weight = weight.reshape(*leading, *weight.shape[-2:])
    value = value.expand(*batch, *value.shape[-2:])
    observations = []
    for index in itertools.product(*map(range, leading)):
        # A matrix broadcast along a dimension maps every row along it.
        rows = value[tuple(
            slice(None) if size == 1 and full > 1 else position
            for position, size, full in zip(index, leading, batch)
        )]
        observations.append((rows, weight[index], orientation))
    return observations


def _observe_matmul(_func, args, _kwargs, _result):
    """``x @ weight``, ``weight @ x``, and their batched forms."""
    return _matrix_products(*args[:2]) if len(args) >= 2 else ()


def _observe_baddbmm(_func, args, _kwargs, _result):
    return _matrix_products(*args[1:3]) if len(args) >= 3 else ()


def _observe_mv(_func, args, _kwargs, _result):
    """``weight @ vector``: a linear layer on one row."""
    if len(args) < 2 or not isinstance(args[1], torch.Tensor):
        return ()
    return ((args[1].unsqueeze(0), args[0], "linear"),)


def _observe_addmv(_func, args, _kwargs, _result):
    if len(args) < 3 or not isinstance(args[2], torch.Tensor):
        return ()
    return ((args[2].unsqueeze(0), args[1], "linear"),)


def _einsum_subscripts(equation: str, operands) -> tuple[list[str], str] | None:
    """Operand and output subscripts, with each ``...`` spelled out."""
    equation = equation.replace(" ", "")
    inputs, arrow, output = equation.partition("->")
    terms = inputs.split(",")
    if len(terms) != len(operands):
        return None
    spare = iter(sorted(
        set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") - set(equation)
    ))
    widest = max(
        (operand.ndim - len(term.replace("...", "")) for term, operand in zip(terms, operands)
         if "..." in term),
        default=0,
    )
    broadcast = "".join(next(spare) for _ in range(widest))
    spelled = []
    for term, operand in zip(terms, operands):
        if "..." in term:
            count = operand.ndim - len(term.replace("...", ""))
            term = term.replace("...", broadcast[widest - count:])
        if len(term) != operand.ndim or not term.isalpha():
            return None
        spelled.append(term)
    if not arrow:
        counts = {letter: inputs.count(letter) for letter in "".join(spelled)}
        output = broadcast + "".join(
            sorted(letter for letter, count in counts.items() if count == 1 and letter not in broadcast)
        )
    else:
        output = output.replace("...", broadcast)
    return spelled, output


def _einsum_weight(x, x_terms: str, weight, weight_terms: str, output: str):
    """Observations of ``weight`` as one matrix per batch in an einsum.

    ``weight`` must contract exactly one index with ``x``, put indices
    shared with ``x`` and the output first, and keep the rest, all outputs,
    on one side of the contracted index.
    """
    if len(set(x_terms)) != len(x_terms) or len(set(weight_terms)) != len(weight_terms):
        return []
    contracted = [c for c in weight_terms if c in x_terms and c not in output]
    batch = [c for c in weight_terms if c in x_terms and c in output]
    outputs = [c for c in weight_terms if c not in x_terms]
    if (
        len(contracted) != 1 or not outputs
        or any(c not in output for c in outputs)
        or any(c not in weight_terms and c not in output for c in x_terms)
        or weight_terms[:len(batch)] != "".join(batch)
        or any(x.shape[x_terms.index(c)] != weight.shape[weight_terms.index(c)] for c in batch)
    ):
        return []
    (inner,) = contracted
    matrix_terms = weight_terms[len(batch):]
    if matrix_terms[0] == inner:
        orientation = "rhs_matmul"
    elif matrix_terms[-1] == inner:
        orientation = "linear"
    else:
        return []
    width = x.shape[x_terms.index(inner)]
    rows = [c for c in x_terms if c not in batch and c != inner]
    ordered = x.permute([x_terms.index(c) for c in (*batch, *rows, inner)])
    observations = []
    batches = [weight.shape[index] for index in range(len(batch))]
    for flat in range(math.prod(batches)):
        index = []
        for size in reversed(batches):
            index.append(flat % size)
            flat //= size
        position = tuple(reversed(index))
        matrix = weight[position]
        if matrix.ndim > 2:
            shape = (width, -1) if orientation == "rhs_matmul" else (-1, width)
            try:
                matrix = matrix.view(shape)
            except RuntimeError:
                return []
        observations.append((ordered[position].reshape(-1, width), matrix, orientation))
    return observations


def _observe_einsum(_func, args, _kwargs, _result):
    """A two-operand einsum that applies one operand to the other as a matrix."""
    if len(args) < 2 or not isinstance(args[0], str) or len(args[1]) != 2:
        return ()
    operands = args[1]
    if not all(isinstance(operand, torch.Tensor) for operand in operands):
        return ()
    spelled = _einsum_subscripts(args[0], operands)
    if spelled is None:
        return ()
    (left_terms, right_terms), output = spelled
    left, right = operands
    return (
        *_einsum_weight(left, left_terms, right, right_terms, output),
        *_einsum_weight(right, right_terms, left, left_terms, output),
    )


def _observe_tensordot(_func, args, _kwargs, _result):
    """A matrix contracted over one of its dimensions with the other operand."""
    if len(args) < 4:
        return ()
    left, right, left_dims, right_dims = args[:4]
    observations = []
    for x, x_dims, weight, weight_dims in (
        (left, left_dims, right, right_dims), (right, right_dims, left, left_dims),
    ):
        if (
            not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor)
            or weight.ndim != 2 or len(weight_dims) != 1 or len(x_dims) != 1
        ):
            continue
        inner = weight_dims[0] % 2
        rows = x.movedim(x_dims[0], -1)
        observations.append((rows, weight, "rhs_matmul" if inner == 0 else "linear"))
    return observations


def _observe_bilinear(_func, args, _kwargs, _result):
    """``left`` applied to each output's ``[left, right]`` matrix.

    cattorch lowers the product as one matrix product of ``left`` with every
    output's matrix side by side, so each matrix is grouped over ``left``.
    """
    if len(args) < 3 or not isinstance(args[2], torch.Tensor) or args[2].ndim != 3:
        return ()
    left, weight = args[0], args[2]
    return tuple((left, weight[output], "rhs_matmul") for output in range(weight.shape[0]))


def _observe_native_attention(_func, args, _kwargs, result):
    """``nn.MultiheadAttention``'s fused inference path.

    The packed input projection sees the query, key and value. The output
    projection's input is reconstructed when there is no mask, and kept only
    if it reproduces the operation's result.
    """
    if len(args) < 9:
        return ()
    query, key, value, embed_dim, heads, qkv_weight, qkv_bias, proj_weight, proj_bias = args[:9]
    if not all(
        isinstance(item, torch.Tensor)
        for item in (query, key, value, qkv_weight, proj_weight)
    ):
        return ()
    if query is key and key is value:
        observations = [(query, qkv_weight, "linear")]
    else:
        observations = [
            (source, qkv_weight[index * embed_dim:(index + 1) * embed_dim], "linear")
            for index, source in enumerate((query, key, value))
        ]
    mask = args[9] if len(args) > 9 else None
    if mask is None and query.ndim == 3 and embed_dim % heads == 0:
        biases = (
            qkv_bias.chunk(3) if isinstance(qkv_bias, torch.Tensor) else (None,) * 3
        )
        projected = [
            torch.nn.functional.linear(source, weight, bias)
            for source, weight, bias in zip(
                (query, key, value), qkv_weight.chunk(3), biases,
            )
        ]
        batch = query.shape[0]
        q, k, v = (
            item.reshape(batch, item.shape[1], heads, -1).transpose(1, 2) for item in projected
        )
        attended = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(1, 2).reshape(batch, query.shape[1], embed_dim)
        output = result[0] if isinstance(result, tuple) else result
        replay = torch.nn.functional.linear(attended, proj_weight, proj_bias)
        if (
            isinstance(output, torch.Tensor) and output.shape == replay.shape
            and torch.allclose(output, replay, rtol=1e-4, atol=1e-5)
        ):
            observations.append((attended, proj_weight, "linear"))
    return tuple(observations)


def _observe_recurrent(_func, args, _kwargs, result):
    """``nn.RNN``, ``nn.LSTM`` and ``nn.GRU`` weights that see known inputs.

    Every first-layer input weight reads the sequence. A single-layer,
    one-direction hidden weight reads the previous hidden states, which are
    the output shifted by one step after the initial state.
    """
    if len(args) < 9 or not isinstance(args[0], torch.Tensor):
        return ()
    sequence, hidden, params, biases, layers, _dropout, _train, bidirectional, batch_first = args[:9]
    directions = 2 if bidirectional else 1
    per_layer = 4 if biases else 2
    if len(params) != layers * directions * per_layer:
        return ()  # Projected LSTMs carry another weight per layer.
    observations = [
        (sequence, params[direction * per_layer], "linear") for direction in range(directions)
    ]
    output = result[0] if isinstance(result, tuple) else result
    initial = hidden[0] if isinstance(hidden, (list, tuple)) else hidden
    if layers == 1 and directions == 1 and isinstance(output, torch.Tensor):
        time = 1 if batch_first and sequence.ndim == 3 else 0
        first = initial[0].unsqueeze(time)
        steps = output.narrow(time, 0, output.shape[time] - 1)
        observations.append((torch.cat((first, steps), time), params[1], "linear"))
    return tuple(observations)


def _observe_convolution(func, args, kwargs, _result):
    """Each group of a convolution's kernels, with the windows it reads.

    A convolution is a linear map of every input window (im2col), so each
    group's kernels are observed as one matrix over its input channels and
    kernel positions, in the order storage keeps them. A transposed
    convolution is the plain convolution, with its kernel flipped, of the
    input spread by its stride and padded; flipping that input instead keeps
    the windows in the weight's own kernel order, and only reorders them.

    The windows are returned as views, ``[..., channels, *kernel]``, for the
    capture to read a slice at a time; the orientation tells it the weight
    is a group of output-major (``conv``) or input-major
    (``conv_transpose``) kernels.
    """
    schema = getattr(func, "_schema", None)
    if schema is None:
        return ()
    bound = {
        argument.name: argument.default_value
        for argument in schema.arguments if argument.has_default_value()
    }
    bound.update(zip((argument.name for argument in schema.arguments), args))
    bound.update(kwargs)
    value, weight = bound.get("input"), bound.get("weight")
    if (
        not isinstance(value, torch.Tensor) or not isinstance(weight, torch.Tensor)
        or weight.ndim < 3
    ):
        return ()
    dimensions = weight.ndim - 2
    transposed = bool(bound.get("transposed", "conv_transpose" in str(func)))
    groups = int(bound.get("groups", 1))
    kernel = list(weight.shape[2:])

    def per_dimension(name, default):
        item = bound.get(name, default)
        item = list(item) if isinstance(item, (list, tuple)) else [item]
        return item * dimensions if len(item) == 1 else item

    stride = per_dimension("stride", 1)
    dilation = per_dimension("dilation", 1)
    padding = bound.get("padding", 0)
    if value.ndim == dimensions + 1:
        value = value.unsqueeze(0)
    if value.ndim != dimensions + 2 or len(stride) != dimensions:
        return ()
    if transposed:
        output_padding = per_dimension("output_padding", 0)
        for index in range(dimensions):
            axis = 2 + index
            if stride[index] > 1:
                shape = list(value.shape)
                shape[axis] = (shape[axis] - 1) * stride[index] + 1
                spread = value.new_zeros(shape)
                every = [slice(None)] * value.ndim
                every[axis] = slice(None, None, stride[index])
                spread[tuple(every)] = value
                value = spread
        explicit = per_dimension("padding", 0)
        pads = []
        for index in range(dimensions):
            before = dilation[index] * (kernel[index] - 1) - int(explicit[index])
            pads.append((before, before + int(output_padding[index])))
        stride = [1] * dimensions
    elif padding == "valid":
        pads = [(0, 0)] * dimensions
    elif padding == "same":
        pads = []
        for index in range(dimensions):
            total = dilation[index] * (kernel[index] - 1)
            pads.append((total // 2, total - total // 2))
    else:
        pads = [(int(item), int(item)) for item in per_dimension("padding", 0)]
    flat_pads: list[int] = []
    for before, after in reversed(pads):
        flat_pads += [before, after]
    if any(flat_pads):
        # Negative padding crops, as a transposed convolution's does.
        value = torch.nn.functional.pad(value, flat_pads)
    if transposed:
        value = value.flip(list(range(2, 2 + dimensions)))
    for index in range(dimensions):
        span = dilation[index] * (kernel[index] - 1) + 1
        if value.shape[2 + index] < span:
            return ()
        value = value.unfold(2 + index, span, stride[index])
        if dilation[index] > 1:
            value = value[..., ::dilation[index]]
    # [batch, *positions, channels, *kernel]
    windows = value.permute(
        0, *range(2, 2 + dimensions), 1, *range(2 + dimensions, 2 + 2 * dimensions),
    )
    channels = windows.shape[dimensions + 1] // groups
    outputs = weight.shape[0] // groups
    observations: list[Observation] = []
    for group in range(groups):
        rows = windows.narrow(dimensions + 1, group * channels, channels)
        if transposed:
            kernels = weight.narrow(0, group * channels, channels)
            observations.append((rows, kernels, "conv_transpose"))
        else:
            kernels = weight.narrow(0, group * outputs, outputs)
            observations.append((rows, kernels, "conv"))
    return observations


# Operations the frontend rewrites before lowering, so they have no rule of
# their own, but which eager calibration still sees under their own name.
_DECOMPOSED_OBSERVERS: dict[str, Observer] = {
    "aten.addmm.default": _observe_addmm,
    "aten.baddbmm.default": _observe_baddbmm,
    "aten.mv.default": _observe_mv,
    "aten.addmv.default": _observe_addmv,
    "aten.einsum.default": _observe_einsum,
    "aten.tensordot.default": _observe_tensordot,
    "aten.bilinear.default": _observe_bilinear,
    "aten._native_multi_head_attention.default": _observe_native_attention,
    "aten.lstm.input": _observe_recurrent,
    "aten.gru.input": _observe_recurrent,
    "aten.rnn_tanh.input": _observe_recurrent,
    "aten.rnn_relu.input": _observe_recurrent,
    **{
        target: _observe_convolution
        for target in (
            "aten.conv1d.default", "aten.conv2d.default", "aten.conv3d.default",
            "aten.conv1d.padding", "aten.conv2d.padding", "aten.conv3d.padding",
            "aten.conv_transpose1d.default", "aten.conv_transpose2d.input",
            "aten.conv_transpose3d.input", "aten.convolution.default",
        )
    },
}


def _observe_stacked_swiglu(_func, args, _kwargs, _result):
    """Collect true selected-expert inputs, including down activations."""
    if len(args) < 7:
        return ()
    value, router, gate, up, down, top_k, normalize = args[:7]
    if not all(isinstance(item, torch.Tensor) for item in (value, router, gate, up, down)):
        return ()
    from cattorch.experts import RoutingScores, select_routes

    flat = value.detach().reshape(-1, value.shape[-1])
    probabilities = torch.softmax(
        torch.nn.functional.linear(flat.float(), router.float()), dim=-1,
    )
    ids, _ = select_routes(
        RoutingScores(probabilities, probabilities, bool(normalize)), int(top_k),
    )
    flat_ids = ids.reshape(-1, int(top_k))
    observations = [(flat, router, "linear")]
    for expert_id in range(gate.shape[0]):
        token = torch.nonzero(
            flat_ids == expert_id, as_tuple=False,
        )[:, 0].unique()
        if token.numel() == 0:
            continue
        selected = flat.index_select(0, token)
        observations.extend((
            (selected, gate[expert_id], "linear"),
            (selected, up[expert_id], "linear"),
        ))
        hidden = (
            torch.nn.functional.silu(torch.nn.functional.linear(selected, gate[expert_id]))
            * torch.nn.functional.linear(selected, up[expert_id])
        )
        observations.append((hidden, down[expert_id], "linear"))
    return tuple(observations)


def _observe_expert_family(_func, args, _kwargs, _result):
    """Replay only routed experts so ordinary kernel observers see true inputs."""
    if len(args) < 8:
        return ()
    value, selection, combination, banks, handle, top_k, normalize, _ = args[:8]
    if not isinstance(value, torch.Tensor):
        return ()
    from cattorch.experts import RoutingScores, select_routes
    ids, _weights = select_routes(
        RoutingScores(selection, combination, bool(normalize)), int(top_k),
    )
    return _observe_routed_family(value, ids, banks, handle)


def _observe_routed_experts(_func, args, _kwargs, _result):
    if len(args) < 6:
        return ()
    value, ids, _weights, banks, handle, _ = args[:6]
    if not isinstance(value, torch.Tensor) or not isinstance(ids, torch.Tensor):
        return ()
    return _observe_routed_family(value, ids, banks, handle)


def _observe_routed_family(value, ids, banks, handle):
    from cattorch.experts import _ACTIVE_EXPERT_FAMILIES
    family = _ACTIVE_EXPERT_FAMILIES.get().get(int(handle))
    if family is None:
        return ()
    flat = value.detach().reshape(-1, value.shape[-1])
    flat_ids = ids.reshape(-1, ids.shape[-1])
    observations = []
    from cattorch.frontend import capture_model
    captured = getattr(family, "_gptq_captured_template", None)
    if captured is None:
        captured = capture_model(
            family.template, (family.example_input,), frontend="fx",
        )
    template_state = {}
    template_state.update(dict(
        family.template.named_parameters(remove_duplicate=False),
    ))
    template_state.update(dict(
        family.template.named_buffers(remove_duplicate=False),
    ))
    locations = tuple(dict.fromkeys(family._bank_names.values()))
    location_indices = {location: index for index, location in enumerate(locations)}
    captured_locations = {}
    for captured_name, captured_value in captured.state_inputs.items():
        for logical_name, candidate in template_state.items():
            if (
                candidate.untyped_storage().data_ptr()
                == captured_value.untyped_storage().data_ptr()
                and candidate.storage_offset() == captured_value.storage_offset()
            ):
                captured_locations[captured_name] = location_indices[
                    family._bank_names[logical_name]
                ]
                break

    class _ExpertInterpreter(torch.fx.Interpreter):
        def __init__(self, replacements):
            super().__init__(captured.graph_module)
            self.replacements = replacements

        def get_attr(self, target, args, kwargs):
            if target in self.replacements:
                return self.replacements[target]
            return super().get_attr(target, args, kwargs)

        def call_function(self, target, args, kwargs):
            result = super().call_function(target, args, kwargs)
            if str(target) == "aten.linear.default" and len(args) >= 2:
                observations.append((args[0].detach(), args[1], "linear"))
            return result

    for expert_id in range(family.expert_count):
        token = torch.nonzero(
            flat_ids == expert_id, as_tuple=False,
        )[:, 0].unique()
        if token.numel() == 0:
            continue
        replacements = {
            name: banks[index][expert_id]
            for name, index in captured_locations.items()
        }
        _ExpertInterpreter(replacements).run(flat.index_select(0, token))
    return tuple(observations)


def _build_default_registry() -> OperatorRegistry:
    registry = OperatorRegistry()
    exact: dict[str, KernelFactory] = {
        **{
            kernel.target_name: kernel.instruction_class()
            for kernel in (
                EXPERT_FAMILY_MOE_KERNEL, ROUTED_EXPERTS_KERNEL, STACKED_SWIGLU_MOE_KERNEL,
            )
        },
        "aten.linear.default": LINEAR_KERNEL.instruction_class(),
        **{
            kernel.target_name: kernel.instruction_class()
            for kernel in (*SOFTMAX_KERNELS, *LOG_SOFTMAX_KERNELS)
        },
        "aten.mean.dim": _mean_kernel,
        **{
            operation: _REDUCTION_INSTRUCTIONS[operation]
            for operation in REDUCTIONS if operation != "aten.mean.dim"
        },
        **{operation: INDEX_MAP_KERNEL.instruction_class() for operation in INDEX_MAP_OPS},
        **{
            kernel.target_name: kernel.instruction_class()
            for kernel in (
                LAYER_NORM_KERNEL, RMS_NORM_KERNEL, BATCH_NORM_KERNEL,
                EMBEDDING_KERNEL, MASKED_FILL_KERNEL, *TRANSPOSE_KERNELS,
            )
        },
        **{
            kernel.target_name: kernel.instruction_class()
            for kernel in (*CONVOLUTION_KERNELS, *POOLING_KERNELS)
        },
        **{
            operation: elementwise_kernel(operation).instruction_class()
            for operation in ELEMENTWISE_OPS
        },
        **{
            operation: matmul_kernel(operation).instruction_class() for operation in MATMUL_OPS
        },
    }
    validators = {
        "aten.add.Tensor": _validate_add_sub,
        "aten.sub.Tensor": _validate_add_sub,
        "aten.clamp.default": _validate_clamp,
        "aten.bitwise_not.default": _validate_bitwise_not,
        "aten.conv1d.default": _validate_convolution,
        "aten.conv2d.default": _validate_convolution,
        "aten.max_pool1d.default": _validate_max_pool,
        "aten.max_pool2d.default": _validate_max_pool,
        "aten.avg_pool1d.default": _validate_avg_pool,
        "aten.avg_pool2d.default": _validate_avg_pool,
        "aten.adaptive_avg_pool2d.default": _validate_adaptive_pool,
        "aten.batch_norm.default": _validate_batch_norm,
        **{operation: _validate_matmul for operation in MATMUL_OPS},
    }
    observers = {
        "cattorch.stacked_swiglu_moe.default": _observe_stacked_swiglu,
        "cattorch.expert_family_moe.default": _observe_expert_family,
        "cattorch.routed_experts.default": _observe_routed_experts,
        "aten.linear.default": _observe_linear,
        "aten.mm.default": _observe_matmul,
        "aten.matmul.default": _observe_matmul,
        "aten.bmm.default": _observe_matmul,
    }
    fast: dict[
        str,
        tuple[
            KernelFactory | None,
            Literal["activations", "layer_norm", "softmax"] | None,
        ],
    ] = {
        "aten.gelu.default": (FAST_GELU_KERNEL.instruction_class(), "activations"),
        "aten.layer_norm.default": (FAST_LAYER_NORM_KERNEL.instruction_class(), "layer_norm"),
        "aten.softmax.int": (FAST_SOFTMAX_KERNEL.instruction_class(), "softmax"),
    }
    for target, kernel in exact.items():
        fast_kernel, fast_flag = fast.get(target, (None, None))
        registry.register_operation(OperationRule(
            target=target,
            exact_kernel=kernel,
            fast_kernel=fast_kernel,
            fast_flag=fast_flag,
            validator=validators.get(target),
            observer=observers.get(target),
        ))
    for kernel in BUILTIN_KERNELS:
        registry.register_kernel(kernel)
    return registry


_DEFAULT_REGISTRY = _build_default_registry()


def default_registry() -> OperatorRegistry:
    return _DEFAULT_REGISTRY


__all__ = [
    "ELEMENTWISE_OPS", "INDEX_MAP_OPS", "MATMUL_OPS", "OperationRule", "OperatorRegistry",
    "default_registry",
]
