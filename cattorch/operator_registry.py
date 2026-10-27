"""Private registry for frontend-neutral operation and module lowering rules."""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Literal

import torch

from cattorch.fast import FastConfig
from cattorch.huggingface import (
    convert_huggingface_experts, is_huggingface_experts, prepare_huggingface_models,
)
from cattorch.builtin_kernels import BUILTIN_KERNELS
from cattorch.kernel_api import Kernel
from cattorch.moe_detection import detect_custom_moe
from cattorch.adapters import AdapterContext, ModuleAdapter as PublicModuleAdapter
from cattorch.util.instruction.instruction import Instruction
from cattorch.util.instruction.optimized import (
    FastElementwiseInstruction,
    FastLayerNormInstruction,
    FastSoftmaxInstruction,
    LinearInstruction,
    OptimizedBatchNormInstruction,
    OptimizedConvolutionInstruction,
    OptimizedElementwiseInstruction,
    OptimizedEmbeddingInstruction,
    OptimizedLayerNormInstruction,
    OptimizedLogSoftmaxInstruction,
    OptimizedMaskedFillInstruction,
    OptimizedMatMulInstruction,
    OptimizedMeanInstruction,
    OptimizedPoolingInstruction,
    OptimizedRMSNormInstruction,
    OptimizedSoftmaxInstruction,
    OptimizedTransposeInstruction,
)
from cattorch.util.instruction.optimized_moe import (
    ExpertFamilyMoEInstruction, RoutedExpertsInstruction, StackedSwiGLUMoEInstruction,
)
from cattorch.util.instruction.generic import (
    IndexMapInstruction, REDUCTIONS, ReductionInstruction,
)


# An Instruction subclass, or a function that picks one from the arguments.
KernelFactory = Callable[..., Instruction]
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

    def clone(self) -> OperatorRegistry:
        result = OperatorRegistry()
        result._operations = self._operations.copy()
        result._module_adapters = self._module_adapters.copy()
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
            rule = replace(rule, exact_kernel=instruction)
        else:
            rule = replace(
                rule, fast_kernel=instruction, fast_flag=kernel.fast_flag,
            )
        self._operations[kernel.target_name] = rule

    def operation(self, target: str) -> OperationRule | None:
        return self._operations.get(target)

    def kernel(
        self, target: str, *, fast: bool, config: FastConfig,
    ) -> KernelFactory | None:
        rule = self.operation(target)
        return None if rule is None else rule.kernel(fast=fast, config=config)

    def validate(self, node: torch.fx.Node, *, fast: bool, config: FastConfig) -> str | None:
        rule = self.operation(str(node.target))
        if rule is None:
            return None
        detail = None if rule.validator is None else rule.validator(node, config, fast)
        if detail is not None:
            return detail
        # A context kernel validates only when it is the one that will run.
        selected = rule.kernel(fast=fast, config=config)
        validate_node = getattr(selected, "validate_node", None)
        return None if validate_node is None else validate_node(node)

    def observe(
        self, target: str, func: object, args: tuple, kwargs: dict, result: object,
    ):
        rule = self.operation(target)
        return () if rule is None or rule.observer is None else tuple(
            rule.observer(func, args, kwargs, result)
        )

    def register_module_adapter(
        self, module_type: type[torch.nn.Module], adapter: Callable[[torch.nn.Module], torch.nn.Module],
    ) -> None:
        """Register a private one-argument adapter compatibility hook."""
        self._module_adapters[module_type] = lambda module, _context: adapter(module)

    def register_adapter(self, adapter: PublicModuleAdapter) -> None:
        """Register one scoped public adapter on this export-owned registry."""
        if not isinstance(adapter, PublicModuleAdapter):
            raise TypeError("adapter must be a cattorch.experimental.ModuleAdapter")
        self._module_adapters[adapter.module_type] = adapter.factory

    def adapt_model(
        self,
        model: torch.nn.Module,
        *,
        state_names: frozenset[str] = frozenset(),
    ) -> torch.nn.Module:
        """Apply registered adapters to an export-owned model copy."""
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
            from cattorch.moe import (
                ExpertFamily, SparseMoE, _BankedSparseMoE, _specialize_sparse_swiglu,
            )
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
                specialized = _specialize_sparse_swiglu(sparse)
                if specialized is not None:
                    replacements[identity] = specialized
                    return specialized
                handle = self._register_expert_family(sparse.expert_family)
                result = _BankedSparseMoE(sparse, handle)
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
    "aten.exp2.default", "aten.log2.default", "aten.erfc.default",
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


def _mean_kernel(torch_name, output, *args):
    """Use the tuned mean kernel for consecutive dimensions, else the generic one."""
    dims = args[1].value
    if isinstance(dims, int):
        dims = (dims,)
    ndim = len(args[0].shape)
    if ndim == 0:
        return ReductionInstruction(torch_name, output, *args)
    normalized = sorted({dim % ndim for dim in dims or range(ndim)})
    if normalized and normalized == list(range(normalized[0], normalized[-1] + 1)):
        return OptimizedMeanInstruction(torch_name, output, *args)
    return ReductionInstruction(torch_name, output, *args)


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


def _observe_rhs_matmul(_func, args, _kwargs, _result):
    if len(args) < 2:
        return ()
    return ((args[0], args[1], "rhs_matmul"),)


def _observe_stacked_swiglu(_func, args, _kwargs, _result):
    """Collect true selected-expert inputs, including down activations."""
    if len(args) < 7:
        return ()
    value, router, gate, up, down, top_k, normalize = args[:7]
    if not all(isinstance(item, torch.Tensor) for item in (value, router, gate, up, down)):
        return ()
    from cattorch.moe import RoutingScores, select_routes

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
    from cattorch.moe import RoutingScores, select_routes
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
    from cattorch.moe import _ACTIVE_EXPERT_FAMILIES
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
        "cattorch.expert_family_moe.default": ExpertFamilyMoEInstruction,
        "cattorch.routed_experts.default": RoutedExpertsInstruction,
        "cattorch.stacked_swiglu_moe.default": StackedSwiGLUMoEInstruction,
        "aten.linear.default": LinearInstruction,
        "aten.softmax.int": OptimizedSoftmaxInstruction,
        "aten.log_softmax.int": OptimizedLogSoftmaxInstruction,
        "aten._log_softmax.default": OptimizedLogSoftmaxInstruction,
        "aten.mean.dim": _mean_kernel,
        **{
            operation: ReductionInstruction
            for operation in REDUCTIONS if operation != "aten.mean.dim"
        },
        **{operation: IndexMapInstruction for operation in INDEX_MAP_OPS},
        "aten.layer_norm.default": OptimizedLayerNormInstruction,
        "aten.rms_norm.default": OptimizedRMSNormInstruction,
        "aten.batch_norm.default": OptimizedBatchNormInstruction,
        "aten.embedding.default": OptimizedEmbeddingInstruction,
        "aten.numpy_T.default": OptimizedTransposeInstruction,
        "aten.transpose.int": OptimizedTransposeInstruction,
        "aten.permute.default": OptimizedTransposeInstruction,
        "aten.masked_fill.Scalar": OptimizedMaskedFillInstruction,
        "aten.conv1d.default": OptimizedConvolutionInstruction,
        "aten.conv2d.default": OptimizedConvolutionInstruction,
        "aten.max_pool1d.default": OptimizedPoolingInstruction,
        "aten.max_pool2d.default": OptimizedPoolingInstruction,
        "aten.avg_pool1d.default": OptimizedPoolingInstruction,
        "aten.avg_pool2d.default": OptimizedPoolingInstruction,
        "aten.adaptive_avg_pool2d.default": OptimizedPoolingInstruction,
        **{operation: OptimizedElementwiseInstruction for operation in ELEMENTWISE_OPS},
        **{operation: OptimizedMatMulInstruction for operation in MATMUL_OPS},
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
        "aten.mm.default": _observe_rhs_matmul,
        "aten.matmul.default": _observe_rhs_matmul,
    }
    fast: dict[
        str,
        tuple[
            KernelFactory | None,
            Literal["activations", "layer_norm", "softmax"] | None,
        ],
    ] = {
        "aten.gelu.default": (FastElementwiseInstruction, "activations"),
        "aten.layer_norm.default": (FastLayerNormInstruction, "layer_norm"),
        "aten.softmax.int": (FastSoftmaxInstruction, "softmax"),
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
