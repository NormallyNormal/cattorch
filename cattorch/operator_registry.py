"""Private registry for frontend-neutral operation and module lowering rules."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import torch

from cattorch.fast import FastConfig
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
    OptimizedMaskedFillInstruction,
    OptimizedMatMulInstruction,
    OptimizedMeanInstruction,
    OptimizedPoolingInstruction,
    OptimizedRMSNormInstruction,
    RotaryEmbeddingInstruction,
    OptimizedSoftmaxInstruction,
    OptimizedTransposeInstruction,
)
from cattorch.util.instruction.optimized_moe import (
    ExpertFamilyMoEInstruction, StackedSwiGLUMoEInstruction,
)


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
    exact_kernel: type[Instruction] | None = None
    fast_kernel: type[Instruction] | None = None
    fast_flag: Literal["activations", "layer_norm", "softmax"] | None = None
    validator: Validator | None = None
    observer: Observer | None = None
    lowering_target: object | None = None

    def kernel(self, *, fast: bool, config: FastConfig) -> type[Instruction] | None:
        if fast and self.fast_kernel is not None:
            enabled = self.fast_flag is None or bool(getattr(config, self.fast_flag))
            if enabled:
                return self.fast_kernel
        return self.exact_kernel


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
        handle = self._next_expert_family
        self._next_expert_family += 1
        self._expert_families[handle] = family
        return handle

    def register_operation(self, rule: OperationRule) -> None:
        if not rule.target:
            raise ValueError("operation target must be non-empty")
        self._operations[rule.target] = rule

    def operation(self, target: str) -> OperationRule | None:
        return self._operations.get(target)

    def kernel(
        self, target: str, *, fast: bool, config: FastConfig,
    ) -> type[Instruction] | None:
        rule = self.operation(target)
        return None if rule is None else rule.kernel(fast=fast, config=config)

    def validate(self, node: torch.fx.Node, *, fast: bool, config: FastConfig) -> str | None:
        rule = self.operation(str(node.target))
        if rule is None or rule.validator is None:
            return None
        return rule.validator(node, config, fast)

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
            from cattorch.moe import SparseMoE, _BankedSparseMoE, _specialize_sparse_swiglu
            if isinstance(module, SparseMoE):
                specialized = _specialize_sparse_swiglu(module)
                if specialized is not None:
                    replacements[identity] = specialized
                    return specialized
                handle = self._register_expert_family(module.expert_family)
                result = _BankedSparseMoE(module, handle)
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
})

MATMUL_OPS = frozenset({
    "aten.matmul.default", "aten.mm.default", "aten.bmm.default",
})


def _shape(value) -> torch.Size:
    return value.meta["val"].shape if hasattr(value, "meta") and "val" in value.meta else torch.Size([])


def _validate_add_sub(node, _config, _fast) -> str | None:
    alpha = node.kwargs.get("alpha", 1)
    return None if alpha == 1 else f"The alpha={alpha!r} variant is not supported"


def _validate_gelu(node, config, fast) -> str | None:
    approximate = node.args[1] if len(node.args) > 1 else node.kwargs.get("approximate", "none")
    if approximate == "tanh" or (fast and config.activations):
        return None
    return "Exact GELU requires approximate='tanh'; the default erf form is not implemented"


def _validate_mean(node, _config, _fast) -> str | None:
    dims = node.args[1]
    if isinstance(dims, int):
        dims = (dims,)
    ndim = len(_shape(node.args[0]))
    normalized = sorted({dim % ndim for dim in dims})
    if normalized and normalized == list(range(normalized[0], normalized[-1] + 1)):
        return None
    return "Mean reduction dimensions must be consecutive"


def _validate_power(node, _config, _fast) -> str | None:
    exponent = node.args[1]
    if exponent in {0, 2}:
        return None
    return "Only exponents 0 and 2 are currently exact for arbitrary Scratch inputs"


def _validate_convolution(node, _config, _fast) -> str | None:
    dimensions = 1 if str(node.target) == "aten.conv1d.default" else 2
    if len(_shape(node.args[0])) != dimensions + 2:
        return "Unbatched convolution is not supported"
    dilation = node.args[5] if len(node.args) > 5 else [1] * dimensions
    groups = node.args[6] if len(node.args) > 6 else 1
    if any(value != 1 for value in dilation) or groups != 1:
        return "Convolution dilation must be 1 and groups must be 1"
    return None


def _validate_max_pool(node, _config, _fast) -> str | None:
    dimensions = 1 if str(node.target) == "aten.max_pool1d.default" else 2
    if len(_shape(node.args[0])) != dimensions + 2:
        return "Unbatched pooling is not supported"
    dilation = node.args[4] if len(node.args) > 4 else [1] * dimensions
    ceil_mode = node.args[5] if len(node.args) > 5 else False
    if any(value != 1 for value in dilation) or ceil_mode:
        return "Max pooling dilation must be 1 and ceil_mode must be False"
    return None


def _validate_avg_pool(node, _config, _fast) -> str | None:
    dimensions = 1 if str(node.target) == "aten.avg_pool1d.default" else 2
    if len(_shape(node.args[0])) != dimensions + 2:
        return "Unbatched pooling is not supported"
    ceil_mode = node.args[4] if len(node.args) > 4 else False
    count_include_pad = node.args[5] if len(node.args) > 5 else True
    divisor_override = node.args[6] if len(node.args) > 6 else None
    if ceil_mode or not count_include_pad or divisor_override is not None:
        return (
            "Average pooling requires ceil_mode=False, count_include_pad=True, "
            "and no divisor_override"
        )
    return None


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


def _validate_rotary_embedding(node, _config, _fast) -> str | None:
    value_shape = _shape(node.args[0])
    if not value_shape or value_shape[-1] % 2:
        return "Rotary embedding requires an even final dimension"
    try:
        output_shape = torch.broadcast_shapes(*(_shape(arg) for arg in node.args[:3]))
    except RuntimeError:
        return "Rotary cosine and sine must broadcast with the input"
    if tuple(output_shape) != tuple(value_shape):
        return "Rotary cosine and sine may not expand the input shape"
    return None


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
    from cattorch.moe import (
        RoutingScores, _ACTIVE_EXPERT_FAMILIES, select_routes,
    )
    family = _ACTIVE_EXPERT_FAMILIES.get().get(int(handle))
    if family is None:
        return ()
    ids, _weights = select_routes(
        RoutingScores(selection, combination, bool(normalize)), int(top_k),
    )
    flat = value.detach().reshape(-1, value.shape[-1])
    flat_ids = ids.reshape(-1, int(top_k))
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
    exact: dict[str, type[Instruction]] = {
        "cattorch.rotary_embedding.default": RotaryEmbeddingInstruction,
        "cattorch.expert_family_moe.default": ExpertFamilyMoEInstruction,
        "cattorch.stacked_swiglu_moe.default": StackedSwiGLUMoEInstruction,
        "aten.linear.default": LinearInstruction,
        "aten.softmax.int": OptimizedSoftmaxInstruction,
        "aten.mean.dim": OptimizedMeanInstruction,
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
        "cattorch.rotary_embedding.default": _validate_rotary_embedding,
        "aten.add.Tensor": _validate_add_sub,
        "aten.sub.Tensor": _validate_add_sub,
        "aten.gelu.default": _validate_gelu,
        "aten.mean.dim": _validate_mean,
        "aten.pow.Tensor_Scalar": _validate_power,
        "aten.conv1d.default": _validate_convolution,
        "aten.conv2d.default": _validate_convolution,
        "aten.max_pool1d.default": _validate_max_pool,
        "aten.max_pool2d.default": _validate_max_pool,
        "aten.avg_pool1d.default": _validate_avg_pool,
        "aten.avg_pool2d.default": _validate_avg_pool,
        "aten.batch_norm.default": _validate_batch_norm,
        **{operation: _validate_matmul for operation in MATMUL_OPS},
    }
    observers = {
        "cattorch.stacked_swiglu_moe.default": _observe_stacked_swiglu,
        "cattorch.expert_family_moe.default": _observe_expert_family,
        "aten.linear.default": _observe_linear,
        "aten.mm.default": _observe_rhs_matmul,
        "aten.matmul.default": _observe_rhs_matmul,
    }
    fast: dict[
        str,
        tuple[
            type[Instruction] | None,
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
    return registry


_DEFAULT_REGISTRY = _build_default_registry()


def default_registry() -> OperatorRegistry:
    return _DEFAULT_REGISTRY


__all__ = [
    "ELEMENTWISE_OPS", "MATMUL_OPS", "OperationRule", "OperatorRegistry",
    "default_registry",
]
