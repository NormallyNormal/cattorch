"""Experimental sparse mixture-of-experts building blocks.

These classes provide one eager reference contract for routing and expert
execution.  The FX compiler can therefore recognize a semantic MoE boundary
without depending on a training library's Python token-grouping code.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import torch
from torch import nn

from cattorch.adapters import ModuleAdapter


_ACTIVE_EXPERT_FAMILIES: ContextVar[Mapping[int, "ExpertFamily"]] = ContextVar(
    "cattorch_active_expert_families", default={},
)


@contextmanager
def _expert_family_context(families: Mapping[int, "ExpertFamily"]):
    token = _ACTIVE_EXPERT_FAMILIES.set(families)
    try:
        yield
    finally:
        _ACTIVE_EXPERT_FAMILIES.reset(token)


@dataclass(frozen=True)
class RoutingScores:
    """Selection and combination scores produced by a router.

    ``combination=None`` reuses ``selection``.  Scores are not implicitly
    transformed; callers may construct them with any cattorch-supported FX
    tensor graph.
    """

    selection: torch.Tensor
    combination: torch.Tensor | None = None
    normalize_selected: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.selection, torch.Tensor):
            raise TypeError("RoutingScores.selection must be a tensor")
        if self.combination is not None and not isinstance(self.combination, torch.Tensor):
            raise TypeError("RoutingScores.combination must be a tensor or None")
        if not isinstance(self.normalize_selected, bool):
            raise TypeError("normalize_selected must be a bool")


def select_routes(scores: RoutingScores, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Select routes with deterministic lower-expert-ID tie breaking."""
    selection = scores.selection
    combination = selection if scores.combination is None else scores.combination
    if selection.ndim < 1 or selection.shape[-1] < 1:
        raise ValueError("routing scores must have a nonempty expert dimension")
    if combination.shape != selection.shape:
        raise ValueError("selection and combination score tensors must have equal shapes")
    experts = selection.shape[-1]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= experts:
        raise ValueError(f"top_k must be between 1 and {experts}")
    # Stable sorting preserves the original ascending expert order for ties.
    expert_ids = torch.argsort(
        selection, dim=-1, descending=True, stable=True,
    )[..., :top_k]
    weights = combination.gather(-1, expert_ids)
    if scores.normalize_selected:
        denominator = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(denominator != 0, weights / denominator, torch.zeros_like(weights))
    return expert_ids, weights


def _module_signature(module: nn.Module) -> tuple:
    module_types = tuple(
        (
            name,
            f"{child.__class__.__module__}.{child.__class__.__qualname__}",
            child.training,
        )
        for name, child in module.named_modules()
    )
    parameters = dict(module.named_parameters(remove_duplicate=False))
    buffers = dict(module.named_buffers(remove_duplicate=False))
    state_specs = []
    aliases = {}
    for name, value in (*parameters.items(), *buffers.items()):
        if name in parameters:
            role = ("parameter", value.requires_grad)
        else:
            path, _, leaf = name.rpartition(".")
            owner = module.get_submodule(path) if path else module
            role = ("buffer", leaf not in owner._non_persistent_buffers_set)
        identity = id(value)
        aliases.setdefault(identity, len(aliases))
        state_specs.append((
            name, tuple(value.shape), value.dtype, role, aliases[identity],
        ))
    state_specs = tuple(state_specs)
    return module_types, state_specs


def _bit_identical(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    left_bytes = left.detach().contiguous().view(torch.uint8)
    right_bytes = right.detach().to(left.device).contiguous().view(torch.uint8)
    return torch.equal(left_bytes, right_bytes)


def _expert_graph_signature(module: nn.Module, example_input: torch.Tensor) -> tuple:
    """Capture scalar constants and dataflow, ignoring parameter values."""
    from cattorch.frontend import capture_model

    captured = capture_model(module, (example_input,), frontend="fx")
    named_state = {}
    named_state.update(dict(module.named_parameters(remove_duplicate=False)))
    named_state.update(dict(module.named_buffers(remove_duplicate=False)))

    def state_name(tensor):
        for name, candidate in named_state.items():
            if (
                candidate.untyped_storage().data_ptr()
                == tensor.untyped_storage().data_ptr()
                and candidate.storage_offset() == tensor.storage_offset()
                and tuple(candidate.shape) == tuple(tensor.shape)
            ):
                return name
        raise ValueError("captured expert state could not be mapped to its template name")

    def argument(value):
        if isinstance(value, torch.fx.Node):
            if value.op == "placeholder":
                return ("input",)
            if value.op == "get_attr":
                tensor = captured.state_inputs[value.name]
                return ("state", state_name(tensor), tuple(tensor.shape), tensor.dtype)
            return ("node", value.name)
        if isinstance(value, (tuple, list)):
            return tuple(argument(item) for item in value)
        if isinstance(value, dict):
            return tuple(sorted((key, argument(item)) for key, item in value.items()))
        return ("literal", value)

    names = {}
    result = []
    for node in captured.graph.nodes:
        if node.op in {"placeholder", "get_attr"}:
            continue
        # Node names are deterministic within a capture, but normalize them to
        # positional IDs so harmless source attribute names do not matter.
        names[node.name] = len(names)

        def normalized(value):
            item = argument(value)
            if item and item[0] == "node":
                return ("node", names.get(item[1], item[1]))
            if isinstance(item, tuple):
                return tuple(normalized_part(part) for part in item)
            return item

        def normalized_part(part):
            if isinstance(part, tuple) and part and part[0] == "node":
                return ("node", names.get(part[1], part[1]))
            if isinstance(part, tuple):
                return tuple(normalized_part(value) for value in part)
            return part

        result.append((node.op, str(node.target), normalized(node.args), normalized(node.kwargs)))
    return tuple(result)


class ExpertFamily(nn.Module):
    """One expert program backed by an expert-major tensor bank.

    The template describes the computation once. Every tensor in
    ``stacked_state`` has an additional leading expert dimension. This is the
    representation consumed by Scratch lowering, and avoids keeping one Python
    module (and one copy of its graph) per expert.
    """

    def __init__(
        self,
        template: nn.Module,
        stacked_state: Mapping[str, torch.Tensor],
        *,
        example_input: torch.Tensor,
    ):
        super().__init__()
        if not isinstance(template, nn.Module):
            raise TypeError("template must be an nn.Module")
        if not isinstance(example_input, torch.Tensor):
            raise TypeError("example_input must be a tensor")
        if example_input.ndim != 2 or example_input.shape[0] != 1:
            raise ValueError("example_input must have shape [1, input_width]")

        parameter_names = dict(template.named_parameters(remove_duplicate=False))
        buffer_names = dict(template.named_buffers(remove_duplicate=False))
        persistent_buffers = {}
        for name, value in buffer_names.items():
            path, _, leaf = name.rpartition(".")
            owner = template.get_submodule(path) if path else template
            persistent_buffers[name] = leaf not in owner._non_persistent_buffers_set
        state_names = (*parameter_names, *buffer_names)
        if not state_names:
            raise ValueError("ExpertFamily templates must contain parameter or buffer state")
        if set(stacked_state) != set(state_names):
            missing = sorted(set(state_names) - set(stacked_state))
            unexpected = sorted(set(stacked_state) - set(state_names))
            raise ValueError(
                f"stacked_state keys differ from the template; missing={missing}, "
                f"unexpected={unexpected}"
            )

        expert_count: int | None = None
        self._bank_names: dict[str, tuple[str, int]] = {}
        self._parameter_banks = nn.ParameterList()
        self._buffer_bank_count = 0
        canonical: dict[int, tuple[str, int]] = {}
        for name in state_names:
            source = parameter_names.get(name, buffer_names.get(name))
            assert source is not None
            bank = stacked_state[name]
            if not isinstance(bank, torch.Tensor):
                raise TypeError(f"stacked state {name!r} must be a tensor")
            if bank.ndim < 1 or tuple(bank.shape[1:]) != tuple(source.shape):
                raise ValueError(
                    f"stacked state {name!r} has shape {tuple(bank.shape)}; "
                    f"expected [experts, {', '.join(map(str, source.shape))}]"
                )
            if expert_count is None:
                expert_count = bank.shape[0]
                if expert_count < 1:
                    raise ValueError("ExpertFamily requires at least one expert")
            elif bank.shape[0] != expert_count:
                raise ValueError("all stacked state tensors must have the same expert count")

            # Preserve tied template aliases as one physical bank. Supplying
            # different tensors for tied names would make the eager and Scratch
            # meanings ambiguous, so reject it explicitly.
            identity = id(source)
            if identity in canonical:
                kind, index = canonical[identity]
                existing = (
                    self._parameter_banks[index]
                    if kind == "parameter"
                    else getattr(self, f"_buffer_bank_{index}")
                )
                if not _bit_identical(existing, bank):
                    raise ValueError(
                        f"stacked state {name!r} disagrees with its tied template alias"
                    )
                self._bank_names[name] = (kind, index)
                continue

            if name in parameter_names:
                index = len(self._parameter_banks)
                value = nn.Parameter(bank.detach().clone(), requires_grad=source.requires_grad)
                self._parameter_banks.append(value)
                kind = "parameter"
            else:
                index = self._buffer_bank_count
                persistence = any(
                    persistent_buffers[alias]
                    for alias, candidate in buffer_names.items()
                    if id(candidate) == identity
                )
                self.register_buffer(
                    f"_buffer_bank_{index}", bank.detach().clone(),
                    persistent=persistence,
                )
                self._buffer_bank_count += 1
                kind = "buffer"
            canonical[identity] = (kind, index)
            self._bank_names[name] = (kind, index)

        # Deliberately keep the canonical program outside the module registry:
        # its values are examples, not another stored expert. ``functional_call``
        # supplies every state tensor from the registered banks.
        object.__setattr__(self, "_template", copy.deepcopy(template))
        super().train(self._template.training)
        assert expert_count is not None
        self._expert_count = expert_count
        self._state_order = tuple(state_names)
        probe_banks = [value.detach().clone() for value in self.unique_banks()]
        probe_before = [value.clone() for value in probe_banks]
        probe_state = self._state_from_banks(probe_banks, 0)
        with torch.no_grad():
            output = torch.func.functional_call(
                self._template, probe_state, (example_input,),
                strict=False, tie_weights=True,
            )
        if any(
            not _bit_identical(before, after)
            for before, after in zip(probe_before, probe_banks)
        ):
            raise ValueError("expert templates must not mutate parameter or buffer state")
        if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != 1:
            raise ValueError("expert template must return one tensor shaped [1, output_width]")
        if output.dtype != example_input.dtype:
            raise ValueError("expert templates must preserve the input dtype")
        self._output_width = output.shape[-1]
        self._output_dtype = output.dtype
        self.register_buffer(
            "_example_input", example_input.detach().clone(), persistent=False,
        )

    @staticmethod
    def _validate_modules(experts: Sequence[nn.Module], example_input: torch.Tensor) -> None:
        if not experts:
            raise ValueError("ExpertFamily requires at least one expert")
        signature = _module_signature(experts[0])
        graph_signature = _expert_graph_signature(experts[0], example_input)
        reference = experts[0](example_input)
        for index, expert in enumerate(experts[1:], 1):
            candidate = _module_signature(expert)
            if candidate != signature:
                raise ValueError(
                    f"expert {index} is not structurally compatible with expert 0"
                )
            if _expert_graph_signature(expert, example_input) != graph_signature:
                raise ValueError(
                    f"expert {index} has different dataflow or scalar constants from expert 0"
                )
            output = expert(example_input)
            if not isinstance(output, torch.Tensor) or output.shape != reference.shape:
                raise ValueError(f"expert {index} output is incompatible with expert 0")

    @classmethod
    def from_modules(
        cls,
        experts: Sequence[nn.Module],
        *,
        example_input: torch.Tensor,
        copy_modules: bool = True,
    ) -> "ExpertFamily":
        values = list(experts)
        if copy_modules:
            values = copy.deepcopy(values)
        cls._validate_modules(values, example_input)
        template = values[0]
        names = (
            *dict(template.named_parameters(remove_duplicate=False)),
            *dict(template.named_buffers(remove_duplicate=False)),
        )
        banks = {}
        for name in names:
            tensors = []
            for expert in values:
                state = dict(expert.named_parameters(remove_duplicate=False))
                state.update(dict(expert.named_buffers(remove_duplicate=False)))
                tensors.append(state[name].detach())
            banks[name] = torch.stack(tensors)
        return cls(template, banks, example_input=example_input)

    @classmethod
    def from_stacked(
        cls,
        template: nn.Module,
        stacked_state: Mapping[str, torch.Tensor],
        *,
        example_input: torch.Tensor,
    ) -> "ExpertFamily":
        return cls(template, stacked_state, example_input=example_input)

    @property
    def template(self) -> nn.Module:
        return self._template

    def _bank(self, name: str) -> torch.Tensor:
        kind, index = self._bank_names[name]
        if kind == "parameter":
            return self._parameter_banks[index]
        return getattr(self, f"_buffer_bank_{index}")

    @property
    def stacked_state(self) -> Mapping[str, torch.Tensor]:
        return MappingProxyType({name: self._bank(name) for name in self._state_order})

    def unique_banks(self) -> tuple[torch.Tensor, ...]:
        """Physical banks in stable custom-op argument order."""
        result = []
        for kind, index in dict.fromkeys(self._bank_names.values()):
            result.append(
                self._parameter_banks[index]
                if kind == "parameter"
                else getattr(self, f"_buffer_bank_{index}")
            )
        return tuple(result)

    def _state_from_banks(
        self, banks: Sequence[torch.Tensor], expert_id: int,
    ) -> dict[str, torch.Tensor]:
        order = tuple(dict.fromkeys(self._bank_names.values()))
        locations = {location: index for index, location in enumerate(order)}
        state = {}
        emitted = set()
        for name in self._state_order:
            location = self._bank_names[name]
            if location in emitted:
                continue
            emitted.add(location)
            state[name] = banks[locations[location]][expert_id]
        return state

    @property
    def expert_count(self) -> int:
        return self._expert_count

    @property
    def output_width(self) -> int:
        return self._output_width

    @property
    def example_input(self) -> torch.Tensor:
        return self._example_input

    def train(self, mode: bool = True):
        super().train(mode)
        self._template.train(mode)
        return self

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        self._template._apply(fn, recurse=recurse)
        return self

    def forward(self, expert_id: int, value: torch.Tensor) -> torch.Tensor:
        if isinstance(expert_id, bool) or not isinstance(expert_id, int):
            raise TypeError("expert_id must be an integer")
        if not 0 <= expert_id < self.expert_count:
            raise IndexError("expert_id is outside the expert bank")
        state = {}
        emitted = set()
        for name in self._state_order:
            location = self._bank_names[name]
            if location in emitted:
                continue
            emitted.add(location)
            state[name] = self._bank(name)[expert_id]
        return torch.func.functional_call(
            self._template, state, (value,), strict=False, tie_weights=True,
        )


@torch.library.custom_op("cattorch::expert_family_moe", mutates_args=())
def _expert_family_moe(
    value: torch.Tensor,
    selection: torch.Tensor,
    combination: torch.Tensor,
    banks: Sequence[torch.Tensor],
    family_handle: int,
    top_k: int,
    normalize_selected: bool,
    output_width: int,
) -> torch.Tensor:
    """Opaque semantic boundary for generic banked expert execution."""
    family = _ACTIVE_EXPERT_FAMILIES.get().get(family_handle)
    if family is None:
        raise RuntimeError("generic ExpertFamily executed outside its export context")
    expected_scores = (*value.shape[:-1], family.expert_count)
    if tuple(selection.shape) != expected_scores:
        raise ValueError(
            f"router selection shape must be {expected_scores}, got {tuple(selection.shape)}"
        )
    if combination.shape != selection.shape:
        raise ValueError("selection and combination score tensors must have equal shapes")
    ids, weights = select_routes(
        RoutingScores(selection, combination, normalize_selected), top_k,
    )
    flat = value.reshape(-1, value.shape[-1])
    flat_ids = ids.reshape(-1, top_k)
    flat_weights = weights.reshape(-1, top_k)
    output = flat.new_zeros((flat.shape[0], output_width))
    for expert_id in range(family.expert_count):
        token, route = torch.nonzero(flat_ids == expert_id, as_tuple=True)
        if token.numel() == 0:
            continue
        selected = flat.index_select(0, token)
        state = family._state_from_banks(banks, expert_id)
        expert_output = torch.func.functional_call(
            family.template, state, (selected,), strict=False, tie_weights=True,
        )
        output.index_add_(
            0, token,
            expert_output * flat_weights[token, route].to(
                expert_output.dtype,
            ).unsqueeze(-1),
        )
    return output.reshape(*value.shape[:-1], output_width)


@_expert_family_moe.register_fake
def _expert_family_moe_fake(
    value, selection, combination, banks, family_handle, top_k,
    normalize_selected, output_width,
):
    return value.new_empty((*value.shape[:-1], output_width))


class _BankedSparseMoE(nn.Module):
    """Export-owned form whose Python routing loops are a semantic op."""

    def __init__(self, source: "SparseMoE", family_handle: int):
        super().__init__()
        self.router = source.router
        self.expert_family = source.expert_family
        self.family_handle = family_handle
        self.top_k = source.top_k
        self.normalize_selected = source.normalize_selected

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        raw = self.router(value)
        normalize = self.normalize_selected
        if isinstance(raw, RoutingScores):
            selection = raw.selection
            combination = raw.selection if raw.combination is None else raw.combination
            normalize = raw.normalize_selected
        elif isinstance(raw, torch.Tensor):
            selection = combination = raw
        elif isinstance(raw, tuple) and len(raw) == 2:
            selection, combination = raw
        else:
            raise TypeError(
                "router must return a Tensor, RoutingScores, or a two-Tensor tuple"
            )
        return _expert_family_moe(
            value, selection, combination,
            list(self.expert_family.unique_banks()),
            self.family_handle, self.top_k, normalize,
            self.expert_family.output_width,
        )


def _specialize_sparse_swiglu(source: "SparseMoE") -> "StackedSwiGLUMoE | None":
    """Conservatively recognize exact bias-free softmax SwiGLU semantics."""
    from cattorch.frontend import capture_model

    try:
        router_graph = capture_model(
            source.router.eval(), (source.expert_family.example_input,), frontend="fx",
        )
        router_calls = [
            node for node in router_graph.graph.nodes if node.op == "call_function"
        ]
        if [str(node.target) for node in router_calls] != [
            "aten.linear.default", "aten._softmax.default",
        ] and [str(node.target) for node in router_calls] != [
            "aten.linear.default", "aten.softmax.int",
        ]:
            return None
        linear, softmax = router_calls
        if len(linear.args) > 2 and linear.args[2] is not None:
            return None
        if softmax.args[0] is not linear or int(softmax.args[1]) != -1:
            return None
        router_weight = router_graph.state_inputs[linear.args[1].name]

        family = source.expert_family
        expert_graph = capture_model(
            family.template.eval(), (family.example_input,), frontend="fx",
        )
        calls = [node for node in expert_graph.graph.nodes if node.op == "call_function"]
        linears = [node for node in calls if str(node.target) == "aten.linear.default"]
        silus = [node for node in calls if str(node.target) == "aten.silu.default"]
        multiplies = [node for node in calls if str(node.target) == "aten.mul.Tensor"]
        if (
            len(calls) != 5 or len(linears) != 3
            or len(silus) != 1 or len(multiplies) != 1
        ):
            return None
        if any(len(node.args) > 2 and node.args[2] is not None for node in linears):
            return None
        placeholders = [
            node for node in expert_graph.graph.nodes if node.op == "placeholder"
        ]
        if len(placeholders) != 1:
            return None
        gate = silus[0].args[0]
        product = multiplies[0]
        if gate not in linears or silus[0] not in product.args:
            return None
        up = product.args[1] if product.args[0] is silus[0] else product.args[0]
        if up not in linears:
            return None
        # The stacked executor applies gate and up directly to the token.
        if gate.args[0] is not placeholders[0] or up.args[0] is not placeholders[0]:
            return None
        down = next((node for node in linears if node.args[0] is product), None)
        if down is None:
            return None
        output = next(node for node in expert_graph.graph.nodes if node.op == "output")
        returned = output.args[0]
        if isinstance(returned, (tuple, list)):
            returned = returned[0] if len(returned) == 1 else None
        if returned is not down:
            return None

        captured_to_logical = {}
        template_state = {}
        template_state.update(dict(family.template.named_parameters(remove_duplicate=False)))
        template_state.update(dict(family.template.named_buffers(remove_duplicate=False)))
        for captured_name, value in expert_graph.state_inputs.items():
            for logical_name, candidate in template_state.items():
                if (
                    candidate.untyped_storage().data_ptr()
                    == value.untyped_storage().data_ptr()
                    and candidate.storage_offset() == value.storage_offset()
                ):
                    captured_to_logical[captured_name] = logical_name
                    break

        def bank(node):
            return family._bank(captured_to_logical[node.args[1].name])

        return StackedSwiGLUMoE(
            router_weight, bank(gate), bank(up), bank(down),
            top_k=source.top_k,
            normalize_selected=source.normalize_selected,
        )
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
        return None


class SparseMoE(nn.Module):
    """Dropless token-wise sparse expert execution with generic top-k routing.

    A router may return a tensor, ``RoutingScores``, or a two-tensor
    ``(selection, combination)`` tuple.  Only selected experts execute in the
    eager reference implementation.
    """

    def __init__(
        self,
        router: nn.Module,
        experts: ExpertFamily,
        *,
        top_k: int = 1,
        normalize_selected: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(router, nn.Module):
            raise TypeError("router must be an nn.Module")
        if not isinstance(experts, ExpertFamily):
            raise TypeError("experts must be an ExpertFamily")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if not 1 <= top_k <= experts.expert_count:
            raise ValueError(f"top_k must be between 1 and {experts.expert_count}")
        if not isinstance(normalize_selected, bool):
            raise TypeError("normalize_selected must be a bool")
        self.router = router
        self.expert_family = experts
        self.top_k = top_k
        self.normalize_selected = normalize_selected

    def _scores(self, value: torch.Tensor) -> RoutingScores:
        raw = self.router(value)
        if isinstance(raw, RoutingScores):
            scores = raw
        elif isinstance(raw, torch.Tensor):
            scores = RoutingScores(raw, normalize_selected=self.normalize_selected)
        elif (
            isinstance(raw, tuple)
            and len(raw) == 2
            and all(isinstance(item, torch.Tensor) for item in raw)
        ):
            scores = RoutingScores(
                raw[0], raw[1], normalize_selected=self.normalize_selected,
            )
        else:
            raise TypeError(
                "router must return a Tensor, RoutingScores, or a two-Tensor tuple"
            )
        expected = (*value.shape[:-1], self.expert_family.expert_count)
        if tuple(scores.selection.shape) != expected:
            raise ValueError(
                f"router selection shape must be {expected}, "
                f"got {tuple(scores.selection.shape)}"
            )
        combination = scores.selection if scores.combination is None else scores.combination
        if combination.shape != scores.selection.shape:
            raise ValueError("selection and combination score tensors must have equal shapes")
        return scores

    def routes(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return select_routes(self._scores(value), self.top_k)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim < 1:
            raise ValueError("SparseMoE input must have a feature dimension")
        expert_ids, weights = self.routes(value)
        flat = value.reshape(-1, value.shape[-1])
        flat_ids = expert_ids.reshape(-1, self.top_k)
        flat_weights = weights.reshape(-1, self.top_k)
        if flat.shape[0] == 0:
            return torch.empty(
                (*value.shape[:-1], self.expert_family.output_width),
                dtype=self.expert_family._output_dtype,
                device=value.device,
            )
        output: torch.Tensor | None = None
        for expert_id in range(self.expert_family.expert_count):
            token, route = torch.nonzero(
                flat_ids == expert_id, as_tuple=True,
            )
            if token.numel() == 0:
                continue
            selected = flat.index_select(0, token)
            expert_output = self.expert_family(expert_id, selected)
            if expert_output.ndim != 2 or expert_output.shape[0] != token.shape[0]:
                raise ValueError("experts must preserve the flattened token dimension")
            if output is None:
                output = expert_output.new_zeros((flat.shape[0], expert_output.shape[-1]))
            contribution = expert_output * flat_weights[token, route].to(
                expert_output.dtype,
            ).unsqueeze(-1)
            output.index_add_(0, token, contribution)
        if output is None:
            raise RuntimeError("routing selected no experts")
        return output.reshape(*value.shape[:-1], output.shape[-1])


@torch.library.custom_op("cattorch::stacked_swiglu_moe", mutates_args=())
def _stacked_swiglu_moe(
    value: torch.Tensor,
    router_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    top_k: int,
    normalize_selected: bool,
) -> torch.Tensor:
    """Opaque eager reference for the first bank-aware Scratch executor."""
    flat = value.reshape(-1, value.shape[-1])
    probabilities = torch.softmax(
        torch.nn.functional.linear(flat.float(), router_weight.float()), dim=-1,
    )
    expert_ids, weights = select_routes(
        RoutingScores(
            probabilities,
            probabilities,
            normalize_selected=normalize_selected,
        ),
        top_k,
    )
    output = flat.new_zeros((flat.shape[0], down_weight.shape[1]))
    for route in range(top_k):
        route_ids = expert_ids[:, route]
        for expert_id in range(gate_weight.shape[0]):
            token = torch.nonzero(route_ids == expert_id, as_tuple=False).flatten()
            if token.numel() == 0:
                continue
            selected = flat.index_select(0, token)
            hidden = (
                torch.nn.functional.silu(torch.nn.functional.linear(
                    selected, gate_weight[expert_id],
                ))
                * torch.nn.functional.linear(selected, up_weight[expert_id])
            )
            contribution = torch.nn.functional.linear(
                hidden, down_weight[expert_id],
            )
            contribution = contribution * weights[token, route].to(
                contribution.dtype,
            ).unsqueeze(-1)
            output.index_add_(0, token, contribution)
    return output.reshape(*value.shape[:-1], output.shape[-1])


@_stacked_swiglu_moe.register_fake
def _stacked_swiglu_moe_fake(
    value, router_weight, gate_weight, up_weight, down_weight,
    top_k, normalize_selected,
):
    return value.new_empty((*value.shape[:-1], down_weight.shape[1]))


class StackedSwiGLUMoE(nn.Module):
    """Top-k MoE with stacked, bias-free SwiGLU experts.

    A faster implementation for experts whose gate, up, and down weights are
    stored as stacked tensors. ``SparseMoE`` handles any supported expert
    structure; use ``stacked_swiglu_moe_adapter`` to convert a compatible
    module to this class.
    """

    def __init__(
        self,
        router_weight: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        *,
        top_k: int = 1,
        normalize_selected: bool = False,
    ) -> None:
        super().__init__()
        shapes = (
            tuple(router_weight.shape), tuple(gate_weight.shape),
            tuple(up_weight.shape), tuple(down_weight.shape),
        )
        if any(value.ndim != expected for value, expected in zip(
            (router_weight, gate_weight, up_weight, down_weight), (2, 3, 3, 3),
        )):
            raise ValueError("stacked SwiGLU weights must have ranks 2, 3, 3, and 3")
        experts, hidden, width = gate_weight.shape
        if shapes[0] != (experts, width):
            raise ValueError("router weight shape must be [experts, width]")
        if shapes[2] != (experts, hidden, width):
            raise ValueError("up weight shape must match gate weight shape")
        if shapes[3] != (experts, width, hidden):
            raise ValueError("down weight shape must be [experts, width, hidden]")
        if not 1 <= top_k <= experts:
            raise ValueError(f"top_k must be between 1 and {experts}")
        self.router_weight = nn.Parameter(router_weight.detach().clone())
        self.gate_weight = nn.Parameter(gate_weight.detach().clone())
        self.up_weight = nn.Parameter(up_weight.detach().clone())
        self.down_weight = nn.Parameter(down_weight.detach().clone())
        self.top_k = top_k
        self.normalize_selected = normalize_selected

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _stacked_swiglu_moe(
            value,
            self.router_weight,
            self.gate_weight,
            self.up_weight,
            self.down_weight,
            self.top_k,
            self.normalize_selected,
        )


def stacked_swiglu_moe_adapter(
    module_type: type[nn.Module],
    *,
    top_k: int | None = None,
    normalize_selected: bool = False,
) -> ModuleAdapter:
    """Build an adapter that converts a stacked SwiGLU MoE module.

    Modules of ``module_type`` must have ``router.weight``, ``gate.weight``,
    ``up.weight``, and ``down.weight``, with expert weights stacked along the
    first dimension. They are converted to ``StackedSwiGLUMoE`` during export.
    """
    def convert(module: nn.Module, context) -> nn.Module:
        try:
            router = module.router.weight
            gate = module.gate.weight
            up = module.up.weight
            down = module.down.weight
        except AttributeError as error:
            raise ValueError(
                f"stacked SwiGLU adapter at {context.module_path or '<root>'} "
                "requires router/gate/up/down weight attributes"
            ) from error
        selected_k = top_k
        if selected_k is None:
            selected_k = int(getattr(module, "top_k", getattr(module, "moe_top_k", 1)))
        return StackedSwiGLUMoE(
            router, gate, up, down,
            top_k=selected_k,
            normalize_selected=normalize_selected,
        )

    return ModuleAdapter(module_type, convert)


__all__ = [
    "ExpertFamily", "RoutingScores", "SparseMoE", "StackedSwiGLUMoE",
    "select_routes", "stacked_swiglu_moe_adapter",
]
