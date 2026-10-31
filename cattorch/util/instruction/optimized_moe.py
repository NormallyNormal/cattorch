"""Scratch-native banked sparse mixture-of-experts executors."""

from __future__ import annotations

import math
import operator

import torch

from dataclasses import dataclass, field

from cattorch.kernel_api import Kernel, KernelContext
from cattorch.util.instruction.optimized_common import (
    _static_unrolled_repeat, quick_gelu_of, sigmoid_of, tanh_gelu_inner, tanh_of,
)
from cattorch.util.instruction.optimized_elementwise import _erfc_statements
from cattorch.util.scratch.dsl import (
    add, append, change_var, clear, div, eq, gt, if_, index_of, item,
    if_else, lt, mathop, mod, mul, not_, repeat, replace, set_var, sub, var,
)


# Data movement inside an expert, lowered through static index tables.
_EXPERT_MOVES = frozenset({
    "aten.slice.Tensor", "aten.narrow.default", "aten.select.int",
    "aten.transpose.int", "aten.permute.default", "aten.t.default",
})
# Multi-output moves; each ``getitem`` of one is lowered like a move.
_EXPERT_SPLITS = frozenset({
    "aten.split.Tensor", "aten.split_with_sizes.default", "aten.chunk.default",
    "aten.unbind.int",
})


def _emit_stacked_swiglu_moe(ctx: KernelContext):
    """Generic top-k, full-softmax router with banked SwiGLU experts."""
    inputs = ctx.inputs
    value, router = inputs["value"].list, inputs["router_weight"].list
    gate, up, down = inputs["gate_weight"].list, inputs["up_weight"].list, inputs["down_weight"].list
    value_shape = inputs["value"].shape
    width = value_shape[-1]
    tokens = math.prod(value_shape[:-1]) if len(value_shape) > 1 else 1
    experts = inputs["router_weight"].shape[0]
    hidden_size = inputs["gate_weight"].shape[1]
    output_width = inputs["down_weight"].shape[1]
    top_k = int(ctx.scalars["top_k"])
    normalize_selected = bool(ctx.scalars["normalize_selected"])
    for name in (
        "token", "input start", "expert", "left index", "weight index",
        "sum", "maximum", "denominator", "route", "best", "selected",
        "hidden row", "gate index", "up index", "gate sum", "up sum",
        "feature", "hidden index", "down index", "output index",
        "selected denominator", "gate expert base", "up expert base",
        "down expert base", "route weight",
        *(f"gate index {offset}" for offset in range(1, 5)),
        *(f"up index {offset}" for offset in range(1, 5)),
        *(f"gate sum {offset}" for offset in range(1, 5)),
        *(f"up sum {offset}" for offset in range(1, 5)),
        *(f"down index {offset}" for offset in range(1, 5)),
        *(f"down sum {offset}" for offset in range(1, 5)),
    ):
        ctx.temp_var(name)
    out = ctx.output.list
    logits = ctx.temp_list("_moe logits", experts)
    route_ids = ctx.temp_list("_moe route ids", top_k)
    route_weights = ctx.temp_list("_moe route weights", top_k)
    hidden = ctx.temp_list("_moe hidden", hidden_size)
    router_dot = (
        change_var("sum", mul(
            item(value, var("left index")),
            item(router, var("weight index")),
        )),
        change_var("left index", 1),
        change_var("weight index", 1),
    )
    gate_up_dot = (
        change_var("gate sum", mul(
            item(value, var("left index")),
            item(gate, var("gate index")),
        )),
        change_var("up sum", mul(
            item(value, var("left index")),
            item(up, var("up index")),
        )),
        change_var("left index", 1),
        change_var("gate index", 1),
        change_var("up index", 1),
    )
    down_dot = (
        change_var("sum", mul(
            item(hidden, var("hidden index")),
            item(down, var("down index")),
        )),
        change_var("hidden index", 1),
        change_var("down index", 1),
    )
    normalize = ()
    if normalize_selected and top_k > 1:
        normalize = (
            set_var("selected denominator", 0),
            set_var("route", 1),
            repeat(top_k, (
                change_var(
                    "selected denominator", item(route_weights, var("route")),
                ),
                change_var("route", 1),
            )),
            set_var("route", 1),
            repeat(top_k, (
                replace(
                    route_weights,
                    var("route"),
                    div(
                        item(route_weights, var("route")),
                        var("selected denominator"),
                    ),
                ),
                change_var("route", 1),
            )),
        )
    router_softmax = ()
    route_weight = 1
    if not normalize_selected:
        router_softmax = (
            set_var("denominator", 0),
            set_var("expert", 1),
            repeat(experts, (
                set_var(
                    "sum", mathop("e ^", sub(
                        item(logits, var("expert")), var("maximum"),
                    )),
                ),
                replace(logits, var("expert"), var("sum")),
                change_var("denominator", var("sum")),
                change_var("expert", 1),
            )),
        )
        route_weight = div(var("best"), var("denominator"))
    elif top_k > 1:
        # Renormalizing selected softmax probabilities cancels the global
        # denominator. Select raw logits, exponentiate only the winners,
        # then normalize those k values.
        route_weight = mathop("e ^", sub(var("best"), var("maximum")))
    silu_products = tuple(
        append(hidden, mul(
            div(
                var(f"gate sum {offset}"),
                add(1, mathop(
                    "e ^", mul(-1, var(f"gate sum {offset}")),
                )),
            ),
            var(f"up sum {offset}"),
        ))
        for offset in range(1, 5)
    )
    grouped_gate_up_dot = tuple(
        statement
        for offset in range(1, 5)
        for statement in (
            change_var(f"gate sum {offset}", mul(
                item(value, var("left index")),
                item(gate, var(f"gate index {offset}")),
            )),
            change_var(f"up sum {offset}", mul(
                item(value, var("left index")),
                item(up, var(f"up index {offset}")),
            )),
        )
    ) + (
        change_var("left index", 1),
        *(change_var(f"gate index {offset}", 1) for offset in range(1, 5)),
        *(change_var(f"up index {offset}", 1) for offset in range(1, 5)),
    )
    grouped_hidden = ()
    grouped_hidden_rows = hidden_size - hidden_size % 4
    if grouped_hidden_rows:
        grouped_hidden = (
            repeat(grouped_hidden_rows // 4, (
                set_var("left index", var("input start")),
                set_var("gate index 1", add(
                    var("gate expert base"),
                    add(mul(var("hidden row"), width), 1),
                )),
                set_var("up index 1", add(
                    var("up expert base"),
                    add(mul(var("hidden row"), width), 1),
                )),
                *(set_var(
                    f"gate index {offset}",
                    add(var("gate index 1"), (offset - 1) * width),
                ) for offset in range(2, 5)),
                *(set_var(
                    f"up index {offset}",
                    add(var("up index 1"), (offset - 1) * width),
                ) for offset in range(2, 5)),
                *(set_var(f"gate sum {offset}", 0) for offset in range(1, 5)),
                *(set_var(f"up sum {offset}", 0) for offset in range(1, 5)),
                *_static_unrolled_repeat(width, grouped_gate_up_dot),
                *silu_products,
                change_var("hidden row", 4),
            )),
        )
    scalar_hidden = ()
    if grouped_hidden_rows < hidden_size:
        scalar_hidden = (
            repeat(hidden_size - grouped_hidden_rows, (
                set_var("left index", var("input start")),
                set_var("gate index", add(
                    var("gate expert base"),
                    add(mul(var("hidden row"), width), 1),
                )),
                set_var("up index", add(
                    var("up expert base"),
                    add(mul(var("hidden row"), width), 1),
                )),
                set_var("gate sum", 0),
                set_var("up sum", 0),
                *_static_unrolled_repeat(width, gate_up_dot),
                append(hidden, mul(
                    div(
                        var("gate sum"),
                        add(1, mathop("e ^", mul(-1, var("gate sum")))),
                    ),
                    var("up sum"),
                )),
                change_var("hidden row", 1),
            )),
        )

    def store_output(sum_name, feature_offset=0):
        contribution = mul(var(sum_name), var("route weight"))
        if top_k == 1:
            return (append(out, contribution),)
        output_index = add(
            mul(var("token"), output_width),
            add(var("feature"), feature_offset + 1),
        )
        return (
            set_var("output index", output_index),
            replace(out, var("output index"), add(
                item(out, var("output index")), contribution,
            )),
        )

    grouped_down_dot = tuple(
        change_var(f"down sum {offset}", mul(
            item(hidden, var("hidden index")),
            item(down, var(f"down index {offset}")),
        ))
        for offset in range(1, 5)
    ) + (
        change_var("hidden index", 1),
        *(change_var(f"down index {offset}", 1) for offset in range(1, 5)),
    )
    grouped_output = ()
    grouped_output_rows = output_width - output_width % 4
    if grouped_output_rows:
        grouped_output = (
            repeat(grouped_output_rows // 4, (
                set_var("hidden index", 1),
                set_var("down index 1", add(
                    var("down expert base"),
                    add(mul(var("feature"), hidden_size), 1),
                )),
                *(set_var(
                    f"down index {offset}",
                    add(var("down index 1"), (offset - 1) * hidden_size),
                ) for offset in range(2, 5)),
                *(set_var(f"down sum {offset}", 0) for offset in range(1, 5)),
                *_static_unrolled_repeat(hidden_size, grouped_down_dot),
                *(statement
                  for offset in range(1, 5)
                  for statement in store_output(f"down sum {offset}", offset - 1)),
                change_var("feature", 4),
            )),
        )
    scalar_output = ()
    if grouped_output_rows < output_width:
        scalar_output = (
            repeat(output_width - grouped_output_rows, (
                set_var("hidden index", 1),
                set_var("down index", add(
                    var("down expert base"),
                    add(mul(var("feature"), hidden_size), 1),
                )),
                set_var("sum", 0),
                *_static_unrolled_repeat(hidden_size, down_dot),
                *store_output("sum"),
                change_var("feature", 1),
            )),
        )

    initialize_output = (
        () if top_k == 1
        else (repeat(output_width, (append(out, 0),)),)
    )
    return (
        clear(out),
        set_var("token", 0),
        set_var("input start", 1),
        repeat(tokens, (
            clear(logits),
            set_var("expert", 0),
            set_var("maximum", "-Infinity"),
            repeat(experts, (
                set_var("left index", var("input start")),
                set_var("weight index", add(mul(var("expert"), width), 1)),
                set_var("sum", 0),
                *_static_unrolled_repeat(width, router_dot),
                append(logits, var("sum")),
                if_(gt(var("sum"), var("maximum")), (
                    set_var("maximum", var("sum")),
                )),
                change_var("expert", 1),
            )),
            *router_softmax,
            clear(route_ids),
            clear(route_weights),
            set_var("route", 0),
            repeat(top_k, (
                set_var("expert", 1),
                set_var("best", "-Infinity"),
                set_var("selected", 1),
                repeat(experts, (
                    if_(eq(index_of(route_ids, var("expert")), 0), (
                        if_(gt(item(logits, var("expert")), var("best")), (
                            set_var("best", item(logits, var("expert"))),
                            set_var("selected", var("expert")),
                        )),
                    )),
                    change_var("expert", 1),
                )),
                append(route_ids, var("selected")),
                append(route_weights, route_weight),
                change_var("route", 1),
            )),
            *normalize,
            *initialize_output,
            set_var("route", 1),
            repeat(top_k, (
                set_var("expert", sub(item(route_ids, var("route")), 1)),
                set_var("route weight", item(route_weights, var("route"))),
                set_var(
                    "gate expert base",
                    mul(var("expert"), hidden_size * width),
                ),
                set_var("up expert base", var("gate expert base")),
                set_var(
                    "down expert base",
                    mul(var("expert"), output_width * hidden_size),
                ),
                clear(hidden),
                set_var("hidden row", 0),
                *grouped_hidden,
                *scalar_hidden,
                set_var("feature", 0),
                *grouped_output,
                *scalar_output,
                change_var("route", 1),
            )),
            change_var("token", 1),
            change_var("input start", width),
        )),
    )


STACKED_SWIGLU_MOE_KERNEL = Kernel(
    "cattorch.stacked_swiglu_moe.default", _emit_stacked_swiglu_moe, name="stacked_swiglu_moe",
)


@dataclass
class ExpertProgram:
    """One expert's forward pass, run on the active expert's bank rows."""

    statements: tuple
    #: The private list holding the expert's output.
    output: str
    #: Private lists in declaration order; index tables have ``list_values``.
    lists: list[str] = field(default_factory=list)
    list_values: dict[str, list] = field(default_factory=dict)
    sizes: dict[str, int] = field(default_factory=dict)
    #: Variables only some expert programs need.
    extra_variables: list[str] = field(default_factory=list)
    #: Operations lowered with fast mode's approximations, once per call.
    approximated: list[str] = field(default_factory=list)


def _expert_state_banks(family, captured, bank_lists):
    """Each captured state input's bank list and per-expert shape."""
    unique_locations = tuple(dict.fromkeys(family._bank_names.values()))
    location_index = {location: index for index, location in enumerate(unique_locations)}
    template_state = dict(family.template.named_parameters(remove_duplicate=False))
    template_state.update(dict(family.template.named_buffers(remove_duplicate=False)))
    template_values = {id(value): name for name, value in template_state.items()}
    banks = {}
    for captured_name, value in captured.state_inputs.items():
        logical_name = template_values.get(id(value))
        if logical_name is None:
            # Fake/capture wrappers can replace Tensor Python identities;
            # storage identity remains stable for the underlying state.
            for name in family._state_order:
                candidate = template_state[name]
                if (
                    candidate.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
                    and candidate.storage_offset() == value.storage_offset()
                ):
                    logical_name = name
                    break
        if logical_name is None:
            raise ValueError(f"unable to map expert state {captured_name!r} to its bank")
        location = family._bank_names[logical_name]
        banks[captured_name] = (
            bank_lists[location_index[location]], family._bank(logical_name).shape[1:],
        )
    return banks


def _unary(target, value, fast_activations):
    if target == "aten.relu.default":
        return div(add(value, mathop("abs", value)), 2)
    if target == "aten.sigmoid.default":
        return sigmoid_of(value)
    if target == "aten.silu.default":
        return div(value, add(1, mathop("e ^", mul(-1, value))))
    if target == "aten.tanh.default":
        return tanh_of(value)
    if target == "aten.gelu.default":
        if fast_activations:
            return quick_gelu_of(value)
        return mul(0.5, mul(value, add(1, tanh_of(tanh_gelu_inner(value)))))
    if target == "aten.neg.default":
        return mul(-1, value)
    raise ValueError(f"unsupported generic expert activation {target}")


def expert_program(
    family, bank_lists, *, fast_activations: bool = False, fast_layer_norm: bool = False,
) -> ExpertProgram:
    """Lower ``family``'s expert template, reading its state from ``bank_lists``.

    Raises ``ValueError`` for an operation the expert executor can't run.
    """
    from cattorch.frontend import capture_model

    captured = capture_model(family.template.eval(), (family.example_input,), frontend="fx")
    state_banks = _expert_state_banks(family, captured, bank_lists)
    nodes = list(captured.graph.nodes)
    placeholder = next(node for node in nodes if node.op == "placeholder")
    names = {placeholder.name: "_expert input"}
    shapes = {placeholder.name: tuple(placeholder.meta["val"].shape)}
    statements = []
    declared = ["_expert input"]
    sizes = {"_expert input": math.prod(shapes[placeholder.name])}
    list_values = {}
    extra_variables = []
    approximated: list[str] = []

    def tensor_name(argument):
        if argument.op == "get_attr":
            return state_banks[argument.name][0]
        return names[argument.name]

    def tensor_shape(argument):
        if argument.op == "get_attr":
            return tuple(state_banks[argument.name][1])
        return shapes[argument.name]

    # Nodes whose list is a state bank (a parameter or buffer, or a view of
    # one): the active expert's values start ``active expert * size`` in.
    banked = set()
    copies = {}

    def is_banked(argument):
        return argument.op == "get_attr" or argument.name in banked

    def materialized(argument):
        """``argument``'s list; a bank's active-expert slice is copied first."""
        if not is_banked(argument):
            return tensor_name(argument)
        if argument.name not in copies:
            size = math.prod(tensor_shape(argument))
            copy = f"_expert {len(declared)}"
            declared.append(copy)
            sizes[copy] = size
            statements.extend((
                clear(copy),
                set_var("expert index", 1),
                repeat(size, (
                    append(copy, item(
                        tensor_name(argument),
                        add(mul(var("active expert"), size), var("expert index")),
                    )),
                    change_var("expert index", 1),
                )),
            ))
            copies[argument.name] = copy
        return copies[argument.name]

    def element(argument, index, target_shape=None):
        if hasattr(argument, "op"):
            source = tensor_name(argument)
            shape = tensor_shape(argument)
            size = math.prod(shape)
            local = 1 if size == 1 else index
            if target_shape is not None and size != 1 and tuple(shape) != tuple(target_shape):
                trimmed = tuple(shape)
                while trimmed and trimmed[0] == 1:
                    trimmed = trimmed[1:]
                target_shape = tuple(target_shape)
                if (
                    trimmed
                    and len(trimmed) <= len(target_shape)
                    and trimmed == target_shape[-len(trimmed):]
                ):
                    local = add(mod(sub(index, 1), math.prod(trimmed)), 1)
                else:
                    padded = (1,) * (len(target_shape) - len(shape)) + tuple(shape)
                    try:
                        indices = torch.arange(size).reshape(padded).expand(
                            target_shape,
                        ).reshape(-1)
                    except RuntimeError as error:
                        raise ValueError(
                            "unsupported broadcasting inside generic ExpertFamily"
                        ) from error
                    map_name = f"_expert broadcast {len(list_values)}"
                    list_values[map_name] = (indices + 1).tolist()
                    declared.append(map_name)
                    local = item(map_name, index)
            if is_banked(argument):
                local = add(mul(var("active expert"), size), local)
            return item(source, local)
        return argument

    for node in nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        if target == "aten.dropout.default":
            training = bool(node.args[2]) if len(node.args) > 2 else True
            if training:
                raise ValueError(
                    "unsupported operation inside generic ExpertFamily: "
                    "training-mode dropout"
                )
            names[node.name] = tensor_name(node.args[0])
            shapes[node.name] = tuple(node.meta["val"].shape)
            if is_banked(node.args[0]):
                banked.add(node.name)
            continue
        if target in {
            "aten.view.default", "aten.reshape.default", "aten._unsafe_view.default",
            "aten.flatten.using_ints", "aten.contiguous.default", "aten.clone.default",
            "aten.detach.default", "aten.alias.default",
        } or (
            # Scratch keeps one number type, so float casts change nothing.
            target in {"aten.to.dtype", "aten._to_copy.default"}
            and node.meta["val"].dtype.is_floating_point
            and node.args[0].meta["val"].dtype.is_floating_point
        ):
            names[node.name] = tensor_name(node.args[0])
            shapes[node.name] = tuple(node.meta["val"].shape)
            if is_banked(node.args[0]):
                banked.add(node.name)
            continue
        if target in _EXPERT_SPLITS:
            # Each output is lowered at its ``getitem``.
            shapes[node.name] = tuple(node.args[0].meta["val"].shape)
            continue

        output_name = f"_expert {len(declared)}"
        names[node.name] = output_name
        output_shape = tuple(node.meta["val"].shape)
        shapes[node.name] = output_shape
        declared.append(output_name)
        sizes[output_name] = math.prod(output_shape)
        statements.append(clear(output_name))

        if target in _EXPERT_MOVES or (
            node.target is operator.getitem and str(node.args[0].target) in _EXPERT_SPLITS
        ):
            # A static index table found by running the move on positions.
            parent = node.args[0] if target in _EXPERT_MOVES else node.args[0].args[0]
            source_shape = tensor_shape(parent)
            positions = torch.arange(1, math.prod(source_shape) + 1, dtype=torch.float64)
            positions = positions.reshape(source_shape)
            with torch.no_grad():
                if target in _EXPERT_MOVES:
                    moved = node.target(positions, *node.args[1:], **node.kwargs)
                else:
                    split = node.args[0]
                    moved = split.target(positions, *split.args[1:], **split.kwargs)
                    moved = moved[node.args[1]]
            map_name = f"_expert map {len(list_values)}"
            list_values[map_name] = [int(value) for value in moved.reshape(-1).tolist()]
            declared.append(map_name)
            statements.extend((
                set_var("expert index", 1),
                repeat(moved.numel(), (
                    append(output_name, item(
                        materialized(parent), item(map_name, var("expert index")),
                    )),
                    change_var("expert index", 1),
                )),
            ))
            continue

        if target in {"aten.clamp.default", "aten.clamp_min.default", "aten.clamp_max.default"}:
            low = high = None
            if target == "aten.clamp_min.default":
                low = node.args[1]
            elif target == "aten.clamp_max.default":
                high = node.args[1]
            else:
                low = node.args[1] if len(node.args) > 1 else node.kwargs.get("min")
                high = node.args[2] if len(node.args) > 2 else node.kwargs.get("max")
            if any(hasattr(bound, "op") for bound in (low, high)):
                raise ValueError(
                    "unsupported operation inside generic ExpertFamily: "
                    "clamp with tensor bounds"
                )
            if "expert clamp value" not in extra_variables:
                extra_variables.append("expert clamp value")
            value = var("expert clamp value")
            limits = []
            if low is not None:
                limits.append(if_(lt(value, low), (set_var("expert clamp value", low),)))
            if high is not None:
                limits.append(if_(gt(value, high), (set_var("expert clamp value", high),)))
            statements.extend((
                set_var("expert index", 1),
                repeat(math.prod(output_shape), (
                    set_var("expert clamp value", item(
                        materialized(node.args[0]), var("expert index"),
                    )),
                    *limits,
                    append(output_name, value),
                    change_var("expert index", 1),
                )),
            ))
            continue

        if target == "aten.linear.default":
            source, weight = node.args[:2]
            bias = node.args[2] if len(node.args) > 2 else None
            source_name = materialized(source)
            weight_name = tensor_name(weight)
            rows, inner = tensor_shape(weight)
            input_rows = math.prod(tensor_shape(source)) // inner
            bias_name = None if bias is None else tensor_name(bias)
            # A stored weight or bias is read at the active expert's slice; a
            # computed one holds only this expert's values.
            weight_start = mul(var("active expert"), rows * inner) if is_banked(weight) else 0
            bias_start = (
                mul(var("active expert"), rows) if bias is not None and is_banked(bias) else 0
            )
            statements.extend((
                set_var("expert source base", 1),
                repeat(input_rows, (
                    set_var("expert row", 0),
                    repeat(rows, (
                        set_var(
                            "expert sum", 0 if bias_name is None else item(
                                bias_name, add(bias_start, add(var("expert row"), 1)),
                            ),
                        ),
                        set_var("expert input index", var("expert source base")),
                        set_var("expert weight index", add(
                            weight_start, add(mul(var("expert row"), inner), 1),
                        )),
                        repeat(inner, (
                            change_var("expert sum", mul(
                                item(source_name, var("expert input index")),
                                item(weight_name, var("expert weight index")),
                            )),
                            change_var("expert input index", 1),
                            change_var("expert weight index", 1),
                        )),
                        append(output_name, var("expert sum")),
                        change_var("expert row", 1),
                    )),
                    change_var("expert source base", inner),
                )),
            ))
            continue

        if target in {"aten.matmul.default", "aten.mm.default"} and len(
            tensor_shape(node.args[1]) if hasattr(node.args[1], "op") else (),
        ) == 2:
            # ``x @ W`` with a matrix ``W``, as in a low-rank layer.
            source, weight = node.args[:2]
            source_name = materialized(source)
            weight_name = tensor_name(weight)
            inner, columns = tensor_shape(weight)
            input_rows = math.prod(tensor_shape(source)) // inner
            weight_start = mul(var("active expert"), inner * columns) if is_banked(weight) else 0
            statements.extend((
                set_var("expert source base", 1),
                repeat(input_rows, (
                    set_var("expert row", 0),
                    repeat(columns, (
                        set_var("expert sum", 0),
                        set_var("expert input index", var("expert source base")),
                        set_var("expert weight index", add(
                            weight_start, add(var("expert row"), 1),
                        )),
                        repeat(inner, (
                            change_var("expert sum", mul(
                                item(source_name, var("expert input index")),
                                item(weight_name, var("expert weight index")),
                            )),
                            change_var("expert input index", 1),
                            change_var("expert weight index", columns),
                        )),
                        append(output_name, var("expert sum")),
                        change_var("expert row", 1),
                    )),
                    change_var("expert source base", inner),
                )),
            ))
            continue

        if target == "aten.pow.Tensor_Scalar":
            # The elementwise kernel's exact exponents: integers up to 4, and ±0.5.
            exponent = node.args[1]
            magnitude = abs(exponent)
            if magnitude not in {0, 0.5, 1, 2, 3, 4}:
                raise ValueError(
                    "unsupported operation inside generic ExpertFamily: "
                    f"pow exponent {exponent} (integers from -4 to 4 and ±0.5 work)"
                )
            source_name = materialized(node.args[0])
            element_value = item(source_name, var("expert index"))
            if magnitude == 0:
                power = 1
            elif magnitude == 0.5:
                power = mathop("sqrt", element_value)
            else:
                power = element_value
                for _ in range(int(magnitude) - 1):
                    power = mul(power, element_value)
            statements.extend((
                set_var("expert index", 1),
                repeat(math.prod(output_shape), (
                    append(output_name, div(1, power) if exponent < 0 else power),
                    change_var("expert index", 1),
                )),
            ))
            continue

        if target in {
            "aten.relu.default", "aten.sigmoid.default", "aten.silu.default",
            "aten.tanh.default", "aten.gelu.default", "aten.neg.default",
        }:
            source_name = materialized(node.args[0])
            size = math.prod(output_shape)
            if target == "aten.gelu.default" and fast_activations:
                approximated.append(target)
            approximate = (
                node.args[1] if len(node.args) > 1
                else node.kwargs.get("approximate", "none")
            )
            if (
                target == "aten.gelu.default" and approximate != "tanh"
                and not fast_activations
            ):
                # 0.5 x (1 + erf(x / sqrt(2))) through erfc, as in the
                # elementwise kernel: 1 + erf(z) is 2 - erfc(|z|) or erfc(|z|).
                for name in ("expert gelu value", "expert gelu erfc"):
                    if name not in extra_variables:
                        extra_variables.append(name)
                value = var("expert gelu value")
                erfc = var("expert gelu erfc")
                statements.extend((
                    set_var("expert index", 1),
                    repeat(size, (
                        set_var("expert gelu value", item(
                            source_name, var("expert index"),
                        )),
                        *_erfc_statements(
                            mul(value, 0.7071067811865476), "expert gelu erfc",
                        ),
                        if_else(
                            lt(value, 0),
                            (append(output_name, mul(mul(0.5, value), erfc)),),
                            (append(output_name, mul(value, sub(1, mul(0.5, erfc)))),),
                        ),
                        change_var("expert index", 1),
                    )),
                ))
                continue
            statements.extend((
                set_var("expert index", 1),
                repeat(size, (
                    append(output_name, _unary(
                        target, item(source_name, var("expert index")), fast_activations,
                    )),
                    change_var("expert index", 1),
                )),
            ))
            continue

        if target in {
            "aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor", "aten.div.Tensor",
        }:
            if target in {"aten.add.Tensor", "aten.sub.Tensor"}:
                alpha = (
                    node.args[2] if len(node.args) > 2
                    else node.kwargs.get("alpha", 1)
                )
                if alpha != 1:
                    raise ValueError(
                        "unsupported operation inside generic ExpertFamily: "
                        f"{target} alpha must be 1"
                    )
            left, right = node.args[:2]
            size = math.prod(output_shape)
            operation = {
                "aten.add.Tensor": add, "aten.sub.Tensor": sub,
                "aten.mul.Tensor": mul, "aten.div.Tensor": div,
            }[target]
            statements.extend((
                set_var("expert index", 1),
                repeat(size, (
                    append(output_name, operation(
                        element(left, var("expert index"), output_shape),
                        element(right, var("expert index"), output_shape),
                    )),
                    change_var("expert index", 1),
                )),
            ))
            continue

        if target in {"aten.rms_norm.default", "aten.layer_norm.default"}:
            source = materialized(node.args[0])
            size = math.prod(output_shape)
            norm_size = math.prod(node.args[1])
            groups = size // norm_size
            weight_arg = node.args[2] if len(node.args) > 2 else None
            bias_arg = (
                node.args[3]
                if target == "aten.layer_norm.default" and len(node.args) > 3
                else None
            )
            epsilon_index = 4 if target == "aten.layer_norm.default" else 3
            epsilon = (
                node.args[epsilon_index] if len(node.args) > epsilon_index
                else (
                    torch.finfo(node.meta["val"].dtype).eps
                    if target == "aten.rms_norm.default" else 1e-5
                )
            )
            group_body = [
                set_var("expert mean", 0),
                set_var("expert variance", 0),
            ]
            if target == "aten.layer_norm.default" and fast_layer_norm:
                approximated.append(target)
            if target == "aten.layer_norm.default" and fast_layer_norm:
                group_body.extend((
                    set_var("expert index", 1),
                    repeat(norm_size, (
                        set_var(
                            "expert centered",
                            item(source, var("expert norm index")),
                        ),
                        change_var("expert mean", var("expert centered")),
                        change_var("expert variance", mul(
                            var("expert centered"), var("expert centered"),
                        )),
                        change_var("expert index", 1),
                        change_var("expert norm index", 1),
                    )),
                    change_var("expert norm index", -norm_size),
                    set_var("expert mean", div(var("expert mean"), norm_size)),
                    set_var("expert variance", sub(
                        div(var("expert variance"), norm_size),
                        mul(var("expert mean"), var("expert mean")),
                    )),
                ))
            elif target == "aten.layer_norm.default":
                group_body.extend((
                    set_var("expert index", 1),
                    repeat(norm_size, (
                        change_var(
                            "expert mean", item(source, var("expert norm index")),
                        ),
                        change_var("expert index", 1),
                        change_var("expert norm index", 1),
                    )),
                    change_var("expert norm index", -norm_size),
                    set_var("expert mean", div(var("expert mean"), norm_size)),
                ))
            if not (target == "aten.layer_norm.default" and fast_layer_norm):
                group_body.extend((
                    set_var("expert index", 1),
                    repeat(norm_size, (
                        set_var(
                            "expert centered",
                            item(source, var("expert norm index"))
                            if target == "aten.rms_norm.default"
                            else sub(
                                item(source, var("expert norm index")),
                                var("expert mean"),
                            ),
                        ),
                        change_var("expert variance", mul(
                            var("expert centered"), var("expert centered"),
                        )),
                        change_var("expert index", 1),
                        change_var("expert norm index", 1),
                    )),
                    change_var("expert norm index", -norm_size),
                ))
            group_body.extend((
                set_var("expert scale", div(1, mathop("sqrt", add(
                    (
                        var("expert variance")
                        if target == "aten.layer_norm.default" and fast_layer_norm
                        else div(var("expert variance"), norm_size)
                    ),
                    epsilon,
                )))),
                set_var("expert index", 1),
                repeat(norm_size, (
                    set_var(
                        "expert centered",
                        item(source, var("expert norm index"))
                        if target == "aten.rms_norm.default"
                        else sub(
                            item(source, var("expert norm index")),
                            var("expert mean"),
                        ),
                    ),
                    append(output_name, add(
                        mul(
                            mul(var("expert centered"), var("expert scale")),
                            1 if weight_arg is None else element(
                                weight_arg, var("expert index"),
                            ),
                        ),
                        0 if bias_arg is None else element(bias_arg, var("expert index")),
                    )),
                    change_var("expert index", 1),
                    change_var("expert norm index", 1),
                )),
            ))
            statements.extend((
                set_var("expert norm index", 1),
                repeat(groups, tuple(group_body)),
            ))
            continue

        raise ValueError(f"unsupported operation inside generic ExpertFamily: {target}")

    output_node = next(node for node in nodes if node.op == "output")
    result = output_node.args[0]
    if isinstance(result, (tuple, list)):
        if len(result) != 1:
            raise ValueError("generic experts must return one tensor")
        result = result[0]
    return ExpertProgram(
        tuple(statements), tensor_name(result), declared, list_values, sizes, extra_variables,
        approximated,
    )


_EXPERT_VARIABLES = (
    "token", "expert", "best", "selected", "route", "active expert",
    "input index", "output index", "selected denominator", "expert row",
    "expert index", "expert input index", "expert weight index", "expert sum",
    "expert mean", "expert variance", "expert centered", "expert scale",
    "expert norm index", "expert source base", "candidate score",
)


def _selected_routes(ctx, routes, weights, top_k, experts):
    """Fill ``routes`` (one-based expert numbers) and ``weights`` for ``token``
    from the router's selection scores and combination weights."""
    selection = ctx.inputs[ctx.argument_names[1]].list
    combination = ctx.inputs[ctx.argument_names[2]].list
    normalize = ()
    if ctx.options.get("normalize_selected"):
        normalize = (
            set_var("selected denominator", 0),
            set_var("route", 1),
            repeat(top_k, (
                change_var("selected denominator", item(weights, var("route"))),
                change_var("route", 1),
            )),
            if_else(
                not_(eq(var("selected denominator"), 0)),
                (
                    set_var("route", 1),
                    repeat(top_k, (
                        replace(weights, var("route"), div(
                            item(weights, var("route")), var("selected denominator"),
                        )),
                        change_var("route", 1),
                    )),
                ),
                (
                    set_var("route", 1),
                    repeat(top_k, (
                        replace(weights, var("route"), 0),
                        change_var("route", 1),
                    )),
                ),
            ),
        )
    candidate = var("candidate score")
    return (
        set_var("route", 0),
        repeat(top_k, (
            set_var("expert", 1), set_var("best", "-Infinity"),
            set_var("selected", 0),
            repeat(experts, (
                if_(eq(index_of(routes, var("expert")), 0), (
                    set_var("candidate score", item(
                        selection, add(mul(var("token"), experts), var("expert")),
                    )),
                    if_else(
                        eq(var("selected"), 0),
                        (
                            set_var("best", candidate),
                            set_var("selected", var("expert")),
                        ),
                        (if_else(
                            not_(eq(candidate, candidate)),
                            (if_(eq(var("best"), var("best")), (
                                set_var("best", candidate),
                                set_var("selected", var("expert")),
                            )),),
                            (if_(eq(var("best"), var("best")), (
                                if_(gt(candidate, var("best")), (
                                    set_var("best", candidate),
                                    set_var("selected", var("expert")),
                                )),
                            )),),
                        ),),
                    ),
                )),
                change_var("expert", 1),
            )),
            append(routes, var("selected")),
            append(weights, item(
                combination, add(mul(var("token"), experts), var("selected")),
            )),
            change_var("route", 1),
        )),
        *normalize,
    )


def _given_routes(ctx, routes, weights, top_k, _experts):
    """Copy ``token``'s zero-based expert IDs and weights from the inputs."""
    ids = ctx.inputs[ctx.argument_names[1]].list
    given = ctx.inputs[ctx.argument_names[2]].list
    position = add(mul(var("token"), top_k), var("route"))
    return (
        set_var("route", 1),
        repeat(top_k, (
            append(routes, add(item(ids, position), 1)),
            append(weights, item(given, position)),
            change_var("route", 1),
        )),
    )


def _emit_expert_family(ctx: KernelContext, *, routed: bool):
    """Topology-independent routed executor for a canonical expert FX graph.

    The arguments are the value, the routing inputs, then each state bank,
    stored expert-major. The generated program is independent of the number
    of experts: choosing another expert changes only the base index used by
    static-list reporters. Options: ``family``, ``top_k`` and
    ``normalize_selected`` (router-selected routes), ``fast_activations`` and
    ``fast_layer_norm``.
    """
    names = ctx.argument_names
    value = ctx.inputs[names[0]]
    family = ctx.options["family"]
    program = expert_program(
        family, [ctx.inputs[name].list for name in names[3:]],
        fast_activations=bool(ctx.options.get("fast_activations")),
        fast_layer_norm=bool(ctx.options.get("fast_layer_norm")),
    )
    width = value.shape[-1]
    tokens = math.prod(value.shape[:-1])
    output_width = family.output_width
    top_k = ctx.inputs[names[1]].shape[-1] if routed else int(ctx.options["top_k"])
    for name in (*_EXPERT_VARIABLES, *program.extra_variables):
        ctx.temp_var(name)
    ctx.approximated.extend(program.approximated)
    out = ctx.output.list
    routes = ctx.temp_list("_moe route ids", top_k)
    weights = ctx.temp_list("_moe route weights", top_k)
    for name in program.lists:
        if name in program.list_values:
            ctx.static_list(name, program.list_values[name])
        else:
            ctx.temp_list(name, program.sizes[name])
    route_statements = (_given_routes if routed else _selected_routes)(
        ctx, routes, weights, top_k, family.expert_count,
    )
    return (
        clear(out),
        set_var("token", 0),
        repeat(tokens, (
            repeat(output_width, (append(out, 0),)),
            clear(routes), clear(weights),
            *route_statements,
            set_var("route", 1),
            repeat(top_k, (
                set_var("active expert", sub(item(routes, var("route")), 1)),
                clear("_expert input"),
                set_var("input index", add(mul(var("token"), width), 1)),
                repeat(width, (
                    append("_expert input", item(value.list, var("input index"))),
                    change_var("input index", 1),
                )),
                *program.statements,
                set_var("expert index", 1),
                set_var("output index", add(mul(var("token"), output_width), 1)),
                repeat(output_width, (
                    replace(out, var("output index"), add(
                        item(out, var("output index")),
                        mul(item(program.output, var("expert index")), item(weights, var("route"))),
                    )),
                    change_var("expert index", 1),
                    change_var("output index", 1),
                )),
                change_var("route", 1),
            )),
            change_var("token", 1),
        )),
    )


EXPERT_FAMILY_MOE_KERNEL = Kernel(
    "cattorch.expert_family_moe.default",
    lambda ctx: _emit_expert_family(ctx, routed=False),
    name="expert_family_moe", arguments=("value", "selection", "combination"),
)
ROUTED_EXPERTS_KERNEL = Kernel(
    "cattorch.routed_experts.default",
    lambda ctx: _emit_expert_family(ctx, routed=True),
    name="routed_experts", arguments=("value", "expert_ids", "weights"),
)


__all__ = [
    "EXPERT_FAMILY_MOE_KERNEL", "ExpertProgram", "ROUTED_EXPERTS_KERNEL",
    "STACKED_SWIGLU_MOE_KERNEL", "expert_program",
]
