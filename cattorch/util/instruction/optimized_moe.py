"""Scratch-native banked sparse mixture-of-experts executors."""

from __future__ import annotations

import math

import torch

from cattorch.util.instruction.instruction import Instruction
from cattorch.util.instruction.optimized_common import _static_unrolled_repeat
from cattorch.util.scratch.dsl import (
    Program, add, append, change_var, clear, div, eq, gt, if_, index_of, item,
    if_else, length, mathop, mod, mul, not_, repeat, replace, set_var, sub, var,
)


class StackedSwiGLUMoEInstruction(Instruction):
    """Generic top-k, full-softmax router with banked SwiGLU experts."""

    def prepare(self):
        value, router, gate, up, down = self.args[:5]
        self.width = value.shape[-1]
        self.tokens = math.prod(value.shape[:-1]) if len(value.shape) > 1 else 1
        self.experts = router.shape[0]
        self.hidden = gate.shape[1]
        self.output_width = down.shape[1]
        self.top_k = int(self.args[5].value)
        self.normalize_selected = bool(self.args[6].value)

    def finalize(self):
        logits = "_moe logits"
        route_ids = "_moe route ids"
        route_weights = "_moe route weights"
        hidden = "_moe hidden"
        router_dot = (
            change_var("sum", mul(
                item("T1", var("left index")),
                item("T2", var("weight index")),
            )),
            change_var("left index", 1),
            change_var("weight index", 1),
        )
        gate_up_dot = (
            change_var("gate sum", mul(
                item("T1", var("left index")),
                item("T3", var("gate index")),
            )),
            change_var("up sum", mul(
                item("T1", var("left index")),
                item("T4", var("up index")),
            )),
            change_var("left index", 1),
            change_var("gate index", 1),
            change_var("up index", 1),
        )
        down_dot = (
            change_var("sum", mul(
                item(hidden, var("hidden index")),
                item("T5", var("down index")),
            )),
            change_var("hidden index", 1),
            change_var("down index", 1),
        )
        normalize = ()
        if self.normalize_selected and self.top_k > 1:
            normalize = (
                set_var("selected denominator", 0),
                set_var("route", 1),
                repeat(self.top_k, (
                    change_var(
                        "selected denominator", item(route_weights, var("route")),
                    ),
                    change_var("route", 1),
                )),
                set_var("route", 1),
                repeat(self.top_k, (
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
        if not self.normalize_selected:
            router_softmax = (
                set_var("denominator", 0),
                set_var("expert", 1),
                repeat(self.experts, (
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
        elif self.top_k > 1:
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
                    item("T1", var("left index")),
                    item("T3", var(f"gate index {offset}")),
                )),
                change_var(f"up sum {offset}", mul(
                    item("T1", var("left index")),
                    item("T4", var(f"up index {offset}")),
                )),
            )
        ) + (
            change_var("left index", 1),
            *(change_var(f"gate index {offset}", 1) for offset in range(1, 5)),
            *(change_var(f"up index {offset}", 1) for offset in range(1, 5)),
        )
        grouped_hidden = ()
        grouped_hidden_rows = self.hidden - self.hidden % 4
        if grouped_hidden_rows:
            grouped_hidden = (
                repeat(grouped_hidden_rows // 4, (
                    set_var("left index", var("input start")),
                    set_var("gate index 1", add(
                        var("gate expert base"),
                        add(mul(var("hidden row"), self.width), 1),
                    )),
                    set_var("up index 1", add(
                        var("up expert base"),
                        add(mul(var("hidden row"), self.width), 1),
                    )),
                    *(set_var(
                        f"gate index {offset}",
                        add(var("gate index 1"), (offset - 1) * self.width),
                    ) for offset in range(2, 5)),
                    *(set_var(
                        f"up index {offset}",
                        add(var("up index 1"), (offset - 1) * self.width),
                    ) for offset in range(2, 5)),
                    *(set_var(f"gate sum {offset}", 0) for offset in range(1, 5)),
                    *(set_var(f"up sum {offset}", 0) for offset in range(1, 5)),
                    *_static_unrolled_repeat(self.width, grouped_gate_up_dot),
                    *silu_products,
                    change_var("hidden row", 4),
                )),
            )
        scalar_hidden = ()
        if grouped_hidden_rows < self.hidden:
            scalar_hidden = (
                repeat(self.hidden - grouped_hidden_rows, (
                    set_var("left index", var("input start")),
                    set_var("gate index", add(
                        var("gate expert base"),
                        add(mul(var("hidden row"), self.width), 1),
                    )),
                    set_var("up index", add(
                        var("up expert base"),
                        add(mul(var("hidden row"), self.width), 1),
                    )),
                    set_var("gate sum", 0),
                    set_var("up sum", 0),
                    *_static_unrolled_repeat(self.width, gate_up_dot),
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
            if self.top_k == 1:
                return (append("T6", contribution),)
            output_index = add(
                mul(var("token"), self.output_width),
                add(var("feature"), feature_offset + 1),
            )
            return (
                set_var("output index", output_index),
                replace("T6", var("output index"), add(
                    item("T6", var("output index")), contribution,
                )),
            )

        grouped_down_dot = tuple(
            change_var(f"down sum {offset}", mul(
                item(hidden, var("hidden index")),
                item("T5", var(f"down index {offset}")),
            ))
            for offset in range(1, 5)
        ) + (
            change_var("hidden index", 1),
            *(change_var(f"down index {offset}", 1) for offset in range(1, 5)),
        )
        grouped_output = ()
        grouped_output_rows = self.output_width - self.output_width % 4
        if grouped_output_rows:
            grouped_output = (
                repeat(grouped_output_rows // 4, (
                    set_var("hidden index", 1),
                    set_var("down index 1", add(
                        var("down expert base"),
                        add(mul(var("feature"), self.hidden), 1),
                    )),
                    *(set_var(
                        f"down index {offset}",
                        add(var("down index 1"), (offset - 1) * self.hidden),
                    ) for offset in range(2, 5)),
                    *(set_var(f"down sum {offset}", 0) for offset in range(1, 5)),
                    *_static_unrolled_repeat(self.hidden, grouped_down_dot),
                    *(statement
                      for offset in range(1, 5)
                      for statement in store_output(f"down sum {offset}", offset - 1)),
                    change_var("feature", 4),
                )),
            )
        scalar_output = ()
        if grouped_output_rows < self.output_width:
            scalar_output = (
                repeat(self.output_width - grouped_output_rows, (
                    set_var("hidden index", 1),
                    set_var("down index", add(
                        var("down expert base"),
                        add(mul(var("feature"), self.hidden), 1),
                    )),
                    set_var("sum", 0),
                    *_static_unrolled_repeat(self.hidden, down_dot),
                    *store_output("sum"),
                    change_var("feature", 1),
                )),
            )

        initialize_output = (
            () if self.top_k == 1
            else (repeat(self.output_width, (append("T6", 0),)),)
        )
        return Program(
            "stacked_swiglu_moe",
            variables=(
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
            ),
            lists=(
                "T1", "T2", "T3", "T4", "T5", "T6", logits,
                route_ids, route_weights, hidden,
            ),
            body=(
                clear("T6"),
                set_var("token", 0),
                set_var("input start", 1),
                repeat(self.tokens, (
                    clear(logits),
                    set_var("expert", 0),
                    set_var("maximum", "-Infinity"),
                    repeat(self.experts, (
                        set_var("left index", var("input start")),
                        set_var("weight index", add(mul(var("expert"), self.width), 1)),
                        set_var("sum", 0),
                        *_static_unrolled_repeat(self.width, router_dot),
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
                    repeat(self.top_k, (
                        set_var("expert", 1),
                        set_var("best", "-Infinity"),
                        set_var("selected", 1),
                        repeat(self.experts, (
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
                    repeat(self.top_k, (
                        set_var("expert", sub(item(route_ids, var("route")), 1)),
                        set_var("route weight", item(route_weights, var("route"))),
                        set_var(
                            "gate expert base",
                            mul(var("expert"), self.hidden * self.width),
                        ),
                        set_var("up expert base", var("gate expert base")),
                        set_var(
                            "down expert base",
                            mul(var("expert"), self.output_width * self.hidden),
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
                    change_var("input start", self.width),
                )),
            ),
        ).compile()


class ExpertFamilyMoEInstruction(Instruction):
    """Topology-independent routed executor for a canonical expert FX graph.

    Each expert state tensor is stored expert-major. The generated program is
    independent of the number of experts: choosing another expert changes only
    the base index used by static-list reporters.
    """

    def __init__(
        self, torch_name, output, *args, family, top_k: int,
        normalize_selected: bool, fast_activations: bool = False,
        fast_layer_norm: bool = False,
    ):
        self.family = family
        self.top_k = top_k
        self.normalize_selected = normalize_selected
        self.fast_activations = fast_activations
        self.fast_layer_norm = fast_layer_norm
        super().__init__(torch_name, output, *args)

    def prepare(self):
        from cattorch.frontend import capture_model

        self.width = self.args[0].shape[-1]
        self.tokens = math.prod(self.args[0].shape[:-1])
        self.experts = self.family.expert_count
        self.output_width = self.family.output_width
        self.captured = capture_model(
            self.family.template.eval(), (self.family.example_input,), frontend="fx",
        )

        unique_locations = tuple(dict.fromkeys(self.family._bank_names.values()))
        location_index = {location: index for index, location in enumerate(unique_locations)}
        template_values = {}
        template_state = dict(
            self.family.template.named_parameters(remove_duplicate=False),
        )
        template_state.update(dict(
            self.family.template.named_buffers(remove_duplicate=False),
        ))
        for name, value in self.family.template.named_parameters(remove_duplicate=False):
            template_values[id(value)] = name
        for name, value in self.family.template.named_buffers(remove_duplicate=False):
            template_values[id(value)] = name
        self.state_banks = {}
        for captured_name, value in self.captured.state_inputs.items():
            logical_name = template_values.get(id(value))
            if logical_name is None:
                # Fake/capture wrappers can replace Tensor Python identities;
                # storage identity remains stable for the underlying state.
                for name in self.family._state_order:
                    candidate = template_state[name]
                    if (
                        candidate.untyped_storage().data_ptr()
                        == value.untyped_storage().data_ptr()
                        and candidate.storage_offset() == value.storage_offset()
                    ):
                        logical_name = name
                        break
            if logical_name is None:
                raise ValueError(f"unable to map expert state {captured_name!r} to its bank")
            location = self.family._bank_names[logical_name]
            self.state_banks[captured_name] = (
                f"T{4 + location_index[location]}",
                self.family._bank(logical_name).shape[1:],
            )

    def _unary(self, target, value):
        if target == "aten.relu.default":
            return div(add(value, mathop("abs", value)), 2)
        if target == "aten.sigmoid.default":
            return div(1, add(1, mathop("e ^", mul(-1, value))))
        if target == "aten.silu.default":
            return div(value, add(1, mathop("e ^", mul(-1, value))))
        if target == "aten.tanh.default":
            return sub(div(2, add(1, mathop("e ^", mul(-2, value)))), 1)
        if target == "aten.gelu.default":
            if self.fast_activations:
                return mul(value, div(
                    1, add(1, mathop("e ^", mul(-1.702, value))),
                ))
            cube = mul(value, mul(value, value))
            inner = mul(0.7978845608028654, add(value, mul(0.044715, cube)))
            tanh = sub(div(
                2, add(1, mathop("e ^", mul(-2, inner))),
            ), 1)
            return mul(0.5, mul(value, add(1, tanh)))
        if target == "aten.neg.default":
            return mul(-1, value)
        raise ValueError(f"unsupported generic expert activation {target}")

    def _expert_program(self):
        nodes = list(self.captured.graph.nodes)
        placeholder = next(node for node in nodes if node.op == "placeholder")
        names = {placeholder.name: "_expert input"}
        shapes = {placeholder.name: tuple(placeholder.meta["val"].shape)}
        statements = []
        declared = ["_expert input"]
        list_values = {}

        def tensor_name(argument):
            if argument.op == "get_attr":
                return self.state_banks[argument.name][0]
            return names[argument.name]

        def tensor_shape(argument):
            if argument.op == "get_attr":
                return tuple(self.state_banks[argument.name][1])
            return shapes[argument.name]

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
                if argument.op == "get_attr":
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
                continue
            if target in {
                "aten.view.default", "aten.reshape.default", "aten._unsafe_view.default",
                "aten.flatten.using_ints", "aten.contiguous.default", "aten.clone.default",
                "aten.detach.default", "aten.alias.default",
            }:
                names[node.name] = tensor_name(node.args[0])
                shapes[node.name] = tuple(node.meta["val"].shape)
                continue

            output_name = f"_expert {len(declared)}"
            names[node.name] = output_name
            output_shape = tuple(node.meta["val"].shape)
            shapes[node.name] = output_shape
            declared.append(output_name)
            statements.append(clear(output_name))

            if target == "aten.linear.default":
                source, weight = node.args[:2]
                bias = node.args[2] if len(node.args) > 2 else None
                source_name = tensor_name(source)
                weight_name = tensor_name(weight)
                rows, inner = tensor_shape(weight)
                input_rows = math.prod(tensor_shape(source)) // inner
                bias_name = None if bias is None else tensor_name(bias)
                statements.extend((
                    set_var("expert source base", 1),
                    repeat(input_rows, (
                        set_var("expert row", 0),
                        repeat(rows, (
                            set_var(
                                "expert sum", 0 if bias_name is None else item(
                                    bias_name,
                                    add(
                                        mul(var("active expert"), rows),
                                        add(var("expert row"), 1),
                                    ),
                                ),
                            ),
                            set_var("expert input index", var("expert source base")),
                            set_var("expert weight index", add(
                                mul(var("active expert"), rows * inner),
                                add(mul(var("expert row"), inner), 1),
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

            if target in {
                "aten.relu.default", "aten.sigmoid.default", "aten.silu.default",
                "aten.tanh.default", "aten.gelu.default", "aten.neg.default",
            }:
                if target == "aten.gelu.default":
                    approximate = (
                        node.args[1] if len(node.args) > 1
                        else node.kwargs.get("approximate", "none")
                    )
                    if approximate != "tanh" and not self.fast_activations:
                        raise ValueError(
                            "unsupported operation inside generic ExpertFamily: "
                            "exact GELU requires approximate='tanh'"
                        )
                source_name = tensor_name(node.args[0])
                size = math.prod(output_shape)
                statements.extend((
                    set_var("expert index", 1),
                    repeat(size, (
                        append(output_name, self._unary(
                            target, item(source_name, var("expert index")),
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
                source = tensor_name(node.args[0])
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
                if target == "aten.layer_norm.default" and self.fast_layer_norm:
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
                if not (target == "aten.layer_norm.default" and self.fast_layer_norm):
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
                            if target == "aten.layer_norm.default" and self.fast_layer_norm
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
        return tuple(statements), tensor_name(result), tuple(declared), list_values

    def finalize(self):
        expert_body, expert_output, expert_lists, expert_list_values = (
            self._expert_program()
        )
        routes = "_moe route ids"
        weights = "_moe route weights"
        normalize = ()
        if self.normalize_selected:
            normalize = (
                set_var("selected denominator", 0),
                set_var("route", 1),
                repeat(self.top_k, (
                    change_var("selected denominator", item(weights, var("route"))),
                    change_var("route", 1),
                )),
                if_else(
                    not_(eq(var("selected denominator"), 0)),
                    (
                        set_var("route", 1),
                        repeat(self.top_k, (
                            replace(weights, var("route"), div(
                                item(weights, var("route")), var("selected denominator"),
                            )),
                            change_var("route", 1),
                        )),
                    ),
                    (
                        set_var("route", 1),
                        repeat(self.top_k, (
                            replace(weights, var("route"), 0),
                            change_var("route", 1),
                        )),
                    ),
                ),
            )
        bank_lists = tuple(f"T{index}" for index in range(4, 4 + len(self.args) - 3))
        target_list = f"T{len(self.args) + 1}"
        return Program(
            "expert_family_moe",
            variables=(
                "token", "expert", "best", "selected", "route", "active expert",
                "input index", "output index", "selected denominator", "expert row",
                "expert index", "expert input index", "expert weight index", "expert sum",
                "expert mean", "expert variance", "expert centered", "expert scale",
                "expert norm index", "expert source base", "candidate score",
            ),
            lists=("T1", "T2", "T3", *bank_lists, target_list, routes, weights, *expert_lists),
            list_values=expert_list_values,
            body=(
                clear(target_list),
                set_var("token", 0),
                repeat(self.tokens, (
                    repeat(self.output_width, (append(target_list, 0),)),
                    clear(routes), clear(weights),
                    set_var("route", 0),
                    repeat(self.top_k, (
                        set_var("expert", 1), set_var("best", "-Infinity"),
                        set_var("selected", 0),
                        repeat(self.experts, (
                            if_(eq(index_of(routes, var("expert")), 0), (
                                set_var("candidate score", item(
                                    "T2", add(
                                        mul(var("token"), self.experts),
                                        var("expert"),
                                    ),
                                )),
                                if_else(
                                    eq(var("selected"), 0),
                                    (
                                        set_var("best", var("candidate score")),
                                        set_var("selected", var("expert")),
                                    ),
                                    (if_else(
                                        not_(eq(
                                            var("candidate score"),
                                            var("candidate score"),
                                        )),
                                        (if_(eq(var("best"), var("best")), (
                                            set_var("best", var("candidate score")),
                                            set_var("selected", var("expert")),
                                        )),),
                                        (if_(eq(var("best"), var("best")), (
                                            if_(gt(
                                                var("candidate score"), var("best"),
                                            ), (
                                                set_var(
                                                    "best", var("candidate score"),
                                                ),
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
                            "T3", add(mul(var("token"), self.experts), var("selected")),
                        )),
                        change_var("route", 1),
                    )),
                    *normalize,
                    set_var("route", 1),
                    repeat(self.top_k, (
                        set_var("active expert", sub(item(routes, var("route")), 1)),
                        clear("_expert input"),
                        set_var("input index", add(mul(var("token"), self.width), 1)),
                        repeat(self.width, (
                            append("_expert input", item("T1", var("input index"))),
                            change_var("input index", 1),
                        )),
                        *expert_body,
                        set_var("expert index", 1),
                        set_var("output index", add(mul(var("token"), self.output_width), 1)),
                        repeat(self.output_width, (
                            replace(target_list, var("output index"), add(
                                item(target_list, var("output index")),
                                mul(
                                    item(expert_output, var("expert index")),
                                    item(weights, var("route")),
                                ),
                            )),
                            change_var("expert index", 1),
                            change_var("output index", 1),
                        )),
                        change_var("route", 1),
                    )),
                    change_var("token", 1),
                )),
            ),
        ).compile()


__all__ = ["ExpertFamilyMoEInstruction", "StackedSwiGLUMoEInstruction"]
