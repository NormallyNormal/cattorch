import json
import time

import torch
from torch.export import export

from src.cattorch.util.argument import Argument
from src.cattorch.util.instruction.instruction import Instruction
from src.cattorch.util.instruction.matmul import MatMulInstruction
from src.cattorch.util.instruction.relu import ReLUInstruction
from src.cattorch.util.instruction.tensor_add import TensorAddInstruction
from src.cattorch.util.scope import ScopeManager
from src.cattorch.util.scratch.block_combiner import combine
from src.cattorch.util.scratch.block_manager import BlockManager
from src.cattorch.util.scratch.finalize_scratch import finalize_sprite
from src.cattorch.util.scratch.tensor_adder import TensorAdder
from src.cattorch.util.scratch.tensor_replacer import TensorReplacer


def transpile(model: torch.nn.Module, input_shape: torch.Size, sprite_name: str):
    exported = export(model, (torch.randn(input_shape),))
    nodes = list(exported.graph.nodes)
    normalised_state = {k.lower(): v for k, v in exported.state_dict.items()}

    def resolve_weight(arg_name: str):
        key = arg_name.removeprefix("p_")
        return normalised_state.get(key)

    def get_shape(arg) -> torch.Size:
        if hasattr(arg, 'meta') and 'val' in arg.meta:
            return arg.meta['val'].shape
        return torch.Size([])

    scope = ScopeManager()
    scope.analyze_lifetimes(nodes)

    dynamic_lists_set = set()
    static_lists_set = {}
    constants_set = []

    scratch_output = None
    block_manager = BlockManager()

    for node in nodes:
        if node.op == 'placeholder':
            continue

        if node.op == 'call_function':
            target_list = scope.get_list_for_node(node)

            input_lists = []
            for arg in node.args:
                if hasattr(arg, 'name'):
                    if arg.name in scope.assignments:
                        input_lists.append(f"T{scope.assignments[arg.name]}")
                        dynamic_lists_set.add(f"T{scope.assignments[arg.name]}")
                    else:
                        input_lists.append(f"W_{arg.name}")
                        tensor = resolve_weight(arg.name)
                        static_lists_set[arg.name] = tensor if tensor is not None else None
                else:
                    input_lists.append(f"C_{arg}")
                    constants_set.append(arg)

            print(f"SCRATCH: {node.target} -> {target_list} ({node.name}) (Inputs: {input_lists})")

            instruction = None
            if str(node.target) == "aten.matmul.default":
                instruction = MatMulInstruction(
                    node.target,
                    target_list,
                    Argument(input_lists[0], get_shape(node.args[0])),
                    Argument(input_lists[1], get_shape(node.args[1])),
                )
            if str(node.target) == "aten.relu.default":
                instruction = ReLUInstruction(
                    node.target,
                    target_list,
                    Argument(input_lists[0], get_shape(node.args[0]))
                )
            if str(node.target) == "aten.add.Tensor":
                instruction = TensorAddInstruction(
                    node.target,
                    target_list,
                    Argument(input_lists[0], get_shape(node.args[0])),
                    Argument(input_lists[1], get_shape(node.args[1])),
                )
            data = instruction.finalize() # get the sprite dict with blocks

            block_manager.new_context()
            data["blocks"] = block_manager.apply_to_blocks(data["blocks"])

            tensor_adder = TensorAdder()
            data = tensor_adder.apply(data, input_lists + [target_list])

            # T1 = input A, T2 = input B, T3 = output
            tensor_replacer = TensorReplacer(data, input_lists + [target_list])
            data["blocks"] = tensor_replacer.apply(data["blocks"])

            if scratch_output is None:
                scratch_output = data
            else:
                scratch_output = combine(scratch_output, data)

            scope.release_dependencies(node)

    # Add static weight lists with preloaded data
    tensor_adder = TensorAdder()
    scratch_output = tensor_adder.apply(
        scratch_output,
        [f"W_{k}" for k in static_lists_set.keys()],
        weights={f"W_{k}": v for k, v in static_lists_set.items() if v is not None}
    )

    print(f"Scratch dynamic lists required: {list(dynamic_lists_set)}")
    print(f"Scratch static lists required: {list(static_lists_set.keys())}")
    print(f"Scratch static weights: { {k: v.shape for k, v in static_lists_set.items() if v is not None} }")
    print(f"Scratch constants required: {list(constants_set)}")
    finalize_sprite(scratch_output, f"{sprite_name}.sprite3", sprite_name=sprite_name)
