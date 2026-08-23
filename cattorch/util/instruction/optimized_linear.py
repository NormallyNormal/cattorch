"""Linear, matmul, and attention production kernels."""

import math

import torch

from cattorch.util.instruction.instruction import Instruction
from cattorch.util.scratch.dsl import (
    Program,
    add,
    append,
    change_var,
    clear,
    div,
    eq,
    gt,
    if_,
    if_else,
    item,
    length,
    lt,
    mathop,
    mod,
    mul,
    repeat,
    replace,
    set_var,
    sub,
    var,
)


from cattorch.util.instruction.optimized_common import (
    _broadcast_index,
    _static_unrolled_repeat,
    _unrolled_copy_statements,
    _unrolled_dot_statements,
    _unrolled_offset_dot_statements,
)

class LinearInstruction(Instruction):
    """Direct ``aten.linear`` lowering using the original [M, K] weight.

    Unlike the old transpose-first lowering, this does not transpose the weight at
    runtime and folds bias initialization into the dot product.
    """

    aten_op = "aten.linear.default"
    # Browser Scratch screens show that sharing one activation read across
    # four output accumulators pays off at the transformer-sized 49,152-MAC
    # matrices, but not at 128 x 128. Kept as a class attribute so benchmark
    # builders can produce isolated before/after exports without a public flag.
    grouped_min_macs = 49_152

    def __init__(
        self,
        torch_name,
        output,
        *args,
        epilogue=(),
        fast=False,
        weight_shard_rows=None,
        interleaved=False,
    ):
        self.epilogue = tuple(epilogue)
        self.fast = fast
        self.weight_shard_rows = tuple(weight_shard_rows or ())
        self.interleaved = interleaved
        super().__init__(torch_name, output, *args)

    def prepare(self):
        input_shape = self.args[0].shape
        weight_shape = self.args[1].shape
        self.rows = math.prod(input_shape[:-1]) if len(input_shape) > 1 else 1
        self.inner = input_shape[-1]
        self.columns = weight_shape[0]
        self.has_bias = self.args[2].value is not None or bool(self.args[2].shape)
        self.has_residual = self.args[3].value is not None or bool(self.args[3].shape)
        self.sparse_block_starts = None
        if not self.weight_shard_rows:
            self.weight_shard_rows = (self.columns,)
        if sum(self.weight_shard_rows) != self.columns:
            raise ValueError("Linear weight shards must cover every output row")
        shard_count = len(self.weight_shard_rows)
        self.weight_lists = tuple(f"T{index + 2}" for index in range(shard_count))
        self.bias_list = f"T{shard_count + 2}"
        self.residual_list = f"T{shard_count + 3}"
        self.output_list = f"T{shard_count + 4}"

    def transform_weights(self, static_lists):
        if self.interleaved or len(self.weight_shard_rows) > 1:
            return
        raw_key = self.args[1].name.removeprefix("W_")
        weight = static_lists.get(raw_key)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or self.inner < 4:
            return
        full_width = self.inner - self.inner % 4
        starts = []
        counts = []
        offsets = []
        total_blocks = self.columns * (full_width // 4)
        for column in range(self.columns):
            starts.append(len(offsets) + 1)
            before = len(offsets)
            for start in range(0, full_width, 4):
                if bool(torch.any(weight[column, start:start + 4] != 0)):
                    offsets.append(start)
            counts.append(len(offsets) - before)
        # Sparse indexing has overhead; require at least one quarter of the
        # four-wide blocks to disappear before changing kernels.
        if total_blocks and len(offsets) <= total_blocks * 0.75:
            self.sparse_block_starts = starts
            self.sparse_block_counts = counts
            self.sparse_block_offsets = offsets

    def finalize(self):
        accumulator = item(self.bias_list, var("bias_index")) if self.has_bias else 0

        def epilogue_output(sum_name):
            result = var(sum_name)
            setup = []
            terminal_activation = None
            for operation, operand in self.epilogue:
                if operation == "mul":
                    result = mul(result, operand)
                elif operation == "div":
                    result = mul(result, 1.0 / operand)
                elif operation == "add":
                    result = add(result, operand)
                elif operation == "sub":
                    result = add(result, -operand)
                elif operation == "relu":
                    if result != var(sum_name):
                        setup.append(set_var("value", result))
                        result = var("value")
                    terminal_activation = "relu"
                elif operation == "sigmoid":
                    result = div(1, add(1, mathop("e ^", mul(-1, result))))
                elif operation == "tanh":
                    setup.append(
                        set_var("exponential", mathop("e ^", mul(2, result))),
                    )
                    result = div(
                        add(var("exponential"), -1),
                        add(var("exponential"), 1),
                    )
                elif operation == "silu":
                    if result != var(sum_name):
                        setup.append(set_var("value", result))
                        result = var("value")
                    result = mul(
                        result,
                        div(1, add(1, mathop("e ^", mul(-1, result)))),
                    )
                elif operation == "gelu":
                    setup.append(set_var("value", result))
                    result = var("value")
                    if self.fast:
                        result = mul(
                            result,
                            div(1, add(1, mathop("e ^", mul(-1.702, result)))),
                        )
                    else:
                        inner = mul(
                            0.7978845608028654,
                            add(result, mul(0.044715, mul(mul(result, result), result))),
                        )
                        setup.extend((
                            set_var("inner", inner),
                            set_var(
                                "exponential",
                                mathop("e ^", mul(2, var("inner"))),
                            ),
                        ))
                        tanh_inner = div(
                            add(var("exponential"), -1),
                            add(var("exponential"), 1),
                        )
                        result = mul(mul(0.5, result), add(1, tanh_inner))
                elif operation == "tensor_add":
                    result = add(result, item(self.residual_list, var("output_index")))

            if terminal_activation == "relu":
                output = if_else(
                    gt(result, 0),
                    (append(self.output_list, result),),
                    (append(self.output_list, 0),),
                )
            else:
                output = append(self.output_list, result)
            return (*setup, output)

        epilogue_setup_and_output = epilogue_output("sum")
        if self.sparse_block_starts is None:
            dot = _unrolled_offset_dot_statements(
                self.inner,
                right_list=self.weight_lists[0],
                right_offset="weight_offset",
            )
            dot_setup = (
                set_var("left_index", var("left_start")),
            )
            row_setup = (set_var("weight_offset", sub(1, var("left_start"))),)
            weight_finish = ()
            column_finish = (change_var("weight_offset", self.inner),)
            auxiliary_lists = ()
            auxiliary_values = {}
        else:
            block_dot = _unrolled_dot_statements(
                4,
                right_list=self.weight_lists[0],
                right_index="weight_index",
                right_step=1,
            )
            remainder = self.inner % 4
            remainder_dot = ()
            if remainder:
                remainder_start = self.inner - remainder
                remainder_dot = (
                    set_var("left_index", add(var("left_start"), remainder_start)),
                    set_var("weight_index", add(var("weight_start"), remainder_start)),
                    *_unrolled_dot_statements(
                        remainder,
                        right_list="T2",
                        right_index="weight_index",
                        right_step=1,
                    ),
                )
            dot_setup = (
                set_var(
                    "block_map_index",
                    item("_linear_block_starts", var("column_index")),
                ),
            )
            dot = (
                repeat(item("_linear_block_counts", var("column_index")), (
                    set_var(
                        "block_offset",
                        item("_linear_block_offsets", var("block_map_index")),
                    ),
                    set_var("left_index", add(var("left_start"), var("block_offset"))),
                    set_var("weight_index", add(var("weight_start"), var("block_offset"))),
                    *block_dot,
                    change_var("block_map_index", 1),
                )),
                *remainder_dot,
            )
            row_setup = (
                set_var("weight_start", 1),
                set_var("column_index", 1),
            )
            weight_finish = (change_var("weight_start", self.inner),)
            column_finish = (change_var("column_index", 1),)
            auxiliary_lists = (
                "_linear_block_starts", "_linear_block_counts", "_linear_block_offsets",
            )
            auxiliary_values = {
                "_linear_block_starts": self.sparse_block_starts,
                "_linear_block_counts": self.sparse_block_counts,
                "_linear_block_offsets": self.sparse_block_offsets,
            }
        grouped = (
            self.sparse_block_starts is None
            and self.columns % 4 == 0
            and self.inner * self.columns >= self.grouped_min_macs
        )
        if grouped:
            group_names = ("sum a", "sum b", "sum c", "sum d")
            group_setup = []
            for offset, name in enumerate(group_names):
                if self.has_bias:
                    bias_position = (
                        add(var("bias_index"), offset)
                        if offset else var("bias_index")
                    )
                    initial = item(self.bias_list, bias_position)
                else:
                    initial = 0
                group_setup.append(set_var(name, initial))
            group_setup.append(set_var("left_index", var("left_start")))

            group_outputs = []
            for name in group_names:
                group_outputs.extend(epilogue_output(name))
                if self.has_bias:
                    group_outputs.append(change_var("bias_index", 1))
                if self.has_residual:
                    group_outputs.append(change_var("output_index", 1))

            grouped_shards = []
            for weight_list, shard_rows in zip(
                self.weight_lists, self.weight_shard_rows,
            ):
                if shard_rows % 4:
                    raise ValueError("Grouped Linear shards must contain four-row groups")
                if self.interleaved:
                    grouped_dot_step = (
                        set_var("input value", item("T1", var("left_index"))),
                        change_var(
                            "sum a",
                            mul(item(weight_list, var("weight")), var("input value")),
                        ),
                        change_var(
                            "sum b",
                            mul(item(weight_list, add(var("weight"), 1)), var("input value")),
                        ),
                        change_var(
                            "sum c",
                            mul(item(weight_list, add(var("weight"), 2)), var("input value")),
                        ),
                        change_var(
                            "sum d",
                            mul(item(weight_list, add(var("weight"), 3)), var("input value")),
                        ),
                        change_var("weight", 4),
                        change_var("left_index", 1),
                    )
                    shard_setup = (set_var("weight", 1),)
                    shard_finish = ()
                else:
                    grouped_dot_step = [
                        set_var("input value", item("T1", var("left_index"))),
                    ]
                    for suffix in ("a", "b", "c", "d"):
                        grouped_dot_step.extend((
                            change_var(
                                f"sum {suffix}",
                                mul(
                                    item(weight_list, var(f"weight {suffix}")),
                                    var("input value"),
                                ),
                            ),
                            change_var(f"weight {suffix}", 1),
                        ))
                    grouped_dot_step.append(change_var("left_index", 1))
                    grouped_dot_step = tuple(grouped_dot_step)
                    shard_setup = (
                        set_var("weight a", 1),
                        set_var("weight b", self.inner + 1),
                        set_var("weight c", 2 * self.inner + 1),
                        set_var("weight d", 3 * self.inner + 1),
                    )
                    shard_finish = tuple(
                        change_var(f"weight {suffix}", 3 * self.inner)
                        for suffix in ("a", "b", "c", "d")
                    )
                grouped_shards.extend((
                    *shard_setup,
                    repeat(shard_rows // 4, (
                        *group_setup,
                        repeat(self.inner, grouped_dot_step),
                        *group_outputs,
                        *shard_finish,
                    )),
                ))

            grouped_row = (
                *((set_var("bias_index", 1),) if self.has_bias else ()),
                *grouped_shards,
                change_var("left_start", self.inner),
            )
            body = (
                clear(self.output_list),
                set_var("left_start", 1),
                *((set_var("output_index", 1),) if self.has_residual else ()),
                repeat(self.rows, grouped_row),
            )
        elif self.sparse_block_starts is not None:
            body = (
                clear(self.output_list),
                set_var("left_start", 1),
                *((set_var("output_index", 1),) if self.has_residual else ()),
                repeat(self.rows, (
                    *row_setup,
                    *((set_var("bias_index", 1),) if self.has_bias else ()),
                    repeat(self.columns, (
                        set_var("sum", accumulator),
                        *dot_setup,
                        *dot,
                        *epilogue_setup_and_output,
                        *weight_finish,
                        *((change_var("bias_index", 1),) if self.has_bias else ()),
                        *((change_var("output_index", 1),) if self.has_residual else ()),
                        *column_finish,
                    )),
                    change_var("left_start", self.inner),
                )),
            )
        else:
            dense_shards = []
            for weight_list, shard_rows in zip(
                self.weight_lists, self.weight_shard_rows,
            ):
                shard_dot = _unrolled_offset_dot_statements(
                    self.inner,
                    right_list=weight_list,
                    right_offset="weight_offset",
                )
                dense_shards.extend((
                    set_var("weight_offset", sub(1, var("left_start"))),
                    repeat(shard_rows, (
                        set_var("sum", accumulator),
                        set_var("left_index", var("left_start")),
                        *shard_dot,
                        *epilogue_setup_and_output,
                        change_var("weight_offset", self.inner),
                        *((change_var("bias_index", 1),) if self.has_bias else ()),
                        *((change_var("output_index", 1),) if self.has_residual else ()),
                    )),
                ))
            body = (
                clear(self.output_list),
                set_var("left_start", 1),
                *((set_var("output_index", 1),) if self.has_residual else ()),
                repeat(self.rows, (
                    *((set_var("bias_index", 1),) if self.has_bias else ()),
                    *dense_shards,
                    change_var("left_start", self.inner),
                )),
            )
        return Program(
            "linear_optimized",
            variables=(
                "sum", "left_start", "left_index", "weight_start",
                "weight_index", "weight_offset", "bias_index", "output_index", "value",
                "inner", "exponential",
                "column_index", "block_map_index", "block_offset",
                "sum a", "sum b", "sum c", "sum d", "input value",
                "weight", "weight a", "weight b", "weight c", "weight d",
            ),
            lists=tuple(
                f"T{index}" for index in range(1, len(self.weight_lists) + 5)
            ) + auxiliary_lists,
            list_values=auxiliary_values,
            body=body,
        ).compile()


class OptimizedMatMulInstruction(Instruction):
    """Matmul kernels whose innermost loops use running flat indices."""

    def prepare(self):
        left_shape = self.args[0].shape
        right_shape = self.args[1].shape
        self.inner = left_shape[-1]
        self.tensor_rhs = len(right_shape) > 2

        if self.tensor_rhs:
            self.batches = math.prod(right_shape[:-2])
            self.rows = left_shape[-2]
            self.columns = right_shape[-1]
        else:
            self.batches = 1
            self.rows = math.prod(left_shape[:-1]) if len(left_shape) > 1 else 1
            self.columns = right_shape[-1] if len(right_shape) > 1 else 1

    def _dot_body(self, left_start, right_start, right_step):
        if right_step == 1:
            dot_setup = (
                set_var("left_index", left_start),
                set_var("right_offset", sub(right_start, left_start)),
            )
            dot = _unrolled_offset_dot_statements(
                self.inner,
                right_list="T2",
                right_offset="right_offset",
            )
        else:
            dot_setup = (
                set_var("left_index", left_start),
                set_var("right_index", right_start),
            )
            dot = _unrolled_dot_statements(
                self.inner,
                right_list="T2",
                right_index="right_index",
                right_step=right_step,
            )
        return (
            set_var("sum", 0),
            *dot_setup,
            *dot,
            append("T3", var("sum")),
        )

    def finalize(self):
        if self.tensor_rhs:
            right_batch_size = self.inner * self.columns
            dot = self._dot_body(
                var("left_start"),
                var("right_start"),
                self.columns,
            )
            body = (
                clear("T3"),
                set_var("left_start", 1),
                set_var("right_batch_start", 1),
                repeat(self.batches, (
                    repeat(self.rows, (
                        set_var("right_start", var("right_batch_start")),
                        repeat(self.columns, dot + (change_var("right_start", 1),)),
                        change_var("left_start", self.inner),
                    )),
                    change_var("right_batch_start", right_batch_size),
                )),
            )
        else:
            right_step = self.columns if len(self.args[1].shape) > 1 else 1
            dot = self._dot_body(
                var("left_start"),
                var("right_start"),
                right_step,
            )
            body = (
                clear("T3"),
                set_var("left_start", 1),
                repeat(self.rows, (
                    set_var("right_start", 1),
                    repeat(self.columns, dot + (change_var("right_start", 1),)),
                    change_var("left_start", self.inner),
                )),
            )

        return Program(
            "matmul_optimized",
            variables=(
                "sum", "left_start", "right_start", "right_batch_start",
                "left_index", "right_index", "right_offset",
            ),
            lists=("T1", "T2", "T3"),
            body=body,
        ).compile()


class CausalScoreInstruction(Instruction):
    """Exact QK score matmul specialized for an upper-triangular ``-inf`` mask."""

    def prepare(self):
        shape = self.args[0].shape
        self.groups = math.prod(shape[:-2])
        self.length = shape[-2]
        self.width = shape[-1]
        self.scale = self.args[2].value
        self.query_offset = self.args[3].value if len(self.args) > 3 else 0
        self.key_offset = self.args[4].value if len(self.args) > 4 else 0

    def finalize(self):
        dot = _unrolled_offset_dot_statements(
            self.width,
            right_list="T2",
            right_offset="key_offset",
        )
        return Program(
            "causal_score_optimized",
            variables=(
                "query_start", "left_index", "key_group_start", "key_offset",
                "valid", "sum",
            ),
            lists=("T1", "T2", "T3"),
            body=(
                clear("T3"),
                set_var("query_start", self.query_offset + 1),
                set_var("key_group_start", self.key_offset + 1),
                repeat(self.groups, (
                    set_var("valid", 1),
                    repeat(self.length, (
                        set_var("key_offset", sub(var("key_group_start"), var("query_start"))),
                        repeat(var("valid"), (
                            set_var("sum", 0),
                            set_var("left_index", var("query_start")),
                            *dot,
                            append("T3", mul(var("sum"), self.scale)),
                            change_var("key_offset", self.width),
                        )),
                        repeat(sub(self.length, var("valid")), (
                            append("T3", float("-inf")),
                        )),
                        change_var("query_start", self.width),
                        change_var("valid", 1),
                    )),
                    change_var("key_group_start", self.length * self.width),
                )),
            ),
        ).compile()


class CausalSoftmaxInstruction(Instruction):
    """Stable softmax that skips known ``-inf`` causal positions."""

    def prepare(self):
        shape = self.args[0].shape
        self.length = shape[-1]
        self.groups = math.prod(shape[:-2])

    def finalize(self):
        current = item("T1", var("index"))
        return Program(
            "causal_softmax_optimized",
            variables=("row_start", "index", "valid", "max", "sum"),
            lists=("T1", "T2"),
            body=(
                clear("T2"),
                set_var("row_start", 1),
                repeat(self.groups, (
                    set_var("valid", 1),
                    repeat(self.length, (
                        set_var("index", var("row_start")),
                        set_var("max", current),
                        change_var("index", 1),
                        repeat(sub(var("valid"), 1), (
                            if_(gt(current, var("max")), (set_var("max", current),)),
                            change_var("index", 1),
                        )),
                        set_var("index", var("row_start")),
                        set_var("sum", 0),
                        repeat(var("valid"), (
                            change_var("sum", mathop("e ^", sub(current, var("max")))),
                            change_var("index", 1),
                        )),
                        set_var("index", var("row_start")),
                        repeat(var("valid"), (
                            append(
                                "T2",
                                div(mathop("e ^", sub(current, var("max"))), var("sum")),
                            ),
                            change_var("index", 1),
                        )),
                        repeat(sub(self.length, var("valid")), (append("T2", 0),)),
                        change_var("row_start", self.length),
                        change_var("valid", 1),
                    )),
                )),
            ),
        ).compile()


class FastCausalSoftmaxInstruction(CausalSoftmaxInstruction):
    """Unstabilized causal softmax used by fast mode."""

    def finalize(self):
        current = item("T1", var("index"))
        stored = item("T2", var("index"))
        return Program(
            "causal_softmax_fast",
            variables=("row_start", "index", "valid", "sum", "exponential"),
            lists=("T1", "T2"),
            body=(
                clear("T2"),
                set_var("row_start", 1),
                repeat(self.groups, (
                    set_var("valid", 1),
                    repeat(self.length, (
                        set_var("index", var("row_start")),
                        set_var("sum", 0),
                        repeat(var("valid"), (
                            set_var("exponential", mathop("e ^", current)),
                            change_var("sum", var("exponential")),
                            append("T2", var("exponential")),
                            change_var("index", 1),
                        )),
                        repeat(sub(self.length, var("valid")), (append("T2", 0),)),
                        set_var("index", var("row_start")),
                        repeat(var("valid"), (
                            replace("T2", var("index"), div(stored, var("sum"))),
                            change_var("index", 1),
                        )),
                        change_var("row_start", self.length),
                        change_var("valid", 1),
                    )),
                )),
            ),
        ).compile()


class CachedCausalScoreInstruction(Instruction):
    """Append current K and score Q against an MHA/GQA/MQA persistent cache."""

    def __init__(self, torch_name, output, *args, cache_prepopulated=False):
        self.cache_prepopulated = cache_prepopulated
        super().__init__(torch_name, output, *args)

    def prepare(self):
        q_shape = self.args[0].shape
        k_shape = self.args[1].shape
        self.heads = q_shape[-3]
        self.width = q_shape[-1]
        self.scale = self.args[3].value
        self.query_offset = self.args[4].value
        self.key_offset = self.args[5].value
        self.kv_heads = k_shape[-3]
        self.cache_embed = self.kv_heads * self.width
        self.query_heads_per_kv = self.heads // self.kv_heads

    def finalize(self):
        dot = _unrolled_offset_dot_statements(
            self.width,
            left_list="T1",
            right_list="T3",
            right_offset="key_offset",
        )
        cache_append = () if self.cache_prepopulated else (
            set_var("copy_index", self.key_offset + 1),
            *_static_unrolled_repeat(self.cache_embed, (
                append("T3", item("T2", var("copy_index"))),
                change_var("copy_index", 1),
            )),
        )
        return Program(
            "causal_score_cached",
            variables=(
                "copy_index", "cache_tokens", "query_start", "left_index",
                "key_head", "key_offset", "heads_on_key", "sum",
            ),
            lists=("T1", "T2", "T3", "T4"),
            body=(
                *cache_append,
                clear("T4"),
                set_var("cache_tokens", div(length("T3"), self.cache_embed)),
                set_var("query_start", self.query_offset + 1),
                set_var("key_head", 1),
                set_var("heads_on_key", 0),
                repeat(self.heads, (
                    set_var("key_offset", sub(var("key_head"), var("query_start"))),
                    repeat(var("cache_tokens"), (
                        set_var("sum", 0),
                        set_var("left_index", var("query_start")),
                        *dot,
                        append("T4", mul(var("sum"), self.scale)),
                        change_var("key_offset", self.cache_embed),
                    )),
                    change_var("query_start", self.width),
                    change_var("heads_on_key", 1),
                    if_(gt(var("heads_on_key"), self.query_heads_per_kv - 1), (
                        set_var("heads_on_key", 0),
                        change_var("key_head", self.width),
                    )),
                )),
            ),
        ).compile()


class CachedSoftmaxInstruction(Instruction):
    """Stable softmax over the dynamic cache length for each attention head."""

    def prepare(self):
        self.heads = self.args[2].value

    def finalize(self):
        current = item("T1", var("index"))
        return Program(
            "softmax_cached",
            variables=("group_start", "group_size", "index", "max", "sum"),
            lists=("T1", "T2"),
            body=(
                clear("T2"),
                set_var("group_start", 1),
                set_var("group_size", div(length("T1"), self.heads)),
                repeat(self.heads, (
                    set_var("index", var("group_start")),
                    set_var("max", current),
                    change_var("index", 1),
                    repeat(sub(var("group_size"), 1), (
                        if_(gt(current, var("max")), (set_var("max", current),)),
                        change_var("index", 1),
                    )),
                    set_var("index", var("group_start")),
                    set_var("sum", 0),
                    repeat(var("group_size"), (
                        change_var("sum", mathop("e ^", sub(current, var("max")))),
                        change_var("index", 1),
                    )),
                    set_var("index", var("group_start")),
                    repeat(var("group_size"), (
                        append(
                            "T2",
                            div(mathop("e ^", sub(current, var("max"))), var("sum")),
                        ),
                        change_var("index", 1),
                    )),
                    change_var("group_start", var("group_size")),
                )),
            ),
        ).compile()


class FastCachedSoftmaxInstruction(CachedSoftmaxInstruction):
    """Unstabilized dynamic-cache softmax for fast mode."""

    def finalize(self):
        current = item("T1", var("index"))
        stored = item("T2", var("index"))
        return Program(
            "softmax_cached_fast",
            variables=("group_start", "group_size", "index", "sum", "exponential"),
            lists=("T1", "T2"),
            body=(
                clear("T2"),
                set_var("group_start", 1),
                set_var("group_size", div(length("T1"), self.heads)),
                repeat(self.heads, (
                    set_var("index", var("group_start")),
                    set_var("sum", 0),
                    repeat(var("group_size"), (
                        set_var("exponential", mathop("e ^", current)),
                        change_var("sum", var("exponential")),
                        append("T2", var("exponential")),
                        change_var("index", 1),
                    )),
                    set_var("index", var("group_start")),
                    repeat(var("group_size"), (
                        replace("T2", var("index"), div(stored, var("sum"))),
                        change_var("index", 1),
                    )),
                    change_var("group_start", var("group_size")),
                )),
            ),
        ).compile()


class CachedSoftmaxValueInstruction(Instruction):
    """Fuse long-context stable softmax with cached probability-times-V."""

    def prepare(self):
        self.heads = self.args[2].value
        self.kv_heads = self.args[3].value
        self.width = self.args[4].value
        self.cache_embed = self.kv_heads * self.width
        self.query_heads_per_kv = self.heads // self.kv_heads

    def finalize(self):
        current = item("T1", var("index"))
        return Program(
            "softmax_value_cached_fused",
            variables=(
                "group start", "group size", "index", "maximum",
                "denominator", "exponential", "value head",
                "heads on value", "feature", "value index", "sum",
            ),
            lists=("T1", "T2", "T3"),
            body=(
                clear("T3"),
                set_var("group start", 1),
                set_var("group size", div(length("T1"), self.heads)),
                set_var("value head", 1),
                set_var("heads on value", 0),
                repeat(self.heads, (
                    set_var("index", var("group start")),
                    set_var("maximum", current),
                    change_var("index", 1),
                    repeat(sub(var("group size"), 1), (
                        if_(
                            gt(current, var("maximum")),
                            (set_var("maximum", current),),
                        ),
                        change_var("index", 1),
                    )),
                    set_var("index", var("group start")),
                    set_var("denominator", 0),
                    repeat(var("group size"), (
                        set_var(
                            "exponential",
                            mathop("e ^", sub(current, var("maximum"))),
                        ),
                        replace("T1", var("index"), var("exponential")),
                        change_var("denominator", var("exponential")),
                        change_var("index", 1),
                    )),
                    set_var("feature", 0),
                    repeat(self.width, (
                        set_var("sum", 0),
                        set_var("index", var("group start")),
                        set_var(
                            "value index",
                            add(var("value head"), var("feature")),
                        ),
                        repeat(var("group size"), (
                            change_var(
                                "sum",
                                mul(
                                    item("T1", var("index")),
                                    item("T2", var("value index")),
                                ),
                            ),
                            change_var("index", 1),
                            change_var("value index", self.cache_embed),
                        )),
                        append("T3", div(var("sum"), var("denominator"))),
                        change_var("feature", 1),
                    )),
                    change_var("group start", var("group size")),
                    change_var("heads on value", 1),
                    if_(
                        gt(
                            var("heads on value"),
                            self.query_heads_per_kv - 1,
                        ),
                        (
                            set_var("heads on value", 0),
                            change_var("value head", self.width),
                        ),
                    ),
                )),
            ),
        ).compile()


class CachedValueMatMulInstruction(Instruction):
    """Append V and apply cached MHA/GQA/MQA values to current probabilities."""

    def __init__(self, torch_name, output, *args, cache_prepopulated=False):
        self.cache_prepopulated = cache_prepopulated
        super().__init__(torch_name, output, *args)

    def prepare(self):
        v_shape = self.args[1].shape
        probability_shape = self.args[0].shape
        self.heads = probability_shape[-3]
        self.kv_heads = v_shape[-3]
        self.width = v_shape[-1]
        self.cache_embed = self.kv_heads * self.width
        self.query_heads_per_kv = self.heads // self.kv_heads
        self.value_offset = self.args[3].value

    def finalize(self):
        cache_append = () if self.cache_prepopulated else (
            set_var("copy_index", self.value_offset + 1),
            *_static_unrolled_repeat(self.cache_embed, (
                append("T3", item("T2", var("copy_index"))),
                change_var("copy_index", 1),
            )),
        )
        return Program(
            "causal_value_matmul_cached",
            variables=(
                "copy_index", "cache_tokens", "probability_start",
                "probability_index", "value_head", "value_index",
                "heads_on_value", "feature", "sum",
            ),
            lists=("T1", "T2", "T3", "T4"),
            body=(
                *cache_append,
                clear("T4"),
                set_var("cache_tokens", div(length("T3"), self.cache_embed)),
                set_var("probability_start", 1),
                set_var("value_head", 1),
                set_var("heads_on_value", 0),
                repeat(self.heads, (
                    set_var("feature", 0),
                    repeat(self.width, (
                        set_var("sum", 0),
                        set_var("probability_index", var("probability_start")),
                        set_var("value_index", add(var("value_head"), var("feature"))),
                        repeat(var("cache_tokens"), (
                            change_var(
                                "sum",
                                mul(
                                    item("T1", var("probability_index")),
                                    item("T3", var("value_index")),
                                ),
                            ),
                            change_var("probability_index", 1),
                            change_var("value_index", self.cache_embed),
                        )),
                        append("T4", var("sum")),
                        change_var("feature", 1),
                    )),
                    change_var("probability_start", var("cache_tokens")),
                    change_var("heads_on_value", 1),
                    if_(gt(var("heads_on_value"), self.query_heads_per_kv - 1), (
                        set_var("heads_on_value", 0),
                        change_var("value_head", self.width),
                    )),
                )),
            ),
        ).compile()


class CausalValueMatMulInstruction(Instruction):
    """Exact attention-probability @ V matmul skipping causal zero entries."""

    def prepare(self):
        left_shape = self.args[0].shape
        right_shape = self.args[1].shape
        self.groups = math.prod(left_shape[:-2])
        self.length = left_shape[-1]
        self.width = right_shape[-1]
        self.value_offset = self.args[2].value if len(self.args) > 2 else 0

    def finalize(self):
        product = change_var(
            "sum",
            mul(item("T1", var("probability_index")), item("T2", var("value_index"))),
        )
        return Program(
            "causal_value_matmul_optimized",
            variables=(
                "probability_row", "probability_index", "value_group_start",
                "value_index", "feature", "valid", "sum",
            ),
            lists=("T1", "T2", "T3"),
            body=(
                clear("T3"),
                set_var("probability_row", 1),
                set_var("value_group_start", self.value_offset + 1),
                repeat(self.groups, (
                    set_var("valid", 1),
                    repeat(self.length, (
                        set_var("feature", 0),
                        repeat(self.width, (
                            set_var("sum", 0),
                            set_var("probability_index", var("probability_row")),
                            set_var("value_index", add(var("value_group_start"), var("feature"))),
                            repeat(var("valid"), (
                                product,
                                change_var("probability_index", 1),
                                change_var("value_index", self.width),
                            )),
                            append("T3", var("sum")),
                            change_var("feature", 1),
                        )),
                        change_var("probability_row", self.length),
                        change_var("valid", 1),
                    )),
                    change_var("value_group_start", self.length * self.width),
                )),
            ),
        ).compile()


class QKVLinearInstruction(Instruction):
    """Write combined MHA/GQA/MQA QKV directly in head-major segments."""

    def __init__(
        self,
        torch_name,
        output,
        *args,
        direct_caches=False,
        hidden_prefill=False,
        weight_shard_specs=(),
    ):
        self.direct_caches = direct_caches
        self.hidden_prefill = hidden_prefill
        self.weight_shard_specs = tuple(weight_shard_specs)
        super().__init__(torch_name, output, *args)

    def prepare(self):
        input_shape = self.args[0].shape
        weight_shape = self.args[1].shape
        self.batches = math.prod(input_shape[:-2]) if len(input_shape) > 2 else 1
        self.length = input_shape[-2]
        self.embed = input_shape[-1]
        self.query_heads = self.args[3].value
        self.kv_heads = self.args[4].value
        self.head_width = self.embed // self.query_heads
        self.projection_heads = self.query_heads + 2 * self.kv_heads
        self.has_bias = self.args[2].value is not None or bool(self.args[2].shape)
        expected_rows = self.projection_heads * self.head_width
        if weight_shape != torch.Size((expected_rows, self.embed)):
            raise ValueError(
                "QKV projection weight must match query and KV head counts"
            )

    def finalize(self):
        accumulator = item("T3", var("bias_index")) if self.has_bias else 0
        dot = _unrolled_offset_dot_statements(
            self.embed,
            right_list="T2",
            right_offset="weight_offset",
        )
        if self.direct_caches:
            if self.weight_shard_specs:
                shard_count = len(self.weight_shard_specs)
                weight_lists = tuple(
                    f"T{index + 2}" for index in range(shard_count)
                )
                bias_list = f"T{shard_count + 2}"
                key_cache = f"T{shard_count + 3}"
                value_cache = f"T{shard_count + 4}"
                output_list = f"T{shard_count + 5}"
                destinations = {
                    "query": output_list,
                    "key": key_cache,
                    "value": value_cache,
                }
                def shard_projection(
                    weight_list, rows, destination, bias_start, interleaved,
                ):
                    if interleaved:
                        group_names = ("sum a", "sum b", "sum c", "sum d")
                        step = (
                            set_var("input value", item("T1", var("left_index"))),
                            *tuple(
                                change_var(
                                    name,
                                    mul(
                                        var("input value"),
                                        item(weight_list, add(var("weight_index"), offset)),
                                    ),
                                )
                                for offset, name in enumerate(group_names)
                            ),
                            change_var("weight_index", 4),
                            change_var("left_index", 1),
                        )
                        groups = []
                        for offset, name in enumerate(group_names):
                            initial = (
                                item(
                                    bias_list,
                                    add(var("bias_index"), offset),
                                )
                                if self.has_bias else 0
                            )
                            groups.append(set_var(name, initial))
                        return (
                            set_var("weight_index", 1),
                            set_var("bias_index", bias_start),
                            repeat(rows // 4, (
                                *groups,
                                set_var("left_index", var("token_input")),
                                repeat(self.embed, step),
                                *(append(destination, var(name)) for name in group_names),
                                change_var("bias_index", 4),
                            )),
                        )
                    shard_dot = _unrolled_offset_dot_statements(
                        self.embed,
                        right_list=weight_list,
                        right_offset="weight_offset",
                    )
                    return (
                        set_var("weight_offset", sub(1, var("token_input"))),
                        set_var("bias_index", bias_start),
                        repeat(rows, (
                            set_var(
                                "sum",
                                item(bias_list, var("bias_index"))
                                if self.has_bias else 0,
                            ),
                            set_var("left_index", var("token_input")),
                            *shard_dot,
                            append(destination, var("sum")),
                            change_var("weight_offset", self.embed),
                            change_var("bias_index", 1),
                        )),
                    )

                projections = []
                bias_start = 1
                for weight_list, (destination, rows, interleaved) in zip(
                    weight_lists, self.weight_shard_specs,
                ):
                    projection = shard_projection(
                        weight_list,
                        rows,
                        destinations[destination],
                        bias_start,
                        interleaved,
                    )
                    if destination == "query" and self.hidden_prefill:
                        projection = (
                            if_(
                                eq(var("cattorch project output"), 1),
                                projection,
                            ),
                        )
                    projections.extend(projection)
                    bias_start += rows
                return Program(
                    "qkv_linear_cached_sharded",
                    variables=(
                        "token_input", "weight_offset", "bias_index",
                        "left_index", "weight_index", "input value", "sum",
                        "sum a", "sum b", "sum c", "sum d",
                        "cattorch project output",
                    ),
                    variable_values={"cattorch project output": 1},
                    lists=tuple(
                        f"T{index}" for index in range(1, shard_count + 6)
                    ),
                    body=(
                        clear(output_list),
                        set_var("token_input", 1),
                        *projections,
                    ),
                ).compile()

            def project(heads, destination):
                return (
                    repeat(heads, (
                        set_var("batch_input", 1),
                        repeat(self.batches, (
                            set_var("token_input", var("batch_input")),
                            repeat(self.length, (
                                set_var(
                                    "weight_offset",
                                    sub(var("segment_weight"), var("token_input")),
                                ),
                                set_var("bias_index", var("segment_bias")),
                                repeat(self.head_width, (
                                    set_var("sum", accumulator),
                                    set_var("left_index", var("token_input")),
                                    *dot,
                                    append(destination, var("sum")),
                                    change_var("weight_offset", self.embed),
                                    change_var("bias_index", 1),
                                )),
                                change_var("token_input", self.embed),
                            )),
                            change_var("batch_input", self.length * self.embed),
                        )),
                        change_var("segment_weight", self.head_width * self.embed),
                        change_var("segment_bias", self.head_width),
                    )),
                )

            query = project(self.query_heads, "T6")
            if self.hidden_prefill:
                query = (
                    if_(eq(var("cattorch project output"), 1), query),
                    if_(eq(var("cattorch project output"), 0), (
                        change_var(
                            "segment_weight",
                            self.query_heads * self.head_width * self.embed,
                        ),
                        change_var(
                            "segment_bias",
                            self.query_heads * self.head_width,
                        ),
                    )),
                )
            return Program(
                "qkv_linear_cached",
                variables=(
                    "segment_weight", "segment_bias", "batch_input",
                    "token_input", "weight_offset", "bias_index",
                    "left_index", "sum", "cattorch project output",
                ),
                variable_values={"cattorch project output": 1},
                lists=("T1", "T2", "T3", "T4", "T5", "T6"),
                body=(
                    clear("T6"),
                    set_var("segment_weight", 1),
                    set_var("segment_bias", 1),
                    *query,
                    *project(self.kv_heads, "T4"),
                    *project(self.kv_heads, "T5"),
                ),
            ).compile()

        return Program(
            "qkv_linear_head_major",
            variables=(
                "segment_weight", "segment_bias", "batch_input",
                "token_input", "weight_offset", "bias_index",
                "left_index", "sum",
            ),
            lists=("T1", "T2", "T3", "T4"),
            body=(
                clear("T4"),
                set_var("segment_weight", 1),
                set_var("segment_bias", 1),
                repeat(self.projection_heads, (
                    set_var("batch_input", 1),
                    repeat(self.batches, (
                        set_var("token_input", var("batch_input")),
                        repeat(self.length, (
                            set_var(
                                "weight_offset",
                                sub(var("segment_weight"), var("token_input")),
                            ),
                            set_var("bias_index", var("segment_bias")),
                            repeat(self.head_width, (
                                set_var("sum", accumulator),
                                set_var("left_index", var("token_input")),
                                *dot,
                                append("T4", var("sum")),
                                change_var("weight_offset", self.embed),
                                change_var("bias_index", 1),
                            )),
                            change_var("token_input", self.embed),
                        )),
                        change_var("batch_input", self.length * self.embed),
                    )),
                    change_var("segment_weight", self.head_width * self.embed),
                    change_var("segment_bias", self.head_width),
                )),
            ),
        ).compile()


class FastPrunedMatMulInstruction(OptimizedMatMulInstruction):
    """Static-RHS matmul that skips zero four-wide reduction blocks."""

    def __init__(self, torch_name, output, *args, weight):
        self.weight = weight
        super().__init__(torch_name, output, *args)

    def prepare(self):
        super().prepare()
        self.block_starts = []
        self.block_counts = []
        self.block_offsets = []
        full_width = self.inner - self.inner % 4
        total_blocks = self.columns * (full_width // 4)
        for column in range(self.columns):
            self.block_starts.append(len(self.block_offsets) + 1)
            before = len(self.block_offsets)
            for start in range(0, full_width, 4):
                if bool(torch.any(self.weight[start:start + 4, column] != 0)):
                    self.block_offsets.append(start)
            self.block_counts.append(len(self.block_offsets) - before)
        self.use_sparse = (
            total_blocks > 0
            and len(self.block_offsets) <= total_blocks * 0.75
        )

    def finalize(self):
        if not self.use_sparse:
            return super().finalize()

        block_dot = _unrolled_dot_statements(
            4,
            right_list="T2",
            right_index="right_index",
            right_step=self.columns,
        )
        remainder = self.inner % 4
        remainder_dot = ()
        if remainder:
            offset = self.inner - remainder
            remainder_dot = (
                set_var("left_index", add(var("left_start"), offset)),
                set_var("right_index", add(var("right_start"), offset * self.columns)),
                *_unrolled_dot_statements(
                    remainder,
                    right_list="T2",
                    right_index="right_index",
                    right_step=self.columns,
                ),
            )
        dot = (
            set_var("sum", 0),
            set_var(
                "block_map_index",
                item("_matmul_block_starts", var("column_index")),
            ),
            repeat(item("_matmul_block_counts", var("column_index")), (
                set_var(
                    "block_offset",
                    item("_matmul_block_offsets", var("block_map_index")),
                ),
                set_var("left_index", add(var("left_start"), var("block_offset"))),
                set_var(
                    "right_index",
                    add(var("right_start"), mul(var("block_offset"), self.columns)),
                ),
                *block_dot,
                change_var("block_map_index", 1),
            )),
            *remainder_dot,
            append("T3", var("sum")),
        )
        return Program(
            "matmul_pruned",
            variables=(
                "sum", "left_start", "right_start", "left_index", "right_index",
                "column_index", "block_map_index", "block_offset",
            ),
            lists=(
                "T1", "T2", "T3", "_matmul_block_starts",
                "_matmul_block_counts", "_matmul_block_offsets",
            ),
            list_values={
                "_matmul_block_starts": self.block_starts,
                "_matmul_block_counts": self.block_counts,
                "_matmul_block_offsets": self.block_offsets,
            },
            body=(
                clear("T3"),
                set_var("left_start", 1),
                repeat(self.rows, (
                    set_var("right_start", 1),
                    set_var("column_index", 1),
                    repeat(self.columns, (
                        *dot,
                        change_var("right_start", 1),
                        change_var("column_index", 1),
                    )),
                    change_var("left_start", self.inner),
                )),
            ),
        ).compile()


class FastLowRankMatMulInstruction(Instruction):
    """Static-RHS matmul lowered as two smaller dense matmuls."""

    def prepare(self):
        left_shape = self.args[0].shape
        self.rows = math.prod(left_shape[:-1]) if len(left_shape) > 1 else 1
        self.inner = left_shape[-1]
        self.rank = self.args[1].shape[1]
        self.columns = self.args[2].shape[1]

    @staticmethod
    def _dot(inner, left_list, right_list, right_step, destination):
        return (
            set_var("sum", 0),
            set_var("left_index", var("left_start")),
            set_var("right_index", var("right_start")),
            *_unrolled_dot_statements(
                inner,
                left_list=left_list,
                right_list=right_list,
                right_index="right_index",
                right_step=right_step,
            ),
            append(destination, var("sum")),
        )

    def finalize(self):
        first_dot = self._dot(self.inner, "T1", "T2", self.rank, "_matmul_low_rank")
        second_dot = self._dot(
            self.rank, "_matmul_low_rank", "T3", self.columns, "T4",
        )
        return Program(
            "matmul_low_rank",
            variables=("sum", "left_start", "right_start", "left_index", "right_index"),
            lists=("T1", "T2", "T3", "T4", "_matmul_low_rank"),
            body=(
                clear("_matmul_low_rank"),
                set_var("left_start", 1),
                repeat(self.rows, (
                    set_var("right_start", 1),
                    repeat(self.rank, (
                        *first_dot,
                        change_var("right_start", 1),
                    )),
                    change_var("left_start", self.inner),
                )),
                clear("T4"),
                set_var("left_start", 1),
                repeat(self.rows, (
                    set_var("right_start", 1),
                    repeat(self.columns, (
                        *second_dot,
                        change_var("right_start", 1),
                    )),
                    change_var("left_start", self.rank),
                )),
            ),
        ).compile()
