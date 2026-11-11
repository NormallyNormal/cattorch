"""Linear, matmul, and attention production kernels."""

import math

import torch

from cattorch.kernel_api import Kernel, KernelContext
from cattorch.util.scratch.dsl import (
    add,
    append,
    change_var,
    clear,
    div,
    gt,
    if_,
    if_else,
    item,
    length,
    mathop,
    mul,
    repeat,
    replace,
    set_var,
    sub,
    var,
)


from cattorch.util.instruction.optimized_common import (
    _unrolled_dot_statements,
    argument_at,
    _unrolled_offset_dot_statements,
    quick_gelu_of,
    sigmoid_of,
    tanh_gelu_inner,
    tanh_of,
)

# Browser Scratch screens show that sharing one activation read across four
# output accumulators pays off at the transformer-sized 49,152-MAC matrices,
# but not at 128 x 128. A module attribute, so benchmark builders and tests
# can produce isolated before/after exports without a public flag.
GROUPED_MIN_MACS = 49_152


def grouped_linear(columns: int, inner: int) -> bool:
    """Whether a ``columns`` x ``inner`` Linear runs four outputs at a time."""
    return columns % 4 == 0 and inner * columns >= GROUPED_MIN_MACS


def _zero_blocks(weight, columns: int, inner: int, column_of):
    """Starts, counts and offsets of each column's nonzero four-wide blocks.

    ``column_of(weight, column, start)`` is a column's block at ``start``.
    ``None`` unless at least a quarter of the blocks are zero: sparse
    indexing has overhead.
    """
    full_width = inner - inner % 4
    starts, counts, offsets = [], [], []
    for column in range(columns):
        starts.append(len(offsets) + 1)
        before = len(offsets)
        for start in range(0, full_width, 4):
            if bool(torch.any(column_of(weight, column, start) != 0)):
                offsets.append(start)
        counts.append(len(offsets) - before)
    total = columns * (full_width // 4)
    if total and len(offsets) <= total * 0.75:
        return starts, counts, offsets
    return None


_LINEAR_VARIABLES = (
    "sum", "left_start", "left_index", "weight_start",
    "weight_index", "weight_offset", "bias_index", "output_index", "value",
    "inner", "exponential",
    "column_index", "block_map_index", "block_offset",
    "sum a", "sum b", "sum c", "sum d", "input value",
    "weight", "weight a", "weight b", "weight c", "weight d",
)


def _linear_epilogue(epilogue, sum_name, out, residual, fast):
    """Statements applying ``epilogue`` to ``sum_name`` and appending it."""
    result = var(sum_name)
    setup = []
    terminal_activation = None
    for operation, operand in epilogue:
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
            result = sigmoid_of(result)
        elif operation == "tanh":
            result = tanh_of(result)
        elif operation == "silu":
            if result != var(sum_name):
                setup.append(set_var("value", result))
                result = var("value")
            result = mul(result, sigmoid_of(result))
        elif operation == "gelu":
            setup.append(set_var("value", result))
            result = var("value")
            if fast:
                result = quick_gelu_of(result)
            else:
                setup.append(set_var("inner", tanh_gelu_inner(result)))
                result = mul(mul(0.5, result), add(1, tanh_of(var("inner"))))
        elif operation == "tensor_add":
            result = add(result, item(residual, var("output_index")))
    if terminal_activation == "relu":
        output = if_else(gt(result, 0), (append(out, result),), (append(out, 0),))
    else:
        output = append(out, result)
    return (*setup, output)


def _emit_linear(ctx: KernelContext):
    """``aten.linear`` on the original [M, K] weight, with fused epilogues.

    The arguments are the input, one or more weight shards of whole output
    rows, the bias, and a residual added by the epilogue. Options:
    ``epilogue`` (``(operation, operand)`` pairs applied to each output),
    ``fast`` (approximate GELU), and ``interleaved`` (four-row groups stored
    side by side). The bias initializes each dot product.
    """
    names = ctx.argument_names
    source = ctx.inputs[names[0]]
    weights = [ctx.inputs[name] for name in names[1:-2]]
    bias, residual = ctx.inputs.get(names[-2]), ctx.inputs.get(names[-1])
    epilogue = tuple(ctx.options.get("epilogue", ()))
    fast = bool(ctx.options.get("fast"))
    interleaved = bool(ctx.options.get("interleaved"))
    input_shape = source.shape
    inner = input_shape[-1]
    rows = (
        div(length(source.list), inner) if source.dynamic_axis is not None
        else (math.prod(input_shape[:-1]) if len(input_shape) > 1 else 1)
    )
    shard_rows = tuple(weight.shape[0] for weight in weights)
    columns = sum(shard_rows)
    first_weight = weights[0]
    sparse = None
    value = first_weight.value
    if (
        not interleaved and len(weights) == 1 and isinstance(value, torch.Tensor)
        and value.ndim == 2 and inner >= 4
    ):
        sparse = _zero_blocks(
            value, columns, inner, lambda weight, column, start: weight[column, start:start + 4],
        )

    for name in _LINEAR_VARIABLES:
        ctx.temp_var(name)
    out = ctx.output.list
    residual_list = residual.list if residual else None
    accumulator = item(bias.list, var("bias_index")) if bias else 0
    epilogue_output = _linear_epilogue(epilogue, "sum", out, residual_list, fast)
    advance_bias = (change_var("bias_index", 1),) if bias else ()
    advance_output = (change_var("output_index", 1),) if residual else ()
    start_output = (set_var("output_index", 1),) if residual else ()
    start_bias = (set_var("bias_index", 1),) if bias else ()

    if sparse is None and grouped_linear(columns, inner):
        group_names = ("sum a", "sum b", "sum c", "sum d")
        group_setup = []
        for offset, name in enumerate(group_names):
            if bias:
                position = add(var("bias_index"), offset) if offset else var("bias_index")
                group_setup.append(set_var(name, item(bias.list, position)))
            else:
                group_setup.append(set_var(name, 0))
        group_setup.append(set_var("left_index", var("left_start")))
        group_outputs = []
        for name in group_names:
            group_outputs.extend(_linear_epilogue(epilogue, name, out, residual_list, fast))
            group_outputs.extend((*advance_bias, *advance_output))

        grouped_shards = []
        for weight, count in zip(weights, shard_rows):
            if count % 4:
                raise ValueError("grouped Linear shards must contain four-row groups")
            input_value = set_var("input value", item(source.list, var("left_index")))
            if interleaved:
                step = (
                    input_value,
                    *(
                        change_var(name, mul(
                            item(weight.list, add(var("weight"), offset) if offset else var("weight")),
                            var("input value"),
                        ))
                        for offset, name in enumerate(group_names)
                    ),
                    change_var("weight", 4),
                    change_var("left_index", 1),
                )
                shard_setup, shard_finish = (set_var("weight", 1),), ()
            else:
                step = [input_value]
                for suffix in "abcd":
                    step.extend((
                        change_var(f"sum {suffix}", mul(
                            item(weight.list, var(f"weight {suffix}")), var("input value"),
                        )),
                        change_var(f"weight {suffix}", 1),
                    ))
                step = (*step, change_var("left_index", 1))
                shard_setup = tuple(
                    set_var(f"weight {suffix}", offset * inner + 1)
                    for offset, suffix in enumerate("abcd")
                )
                shard_finish = tuple(change_var(f"weight {suffix}", 3 * inner) for suffix in "abcd")
            grouped_shards.extend((
                *shard_setup,
                repeat(count // 4, (
                    *group_setup,
                    repeat(inner, step),
                    *group_outputs,
                    *shard_finish,
                )),
            ))
        row = (*start_bias, *grouped_shards, change_var("left_start", inner))
    elif sparse is not None:
        starts, counts, offsets = sparse
        block_starts = ctx.static_list("_linear_block_starts", starts)
        block_counts = ctx.static_list("_linear_block_counts", counts)
        block_offsets = ctx.static_list("_linear_block_offsets", offsets)
        remainder = inner % 4
        remainder_dot = ()
        if remainder:
            remainder_start = inner - remainder
            remainder_dot = (
                set_var("left_index", add(var("left_start"), remainder_start)),
                set_var("weight_index", add(var("weight_start"), remainder_start)),
                *_unrolled_dot_statements(
                    remainder, left_list=source.list, right_list=first_weight.list,
                    right_index="weight_index", right_step=1,
                ),
            )
        row = (
            set_var("weight_start", 1),
            set_var("column_index", 1),
            *start_bias,
            repeat(columns, (
                set_var("sum", accumulator),
                set_var("block_map_index", item(block_starts, var("column_index"))),
                repeat(item(block_counts, var("column_index")), (
                    set_var("block_offset", item(block_offsets, var("block_map_index"))),
                    set_var("left_index", add(var("left_start"), var("block_offset"))),
                    set_var("weight_index", add(var("weight_start"), var("block_offset"))),
                    *_unrolled_dot_statements(
                        4, left_list=source.list, right_list=first_weight.list,
                        right_index="weight_index", right_step=1,
                    ),
                    change_var("block_map_index", 1),
                )),
                *remainder_dot,
                *epilogue_output,
                change_var("weight_start", inner),
                *advance_bias,
                *advance_output,
                change_var("column_index", 1),
            )),
            change_var("left_start", inner),
        )
    else:
        dense_shards = []
        for weight, count in zip(weights, shard_rows):
            dense_shards.extend((
                set_var("weight_offset", sub(1, var("left_start"))),
                repeat(count, (
                    set_var("sum", accumulator),
                    set_var("left_index", var("left_start")),
                    *_unrolled_offset_dot_statements(
                        inner, left_list=source.list, right_list=weight.list,
                        right_offset="weight_offset",
                    ),
                    *epilogue_output,
                    change_var("weight_offset", inner),
                    *advance_bias,
                    *advance_output,
                )),
            ))
        row = (*start_bias, *dense_shards, change_var("left_start", inner))
    return (clear(out), set_var("left_start", 1), *start_output, repeat(rows, row))


LINEAR_KERNEL = Kernel("cattorch.linear", _emit_linear, name="linear_optimized")


class _MatMulLayout:
    def __init__(self, ctx: KernelContext):
        self.left, self.right = argument_at(ctx, 0), argument_at(ctx, 1)
        left_shape, right_shape = self.left.shape, self.right.shape
        self.inner = left_shape[-1]
        self.tensor_rhs = len(right_shape) > 2
        # A stored right side kept as [batch, columns, inner], each matrix
        # transposed, so integer storage groups it over its inputs.
        self.transposed = bool(ctx.options.get("transposed_right", False))
        if self.tensor_rhs:
            self.batches = math.prod(right_shape[:-2])
            self.rows = left_shape[-2]
            self.columns = right_shape[-2] if self.transposed else right_shape[-1]
        else:
            self.batches = 1
            self.rows = (
                div(length(self.left.list), self.inner) if self.left.dynamic_axis is not None
                else (math.prod(left_shape[:-1]) if len(left_shape) > 1 else 1)
            )
            self.columns = right_shape[-1] if len(right_shape) > 1 else 1


def _emit_matmul(ctx: KernelContext):
    """Matmul whose innermost loops use running flat indices."""
    layout = _MatMulLayout(ctx)
    for name in (
        "sum", "left_start", "right_start", "right_batch_start",
        "left_index", "right_index", "right_offset",
    ):
        ctx.temp_var(name)
    out = ctx.output.list
    left, right, inner = layout.left.list, layout.right.list, layout.inner

    def dot_body(right_step, advance=1):
        if right_step == 1:
            setup = (
                set_var("left_index", var("left_start")),
                set_var("right_offset", sub(var("right_start"), var("left_start"))),
            )
            dot = _unrolled_offset_dot_statements(
                inner, left_list=left, right_list=right, right_offset="right_offset",
            )
        else:
            setup = (
                set_var("left_index", var("left_start")),
                set_var("right_index", var("right_start")),
            )
            dot = _unrolled_dot_statements(
                inner, left_list=left, right_list=right, right_index="right_index",
                right_step=right_step,
            )
        return (
            set_var("sum", 0), *setup, *dot, append(out, var("sum")),
            change_var("right_start", advance),
        )

    if layout.tensor_rhs:
        return (
            clear(out),
            set_var("left_start", 1),
            set_var("right_batch_start", 1),
            repeat(layout.batches, (
                repeat(layout.rows, (
                    set_var("right_start", var("right_batch_start")),
                    repeat(
                        layout.columns,
                        dot_body(1, inner) if layout.transposed else dot_body(layout.columns),
                    ),
                    change_var("left_start", inner),
                )),
                change_var("right_batch_start", inner * layout.columns),
            )),
        )
    right_step = layout.columns if len(layout.right.shape) > 1 else 1
    return (
        clear(out),
        set_var("left_start", 1),
        repeat(layout.rows, (
            set_var("right_start", 1),
            repeat(layout.columns, dot_body(right_step)),
            change_var("left_start", inner),
        )),
    )


def matmul_kernel(target: str) -> Kernel:
    """The matmul kernel for ``target``, such as ``aten.mm.default``."""
    return Kernel(target, _emit_matmul, name="matmul_optimized")


def pruned_matmul_blocks(weight: torch.Tensor):
    """The nonzero four-row blocks of a static [inner, columns] right side.

    ``None`` when too few blocks are zero for the pruned kernel to pay off.
    """
    inner, columns = weight.shape
    return _zero_blocks(
        weight, columns, inner, lambda value, column, start: value[start:start + 4, column],
    )


def _emit_pruned_matmul(ctx: KernelContext):
    """Static right side matmul that skips its zero four-wide reduction blocks."""
    layout = _MatMulLayout(ctx)
    starts, counts, offsets = pruned_matmul_blocks(layout.right.value)
    for name in (
        "sum", "left_start", "right_start", "left_index", "right_index",
        "column_index", "block_map_index", "block_offset",
    ):
        ctx.temp_var(name)
    out = ctx.output.list
    block_starts = ctx.static_list("_matmul_block_starts", starts)
    block_counts = ctx.static_list("_matmul_block_counts", counts)
    block_offsets = ctx.static_list("_matmul_block_offsets", offsets)
    left, right = layout.left.list, layout.right.list
    inner, columns = layout.inner, layout.columns

    def block_dot(count):
        return _unrolled_dot_statements(
            count, left_list=left, right_list=right, right_index="right_index", right_step=columns,
        )

    remainder = inner % 4
    remainder_dot = ()
    if remainder:
        offset = inner - remainder
        remainder_dot = (
            set_var("left_index", add(var("left_start"), offset)),
            set_var("right_index", add(var("right_start"), offset * columns)),
            *block_dot(remainder),
        )
    dot = (
        set_var("sum", 0),
        set_var("block_map_index", item(block_starts, var("column_index"))),
        repeat(item(block_counts, var("column_index")), (
            set_var("block_offset", item(block_offsets, var("block_map_index"))),
            set_var("left_index", add(var("left_start"), var("block_offset"))),
            set_var("right_index", add(var("right_start"), mul(var("block_offset"), columns))),
            *block_dot(4),
            change_var("block_map_index", 1),
        )),
        *remainder_dot,
        append(out, var("sum")),
    )
    return (
        clear(out),
        set_var("left_start", 1),
        repeat(layout.rows, (
            set_var("right_start", 1),
            set_var("column_index", 1),
            repeat(columns, (*dot, change_var("right_start", 1), change_var("column_index", 1))),
            change_var("left_start", inner),
        )),
    )


def pruned_matmul_kernel(target: str) -> Kernel:
    """The pruned matmul kernel for ``target``; its right side is stored."""
    return Kernel(target, _emit_pruned_matmul, name="matmul_pruned")


def _emit_low_rank_matmul(ctx: KernelContext):
    """Static right side matmul lowered as two smaller dense matmuls."""
    source, first, second = ctx.inputs["input"], ctx.inputs["first"], ctx.inputs["second"]
    left_shape = source.shape
    traced_rows = math.prod(left_shape[:-1]) if len(left_shape) > 1 else 1
    inner, rank, columns = left_shape[-1], first.shape[1], second.shape[1]
    # A variable-length input sets the rows when the code runs;
    # ``ctx.options["most_rows"]`` bounds them.
    rows = div(length(source.list), inner) if source.dynamic_axis is not None else traced_rows
    for name in ("sum", "left_start", "right_start", "left_index", "right_index"):
        ctx.temp_var(name)
    out = ctx.output.list
    low_rank = ctx.temp_list(
        "_matmul_low_rank", ctx.options.get("most_rows", traced_rows) * rank,
    )

    def dot(count, left_list, right_list, right_step, destination):
        return (
            set_var("sum", 0),
            set_var("left_index", var("left_start")),
            set_var("right_index", var("right_start")),
            *_unrolled_dot_statements(
                count, left_list=left_list, right_list=right_list,
                right_index="right_index", right_step=right_step,
            ),
            append(destination, var("sum")),
        )

    def product(width, count, left_list, right_list, destination):
        return (
            clear(destination),
            set_var("left_start", 1),
            repeat(rows, (
                set_var("right_start", 1),
                repeat(width, (
                    *dot(count, left_list, right_list, width, destination),
                    change_var("right_start", 1),
                )),
                change_var("left_start", count),
            )),
        )

    return (
        *product(rank, inner, source.list, first.list, low_rank),
        *product(columns, rank, low_rank, second.list, out),
    )


LOW_RANK_MATMUL_KERNEL = Kernel(
    "cattorch.low_rank_matmul", _emit_low_rank_matmul, name="matmul_low_rank",
    arguments=("input", "first", "second"),
)


def _emit_causal_score(ctx: KernelContext):
    """Exact QK score matmul specialized for an upper-triangular ``-inf`` mask.

    Queries and keys may start ``query_offset`` and ``key_offset`` values
    into their lists, as in a combined QKV projection.
    """
    query, key = ctx.inputs["query"], ctx.inputs["key"]
    shape = query.shape
    groups, sequence, width = math.prod(shape[:-2]), shape[-2], shape[-1]
    scale = ctx.scalars["scale"]
    for name in ("query_start", "left_index", "key_group_start", "key_offset", "valid", "sum"):
        ctx.temp_var(name)
    out = ctx.output.list
    dot = _unrolled_offset_dot_statements(
        width, left_list=query.list, right_list=key.list, right_offset="key_offset",
    )
    return (
        clear(out),
        set_var("query_start", ctx.scalars.get("query_offset", 0) + 1),
        set_var("key_group_start", ctx.scalars.get("key_offset", 0) + 1),
        repeat(groups, (
            set_var("valid", 1),
            repeat(sequence, (
                set_var("key_offset", sub(var("key_group_start"), var("query_start"))),
                repeat(var("valid"), (
                    set_var("sum", 0),
                    set_var("left_index", var("query_start")),
                    *dot,
                    append(out, mul(var("sum"), scale)),
                    change_var("key_offset", width),
                )),
                repeat(sub(sequence, var("valid")), (append(out, float("-inf")),)),
                change_var("query_start", width),
                change_var("valid", 1),
            )),
            change_var("key_group_start", sequence * width),
        )),
    )


CAUSAL_SCORE_KERNEL = Kernel(
    "cattorch.causal_score", _emit_causal_score, name="causal_score_optimized",
    arguments=("query", "key", "scale", "query_offset", "key_offset"),
)


def _causal_rows(ctx: KernelContext):
    source = argument_at(ctx, 0)
    return source, math.prod(source.shape[:-2]), source.shape[-1]


def _emit_causal_softmax(ctx: KernelContext):
    """Stable softmax that skips known ``-inf`` causal positions."""
    source, groups, sequence = _causal_rows(ctx)
    for name in ("row_start", "index", "valid", "max", "sum"):
        ctx.temp_var(name)
    out = ctx.output.list
    current = item(source.list, var("index"))
    return (
        clear(out),
        set_var("row_start", 1),
        repeat(groups, (
            set_var("valid", 1),
            repeat(sequence, (
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
                    append(out, div(mathop("e ^", sub(current, var("max"))), var("sum"))),
                    change_var("index", 1),
                )),
                repeat(sub(sequence, var("valid")), (append(out, 0),)),
                change_var("row_start", sequence),
                change_var("valid", 1),
            )),
        )),
    )


def _emit_fast_causal_softmax(ctx: KernelContext):
    """Unstabilized causal softmax used by fast mode."""
    source, groups, sequence = _causal_rows(ctx)
    for name in ("row_start", "index", "valid", "sum", "exponential"):
        ctx.temp_var(name)
    out = ctx.output.list
    return (
        clear(out),
        set_var("row_start", 1),
        repeat(groups, (
            set_var("valid", 1),
            repeat(sequence, (
                set_var("index", var("row_start")),
                set_var("sum", 0),
                repeat(var("valid"), (
                    set_var("exponential", mathop("e ^", item(source.list, var("index")))),
                    change_var("sum", var("exponential")),
                    append(out, var("exponential")),
                    change_var("index", 1),
                )),
                repeat(sub(sequence, var("valid")), (append(out, 0),)),
                set_var("index", var("row_start")),
                repeat(var("valid"), (
                    replace(out, var("index"), div(item(out, var("index")), var("sum"))),
                    change_var("index", 1),
                )),
                change_var("row_start", sequence),
                change_var("valid", 1),
            )),
        )),
    )


CAUSAL_SOFTMAX_KERNEL = Kernel(
    "aten.softmax.int", _emit_causal_softmax, name="causal_softmax_optimized",
)
FAST_CAUSAL_SOFTMAX_KERNEL = Kernel(
    "aten.softmax.int", _emit_fast_causal_softmax, name="causal_softmax_fast",
)


def _emit_causal_value_matmul(ctx: KernelContext):
    """Exact attention-probability @ V matmul skipping causal zero entries."""
    probabilities, values = ctx.inputs["probabilities"], ctx.inputs["values"]
    groups = math.prod(probabilities.shape[:-2])
    sequence, width = probabilities.shape[-1], values.shape[-1]
    for name in (
        "probability_row", "probability_index", "value_group_start",
        "value_index", "feature", "valid", "sum",
    ):
        ctx.temp_var(name)
    out = ctx.output.list
    product = change_var("sum", mul(
        item(probabilities.list, var("probability_index")), item(values.list, var("value_index")),
    ))
    return (
        clear(out),
        set_var("probability_row", 1),
        set_var("value_group_start", ctx.scalars.get("value_offset", 0) + 1),
        repeat(groups, (
            set_var("valid", 1),
            repeat(sequence, (
                set_var("feature", 0),
                repeat(width, (
                    set_var("sum", 0),
                    set_var("probability_index", var("probability_row")),
                    set_var("value_index", add(var("value_group_start"), var("feature"))),
                    repeat(var("valid"), (
                        product,
                        change_var("probability_index", 1),
                        change_var("value_index", width),
                    )),
                    append(out, var("sum")),
                    change_var("feature", 1),
                )),
                change_var("probability_row", sequence),
                change_var("valid", 1),
            )),
            change_var("value_group_start", sequence * width),
        )),
    )


CAUSAL_VALUE_MATMUL_KERNEL = Kernel(
    "cattorch.causal_value_matmul", _emit_causal_value_matmul,
    name="causal_value_matmul_optimized",
    arguments=("probabilities", "values", "value_offset"),
)


def _emit_qkv_linear(ctx: KernelContext):
    """Write a combined MHA/GQA/MQA QKV projection in head-major segments."""
    source, weight, bias = ctx.inputs["input"], ctx.inputs["weight"], ctx.inputs.get("bias")
    input_shape = source.shape
    batches = math.prod(input_shape[:-2]) if len(input_shape) > 2 else 1
    sequence, embed = input_shape[-2], input_shape[-1]
    query_heads, kv_heads = ctx.scalars["query_heads"], ctx.scalars["kv_heads"]
    head_width = embed // query_heads
    projection_heads = query_heads + 2 * kv_heads
    if weight.shape != torch.Size((projection_heads * head_width, embed)):
        raise ValueError("QKV projection weight must match query and KV head counts")
    for name in (
        "segment_weight", "segment_bias", "batch_input", "token_input",
        "weight_offset", "bias_index", "left_index", "sum",
    ):
        ctx.temp_var(name)
    out = ctx.output.list
    accumulator = item(bias.list, var("bias_index")) if bias else 0
    dot = _unrolled_offset_dot_statements(
        embed, left_list=source.list, right_list=weight.list, right_offset="weight_offset",
    )
    return (
        clear(out),
        set_var("segment_weight", 1),
        set_var("segment_bias", 1),
        repeat(projection_heads, (
            set_var("batch_input", 1),
            repeat(batches, (
                set_var("token_input", var("batch_input")),
                repeat(sequence, (
                    set_var("weight_offset", sub(var("segment_weight"), var("token_input"))),
                    set_var("bias_index", var("segment_bias")),
                    repeat(head_width, (
                        set_var("sum", accumulator),
                        set_var("left_index", var("token_input")),
                        *dot,
                        append(out, var("sum")),
                        change_var("weight_offset", embed),
                        change_var("bias_index", 1),
                    )),
                    change_var("token_input", embed),
                )),
                change_var("batch_input", sequence * embed),
            )),
            change_var("segment_weight", head_width * embed),
            change_var("segment_bias", head_width),
        )),
    )


QKV_LINEAR_KERNEL = Kernel(
    "cattorch.qkv_linear", _emit_qkv_linear, name="qkv_linear_head_major",
    arguments=("input", "weight", "bias", "query_heads", "kv_heads"),
)


__all__ = [
    "CAUSAL_SCORE_KERNEL", "CAUSAL_SOFTMAX_KERNEL", "CAUSAL_VALUE_MATMUL_KERNEL",
    "FAST_CAUSAL_SOFTMAX_KERNEL", "GROUPED_MIN_MACS", "LINEAR_KERNEL",
    "LOW_RANK_MATMUL_KERNEL", "QKV_LINEAR_KERNEL", "grouped_linear", "matmul_kernel",
    "pruned_matmul_blocks", "pruned_matmul_kernel",
]
