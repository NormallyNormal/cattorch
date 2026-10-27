# Custom kernels and operations

[Documentation home](index.md)

cattorch lowers every operation to Scratch blocks with a *kernel*. When an
operation has no kernel, it exports through PyTorch's decomposition into
simpler operations, which is often slower; `TranspileResult.fallback_operations`
lists those. You can write a kernel for such an operation, replace a built-in
one, or declare an operation of your own.

`Kernel`, `check_kernel`, and `build_kernel_benchmark` come from
`cattorch.experimental`, the statements from `cattorch.scratch`, and `op` from
`cattorch`. The kernel interface may change before 1.0.

## Write a kernel

A kernel is a function that receives a context describing one call and
returns Scratch statements. This one lowers `F.mish`:

```python
import torch
from cattorch import transpile, verify
from cattorch.experimental import Kernel
from cattorch.scratch import (
    add, append, change_var, clear, div, item, mathop, mul, repeat, set_var, sub, var,
)

def emit_mish(ctx):
    value = ctx.inputs["self"]          # the operator's schema argument name
    out = ctx.output.list
    index = ctx.temp_var("mish index")
    x = item(value.list, var(index))
    softplus = mathop("ln", add(1, mathop("e ^", x)))
    tanh = sub(div(2, add(1, mathop("e ^", mul(-2, softplus)))), 1)
    return (
        clear(out),
        set_var(index, 1),
        repeat(value.numel, (
            append(out, mul(x, tanh)),
            change_var(index, 1),
        )),
    )

mish = Kernel(torch.ops.aten.mish, emit_mish, name="mish")

model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Mish()).eval()
example = torch.randn(2, 4)
result = transpile(model, example, "model.sprite3", kernels=[mish])
assert result.fallback_operations == ()
assert verify(model, example, result).passed
```

`kernels=` applies to one export, including `GenerationProgram` and
`ExportProgram` exports and `analyze`. A kernel for an operation cattorch
already lowers replaces the built-in kernel for that export only.

### The context

- `ctx.inputs` maps the operator's schema argument names to tensors. Each has
  `list` (the Scratch list holding it, flattened in row-major order), `shape`,
  `dtype`, `numel`, `strides`, and `static` (true for a weight stored in the
  sprite). Find the names in the schema, for example with
  `print(torch.ops.aten.mish.default._schema)`.
- `ctx.scalars` holds the other arguments, with the schema's defaults filled
  in.
- `ctx.output` is the result tensor. Fill its `list` from empty.
- `ctx.temp_var(name)` declares a variable, and `ctx.temp_list(name, length)`
  a scratch list of up to `length` items. `ctx.static_list(name, values)`
  declares a list filled at export, for index tables and constants. Variables
  are shared by name across the sprite, so don't expect a value to survive
  between calls.
- Don't put per-call data, such as a layer number, in a kernel's `name` or
  body. Repeated layers share code only when their kernels are identical.

### Scratch statements

`cattorch.scratch` has expressions (`var`, `item`, `length`, `add`, `sub`,
`mul`, `div`, `mod`, `eq`, `lt`, `gt`, `not_`, `mathop`, `round_`), statements
(`set_var`, `change_var`, `clear`, `append`, `replace`, `delete`), loops
(`repeat`, `repeat_until`, `for_each`, and `static_unrolled_repeat` for a count
known at export), and conditions (`if_`, `if_else`). Numbers are literals.
`render(body)` prints a body as pseudocode.

Scratch lists are indexed from 1, its trigonometry uses degrees, and
`mathop` takes Scratch's names: `abs`, `floor`, `ceiling`, `sqrt`, `sin`,
`cos`, `tan`, `asin`, `acos`, `atan`, `ln`, `log`, `e ^`, and `10 ^`.

### Validators and approximate kernels

`Kernel(validator=...)` receives the same context and returns `None` to
accept the call or a reason to reject it. The reason is raised as
`UnsupportedOperationError`, so a kernel can refuse shapes it doesn't handle.

`Kernel(approximate=True)` marks a kernel that trades accuracy for speed. It
is used only with `optimization="fast"`; exact exports keep the built-in
lowering. `TranspileResult.approximate_operations` lists the operations that
used an approximate kernel, built-in or yours, and `verify` repeats the list
in its result.

## Declare an operation

`op` turns a function into an operation that exports whole, so a kernel can
lower it. Annotate every argument and the return type; they become the
operation's schema.

```python
from torch import Tensor
from cattorch import op

@op
def squash(x: Tensor, alpha: float = 1.0) -> Tensor:
    return x / (1 + alpha * x.abs())

@squash.kernel
def emit_squash(ctx):
    value, out = ctx.inputs["x"], ctx.output.list
    index = ctx.temp_var("squash index")
    current = item(value.list, var(index))
    return (
        clear(out),
        set_var(index, 1),
        repeat(value.numel, (
            append(out, div(current, add(1, mul(ctx.scalars["alpha"], mathop("abs", current))))),
            change_var(index, 1),
        )),
    )
```

Call `squash` in your model like the original function. The function is the
reference `verify` compares against. A kernel attached with
`@squash.kernel` is used in every export, without `kernels=`; it takes the
same `name`, `validator`, and `approximate` options as `Kernel`. Without a
kernel, the operation exports through its function, like any other
decomposed operation.

`op(name="mylib::squash")` sets the operation's qualified name. Shapes during
tracing come from running the function on shape-only tensors; pass
`fake=` with a function that returns empty tensors of the right shapes if
yours depend on values. The function must not modify its inputs.

## Test and time a kernel

`check_kernel` exports each sample with the kernel, runs it in cattorch's
Scratch emulator, and compares it with PyTorch. It raises `AssertionError`
for a mismatch, or when the export didn't use the kernel at all:

```python
from cattorch.experimental import check_kernel

check_kernel(mish, [(torch.randn(8),), (torch.randn(2, 3, 4) * 5,)])
check_kernel(squash.kernels[0], [(torch.randn(4), 0.5)])
```

Each sample is a tuple of arguments, or an `(args, kwargs)` pair. Tensors are
Scratch inputs, except the arguments named in `static=("weight",)`, which are
stored as weights.

The emulator checks results, not speed. `build_kernel_benchmark` writes a
project that times the export with and without your kernel in Scratch:

```python
from cattorch.experimental import build_kernel_benchmark

build_kernel_benchmark(mish, [("mish", model, (example,))], "mish.sb3")
```

Run it on [scratch.mit.edu](https://scratch.mit.edu) or in the desktop editor,
save it, and read the timings with
[`analyze_benchmark`](api-reference.md#analyzebenchmark).
