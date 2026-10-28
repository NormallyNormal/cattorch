# Troubleshooting exports

[Documentation home](index.md)

Problems usually happen at one of three stages:

1. **Tracing.** PyTorch couldn't trace the model's Python code. The exception
   or its cause comes from `torch.fx` or `torch.export`.
2. **Conversion.** cattorch traced the model but can't convert an operation.
   The exception is an `UnsupportedOperationError` or `UnsupportedModelError`.
3. **Running in Scratch.** The sprite exported but doesn't work in a project.
   Check `result.warnings`, `expanded_json_bytes`, and `cattorch status`, then
   try the sprite with the smallest possible surrounding script.

## Put the model in evaluation mode

Call `model.eval()` before export. A model with modules still in training mode
is exported as an evaluation-mode copy with a warning: dropout is disabled
and batch norm uses its running statistics. Your model itself is unchanged.

## Keep one inference return path

Tracing records a single path through `forward`. It can't follow `if`
statements or loops that depend on the model's inputs. Branches on values
that only depend on shapes, parameters, buffers, or constants, such as a mask
built from `torch.arange`, are fine: they are evaluated once during export.
If training and inference return different things, mark an inference-only
method with [`@export_forward`](markers.md#an-export-only-forward). cattorch
traces it instead of `forward`, and your model is unchanged:

```python
from cattorch import export_forward

class Net(nn.Module):
    def forward(self, x, targets=None):
        logits = self.head(x)
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits, targets)

    @export_forward
    def inference(self, x):
        return self.head(x)
```

The marked method takes the arguments `transpile` passes and should return
what `forward` returns for them, since `verify` compares the export with
`forward`. It works on any module in the model, not only the top one.

## Unpack module tuples in the wrapper

Some transformer modules return tuples. Take the tensor out before passing it
to the next layer:

```python
# Incorrect when block returns (hidden_states, attention_weights, ...)
x = block(x)

# Correct
x = block(x)[0]
```

Several returned tensors, even in nested tuples, lists, or dictionaries,
become `output`, `output_1`, and so on. Anything that isn't a tensor, such as
`None` or a Hugging Face `DynamicCache`, is skipped with a warning naming it.
Hugging Face models have their cache turned off on the export copy, so this
warning only appears if you pass `example_kwargs={"use_cache": True}`.

## Treat example inputs as the interface

The sprite's input lists have exactly the shapes and dtypes of the example
inputs. Pass examples with the shape your Scratch project will use. Extra
tensor arguments map to `input_1`, `input_2`, and so on.

GPTQ calibration inputs are different: they should be realistic data, and are
only used to choose quantized weights. See
[GPTQ calibration](storage.md#gptq-calibration).

## Check sprite-local data

The generated `input` and `output` lists and My Blocks belong to the generated
sprite, and other sprites can't use them directly. If clicking the blocks works
but calling from another sprite doesn't, add the
[global-list bridge](getting-started.md#connect-another-sprite).

Don't replace a generated list with a new list of the same name. Blocks refer to
lists by ID, so they keep using the original.

## Preserve processor costumes

If the model runs but gives wrong output, check whether the generated sprite's
costumes were changed. They store the compressed weights; see
[preserve processor costumes](storage.md#preserve-processor-costumes).

## Read the error

`UnsupportedOperationError` names the operation that failed and the module it
came from. `UnsupportedModelError` covers other problems, such as a non-tensor
return value. Both subclass `CattorchError`, so you can catch that one class.
See [exceptions](api-reference.md#exceptions).

## Report an issue

When [reporting an issue](https://github.com/NormallyNormal/cattorch/issues),
include:

- The full exception, including any chained cause.
- A minimal model definition and export call.
- Your cattorch, PyTorch, and Python versions, and for Scratch problems, the
  browser and Scratch client.
- Example input shapes and dtypes, and for GPTQ, the calibration data shape and
  count.
- Whether the model was in evaluation mode.
- `artifact.warnings` and `expanded_json_bytes`, if export succeeded.
- Whether `verify` passed, and whether the problem happens on import, run, or
  save.
