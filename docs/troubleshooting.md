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

Call `model.eval()` before export. cattorch raises an error on training-mode
dropout instead of quietly removing it.

## Keep one inference return path

Tracing records a single path through `forward`. It can't follow `if`
statements or loops that depend on tensor values. If training and inference
return different things, export an inference-only method:

```python
def forward(self, x, targets=None):
    logits = self.head(x)
    if targets is None:
        return logits
    return logits, F.cross_entropy(logits, targets)

def forward_inference(self, x):
    return self.head(x)

model.eval()
model.forward = model.forward_inference
```

## Unpack module tuples in the wrapper

Some transformer modules return tuples. Take the tensor out before passing it
to the next layer:

```python
# Incorrect when block returns (hidden_states, attention_weights, ...)
x = block(x)

# Correct
x = block(x)[0]
```

The model's return value can only contain tensors. Several tensors, even in
nested tuples, lists, or dictionaries, become `output`, `output_1`, and so on.
Remove anything that isn't a tensor.

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
