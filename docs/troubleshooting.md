# Troubleshooting exports

## Put the model in evaluation mode

Call `model.eval()` before export. cattorch folds some evaluation-only patterns
and deliberately rejects training dropout rather than silently changing its
meaning.

## Keep one inference return path

`torch.export` captures one graph and cannot follow data-dependent Python
control flow. Expose an inference-only method when training and inference have
different returns:

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

Some transformer modules return tuples. Extract the tensor before sending it to
the next layer:

```python
# Incorrect when block returns (hidden_states, attention_weights, ...)
x = block(x)

# Export the tensor dataflow explicitly.
x = block(x)[0]
```

The model itself must ultimately return one tensor. Multiple outputs are
rejected because the generated sprite has one `output` list.

## Treat example inputs as the interface

cattorch specializes list shapes from the supplied values. Use the same rank,
shape, and dtype expected by the Scratch project. Additional tensor arguments
map to `input_1`, `input_2`, and so on.

## Read contextual errors

`UnsupportedOperationError` reports the exported operation and, when available,
its node and owning module. `UnsupportedModelError` reports unsupported graph
contracts such as unresolved state or multiple outputs. These are subclasses of
`CattorchError`, so callers can catch the common base while still presenting a
specific message.

When reporting an issue, include the full exception, a minimal model definition,
PyTorch and cattorch versions, the example input shapes/dtypes, and whether the
model was in evaluation mode.
