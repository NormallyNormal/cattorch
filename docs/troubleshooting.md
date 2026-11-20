# Troubleshooting exports

[Documentation home](index.md)

Problems usually happen at one of three stages:

1. **Tracing.** PyTorch couldn't trace the model's Python code. The exception
   or its cause comes from `torch.fx`.
2. **Conversion.** cattorch traced the model but can't convert an operation.
   The exception is an `UnsupportedOperationError` or `UnsupportedModelError`.
3. **Running in Scratch.** The sprite exported but doesn't work in a project.
   Check `result.warnings`, `expanded_json_bytes`, and `NAME status`
   (its values are in the [status table](scratch-interface.md#status-values)),
   then try the sprite with the smallest possible surrounding script. Call
   the sprite's blocks from one script at a time; see
   [overlapping calls](scratch-interface.md#overlapping-calls).

Code or a Scratch project written for cattorch 0.5.0, for example one that
fails with a `TypeError` naming a parameter or reads a list the sprite
doesn't have, needs the changes in [migrating to 1.0](migrating-to-1.0.md).

## Put the model in evaluation mode

Call `model.eval()` before export. A model is always exported as an
evaluation-mode copy: dropout is disabled and batch norm uses its running
statistics. When modules whose output depends on the mode (dropout, batch
norm, or a `forward` that reads `self.training`) are still in training mode,
the export warns, naming them. Your model itself is unchanged.

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

## Check the sprite's globals and broadcasts

A sprite is used through global lists and variables named after it
(`NAME input`, `NAME output`, `NAME status`) and its broadcasts
(`NAME forward`); see
[using a sprite without opening it](scratch-interface.md#using-a-sprite-without-opening-it).
If a call seems to do nothing:

- Check NAME. It is the export's `name` (by default the file name), not
  the sprite's name in your project: Scratch renames a second sprite of one
  name (`NAME2`), but its globals and broadcasts keep NAME.
- Two sprites exported under one NAME, such as two uploads of the same
  file, share their globals and both answer the same broadcasts, so a call
  runs in both and they overwrite each other's outputs. To use two copies
  of one model, export them with different `name`s.
- Read `NAME status` right after `broadcast [NAME forward] and wait`. It
  reads `not finished` when a second broadcast of the same message
  restarted the call, or after the stop sign stopped it; see
  [overlapping calls](scratch-interface.md#overlapping-calls).
- Don't make a new local list or variable with a global's name in one of
  your sprites: Scratch then gives that sprite its own copy, which the
  model never sees.

You never need to open the model's sprite, and a large one makes the editor
slow while it's selected. Uploading a sprite selects it, so select another
sprite afterwards.

## The editor freezes or crashes after uploading a sprite

Uploading a sprite (**Upload Sprite**) selects it, and the editor then shows
all its blocks, which for a large model can take minutes, run out of memory,
or crash the tab. Don't upload a large sprite: add it to your project outside
the editor with [`add_to_project`](projects-and-cli.md#add-sprites-to-your-own-project)
or `cattorch add PROJECT.sb3 SPRITE.sprite3 --output OUT.sb3`, then load
`OUT.sb3` with **File > Load from your computer**. The sprite goes after your
own, so the project opens on your first sprite and the model's is never shown.
For a new project, [`build_project`](projects-and-cli.md#build-a-project)
writes one that opens on its small Demo sprite. To put a retrained model in
place of one you added, use `add_to_project(..., replace=True)`
(`cattorch add --replace`) rather than uploading it.

## Preserve processor costumes

If the model runs but gives wrong output, check whether the generated sprite's
costumes were changed. The sprite uses them to decode the compressed weights; see
[preserve processor costumes](storage.md#preserve-processor-costumes).

## Generate with a recurrent model

A `GenerationProgram` only handles transformers whose tokens mix in causal
attention. For a Mamba, Mamba-2, or RWKV model from Hugging Face, use
`RecurrentGenerationProgram()`; for your own GRU, LSTM, or state-space model,
write a one-token `step` and pass its states. If a Hugging Face model's export
fails the reference check or reports a missing attribute, the installed
transformers version may compute the model differently from the one cattorch
was tested with. See [recurrent generation](recurrent-generation.md).

## Read the error

`UnsupportedOperationError` names the operation that failed and the module it
came from. `UnsupportedModelError` covers other problems with the model, such
as a non-tensor return value or tensors returned inside a dataclass, a
requirement of a `GenerationProgram` it doesn't meet, or an error the model
(or an `@entrypoint` method) raises on the example inputs (the message gives
their shapes). Both subclass `CattorchError`, so you can catch
that one class. A `TypeError` or `ValueError` means an argument is wrong.
Every warning is a `CattorchWarning` and is also in `result.warnings`. See
[exceptions](api-reference.md#exceptions).

## Report an issue

When [reporting an issue](https://github.com/NormallyNormal/cattorch/issues),
include:

- The full exception, including any chained cause.
- A minimal model definition and export call.
- Your cattorch (`cattorch --version`), PyTorch, and Python versions, and for Scratch problems, the
  browser and Scratch client.
- Example input shapes and dtypes, and for GPTQ, the calibration data shape and
  count.
- Whether the model was in evaluation mode.
- `artifact.warnings` and `expanded_json_bytes`, if export succeeded.
- Whether `verify` passed, and whether the problem happens on import, run, or
  save.
