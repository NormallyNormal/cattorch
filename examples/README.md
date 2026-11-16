# Examples

Each script trains a small model for a few seconds, exports it, checks it
against PyTorch, and writes a runnable Scratch project to `build/`. Open the
`.sb3` with Scratch's **File > Load from your computer**.

| Script | Model | In Scratch |
|---|---|---|
| [`mlp.py`](mlp.py) | MLP that names the quadrant of a point | Edit the point in `quadrants input` and press space. |
| [`cnn.py`](cnn.py) | CNN that tells horizontal lines from vertical ones in 8×8 images | Set pixels in `lines input` (row by row) and press space. |
| [`text_generator.py`](text_generator.py) | Character-level transformer trained on a nursery rhyme | Click the green flag and type the start of a line. |
| [`recurrent_generator.py`](recurrent_generator.py) | Character-level GRU trained on the same rhyme, exported with a [`RecurrentGenerationProgram`](../docs/recurrent-generation.md) | Click the green flag and type the start of a line. |

```bash
python examples/mlp.py
python examples/cnn.py
pip install "cattorch[tokenizers]"   # for the text generator's tokenizer
python examples/text_generator.py
python examples/recurrent_generator.py
```

## Use a model from your own sprites

Each project holds the exported sprite, named after it (`quadrants`,
`lines`, `rhyme`, `gru`), and a small Demo sprite, which the project opens
with. You never need to open the model's sprite. From a sprite of your own,
fill its input list, broadcast its call and wait, then read its status:

```text
when green flag clicked
delete all of [quadrants input]
add (0.5) to [quadrants input]
add (-0.25) to [quadrants input]
broadcast [quadrants forward] and wait
if <(quadrants status) = [ok]> then
  broadcast [quadrants predict] and wait
  say (quadrants prediction)
end
```

To put a model in a project of your own, upload `build/quadrants.sprite3`
(**Upload Sprite**), then select another sprite: Scratch creates the
`quadrants ...` lists and variables, and the script above works, except
`quadrants predict`, which the built project adds: pick the label of the
largest item of `quadrants output` yourself. See
[using a sprite without opening it](../docs/scratch-interface.md#using-a-sprite-without-opening-it).

## Options

`--output DIR` writes somewhere other than `build/`. The models export from
the command line too, for example:

```bash
cattorch export examples/mlp.py:trained --project \
    --labels "top right,top left,bottom left,bottom right"
cattorch export examples/cnn.py:trained --shape 1,1,8,8 --project --labels horizontal,vertical
```

See [projects and the command line](../docs/projects-and-cli.md).
