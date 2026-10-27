# Examples

Each script trains a small model for a few seconds, exports it, checks it
against PyTorch, and writes a runnable Scratch project to `build/`. Open the
`.sb3` with Scratch's **File > Load from your computer**.

| Script | Model | In Scratch |
|---|---|---|
| [`mlp.py`](mlp.py) | MLP that names the quadrant of a point | Edit the point in `quadrants input` and press space. |
| [`cnn.py`](cnn.py) | CNN that tells horizontal lines from vertical ones in 8×8 images | Set pixels in `lines input` (row by row) and press space. |
| [`text_generator.py`](text_generator.py) | Character-level transformer trained on a nursery rhyme | Click the green flag and type the start of a line. |

```bash
python examples/mlp.py
python examples/cnn.py
pip install "cattorch[tokenizers]"   # for the text generator's tokenizer
python examples/text_generator.py
```

`--output DIR` writes somewhere other than `build/`. The models export from
the command line too, for example:

```bash
cattorch export examples/mlp.py:trained --project \
    --labels "top right,top left,bottom left,bottom right"
cattorch export examples/cnn.py:trained --shape 1,1,8,8 --project --labels horizontal,vertical
```

See [projects and the command line](../docs/projects-and-cli.md).
