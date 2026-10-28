"""Train a tiny character-level transformer and export a Scratch text generator.

    pip install "cattorch[tokenizers]"
    python examples/text_generator.py

writes ``build/rhyme.sprite3``, its tokenizer ``build/rhyme_tokenizer.sprite3``,
and a runnable ``build/rhyme.sb3``: click the green flag, type the start of a
line such as "twinkle", and the Demo sprite continues it a character at a time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

import cattorch

TEXT = (
    "twinkle twinkle little star. how i wonder what you are. "
    "up above the world so high. like a diamond in the sky. "
) * 4
CHARACTERS = "?" + "".join(sorted(set(TEXT)))
CONTEXT = 48


class TinyGPT(nn.Module):
    """A decoder-only transformer: ``forward(tokens)`` returns next-token logits."""

    def __init__(self, vocab: int = len(CHARACTERS), width: int = 32, layers: int = 2):
        super().__init__()
        self.tokens = nn.Embedding(vocab, width)
        self.positions = nn.Embedding(CONTEXT, width)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(width, 4, 4 * width, dropout=0.0, batch_first=True)
            for _ in range(layers)
        )
        self.head = nn.Linear(width, vocab)

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.tokens(tokens) + self.positions(torch.arange(length))
        mask = nn.Transformer.generate_square_subsequent_mask(length)
        for block in self.blocks:
            x = block(x, src_mask=mask, is_causal=True)
        return self.head(x)


def tokenizer():
    """A Hugging Face character tokenizer, which cattorch can export."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {character: index for index, character in enumerate(CHARACTERS)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="?"))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="?")


def trained(steps: int = 400) -> TinyGPT:
    torch.manual_seed(0)
    model = TinyGPT()
    ids = torch.tensor([CHARACTERS.index(character) for character in TEXT])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        starts = torch.randint(0, len(ids) - CONTEXT - 1, (16,))
        window = torch.stack([ids[start:start + CONTEXT + 1] for start in starts])
        logits = model(window[:, :-1])
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), window[:, 1:].flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


def main(
    output: Path = Path("build"), steps: int = 400, max_new_tokens: int = 30,
) -> cattorch.ProjectResult:
    model = trained(steps)
    result = cattorch.transpile(
        model, cattorch.GenerationProgram(max_context=CONTEXT), output / "rhyme",
    )
    words = cattorch.CharTokenizer(tokenizer()).save(
        output / "rhyme_tokenizer", name="tokenizer",
    )
    return cattorch.build_project(
        output / "rhyme", result, tokenizer=words, max_new_tokens=max_new_tokens,
        temperature=0.5,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("build"))
    parser.add_argument("--steps", type=int, default=400)
    arguments = parser.parse_args()
    print(main(arguments.output, arguments.steps).path)
