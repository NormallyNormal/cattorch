"""Train a tiny character-level GRU and export it as a recurrent Scratch text generator.

    pip install "cattorch[tokenizers]"
    python examples/recurrent_generator.py

writes ``build/gru.sprite3``, its tokenizer ``build/gru_tokenizer.sprite3``,
and a runnable ``build/gru.sb3``: click the green flag, type the start of a
line such as "twinkle", and the Demo sprite continues it a character at a
time. Unlike a transformer, the GRU has no context limit.
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
WIDTH = 48


class CharGRU(nn.Module):
    """``forward(tokens)`` for training; ``step(token, hidden)`` for generation."""

    def __init__(self, vocab: int = len(CHARACTERS), width: int = WIDTH):
        super().__init__()
        self.tokens = nn.Embedding(vocab, width)
        self.gru = nn.GRU(width, width, batch_first=True)
        self.head = nn.Linear(width, vocab)

    def forward(self, tokens):
        output, _ = self.gru(self.tokens(tokens))
        return self.head(output)

    def step(self, token, hidden):
        output, hidden = self.gru(self.tokens(token), hidden)
        return self.head(output), hidden


def tokenizer():
    """A Hugging Face character tokenizer, which cattorch can export."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {character: index for index, character in enumerate(CHARACTERS)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="?"))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="?")


def trained(steps: int = 300) -> CharGRU:
    torch.manual_seed(0)
    model = CharGRU()
    ids = torch.tensor([CHARACTERS.index(character) for character in TEXT])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        starts = torch.randint(0, len(ids) - 33, (16,))
        window = torch.stack([ids[start:start + 33] for start in starts])
        logits = model(window[:, :-1])
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), window[:, 1:].flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


def main(
    output: Path = Path("build"), steps: int = 300, max_new_tokens: int = 30,
) -> cattorch.ProjectResult:
    model = trained(steps)
    program = cattorch.RecurrentGenerationProgram(
        (cattorch.State("hidden", torch.zeros(1, 1, WIDTH)),), reference="forward",
    )
    result = cattorch.transpile(model, program, output / "gru")
    words = cattorch.CharTokenizer(tokenizer()).save(output / "gru_tokenizer", name="tokenizer")
    return cattorch.build_project(
        output / "gru", result, tokenizer=words, max_new_tokens=max_new_tokens,
        temperature=0.5,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("build"))
    parser.add_argument("--steps", type=int, default=300)
    arguments = parser.parse_args()
    print(main(arguments.output, arguments.steps).path)
