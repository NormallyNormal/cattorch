"""A tied embedding and output head larger than a Scratch list is stored once.

Each case exports a tied model and its untied twin, whose head is a copy of
the same (storage-rounded) table, with the same settings. The tied export
must hold exactly one table fewer and compute exactly what the twin computes.
"""

import copy
import json
import sys
import zipfile
from pathlib import Path

import pytest
import torch
from torch import nn

from cattorch import (
    EntryPoint, ExportProgram, GenerationProgram, Input, Output, QuantizationConfig,
    RecurrentGenerationProgram, StorageConfig, transpile, verify,
)
from cattorch.quantization import _symmetric_matrix
from cattorch.util.scratch.emulator import ScratchEmulator

sys.path.insert(0, str(Path(__file__).parent))

LIMIT = 4000

FORMATS = {
    "int8": {"quantization": QuantizationConfig(bits=8, min_quantized_values=16)},
    "int4": {"quantization": QuantizationConfig(bits=4, min_quantized_values=16)},
    "float16": {"storage": StorageConfig(precision="float16")},
    "float32": {},
    "uncompressed": {"storage": StorageConfig(compression=False)},
}
# Of the largest logit, against PyTorch with the original weights.
TOLERANCE = {"int8": 0.04, "int4": 0.3, "float16": 0.01, "float32": 1e-5, "uncompressed": 1e-6}


class Tied(nn.Module):
    def __init__(self, vocabulary, width, index=False):
        super().__init__()
        self.index = index
        self.embedding = nn.Embedding(vocabulary, width)
        self.head = nn.Linear(width, vocabulary, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, tokens):
        logits = self.head(torch.tanh(self.embedding(tokens)))
        return (logits, self.embedding.weight[tokens]) if self.index else logits


def _round_tied(model, head, options):
    """Put the tied table on the quantization grid, as quantization would."""
    quantization = options.get("quantization")
    if quantization is not None:
        with torch.no_grad():
            head.weight.copy_(_symmetric_matrix(head.weight, quantization)[0])
    return model


def _untied(model, head, table):
    """A copy of ``model`` whose ``head`` has its own copy of ``table``'s weight."""
    twin = copy.deepcopy(model)
    twin.get_submodule(head).weight = nn.Parameter(twin.get_submodule(table).weight.detach().clone())
    return twin


def _sprite(result):
    with zipfile.ZipFile(result.path) as archive:
        return json.loads(archive.read("sprite.json"))


def _stored_weights(emulator):
    """Values in every weight list once the weights are decoded."""
    emulator.run_procedure("cattorch init")
    # A program prefixes each entry point's weights with its name.
    return sum(len(emulator.lists[name]) for name in emulator._list_name_to_ids if "W_" in name)


def _assert_stored_once(tied, untied, table_values):
    for result in (tied, untied):
        assert not [warning for warning in result.warnings if "tied" in warning], result.warnings
    stored = _stored_weights(ScratchEmulator(_sprite(untied))) - _stored_weights(ScratchEmulator(_sprite(tied)))
    assert stored == table_values


def _close(actual, expected, tolerance):
    actual = torch.as_tensor(actual, dtype=torch.float64).flatten()
    expected = expected.double().flatten()
    assert actual.shape == expected.shape
    assert (actual - expected).abs().max() <= tolerance * expected.abs().max()


@pytest.mark.parametrize("storage", sorted(FORMATS))
@pytest.mark.parametrize("vocabulary", [512, 514], ids=["grouped", "rows"])
def test_tied_table_is_stored_once(tmp_path, small_list_limit, vocabulary, storage):
    # 512 output rows run four at a time, so the table is stored in four-row
    # groups; 514 rows can't be, so it is stored in plain rows.
    small_list_limit(LIMIT)
    options = FORMATS[storage]
    torch.manual_seed(0)
    model = Tied(vocabulary, 128).eval()
    assert model.embedding.weight.numel() > LIMIT
    _round_tied(model, model.head, options)
    twin = _untied(model, "head", "embedding")
    tokens = torch.tensor([[7, 3, vocabulary - 1]])
    tied = transpile(model, tokens, tmp_path / "tied", **options)
    untied = transpile(twin, tokens, tmp_path / "untied", **options)
    _assert_stored_once(tied, untied, model.embedding.weight.numel())

    outputs = []
    for result in (tied, untied):
        emulator = ScratchEmulator(_sprite(result))
        emulator.lists["input"] = tokens.flatten().tolist()
        emulator.run_procedure("cattorch forward")
        outputs.append(list(emulator.lists["output"]))
    assert outputs[0] == outputs[1]
    with torch.no_grad():
        _close(outputs[0], model(tokens), TOLERANCE[storage])


def test_tied_table_with_another_reader_reports_its_second_copy(tmp_path, small_list_limit):
    # The third reader needs the plain layout, the grouped head its four-row
    # groups: two copies, and the result says so.
    small_list_limit(LIMIT)
    torch.manual_seed(0)
    model = Tied(512, 128, index=True).eval()
    tokens = torch.tensor([[7, 3]])
    result = transpile(model, tokens, tmp_path / "tied")
    (warning,) = [warning for warning in result.warnings if "tied" in warning]
    assert "stored twice" in warning
    assert verify(model, tokens, result).passed


def _gpt2(vocabulary, width):
    import transformers

    config = transformers.GPT2Config(
        vocab_size=vocabulary, n_positions=8, n_embd=width, n_layer=1, n_head=2,
        bos_token_id=0, eos_token_id=0,
    )
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(config).eval()


def _generate(result, tokens):
    emulator = ScratchEmulator(_sprite(result))
    emulator.run_procedure("cattorch init")
    emulator.lists["cattorch tokens"] = list(tokens)
    emulator.run_procedure("cattorch prefill")
    assert emulator.variables["cattorch status"] == "ok"
    return list(emulator.lists["cattorch logits"])


@pytest.mark.parametrize("storage", ["int8", "float16", "float32"])
@pytest.mark.parametrize(("vocabulary", "width"), [(512, 96), (257, 32)], ids=["grouped", "rows"])
def test_gpt2_style_tied_head_is_stored_once(tmp_path, small_list_limit, vocabulary, width, storage):
    small_list_limit(LIMIT)
    options = FORMATS[storage]
    model = _gpt2(vocabulary, width)
    assert model.lm_head.weight is model.transformer.wte.weight
    _round_tied(model, model.lm_head, options)
    twin = _untied(model, "lm_head", "transformer.wte")
    twin.config.tie_word_embeddings = False
    program = GenerationProgram(max_context=8)
    tied = transpile(model, program, tmp_path / "tied", **options)
    untied = transpile(twin, program, tmp_path / "untied", **options)
    _assert_stored_once(tied, untied, vocabulary * width)

    tokens = [3, 1, vocabulary - 1]
    logits = _generate(tied, tokens)
    assert logits == _generate(untied, tokens)
    with torch.no_grad():
        _close(logits, model(torch.tensor([tokens])).logits[0, -1], TOLERANCE[storage])


@pytest.mark.parametrize("storage", ["int8", "float16"])
def test_recurrent_tied_head_is_stored_once(tmp_path, small_list_limit, storage):
    from test_huggingface_recurrent import mamba2

    small_list_limit(LIMIT)
    options = FORMATS[storage]
    model = mamba2(
        seed=3, vocab_size=1024, hidden_size=48, head_dim=8, num_heads=12,
        tie_word_embeddings=True,
    )
    assert model.lm_head.weight is model.backbone.embeddings.weight
    _round_tied(model, model.lm_head, options)
    twin = _untied(model, "lm_head", "backbone.embeddings")
    twin.config.tie_word_embeddings = False
    tied = transpile(model, RecurrentGenerationProgram(), tmp_path / "tied", **options)
    untied = transpile(twin, RecurrentGenerationProgram(), tmp_path / "untied", **options)
    _assert_stored_once(tied, untied, model.lm_head.weight.numel())

    tokens = [3, 17, 1023]
    logits = _generate(tied, tokens)
    assert logits == _generate(untied, tokens)
    with torch.no_grad():
        expected = model(torch.tensor([tokens]), use_cache=False).logits[0, -1]
    _close(logits, expected, TOLERANCE[storage])


class Entries(Tied):
    def lookup(self, tokens):
        return self.embedding(tokens)

    def logits(self, hidden):
        return self.head(hidden)

    def both(self, tokens):
        return self.head(self.embedding(tokens))


@pytest.mark.parametrize("storage", ["int8", "float32"])
def test_program_entry_point_with_a_tied_head_stores_it_once(tmp_path, small_list_limit, storage):
    small_list_limit(LIMIT)
    options = FORMATS[storage]
    torch.manual_seed(0)
    model = Entries(512, 128).eval()
    _round_tied(model, model.head, options)
    program = ExportProgram(entrypoints=(
        EntryPoint("lookup", method="lookup", arguments=(Input("tokens", torch.tensor([3])),),
                   returns=(Output("rows"),)),
        EntryPoint("both", method="both", arguments=(Input("tokens", torch.tensor([3])),),
                   returns=(Output("logits"),)),
    ))
    tied = transpile(model, program, tmp_path / "tied", **options)
    untied = transpile(_untied(model, "head", "embedding"), program, tmp_path / "untied", **options)
    assert not [warning for warning in tied.warnings if "tied" in warning]
    stored = _stored_weights(ScratchEmulator(_sprite(untied))) - _stored_weights(ScratchEmulator(_sprite(tied)))
    assert stored == model.embedding.weight.numel()
    comparison = verify(
        model, program, tied, atol=1e-3 if storage == "int8" else 1e-4,
        quantization=options.get("quantization"),
    )
    assert comparison.passed, comparison
