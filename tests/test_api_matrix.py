"""Every ``transpile`` option against every kind of export, and ``verify`` for each kind.

An option either works on a kind of export or raises a clear error naming
it; nothing is silently dropped. ``verify`` takes the same model-changing
options as ``transpile`` and passes with tight tolerances on every kind.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

from cattorch import (
    CattorchWarning, CodegenConfig, EntryPoint, ExportProgram, FastConfig, GenerationProgram,
    GenerationResult, Input, ModuleAdapter, Output, ProgramCall, ProgramResult,
    QuantizationConfig, RecurrentGenerationProgram, State, StorageConfig, TranspileResult,
    entrypoint, transpile, verify,
)

sys.path.insert(0, str(Path(__file__).parent))
from test_autoregressive import NanoGPT  # noqa: E402


class Doubler(nn.Module):
    """A module the adapter below replaces, changing the result."""

    def forward(self, value):
        return value * 2


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(4, 8)
        self.doubler = Doubler()
        self.second = nn.Linear(8, 3)

    def forward(self, value):
        return self.second(torch.relu(self.doubler(self.first(value))))


class Marked(Net):
    @entrypoint(example_inputs=lambda: (torch.randn(2, 4),))
    def run(self, value):
        return self.forward(value)


class Recurrent(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.doubler = Doubler()
        self.gru = nn.GRU(8, 8, batch_first=True)
        self.head = nn.Linear(8, 16)

    def step(self, token, hidden):
        output, hidden = self.gru(self.doubler(self.embed(token)), hidden)
        return self.head(output), hidden


class Generator(NanoGPT):
    def __init__(self):
        super().__init__(layers=1, width=8, heads=2, context=8)
        self.doubler = Doubler()

    def forward(self, tokens):
        return self.doubler(super().forward(tokens))


ADAPTER = ModuleAdapter(Doubler, lambda module, context: nn.Identity())
TOKENS = torch.tensor([[1, 2, 3]])


def kinds():
    torch.manual_seed(0)
    net = Net().eval()
    program = ExportProgram((EntryPoint(
        "run", arguments=(Input("value", torch.randn(2, 4)),), returns=(Output("y"),),
    ),))
    return {
        "tensors": (net, torch.randn(2, 4)),
        "program": (Net().eval(), program),
        "markers": (Marked().eval(), None),
        "generation": (Generator().eval(), GenerationProgram(8, top_k=2)),
        "recurrent": (
            Recurrent().eval(),
            RecurrentGenerationProgram((State("hidden", torch.zeros(1, 1, 8)),), max_context=8),
        ),
    }


def overrides(kind, bits=None):
    """Symmetric int8 with another setting for each kind's output layer."""
    layer = "second" if kind in {"tensors", "program", "markers"} else "head"
    return QuantizationConfig(bits=8, min_quantized_values=1, overrides={layer: bits})


GPTQ = QuantizationConfig(bits=8, method="gptq", min_quantized_values=1, min_calibration_rows=1)
SYMMETRIC = QuantizationConfig(bits=8, min_quantized_values=1)
# int4 with an explicit int6 output head; generation exports only.
HEAD4 = QuantizationConfig(bits=4, min_quantized_values=1, head_bits=6)


def calibration(kind):
    """GPTQ calibration data, in the form each kind of export takes."""
    if kind == "tensors":
        return [torch.randn(2, 4) for _ in range(2)]
    if kind in {"program", "markers"}:
        return [ProgramCall("run", (torch.randn(2, 4),)) for _ in range(2)]
    return [TOKENS, TOKENS.flip(1)]


# Each option: (keyword arguments, or a function of the kind giving them;
# the kinds it raises for, with the error and message).
OPTIONS = {
    "name": ({"name": "Sprite"}, {}),
    "optimization": ({"optimization": "fast"}, {}),
    "fast_config": ({"optimization": "fast", "fast_config": FastConfig(softmax=False)}, {}),
    "fast_config_without_fast": ({"fast_config": FastConfig()}, {
        kind: (ValueError, "fast_config can only be used with optimization='fast'")
        for kind in ("tensors", "program", "markers", "generation", "recurrent")
    }),
    "storage": ({"storage": StorageConfig(precision="float16")}, {}),
    "quantization": ({"quantization": SYMMETRIC}, {}),
    "quantization_overrides": (lambda kind: {"quantization": overrides(kind, 4)}, {}),
    "head_bits": ({"quantization": HEAD4}, {
        kind: (ValueError, "head_bits applies to generation exports.*overrides")
        for kind in ("tensors", "program", "markers")
    }),
    "gptq": (lambda kind: {"quantization": GPTQ, "calibration": calibration(kind)}, {}),
    "gptq_without_calibration": ({"quantization": GPTQ}, {
        kind: (ValueError, "GPTQ needs calibration=")
        for kind in ("tensors", "program", "markers", "generation", "recurrent")
    }),
    "calibration_without_gptq": (lambda kind: {"calibration": calibration(kind)}, {
        kind: (ValueError, "calibration= is only used by GPTQ")
        for kind in ("tensors", "program", "markers", "generation", "recurrent")
    }),
    "sig_figs": ({"sig_figs": 5}, {}),
    "codegen": ({"codegen": CodegenConfig(unrolling="compact", id_namespace="abc")}, {}),
    "adapters": ({"adapters": (ADAPTER,)}, {}),
    "kernels": ({"kernels": ()}, {}),
    "example_kwargs": ({"example_kwargs": {}}, {
        "program": (ValueError, "example_kwargs can't be used with an ExportProgram"),
        "markers": (ValueError, "example_kwargs can't be used with an ExportProgram"),
        "generation": (ValueError, "example_kwargs can't be used with a GenerationProgram"),
        "recurrent": (ValueError, "example_kwargs can't be used with a RecurrentGenerationProgram"),
    }),
    "dynamic": ({"dynamic": {0: (1, 4)}}, {
        "program": (ValueError, r"Input\(\.\.\., dynamic=\.\.\.\)"),
        "markers": (ValueError, r"Input\(\.\.\., dynamic=\.\.\.\)"),
        "generation": (ValueError, "dynamic= can't be used with a GenerationProgram"),
        "recurrent": (ValueError, "dynamic= can't be used with a RecurrentGenerationProgram"),
    }),
}
RESULTS = {
    "tensors": TranspileResult, "program": ProgramResult, "markers": ProgramResult,
    "generation": GenerationResult, "recurrent": GenerationResult,
}


@pytest.mark.parametrize("kind", sorted(RESULTS))
@pytest.mark.parametrize("option", sorted(OPTIONS))
def test_every_option_works_or_is_rejected_on_every_kind(tmp_path, option, kind):
    model, example = kinds()[kind]
    arguments, rejected = OPTIONS[option]
    if callable(arguments):
        arguments = arguments(kind)
    if kind in rejected:
        error, message = rejected[kind]
        with pytest.raises(error, match=message):
            transpile(model, example, tmp_path / "sprite", **arguments)
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CattorchWarning)
        result = transpile(model, example, tmp_path / "sprite", **arguments)
    assert isinstance(result, RESULTS[kind])
    if option == "name":
        assert result.sprite_name == "Sprite"
    if option == "head_bits":
        assert [item.override for item in result.quantization.tensors if item.bits == 6] == [
            "head_bits",
        ]


@pytest.mark.parametrize("quantization", ["symmetric", "overrides"])
@pytest.mark.parametrize("kind", sorted(RESULTS))
def test_verify_reproduces_the_export_options_on_every_kind(tmp_path, kind, quantization):
    model, example = kinds()[kind]
    options = dict(
        adapters=(ADAPTER,),
        quantization=SYMMETRIC if quantization == "symmetric" else overrides(kind, 4),
        storage=StorageConfig(precision="float16"), sig_figs=5,
    )
    result = transpile(model, example, tmp_path / "sprite", **options)
    report = verify(model, example, result, atol=1e-3, rtol=1e-4, **options)
    assert report.passed, report
    # Without the export's options PyTorch runs another model: the adapter
    # changes the result.
    assert not verify(model, example, result, atol=1e-3, rtol=1e-4).passed
    if quantization == "overrides":
        # The override is reproduced too: without it, the layer is int8.
        options["quantization"] = SYMMETRIC
        assert not verify(model, example, result, atol=1e-3, rtol=1e-4, **options).passed


@pytest.mark.parametrize("head_bits", ["auto", 4, None])
@pytest.mark.parametrize("kind", ["generation", "recurrent"])
def test_verify_reproduces_the_output_head_of_an_int4_generation_export(
    tmp_path, kind, head_bits,
):
    # The default int8 head, an int4 one, and a float16 one: each sprite
    # passes verify with its own config, and fails it with another head.
    model, example = kinds()[kind]
    config = QuantizationConfig(bits=4, min_quantized_values=1, head_bits=head_bits)
    result = transpile(model, example, tmp_path / "sprite", quantization=config)
    head = {"auto": 8, 4: 4, None: None}[head_bits]
    assert {
        item.bits for item in result.quantization.tensors if item.name.startswith("head")
    } == {head}
    assert verify(model, example, result, atol=1e-5, rtol=1e-5, quantization=config).passed
    other = QuantizationConfig(
        bits=4, min_quantized_values=1, head_bits=6 if head_bits == "auto" else "auto",
    )
    assert not verify(model, example, result, atol=1e-5, rtol=1e-5, quantization=other).passed
    with pytest.raises(ValueError, match="head_bits applies to generation exports"):
        verify(
            *kinds()["tensors"], result, quantization=QuantizationConfig(bits=4, head_bits=8),
        )


def test_a_neuron_pruned_model_torch_fx_cant_trace_is_rejected(tmp_path):
    model, program = kinds()["generation"]
    with pytest.raises(Exception, match="neuron_pruning"):
        transpile(
            model, program, tmp_path / "pruned", optimization="fast",
            fast_config=FastConfig(neuron_pruning=0.5),
        )


def test_an_explicit_method_wins_over_generate(tmp_path):
    from cattorch import generate

    class Marked(Generator):
        @generate
        def other(self, tokens):
            return -self.forward(tokens)

    model = Marked().eval()
    for method, sign in ((None, -1), ("forward", 1)):
        program = GenerationProgram(8, method=method)
        result = transpile(model, program, tmp_path / str(method))
        assert verify(model, program, result).passed
        reference = GenerationProgram(8, method="other" if sign < 0 else "forward")
        assert verify(model, reference, result).passed
