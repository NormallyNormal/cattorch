"""Hugging Face Mamba, Mamba-2 and RWKV causal LMs as recurrent generators."""

from __future__ import annotations

import json
import zipfile

import pytest
import torch

from cattorch import (
    GenerationProgram, QuantizationConfig, RecurrentGenerationProgram, State,
    UnsupportedModelError, transpile,
)
from cattorch.util.scratch.emulator import ScratchEmulator

transformers = pytest.importorskip("transformers")

TOLERANCE = 1e-4
VOCAB = 32


def mamba2(seed=0, **overrides):
    """Mamba-2 with chunk_size 4, and decay and skip weights spread out so the
    state carries real information across chunk boundaries."""
    options = dict(
        vocab_size=VOCAB, hidden_size=16, num_hidden_layers=2, expand=2, num_heads=4,
        head_dim=8, state_size=4, n_groups=2, conv_kernel=3, chunk_size=4,
    )
    options.update(overrides)
    torch.manual_seed(seed)
    model = transformers.Mamba2ForCausalLM(transformers.Mamba2Config(**options)).eval()
    with torch.no_grad():
        for block in model.backbone.layers:
            block.mixer.dt_bias.normal_(0.0, 1.0)
            block.mixer.A_log.uniform_(-3.0, 0.0)
            block.mixer.D.normal_()
    return model


def mamba(seed=0, **overrides):
    options = dict(
        vocab_size=VOCAB, hidden_size=16, num_hidden_layers=2, expand=2, state_size=4,
        conv_kernel=3, time_step_rank=3,
    )
    options.update(overrides)
    torch.manual_seed(seed)
    return transformers.MambaForCausalLM(transformers.MambaConfig(**options)).eval()


def rwkv(seed=0, **overrides):
    options = dict(
        vocab_size=VOCAB, hidden_size=16, num_hidden_layers=2, attention_hidden_size=16,
        intermediate_size=32, context_length=64, rescale_every=0,
    )
    options.update(overrides)
    torch.manual_seed(seed)
    return transformers.RwkvForCausalLM(transformers.RwkvConfig(**options)).eval()


BUILDERS = {"mamba2": mamba2, "mamba": mamba, "rwkv": rwkv}


def load(result):
    with zipfile.ZipFile(result.path) as archive:
        return ScratchEmulator(json.loads(archive.read("sprite.json")))


def run(emulator, procedure, tokens=None):
    if tokens is not None:
        emulator.lists["cattorch tokens"] = list(tokens)
    emulator.run_procedure(procedure)
    return emulator.variables["cattorch status"]


def logits(emulator):
    return torch.tensor(emulator.lists["cattorch logits"], dtype=torch.float64)


def expected_rows(model, tokens):
    with torch.no_grad():
        return model(torch.tensor([tokens]), use_cache=False).logits[0].double()


KINDS = {
    "mamba2": ("conv", "ssm"),
    "mamba": ("conv", "ssm"),
    "rwkv": ("attention shift", "feed forward shift", "numerator", "denominator", "maximum"),
}


def state_names(model):
    """One state per layer and kind, so the largest kind stays unsharded longest."""
    return [
        f"{kind} {index}" for index in range(model.config.num_hidden_layers)
        for kind in KINDS[model.config.model_type]
    ]


def states(emulator, model):
    return {name: list(emulator.lists[f"_cattorch state {name}"]) for name in state_names(model)}


TOKENS = [3, 17, 8, 29, 1, 12, 25, 6, 14, 30, 2, 19, 9, 22]


@pytest.fixture(scope="module", params=sorted(BUILDERS))
def exported(request, tmp_path_factory):
    model = BUILDERS[request.param]()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    result = transpile(
        model, RecurrentGenerationProgram(top_k=3), tmp_path_factory.mktemp(request.param) / "lm",
    )
    return model, load(result), result, before


def test_prefill_and_decode_match_the_stateless_forward(exported):
    model, emulator, result, _ = exported
    expected = expected_rows(model, TOKENS)
    assert run(emulator, "cattorch init") == "ok"
    assert run(emulator, "cattorch prefill", TOKENS[:5]) == "ok"
    rows = [logits(emulator)]
    # Past several of Mamba-2's four-token chunks.
    for token in TOKENS[5:]:
        assert run(emulator, "cattorch decode", [token]) == "ok"
        rows.append(logits(emulator))
    torch.testing.assert_close(torch.stack(rows), expected[4:], atol=TOLERANCE, rtol=0)
    assert emulator.lists["cattorch top k ids"] == expected[-1].topk(3).indices.tolist()
    assert emulator.variables["cattorch cache length"] == len(TOKENS)
    assert emulator.variables["cattorch max context"] == "Infinity"
    assert result.output.numel == VOCAB
    assert result.procedures == (
        "cattorch init", "cattorch reset", "cattorch prefill", "cattorch decode",
    )


def test_reset_restores_the_initial_states(exported):
    model, emulator, _, _ = exported
    run(emulator, "cattorch reset")
    initial = states(emulator, model)
    run(emulator, "cattorch prefill", TOKENS[:4])
    assert states(emulator, model) != initial
    assert run(emulator, "cattorch reset") == "ok"
    assert states(emulator, model) == initial
    assert emulator.variables["cattorch cache length"] == 0
    assert run(emulator, "cattorch decode", [TOKENS[0]]) == "ok"
    torch.testing.assert_close(
        logits(emulator), expected_rows(model, TOKENS[:1])[0], atol=TOLERANCE, rtol=0,
    )


def test_the_users_model_is_not_changed(exported):
    model, _, _, before = exported
    after = model.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key], after[key]) for key in before)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_hidden_prefill_gives_the_same_logits(tmp_path, name):
    model = BUILDERS[name]()
    results = {}
    for hidden in (True, False):
        emulator = load(transpile(
            model, RecurrentGenerationProgram(hidden_prefill=hidden), tmp_path / f"{name}{hidden}",
        ))
        assert run(emulator, "cattorch prefill", TOKENS[:6]) == "ok"
        results[hidden] = (logits(emulator), states(emulator, model))
    assert torch.equal(results[True][0], results[False][0])
    assert results[True][1] == results[False][1]


def test_max_tokens_limits_the_tokens_between_resets(tmp_path):
    model = mamba2()
    emulator = load(transpile(model, RecurrentGenerationProgram(max_tokens=3), tmp_path / "m"))
    assert emulator.variables["cattorch max context"] == 3
    assert run(emulator, "cattorch prefill", TOKENS[:3]) == "ok"
    assert run(emulator, "cattorch decode", [TOKENS[3]]) == "maximum context exceeded"
    assert run(emulator, "cattorch prefill", TOKENS[:4]) == "prefill exceeds maximum context"


def test_rescaled_rwkv_layers_follow_transformers_without_changing_the_model(tmp_path):
    model = rwkv(rescale_every=1)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    emulator = load(transpile(model, RecurrentGenerationProgram(), tmp_path / "rescaled"))
    assert not model.rwkv.layers_are_rescaled
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    # transformers rescales the weights in place on the first evaluation call.
    expected = expected_rows(model, TOKENS[:6])
    assert model.rwkv.layers_are_rescaled
    run(emulator, "cattorch prefill", TOKENS[:6])
    torch.testing.assert_close(logits(emulator), expected[-1], atol=TOLERANCE, rtol=0)
    # A model whose weights are already rescaled exports the same way.
    emulator = load(transpile(model, RecurrentGenerationProgram(), tmp_path / "already"))
    run(emulator, "cattorch prefill", TOKENS[:6])
    torch.testing.assert_close(logits(emulator), expected[-1], atol=TOLERANCE, rtol=0)


class Checked(Exception):
    """Raised in place of lowering, once the export's checks have passed."""


@pytest.fixture
def checks_only(monkeypatch):
    """Run an export's step and reference checks, but not the lowering."""
    from cattorch import recurrent_generation

    def stop(*_args, **_kwargs):
        raise Checked

    monkeypatch.setattr(recurrent_generation, "transpile_program", stop)


@pytest.mark.parametrize(("name", "overrides"), [
    ("mamba2", {"n_groups": 1}),
    ("mamba2", {"n_groups": 4, "conv_kernel": 4}),
    ("mamba2", {"time_step_limit": (0.05, 0.5)}),
    ("mamba2", {"use_conv_bias": False, "use_bias": True}),
    ("mamba2", {"residual_in_fp32": False, "hidden_act": "gelu"}),
    ("mamba2", {"conv_kernel": 1}),
    ("mamba2", {"tie_word_embeddings": True}),
    ("mamba", {"time_step_rank": 1, "state_size": 3}),
    ("mamba", {"use_mambapy": True, "use_associative_scan": False}),
    ("mamba", {"use_conv_bias": False, "use_bias": True, "residual_in_fp32": False}),
    ("mamba", {"conv_kernel": 2, "hidden_act": "gelu"}),
    ("rwkv", {"attention_hidden_size": 12}),
    ("rwkv", {"intermediate_size": 20}),
    ("rwkv", {"num_hidden_layers": 3, "rescale_every": 1}),
    ("rwkv", {"tie_word_embeddings": True}),
])
def test_config_variants_pass_the_reference_check(tmp_path, checks_only, name, overrides):
    with pytest.raises(Checked):
        transpile(BUILDERS[name](**overrides), RecurrentGenerationProgram(), tmp_path / "variant")


@pytest.mark.parametrize(("overrides", "message"), [
    ({"n_groups": 3}, r"config n_groups=3 isn't supported: num_heads \(4\) must be a multiple"),
    ({"time_step_limit": (0.0, 1.0, 2.0)}, r"config time_step_limit=\(0.0, 1.0, 2.0\) isn't supported"),
])
def test_unsupported_config_variants_name_the_field(tmp_path, overrides, message):
    with pytest.raises(UnsupportedModelError, match=message):
        transpile(mamba2(**overrides), RecurrentGenerationProgram(), tmp_path / "unsupported")


def test_a_missing_attribute_is_reported_before_export(tmp_path):
    model = mamba()
    del model.backbone.layers[1].mixer.dt_proj
    with pytest.raises(UnsupportedModelError, match=r"backbone.layers.1.mixer has no dt_proj"):
        transpile(model, RecurrentGenerationProgram(), tmp_path / "missing")


def test_upstream_changes_fail_the_reference_check(tmp_path, monkeypatch):
    from transformers.models.mamba2 import modeling_mamba2

    original = modeling_mamba2.Mamba2Mixer.forward

    def changed(self, *args, **kwargs):
        return original(self, *args, **kwargs) + 0.01

    monkeypatch.setattr(modeling_mamba2.Mamba2Mixer, "forward", changed)
    with pytest.raises(UnsupportedModelError, match=(
        r"disagrees with reference at logits row 0.*installed transformers version "
        r"\(.+\) may be unsupported"
    )):
        transpile(mamba2(), RecurrentGenerationProgram(), tmp_path / "changed")


def test_int8_quantized_mamba2_stays_close(tmp_path):
    model = mamba2()
    result = transpile(
        model, RecurrentGenerationProgram(), tmp_path / "q8",
        quantization=QuantizationConfig(bits=8, min_quantized_values=16),
    )
    assert result.quantization is not None and result.quantization.tensors
    emulator = load(result)
    expected = expected_rows(model, TOKENS)
    run(emulator, "cattorch prefill", TOKENS[:4])
    rows = [logits(emulator)]
    for token in TOKENS[4:]:
        run(emulator, "cattorch decode", [token])
        rows.append(logits(emulator))
    error = float((torch.stack(rows) - expected[3:]).abs().max())
    assert error < 0.04 * float(expected.abs().max())


def test_generation_program_points_to_the_recurrent_program(tmp_path):
    for builder in BUILDERS.values():
        with pytest.raises(UnsupportedModelError, match="RecurrentGenerationProgram"):
            transpile(builder(), GenerationProgram(max_context=8), tmp_path / "kv")


def test_states_are_needed_for_other_models(tmp_path):
    with pytest.raises(UnsupportedModelError, match=(
        r"RecurrentGenerationProgram needs states=\.\.\. unless the model is a supported "
        r"Hugging Face recurrent model \(Mamba, Mamba-2, RWKV\)"
    )):
        transpile(torch.nn.Linear(2, 2), RecurrentGenerationProgram(), tmp_path / "linear")
    torch.manual_seed(0)
    base = transformers.Mamba2Model(mamba2().config).eval()
    with pytest.raises(UnsupportedModelError, match="without a language-model head"):
        transpile(base, RecurrentGenerationProgram(), tmp_path / "base")


def test_explicit_states_bypass_the_adapter(tmp_path):
    program = RecurrentGenerationProgram((State("h", torch.zeros(2)),))
    with pytest.raises(ValueError, match="model has no callable recurrent step 'step'"):
        transpile(mamba(), program, tmp_path / "explicit")


def test_the_generation_benchmark_rejects_recurrent_models(tmp_path):
    from cattorch.benchmark import build_generation_benchmark_suite

    tokens = torch.tensor([TOKENS[:4]])
    with pytest.raises(ValueError, match="'ssm' is a Mamba model"):
        build_generation_benchmark_suite([("ssm", mamba(), tokens, tokens)], tmp_path / "b")
