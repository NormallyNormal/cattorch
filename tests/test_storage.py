import json
import math
import zipfile
from collections import Counter

import pytest
import torch

from cattorch import (
    CodegenConfig,
    StorageConfig,
    build_storage_benchmark_suite,
    transpile,
    verify,
)
from cattorch.sprite import _static_storage_shard_limits
from cattorch.storage import (
    BASE92_ALPHABET,
    EncodedList,
    _canonical_huffman_codes,
    _encode_bytes,
    _encode_huffman_symbols,
    _huffman_prefix_table,
    _length_limited_huffman_lengths,
    build_shared_unpack_program,
    build_unpack_program,
    encode_quantized_values,
    quantize_codebook_values,
    quantize_values,
    encode_values,
    rounded_values,
)
from cattorch.util.scratch.dsl import Program, append, delete, item, length, replace
from cattorch.util.scratch.emulator import ScratchEmulator
from cattorch.util.scratch.finalize_scratch import (
    SCRATCH_ONLINE_JSON_LIMIT,
    SCRATCH_ONLINE_JSON_WARN_SIZE,
    online_json_size_warning,
)
from cattorch.util.scratch.sharding import SCRATCH_LIST_LIMIT, shard_sprite_lists


class TinyLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, value):
        return self.linear(value)


def _load(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("sprite.json"))


def _codec_sprite(program):
    sprite = program.compile()
    sprite["currentCostume"] = 0
    sprite["costumes"] = [{"name": character} for character in BASE92_ALPHABET]
    return sprite


def test_base92_uses_all_safe_costume_digits_and_exact_52_bit_groups():
    assert len(BASE92_ALPHABET) == 92
    assert '"' not in BASE92_ALPHABET
    assert "\\" not in BASE92_ALPHABET
    assert all(33 <= ord(character) <= 126 for character in BASE92_ALPHABET)
    assert len(_encode_bytes(bytes(range(13)))) == 17  # one pad header + 16 digits
    assert len(_encode_bytes(bytes(range(26)))) == 33


def test_length_limited_huffman_round_trips_q4_symbols():
    symbols = [7] * 500 + [6] * 250 + [8] * 125 + list(range(15)) * 3
    frequencies = Counter(symbols)
    lengths = _length_limited_huffman_lengths(frequencies)
    codes = _canonical_huffman_codes(lengths)
    table = _huffman_prefix_table(codes)
    payload = _encode_huffman_symbols(symbols, codes)

    assert max(lengths.values()) <= 8
    assert len(payload) * 8 < len(symbols) * 4
    decoded = []
    bits = 0
    bit_count = 0
    cursor = 0
    while len(decoded) < len(symbols):
        if bit_count < 8:
            bits = bits * 256 + payload[cursor]
            bit_count += 8
            cursor += 1
        entry = table[bits // 2 ** (bit_count - 8)]
        length_ = entry % 16
        decoded.append(entry // 16)
        bit_count -= length_
        bits %= 2**bit_count
    assert decoded == symbols


def test_length_limiter_handles_an_extreme_fifteen_symbol_distribution():
    frequencies = Counter({0: 10**9, **{symbol: 1 for symbol in range(1, 15)}})
    lengths = _length_limited_huffman_lengths(frequencies)
    assert set(lengths) == set(range(15))
    assert max(lengths.values()) <= 8
    assert sum(2 ** (8 - length_) for length_ in lengths.values()) <= 256


@pytest.mark.parametrize(
    "precision,values",
    [
        ("float32", [0.0, -0.0, 1.0, -2.5, 1e-30, 12345.25]),
        ("float16", [0.0, -0.0, 1.0, -2.5, 0.001, 123.5, 70_000, math.inf, -math.inf, math.nan]),
    ],
)
def test_scratch_base92_decoder_round_trips_ieee_values(precision, values):
    payload = encode_values(values, precision)
    program = build_unpack_program([
        EncodedList("weights", "payload", payload, precision),
    ])
    emulator = ScratchEmulator(_codec_sprite(program))
    emulator.run()
    expected = rounded_values(values, precision)
    actual = emulator.lists["weights"]
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert (math.isnan(got) and math.isnan(want)) or got == want


@pytest.mark.parametrize("precision", ["int8", "int4"])
def test_scratch_integer_decoder_dequantizes_groups_once(precision):
    values = [
        -3.0, -1.25, 0.0, 0.75, 2.5,
        -0.02, 0.01, 0.03, 0.07, -0.05,
        100.0,
    ]
    group_size = 5
    payload, scale_payload = encode_quantized_values(values, precision, group_size)
    program = build_unpack_program([
        EncodedList(
            "weights", "payload", payload, precision,
            value_count=len(values), group_size=group_size,
            scale_list_name="scales", scale_payload_name="scale payload",
            scale_payload=scale_payload,
        ),
    ])
    emulator = ScratchEmulator(_codec_sprite(program))
    emulator.run()
    assert emulator.lists["weights"] == rounded_values(values, precision, group_size)
    assert emulator.lists["scales"] == []


@pytest.mark.parametrize("count", [1, 2, 3])
def test_costume_base92_decoder_handles_partial_byte_groups(count):
    values = [-0.75, 0.25, 1.5][:count]
    payload, scale_payload = encode_quantized_values(values, "int8", count)
    program = build_unpack_program([
        EncodedList(
            "weights", "payload", payload, "int8",
            value_count=count, group_size=count,
            scale_list_name="scales", scale_payload_name="scale payload",
            scale_payload=scale_payload,
        ),
    ])
    emulator = ScratchEmulator(_codec_sprite(program))
    emulator.run()
    assert emulator.lists["weights"] == rounded_values(values, "int8", count)


def test_packed_int4_payload_has_half_the_int8_base92_data_groups():
    values = [math.sin(index) for index in range(96)]
    int8_payload, _ = encode_quantized_values(values, "int8", 32)
    int4_payload, _ = encode_quantized_values(values, "int4", 32)
    assert len(int4_payload) - 1 == (len(int8_payload) - 1) // 2


def test_bitpacked_int6_uses_base92_groups_after_one_padding_digit():
    values = [math.sin(index) for index in range(96)]
    payload, _ = encode_quantized_values(values, "int6", 32)
    assert len(payload) == 1 + math.ceil((96 * 6 / 8) / 13) * 16


def test_learned_codebook_quantizer_is_deterministic_and_bounded():
    values = [math.sin(index / 7) * (1 + index % 5) for index in range(513)]
    first = quantize_codebook_values(values)
    second = quantize_codebook_values(values)
    assert first == second
    codes, centers = first
    assert len(codes) == len(values)
    assert len(centers) == 64
    assert min(codes) >= 0 and max(codes) < 64
    assert centers == sorted(centers)


def test_default_export_unpacks_weights_once_and_prepare_save_rearms_it(tmp_path):
    torch.manual_seed(40)
    model = TinyLinear()
    value = torch.randn(2, 3)
    path = tmp_path / "compressed"
    transpile(model, value, str(path))
    sprite = _load(path.with_suffix(".sprite3"))

    weight_lists = [entry for entry in sprite["lists"].values() if entry[0].startswith("W_")]
    assert weight_lists and all(not entry[1] for entry in weight_lists)
    weight_banks = [
        entry for entry in sprite["variables"].values()
        if "weight bank payload" in entry[0]
    ]
    assert len(weight_banks) == 1
    assert [costume["name"] for costume in sprite["costumes"]] == list(BASE92_ALPHABET)
    assert len({costume["md5ext"] for costume in sprite["costumes"]}) == 1

    emulator = ScratchEmulator(sprite)
    emulator.lists["input"] = value.flatten().tolist()
    emulator.run()
    torch.testing.assert_close(
        torch.tensor(emulator.lists["output"]),
        model(value).flatten(),
        atol=1e-5,
        rtol=0,
    )

    prepare_root = emulator._procedures["cattorch prepare for save"]
    emulator._exec_chain(prepare_root)
    assert emulator.variables["cattorch initialized"] == 0
    assert all(not emulator.lists[entry[0]] for entry in weight_lists)

    emulator.lists["input"] = value.flatten().tolist()
    emulator.run()
    torch.testing.assert_close(
        torch.tensor(emulator.lists["output"]),
        model(value).flatten(),
        atol=1e-5,
        rtol=0,
    )


def test_shared_unpack_splits_temporary_byte_banks_at_scratch_list_limit():
    first = bytes(120_000)
    second = bytes(120_000)
    specs = [
        EncodedList("first", "first payload", "", "float32", payload_bytes=first),
        EncodedList("second", "second payload", "", "float32", payload_bytes=second),
    ]

    program = build_shared_unpack_program(specs)
    banks = {
        name: payload
        for name, payload in program.variable_values.items()
        if "weight bank" in name
    }

    assert set(banks) == {
        "cattorch float32 weight bank 1 payload",
        "cattorch float32 weight bank 2 payload",
    }
    expected = 1 + math.ceil(120_000 / 13) * 16
    assert all(len(payload) == expected for payload in banks.values())


def test_shared_unpack_splits_quantized_banks_on_combined_scale_bytes():
    specs = [
        EncodedList(
            f"weights {index}", f"payload {index}", "", "int8",
            value_count=30_000, group_size=1,
            scale_precision="float32", payload_bytes=bytes(30_000),
            scale_values=(1.0,) * 30_000,
        )
        for index in range(2)
    ]
    program = build_shared_unpack_program(specs)

    scale_banks = {
        name: payload
        for name, payload in program.variable_values.items()
        if "scale bank" in name
    }
    weight_banks = {
        name: payload
        for name, payload in program.variable_values.items()
        if "weight bank" in name
    }
    assert len(scale_banks) == 2
    assert len(weight_banks) == 2
    expected = 1 + math.ceil(120_000 / 13) * 16
    assert all(len(payload) == expected for payload in scale_banks.values())


def test_split_scale_banks_decode_every_tensor_value(tmp_path):
    class TwoMatrices(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(200, 150, bias=False)
            self.second = torch.nn.Linear(200, 150, bias=False)
            self.first.weight.data.fill_(1)
            self.second.weight.data.fill_(2)

        def forward(self, value):
            return self.first(value) + self.second(value)

    model = TwoMatrices().eval()
    with pytest.warns(DeprecationWarning, match="QuantizationConfig"):
        result = transpile(
            model,
            torch.ones(1, 200),
            tmp_path / "split-scales",
            storage=StorageConfig(
                precision="int8", group_size=1, min_quantized_values=1,
            ),
            codegen=CodegenConfig(unrolling="compact", id_namespace=""),
        )
    emulator = ScratchEmulator(_load(result.path))
    emulator.run_procedure("cattorch init")

    decoded = [
        emulator._lists[identifiers[0]]
        for name, identifiers in sorted(emulator._list_name_to_ids.items())
        if name.startswith("W_")
    ]
    assert [len(values) for values in decoded] == [30_000, 30_000]
    assert all(all(value != 0 for value in values) for values in decoded)


def test_shared_unpack_rejects_one_oversized_scale_stream():
    spec = EncodedList(
        "weights", "payload", "", "int8",
        value_count=50_001, group_size=1,
        scale_precision="float32", payload_bytes=bytes(50_001),
        scale_values=(1.0,) * 50_001,
    )
    with pytest.raises(ValueError, match="temporary scale bytes"):
        build_shared_unpack_program([spec])


def test_compressed_static_shard_limits_cover_values_and_scales():
    tensors = {
        "f32": torch.ones(50_001),
        "f16": torch.ones(100_001),
        "int": torch.ones(200_000, 1),
    }
    assert _static_storage_shard_limits(tensors, StorageConfig()) == {
        "W_f32": 50_000, "W_f16": 50_000, "W_int": 50_000,
    }
    assert _static_storage_shard_limits(
        tensors,
        StorageConfig(precision="float16"),
    ) == {"W_f32": 100_000, "W_f16": 100_000, "W_int": 100_000}
    assert _static_storage_shard_limits(
        {"int": tensors["int"]},
        StorageConfig(
            precision="int8", group_size=1, min_quantized_values=1,
        ),
    ) == {"W_int": 50_000}
    assert _static_storage_shard_limits(
        {"int": tensors["int"]},
        StorageConfig(
            precision="int8", group_size=1, min_quantized_values=1,
            scale_precision="float16",
        ),
    ) == {"W_int": 100_000}


def test_default_float32_export_shards_for_temporary_decode_limit(tmp_path):
    torch.manual_seed(81)
    model = torch.nn.Linear(256, 256, bias=False).eval()
    result = transpile(
        model,
        torch.randn(1, 256),
        tmp_path / "large-f32",
        codegen=CodegenConfig(unrolling="compact", id_namespace=""),
    )
    sprite = _load(result.path)
    emulator = ScratchEmulator(sprite)
    emulator.run_procedure("cattorch init")

    weight_lengths = [
        len(emulator.lists[name])
        for name in emulator._list_name_to_ids
        if name.startswith("W_")
    ]
    assert weight_lengths == [50_000, 15_536]
    assert sum(weight_lengths) == model.weight.numel()


def test_quantization_scale_rounding_handles_underflow_and_rejects_overflow():
    for precision in ("int8", "int6", "int4"):
        codes, scales = quantize_values(
            [1e-7], precision, group_size=1, scale_precision="float16",
        )
        assert codes and math.isfinite(scales[0]) and scales[0] > 0
        with pytest.raises(ValueError, match="finite float16 range"):
            quantize_values(
                [1e8], precision, group_size=1, scale_precision="float16",
            )


def test_emulator_enforces_vanilla_scratch_list_item_limit(monkeypatch):
    import cattorch.util.scratch.emulator as emulator_module

    monkeypatch.setattr(emulator_module, "SCRATCH_LIST_LIMIT", 2)
    program = Program(
        "list cap",
        lists=("values",),
        body=(append("values", 1), append("values", 2), append("values", 3)),
    )
    emulator = ScratchEmulator(program.compile())
    emulator.run()

    assert emulator.lists["values"] == [1, 2]


def test_float16_storage_is_explicitly_lossy_in_exact_mode(tmp_path):
    torch.manual_seed(41)
    model = TinyLinear()
    value = torch.randn(2, 3)
    path = tmp_path / "float16"
    transpile(
        model,
        value,
        str(path),
        optimization="exact",
        storage=StorageConfig(precision="float16"),
    )
    sprite = _load(path.with_suffix(".sprite3"))
    emulator = ScratchEmulator(sprite)
    emulator.lists["input"] = value.flatten().tolist()
    emulator.run()
    torch.testing.assert_close(
        torch.tensor(emulator.lists["output"]),
        model(value).flatten(),
        atol=1e-3,
        rtol=1e-3,
    )
    assert emulator.variables["cattorch storage precision"] == "float16"


@pytest.mark.parametrize("precision", ["int8", "int6", "int4"])
def test_integer_storage_dequantizes_before_forward_and_reinitializes(
    tmp_path, precision,
):
    model = TinyLinear().eval()
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([
            [-2.0, -0.25, 1.0],
            [0.125, 0.5, 3.0],
        ]))
        model.linear.bias.copy_(torch.tensor([0.2, -0.4]))
    value = torch.tensor([[0.5, -1.0, 2.0], [-0.25, 0.75, 1.5]])
    path = tmp_path / precision
    group_size = 4
    with pytest.warns(DeprecationWarning, match="QuantizationConfig"):
        transpile(
            model,
            value,
            str(path),
            storage=StorageConfig(
                precision=precision,
                group_size=group_size,
                min_quantized_values=1,
            ),
        )
    sprite = _load(path.with_suffix(".sprite3"))
    emulator = ScratchEmulator(sprite)

    quantized_weight = torch.tensor(
        rounded_values(
            model.linear.weight.detach().flatten(), precision, group_size,
        ),
        dtype=model.linear.weight.dtype,
    ).reshape_as(model.linear.weight)
    rounded_bias = torch.tensor(
        rounded_values(model.linear.bias.detach(), "float16"),
        dtype=model.linear.bias.dtype,
    )
    expected = torch.nn.functional.linear(value, quantized_weight, rounded_bias)

    for _ in range(2):
        emulator.lists["input"] = value.flatten().tolist()
        emulator.run()
        torch.testing.assert_close(
            torch.tensor(emulator.lists["output"]),
            expected.flatten(),
            atol=1e-5,
            rtol=0,
        )
        emulator._exec_chain(emulator._procedures["cattorch prepare for save"])
    assert emulator.variables["cattorch storage precision"] == precision
    assert all(
        not emulator.lists[name]
        for name in emulator._list_name_to_ids
        if name.startswith("cattorch quant scales")
    )


def test_int8_has_lower_export_error_than_int4_on_same_weights(tmp_path):
    model = TinyLinear().eval()
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([
            [-1.73, -0.19, 0.84],
            [0.11, 0.57, 2.62],
        ]))
        model.linear.bias.copy_(torch.tensor([0.17, -0.31]))
    value = torch.tensor([
        [0.51, -1.07, 1.93],
        [-0.28, 0.79, 1.41],
        [1.17, 0.33, -0.62],
    ])
    reports = {}
    for precision in ("int8", "int4"):
        with pytest.warns(DeprecationWarning, match="QuantizationConfig"):
            result = transpile(
                model,
                value,
                tmp_path / precision,
                storage=StorageConfig(
                    precision=precision,
                    group_size=4,
                    min_quantized_values=1,
                ),
            )
        reports[precision] = verify(model, value, result, atol=0, rtol=0)

    assert reports["int8"].mean_abs_error > 0
    assert reports["int8"].mean_abs_error < reports["int4"].mean_abs_error


def test_uncompressed_prepare_save_keeps_static_weights(tmp_path):
    model = TinyLinear()
    value = torch.randn(1, 3)
    path = tmp_path / "plain"
    transpile(
        model, value, str(path),
        storage=StorageConfig(compression=False),
    )
    emulator = ScratchEmulator(_load(path.with_suffix(".sprite3")))
    before = {
        name: list(emulator.lists[name])
        for name in emulator._list_name_to_ids
        if name.startswith("W_")
    }
    emulator._exec_chain(emulator._procedures["cattorch prepare for save"])
    assert before
    assert all(emulator.lists[name] == values for name, values in before.items())


def test_nonfloating_static_tensor_cannot_silently_round_through_float32(tmp_path):
    class IntegerBuffer(torch.nn.Module):
        def __init__(self, number):
            super().__init__()
            self.register_buffer("offset", torch.tensor([number], dtype=torch.int64))

        def forward(self, value):
            return value + self.offset

    model = IntegerBuffer(16_777_217).eval()
    value = torch.tensor([0], dtype=torch.int64)
    with pytest.raises(ValueError, match="not exactly representable"):
        transpile(model, value, tmp_path / "compressed-int")

    path = tmp_path / "plain-int"
    transpile(
        model, value, path,
        storage=StorageConfig(compression=False),
    )
    emulator = ScratchEmulator(_load(path.with_suffix(".sprite3")))
    emulator.lists["input"] = [0]
    emulator.run_procedure("cattorch forward")
    assert emulator.lists["output"] == [16_777_217]

    with pytest.raises(ValueError, match="not exactly representable"):
        transpile(
            IntegerBuffer(2**53 + 1), value, tmp_path / "outside-double",
            storage=StorageConfig(compression=False),
        )


@pytest.mark.parametrize("precision", ["int8", "int4"])
def test_uncompressed_integer_storage_still_applies_quantization(tmp_path, precision):
    model = TinyLinear().eval()
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([
            [-1.73, -0.19, 0.84],
            [0.11, 0.57, 2.62],
        ]))
    value = torch.tensor([[0.25, -0.5, 1.25]])
    with pytest.warns(DeprecationWarning, match="QuantizationConfig"):
        result = transpile(
            model,
            value,
            tmp_path / f"plain-{precision}",
            storage=StorageConfig(
                compression=False,
                precision=precision,
                group_size=4,
                min_quantized_values=1,
            ),
        )
    report = verify(model, value, result, atol=0, rtol=0)
    assert report.values_compared == 2
    assert report.mean_abs_error > 0


def test_oversized_list_routes_reads_writes_and_appends_to_public_shards():
    initial = list(range(SCRATCH_LIST_LIMIT + 2))
    program = Program(
        "sharding",
        lists=("input", "output"),
        list_values={"input": initial},
        body=(
            append("output", item("input", SCRATCH_LIST_LIMIT + 1)),
            replace("input", SCRATCH_LIST_LIMIT + 2, 77),
            append("input", 88),
            delete("input", length("input")),
        ),
    )
    sprite = program.compile()
    layouts = shard_sprite_lists(
        sprite,
        {"input": SCRATCH_LIST_LIMIT + 3},
    )
    emulator = ScratchEmulator(sprite)
    emulator.run()

    second = layouts["input"].shards[1]
    assert emulator.lists["output"] == [SCRATCH_LIST_LIMIT]
    assert emulator._lists[second.identifier] == [SCRATCH_LIST_LIMIT, 77]
    assert second.name == "input shard 2"
    assert sprite["variables"]["cattorch_input_shard_count"][1] == 2
    assert sprite["variables"]["cattorch_input_logical_length"][1] == SCRATCH_LIST_LIMIT + 3


def test_storage_config_validation():
    with pytest.raises(ValueError, match="precision"):
        StorageConfig(precision="int7")
    with pytest.raises(TypeError, match="compression"):
        StorageConfig(compression=1)
    with pytest.raises(ValueError, match="group_size"):
        StorageConfig(group_size=0)
    with pytest.raises(ValueError, match="group_size"):
        StorageConfig(group_size=True)
    with pytest.raises(ValueError, match="min_quantized_values"):
        StorageConfig(min_quantized_values=0)
    with pytest.raises(ValueError, match="min_quantized_values"):
        StorageConfig(min_quantized_values=True)


def test_size_guidance_applies_to_expanded_online_json_not_archives():
    assert online_json_size_warning(1_000, "Project JSON") is None
    approaching = online_json_size_warning(
        SCRATCH_ONLINE_JSON_WARN_SIZE + 1, "Project JSON",
    )
    exceeded = online_json_size_warning(
        SCRATCH_ONLINE_JSON_LIMIT + 1, "Project JSON",
    )

    assert approaching is not None and "approaching" in approaching
    assert exceeded is not None and "ordinary online save/upload" in exceeded
    assert "archive-based path may still work" in exceeded


@pytest.mark.parametrize("precision", ["int8", "int6", "int4"])
def test_small_matrices_use_float16_instead_of_integer_decoder(tmp_path, precision):
    model = TinyLinear().eval()
    value = torch.randn(1, 3)
    path = tmp_path / f"hybrid-{precision}"
    with pytest.warns(DeprecationWarning, match="QuantizationConfig"):
        transpile(
            model,
            value,
            path,
            storage=StorageConfig(precision=precision),
        )
    sprite = _load(path.with_suffix(".sprite3"))

    assert not any(
        "scale bank payload" in entry[0]
        for entry in sprite["variables"].values()
    )
    emulator = ScratchEmulator(sprite)
    emulator.lists["input"] = value.flatten().tolist()
    emulator.run()
    expected = torch.nn.functional.linear(
        value,
        torch.tensor(
            rounded_values(model.linear.weight.detach().flatten(), "float16")
        ).reshape_as(model.linear.weight),
        torch.tensor(
            rounded_values(model.linear.bias.detach().flatten(), "float16")
        ),
    )
    torch.testing.assert_close(
        torch.tensor(emulator.lists["output"]),
        expected.flatten(),
        atol=1e-5,
        rtol=0,
    )


@pytest.mark.parametrize("precision", ["int8", "int6", "int4"])
def test_large_matrices_use_requested_integer_storage(tmp_path, precision):
    model = torch.nn.Linear(128, 128, bias=False).eval()
    value = torch.randn(1, 128)
    path = tmp_path / f"large-{precision}"
    with pytest.warns(DeprecationWarning, match="QuantizationConfig"):
        transpile(model, value, path, storage=StorageConfig(precision=precision))
    sprite = _load(path.with_suffix(".sprite3"))

    assert any(
        "scale bank payload" in entry[0]
        for entry in sprite["variables"].values()
    )
    if precision == "int4":
        weight_payloads = [
            entry[1] for entry in sprite["variables"].values()
            if "int4 weight bank" in entry[0]
        ]
        assert weight_payloads
        assert len(weight_payloads[0]) < len(_encode_bytes(bytes(128 * 128 // 2)))
        assert any(
            "int4 huffman bank" in entry[0]
            for entry in sprite["variables"].values()
        )
        assert next(
            entry[1] for entry in sprite["variables"].values()
            if entry[0] == "cattorch storage compression"
        ) == "costume-base92-huffman-int4"


def test_storage_benchmark_bundles_all_variants(tmp_path):
    project_path = build_storage_benchmark_suite(
        [("tiny", TinyLinear(), torch.randn(1, 3))],
        tmp_path / "storage_suite",
        iterations=2,
    )
    with zipfile.ZipFile(project_path) as archive:
        project = json.loads(archive.read("project.json"))
    assert {target["name"] for target in project["targets"]} == {
        "Stage", "tiny plain-f32", "tiny base92-f32", "tiny base92-f16",
        "tiny base92-int8", "tiny base92-int6", "tiny base92-int4-huffman",
    }
    stage = next(target for target in project["targets"] if target["isStage"])
    lists = {entry[0]: entry[1] for entry in stage["lists"].values()}
    assert len(lists["cattorch storage sizes"]) == 6
    assert any(
        block["opcode"] == "event_whenflagclicked"
        for block in stage["blocks"].values()
    )


def test_int4_huffman_banks_split_when_compressed_stream_exceeds_list_limit():
    import random

    from cattorch.storage import (
        BASE92_ALPHABET, EncodedList, _build_banked_unpack_program,
    )
    from cattorch.util.scratch.sharding import SCRATCH_LIST_LIMIT

    def spec(index, symbols):
        packed = bytes(
            symbols[i] * 16 + (symbols[i + 1] if i + 1 < len(symbols) else 7)
            for i in range(0, len(symbols), 2)
        )
        return EncodedList(
            f"W_{index}", f"payload {index}", "", "int4",
            value_count=len(symbols), group_size=len(symbols),
            scale_precision="float16", payload_bytes=packed, scale_values=(1.0,),
        )

    # Two full-size shards of 13 common codes plus many two-value tensors holding
    # a globally rare code. Each tiny tensor needs 9 Huffman bits, so byte
    # alignment makes the stream longer than the 4-bit payload it replaces.
    common = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
    tiny = 10_000
    symbols = [common[i % 13] for i in range(2 * (SCRATCH_LIST_LIMIT - tiny))]
    random.Random(0).shuffle(symbols)
    half = len(symbols) // 2
    specs = [spec(-1, symbols[:half]), spec(0, symbols[half:])] + [
        spec(i + 1, [(7, 14)[i % 2], common[i % 13]]) for i in range(tiny)
    ]
    assert sum(len(item.payload_bytes) for item in specs) == SCRATCH_LIST_LIMIT

    program = _build_banked_unpack_program(specs)
    banks = [
        value for name, value in program.variable_values.items()
        if name.startswith("cattorch int4 weight bank")
    ]
    assert banks
    for payload in banks:
        decoded = 13 * ((len(payload) - 1) // 16) - BASE92_ALPHABET.index(payload[0])
        assert decoded <= SCRATCH_LIST_LIMIT
