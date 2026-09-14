import json
import math
import zipfile

import pytest
import torch

from cattorch import CodegenConfig, QuantizationConfig, StorageConfig, transpile
from cattorch.util.scratch.emulator import ScratchEmulator
from cattorch.util.scratch.finalize_scratch import finalize_sprite
from cattorch.util.scratch.ids import (
    ID_ALPHABET,
    SCRATCH_TOOLBOX_IDS,
    compact_sprite_ids,
)
from cattorch.util.scratch.remap import remap_ids
from cattorch.sprite import _build_compact_top_k_program, _build_top_k_program


def _load(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("sprite.json"))


def test_codegen_config_validation():
    with pytest.raises(ValueError, match="target_json_bytes"):
        CodegenConfig(target_json_bytes=0)
    with pytest.raises(ValueError, match="unrolling"):
        CodegenConfig(unrolling="maximum")
    with pytest.raises(ValueError, match="Base64url"):
        CodegenConfig(id_namespace="bad namespace")
    with pytest.raises(TypeError, match="id_namespace"):
        CodegenConfig(id_namespace=3)
    with pytest.raises(ValueError, match="exactly three"):
        CodegenConfig(id_namespace="x_")


def test_compact_ids_use_configurable_base64url_namespace():
    sprite = {
        "blocks": {"long block": {"next": None}},
        "variables": {"long variable": ["value", 0]},
        "lists": {"long list": ["items", []]},
        "broadcasts": {},
        "comments": {},
    }
    compact = compact_sprite_ids(sprite, namespace="x_Z")
    identifiers = [
        identifier
        for section in ("blocks", "variables", "lists")
        for identifier in compact[section]
    ]
    assert identifiers == ["ctx_ZA", "ctx_ZB", "ctx_ZC"]
    assert all(set(identifier) <= set(ID_ALPHABET) for identifier in identifiers)

    no_namespace = compact_sprite_ids(sprite, namespace="")
    assert list(no_namespace["blocks"]) == ["ctA"]
    assert list(no_namespace["variables"]) == ["ctB"]


def test_compact_id_namespaces_cannot_have_ambiguous_lengths():
    sprite = {
        "blocks": {
            f"block {index}": {"next": None}
            for index in range(65)
        },
        "variables": {}, "lists": {}, "broadcasts": {}, "comments": {},
    }
    with pytest.raises(ValueError, match="exactly three"):
        compact_sprite_ids(sprite, namespace="A")
    with pytest.raises(ValueError, match="exactly three"):
        compact_sprite_ids(sprite, namespace="AB")

    first = compact_sprite_ids(sprite, namespace="AAA")
    second = compact_sprite_ids(sprite, namespace="AAB")
    assert set(first["blocks"]).isdisjoint(second["blocks"])


def test_compact_ids_never_collide_with_fixed_scratch_toolbox_ids():
    sprite = {
        "blocks": {
            f"long block {index}": {"next": None}
            for index in range(3_000)
        },
        "variables": {},
        "lists": {},
        "broadcasts": {},
        "comments": {},
    }
    compact = compact_sprite_ids(sprite, namespace="")
    assert not (set(compact["blocks"]) & SCRATCH_TOOLBOX_IDS)
    assert "of" not in compact["blocks"]


def test_id_remap_preserves_literal_strings_and_list_payloads():
    sprite = {
        "blocks": {
            "a": {
                "opcode": "data_setvariableto",
                "next": "b",
                "parent": None,
                "inputs": {
                    "VALUE": [1, [10, "a"]],
                    "REPORTER": [3, [12, "shown", "v"], [10, "b"]],
                },
                "fields": {"VARIABLE": ["shown", "v"]},
            },
            "b": {
                "opcode": "operator_join",
                "next": None,
                "parent": "a",
                "inputs": {},
                "fields": {},
            },
        },
        "variables": {"v": ["shown", "a"]},
        "lists": {"l": ["characters", ["a", "b", "v", "l"]]},
        "broadcasts": {},
        "comments": {},
    }
    remapped = remap_ids(
        sprite,
        {"a": "ctA", "b": "ctB", "v": "ctV", "l": "ctL"},
    )
    assert set(remapped["blocks"]) == {"ctA", "ctB"}
    assert remapped["blocks"]["ctA"]["next"] == "ctB"
    assert remapped["blocks"]["ctB"]["parent"] == "ctA"
    assert remapped["blocks"]["ctA"]["inputs"]["VALUE"] == [1, [10, "a"]]
    assert remapped["blocks"]["ctA"]["inputs"]["REPORTER"][1][2] == "ctV"
    assert remapped["blocks"]["ctA"]["inputs"]["REPORTER"][2] == [10, "b"]
    assert remapped["variables"]["ctV"] == ["shown", "a"]
    assert remapped["lists"]["ctL"] == ["characters", ["a", "b", "v", "l"]]


def test_codegen_config_validates_new_compaction_controls():
    with pytest.raises(TypeError, match="compact_internal_names"):
        CodegenConfig(compact_internal_names=1)
    with pytest.raises(TypeError, match="compact_schema"):
        CodegenConfig(compact_schema=1)
    with pytest.raises(ValueError, match="layer_sharing"):
        CodegenConfig(layer_sharing="force")


def test_finalizer_repairs_costume_menu_shadow_for_editor(tmp_path):
    sprite = {
        "blocks": {
            "switch": {
                "opcode": "looks_switchcostumeto",
                "next": None,
                "parent": None,
                "inputs": {"COSTUME": [3, "letter", [4, 0]]},
                "fields": {},
                "shadow": False,
                "topLevel": True,
                "x": 0,
                "y": 0,
            },
            "letter": {
                "opcode": "operator_letter_of",
                "next": None,
                "parent": "switch",
                "inputs": {
                    "LETTER": [1, [4, 1]],
                    "STRING": [1, [10, "A"]],
                },
                "fields": {},
                "shadow": False,
                "topLevel": False,
            },
        },
        "variables": {},
        "lists": {},
        "broadcasts": {},
        "comments": {},
    }
    path = tmp_path / "costume-shadow.sprite3"
    finalize_sprite(
        sprite,
        path,
        codegen=CodegenConfig(id_namespace="", compact_schema=True),
    )
    finalized = _load(path)
    switch = next(
        block for block in finalized["blocks"].values()
        if block["opcode"] == "looks_switchcostumeto"
    )
    shadow = finalized["blocks"][switch["inputs"]["COSTUME"][2]]
    assert shadow["opcode"] == "looks_costume"
    assert shadow["fields"]["COSTUME"] == ["!", None]
    assert shadow["shadow"] is True


def test_compact_top_k_matches_unrolled_with_fewer_blocks():
    logits = [math.sin(index * 0.37) for index in range(128)]
    programs = {
        "unrolled": _build_top_k_program(8),
        "compact": _build_compact_top_k_program(8),
    }
    outputs = {}
    block_counts = {}
    for name, program in programs.items():
        sprite = program.compile()
        emulator = ScratchEmulator(sprite)
        emulator.lists["output"] = logits
        emulator.run()
        outputs[name] = (
            emulator.lists["cattorch top k values"],
            emulator.lists["cattorch top k ids"],
        )
        block_counts[name] = len(sprite["blocks"])
    assert outputs["compact"] == outputs["unrolled"]
    assert block_counts["compact"] < block_counts["unrolled"]


def test_top_k_accepts_logits_below_old_finite_sentinel():
    for builder in (_build_top_k_program, _build_compact_top_k_program):
        sprite = builder(2).compile()
        emulator = ScratchEmulator(sprite)
        emulator.lists["output"] = [-1e31, -2e31, -3e31]
        emulator.run()
        assert [int(value) for value in emulator.lists["cattorch top k ids"]] == [0, 1]
        assert [float(value) for value in emulator.lists["cattorch top k values"]] == [
            -1e31, -2e31,
        ]


def test_internal_name_compaction_preserves_public_interface(tmp_path):
    model = torch.nn.Linear(4, 3).eval()
    path = tmp_path / "compact_names"
    transpile(
        model,
        torch.ones(1, 4),
        path,
        codegen=CodegenConfig(
            id_namespace="", compact_internal_names=True,
            compact_schema=True,
        ),
    )
    sprite = _load(path.with_suffix(".sprite3"))
    data_names = {
        entry[0]
        for section in ("variables", "lists")
        for entry in sprite[section].values()
    }
    assert {"input", "output"} <= data_names
    assert any(name.startswith("l") for name in data_names)
    assert all(
        block.get("shadow") is True or "shadow" not in block
        for block in sprite["blocks"].values()
        if block["opcode"] != "procedures_prototype"
    )
    emulator = ScratchEmulator(sprite)
    emulator.lists["input"] = [1, 2, 3, 4]
    emulator.run_procedure("cattorch forward")
    torch.testing.assert_close(
        torch.tensor(emulator.lists["output"]),
        model(torch.tensor([[1.0, 2.0, 3.0, 4.0]])).flatten(),
    )


class TiedEmbeddingHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(384, 128)
        self.head = torch.nn.Linear(128, 384, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, tokens):
        return self.head(self.embedding(tokens))


class _RepeatedResidualBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(8, 16, bias=False)
        self.down = torch.nn.Linear(16, 8, bias=False)

    def forward(self, value):
        return value + self.down(torch.nn.functional.silu(self.up(value)))


class _RepeatedResidualModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(8, 8, bias=False)
        self.blocks = torch.nn.ModuleList(_RepeatedResidualBlock() for _ in range(3))
        self.output = torch.nn.Linear(8, 4, bias=False)

    def forward(self, value):
        value = self.input(value)
        for block in self.blocks:
            value = block(value)
        return self.output(value)


def test_repeated_layers_share_weight_banks_and_procedure(tmp_path):
    torch.manual_seed(91)
    model = _RepeatedResidualModel().eval()
    value = torch.randn(1, 8)
    results = {}
    sprites = {}
    for mode in ("off", "auto"):
        results[mode] = transpile(
            model,
            value,
            tmp_path / mode,
            codegen=CodegenConfig(
                id_namespace="", unrolling="compact", layer_sharing=mode,
            ),
        )
        sprites[mode] = _load(results[mode].path)
    assert results["auto"].block_count < results["off"].block_count
    auto = ScratchEmulator(sprites["auto"])
    off = ScratchEmulator(sprites["off"])
    for emulator in (auto, off):
        emulator.lists["input"] = value.flatten().tolist()
        emulator.run_procedure("cattorch forward")
    torch.testing.assert_close(
        torch.tensor(auto.lists["output"]),
        torch.tensor(off.lists["output"]),
        atol=1e-6,
        rtol=0,
    )
    assert "cattorch shared transformer layer" in auto._procedures


def test_shared_quantized_bank_can_cross_physical_list_boundary(tmp_path):
    """A layer base may land in either shard of one logical weight bank."""
    width = 320  # two matrices contain 204,800 values, just over Scratch's cap

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(width, width, bias=False)

        def forward(self, value):
            return torch.relu(self.linear(value))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList((Block(), Block()))

        def forward(self, value):
            # Give the shared slab a distinct carried input buffer.
            value = value + 0.0
            for block in self.blocks:
                value = block(value)
            return value

    torch.manual_seed(18)
    model = Model().eval()
    value = torch.randn(1, width)
    sprites = {}
    for mode in ("off", "auto"):
        result = transpile(
            model,
            value,
            tmp_path / f"large-bank-{mode}",
            quantization=QuantizationConfig(
                bits=4, method="symmetric", min_quantized_values=1,
            ),
            codegen=CodegenConfig(
                id_namespace="", unrolling="compact", layer_sharing=mode,
            ),
        )
        sprites[mode] = ScratchEmulator(_load(result.path))
        sprites[mode].lists["input"] = value.flatten().tolist()
        sprites[mode].run_procedure("cattorch forward")

    assert "cattorch shared transformer layer" in sprites["auto"]._procedures
    torch.testing.assert_close(
        torch.tensor(sprites["auto"].lists["output"]),
        torch.tensor(sprites["off"].lists["output"]),
        atol=1e-6,
        rtol=0,
    )


def test_tied_grouped_embedding_head_uses_one_physical_weight(tmp_path):
    torch.manual_seed(72)
    model = TiedEmbeddingHead().eval()
    value = torch.tensor([[7]])
    path = tmp_path / "tied"
    transpile(
        model,
        value,
        path,
        storage=StorageConfig(compression=False),
        codegen=CodegenConfig(id_namespace=""),
    )
    sprite = _load(path.with_suffix(".sprite3"))
    weight_lists = [
        entry for entry in sprite["lists"].values() if entry[0].startswith("W_")
    ]
    assert len(weight_lists) == 1
    assert len(weight_lists[0][1]) == model.embedding.weight.numel()

    emulator = ScratchEmulator(sprite)
    emulator.lists["input"] = value.flatten().tolist()
    emulator.run_procedure("cattorch forward")
    torch.testing.assert_close(
        torch.tensor(emulator.lists["output"]),
        model(value).detach().flatten(),
        atol=2e-5,
        rtol=1e-5,
    )


def test_oversized_tied_embedding_head_shares_aligned_shards(tmp_path):
    class OversizedTied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(2048, 128)
            self.head = torch.nn.Linear(128, 2048, bias=False)
            self.head.weight = self.embedding.weight

        def forward(self, tokens):
            return self.head(self.embedding(tokens))

    model = OversizedTied().eval()
    path = tmp_path / "oversized_tied"
    transpile(
        model,
        torch.tensor([[3]]),
        path,
        storage=StorageConfig(compression=False),
        codegen=CodegenConfig(unrolling="compact", id_namespace=""),
    )
    sprite = _load(path.with_suffix(".sprite3"))
    weight_lists = [
        entry for entry in sprite["lists"].values() if entry[0].startswith("W_")
    ]
    assert [len(entry[1]) for entry in weight_lists] == [199_680, 62_464]
    assert sum(len(entry[1]) for entry in weight_lists) == model.embedding.weight.numel()

    # Float32 compression requires smaller temporary-byte-safe shards. When
    # those no longer align with the grouped head representation, export must
    # retain both copies rather than fail or alias mismatched physical lists.
    compressed = transpile(
        model,
        torch.tensor([[3]]),
        tmp_path / "oversized_tied_compressed",
        codegen=CodegenConfig(
            unrolling="compact", id_namespace="", target_json_bytes=None,
        ),
    )
    compressed_sprite = _load(compressed.path)
    payloads = [
        entry[1]
        for entry in compressed_sprite["variables"].values()
        if "weight bank" in entry[0]
    ]
    assert payloads
    assert all(len(payload) <= 250_000 for payload in payloads)


def test_compact_unrolling_emits_fewer_blocks(tmp_path):
    model = torch.nn.Linear(128, 128, bias=False).eval()
    value = torch.randn(1, 128)
    speed = transpile(
        model,
        value,
        tmp_path / "speed",
        storage=StorageConfig(compression=False),
        codegen=CodegenConfig(unrolling="speed", id_namespace=""),
    )
    compact = transpile(
        model,
        value,
        tmp_path / "compact",
        storage=StorageConfig(compression=False),
        codegen=CodegenConfig(unrolling="compact", id_namespace=""),
    )
    assert compact.block_count < speed.block_count
    assert compact.expanded_json_bytes < speed.expanded_json_bytes
