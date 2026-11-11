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


def test_remap_can_share_the_blocks_it_leaves_unchanged():
    blocks = {
        "a": {"opcode": "data_addtolist", "next": "b", "parent": None,
              "inputs": {"ITEM": [1, [10, "x"]]}, "fields": {"LIST": ["items", "old"]}},
        "b": {"opcode": "data_addtolist", "next": None, "parent": "a",
              "inputs": {"ITEM": [1, [10, "y"]]}, "fields": {"LIST": ["items", "kept"]}},
    }
    snapshot = json.loads(json.dumps(blocks))
    copied = remap_ids(blocks, {"old": "kept"})
    shared = remap_ids(blocks, {"old": "kept"}, share_unchanged=True)
    assert shared == copied
    assert list(shared) == list(copied)
    assert blocks == snapshot
    assert shared["b"] is blocks["b"]
    assert shared["a"] is not blocks["a"]
    assert copied["b"] is not blocks["b"]


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
    assert "_cattorch shared transformer layer" in auto._procedures


def test_shared_quantized_bank_can_cross_physical_list_boundary(
    tmp_path, small_list_limit, monkeypatch,
):
    """A layer base may land in either shard of one logical weight bank."""
    from cattorch.util.instruction import optimized_linear

    small_list_limit(2_000)
    # Keep the grouped linear layout a full-size model would use.
    monkeypatch.setattr(optimized_linear, "GROUPED_MIN_MACS", 1)
    width = 32  # two matrices contain 2,048 values, just over the lowered cap

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

    assert "_cattorch shared transformer layer" in sprites["auto"]._procedures
    assert "W_linear_linear_weight_1 shard 2" in sprites["auto"]._list_name_to_ids
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

    # Float32 compression requires smaller temporary-byte-safe shards; the
    # head's shards are cut to them, so the table is still stored once.
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
    assert not compressed.warnings
    # Base92 holds 13 bytes in 16 characters.
    assert sum(map(len, payloads)) < model.embedding.weight.numel() * 4 * 16 / 13 * 1.01


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


def test_batched_renames_apply_in_order_and_walk_the_blocks_once(monkeypatch):
    import cattorch.util.scratch.interface as interface

    walks = []
    original = interface.refresh_data_display_names
    monkeypatch.setattr(
        interface, "refresh_data_display_names",
        lambda blocks, names: (walks.append(dict(names)), original(blocks, names)),
    )
    sprite = {
        "lists": {"a": ["x", []], "b": ["x shard 2", []], "c": ["y", []]},
        "variables": {"n": ["cattorch x shard count", 2], "v": ["value", 0]},
        "blocks": {
            "read": {"fields": {"LIST": ["x shard 2", "b"]}},
            "show": {"inputs": {"VALUE": [3, [12, "value", "v"], [10, ""]]}},
        },
    }
    # Later renames see earlier ones, as separate calls would.
    interface.rename_lists(sprite, {"x": "y2", "y": "z", "y2": "w"})
    interface.rename_variables(sprite, {"value": "total"})
    assert len(walks) == 2
    assert [entry[0] for entry in sprite["lists"].values()] == ["w", "w shard 2", "z"]
    assert sprite["variables"]["n"][0] == "cattorch w shard count"
    assert sprite["blocks"]["read"]["fields"]["LIST"] == ["w shard 2", "b"]
    assert sprite["blocks"]["show"]["inputs"]["VALUE"][1] == [12, "total", "v"]


def test_program_export_walks_the_blocks_independently_of_the_weight_count(tmp_path, monkeypatch):
    import cattorch.program_export as program_export
    import cattorch.util.scratch.interface as interface
    from cattorch import EntryPoint, ExportProgram, Input, Output

    class Stack(torch.nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = torch.nn.Sequential(*(torch.nn.Linear(4, 4) for _ in range(layers)))

        def run(self, value):
            return self.layers(value)

    walks = []
    original = interface.refresh_data_display_names
    for module in (interface, program_export):
        monkeypatch.setattr(
            module, "refresh_data_display_names",
            lambda blocks, names: (walks.append(names), original(blocks, names)),
        )

    def count(layers):
        walks.clear()
        program = ExportProgram(entrypoints=(EntryPoint(
            "run", method="run", arguments=(Input("value", torch.ones(4)),),
            returns=(Output("out"),),
        ),))
        transpile(Stack(layers).eval(), program, tmp_path / f"stack{layers}")
        return len(walks)

    assert count(2) == count(8)


def test_tensor_adder_copies_only_the_lists():
    from cattorch.util.scratch.tensor_adder import TensorAdder

    sprite = {"blocks": {"b": {"opcode": "x"}}, "lists": {"l": ["W_a", [1.0]]}, "variables": {}}
    result = TensorAdder().apply(sprite, ["W_a", "W_b"], weights={"W_a": torch.tensor([2.0])})
    assert sprite["lists"] == {"l": ["W_a", [1.0]]}
    assert result["lists"]["l"] == ["W_a", [2.0]]
    assert result["lists"]["cattorch_tensor_W_b"] == ["W_b", []]
    assert result["blocks"] is sprite["blocks"]


def test_static_values_stand_in_for_the_list_tolist_makes():
    import numpy as np
    from cattorch.sprite import _deinterleave_static_values
    from cattorch.util.scratch.tensor_adder import StaticValues, TensorAdder

    tensor = torch.tensor([1.5, -0.0, float("nan"), 3.25, 1e-30, 7.0, 2.0, -4.0])
    values = TensorAdder().apply({"lists": {}}, ["W_a"], weights={"W_a": tensor})["lists"][
        "cattorch_tensor_W_a"
    ][1]
    assert isinstance(values, StaticValues)
    expected = tensor.tolist()
    assert len(values) == 8 and values[3] == 3.25
    assert values.tolist()[:2] == expected[:2] and list(values)[3:] == expected[3:]
    assert values[4:] == expected[4:] and values[4:].array.base is not None
    # As lists of floats compare, NaN never equals NaN.
    assert values != StaticValues(tensor.numpy().copy())
    assert values[3:] == StaticValues(tensor.numpy()[3:].copy())
    assert np.asarray(values, dtype=np.float64).tolist()[3:] == expected[3:]
    with pytest.raises(TypeError):
        values + values  # noqa: B018
    # Integer tensors keep their exact list; bfloat16 widens exactly.
    integral = TensorAdder().apply({"lists": {}}, ["W_i"], weights={"W_i": torch.arange(3)})
    assert integral["lists"]["cattorch_tensor_W_i"][1] == [0, 1, 2]
    half = StaticValues.from_tensor(torch.tensor([0.1, 3.0], dtype=torch.bfloat16))
    assert half.tolist() == torch.tensor([0.1, 3.0], dtype=torch.bfloat16).tolist()

    # The array path reorders four-row groups exactly as the list path does.
    grouped = torch.arange(2 * 4 * 3 + 5, dtype=torch.float32)
    segments = ((1, 8, 3),)
    assert (
        _deinterleave_static_values(StaticValues.from_tensor(grouped), segments).tolist()
        == _deinterleave_static_values(grouped.tolist(), segments)
    )


def test_slot_merges_leave_the_other_blocks_as_a_full_remap_would(tmp_path, monkeypatch):
    # A merge that only remaps the blocks just added is equivalent to
    # remapping every block only if the others are already what remap_ids
    # makes of them: copied (no sub-object shared between blocks) and in
    # normal form.
    import cattorch.sprite as sprite_module
    from cattorch import (
        EntryPoint, ExportProgram, GenerationProgram, Input, Output, RecurrentGenerationProgram,
        State, StateInput, StateUpdate,
    )
    from cattorch.util.scratch.remap import _remap_blocks

    checked = 0
    original = sprite_module._merge_slots_by_name

    def checked_merge(sprite, section, names, added):
        nonlocal checked
        if added is not None:
            others = {key: block for key, block in sprite["blocks"].items() if key not in added[1]}
            assert _remap_blocks(others, {"unused": "unused"}) == others
            seen = set()
            stack = list(others.values())
            while stack:
                value = stack.pop()
                if isinstance(value, (list, dict)):
                    assert id(value) not in seen, value
                    seen.add(id(value))
                    stack.extend(value.values() if isinstance(value, dict) else value)
            checked += 1
        original(sprite, section, names, added)

    monkeypatch.setattr(sprite_module, "_merge_slots_by_name", checked_merge)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 4)
            self.gru = torch.nn.GRU(4, 4, batch_first=True)
            self.head = torch.nn.Linear(4, 8)

        def forward(self, tokens):
            return self.head(self.gru(self.embed(tokens))[0])

        def step(self, token, hidden):
            output, hidden = self.gru(self.embed(token), hidden)
            return self.head(output), hidden

        def accumulate(self, value, total):
            updated = total + self.head(value)
            return updated.sum().reshape(1), updated

    torch.manual_seed(0)
    model = Model().eval()
    program = ExportProgram(
        states=(State("total", torch.zeros(8)),),
        entrypoints=(EntryPoint(
            "add", method="accumulate", arguments=(Input("value", torch.ones(4)), StateInput("total")),
            returns=(Output("sum"), StateUpdate("total")),
        ),),
    )
    transpile(model, program, tmp_path / "program")
    transpile(
        model, RecurrentGenerationProgram((State("hidden", torch.zeros(1, 1, 4)),)),
        tmp_path / "recurrent",
    )
    from test_autoregressive import NanoGPT

    transpile(
        NanoGPT(layers=1, width=8, heads=2, context=4).eval(), GenerationProgram(max_context=4),
        tmp_path / "generation",
    )
    assert checked
