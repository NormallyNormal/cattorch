"""Build and analyze paired performance projects for the real Scratch editor.

Timing deliberately happens inside Scratch.  This module only constructs the
project and reads the timing lists after the user saves the completed project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import tempfile
import uuid
import warnings
import zipfile
from pathlib import Path

import torch

from cattorch.fast import FastConfig
from cattorch.storage import StorageConfig
from cattorch.templates.template import TEMPLATE_DIR
from cattorch.transpiler import (
    GenerationConfig,
    _add_warp_procedure,
    _merge_lists_by_name,
    transpile,
)
from cattorch.util.scratch.finalize_scratch import online_json_size_warning


LEGACY_TIMINGS = "cattorch legacy timings"
EXACT_TIMINGS = "cattorch exact timings"
FAST_TIMINGS = "cattorch fast timings"
SUITE_RESULTS = "cattorch benchmark results"
STORAGE_SIZES = "cattorch storage sizes"
STORAGE_VARIANTS = (
    "plain-f32", "base85-f32", "base85-f16", "base85-int8", "base85-int6",
    "base85-int4",
)


def _load_sprite(path: Path) -> tuple[dict, dict[str, bytes]]:
    with zipfile.ZipFile(path) as archive:
        sprite = json.loads(archive.read("sprite.json"))
        assets = {
            name: archive.read(name)
            for name in archive.namelist()
            if name != "sprite.json"
        }
    return sprite, assets


def _set_inputs(sprite: dict, example_inputs: tuple[torch.Tensor, ...]) -> None:
    values = {
        "input" if index == 0 else f"input_{index}": tensor.detach().flatten().tolist()
        for index, tensor in enumerate(example_inputs)
    }
    for entry in sprite.get("lists", {}).values():
        if entry[0] in values:
            entry[1] = values[entry[0]]


def _add_input_restore_procedure(
    sprite: dict,
    example_inputs: tuple[torch.Tensor, ...],
) -> None:
    """Add an untimed benchmark-only procedure that restores model inputs."""
    from cattorch.util.scratch.dsl import Program, append, clear

    values = {
        "input" if index == 0 else f"input_{index}": tensor.detach().flatten().tolist()
        for index, tensor in enumerate(example_inputs)
    }
    body = []
    for name, items in values.items():
        body.append(clear(name))
        body.extend(append(name, value) for value in items)
    _add_warp_procedure(
        sprite,
        "cattorch restore benchmark input",
        Program("restore_benchmark_input", lists=values, body=body),
        x=640,
        y=800,
    )
    _merge_lists_by_name(sprite, set(values))


def _add_cached_decode_procedure(sprite: dict, token: int) -> None:
    """Install a benchmark decode call with a valid one-token model input."""
    from cattorch.util.scratch.dsl import Program, append, call, clear

    _add_warp_procedure(
        sprite,
        "cattorch benchmark decode",
        Program(
            "cached_benchmark_decode",
            lists=("input",),
            body=(
                clear("input"),
                append("input", int(token)),
                call("cattorch decode"),
            ),
        ),
        x=640,
        y=960,
    )
    _merge_lists_by_name(sprite, {"input"})


def _add_broadcast_entry(
    sprite: dict,
    message: str,
    broadcast_id: str,
    procedure: str = "cattorch forward",
) -> None:
    prefix = f"cattorch_benchmark_{uuid.uuid4().hex[:8]}"
    event_id = f"{prefix}_event"
    call_id = f"{prefix}_call"
    sprite["blocks"][event_id] = {
        "opcode": "event_whenbroadcastreceived", "next": call_id, "parent": None,
        "inputs": {}, "fields": {"BROADCAST_OPTION": [message, broadcast_id]},
        "shadow": False, "topLevel": True, "x": 640, "y": 0,
    }
    sprite["blocks"][call_id] = {
        "opcode": "procedures_call", "next": None, "parent": event_id,
        "inputs": {}, "fields": {}, "shadow": False, "topLevel": False,
        "mutation": {
            "tagName": "mutation", "children": [],
            "proccode": procedure, "argumentids": "[]",
        },
    }


class _StageBlocks:
    def __init__(self, broadcast_ids: dict[str, str], list_ids: dict[str, str]):
        self.broadcast_ids = broadcast_ids
        self.list_ids = list_ids
        self.blocks = {}
        self.counter = 0

    def _id(self) -> str:
        self.counter += 1
        return f"cattorch_benchmark_block_{self.counter}"

    def _base(self, opcode, parent):
        return {
            "opcode": opcode, "next": None, "parent": parent,
            "inputs": {}, "fields": {}, "shadow": False, "topLevel": False,
        }

    def _chain(self, specs, parent):
        first = previous = None
        for spec in specs:
            block_id = self._command(spec, parent if previous is None else previous)
            if previous is not None:
                self.blocks[previous]["next"] = block_id
            else:
                first = block_id
            previous = block_id
        return first, previous

    def _command(self, spec, parent):
        kind = spec[0]
        block_id = self._id()
        if kind == "clear":
            name = spec[1]
            block = self._base("data_deletealloflist", parent)
            block["fields"]["LIST"] = [name, self.list_ids[name]]
        elif kind == "reset_timer":
            block = self._base("sensing_resettimer", parent)
        elif kind == "broadcast":
            message = spec[1]
            block = self._base("event_broadcastandwait", parent)
            block["inputs"]["BROADCAST_INPUT"] = [
                1, [11, message, self.broadcast_ids[message]],
            ]
        elif kind == "record":
            name = spec[1]
            block = self._base("data_addtolist", parent)
            block["fields"]["LIST"] = [name, self.list_ids[name]]
            reporter_id = self._id()
            block["inputs"]["ITEM"] = [3, reporter_id, [4, 0]]
            self.blocks[reporter_id] = {
                **self._base("sensing_timer", block_id),
            }
        elif kind == "record_result":
            name, label = spec[1], spec[2]
            block = self._base("data_addtolist", parent)
            block["fields"]["LIST"] = [name, self.list_ids[name]]
            join_id = self._id()
            timer_id = self._id()
            block["inputs"]["ITEM"] = [3, join_id, [10, ""]]
            self.blocks[join_id] = {
                **self._base("operator_join", block_id),
                "inputs": {
                    "STRING1": [1, [10, f"{label}: "]],
                    "STRING2": [3, timer_id, [10, ""]],
                },
            }
            self.blocks[timer_id] = {
                **self._base("sensing_timer", join_id),
            }
        elif kind == "set":
            name, var_id, value = spec[1:]
            block = self._base("data_setvariableto", parent)
            block["fields"]["VARIABLE"] = [name, var_id]
            block["inputs"]["VALUE"] = [1, [4, value]]
        elif kind == "change_timer":
            name, var_id = spec[1:]
            block = self._base("data_changevariableby", parent)
            block["fields"]["VARIABLE"] = [name, var_id]
            timer_id = self._id()
            block["inputs"]["VALUE"] = [3, timer_id, [4, 0]]
            self.blocks[timer_id] = {**self._base("sensing_timer", block_id)}
        elif kind == "record_result_var":
            list_name, label, var_name, var_id = spec[1:]
            block = self._base("data_addtolist", parent)
            block["fields"]["LIST"] = [list_name, self.list_ids[list_name]]
            join_id = self._id()
            block["inputs"]["ITEM"] = [3, join_id, [10, ""]]
            self.blocks[join_id] = {
                **self._base("operator_join", block_id),
                "inputs": {
                    "STRING1": [1, [10, f"{label}: "]],
                    "STRING2": [3, [12, var_name, var_id], [10, ""]],
                },
            }
        elif kind == "repeat":
            times, body = spec[1], spec[2]
            block = self._base("control_repeat", parent)
            block["inputs"]["TIMES"] = [1, [4, times]]
            self.blocks[block_id] = block
            substack, _ = self._chain(body, block_id)
            block["inputs"]["SUBSTACK"] = [2, substack]
            return block_id
        else:
            raise ValueError(kind)
        self.blocks[block_id] = block
        return block_id

    def build(self, warmups: int, repeats: int):
        exact_message = "cattorch benchmark exact"
        fast_message = "cattorch benchmark fast"
        specs = [
            ("clear", EXACT_TIMINGS),
            ("clear", FAST_TIMINGS),
            ("repeat", warmups, (
                ("broadcast", exact_message),
                ("broadcast", fast_message),
            )),
        ]
        for index in range(repeats):
            order = (
                ((exact_message, EXACT_TIMINGS), (fast_message, FAST_TIMINGS))
                if index % 2 == 0 else
                ((fast_message, FAST_TIMINGS), (exact_message, EXACT_TIMINGS))
            )
            for message, timings in order:
                specs.extend((
                    ("reset_timer",),
                    ("broadcast", message),
                    ("record", timings),
                ))

        hat_id = self._id()
        first, _ = self._chain(tuple(specs), hat_id)
        self.blocks[hat_id] = {
            "opcode": "event_whenflagclicked", "next": first, "parent": None,
            "inputs": {}, "fields": {}, "shadow": False, "topLevel": True,
            "x": 0, "y": 0,
        }
        return self.blocks

    def build_suite(
        self,
        cases: list[dict],
        iterations: int,
        warmups: int,
        variants: tuple[str, str],
    ):
        specs = [("clear", SUITE_RESULTS)]
        for case in cases:
            for variant in variants:
                messages = case[variant]
                specs.append(("broadcast", messages["init"]))
                if warmups:
                    specs.append((
                        "repeat", warmups,
                        (("broadcast", messages["forward"]),),
                    ))
                specs.extend((
                    ("reset_timer",),
                    (
                        "repeat", iterations,
                        (("broadcast", messages["forward"]),),
                    ),
                    (
                        "record_result", SUITE_RESULTS,
                        f"{case['name']} {variant}",
                    ),
                ))

        hat_id = self._id()
        first, _ = self._chain(tuple(specs), hat_id)
        self.blocks[hat_id] = {
            "opcode": "event_whenflagclicked", "next": first, "parent": None,
            "inputs": {}, "fields": {}, "shadow": False, "topLevel": True,
            "x": 0, "y": 0,
        }
        return self.blocks

    def build_storage(self, cases: list[dict], iterations: int):
        specs = [("clear", SUITE_RESULTS)]
        for case in cases:
            for variant in STORAGE_VARIANTS:
                messages = case[variant]
                variable_name = f"storage total {case['index']} {variant}"
                variable_id = f"cattorch_storage_total_{case['index']}_{variant}"
                specs.append(("set", variable_name, variable_id, 0))
                specs.append((
                    "repeat",
                    iterations,
                    (
                        ("broadcast", messages["prepare"]),
                        ("reset_timer",),
                        ("broadcast", messages["init"]),
                        ("change_timer", variable_name, variable_id),
                    ),
                ))
                specs.append((
                    "record_result_var", SUITE_RESULTS,
                    f"{case['name']} {variant} init", variable_name, variable_id,
                ))
                specs.extend((
                    ("broadcast", messages["restore"]),
                    ("reset_timer",),
                    ("repeat", iterations, (("broadcast", messages["forward"]),)),
                    ("record_result", SUITE_RESULTS, f"{case['name']} {variant} forward"),
                ))

        hat_id = self._id()
        first, _ = self._chain(tuple(specs), hat_id)
        self.blocks[hat_id] = {
            "opcode": "event_whenflagclicked", "next": first, "parent": None,
            "inputs": {}, "fields": {}, "shadow": False, "topLevel": True,
            "x": 0, "y": 0,
        }
        return self.blocks


def _stage_target(
    blocks: dict,
    broadcasts: dict[str, str],
    lists: dict[str, list],
    variables: dict[str, list] | None = None,
) -> tuple[dict, str, bytes]:
    source_svg = next((TEMPLATE_DIR / "sprite").glob("*.svg"))
    source_bytes = source_svg.read_bytes()
    stage_asset_id = hashlib.md5(source_bytes).hexdigest()
    stage_md5ext = f"{stage_asset_id}.svg"
    stage = {
        "isStage": True,
        "name": "Stage",
        "variables": variables or {},
        "lists": lists,
        "broadcasts": {identifier: name for name, identifier in broadcasts.items()},
        "blocks": blocks,
        "comments": {},
        "currentCostume": 0,
        "costumes": [{
            "name": "backdrop", "bitmapResolution": 1, "dataFormat": "svg",
            "assetId": stage_asset_id, "md5ext": stage_md5ext,
            "rotationCenterX": 48, "rotationCenterY": 50,
        }],
        "sounds": [], "volume": 100, "layerOrder": 0,
        "tempo": 60, "videoTransparency": 50, "videoState": "on",
        "textToSpeechLanguage": None,
    }
    return stage, stage_md5ext, source_bytes


def _write_project(
    output_path: Path,
    stage: dict,
    sprites: list[dict],
    assets: dict[str, bytes],
    stage_asset: tuple[str, bytes],
    description: str,
    monitors: list[dict] | None = None,
) -> Path:
    project = {
        "targets": [stage, *sprites],
        "monitors": monitors or [], "extensions": [],
        "meta": {"semver": "3.0.0", "vm": "0.4.0-cattorch", "agent": "cattorch"},
    }
    stage_md5ext, stage_bytes = stage_asset
    assets[stage_md5ext] = stage_bytes
    project_json = json.dumps(project, separators=(",", ":"))
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("project.json", project_json)
        for name, data in assets.items():
            archive.writestr(name, data)
    size_warning = online_json_size_warning(
        len(project_json.encode("utf-8")), f"{description} project JSON",
    )
    if size_warning is not None:
        warnings.warn(size_warning, stacklevel=2)
    return output_path


def _normalise_inputs(
    example_inputs: torch.Tensor | tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    return (example_inputs,) if isinstance(example_inputs, torch.Tensor) else example_inputs


def _compile_suite_variant(
    model: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    base: Path,
    optimization: str,
    sig_figs: int | None,
    fast_config: FastConfig | None = None,
    storage: StorageConfig | None = None,
) -> tuple[dict, dict[str, bytes]]:
    transpile(
        model, example_inputs, str(base), sig_figs,
        optimization=optimization,
        storage=storage,
        **({"fast_config": fast_config} if optimization == "fast" else {}),
    )
    sprite, assets = _load_sprite(base.with_suffix(".sprite3"))
    _set_inputs(sprite, example_inputs)
    return sprite, assets


def build_benchmark_suite(
    cases,
    output_path: str | Path,
    *,
    iterations: int = 100,
    warmups: int = 0,
    sig_figs: int | None = None,
    fast_config: FastConfig | None = None,
    storage: StorageConfig | None = None,
) -> Path:
    """Bundle named comparison benchmarks into one sequential Scratch project.

    ``cases`` contains ``(name, model, example_inputs)`` tuples. An optional
    fourth item supplies that case's ``FastConfig``
    and replaces the suite-wide ``fast_config``. Each variant is explicitly
    initialized, optionally warmed up, run ``iterations`` times, and recorded
    as ``"<name> <variant>: <seconds>"`` in one visible stage list. Timing
    still happens entirely in stock Scratch.
    """
    cases = list(cases)
    if not cases:
        raise ValueError("At least one benchmark case is required")
    if iterations < 1 or warmups < 0:
        raise ValueError("iterations must be positive and warmups must be non-negative")
    if storage is not None and not isinstance(storage, StorageConfig):
        raise TypeError("storage must be a StorageConfig")
    variants = (("exact", "exact"), ("fast", "fast"))
    normalized_cases = []
    for case in cases:
        if len(case) not in {3, 4}:
            raise ValueError("Benchmark cases must contain three or four items")
        name, model, raw_inputs = case[:3]
        case_fast_config = case[3] if len(case) == 4 else fast_config
        if case_fast_config is not None and not isinstance(case_fast_config, FastConfig):
            raise TypeError("Per-case fast configuration must be a FastConfig")
        normalized_cases.append((name, model, raw_inputs, case_fast_config))

    names = [case[0] for case in normalized_cases]
    if len(set(names)) != len(names):
        raise ValueError("Benchmark case names must be unique")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Benchmark case names must be non-empty strings")

    output_path = Path(output_path)
    if output_path.suffix != ".sb3":
        output_path = output_path.with_suffix(".sb3")

    broadcasts: dict[str, str] = {}
    controller_cases: list[dict] = []
    sprites: list[dict] = []
    assets: dict[str, bytes] = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        for index, (name, model, raw_inputs, case_fast_config) in enumerate(normalized_cases):
            example_inputs = _normalise_inputs(raw_inputs)
            controller_case = {"name": name}
            for variant, optimization in variants:
                sprite, sprite_assets = _compile_suite_variant(
                    model,
                    example_inputs,
                    temp / f"case_{index}_{variant}",
                    optimization,
                    sig_figs,
                    case_fast_config,
                    storage,
                )
                sprite_name = f"{name} {variant}"
                sprite["name"] = sprite_name
                sprite["layerOrder"] = len(sprites) + 1
                sprite["visible"] = False

                variant_messages = {}
                for procedure in ("init", "forward"):
                    message = f"cattorch suite {index} {variant} {procedure}"
                    broadcast_id = f"cattorch_suite_{index}_{variant}_{procedure}"
                    broadcasts[message] = broadcast_id
                    _add_broadcast_entry(
                        sprite,
                        message,
                        broadcast_id,
                        f"cattorch {procedure}",
                    )
                    variant_messages[procedure] = message
                controller_case[variant] = variant_messages
                sprites.append(sprite)
                assets.update(sprite_assets)
            controller_cases.append(controller_case)

    results_id = "cattorch_benchmark_suite_results"
    list_ids = {SUITE_RESULTS: results_id}
    stage_blocks = _StageBlocks(broadcasts, list_ids).build_suite(
        controller_cases, iterations, warmups,
        tuple(variant for variant, _ in variants),
    )
    stage, stage_md5ext, stage_bytes = _stage_target(
        stage_blocks,
        broadcasts,
        {results_id: [SUITE_RESULTS, []]},
        {
            "cattorch_benchmark_iterations": [
                "cattorch benchmark iterations", iterations,
            ],
            "cattorch_benchmark_warmups": [
                "cattorch benchmark warmups", warmups,
            ],
            "cattorch_benchmark_comparison": [
                "cattorch benchmark comparison", "fast",
            ],
        },
    )
    monitor = {
        "id": results_id,
        "mode": "list",
        "opcode": "data_listcontents",
        "params": {"LIST": SUITE_RESULTS},
        "spriteName": None,
        "value": [],
        "width": 460,
        "height": 330,
        "x": 10,
        "y": 10,
        "visible": True,
    }
    return _write_project(
        output_path,
        stage,
        sprites,
        assets,
        (stage_md5ext, stage_bytes),
        "Benchmark suite",
        monitors=[monitor],
    )


def build_generation_benchmark_suite(
    cases,
    output_path: str | Path,
    *,
    iterations: int = 16,
    sig_figs: int | None = None,
) -> Path:
    """Bundle stateless-forward versus KV-cached decode benchmarks.

    Each case is ``(name, model, stateless_input, prompt)``.  The cached arm
    prefills ``prompt`` outside the timer and then performs ``iterations``
    decode calls.  Its context capacity is chosen to fit the prompt plus all
    timed calls.
    """
    cases = list(cases)
    if not cases or iterations < 1:
        raise ValueError("generation cases and positive iterations are required")
    output_path = Path(output_path).with_suffix(".sb3")
    broadcasts: dict[str, str] = {}
    controller_cases = []
    sprites = []
    assets: dict[str, bytes] = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        for index, case in enumerate(cases):
            if len(case) != 4:
                raise ValueError(
                    "generation benchmark cases must contain name, model, "
                    "stateless_input, and prompt"
                )
            name, model, stateless_input, prompt = case
            stateless_inputs = _normalise_inputs(stateless_input)
            prompt_inputs = _normalise_inputs(prompt)
            if len(prompt_inputs) != 1 or prompt_inputs[0].ndim != 2:
                raise ValueError("generation benchmark prompts must be [1, T] tensors")
            controller = {"name": name}

            stateless, stateless_assets = _compile_suite_variant(
                model, stateless_inputs, temp / f"generation_{index}_stateless",
                "exact", sig_figs,
            )
            stateless["name"] = f"{name} stateless"
            stateless["visible"] = False
            stateless["layerOrder"] = len(sprites) + 1
            stateless_messages = {}
            for label, procedure in (("init", "cattorch init"), ("forward", "cattorch forward")):
                message = f"cattorch generation {index} stateless {label}"
                identifier = f"cattorch_generation_{index}_stateless_{label}"
                broadcasts[message] = identifier
                _add_broadcast_entry(stateless, message, identifier, procedure)
                stateless_messages[label] = message
            controller["stateless"] = stateless_messages
            sprites.append(stateless)
            assets.update(stateless_assets)

            cached_base = temp / f"generation_{index}_cached"
            transpile(
                model,
                prompt_inputs[0][:, :1],
                str(cached_base),
                sig_figs,
                optimization="exact",
                generation=GenerationConfig(prompt_inputs[0].shape[1] + iterations),
            )
            cached, cached_assets = _load_sprite(cached_base.with_suffix(".sprite3"))
            _set_inputs(cached, prompt_inputs)
            prompt_length = prompt_inputs[0].shape[1]
            full_tokens = stateless_inputs[0]
            decode_index = min(prompt_length, full_tokens.shape[1] - 1)
            _add_cached_decode_procedure(
                cached, int(full_tokens[0, decode_index]),
            )
            cached["name"] = f"{name} cached"
            cached["visible"] = False
            cached["layerOrder"] = len(sprites) + 1
            cached_messages = {}
            # Prefill includes initialization and is deliberately outside timing.
            for label, procedure in (
                ("init", "cattorch prefill"),
                ("forward", "cattorch benchmark decode"),
            ):
                message = f"cattorch generation {index} cached {label}"
                identifier = f"cattorch_generation_{index}_cached_{label}"
                broadcasts[message] = identifier
                _add_broadcast_entry(cached, message, identifier, procedure)
                cached_messages[label] = message
            controller["cached"] = cached_messages
            sprites.append(cached)
            assets.update(cached_assets)
            controller_cases.append(controller)

    results_id = "cattorch_generation_benchmark_results"
    stage_blocks = _StageBlocks(broadcasts, {SUITE_RESULTS: results_id}).build_suite(
        controller_cases, iterations, 0, ("stateless", "cached"),
    )
    stage, stage_md5ext, stage_bytes = _stage_target(
        stage_blocks,
        broadcasts,
        {results_id: [SUITE_RESULTS, []]},
        {"cattorch_generation_iterations": ["cattorch benchmark iterations", iterations]},
    )
    monitor = {
        "id": results_id, "mode": "list", "opcode": "data_listcontents",
        "params": {"LIST": SUITE_RESULTS}, "spriteName": None, "value": [],
        "width": 460, "height": 330, "x": 10, "y": 10, "visible": True,
    }
    return _write_project(
        output_path, stage, sprites, assets, (stage_md5ext, stage_bytes),
        "Generation benchmark suite", monitors=[monitor],
    )


def build_storage_benchmark_suite(
    cases,
    output_path: str | Path,
    *,
    iterations: int = 10,
    sig_figs: int | None = None,
) -> Path:
    """Benchmark startup unpacking and forward speed for all storage modes.

    Each case is ``(name, model, example_inputs)``. The resulting project
    records accumulated initialization time separately from forward time and
    exposes the individual ``.sprite3`` byte sizes in a second stage list.
    """
    cases = list(cases)
    if not cases or iterations < 1:
        raise ValueError("storage cases and positive iterations are required")
    names = [case[0] for case in cases if len(case) == 3]
    if len(names) != len(cases) or len(set(names)) != len(names):
        raise ValueError("storage cases must contain unique name, model, and inputs")

    output_path = Path(output_path).with_suffix(".sb3")
    variants = {
        "plain-f32": StorageConfig(compression=False, precision="float32"),
        "base85-f32": StorageConfig(compression=True, precision="float32"),
        "base85-f16": StorageConfig(compression=True, precision="float16"),
        "base85-int8": StorageConfig(compression=True, precision="int8"),
        "base85-int6": StorageConfig(compression=True, precision="int6"),
        "base85-int4": StorageConfig(compression=True, precision="int4"),
    }
    broadcasts = {}
    controller_cases = []
    sprites = []
    assets = {}
    sizes = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        for index, (name, model, raw_inputs) in enumerate(cases):
            example_inputs = _normalise_inputs(raw_inputs)
            controller = {"name": name, "index": index}
            for variant, storage in variants.items():
                base = temp / f"storage_{index}_{variant}"
                transpile(
                    model,
                    example_inputs,
                    str(base),
                    sig_figs,
                    optimization="exact",
                    storage=storage,
                )
                sprite_path = base.with_suffix(".sprite3")
                sizes.append(f"{name} {variant}: {sprite_path.stat().st_size} bytes")
                sprite, sprite_assets = _load_sprite(sprite_path)
                _set_inputs(sprite, example_inputs)
                _add_input_restore_procedure(sprite, example_inputs)
                sprite["name"] = f"{name} {variant}"
                sprite["visible"] = False
                sprite["layerOrder"] = len(sprites) + 1

                messages = {}
                for label, procedure in (
                    ("prepare", "cattorch prepare for save"),
                    ("init", "cattorch init"),
                    ("restore", "cattorch restore benchmark input"),
                    ("forward", "cattorch forward"),
                ):
                    message = f"cattorch storage {index} {variant} {label}"
                    identifier = f"cattorch_storage_{index}_{variant}_{label}"
                    broadcasts[message] = identifier
                    _add_broadcast_entry(sprite, message, identifier, procedure)
                    messages[label] = message
                controller[variant] = messages
                sprites.append(sprite)
                assets.update(sprite_assets)
            controller_cases.append(controller)

    results_id = "cattorch_storage_benchmark_results"
    sizes_id = "cattorch_storage_benchmark_sizes"
    list_ids = {SUITE_RESULTS: results_id, STORAGE_SIZES: sizes_id}
    stage_builder = _StageBlocks(broadcasts, list_ids)
    stage_blocks = stage_builder.build_storage(controller_cases, iterations)
    stage_variables = {
        f"cattorch_storage_total_{case['index']}_{variant}": [
            f"storage total {case['index']} {variant}", 0,
        ]
        for case in controller_cases
        for variant in variants
    }
    stage_variables["cattorch_storage_iterations"] = [
        "cattorch storage benchmark iterations", iterations,
    ]
    stage, stage_md5ext, stage_bytes = _stage_target(
        stage_blocks,
        broadcasts,
        {
            results_id: [SUITE_RESULTS, []],
            sizes_id: [STORAGE_SIZES, sizes],
        },
        stage_variables,
    )
    monitors = [
        {
            "id": identifier, "mode": "list", "opcode": "data_listcontents",
            "params": {"LIST": name}, "spriteName": None, "value": values,
            "width": 460, "height": 250, "x": x, "y": 10, "visible": True,
        }
        for identifier, name, values, x in (
            (results_id, SUITE_RESULTS, [], 10),
            (sizes_id, STORAGE_SIZES, sizes, 480),
        )
    ]
    return _write_project(
        output_path, stage, sprites, assets, (stage_md5ext, stage_bytes),
        "Storage benchmark suite", monitors=monitors,
    )


def build_paired_benchmark(
    model: torch.nn.Module,
    example_inputs: torch.Tensor | tuple[torch.Tensor, ...],
    output_path: str | Path,
    *,
    sig_figs: int | None = None,
    warmups: int = 2,
    repeats: int = 7,
    fast_config: FastConfig | None = None,
) -> Path:
    """Create an exact-vs-fast .sb3 to run on scratch.mit.edu."""
    if isinstance(example_inputs, torch.Tensor):
        example_inputs = (example_inputs,)
    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be non-negative and repeats must be positive")

    output_path = Path(output_path)
    if output_path.suffix != ".sb3":
        output_path = output_path.with_suffix(".sb3")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        exact_base = temp / "cattorch benchmark exact"
        fast_base = temp / "cattorch benchmark fast"
        transpile(model, example_inputs, str(exact_base), sig_figs, optimization="exact")
        transpile(
            model,
            example_inputs,
            str(fast_base),
            sig_figs,
            optimization="fast",
            fast_config=fast_config,
        )
        exact, exact_assets = _load_sprite(exact_base.with_suffix(".sprite3"))
        fast, fast_assets = _load_sprite(fast_base.with_suffix(".sprite3"))

    _set_inputs(exact, example_inputs)
    _set_inputs(fast, example_inputs)

    messages = {
        "cattorch benchmark exact": "cattorch_broadcast_exact",
        "cattorch benchmark fast": "cattorch_broadcast_fast",
    }
    _add_broadcast_entry(exact, "cattorch benchmark exact", messages["cattorch benchmark exact"])
    _add_broadcast_entry(fast, "cattorch benchmark fast", messages["cattorch benchmark fast"])

    exact["name"] = "cattorch exact"
    fast["name"] = "cattorch fast"
    exact["layerOrder"] = 1
    fast["layerOrder"] = 2

    timing_ids = {
        EXACT_TIMINGS: "cattorch_benchmark_exact_timings",
        FAST_TIMINGS: "cattorch_benchmark_fast_timings",
    }
    stage_blocks = _StageBlocks(messages, timing_ids).build(warmups, repeats)

    stage_asset_id = hashlib.md5(b"cattorch benchmark stage").hexdigest()
    stage_md5ext = f"{stage_asset_id}.svg"
    stage = {
        "isStage": True,
        "name": "Stage",
        "variables": {},
        "lists": {
            timing_ids[EXACT_TIMINGS]: [EXACT_TIMINGS, []],
            timing_ids[FAST_TIMINGS]: [FAST_TIMINGS, []],
        },
        "broadcasts": {identifier: name for name, identifier in messages.items()},
        "blocks": stage_blocks,
        "comments": {},
        "currentCostume": 0,
        "costumes": [{
            "name": "backdrop", "bitmapResolution": 1, "dataFormat": "svg",
            "assetId": stage_asset_id, "md5ext": stage_md5ext,
            "rotationCenterX": 48, "rotationCenterY": 50,
        }],
        "sounds": [], "volume": 100, "layerOrder": 0,
        "tempo": 60, "videoTransparency": 50, "videoState": "on",
        "textToSpeechLanguage": None,
    }
    project = {
        "targets": [stage, exact, fast],
        "monitors": [], "extensions": [],
        "meta": {"semver": "3.0.0", "vm": "0.4.0-cattorch", "agent": "cattorch"},
    }

    source_svg = next((TEMPLATE_DIR / "sprite").glob("*.svg"))
    assets = {**exact_assets, **fast_assets, stage_md5ext: source_svg.read_bytes()}
    project_json = json.dumps(project, separators=(",", ":"))
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("project.json", project_json)
        for name, data in assets.items():
            archive.writestr(name, data)
    size_warning = online_json_size_warning(
        len(project_json.encode("utf-8")), "Paired benchmark project JSON",
    )
    if size_warning is not None:
        warnings.warn(size_warning, stacklevel=2)
    return output_path


def _project_identity(path: str | Path) -> dict:
    return {
        "project_bytes": Path(path).stat().st_size,
        "project_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }


def _target_output(target: dict) -> list[float]:
    output = next(
        (entry[1] for entry in target.get("lists", {}).values() if entry[0] == "output"),
        [],
    )
    return [float(item) for item in output]


def _analyze_suite(path: str | Path, project: dict, stage: dict, entries: list) -> dict:
    parsed: dict[str, dict[str, float]] = {}
    ordered_names = []
    raw_results = [str(item) for item in entries]
    for item in raw_results:
        try:
            label, raw_time = item.rsplit(": ", 1)
            name, variant = label.rsplit(" ", 1)
            elapsed = float(raw_time)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Malformed benchmark suite result: {item!r}") from error
        if variant not in {"legacy", "exact", "fast"}:
            raise ValueError(f"Unknown benchmark suite variant in {item!r}")
        if name not in parsed:
            parsed[name] = {}
            ordered_names.append(name)
        if variant in parsed[name]:
            raise ValueError(f"Duplicate benchmark suite result for {name} {variant}")
        parsed[name][variant] = elapsed

    all_variants = set().union(*(set(values) for values in parsed.values()))
    expected_variants = {"exact", "fast"} if "fast" in all_variants else {"legacy", "exact"}
    baseline_variant, candidate_variant = (
        ("exact", "fast") if expected_variants == {"exact", "fast"}
        else ("legacy", "exact")
    )
    incomplete = [
        name for name, values in parsed.items()
        if set(values) != expected_variants
    ]
    if incomplete:
        raise ValueError(f"Benchmark suite contains incomplete cases: {', '.join(incomplete)}")

    variables = {entry[0]: entry[1] for entry in stage.get("variables", {}).values()}
    targets = {
        target["name"]: target
        for target in project["targets"]
        if not target["isStage"]
    }
    results = []
    for name in ordered_names:
        baseline = parsed[name][baseline_variant]
        candidate = parsed[name][candidate_variant]
        baseline_output = _target_output(targets.get(f"{name} {baseline_variant}", {}))
        candidate_output = _target_output(targets.get(f"{name} {candidate_variant}", {}))
        output_error = None
        output_length_match = None
        if baseline_output or candidate_output:
            output_length_match = len(baseline_output) == len(candidate_output)
        if baseline_output and len(baseline_output) == len(candidate_output):
            output_error = max(
                abs(left - right)
                for left, right in zip(baseline_output, candidate_output)
            )
        result = {
            "name": name,
            f"{baseline_variant}_seconds": baseline,
            f"{candidate_variant}_seconds": candidate,
            "speedup": baseline / candidate if candidate else None,
            "output_max_abs_error": output_error,
            "output_length_match": output_length_match,
            f"{baseline_variant}_output_values": len(baseline_output),
            f"{candidate_variant}_output_values": len(candidate_output),
            f"{baseline_variant}_blocks": len(
                targets.get(f"{name} {baseline_variant}", {}).get("blocks", {})
            ),
            f"{candidate_variant}_blocks": len(
                targets.get(f"{name} {candidate_variant}", {}).get("blocks", {})
            ),
        }
        results.append(result)

    return {
        "kind": "suite",
        "comparison": f"{baseline_variant}_vs_{candidate_variant}",
        "iterations_per_result": variables.get("cattorch benchmark iterations"),
        "warmups": variables.get("cattorch benchmark warmups"),
        "raw_results": raw_results,
        "benchmarks": results,
        **_project_identity(path),
    }


def analyze_benchmark(path: str | Path) -> dict:
    """Read timing lists from a paired or suite project saved after execution."""
    with zipfile.ZipFile(path) as archive:
        project = json.loads(archive.read("project.json"))
    stage = next(target for target in project["targets"] if target["isStage"])
    by_name = {entry[0]: entry[1] for entry in stage.get("lists", {}).values()}
    if STORAGE_SIZES in by_name:
        raw_results = by_name.get(SUITE_RESULTS, [])
        if not raw_results:
            raise ValueError("Storage benchmark suite has not run")
        timings = {
            str(entry).rsplit(": ", 1)[0]: float(str(entry).rsplit(": ", 1)[1])
            for entry in raw_results
        }
        sizes = {
            str(entry).rsplit(": ", 1)[0]: int(
                str(entry).rsplit(": ", 1)[1].removesuffix(" bytes")
            )
            for entry in by_name[STORAGE_SIZES]
        }
        variants = STORAGE_VARIANTS
        targets = [target for target in project["targets"] if not target["isStage"]]
        case_names = []
        for target in targets:
            for variant in variants:
                suffix = f" {variant}"
                if target["name"].endswith(suffix):
                    name = target["name"][:-len(suffix)]
                    if name not in case_names:
                        case_names.append(name)
        results = []
        for name in case_names:
            variants_result = {}
            for variant in variants:
                variants_result[variant] = {
                    "init_seconds": timings[f"{name} {variant} init"],
                    "forward_seconds": timings[f"{name} {variant} forward"],
                    "sprite_bytes": sizes[f"{name} {variant}"],
                }
            results.append({"name": name, "variants": variants_result})
        variables = {entry[0]: entry[1] for entry in stage.get("variables", {}).values()}
        return {
            "kind": "storage_suite",
            "iterations_per_result": variables.get("cattorch storage benchmark iterations"),
            "raw_results": raw_results,
            "benchmarks": results,
            **_project_identity(path),
        }
    if SUITE_RESULTS in by_name:
        entries = by_name[SUITE_RESULTS]
        if not entries:
            raise ValueError("Benchmark suite has not run")
        return _analyze_suite(path, project, stage, entries)

    exact = [float(item) for item in by_name.get(EXACT_TIMINGS, [])]
    fast = [float(item) for item in by_name.get(FAST_TIMINGS, [])]
    legacy = [float(item) for item in by_name.get(LEGACY_TIMINGS, [])]
    if fast:
        baseline_name, candidate_name = "exact", "fast"
        baseline, candidate = exact, fast
    else:
        # Continue to analyze historical pre-0.4 legacy/exact projects even
        # though cattorch no longer generates the retired template backend.
        baseline_name, candidate_name = "legacy", "exact"
        baseline, candidate = legacy, exact
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("Benchmark has not run or contains incomplete timing lists")
    baseline_median = statistics.median(baseline)
    candidate_median = statistics.median(candidate)
    rng = random.Random(0)
    bootstrapped = []
    for _ in range(10_000):
        indices = [rng.randrange(len(baseline)) for _ in baseline]
        baseline_sample = [baseline[index] for index in indices]
        candidate_sample = [candidate[index] for index in indices]
        sampled_candidate_median = statistics.median(candidate_sample)
        if sampled_candidate_median > 0:
            bootstrapped.append(
                statistics.median(baseline_sample) / sampled_candidate_median
            )
    bootstrapped.sort()
    if candidate_median > 0 and bootstrapped:
        lower = bootstrapped[int(len(bootstrapped) * 0.025)]
        upper = bootstrapped[int(len(bootstrapped) * 0.975)]
        speedup_interval = [lower, upper]
    else:
        speedup_interval = None

    sprites = {target["name"]: target for target in project["targets"] if not target["isStage"]}
    outputs = {name: _target_output(target) for name, target in sprites.items()}
    baseline_output = outputs.get(f"cattorch {baseline_name}", [])
    candidate_output = outputs.get(f"cattorch {candidate_name}", [])
    output_error = None
    output_length_match = None
    if baseline_output or candidate_output:
        output_length_match = len(baseline_output) == len(candidate_output)
    if len(baseline_output) == len(candidate_output) and baseline_output:
        output_error = max(
            abs(left - right)
            for left, right in zip(baseline_output, candidate_output)
        )

    baseline_quartiles = (
        statistics.quantiles(baseline, n=4, method="inclusive")
        if len(baseline) > 1 else [baseline[0]] * 3
    )
    candidate_quartiles = (
        statistics.quantiles(candidate, n=4, method="inclusive")
        if len(candidate) > 1 else [candidate[0]] * 3
    )
    return {
        "comparison": f"{baseline_name}_vs_{candidate_name}",
        "runs": len(baseline),
        baseline_name: baseline,
        candidate_name: candidate,
        f"{baseline_name}_median_seconds": baseline_median,
        f"{candidate_name}_median_seconds": candidate_median,
        f"{baseline_name}_quartiles_seconds": [
            baseline_quartiles[0], baseline_quartiles[2],
        ],
        f"{candidate_name}_quartiles_seconds": [
            candidate_quartiles[0], candidate_quartiles[2],
        ],
        "speedup": baseline_median / candidate_median if candidate_median > 0 else None,
        "speedup_95_percent_interval": speedup_interval,
        "output_max_abs_error": output_error,
        "output_length_match": output_length_match,
        f"{baseline_name}_output_values": len(baseline_output),
        f"{candidate_name}_output_values": len(candidate_output),
        f"{baseline_name}_blocks": len(
            sprites.get(f"cattorch {baseline_name}", {}).get("blocks", {})
        ),
        f"{candidate_name}_blocks": len(
            sprites.get(f"cattorch {candidate_name}", {}).get("blocks", {})
        ),
        **_project_identity(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a cattorch Scratch benchmark")
    parser.add_argument("project", help="Benchmark .sb3 saved after it ran in Scratch")
    args = parser.parse_args()
    print(json.dumps(analyze_benchmark(args.project), indent=2))


if __name__ == "__main__":
    main()
