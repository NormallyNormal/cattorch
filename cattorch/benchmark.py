"""Build and analyze paired performance projects for the real Scratch editor.

Timing deliberately happens inside Scratch.  This module only constructs the
project and reads the timing lists after the user saves the completed project.
"""

from __future__ import annotations

import hashlib
import random
import statistics
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from cattorch import interface
from cattorch.codegen import CodegenConfig
from cattorch.errors import warn
from cattorch.export_options import validate_export_options, validate_model_and_inputs
from cattorch.fast import FastConfig
from cattorch.storage import StorageConfig
from cattorch.quantization import QuantizationConfig
from cattorch.program import GenerationProgram, RecurrentGenerationProgram
from cattorch.templates.template import TEMPLATE_DIR
from cattorch.transpiler import transpile
from cattorch.util.paths import output_file
from cattorch.util.scratch.dsl import (
    Program, append, broadcast_and_wait, change_var, clear, for_each, join, length, repeat,
    reset_timer, set_var, timer, var,
)
from cattorch.util.scratch.dsl import item as item_of
from cattorch.util.scratch.remap import remap_ids
from cattorch.util.scratch.archive import (
    PROJECT_JSON, read_archive, read_user_archive, write_archive,
)
from cattorch.util.scratch.finalize_scratch import online_json_size_warning
from cattorch.util.scratch.globals import install
from cattorch.util.scratch.interface import read_logical_list
from cattorch.util.scratch.sharding import SCRATCH_LIST_LIMIT
from cattorch.interface import input_name
from cattorch.interface import TOKEN_IDS


# The public names of a benchmark project, which analyze_benchmark and the
# benchmarks/*.cjs runners read: the results lists and the settings.
LEGACY_TIMINGS = "cattorch legacy timings"
EXACT_TIMINGS = "cattorch exact timings"
FAST_TIMINGS = "cattorch fast timings"
SUITE_RESULTS = "cattorch benchmark results"
STORAGE_SIZES = "cattorch storage sizes"
REPEATS = "cattorch benchmark repeats"
WARMUPS = "cattorch benchmark warmups"
COMPARISON = "cattorch benchmark comparison"
# The internal stage lists a storage benchmark restores each case's inputs
# from, and a generation benchmark its prompts.
_SAVED_INPUT = "_cattorch benchmark saved {}"
_PROMPT = "_cattorch benchmark prompt {}"
_COPY_INDEX = "_cattorch benchmark copy index"
# The sprites of a paired benchmark.
_PAIRED_NAMES = {"exact": "cattorch exact", "fast": "cattorch fast"}
STORAGE_VARIANTS = (
    "plain-f32", "base92-f32", "base92-f16", "base92-int8", "base92-int6",
    "base92-int4-huffman",
)


def _load_sprite(path: Path) -> tuple[dict, dict[str, bytes]]:
    return read_archive(path)


def _input_values(sprite_name: str, example_inputs: tuple[torch.Tensor, ...]) -> dict:
    """The values the sprite's input globals start with, shard by shard, by name."""
    values = {}
    for index, tensor in enumerate(example_inputs):
        flat = tensor.detach().cpu().flatten().tolist()
        name = interface.global_name(sprite_name, input_name(index))
        for position in range(max(1, -(-len(flat) // SCRATCH_LIST_LIMIT))):
            values[interface.shard_name(name, position)] = (
                flat[position * SCRATCH_LIST_LIMIT:(position + 1) * SCRATCH_LIST_LIMIT]
            )
    return values


def _calls(sprite_name: str, actions) -> dict:
    """The stage statements each ``(label, block)`` runs: a ``broadcast and wait`` of its message."""
    return {
        label: (broadcast_and_wait(interface.message(sprite_name, block)),)
        for label, block in actions
    }


def _copy_list(source: str, target: str) -> tuple:
    return (
        clear(target),
        for_each(_COPY_INDEX, length(source), (append(target, item_of(source, var(_COPY_INDEX))),)),
    )


class _Project:
    """A benchmark project's stage, sprites and assets as they are added."""

    def __init__(self) -> None:
        self.stage, self.stage_md5ext, self.stage_bytes = _stage_target({}, {}, {})
        self.sprites: list[dict] = []
        self.assets: dict[str, bytes] = {}

    def add(self, sprite: dict, assets: dict[str, bytes], values=None) -> None:
        """Add an exported sprite, declaring its globals on the stage, hidden."""
        install(self.stage, sprite, values)
        sprite["visible"] = False
        sprite["layerOrder"] = len(self.sprites) + 1
        self.sprites.append(sprite)
        self.assets.update(assets)

    def stage_list(self, name: str, values=()) -> str:
        identifier = f"cattorch_benchmark_list_{len(self.stage['lists'])}"
        self.stage["lists"][identifier] = [name, list(values)]
        return identifier

    def finish(self, statements, variables=None) -> None:
        """Give the stage its green-flag script and its own variables."""
        self.stage["variables"].update(variables or {})
        self.stage["variables"]["cattorch_benchmark_copy_index"] = [_COPY_INDEX, 0]
        self.stage["blocks"] = _stage_blocks(
            statements,
            {name: identifier for identifier, name in self.stage["broadcasts"].items()},
            {entry[0]: identifier for identifier, entry in self.stage["lists"].items()},
            {entry[0]: identifier for identifier, entry in self.stage["variables"].items()},
        )

    def write(self, output_path: Path, description: str, monitors=None) -> Path:
        return _write_project(
            output_path, self.stage, self.sprites, self.assets,
            (self.stage_md5ext, self.stage_bytes), description, monitors=monitors,
        )


def _stage_blocks(
    statements,
    broadcasts: dict[str, str],
    lists: dict[str, str],
    variables: dict[str, str] | None = None,
) -> dict:
    """Blocks for a green-flag stage script, with the stage's data IDs.

    ``broadcasts`` maps each message to its ID, and ``lists`` and
    ``variables`` map each name to the ID the stage declares it with.
    """
    variables = variables or {}
    data = Program(
        "benchmark stage", variables=tuple(variables), lists=tuple(lists),
        body=tuple(statements), optimize_loops=False, when_flag_clicked=True,
    ).compile()
    mapping = {
        **{f"cattorch_dsl_list_{name}": identifier for name, identifier in lists.items()},
        **{f"cattorch_dsl_var_{name}": identifier for name, identifier in variables.items()},
        **{
            identifier: broadcasts[message]
            for identifier, message in data["broadcasts"].items()
        },
    }
    return remap_ids(data["blocks"], mapping)


def _timed(label: str) -> object:
    """``label: <seconds on the timer>``, for a results list."""
    return join(f"{label}: ", timer())


def _paired_statements(warmups: int, repeats: int) -> list:
    """Alternate the exact and fast sprites, recording each forward's time."""
    exact = broadcast_and_wait(interface.message(_PAIRED_NAMES["exact"], interface.FORWARD))
    fast = broadcast_and_wait(interface.message(_PAIRED_NAMES["fast"], interface.FORWARD))
    statements = [
        clear(EXACT_TIMINGS), clear(FAST_TIMINGS),
        repeat(warmups, (exact, fast)),
    ]
    for index in range(repeats):
        order = (
            ((exact, EXACT_TIMINGS), (fast, FAST_TIMINGS)) if index % 2 == 0
            else ((fast, FAST_TIMINGS), (exact, EXACT_TIMINGS))
        )
        for call, timings in order:
            statements += [reset_timer(), call, append(timings, timer())]
    return statements


def _suite_statements(cases: list[dict], repeats: int, warmups: int, variants) -> list:
    """Initialize each case's variants, then time ``repeats`` forwards of each."""
    statements = [clear(SUITE_RESULTS)]
    for case in cases:
        for variant in variants:
            calls = case[variant]
            statements += calls["init"]
            if warmups:
                statements.append(repeat(warmups, calls["forward"]))
            statements += [
                reset_timer(),
                repeat(repeats, calls["forward"]),
                append(SUITE_RESULTS, _timed(f"{case['name']} {variant}")),
            ]
    return statements


def _storage_total(case: dict, variant: str) -> str:
    return f"_cattorch storage total {case['index']} {variant}"


def _storage_statements(cases: list[dict], repeats: int, warmups: int) -> list:
    """Time decoding the weights, from saved storage, and then forwards."""
    statements = [clear(SUITE_RESULTS)]
    for case in cases:
        for variant in STORAGE_VARIANTS:
            calls = case[variant]
            total = _storage_total(case, variant)
            statements += [
                set_var(total, 0),
                repeat(repeats, (
                    *calls["prepare"],
                    reset_timer(),
                    *calls["init"],
                    change_var(total, timer()),
                )),
                append(SUITE_RESULTS, join(f"{case['name']} {variant} init: ", var(total))),
                *calls["restore"],
                *((repeat(warmups, calls["forward"]),) if warmups else ()),
                reset_timer(),
                repeat(repeats, calls["forward"]),
                append(SUITE_RESULTS, _timed(f"{case['name']} {variant} forward")),
            ]
    return statements


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
    project_json = write_archive(output_path, project, assets, PROJECT_JSON)
    size_warning = online_json_size_warning(
        len(project_json.encode("utf-8")), f"{description} project JSON",
    )
    if size_warning is not None:
        warn(size_warning)
    return output_path


def _normalise_inputs(
    model, example_inputs: torch.Tensor | tuple[torch.Tensor, ...], where: str = "",
) -> tuple[torch.Tensor, ...]:
    """A case's ``example_inputs`` as a tuple of tensors, checked before any export.

    A benchmark fills the sprite's input lists with them, so they must be
    tensors: not ``None`` (a model's markers) or a program.
    """
    try:
        if example_inputs is None:
            raise TypeError(
                "example_inputs must be a tensor or a tuple of tensors, which a benchmark "
                "fills the sprite's inputs with, not None"
            )
        validate_model_and_inputs(model, example_inputs, programs=False)
        inputs = (
            (example_inputs,) if isinstance(example_inputs, torch.Tensor)
            else tuple(example_inputs)
        )
        if not inputs or any(not isinstance(value, torch.Tensor) for value in inputs):
            raise TypeError("example_inputs must be a tensor or a non-empty tuple of tensors")
    except TypeError as error:
        raise TypeError(f"{where}{error}") from None
    return inputs


def _export_options(
    *, fast_config=None, storage=None, quantization=None, calibration=None, sig_figs=None,
    codegen=None, adapters=(), kernels=(),
) -> dict:
    """The ``transpile`` options every benchmark builder takes, checked once.

    Every variant of every case is exported with them, so ``calibration``
    must be a sequence that can be read again, such as a list.
    """
    validate_export_options(
        storage=storage, quantization=quantization, sig_figs=sig_figs, codegen=codegen,
        adapters=adapters, calibration=calibration,
    )
    if fast_config is not None and not isinstance(fast_config, FastConfig):
        raise TypeError("fast_config must be a FastConfig")
    if calibration is not None and iter(calibration) is calibration:
        raise TypeError(
            "a benchmark exports each case more than once, so calibration= must be a "
            "list or another sequence that can be read again, not an iterator"
        )
    return dict(
        fast_config=fast_config, storage=storage, quantization=quantization,
        calibration=calibration, sig_figs=sig_figs, codegen=codegen,
        adapters=tuple(adapters), kernels=tuple(kernels),
    )


def _check_counts(repeats: int, warmups: int) -> None:
    for label, value, least in (("repeats", repeats, 1), ("warmups", warmups, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < least:
            raise ValueError(
                f"{label} must be {'a positive' if least else 'a non-negative'} integer"
            )


def _export_variant(
    model, example_inputs, base: Path, *, name: str, optimization: str, options: dict,
    extra_kernels=(),
) -> tuple[dict, dict[str, bytes]]:
    """Export one variant of a case as the sprite ``name`` and load it.

    The sprite's messages and globals are named after it, so the project
    keeps the name.
    """
    options = dict(options)
    if optimization != "fast":
        options.pop("fast_config")
    options["kernels"] = (*options["kernels"], *extra_kernels)
    transpile(model, example_inputs, str(base), name=name, optimization=optimization, **options)
    return _load_sprite(base.with_suffix(".sprite3"))


def _stage_settings(repeats: int, warmups: int, comparison: str | None = None) -> dict:
    """The stage's public variables saying how the benchmark ran."""
    settings = {
        "cattorch_benchmark_repeats": [REPEATS, repeats],
        "cattorch_benchmark_warmups": [WARMUPS, warmups],
    }
    if comparison is not None:
        settings["cattorch_benchmark_comparison"] = [COMPARISON, comparison]
    return settings


def _results_monitor(identifier: str, name: str, values: list, *, height=330, x=10) -> dict:
    return {
        "id": identifier, "mode": "list", "opcode": "data_listcontents",
        "params": {"LIST": name}, "spriteName": None, "value": values,
        "width": 460, "height": height, "x": x, "y": 10, "visible": True,
    }


def build_benchmark_suite(
    cases,
    output_path: str | Path,
    *,
    repeats: int = 100,
    warmups: int = 0,
    fast_config: FastConfig | None = None,
    storage: StorageConfig | None = None,
    quantization: QuantizationConfig | None = None,
    calibration=None,
    sig_figs: int | None = None,
    codegen: CodegenConfig | None = None,
    adapters=(),
    kernels=(),
) -> Path:
    """Bundle exact-versus-fast benchmarks of several models into one Scratch project.

    ``cases`` contains ``(name, model, example_inputs)`` tuples; an optional
    fourth item is that case's ``FastConfig``, replacing ``fast_config``.
    Each case is exported twice, with ``optimization="exact"`` and
    ``"fast"``, and the other options are ``transpile``'s. On the green
    flag each variant is initialized, run ``warmups`` times, then timed over
    ``repeats`` forwards, and recorded as ``"<name> <variant>: <seconds>"``
    in the ``cattorch benchmark results`` stage list. Timing happens entirely
    in stock Scratch; read the results with ``analyze_benchmark``.
    """
    options = _export_options(
        fast_config=fast_config, storage=storage, quantization=quantization,
        calibration=calibration, sig_figs=sig_figs, codegen=codegen, adapters=adapters,
        kernels=kernels,
    )
    return _build_suite(
        cases, output_path, (("exact", "exact", ()), ("fast", "fast", ())),
        repeats=repeats, warmups=warmups, options=options, per_case_fast_config=True,
    )


def build_kernel_benchmark(
    kernel,
    cases,
    output_path: str | Path,
    *,
    repeats: int = 100,
    warmups: int = 0,
    fast_config: FastConfig | None = None,
    storage: StorageConfig | None = None,
    quantization: QuantizationConfig | None = None,
    calibration=None,
    sig_figs: int | None = None,
    codegen: CodegenConfig | None = None,
    adapters=(),
    kernels=(),
) -> Path:
    """Time a kernel against cattorch's own lowering of the same operation.

    ``cases`` contains ``(name, model, example_inputs)`` tuples whose models
    call the kernel's operation. Each runs once as cattorch would export it
    and once with ``kernel`` added to ``kernels``, and the project reports
    both as ``"<name> builtin"`` and ``"<name> kernel"``, timed as
    ``build_benchmark_suite`` times them. An approximate kernel, or a
    ``fast_config``, compares them in fast mode. Read the timings with
    ``analyze_benchmark``.
    """
    options = _export_options(
        fast_config=fast_config, storage=storage, quantization=quantization,
        calibration=calibration, sig_figs=sig_figs, codegen=codegen, adapters=adapters,
        kernels=kernels,
    )
    optimization = "fast" if kernel.approximate or fast_config is not None else "exact"
    return _build_suite(
        cases, output_path,
        (("builtin", optimization, ()), ("kernel", optimization, (kernel,))),
        repeats=repeats, warmups=warmups, options=options, per_case_fast_config=False,
    )


def _build_suite(
    cases, output_path, variants, *, repeats, warmups, options, per_case_fast_config,
) -> Path:
    _check_counts(repeats, warmups)
    cases = list(cases)
    if not cases:
        raise ValueError("at least one benchmark case is required")
    sizes = {3, 4} if per_case_fast_config else {3}
    normalized_cases = []
    for case in cases:
        if not isinstance(case, tuple) or len(case) not in sizes:
            raise ValueError(
                "benchmark cases are (name, model, example_inputs)"
                + (" or (name, model, example_inputs, fast_config)" if per_case_fast_config else "")
                + " tuples"
            )
        name, model, raw_inputs = case[:3]
        case_options = dict(options)
        if len(case) == 4:
            if case[3] is not None and not isinstance(case[3], FastConfig):
                raise TypeError("a case's fourth item must be a FastConfig")
            case_options["fast_config"] = case[3]
        normalized_cases.append((
            name, model, _normalise_inputs(model, raw_inputs, f"case {name!r}: "), case_options,
        ))
    _check_names([case[0] for case in normalized_cases])
    output_path = Path(output_file(output_path, ".sb3"))

    project = _Project()
    controller_cases: list[dict] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        for index, (name, model, example_inputs, case_options) in enumerate(normalized_cases):
            controller_case = {"name": name}
            for variant, optimization, extra_kernels in variants:
                sprite_name = f"{name} {variant}"
                sprite, sprite_assets = _export_variant(
                    model, example_inputs, temp / f"case_{index}_{variant}", name=sprite_name,
                    optimization=optimization, options=case_options,
                    extra_kernels=extra_kernels,
                )
                project.add(sprite, sprite_assets, _input_values(sprite_name, example_inputs))
                controller_case[variant] = _calls(
                    sprite_name, (("init", interface.INIT), ("forward", interface.FORWARD)),
                )
            controller_cases.append(controller_case)

    results_id = project.stage_list(SUITE_RESULTS)
    project.finish(
        _suite_statements(
            controller_cases, repeats, warmups, tuple(variant for variant, _, _ in variants),
        ),
        _stage_settings(repeats, warmups, variants[-1][0]),
    )
    return project.write(
        output_path, "Benchmark suite", monitors=[_results_monitor(results_id, SUITE_RESULTS, [])],
    )


def _check_names(names: list) -> None:
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("benchmark case names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("benchmark case names must be unique")
    for name in names:
        # Each case's sprites are named "NAME VARIANT", after which their
        # globals and messages are named.
        interface.check_export_name(name)


def _generation_baseline(model, program, tokens: torch.Tensor):
    """The stateless model a generation case's cached decode is compared with.

    It computes the logits of every token from all of them, as an export of
    the model without a cache does for each new token.
    """
    from cattorch.autoregressive import MethodLeaves

    if isinstance(program, GenerationProgram):
        from cattorch.markers import generation_method

        method = program.method or generation_method(model) or "forward"
        return model if method == "forward" else MethodLeaves(model, method)
    if program.reference is not None:
        return MethodLeaves(model, program.reference)
    if program.states is None:
        from cattorch.huggingface_recurrent import adapt_recurrent_model

        adapted = adapt_recurrent_model(model)
        if adapted is not None:
            return MethodLeaves(adapted.module, adapted.reference)
    raise ValueError(
        "a RecurrentGenerationProgram case needs reference=, the stateless "
        "method(tokens) -> logits its cached decode is compared with"
    )


def build_generation_benchmark_suite(
    cases,
    output_path: str | Path,
    *,
    repeats: int = 16,
    warmups: int = 0,
    fast_config: FastConfig | None = None,
    storage: StorageConfig | None = None,
    quantization: QuantizationConfig | None = None,
    calibration=None,
    sig_figs: int | None = None,
    codegen: CodegenConfig | None = None,
    adapters=(),
    kernels=(),
) -> Path:
    """Time text generators' cached decode against a stateless forward.

    Each case is ``(name, model, program, prompt)``: ``program`` is the
    ``GenerationProgram`` or ``RecurrentGenerationProgram`` to export the
    model with, as ``transpile`` takes it, and ``prompt`` a ``[1, length]``
    token tensor. The ``cached`` variant prefills the prompt outside the
    timer, then decodes ``warmups`` and ``repeats`` tokens; the program's
    ``max_context`` must hold them all. The ``stateless`` variant exports the
    model's method (for a recurrent program, its ``reference``) on the prompt
    plus one token and times ``repeats`` forwards, the work each new token
    costs without a cache. The other options are ``transpile``'s, with
    ``optimization="fast"`` when ``fast_config`` is given. Read the results
    with ``analyze_benchmark``.
    """
    _check_counts(repeats, warmups)
    options = _export_options(
        fast_config=fast_config, storage=storage, quantization=quantization,
        calibration=calibration, sig_figs=sig_figs, codegen=codegen, adapters=adapters,
        kernels=kernels,
    )
    cases = list(cases)
    if not cases:
        raise ValueError("at least one generation benchmark case is required")
    for case in cases:
        if not isinstance(case, tuple) or len(case) != 4:
            raise ValueError("generation benchmark cases are (name, model, program, prompt) tuples")
        name, _, program, prompt = case
        if not isinstance(program, (GenerationProgram, RecurrentGenerationProgram)):
            raise TypeError(
                f"case {name!r}: the program must be a GenerationProgram or "
                f"RecurrentGenerationProgram, not a {type(program).__name__}"
            )
        if not isinstance(prompt, torch.Tensor) or prompt.ndim != 2 or prompt.shape[0] != 1:
            raise ValueError(f"case {name!r}: the prompt must be a [1, length] token tensor")
        needed = prompt.shape[1] + warmups + repeats
        if program.max_context is not None and program.max_context < needed:
            raise ValueError(
                f"case {name!r}: max_context={program.max_context} can't hold the "
                f"{prompt.shape[1]}-token prompt and {warmups + repeats} timed decodes; "
                f"use max_context={needed} or more"
            )
    _check_names([case[0] for case in cases])
    output_path = Path(output_file(output_path, ".sb3"))
    project = _Project()
    controller_cases = []
    optimization = "fast" if fast_config is not None else "exact"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        for index, (name, model, program, prompt) in enumerate(cases):
            prompt = prompt.long()
            token = int(prompt[0, -1])
            tokens = torch.cat((prompt, prompt[:, -1:]), dim=1)
            controller = {"name": name}
            cached_name, stateless_name = f"{name} cached", f"{name} stateless"
            # The program first: a model it can't export fails with the
            # export's own message.
            cached, cached_assets = _export_variant(
                model, program, temp / f"generation_{index}_cached", name=cached_name,
                optimization=optimization, options=options,
            )
            stateless, stateless_assets = _export_variant(
                _generation_baseline(model, program, tokens), (tokens,),
                temp / f"generation_{index}_stateless", name=stateless_name,
                optimization=optimization, options=options,
            )
            project.add(stateless, stateless_assets, _input_values(stateless_name, (tokens,)))
            controller["stateless"] = _calls(
                stateless_name, (("init", interface.INIT), ("forward", interface.FORWARD)),
            )
            project.add(cached, cached_assets)
            ids = interface.global_name(cached_name, TOKEN_IDS)
            saved = _PROMPT.format(index)
            project.stage_list(saved, prompt.flatten().tolist())
            # The prefill, which decodes the weights, is outside the timer.
            controller["cached"] = {
                "init": (
                    *_copy_list(saved, ids),
                    broadcast_and_wait(interface.message(cached_name, interface.PREFILL)),
                ),
                "forward": (
                    clear(ids), append(ids, token),
                    broadcast_and_wait(interface.message(cached_name, interface.DECODE)),
                ),
            }
            controller_cases.append(controller)

    results_id = project.stage_list(SUITE_RESULTS)
    project.finish(
        _suite_statements(controller_cases, repeats, warmups, ("stateless", "cached")),
        _stage_settings(repeats, warmups, "cached"),
    )
    return project.write(
        output_path, "Generation benchmark suite",
        monitors=[_results_monitor(results_id, SUITE_RESULTS, [])],
    )


def build_storage_benchmark_suite(
    cases,
    output_path: str | Path,
    *,
    repeats: int = 10,
    warmups: int = 0,
    fast_config: FastConfig | None = None,
    sig_figs: int | None = None,
    codegen: CodegenConfig | None = None,
    adapters=(),
    kernels=(),
) -> Path:
    """Time decoding the weights and running the model in every storage mode.

    Each case is ``(name, model, example_inputs)``, exported once per mode in
    ``STORAGE_VARIANTS``; storage and quantization are what the suite
    compares, so it doesn't take them. The project times ``repeats`` weight
    decodes from saved storage, then ``warmups`` untimed and ``repeats``
    timed forwards, and lists each sprite's size in the
    ``cattorch storage sizes`` stage list. The other options are
    ``transpile``'s, with ``optimization="fast"`` when ``fast_config`` is
    given. Read the results with ``analyze_benchmark``.
    """
    _check_counts(repeats, warmups)
    options = _export_options(
        fast_config=fast_config, sig_figs=sig_figs, codegen=codegen, adapters=adapters,
        kernels=kernels,
    )
    optimization = "fast" if fast_config is not None else "exact"
    cases = list(cases)
    if not cases:
        raise ValueError("at least one storage benchmark case is required")
    if any(not isinstance(case, tuple) or len(case) != 3 for case in cases):
        raise ValueError("storage benchmark cases are (name, model, example_inputs) tuples")
    _check_names([case[0] for case in cases])
    cases = [
        (name, model, _normalise_inputs(model, raw_inputs, f"case {name!r}: "))
        for name, model, raw_inputs in cases
    ]

    output_path = Path(output_file(output_path, ".sb3"))
    variants = {
        "plain-f32": (StorageConfig(compression=False, precision="float32"), None),
        "base92-f32": (StorageConfig(compression=True, precision="float32"), None),
        "base92-f16": (StorageConfig(compression=True, precision="float16"), None),
        "base92-int8": (StorageConfig(compression=True), QuantizationConfig(bits=8)),
        "base92-int6": (StorageConfig(compression=True), QuantizationConfig(bits=6)),
        "base92-int4-huffman": (
            StorageConfig(compression=True), QuantizationConfig(bits=4),
        ),
    }
    project = _Project()
    controller_cases = []
    sizes = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        for index, (name, model, example_inputs) in enumerate(cases):
            controller = {"name": name, "index": index}
            for variant, (variant_storage, variant_quantization) in variants.items():
                base = temp / f"storage_{index}_{variant}"
                sprite_name = f"{name} {variant}"
                sprite, sprite_assets = _export_variant(
                    model, example_inputs, base, name=sprite_name, optimization=optimization,
                    options={
                        **options, "storage": variant_storage,
                        "quantization": variant_quantization,
                    },
                )
                sprite_path = base.with_suffix(".sprite3")
                sizes.append(f"{name} {variant}: {sprite_path.stat().st_size} bytes")
                values = _input_values(sprite_name, example_inputs)
                project.add(sprite, sprite_assets, values)
                # The inputs are copied back from saved lists before the
                # forwards, as they were when the sprite was exported.
                restore: list = []
                for physical, chunk in values.items():
                    saved = _SAVED_INPUT.format(physical)
                    project.stage_list(saved, chunk)
                    restore += _copy_list(saved, physical)
                controller[variant] = {
                    **_calls(sprite_name, (
                        ("prepare", interface.PREPARE_FOR_SAVE),
                        ("init", interface.INIT),
                        ("forward", interface.FORWARD),
                    )),
                    "restore": tuple(restore),
                }
            controller_cases.append(controller)

    results_id = project.stage_list(SUITE_RESULTS)
    sizes_id = project.stage_list(STORAGE_SIZES, sizes)
    totals = {
        _storage_total(case, variant): f"cattorch_storage_total_{case['index']}_{variant}"
        for case in controller_cases
        for variant in variants
    }
    stage_variables = {identifier: [name, 0] for name, identifier in totals.items()}
    stage_variables.update(_stage_settings(repeats, warmups))
    project.finish(_storage_statements(controller_cases, repeats, warmups), stage_variables)
    monitors = [
        _results_monitor(results_id, SUITE_RESULTS, [], height=250, x=10),
        _results_monitor(sizes_id, STORAGE_SIZES, sizes, height=250, x=480),
    ]
    return project.write(output_path, "Storage benchmark suite", monitors=monitors)


def build_paired_benchmark(
    model: torch.nn.Module,
    example_inputs: torch.Tensor | tuple[torch.Tensor, ...],
    output_path: str | Path,
    *,
    repeats: int = 7,
    warmups: int = 2,
    fast_config: FastConfig | None = None,
    storage: StorageConfig | None = None,
    quantization: QuantizationConfig | None = None,
    calibration=None,
    sig_figs: int | None = None,
    codegen: CodegenConfig | None = None,
    adapters=(),
    kernels=(),
) -> Path:
    """Create an exact-versus-fast project of one model, to run on scratch.mit.edu.

    The model is exported with ``optimization="exact"`` and ``"fast"``, and
    the other options are ``transpile``'s. On the green flag both run
    ``warmups`` times, then alternate for ``repeats`` timed forwards each,
    recorded one by one in the ``cattorch exact timings`` and
    ``cattorch fast timings`` stage lists. Read them with
    ``analyze_benchmark``, which reports the median and a bootstrap
    interval of the speedup.
    """
    _check_counts(repeats, warmups)
    options = _export_options(
        fast_config=fast_config, storage=storage, quantization=quantization,
        calibration=calibration, sig_figs=sig_figs, codegen=codegen, adapters=adapters,
        kernels=kernels,
    )
    example_inputs = _normalise_inputs(model, example_inputs)
    output_path = Path(output_file(output_path, ".sb3"))

    project = _Project()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        for variant in ("exact", "fast"):
            name = _PAIRED_NAMES[variant]
            sprite, assets = _export_variant(
                model, example_inputs, temp / variant, name=name, optimization=variant,
                options=options,
            )
            project.add(sprite, assets, _input_values(name, example_inputs))
    for list_name in (EXACT_TIMINGS, FAST_TIMINGS):
        project.stage_list(list_name)
    project.finish(_paired_statements(warmups, repeats), _stage_settings(repeats, warmups, "fast"))
    return project.write(output_path, "Paired benchmark")


@dataclass(frozen=True)
class BenchmarkCase:
    """One case of a benchmark project, its two variants compared.

    ``name`` is the case's name. ``baseline_seconds`` and
    ``candidate_seconds`` are the seconds one call of each variant took: the
    median over the runs of a paired benchmark, or a suite's timed total
    divided by its repeats. ``speedup`` is baseline over candidate, ``None``
    when the candidate's time is below the timer's resolution.
    ``baseline_blocks`` and ``candidate_blocks`` count each sprite's blocks.
    ``baseline_output_values`` and ``candidate_output_values`` count the
    values each left in its ``output`` list, ``output_length_match`` says
    whether they match (``None`` when neither ran), and
    ``output_max_abs_error`` is the largest difference between them. A
    paired benchmark also gives each run's seconds, ``baseline_timings`` and
    ``candidate_timings``, and the bootstrap 95% interval of the speedup.
    """

    name: str
    baseline_seconds: float
    candidate_seconds: float
    speedup: float | None
    baseline_blocks: int
    candidate_blocks: int
    baseline_output_values: int
    candidate_output_values: int
    output_length_match: bool | None
    output_max_abs_error: float | None
    baseline_timings: tuple[float, ...] = ()
    candidate_timings: tuple[float, ...] = ()
    speedup_95_percent_interval: tuple[float, float] | None = None


@dataclass(frozen=True)
class StorageBenchmarkVariant:
    """One storage mode of a storage benchmark case.

    ``init_seconds`` is the seconds one weight decode from saved storage
    took, ``forward_seconds`` one forward, and ``sprite_bytes`` the size of
    the exported ``.sprite3``.
    """

    init_seconds: float
    forward_seconds: float
    sprite_bytes: int


@dataclass(frozen=True)
class StorageBenchmarkCase:
    """One case of a storage benchmark: ``variants`` by storage mode name."""

    name: str
    variants: Mapping[str, StorageBenchmarkVariant]


@dataclass(frozen=True)
class BenchmarkAnalysis:
    """The results of a benchmark project saved after it ran in Scratch.

    ``kind`` is ``"paired"`` (``build_paired_benchmark``), ``"suite"``
    (``build_benchmark_suite``, ``build_kernel_benchmark`` and
    ``build_generation_benchmark_suite``) or ``"storage"``
    (``build_storage_benchmark_suite``). ``baseline`` and ``candidate`` name
    the variants compared, such as ``"exact"`` and ``"fast"`` (``None`` for
    storage), and ``cases`` holds one ``BenchmarkCase`` per case, while
    ``storage`` holds the storage benchmark's cases. ``repeats`` and
    ``warmups`` are the builder's (``None`` when the project doesn't record
    them), ``raw_results`` the results list as Scratch saved it, and
    ``project_bytes`` and ``project_sha256`` identify the file analyzed.
    """

    kind: str
    baseline: str | None
    candidate: str | None
    repeats: int | None
    warmups: int | None
    cases: tuple[BenchmarkCase, ...]
    storage: tuple[StorageBenchmarkCase, ...]
    raw_results: tuple[str, ...]
    project_bytes: int
    project_sha256: str


def _project_identity(path: str | Path) -> dict:
    return {
        "project_bytes": Path(path).stat().st_size,
        "project_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }


def _target_output(target: dict, stage: dict) -> list[float]:
    """The values a case's sprite left in its output: the global ``NAME output``.

    A project from before 1.0 has them in the sprite's own ``output`` list.
    """
    stage_lists = {entry[0]: entry[1] for entry in stage.get("lists", {}).values()}
    name = interface.global_name(target.get("name", ""), "output")
    if name in stage_lists:
        output = read_logical_list(stage_lists, name)
    else:
        output = read_logical_list(
            {entry[0]: entry[1] for entry in target.get("lists", {}).values()}, "output",
        )
    try:
        return [float(item) for item in output]
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"sprite {target.get('name')!r} has a non-numeric output value; "
            "save the project after the benchmark finishes"
        ) from error


def _setting(variables: dict, *names: str) -> int | None:
    """A stage setting, under its name or the name older projects used."""
    for name in names:
        if name in variables:
            try:
                return int(float(variables[name]))
            except (TypeError, ValueError):
                return None
    return None


def _compare_outputs(
    name, baseline, candidate, targets, stage, baseline_variant, candidate_variant,
):
    """The fields of a ``BenchmarkCase`` that compare the two sprites."""
    baseline_target = targets.get(f"{name} {baseline_variant}", {}) if name else targets.get(
        f"cattorch {baseline_variant}", {},
    )
    candidate_target = targets.get(f"{name} {candidate_variant}", {}) if name else targets.get(
        f"cattorch {candidate_variant}", {},
    )
    baseline_output = _target_output(baseline_target, stage)
    candidate_output = _target_output(candidate_target, stage)
    output_error = None
    output_length_match = None
    if baseline_output or candidate_output:
        output_length_match = len(baseline_output) == len(candidate_output)
    if baseline_output and len(baseline_output) == len(candidate_output):
        output_error = max(
            abs(left - right) for left, right in zip(baseline_output, candidate_output)
        )
    return dict(
        speedup=baseline / candidate if candidate > 0 else None,
        baseline_blocks=len(baseline_target.get("blocks", {})),
        candidate_blocks=len(candidate_target.get("blocks", {})),
        baseline_output_values=len(baseline_output),
        candidate_output_values=len(candidate_output),
        output_length_match=output_length_match,
        output_max_abs_error=output_error,
    )


_SUITE_PAIRS = (
    ("exact", "fast"), ("builtin", "kernel"), ("stateless", "cached"), ("legacy", "exact"),
)


def _analyze_suite(path, project: dict, stage: dict, entries: list) -> BenchmarkAnalysis:
    variables = {entry[0]: entry[1] for entry in stage.get("variables", {}).values()}
    repeats = _setting(variables, REPEATS, "cattorch benchmark iterations")
    parsed: dict[str, dict[str, float]] = {}
    known = {variant for pair in _SUITE_PAIRS for variant in pair}
    raw_results = tuple(str(item) for item in entries)
    for item in raw_results:
        try:
            label, raw_time = item.rsplit(": ", 1)
            name, variant = label.rsplit(" ", 1)
            elapsed = float(raw_time)
        except (ValueError, TypeError) as error:
            raise ValueError(f"malformed benchmark suite result: {item!r}") from error
        if variant not in known:
            raise ValueError(f"unknown benchmark suite variant in {item!r}")
        if variant in parsed.setdefault(name, {}):
            raise ValueError(f"duplicate benchmark suite result for {name} {variant}")
        parsed[name][variant] = elapsed

    found = set().union(*(set(values) for values in parsed.values()))
    baseline_variant, candidate_variant = next(
        (pair for pair in _SUITE_PAIRS if set(pair) <= found), _SUITE_PAIRS[0],
    )
    incomplete = [
        name for name, values in parsed.items()
        if set(values) != {baseline_variant, candidate_variant}
    ]
    if incomplete:
        raise ValueError(f"benchmark suite contains incomplete cases: {', '.join(incomplete)}")
    targets = {target["name"]: target for target in project["targets"] if not target["isStage"]}
    per_call = repeats or 1
    cases = []
    for name, values in parsed.items():
        baseline = values[baseline_variant] / per_call
        candidate = values[candidate_variant] / per_call
        cases.append(BenchmarkCase(
            name=name,
            baseline_seconds=baseline,
            candidate_seconds=candidate,
            **_compare_outputs(
                name, baseline, candidate, targets, stage, baseline_variant,
                candidate_variant,
            ),
        ))
    return BenchmarkAnalysis(
        kind="suite",
        baseline=baseline_variant,
        candidate=candidate_variant,
        repeats=repeats,
        warmups=_setting(variables, WARMUPS, "cattorch benchmark warmups"),
        cases=tuple(cases),
        storage=(),
        raw_results=raw_results,
        **_project_identity(path),
    )


def _analyze_storage(path, project: dict, stage: dict, by_name: dict) -> BenchmarkAnalysis:
    raw_results = tuple(str(entry) for entry in by_name.get(SUITE_RESULTS, []))
    if not raw_results:
        raise ValueError("storage benchmark suite has not run")
    variables = {entry[0]: entry[1] for entry in stage.get("variables", {}).values()}
    repeats = _setting(variables, REPEATS, "cattorch storage benchmark iterations")
    timings = {
        entry.rsplit(": ", 1)[0]: float(entry.rsplit(": ", 1)[1]) for entry in raw_results
    }
    sizes = {
        str(entry).rsplit(": ", 1)[0]: int(str(entry).rsplit(": ", 1)[1].removesuffix(" bytes"))
        for entry in by_name[STORAGE_SIZES]
    }
    case_names: list[str] = []
    for target in project["targets"]:
        if target["isStage"]:
            continue
        for variant in STORAGE_VARIANTS:
            if target["name"].endswith(f" {variant}"):
                name = target["name"][:-len(variant) - 1]
                if name not in case_names:
                    case_names.append(name)
    per_call = repeats or 1
    cases = []
    for name in case_names:
        variants = {}
        for variant in STORAGE_VARIANTS:
            keys = (f"{name} {variant} init", f"{name} {variant} forward")
            missing = next((key for key in keys if key not in timings), None)
            if missing is None and f"{name} {variant}" not in sizes:
                missing = f"{name} {variant}"
            if missing is not None:
                raise ValueError(f"storage benchmark suite has not finished: missing {missing!r}")
            variants[variant] = StorageBenchmarkVariant(
                init_seconds=timings[keys[0]] / per_call,
                forward_seconds=timings[keys[1]] / per_call,
                sprite_bytes=sizes[f"{name} {variant}"],
            )
        cases.append(StorageBenchmarkCase(name=name, variants=variants))
    return BenchmarkAnalysis(
        kind="storage",
        baseline=None,
        candidate=None,
        repeats=repeats,
        warmups=_setting(variables, WARMUPS),
        cases=(),
        storage=tuple(cases),
        raw_results=raw_results,
        **_project_identity(path),
    )


def _analyze_paired(path, project: dict, stage: dict, by_name: dict) -> BenchmarkAnalysis:
    exact = [float(item) for item in by_name.get(EXACT_TIMINGS, [])]
    fast = [float(item) for item in by_name.get(FAST_TIMINGS, [])]
    legacy = [float(item) for item in by_name.get(LEGACY_TIMINGS, [])]
    if fast:
        baseline_name, candidate_name = "exact", "fast"
        baseline, candidate = exact, fast
    else:
        # Historical pre-0.4 legacy/exact projects still analyze, though
        # cattorch no longer generates the retired template backend.
        baseline_name, candidate_name = "legacy", "exact"
        baseline, candidate = legacy, exact
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("benchmark has not run or contains incomplete timing lists")
    baseline_median = statistics.median(baseline)
    candidate_median = statistics.median(candidate)
    rng = random.Random(0)
    bootstrapped = []
    for _ in range(10_000):
        indices = [rng.randrange(len(baseline)) for _ in baseline]
        sampled_candidate_median = statistics.median(candidate[index] for index in indices)
        if sampled_candidate_median > 0:
            bootstrapped.append(
                statistics.median(baseline[index] for index in indices)
                / sampled_candidate_median
            )
    bootstrapped.sort()
    interval = None
    if candidate_median > 0 and bootstrapped:
        interval = (
            bootstrapped[int(len(bootstrapped) * 0.025)],
            bootstrapped[int(len(bootstrapped) * 0.975)],
        )
    targets = {target["name"]: target for target in project["targets"] if not target["isStage"]}
    variables = {entry[0]: entry[1] for entry in stage.get("variables", {}).values()}
    case = BenchmarkCase(
        name="",
        baseline_seconds=baseline_median,
        candidate_seconds=candidate_median,
        baseline_timings=tuple(baseline),
        candidate_timings=tuple(candidate),
        speedup_95_percent_interval=interval,
        **_compare_outputs(
            "", baseline_median, candidate_median, targets, stage, baseline_name,
            candidate_name,
        ),
    )
    return BenchmarkAnalysis(
        kind="paired",
        baseline=baseline_name,
        candidate=candidate_name,
        repeats=_setting(variables, REPEATS) or len(baseline),
        warmups=_setting(variables, WARMUPS),
        cases=(case,),
        storage=(),
        raw_results=(),
        **_project_identity(path),
    )


def analyze_benchmark(path: str | Path) -> BenchmarkAnalysis:
    """Read the results of a benchmark project saved after it ran in Scratch.

    Takes a project from any of the benchmark builders; see
    ``BenchmarkAnalysis``. Raises ``ValueError`` for a project that hasn't
    run, or ran only part of the way.
    """
    project = read_user_archive(path, PROJECT_JSON, parameter="path")
    stage = next(
        (target for target in project.get("targets", ()) if target.get("isStage")), {},
    )
    by_name = {entry[0]: entry[1] for entry in stage.get("lists", {}).values()}
    if not by_name.keys() & {
        STORAGE_SIZES, SUITE_RESULTS, EXACT_TIMINGS, FAST_TIMINGS, LEGACY_TIMINGS,
    }:
        raise ValueError(f"path {str(path)!r} is not a cattorch benchmark project")
    if STORAGE_SIZES in by_name:
        return _analyze_storage(path, project, stage, by_name)
    if SUITE_RESULTS in by_name:
        if not by_name[SUITE_RESULTS]:
            raise ValueError("benchmark suite has not run")
        return _analyze_suite(path, project, stage, by_name[SUITE_RESULTS])
    return _analyze_paired(path, project, stage, by_name)
