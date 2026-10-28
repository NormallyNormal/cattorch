"""Assemble exported sprites into a complete, runnable Scratch project.

``build_project`` takes the results of ``transpile`` and ``transpile_tokenizer``
and writes an ``.sb3`` with the sprites wired together. Each model or
tokenizer sprite gets bridge scripts: a ``when I receive`` script per
operation that copies global (stage) lists into the sprite's own lists, runs
the operation, and copies the results back, so any sprite can drive it with
``broadcast and wait``. A demo on top of the bridges makes the project
runnable on its own: list monitors and a green-flag run for a model, or a
prompt-and-continue loop for a text generator with its tokenizer.
"""

from __future__ import annotations

import hashlib
import json
import uuid
import warnings
import zipfile
from pathlib import Path
from typing import Any

import torch

from cattorch.results import (
    BoundedDimension, ProgramResult, ProjectResult, TokenizerResult, TranspileResult,
)
from cattorch.sprite import _add_warp_procedure, _merge_lists_by_name
from cattorch.templates.template import TEMPLATE_DIR
from cattorch.util.scratch.dsl import (
    Program,
    add,
    answer,
    append,
    ask,
    broadcast_and_wait,
    call,
    change_var,
    clear,
    div,
    eq,
    for_each,
    gt,
    if_,
    item,
    join,
    length,
    mathop,
    mul,
    not_,
    random,
    repeat_until,
    say,
    set_var,
    sub,
    var,
)
from cattorch.util.scratch.finalize_scratch import online_json_size_warning
from cattorch.util.scratch.remap import remap_ids


_INDEX = "_cattorch bridge index"
_BEST = "_cattorch bridge best"
# Set once the generation bridge has decoded the model's weights.
_READY = "_cattorch bridge ready"
_TOTAL = "_cattorch bridge total"
_PICK = "_cattorch bridge pick"
_BACKDROP = (
    b'<svg xmlns="http://www.w3.org/2000/svg" width="480" height="360" viewBox="0 0 480 360">'
    b'<rect width="480" height="360" fill="#ffffff"/></svg>'
)


def _load_sprite(path: Path) -> tuple[dict, dict[str, bytes]]:
    with zipfile.ZipFile(path) as archive:
        sprite = json.loads(archive.read("sprite.json"))
        assets = {name: archive.read(name) for name in archive.namelist() if name != "sprite.json"}
    return sprite, assets


def _copy(source: str, target: str) -> tuple:
    return (
        clear(target),
        for_each(_INDEX, length(source), (append(target, item(source, var(_INDEX))),)),
    )


def _argmax(values: str) -> tuple:
    """Set ``_BEST`` to the one-based position of the largest item."""
    return (
        set_var(_BEST, 1),
        for_each(_INDEX, length(values), (
            if_(gt(item(values, var(_INDEX)), item(values, var(_BEST))),
                (set_var(_BEST, var(_INDEX)),)),
        )),
    )


def _sample(logits: str, temperature: str) -> tuple:
    """Set ``_BEST`` to a position drawn from ``softmax(logits / temperature)``.

    A temperature of 0 or less picks the largest logit.
    """
    def weight(position):
        return mathop("e ^", div(
            sub(item(logits, position), var(_PICK)), var(temperature),
        ))

    return (
        *_argmax(logits),
        if_(gt(var(temperature), 0), (
            set_var(_PICK, item(logits, var(_BEST))),
            set_var(_TOTAL, 0),
            for_each(_INDEX, length(logits), (change_var(_TOTAL, weight(var(_INDEX))),)),
            # ``pick random`` with whole-number bounds returns whole numbers.
            set_var(_TOTAL, mul(var(_TOTAL), div(random(1, 1000000), 1000000))),
            # Walk the cumulative weights to the drawn point. The last
            # position catches rounding that leaves a little weight over.
            set_var(_BEST, length(logits)),
            for_each(_INDEX, length(logits), (
                if_(gt(var(_TOTAL), 0), (
                    change_var(_TOTAL, mul(-1, weight(var(_INDEX)))),
                    if_(not_(gt(var(_TOTAL), 0)), (set_var(_BEST, var(_INDEX)),)),
                )),
            )),
        )),
    )


class _Builder:
    def __init__(self) -> None:
        backdrop = hashlib.md5(_BACKDROP).hexdigest()
        self.assets: dict[str, bytes] = {f"{backdrop}.svg": _BACKDROP}
        self.stage: dict[str, Any] = {
            "isStage": True, "name": "Stage", "variables": {}, "lists": {},
            "broadcasts": {}, "blocks": {}, "comments": {}, "currentCostume": 0,
            "costumes": [{
                "name": "backdrop1", "bitmapResolution": 1, "dataFormat": "svg",
                "assetId": backdrop, "md5ext": f"{backdrop}.svg",
                "rotationCenterX": 240, "rotationCenterY": 180,
            }],
            "sounds": [], "volume": 100, "layerOrder": 0, "tempo": 60,
            "videoTransparency": 50, "videoState": "on", "textToSpeechLanguage": None,
        }
        self.sprites: list[dict] = []
        self.monitors: list[dict] = []
        self.messages: list[str] = []
        self.global_lists: dict[str, list] = {}
        self.global_variables: dict[str, object] = {}

    # ── targets ────────────────────────────────────────────────────────────

    def add_sprite(self, path: Path, *, visible: bool) -> dict:
        sprite, assets = _load_sprite(path)
        names = {target["name"] for target in self.sprites}
        base, index = sprite["name"], 2
        while sprite["name"] in names or sprite["name"] == "Stage":
            sprite["name"] = f"{base}{index}"
            index += 1
        sprite.update(visible=visible, layerOrder=len(self.sprites) + 1)
        self.sprites.append(sprite)
        self.assets.update(assets)
        return sprite

    def add_demo_sprite(self, name: str) -> dict:
        costume = next((TEMPLATE_DIR / "sprite").glob("*.svg"))
        data = costume.read_bytes()
        asset = hashlib.md5(data).hexdigest()
        self.assets[f"{asset}.svg"] = data
        sprite = {
            "isStage": False, "name": name, "variables": {}, "lists": {}, "broadcasts": {},
            "blocks": {}, "comments": {}, "currentCostume": 0,
            "costumes": [{
                "name": "costume1", "bitmapResolution": 1, "dataFormat": "svg",
                "assetId": asset, "md5ext": f"{asset}.svg",
                "rotationCenterX": 48, "rotationCenterY": 50,
            }],
            "sounds": [], "volume": 100, "layerOrder": len(self.sprites) + 1,
            "visible": True, "x": -150, "y": -100, "size": 100, "direction": 90,
            "draggable": False, "rotationStyle": "all around",
        }
        self.sprites.append(sprite)
        return sprite

    # ── global data ────────────────────────────────────────────────────────

    def global_list(self, name: str, values=()) -> str:
        if name not in self.global_lists:
            self.global_lists[name] = list(values)
        return name

    def global_variable(self, name: str, value: object = 0) -> str:
        if name not in self.global_variables:
            self.global_variables[name] = value
        return name

    def _bind(self, target: dict, data: dict, section: str) -> None:
        """Move entries named as globals to the stage; unify the rest by name."""
        globals_ = self.global_lists if section == "lists" else self.global_variables
        added = data.get(section, {})
        stage_ids = {
            entry[0]: sid for sid, entry in self.stage[section].items() if sid not in added
        }
        # The first existing entry of each name, before this script's.
        local_ids: dict[str, str] = {}
        for identifier, entry in target[section].items():
            if identifier not in added:
                local_ids.setdefault(entry[0], identifier)
        remap = {}
        for identifier, entry in added.items():
            name = entry[0]
            if name in globals_:
                winner = stage_ids.setdefault(name, identifier)
                if winner == identifier:
                    self.stage[section][identifier] = [name, globals_[name]]
                else:
                    remap[identifier] = winner
                if target is not self.stage or winner != identifier:
                    target[section].pop(identifier, None)
            elif name in local_ids:
                remap[identifier] = local_ids[name]
                del target[section][identifier]
            else:
                local_ids[name] = identifier
        if remap:
            target["blocks"] = remap_ids(target["blocks"], remap)

    def _merge(self, target: dict, data: dict) -> None:
        suffix = uuid.uuid4().hex[:8]
        mapping = {
            identifier: f"{identifier}_{suffix}"
            for section in ("blocks", "variables", "lists", "broadcasts")
            for identifier in data.get(section, {})
        }
        data = remap_ids(data, mapping)
        target["blocks"].update(data["blocks"])
        target.setdefault("variables", {}).update(data.get("variables", {}))
        target.setdefault("lists", {}).update(data.get("lists", {}))
        target.setdefault("broadcasts", {}).update(data.get("broadcasts", {}))
        for section in ("lists", "variables"):
            self._bind(target, data, section)

    # ── scripts ────────────────────────────────────────────────────────────

    def procedure(self, target: dict, name: str, body, *, lists=(), variables=()) -> None:
        """A warp procedure, so its loops don't wait for a screen refresh."""
        program = Program(
            name.replace(" ", "_"), lists=lists, variables=variables, body=body,
        )
        before = {section: set(target.get(section, {})) for section in ("lists", "variables")}
        _add_warp_procedure(target, name, program, x=-800, y=200 * len(self.messages))
        added = {
            section: {
                identifier: entry for identifier, entry in target[section].items()
                if identifier not in before[section]
            }
            for section in ("lists", "variables")
        }
        for section in ("lists", "variables"):
            self._bind(target, added, section)

    def script(self, target: dict, hat: dict, body, *, lists=(), variables=(), y=0) -> None:
        """A script under ``hat`` (a block without ``next`` or ``parent``)."""
        data = Program("script", lists=lists, variables=variables, body=body).compile()
        root = next(bid for bid, block in data["blocks"].items() if block.get("topLevel"))
        hat_id = f"cattorch_project_hat_{len(data['blocks'])}"
        data["blocks"][root].update(topLevel=False, parent=hat_id)
        data["blocks"][root].pop("x", None)
        data["blocks"][root].pop("y", None)
        data["blocks"][hat_id] = {
            "next": root, "parent": None, "inputs": {}, "shadow": False,
            "topLevel": True, "x": -400, "y": y, **hat,
        }
        self._merge(target, data)

    def receiver(self, target: dict, message: str, body, *, lists=(), variables=()) -> None:
        """Run ``body`` in a warp procedure whenever ``message`` is broadcast."""
        procedure = f"cattorch bridge {message}"
        self.procedure(target, procedure, body, lists=lists, variables=variables)
        self.script(target, {
            "opcode": "event_whenbroadcastreceived",
            "fields": {"BROADCAST_OPTION": [message, f"cattorch_message_{message}"]},
        }, (call(procedure),), y=200 * len(self.messages))
        self.messages.append(message)

    # ── output ─────────────────────────────────────────────────────────────

    def monitor(self, name: str, *, kind: str, x: int, y: int, **extra) -> None:
        section = "lists" if kind == "list" else "variables"
        identifier = next(
            sid for sid, entry in self.stage[section].items() if entry[0] == name
        )
        self.monitors.append({
            "id": identifier, "mode": kind,
            "opcode": "data_listcontents" if kind == "list" else "data_variable",
            "params": {"LIST" if kind == "list" else "VARIABLE": name},
            "spriteName": None,
            "value": self.global_lists[name] if kind == "list" else self.global_variables[name],
            "width": 0, "height": 0, "x": x, "y": y, "visible": True, **extra,
        })

    def note(self, text: str) -> None:
        self.stage["comments"]["cattorch_project_note"] = {
            "blockId": None, "x": -400, "y": -300, "width": 400, "height": 240,
            "minimized": False, "text": text,
        }

    def _unify_broadcasts(self) -> None:
        """Give each message one ID, declared on the stage as Scratch expects."""
        canonical: dict[str, tuple[str, str]] = {}

        def identify(message: str) -> str:
            key = message.lower()
            if key not in canonical:
                canonical[key] = (f"cattorch_message_{len(canonical) + 1}", message)
            return canonical[key][0]

        for target in (self.stage, *self.sprites):
            remap = {
                identifier: identify(message)
                for identifier, message in target.get("broadcasts", {}).items()
            }
            for block in target["blocks"].values():
                option = block.get("fields", {}).get("BROADCAST_OPTION")
                if option is not None:
                    option[1] = identify(option[0])
                menu = block.get("inputs", {}).get("BROADCAST_INPUT")
                if menu is not None and isinstance(menu[1], list) and menu[1][0] == 11:
                    menu[1][2] = identify(menu[1][1])
            if remap:
                target["blocks"] = remap_ids(target["blocks"], remap)
            target["broadcasts"] = {}
        self.stage["broadcasts"] = dict(canonical.values())

    def write(self, path: Path) -> list[str]:
        self._unify_broadcasts()
        for identifier, entry in self.stage["lists"].items():
            entry[1] = list(self.global_lists[entry[0]])
        for identifier, entry in self.stage["variables"].items():
            entry[1] = self.global_variables[entry[0]]
        project = {
            "targets": [self.stage, *self.sprites], "monitors": self.monitors,
            "extensions": [],
            "meta": {"semver": "3.0.0", "vm": "0.4.0-cattorch", "agent": "cattorch"},
        }
        text = json.dumps(project, separators=(",", ":"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("project.json", text)
            for name, data in self.assets.items():
                archive.writestr(name, data)
        warning = online_json_size_warning(len(text.encode("utf-8")), "project JSON")
        return [] if warning is None else [warning]


def _example_values(spec, example) -> list:
    if example is not None:
        return example.detach().flatten().tolist()
    count = 1
    for extent in spec.shape:
        count *= extent.minimum if isinstance(extent, BoundedDimension) else extent
    return [0] * count


def _bridge_operation(builder, sprite, message, procedure, inputs, outputs, examples, sharded):
    """Copy global inputs in, call ``procedure``, and copy outputs back out."""
    name = sprite["name"]
    lists, body = [], []
    for spec, example in zip(inputs, examples):
        if spec.list_name in sharded:
            raise ValueError(
                f"{spec.list_name!r} is split into several lists, which a bridge can't fill"
            )
        source = builder.global_list(f"{name} {spec.list_name}", _example_values(spec, example))
        lists += [source, spec.list_name]
        body += _copy(source, spec.list_name)
    body.append(call(procedure))
    for spec in outputs:
        target = builder.global_list(f"{name} {spec.list_name}")
        lists += [spec.list_name, target]
        body += _copy(spec.list_name, target)
    builder.receiver(sprite, message, body, lists=lists, variables=(_INDEX,))
    return [f"{name} {spec.list_name}" for spec in (*inputs, *outputs)]


def _layout(builder, names: list[str]) -> None:
    for position, list_name in enumerate(names):
        builder.monitor(list_name, kind="list", x=5 + 120 * (position % 4), y=5 + 180 * (position // 4))


def _forward_project(builder, result, example_inputs, labels) -> None:
    sprite = builder.add_sprite(result.path, visible=False)
    name = sprite["name"]
    message = f"{name} run"
    examples = list(example_inputs) + [None] * (len(result.inputs) - len(example_inputs))
    shown = _bridge_operation(
        builder, sprite, message, "cattorch forward", result.inputs, result.outputs,
        examples, result.sharded_lists,
    )
    _layout(builder, shown)
    if labels is not None:
        output = result.outputs[0]
        if output.numel is None:
            raise ValueError(
                f"labels needs an output of fixed size, but {output.list_name!r} has a "
                "variable-length dimension"
            )
        if output.numel != len(labels):
            raise ValueError(
                f"labels has {len(labels)} entries but output {output.list_name!r} "
                f"has {output.numel} values"
            )
        table = builder.global_list(f"{name} labels", [str(label) for label in labels])
        prediction = builder.global_variable(f"{name} prediction", "")
        builder.receiver(sprite, f"{name} predict", (
            *_argmax(output.list_name),
            set_var(prediction, item(table, var(_BEST))),
        ), lists=(output.list_name, table), variables=(_INDEX, _BEST, prediction))
        builder.monitor(prediction, kind="large", x=5, y=320)
    run: tuple = (broadcast_and_wait(message),)
    if labels is not None:
        run += (broadcast_and_wait(f"{name} predict"),)
    builder.script(builder.stage, {"opcode": "event_whenflagclicked", "fields": {}}, run)
    builder.script(builder.stage, {
        "opcode": "event_whenkeypressed", "fields": {"KEY_OPTION": ["space", None]},
    }, run, y=200)
    inputs = ", ".join(f"{name} {spec.list_name}" for spec in result.inputs)
    builder.note(
        f"Edit {inputs}, then click the green flag or press space to run {name}.\n\n"
        f"Other sprites can fill the same lists and broadcast \"{message}\" and wait; "
        f"the outputs are then in the {name} output lists."
    )


def _program_project(builder, result) -> None:
    sprite = builder.add_sprite(result.path, visible=False)
    name = sprite["name"]
    shown, lines = [], []
    for entry in result.entrypoints:
        message = f"{name} {entry.name}"
        shown += _bridge_operation(
            builder, sprite, message, entry.procedure, entry.inputs, entry.outputs,
            [None] * len(entry.inputs), result.sharded_lists,
        )
        lines.append(f"broadcast \"{message}\" and wait")
    _layout(builder, list(dict.fromkeys(shown)))
    builder.note(
        f"Fill an entry point's input lists, then:\n" + "\n".join(lines)
        + f"\n\nState is kept inside {name} between calls."
    )


def _generation_project(builder, result, tokenizer, *, max_new_tokens, temperature, eos_token_id,
                        prompt) -> None:
    model = builder.add_sprite(result.path, visible=False)
    words = builder.add_sprite(tokenizer.path, visible=False)
    name, tokenizer_name = model["name"], words["name"]
    tokens = builder.global_list(f"{name} tokens")
    status = builder.global_variable(f"{name} status", "")
    next_token = builder.global_variable(f"{name} next token", 0)
    heat = builder.global_variable(f"{name} temperature", float(temperature))
    for operation in ("prefill", "decode"):
        builder.receiver(model, f"{name} {operation}", (
            if_(eq(var(_READY), 0), (call("cattorch init"), set_var(_READY, 1))),
            *_copy(tokens, "cattorch tokens"),
            call(f"cattorch {operation}"),
            set_var(status, var("cattorch status")),
            if_(eq(var("cattorch status"), "ok"), (
                *_sample("cattorch logits", heat),
                set_var(next_token, sub(var(_BEST), 1)),
            )),
        ), lists=(tokens, "cattorch tokens", "cattorch logits"),
            variables=(
                _INDEX, _BEST, _TOTAL, _PICK, _READY, status, next_token, heat, "cattorch status",
            ))

    # Decode the weights again after a restart, in case they were re-encoded
    # for saving.
    builder.script(model, {"opcode": "event_whenflagclicked", "fields": {}},
                   (set_var(_READY, 0),), variables=(_READY,))
    text = builder.global_variable(f"{tokenizer_name} text", "")
    ids = builder.global_list(f"{tokenizer_name} token ids")
    builder.receiver(words, f"{tokenizer_name} tokenize", (
        set_var("input", var(text)),
        call("cattorch tokenize"),
        *_copy("token_ids", ids),
    ), lists=("token_ids", ids), variables=("input", text, _INDEX))
    builder.receiver(words, f"{tokenizer_name} detokenize", (
        *_copy(ids, "token_ids"),
        call("cattorch detokenize"),
        set_var(text, var("output")),
    ), lists=("token_ids", ids), variables=("output", text, _INDEX))

    demo = builder.add_demo_sprite("Demo")
    generated = "_cattorch demo generated"
    count, done, seed = "_cattorch demo count", "_cattorch demo done", "_cattorch demo prompt"
    builder.procedure(demo, "cattorch demo prompt tokens", _copy(ids, tokens),
                      lists=(ids, tokens), variables=(_INDEX,))
    builder.procedure(demo, "cattorch demo generated tokens", _copy(generated, ids),
                      lists=(generated, ids), variables=(_INDEX,))
    stop = -1 if eos_token_id is None else int(eos_token_id)
    check = (
        set_var(done, 0),
        if_(not_(eq(var(status), "ok")), (set_var(done, 1),)),
        if_(eq(var(count), max_new_tokens), (set_var(done, 1),)),
        if_(eq(var(next_token), stop), (set_var(done, 1),)),
    )
    question = prompt or "Type the start of some text (leave it empty to stop)"
    body = (
        ask(question),
        repeat_until(eq(answer(), ""), (
            set_var(seed, answer()),
            set_var(text, answer()),
            broadcast_and_wait(f"{tokenizer_name} tokenize"),
            call("cattorch demo prompt tokens"),
            say(join(var(seed), "...")),
            broadcast_and_wait(f"{name} prefill"),
            if_(not_(eq(var(status), "ok")), (say(var(status)),)),
            clear(generated),
            set_var(count, 0),
            *check,
            repeat_until(eq(var(done), 1), (
                append(generated, var(next_token)),
                call("cattorch demo generated tokens"),
                broadcast_and_wait(f"{tokenizer_name} detokenize"),
                say(join(var(seed), var(text))),
                clear(tokens),
                append(tokens, var(next_token)),
                broadcast_and_wait(f"{name} decode"),
                change_var(count, 1),
                *check,
            )),
            ask(question),
        )),
    )
    builder.script(demo, {"opcode": "event_whenflagclicked", "fields": {}}, body,
                   lists=(generated, tokens), variables=(count, done, seed, text, status,
                                                         next_token))
    builder.monitor(heat, kind="slider", x=5, y=5, sliderMin=0, sliderMax=2, isDiscrete=False)
    builder.note(
        f"Click the green flag and type a prompt. The Demo sprite continues it with "
        f"{name}, up to {max_new_tokens} tokens"
        + ("" if eos_token_id is None else f" or until token {eos_token_id}")
        + ". The temperature slider sets how random the choices are; 0 always picks "
        f"the likeliest token.\n\n"
        f"To drive the sprites yourself: set \"{tokenizer_name} text\" and broadcast "
        f"\"{tokenizer_name} tokenize\" to fill \"{tokenizer_name} token ids\"; fill "
        f"\"{name} tokens\" and broadcast \"{name} prefill\" (a prompt) or "
        f"\"{name} decode\" (one token) to get \"{name} next token\" and "
        f"\"{name} status\"; and broadcast \"{tokenizer_name} detokenize\" to turn "
        f"\"{tokenizer_name} token ids\" back into \"{tokenizer_name} text\"."
    )


def build_project(
    output_path: str | Path,
    model: TranspileResult | ProgramResult,
    *,
    tokenizer: TokenizerResult | None = None,
    example_inputs: torch.Tensor | tuple[torch.Tensor, ...] = (),
    labels=None,
    max_new_tokens: int = 40,
    temperature: float = 0.8,
    eos_token_id: int | None = None,
    prompt: str | None = None,
) -> ProjectResult:
    """Write a runnable Scratch project around exported sprites.

    ``model`` is a ``transpile`` result. For a ``GenerationProgram`` export,
    pass the ``tokenizer`` result too: the project asks for a prompt and
    continues it, sampling at ``temperature`` (0 picks the likeliest token)
    for up to ``max_new_tokens`` tokens or until ``eos_token_id``. ``prompt``
    replaces the question it asks. For any other model, the project shows the
    input and output lists, starting from ``example_inputs`` or zeros, and runs
    the model on the green flag or space key; ``labels`` names the outputs of a
    classifier and shows the predicted label. A model with entry points gets a
    bridge for each, with no demo.

    In every case each operation is also bridged to global lists, so other
    sprites can run it with ``broadcast and wait``; ``ProjectResult`` lists
    the messages and data.
    """
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".sb3":
        output_path = Path(f"{output_path}.sb3")
    if isinstance(example_inputs, torch.Tensor):
        example_inputs = (example_inputs,)
    builder = _Builder()
    if isinstance(model, ProgramResult):
        _program_project(builder, model)
    elif not isinstance(model, TranspileResult):
        raise TypeError("model must be the result of cattorch.transpile")
    elif "cattorch prefill" in model.procedures:
        if tokenizer is None:
            raise ValueError("a text generator's project needs its tokenizer= result")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        _generation_project(
            builder, model, tokenizer, max_new_tokens=max_new_tokens,
            temperature=temperature, eos_token_id=eos_token_id, prompt=prompt,
        )
    else:
        if tokenizer is not None:
            raise ValueError("tokenizer= applies only to a GenerationProgram export")
        _forward_project(builder, model, tuple(example_inputs), labels)
    project_warnings = builder.write(output_path)
    for message in project_warnings:
        warnings.warn(message, stacklevel=2)
    return ProjectResult(
        path=output_path,
        sprites=tuple(sprite["name"] for sprite in builder.sprites),
        messages=tuple(builder.messages),
        lists=tuple(builder.global_lists),
        variables=tuple(builder.global_variables),
        warnings=tuple(project_warnings),
    )
