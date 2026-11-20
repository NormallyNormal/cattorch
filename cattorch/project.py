"""Assemble exported sprites into a complete, runnable Scratch project.

``build_project`` takes the results of ``transpile`` and ``transpile_tokenizer``
and writes an ``.sb3`` with the sprites in it. An exported sprite already
has the interface a project uses: its messages (``NAME forward``, ...) and
its global lists and variables (``NAME input``, ``NAME status``, ...), which
the project declares on the stage, as uploading the sprite would, starting
from the example inputs. A small Demo sprite, the one selected when the
project opens (a model sprite is slow to show in the editor), makes the
project runnable on its own: list monitors and a green-flag run for a model,
or a prompt-and-continue loop for a text generator with its tokenizer, and
holds the project's own extras: a classifier's ``NAME predict`` and a
generator's ``NAME sample``.

``add_to_project`` puts exported sprites into a project of your own instead:
after its sprites, so that it opens with the sprite it opened with, their
globals and messages declared on its stage, and nothing of yours changed.
"""

from __future__ import annotations

import hashlib
import math
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from cattorch.errors import collect_warnings, warn
from cattorch.results import (
    BoundedDimension, GenerationResult, ProgramResult, ProjectResult, TokenizerResult,
    TranspileResult,
)
from cattorch.procedures import _add_warp_procedure
from cattorch.templates.template import TEMPLATE_DIR
from cattorch.util.scratch.archive import (
    PROJECT_JSON, SPRITE_JSON, read_archive, read_user_archive, write_archive,
)
from cattorch.util.paths import output_file
from cattorch.util.scratch.dsl import (
    Program,
    answer,
    append,
    ask,
    broadcast_and_wait,
    change_var,
    clear,
    div,
    eq,
    for_each,
    gt,
    if_,
    if_else,
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
from cattorch.util.scratch.globals import global_references, install
from cattorch.util.scratch.remap import remap_ids
from cattorch.util.scratch.sharding import SCRATCH_LIST_LIMIT
from cattorch import interface
from cattorch.interface import STATUS_VARIABLE, completion, global_name, message, shard_name


_INDEX = "_cattorch demo index"
_BEST = "_cattorch demo best"
_TOTAL = "_cattorch demo total"
_PICK = "_cattorch demo pick"
_DEMO = "Demo"
_BACKDROP = (
    b'<svg xmlns="http://www.w3.org/2000/svg" width="480" height="360" viewBox="0 0 480 360">'
    b'<rect width="480" height="360" fill="#ffffff"/></svg>'
)


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
        # The project's public messages, in order: each sprite's, then the
        # project's own, each with its completion broadcast.
        self.messages: list[str] = []
        self.global_lists: dict[str, list] = {}
        self.global_variables: dict[str, object] = {}
        # Who uses each global and sends each message, for the collision checks.
        self._global_owners: list[tuple[str, str]] = []
        self._message_owners: list[tuple[str, str]] = []

    # ── targets ────────────────────────────────────────────────────────────

    def add_sprite(self, path: Path, *, visible: bool, values=None) -> dict:
        """Add an exported sprite, declaring its globals on the stage as an upload would.

        The sprite keeps its name, which its globals and messages start
        with. ``values`` are the values its globals start with, by name; a
        status and a tokenizer's text start empty.
        """
        sprite, assets = read_archive(path)
        name = sprite["name"]
        if name in {target["name"] for target in self.sprites} or name == "Stage":
            raise ValueError(
                f"two of the project's sprites would be named {name!r}; export one with "
                "another name="
            )
        owner = f"sprite {name!r}"
        references = global_references(sprite)
        for section in ("variables", "lists"):
            self._global_owners += [(owner, used) for used in references[section].values()]
        sent = set(references["broadcasts"].values())
        self._message_owners += [(owner, text) for text in sorted(sent)]
        interface.check_global_names(self._global_owners)
        interface.check_messages(
            (owner, text) for owner, text in self._message_owners
            if text not in interface.SHARED_MESSAGES
        )
        starting = {
            global_name(name, STATUS_VARIABLE): interface.INITIAL_STATUS,
            global_name(name, interface.TEXT): "",
            **(values or {}),
        }
        used = install(self.stage, sprite, starting)
        for list_name in used["lists"]:
            entry = next(entry for entry in self.stage["lists"].values() if entry[0] == list_name)
            self.global_lists.setdefault(list_name, entry[1])
        for variable in used["variables"]:
            entry = next(
                entry for entry in self.stage["variables"].values() if entry[0] == variable
            )
            self.global_variables.setdefault(variable, entry[1])
        calls = sorted(
            text for text in sent
            if text not in interface.SHARED_MESSAGES and not text.endswith(" complete")
        )
        for text in calls:
            self.messages += [text, completion(text)]
        if interface.SHARED_PREPARE_FOR_SAVE in sent and (
            interface.SHARED_PREPARE_FOR_SAVE not in self.messages
        ):
            self.messages.append(interface.SHARED_PREPARE_FOR_SAVE)
        sprite.update(visible=visible)
        self.sprites.append(sprite)
        self.assets.update(assets)
        return sprite

    def add_demo_sprite(self) -> dict:
        """The project's own small sprite, first among the sprites so that it opens selected."""
        taken = {target["name"] for target in self.sprites}
        name, index = _DEMO, 2
        while name in taken:
            name, index = f"{_DEMO}{index}", index + 1
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
            "sounds": [], "volume": 100, "layerOrder": 1,
            "visible": True, "x": -150, "y": -100, "size": 100, "direction": 90,
            "draggable": False, "rotationStyle": "all around",
        }
        self.sprites.insert(0, sprite)
        return sprite

    # ── global data ────────────────────────────────────────────────────────

    def global_list(self, name: str, values=()) -> str:
        """A global list of the project's own, declared on the stage when a script uses it."""
        if name not in self.global_lists:
            self.global_lists[name] = list(values)
            self._global_owners.append(("the project", name))
            interface.check_global_names(self._global_owners)
        return name

    def global_variable(self, name: str, value: object = 0) -> str:
        if name not in self.global_variables:
            self.global_variables[name] = value
            self._global_owners.append(("the project", name))
            interface.check_global_names(self._global_owners)
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
        _add_warp_procedure(target, name, program, x=-800, y=200 * len(target["blocks"]) % 4000)
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

    def receiver(self, target: dict, text: str, body, *, lists=(), variables=()) -> None:
        """One of the project's own messages: run ``body`` whenever ``text`` is broadcast.

        The body runs in a warp procedure, ``_cattorch demo TEXT``, and ends
        with a ``broadcast and wait`` of ``TEXT complete``.
        """
        self._message_owners.append(("the project", text))
        interface.check_messages(
            (owner, sent) for owner, sent in self._message_owners
            if sent not in interface.SHARED_MESSAGES
        )
        procedure = f"_cattorch demo {text}"
        body = (*body, broadcast_and_wait(completion(text)))
        self.procedure(target, procedure, body, lists=lists, variables=variables)
        self.script(target, {
            "opcode": "event_whenbroadcastreceived",
            "fields": {"BROADCAST_OPTION": [text, f"cattorch_message_{text}"]},
        }, (*_call(procedure),), y=200 * len(self.messages))
        self.messages += [text, completion(text)]

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

    def note(self, target: dict, text: str) -> None:
        """A comment on ``target``'s canvas: the Demo's is what the project opens on."""
        target["comments"]["cattorch_project_note"] = {
            "blockId": None, "x": -400, "y": -300, "width": 400, "height": 280,
            "minimized": False, "text": text,
        }

    def _unify_broadcasts(self) -> None:
        """Give each message one ID, declared on the stage as Scratch expects."""
        canonical: dict[str, tuple[str, str]] = {}

        def identify(text: str) -> str:
            key = text.lower()
            if key not in canonical:
                canonical[key] = (f"cattorch_message_{len(canonical) + 1}", text)
            return canonical[key][0]

        for target in (self.stage, *self.sprites):
            remap = {
                identifier: identify(text)
                for identifier, text in target.get("broadcasts", {}).items()
            }
            for block in target["blocks"].values():
                if not isinstance(block, dict):
                    continue
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
        for entry in self.stage["lists"].values():
            if entry[0] in self.global_lists:
                entry[1] = list(self.global_lists[entry[0]])
        for entry in self.stage["variables"].values():
            if entry[0] in self.global_variables:
                entry[1] = self.global_variables[entry[0]]
        for position, sprite in enumerate(self.sprites):
            sprite["layerOrder"] = position + 1
        project = {
            "targets": [self.stage, *self.sprites], "monitors": self.monitors,
            "extensions": [],
            "meta": {"semver": "3.0.0", "vm": "0.4.0-cattorch", "agent": "cattorch"},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        text = write_archive(path, project, self.assets, PROJECT_JSON)
        warning = online_json_size_warning(len(text.encode("utf-8")), "project JSON")
        return [] if warning is None else [warning]


def _call(procedure: str) -> tuple:
    from cattorch.util.scratch.dsl import call

    return (call(procedure),)


def _example_values(spec, example) -> list:
    if example is not None:
        return example.detach().flatten().tolist()
    count = 1
    for extent in spec.shape:
        count *= extent.minimum if isinstance(extent, BoundedDimension) else extent
    return [0] * count


def _shards(builder, list_name: str) -> list[str]:
    """A global list's physical lists in the project: ``LIST``, ``LIST shard 2``, ..."""
    shards = [list_name]
    while shard_name(list_name, len(shards)) in builder.global_lists:
        shards.append(shard_name(list_name, len(shards)))
    return shards


def _starting_values(specs, examples) -> dict:
    """The values input globals start with, shard by shard, by global name."""
    values = {}
    for spec, example in zip(specs, examples):
        flat = _example_values(spec, example)
        for position in range(max(1, math.ceil(len(flat) / SCRATCH_LIST_LIMIT))):
            values[shard_name(spec.list_name, position)] = (
                flat[position * SCRATCH_LIST_LIMIT:(position + 1) * SCRATCH_LIST_LIMIT]
            )
    return values


def _layout(builder, names: list[str]) -> None:
    for position, list_name in enumerate(names):
        builder.monitor(list_name, kind="list", x=5 + 120 * (position % 4), y=5 + 180 * (position // 4))


_FLAG = {"opcode": "event_whenflagclicked", "fields": {}}


def _forward_project(builder, result, example_inputs, labels) -> None:
    examples = list(example_inputs) + [None] * (len(result.inputs) - len(example_inputs))
    sprite = builder.add_sprite(
        result.path, visible=False, values=_starting_values(result.inputs, examples),
    )
    demo = builder.add_demo_sprite()
    name = sprite["name"]
    shown = [
        shard for spec in (*result.inputs, *result.outputs)
        for shard in _shards(builder, spec.list_name)
    ]
    _layout(builder, shown)
    forward = message(name, interface.FORWARD)
    predict = global_name(name, "predict")
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
        table = builder.global_list(global_name(name, "labels"), [str(label) for label in labels])
        prediction = builder.global_variable(global_name(name, "prediction"), "")
        builder.receiver(demo, predict, (
            *_argmax(output.list_name),
            set_var(prediction, item(table, var(_BEST))),
        ), lists=(output.list_name, table), variables=(_INDEX, _BEST, prediction))
        builder.monitor(prediction, kind="large", x=5, y=320)
    run: tuple = (broadcast_and_wait(forward),)
    if labels is not None:
        run += (broadcast_and_wait(predict),)
    builder.script(demo, _FLAG, run)
    builder.script(demo, {
        "opcode": "event_whenkeypressed", "fields": {"KEY_OPTION": ["space", None]},
    }, run, y=200)
    inputs = ", ".join(spec.list_name for spec in result.inputs)
    builder.note(
        demo,
        f"Edit {inputs}, then click the green flag or press space to run {name}.\n\n"
        f"Any sprite can fill the same lists, then broadcast \"{forward}\" and wait, "
        f"and read \"{global_name(name, STATUS_VARIABLE)}\": when it is ok, the outputs "
        f"are in {', '.join(spec.list_name for spec in result.outputs)}."
        + (
            f" Broadcast \"{predict}\" and wait to put the label of the largest output "
            f"in \"{global_name(name, 'prediction')}\"."
            if labels is not None else ""
        )
        + f"\n\nYou never need to open {name}: a large model is slow to show in the editor."
    )


def _program_project(builder, result) -> None:
    # The entry points' inputs start at zeros, as a model's do.
    inputs = [spec for entry in result.entrypoints for spec in entry.inputs]
    sprite = builder.add_sprite(
        result.path, visible=False, values=_starting_values(inputs, [None] * len(inputs)),
    )
    demo = builder.add_demo_sprite()
    name = sprite["name"]
    shown, lines = [], []
    for entry in result.entrypoints:
        shown += [
            shard for spec in (*entry.inputs, *entry.outputs)
            for shard in _shards(builder, spec.list_name)
        ]
        lines.append(f"broadcast \"{entry.broadcast}\" and wait")
    _layout(builder, list(dict.fromkeys(shown)))
    builder.note(
        demo,
        "Fill an entry point's input lists, then:\n" + "\n".join(lines)
        + f"\nand read \"{global_name(name, STATUS_VARIABLE)}\": when it is ok, its outputs "
        f"are in its output lists.\n\nState is kept in {name}'s state lists between calls. "
        f"You never need to open {name}."
    )


def _generation_project(builder, result, tokenizer, *, max_new_tokens, temperature, eos_token_id,
                        prompt) -> None:
    model = builder.add_sprite(result.path, visible=False)
    words = builder.add_sprite(tokenizer.path, visible=False)
    demo = builder.add_demo_sprite()
    name, tokenizer_name = model["name"], words["name"]
    tokens = global_name(name, interface.TOKEN_IDS)
    logits = global_name(name, interface.LOGITS)
    status = global_name(name, STATUS_VARIABLE)
    next_token = builder.global_variable(global_name(name, "next token"), 0)
    heat = builder.global_variable(global_name(name, "temperature"), float(temperature))
    sample = global_name(name, "sample")
    # The project's sampler: the next token from the logits the last
    # prefill or decode wrote.
    builder.receiver(demo, sample, (
        *_sample(logits, heat),
        set_var(next_token, sub(var(_BEST), 1)),
    ), lists=(logits,), variables=(_INDEX, _BEST, _TOTAL, _PICK, next_token, heat))

    text = global_name(tokenizer_name, interface.TEXT)
    ids = global_name(tokenizer_name, interface.TOKEN_IDS)
    words_status = global_name(tokenizer_name, STATUS_VARIABLE)
    generated = "_cattorch demo generated"
    count, done, seed = "_cattorch demo count", "_cattorch demo done", "_cattorch demo prompt"
    problem = "_cattorch demo problem"
    builder.procedure(demo, "_cattorch demo prompt tokens", _copy(ids, tokens),
                      lists=(ids, tokens), variables=(_INDEX,))
    builder.procedure(demo, "_cattorch demo generated tokens", _copy(generated, ids),
                      lists=(generated, ids), variables=(_INDEX,))
    stop = -1 if eos_token_id is None else int(eos_token_id)

    def failed(who: str, operation: str, status_variable: str, *, normal=()) -> tuple:
        """Report ``who``'s status after ``operation`` unless it is ok (or ``normal``)."""
        report: tuple = (
            set_var(problem, join(f"{message(who, operation)}: ", var(status_variable))),
            say(var(problem)),
        )
        for status_value in normal:
            report = (if_(not_(eq(var(status_variable), status_value)), report),)
        return (if_(not_(eq(var(status_variable), interface.OK)), report),)

    def sampled() -> tuple:
        """Sample the next token after a call that ended ok."""
        return (if_(eq(var(status), interface.OK), (broadcast_and_wait(sample),)),)

    check = (
        set_var(done, 0),
        # A full context ends the continuation as the token budget does.
        if_(not_(eq(var(status), interface.OK)), (set_var(done, 1),)),
        if_(not_(eq(var(problem), "")), (set_var(done, 1),)),
        if_(eq(var(count), max_new_tokens), (set_var(done, 1),)),
        if_(eq(var(next_token), stop), (set_var(done, 1),)),
    )
    question = prompt or "Type the start of some text (leave it empty to stop)"
    # The next question replaces whatever the sprite says, so it carries the
    # finished continuation, or the failure that ended it.
    ask_again = if_else(
        eq(var(problem), ""),
        (ask(join(join(var(seed), var(text)), f"\n\n{question}")),),
        (ask(join(join(var(problem), ". "), question)),),
    )
    body = (
        ask(question),
        repeat_until(eq(answer(), ""), (
            set_var(seed, answer()),
            set_var(text, answer()),
            set_var(problem, ""),
            broadcast_and_wait(message(tokenizer_name, interface.TOKENIZE)),
            *failed(tokenizer_name, interface.TOKENIZE, words_status),
            if_(eq(var(problem), ""), (
                *_call("_cattorch demo prompt tokens"),
                say(join(var(seed), "...")),
                set_var(text, ""),
                broadcast_and_wait(message(name, interface.PREFILL)),
                *failed(name, interface.PREFILL, status),
                *sampled(),
                clear(generated),
                set_var(count, 0),
                *check,
                repeat_until(eq(var(done), 1), (
                    append(generated, var(next_token)),
                    *_call("_cattorch demo generated tokens"),
                    broadcast_and_wait(message(tokenizer_name, interface.DETOKENIZE)),
                    *failed(tokenizer_name, interface.DETOKENIZE, words_status),
                    if_(eq(var(problem), ""), (
                        say(join(var(seed), var(text))),
                        clear(tokens),
                        append(tokens, var(next_token)),
                        broadcast_and_wait(message(name, interface.DECODE)),
                        *failed(name, interface.DECODE, status, normal=(interface.CONTEXT_FULL,)),
                        *sampled(),
                        change_var(count, 1),
                    )),
                    *check,
                )),
            )),
            ask_again,
        )),
    )
    builder.script(demo, _FLAG, body,
                   lists=(generated, tokens), variables=(count, done, seed, text, status,
                                                         next_token, words_status, problem))
    builder.monitor(heat, kind="slider", x=5, y=5, sliderMin=0, sliderMax=2, isDiscrete=False)
    builder.note(
        demo,
        f"Click the green flag and type a prompt. This sprite continues it with "
        f"{name}, up to {max_new_tokens} tokens"
        + ("" if eos_token_id is None else f" or until token {eos_token_id}")
        + ". The temperature slider sets how random the choices are; 0 always picks "
        f"the likeliest token.\n\n"
        f"To drive the sprites yourself, broadcast each message and wait, then read the "
        f"status: set \"{text}\" and broadcast \"{message(tokenizer_name, interface.TOKENIZE)}\" "
        f"to fill \"{ids}\"; fill \"{tokens}\" and broadcast "
        f"\"{message(name, interface.PREFILL)}\" (a prompt) or "
        f"\"{message(name, interface.DECODE)}\" (one token), which write \"{logits}\" and "
        f"\"{status}\"; broadcast \"{sample}\" to pick \"{next_token}\" from them; and "
        f"broadcast \"{message(tokenizer_name, interface.DETOKENIZE)}\" to turn \"{ids}\" "
        f"back into \"{text}\". You never need to open {name} or {tokenizer_name}."
    )


# The generation options' defaults, which other kinds of project accept as given.
_GENERATION_DEFAULTS = {
    "max_new_tokens": 40, "temperature": 0.8, "eos_token_id": None, "prompt": None,
}


def _check_options(model, *, tokenizer, example_inputs, labels, **generation) -> None:
    """Reject options of another kind of project, and values a project can't use.

    Every option applies to one kind of model: ``tokenizer`` and the
    generation options to a ``GenerationResult``, ``example_inputs`` and
    ``labels`` to a ``TranspileResult``. Given to another kind, an option
    would be ignored, so it raises ``ValueError``.
    """
    kind = type(model).__name__
    if isinstance(model, GenerationResult):
        if tokenizer is None:
            raise ValueError("a text generator's project needs its tokenizer= result")
        if not isinstance(tokenizer, TokenizerResult):
            raise TypeError("tokenizer must be the result of cattorch.transpile_tokenizer")
        _check_generation_options(model, **generation)
    else:
        if tokenizer is not None:
            raise ValueError("tokenizer= applies only to a text generator, a GenerationResult")
        for option, value in generation.items():
            if value != _GENERATION_DEFAULTS[option]:
                raise ValueError(
                    f"{option}= applies only to a text generator's project, not a {kind}'s"
                )
    if isinstance(model, TranspileResult):
        _check_examples(model, example_inputs)
        if labels is not None and (
            isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence)
        ):
            raise TypeError("labels must be a sequence of labels, such as a list of strings")
    else:
        if not isinstance(example_inputs, (tuple, list)) or len(example_inputs):
            raise ValueError(
                f"example_inputs= applies only to a TranspileResult's project, not a {kind}'s"
            )
        if labels is not None:
            raise ValueError(f"labels= applies only to a TranspileResult's project, not a {kind}'s")


def _check_examples(model: TranspileResult, example_inputs) -> None:
    """``example_inputs`` must fit the export's inputs, in order; the rest start at zeros."""
    if not isinstance(example_inputs, (tuple, list)):
        raise TypeError(
            "example_inputs must be a tensor or a tuple of tensors, not "
            f"{type(example_inputs).__name__}"
        )
    if len(example_inputs) > len(model.inputs):
        raise ValueError(
            f"example_inputs has {len(example_inputs)} tensors, but the export has "
            f"{len(model.inputs)} inputs"
        )
    for index, (spec, value) in enumerate(zip(model.inputs, example_inputs)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"example input {index} is a {type(value).__name__}, not a tensor")
        shape = tuple(value.shape)
        fits = len(shape) == len(spec.shape) and all(
            extent.minimum <= size <= extent.maximum
            if isinstance(extent, BoundedDimension) else size == extent
            for size, extent in zip(shape, spec.shape)
        )
        dtype = str(value.dtype).removeprefix("torch.")
        if not fits or dtype != spec.dtype:
            expected = tuple(
                f"{extent.minimum}..{extent.maximum}" if isinstance(extent, BoundedDimension)
                else extent for extent in spec.shape
            )
            raise ValueError(
                f"example input {index} is a {dtype} tensor of shape {shape}, but "
                f"{spec.list_name!r} takes {spec.dtype} of shape {expected}"
            )


def _check_generation_options(model, *, max_new_tokens, temperature, eos_token_id, prompt):
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError(f"max_new_tokens must be an integer, not {type(max_new_tokens).__name__}")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError(f"temperature must be a number, not {type(temperature).__name__}")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError(f"temperature must be a finite number of 0 or more, not {temperature!r}")
    if eos_token_id is not None:
        if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
            raise TypeError(
                f"eos_token_id must be an integer or None, not {type(eos_token_id).__name__}"
            )
        if not 0 <= eos_token_id < model.vocabulary_size:
            raise ValueError(
                f"eos_token_id {eos_token_id} is not a token ID of the model, whose "
                f"vocabulary has {model.vocabulary_size}"
            )
    if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
        raise ValueError("prompt must be a non-empty string or None")


def _check_vocabularies(model: GenerationResult, tokenizer: TokenizerResult) -> None:
    """Warn when the model and tokenizer don't have the same token IDs.

    A model whose vocabulary is padded past the tokenizer's (50304 for 50257)
    is common and works until the model picks a padding ID; a tokenizer with
    IDs past the model's (an added padding token) works until the text
    contains one. Either way the Demo stops and says the failing status.
    """
    tokens, vocabulary = tokenizer.token_count, model.vocabulary_size
    if tokens < vocabulary:
        warn(
            f"the model has {vocabulary} token IDs and the tokenizer {tokens}: if the model "
            f"picks an ID from {tokens} up, detokenize fails with an invalid item, and the "
            "Demo stops there"
        )
    elif tokens > vocabulary:
        warn(
            f"the tokenizer has {tokens} token IDs and the model {vocabulary}: text that "
            f"tokenizes to an ID from {vocabulary} up fails the model's prefill with an "
            "invalid item, and the Demo stops there"
        )


def build_project(
    model: TranspileResult | ProgramResult | GenerationResult,
    output_path: str | Path,
    *,
    tokenizer: TokenizerResult | None = None,
    example_inputs: torch.Tensor | tuple[torch.Tensor, ...] = (),
    labels: Sequence[str] | None = None,
    max_new_tokens: int = 40,
    temperature: float = 0.8,
    eos_token_id: int | None = None,
    prompt: str | None = None,
) -> ProjectResult:
    """Write a runnable Scratch project around exported sprites.

    ``model`` is a ``transpile`` result. For a text generator (a
    ``GenerationResult``), pass the ``tokenizer`` result too: the project asks for a prompt and
    continues it, sampling at ``temperature`` (0 picks the likeliest token)
    for up to ``max_new_tokens`` tokens or until ``eos_token_id``. ``prompt``
    replaces the question it asks. For any other model, the project shows the
    input and output lists, starting from ``example_inputs`` or zeros, and runs
    the model on the green flag or space key; ``labels`` names the outputs of a
    classifier and shows the predicted label (``NAME predict``). A model
    with entry points gets list monitors and a note on how to call them.

    The exported sprites keep their names and interface: any sprite can run
    them with ``broadcast [NAME OP] and wait`` and read ``NAME status``. The
    project declares their globals on the stage, and a small Demo sprite,
    selected when the project opens, holds the demo and the project's own
    messages. ``ProjectResult`` lists the broadcasts and data. Raises
    ``ValueError`` if two sprites, globals or broadcasts would be the same to
    Scratch. Warnings are also emitted as ``CattorchWarning``.
    """
    if not isinstance(model, (TranspileResult, ProgramResult, GenerationResult)):
        raise TypeError(
            f"model must be the result of cattorch.transpile, not a {type(model).__name__}"
        )
    output_path = Path(output_file(output_path, ".sb3"))
    if isinstance(example_inputs, torch.Tensor):
        example_inputs = (example_inputs,)
    _check_options(
        model, tokenizer=tokenizer, example_inputs=example_inputs, labels=labels,
        max_new_tokens=max_new_tokens, temperature=temperature, eos_token_id=eos_token_id,
        prompt=prompt,
    )
    with collect_warnings() as collected:
        builder = _Builder()
        if isinstance(model, ProgramResult):
            _program_project(builder, model)
        elif isinstance(model, GenerationResult):
            assert isinstance(tokenizer, TokenizerResult)
            _check_vocabularies(model, tokenizer)
            _generation_project(
                builder, model, tokenizer, max_new_tokens=max_new_tokens,
                temperature=float(temperature), eos_token_id=eos_token_id, prompt=prompt,
            )
        else:
            _forward_project(builder, model, tuple(example_inputs), labels)
        for written in builder.write(output_path):
            warn(written)
    project_warnings = list(collected)
    return ProjectResult(
        path=output_path,
        sprites=tuple(sprite["name"] for sprite in builder.sprites),
        broadcasts=tuple(builder.messages),
        lists=tuple(builder.global_lists),
        variables=tuple(builder.global_variables),
        warnings=tuple(project_warnings),
    )


# ── adding sprites to a project of your own ─────────────────────────────────

# A sprite given by its export's result, or by the path of its ``.sprite3``.
SpriteSource = str | Path | TranspileResult | ProgramResult | GenerationResult | TokenizerResult
_RESULTS = (TranspileResult, ProgramResult, GenerationResult, TokenizerResult)
# The sprite an empty project gets, so that a model isn't selected when it opens.
_BLANK = "Sprite1"
# What a replaced sprite passes on to the one that takes its place.
_KEPT_ON_REPLACE = (
    "layerOrder", "visible", "x", "y", "size", "direction", "draggable", "rotationStyle",
)


def _user_file(source, suffix: str, document: str, *, parameter: str) -> Path:
    """The file a path names, read as an export names its file, checked to be of its kind.

    ``suffix`` is added unless the path ends in it, as ``transpile`` and
    ``build_project`` name their files; the path exactly as given is read
    when that file doesn't exist. A missing file or one of the wrong kind
    raises ``ValueError``.
    """
    path = output_file(source, suffix, parameter=parameter)
    if not path.exists() and Path(source).is_file():
        path = Path(source)
    read_user_archive(path, document, parameter=parameter)
    return path


def _project_path(project) -> Path:
    if isinstance(project, ProjectResult):
        read_user_archive(project.path, PROJECT_JSON, parameter="project")
        return Path(project.path)
    if isinstance(project, (str, os.PathLike)) and not isinstance(project, bytes):
        return _user_file(project, ".sb3", PROJECT_JSON, parameter="project")
    raise TypeError(
        f"project must be a path or a ProjectResult, not a {type(project).__name__}"
    )


def _sprite_paths(sprites) -> list[Path]:
    if isinstance(sprites, (*_RESULTS, str, os.PathLike)):
        sprites = [sprites]
    elif isinstance(sprites, (bytes, ProjectResult)) or not isinstance(sprites, Sequence):
        raise TypeError(
            "sprites must be a sprite (a transpile or transpile_tokenizer result, or a "
            f".sprite3 path) or a sequence of them, not a {type(sprites).__name__}"
        )
    if not sprites:
        raise ValueError("sprites is empty; give at least one sprite to add")
    paths = []
    for sprite in sprites:
        if isinstance(sprite, _RESULTS):
            paths.append(Path(sprite.path))
        elif isinstance(sprite, (str, os.PathLike)) and not isinstance(sprite, bytes):
            paths.append(_user_file(sprite, ".sprite3", SPRITE_JSON, parameter="sprite"))
        else:
            raise TypeError(
                "each sprite must be a transpile or transpile_tokenizer result, or a "
                f".sprite3 path, not a {type(sprite).__name__}"
            )
    return paths


def _ids(target: dict) -> set[str]:
    """Every ID ``target`` defines, and the global IDs its blocks refer to."""
    taken: set[str] = set()
    for section in ("blocks", "variables", "lists", "broadcasts", "comments"):
        taken.update(target.get(section) or {})
    for references in global_references(target).values():
        taken.update(references)
    return taken


def _fresh_ids(sprite: dict, taken: set[str]) -> dict:
    """``sprite`` with each of its IDs that ``taken`` already holds replaced by a new one."""
    own = _ids(sprite)
    mapping = {}
    for identifier in sorted(own & taken):
        while True:
            fresh = f"{identifier}_{uuid.uuid4().hex[:8]}"
            if fresh not in taken and fresh not in own:
                break
        mapping[identifier] = fresh
        own.add(fresh)
    return remap_ids(sprite, mapping) if mapping else sprite


def _names(target: dict, section: str) -> dict[str, str]:
    """``target``'s variables or lists, by name: ``{name: id}``."""
    return {
        entry[0]: identifier for identifier, entry in (target.get(section) or {}).items()
        if isinstance(entry, list) and entry
    }


def _media(target: dict) -> set[str]:
    """The files of ``target``'s costumes and sounds."""
    return {
        entry["md5ext"] for entry in (*target.get("costumes", ()), *target.get("sounds", ()))
        if isinstance(entry, dict) and isinstance(entry.get("md5ext"), str)
    }


def _blank_sprite(name: str, assets: dict[str, bytes], layer: int, note: str) -> dict:
    costume = next((TEMPLATE_DIR / "sprite").glob("*.svg"))
    data = costume.read_bytes()
    asset = hashlib.md5(data).hexdigest()
    assets[f"{asset}.svg"] = data
    return {
        "isStage": False, "name": name, "variables": {}, "lists": {}, "broadcasts": {},
        "blocks": {}, "currentCostume": 0,
        "comments": {"cattorch_project_note": {
            "blockId": None, "x": -400, "y": -300, "width": 400, "height": 280,
            "minimized": False, "text": note,
        }},
        "costumes": [{
            "name": "costume1", "bitmapResolution": 1, "dataFormat": "svg",
            "assetId": asset, "md5ext": f"{asset}.svg",
            "rotationCenterX": 48, "rotationCenterY": 50,
        }],
        "sounds": [], "volume": 100, "layerOrder": layer,
        "visible": True, "x": 0, "y": 0, "size": 100, "direction": 90,
        "draggable": False, "rotationStyle": "all around",
    }


def _check_kinds(stage: dict, owner: str, references: dict) -> None:
    """A global the sprite uses as a list mustn't be the stage's variable, or the reverse."""
    for section, other, kind, other_kind in (
        ("lists", "variables", "list", "variable"), ("variables", "lists", "variable", "list"),
    ):
        own, others = _names(stage, section), _names(stage, other)
        for name in sorted(set(references[section].values())):
            if name in others and name not in own:
                raise ValueError(
                    f"{owner} uses the global {kind} {name!r}, but the project's stage has a "
                    f"global {other_kind} of that name; rename or delete the project's "
                    f"{other_kind}"
                )


def _check_local_names(stage: dict, users: list[dict], added: list[dict], new: dict) -> None:
    """Refuse a sprite's own variable or list that a global of the same name and kind hides.

    Loading a project keeps both (scratch-vm's ``installTargets`` only
    reconciles references then), but Scratch renames a sprite's own
    variable when the stage has a global of the same name and kind as the
    sprite comes in (``fixUpVariableReferences``, on upload and backpack
    paste), which breaks the ``([NAME] of [SPRITE])`` reads of an added
    sprite's own variables, and the sprite's palette would show two of that
    name. ``new`` holds the globals the
    added sprites declare, ``{section: names}``, which mustn't be the name
    of a variable of one of the project's own sprites (``users``) either.
    """
    for section, kind in (("variables", "variable"), ("lists", "list")):
        stage_names = _names(stage, section)
        for sprite in added:
            for name in sorted(_names(sprite, section)):
                if name in stage_names:
                    raise ValueError(
                        f"the project's stage has a global {kind} {name!r}, the name of one of "
                        f"sprite {sprite['name']!r}'s own {kind}s: Scratch would rename the "
                        f"sprite's, breaking the ([{name}] of [{sprite['name']}]) block; rename "
                        f"or delete the project's {kind}"
                    )
        for sprite in users:
            for name in sorted(set(_names(sprite, section)) & new[section]):
                raise ValueError(
                    f"sprite {sprite['name']!r} of the project has its own {kind} {name!r}, "
                    f"the name of a global {kind} the added sprites use: its scripts would "
                    f"use its own {kind}, not the global; rename or delete sprite "
                    f"{sprite['name']!r}'s {kind}"
                )


def add_to_project(
    project: str | Path | ProjectResult,
    sprites: SpriteSource | Sequence[SpriteSource],
    output_path: str | Path,
    *,
    replace: bool = False,
) -> ProjectResult:
    """Add exported sprites to a Scratch project of your own, and write it to ``output_path``.

    ``project`` is an ``.sb3`` (a path, or a ``build_project`` result) and
    ``sprites`` one sprite or a sequence of them, each a ``transpile`` or
    ``transpile_tokenizer`` result or a ``.sprite3`` path. The sprites go
    after the project's own, so it still opens with the sprite it opened
    with selected (a model sprite is slow to show in the editor, and
    uploading one selects it). Their globals and messages are declared on
    the stage as uploading the sprites would: a global already on the stage
    with the same name and kind, and a message equal ignoring case, is used,
    so scripts that already use ``NAME input`` or ``NAME forward`` run the
    added sprite. A new ``NAME status`` and tokenizer's ``NAME text`` start
    empty. The project's own sprites, scripts and IDs are kept as they are.
    A project without sprites gets a blank one first, ``Sprite1``.

    ``output_path`` gets ``.sb3`` unless it ends in it; it may be the
    project's own path. ``replace=True`` replaces a sprite of the project
    with an added sprite of the same name, in its place, keeping the stage's
    globals: the way to put a retrained model in. ``ProjectResult`` lists
    every sprite of the written project, in order, and the broadcasts and
    global lists and variables the added sprites use. Raises ``ValueError``
    for a sprite name the project already has (without ``replace``), two
    added sprites of one name, globals or broadcasts of two added sprites
    that would be the same to Scratch, or a global that would hide a
    sprite's own variable or list. Warnings are also emitted as
    ``CattorchWarning``.
    """
    project_path = _project_path(project)
    paths = _sprite_paths(sprites)
    output = output_file(output_path, ".sb3")
    if not isinstance(replace, bool):
        raise TypeError(f"replace must be True or False, not {type(replace).__name__}")
    data, assets = read_archive(project_path, PROJECT_JSON)
    targets = data.get("targets") if isinstance(data, dict) else None
    if not isinstance(targets, list) or not any(
        isinstance(target, dict) and target.get("isStage") for target in targets
    ):
        raise ValueError(f"project {str(project_path)!r} is not a Scratch 3 project")
    stage = next(target for target in targets if target.get("isStage"))
    places = {
        target["name"]: position for position, target in enumerate(targets)
        if not target.get("isStage")
    }

    added: list[dict] = []
    files: dict[str, Path] = {}
    global_owners: list[tuple[str, str]] = []
    message_owners: list[tuple[str, str]] = []
    for path in paths:
        sprite, sprite_assets = read_archive(path)
        name = sprite["name"]
        if name in files:
            raise ValueError(
                f"two of the sprites are named {name!r}, {str(files[name])!r} and "
                f"{str(path)!r}; export one with another name="
            )
        if name in places and not replace:
            raise ValueError(
                f"project {str(project_path)!r} already has a sprite named {name!r}, and "
                f"so does {str(path)!r}; pass replace=True (--replace) to replace the "
                "project's, or export the sprite with another name="
            )
        files[name] = path
        owner = f"sprite {name!r}"
        references = global_references(sprite)
        global_owners += [
            (owner, used) for section in ("variables", "lists")
            for used in references[section].values()
        ]
        message_owners += [(owner, text) for text in sorted(set(references["broadcasts"].values()))]
        interface.check_global_names(global_owners)
        interface.check_messages(
            (owner, text) for owner, text in message_owners
            if text not in interface.SHARED_MESSAGES
        )
        _check_kinds(stage, owner, references)
        added.append(sprite)
        assets.update(sprite_assets)

    replaced = {sprite["name"] for sprite in added if sprite["name"] in places}
    old_sprites = {name: targets[places[name]] for name in replaced}
    users = [
        target for target in targets
        if not target.get("isStage") and target["name"] not in replaced
    ]
    taken: set[str] = {
        identifier for target in targets
        if target.get("isStage") or target["name"] not in replaced
        for identifier in _ids(target)
    }
    taken.update(
        monitor["id"] for monitor in data.get("monitors") or ()
        if isinstance(monitor, dict) and isinstance(monitor.get("id"), str)
    )
    before = {section: set(_names(stage, section)) for section in ("variables", "lists")}
    broadcasts: list[str] = []
    lists: list[str] = []
    variables: list[str] = []
    with collect_warnings() as collected:
        for position, sprite in enumerate(added):
            sprite = added[position] = _fresh_ids(sprite, taken)
            taken |= _ids(sprite)
            name = sprite["name"]
            starting = {
                global_name(name, STATUS_VARIABLE): interface.INITIAL_STATUS,
                global_name(name, interface.TEXT): "",
            }
            used = install(stage, sprite, starting)
            lists += [list_name for list_name in used["lists"] if list_name not in lists]
            variables += [variable for variable in used["variables"] if variable not in variables]
            sent = set(global_references(sprite)["broadcasts"].values())
            for text in sorted(
                text for text in sent
                if text not in interface.SHARED_MESSAGES and not text.endswith(" complete")
            ):
                broadcasts += [text, completion(text)]
            if interface.SHARED_PREPARE_FOR_SAVE in sent and (
                interface.SHARED_PREPARE_FOR_SAVE not in broadcasts
            ):
                broadcasts.append(interface.SHARED_PREPARE_FOR_SAVE)
            sprite["visible"] = False
        new = {
            section: set(_names(stage, section)) - before[section]
            for section in ("variables", "lists")
        }
        _check_local_names(stage, users, added, new)

        layer = max(
            (target.get("layerOrder", 0) for target in targets if not target.get("isStage")),
            default=0,
        )
        monitors = data.get("monitors") or []
        appended = []
        if not users and not replaced:
            names = [sprite["name"] for sprite in added]
            blank, index = _BLANK, 2
            while blank in names:
                blank, index = f"{_BLANK.rstrip('1')}{index}", index + 1
            layer += 1
            appended.append(_blank_sprite(blank, assets, layer, (
                f"This project uses {', '.join(names)} without opening "
                f"{'it' if len(names) == 1 else 'them'}: fill the input lists, broadcast "
                "a call such as \"NAME forward\" and wait, then read \"NAME status\" "
                "(NAME being the sprite's name). When it is ok, the outputs are in the "
                "output lists.\n\nA large model sprite is slow to show in the editor, so "
                "work in this sprite or the stage."
            )))
        for sprite in added:
            name = sprite["name"]
            if name in replaced:
                old = old_sprites[name]
                sprite.update({key: old[key] for key in _KEPT_ON_REPLACE if key in old})
                targets[places[name]] = sprite
                own = set(sprite.get("variables") or {}) | set(sprite.get("lists") or {})
                monitors = [
                    monitor for monitor in monitors
                    if not (
                        isinstance(monitor, dict) and monitor.get("spriteName") == name
                        and monitor.get("opcode") in ("data_variable", "data_listcontents")
                        and monitor.get("id") not in own
                    )
                ]
            else:
                layer += 1
                sprite["layerOrder"] = layer
                appended.append(sprite)
        targets += appended
        # The replaced sprites' costumes and sounds go with them, unless
        # another sprite still uses the same file.
        kept = {file for target in targets for file in _media(target)}
        for name in replaced:
            for file in _media(old_sprites[name]) - kept:
                assets.pop(file, None)
        if "monitors" in data:
            data["monitors"] = monitors

        output.parent.mkdir(parents=True, exist_ok=True)
        # Written beside the output and moved over it, so that a failure never
        # leaves a half-written project in place of the one read.
        partial = output.with_name(f".{output.name}.{uuid.uuid4().hex[:8]}.partial")
        try:
            text = write_archive(partial, data, assets, PROJECT_JSON)
            os.replace(partial, output)
        finally:
            partial.unlink(missing_ok=True)
        size = online_json_size_warning(len(text.encode("utf-8")), "project JSON")
        if size is not None:
            warn(size)
    return ProjectResult(
        path=output,
        sprites=tuple(target["name"] for target in targets if not target.get("isStage")),
        broadcasts=tuple(broadcasts),
        lists=tuple(lists),
        variables=tuple(variables),
        warnings=tuple(collected),
    )
