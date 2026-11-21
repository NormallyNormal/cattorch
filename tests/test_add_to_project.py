"""Exported sprites added to a project of your own with add_to_project."""

from __future__ import annotations

import copy
import json
import zipfile

import pytest
import torch
from torch import nn

from cattorch import ProjectResult, add_to_project, build_project, entrypoint, interface, transpile
from cattorch.util.scratch.archive import PROJECT_JSON, read_archive, write_archive
from cattorch.util.scratch.dsl import append, broadcast_and_wait, clear, for_each, item, length, var
from cattorch.util.scratch.emulator import ProjectEmulator
from cattorch.util.scratch.globals import _broadcast_inputs

FLAG = {"opcode": "event_whenflagclicked", "fields": {}}


def project_json(path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read(PROJECT_JSON))


def sprites_of(data) -> list[dict]:
    return [target for target in data["targets"] if not target.get("isStage")]


def user_project(tmp_path, values, *, message="Net Forward", name="mine") -> object:
    """A project of the user's own: a Game sprite that runs ``net`` and copies its output.

    Its script fills ``net input`` with ``values``, sends ``message``
    (``net forward`` as the user typed it), and copies ``net output`` into
    Game's own list ``game result``. A second sprite, Cat, comes after it.
    """
    from cattorch.project import _Builder

    builder = _Builder()
    builder.global_list("net input", values)
    builder.global_list("net output", [])
    builder.global_variable("net status", "")
    cat = builder.add_demo_sprite()
    cat["name"] = "Cat"
    game = builder.add_demo_sprite()
    game["name"] = "Game"
    builder.script(game, FLAG, (
        clear("net input"),
        *(append("net input", number) for number in values),
        broadcast_and_wait(message),
        clear("game result"),
        for_each("_index", length("net output"), (
            append("game result", item("net output", var("_index"))),
        )),
    ), lists=("net input", "net output", "game result"), variables=("_index",))
    path = tmp_path / f"{name}.sb3"
    builder.write(path)
    return path


@pytest.fixture
def net(tmp_path):
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 3)).eval()
    value = torch.randn(1, 4)
    result = transpile(model, value, tmp_path / "net", name="net")
    return model, value, result


def game_result(path) -> list[float]:
    emulator = ProjectEmulator(path)
    emulator.green_flag()
    assert emulator.stage.variables["net status"] == interface.OK
    return [float(value) for value in emulator.sprites["Game"].lists["game result"]]


def test_a_sprite_added_after_the_users_runs_from_their_script(net, tmp_path):
    model, value, result = net
    path = user_project(tmp_path, value.flatten().tolist())
    before = project_json(path)
    project = add_to_project(path, result, tmp_path / "with net")
    assert isinstance(project, ProjectResult)
    assert project.path == tmp_path / "with net.sb3"
    # The user's sprites stay first, so the project still opens on Game.
    assert project.sprites == ("Game", "Cat", "net")
    assert project.broadcasts == (
        "net forward", "net forward complete", "net init", "net init complete",
        "net prepare for save", "net prepare for save complete", "cattorch prepare for save",
    )
    assert project.lists == ("net input", "net output")
    assert project.variables == ("net status",)
    assert project.warnings == ()
    assert game_result(project.path) == pytest.approx(
        model(value).detach().flatten().tolist(), abs=1e-5,
    )

    after = project_json(project.path)
    stage, *sprites = after["targets"]
    assert [sprite["name"] for sprite in sprites] == ["Game", "Cat", "net"]
    # Nothing of the user's changed: their sprites are as they were, and
    # their stage only gained what the model needs.
    assert sprites[:2] == sprites_of(before)
    old_stage = before["targets"][0]
    for section in ("variables", "lists", "broadcasts"):
        assert stage[section].items() >= old_stage[section].items()
    assert {key: value for key, value in stage.items() if key not in (
        "variables", "lists", "broadcasts",
    )} == {key: value for key, value in old_stage.items() if key not in (
        "variables", "lists", "broadcasts",
    )}
    assert after["meta"] == before["meta"]
    assert sprites[2]["layerOrder"] > max(sprite["layerOrder"] for sprite in sprites[:2])
    assert sprites[2]["visible"] is False


def test_the_users_globals_and_messages_are_reused(net, tmp_path):
    from cattorch.util.scratch.globals import global_references

    _, value, result = net
    path = user_project(tmp_path, value.flatten().tolist())
    old_stage = project_json(path)["targets"][0]
    stage, *sprites = project_json(add_to_project(path, result, path).path)["targets"]
    # One global of each name: the user's, with its ID and value.
    for section in ("lists", "variables"):
        names = [entry[0] for entry in stage[section].values()]
        assert len(names) == len(set(names))
    assert [entry[0] for entry in stage["lists"].values()].count("net input") == 1
    user_input = next(sid for sid, entry in old_stage["lists"].items() if entry[0] == "net input")
    assert stage["lists"][user_input] == ["net input", value.flatten().tolist()]
    # The user typed "Net Forward": the model answers that broadcast, not a
    # second one that Scratch would show as the same message.
    keys = [text.lower() for text in stage["broadcasts"].values()]
    assert keys.count("net forward") == 1
    user_forward = next(sid for sid, text in old_stage["broadcasts"].items() if text == "Net Forward")
    assert stage["broadcasts"][user_forward] == "Net Forward"
    used = global_references(sprites[-1])
    assert user_forward in used["broadcasts"]
    assert user_input in used["lists"]
    # The new globals start as build_project starts them.
    status = next(entry for entry in stage["variables"].values() if entry[0] == "net status")
    assert status[1] == interface.INITIAL_STATUS


def test_writing_over_the_project_needs_its_path_given(net, tmp_path):
    _, value, result = net
    path = user_project(tmp_path, value.flatten().tolist())
    with pytest.raises(TypeError):
        add_to_project(path, result)  # type: ignore[call-arg]
    project = add_to_project(str(path), [result.path], str(path))
    assert project.path == path
    assert [sprite["name"] for sprite in sprites_of(project_json(path))] == ["Game", "Cat", "net"]
    assert not list(tmp_path.glob("*.partial"))


def test_replace_swaps_the_sprite_in_its_place(net, tmp_path):
    model, value, result = net
    path = user_project(tmp_path, value.flatten().tolist())
    first = add_to_project(path, result, tmp_path / "first")
    # Something after the model, so that its place shows.
    second = build_project(transpile(nn.Linear(2, 2).eval(), torch.ones(1, 2), tmp_path / "other"),
                           tmp_path / "unused")
    later = add_to_project(first, tmp_path / "other.sprite3", tmp_path / "later")
    assert later.sprites == ("Game", "Cat", "net", "other")
    assert second.sprites == ("Demo", "other")
    old_net = sprites_of(project_json(later.path))[2]

    with pytest.raises(ValueError, match="already has a sprite named 'net'.*replace=True"):
        add_to_project(later, result, tmp_path / "again")

    torch.manual_seed(11)
    retrained = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 3)).eval()
    new = transpile(retrained, value, tmp_path / "retrained", name="net")
    project = add_to_project(later, new, tmp_path / "replaced", replace=True)
    assert project.sprites == ("Game", "Cat", "net", "other")
    assert game_result(project.path) == pytest.approx(
        retrained(value).detach().flatten().tolist(), abs=1e-5,
    )
    stage, *sprites = project_json(project.path)["targets"]
    assert sprites[2]["layerOrder"] == old_net["layerOrder"]
    assert sprites[2]["blocks"] != old_net["blocks"]
    # The stage's globals stay, and the new sprite binds to them by name.
    names = [entry[0] for entry in stage["lists"].values()]
    assert names.count("net input") == names.count("net output") == 1
    assert [entry[0] for entry in stage["variables"].values()].count("net status") == 1
    # The old sprite's files went with it.
    with zipfile.ZipFile(project.path) as archive:
        files = set(archive.namelist()) - {PROJECT_JSON}
    assert files == {
        entry["md5ext"] for target in (stage, *sprites)
        for entry in (*target["costumes"], *target["sounds"])
    }
    # A replacement that names no sprite of the project just adds it.
    other = transpile(nn.Linear(4, 1).eval(), value, tmp_path / "extra")
    assert add_to_project(project, other, tmp_path / "extra", replace=True).sprites == (
        "Game", "Cat", "net", "other", "extra",
    )


def test_a_project_without_sprites_gets_one_to_open_on(net, tmp_path):
    from cattorch.project import _Builder

    _, value, result = net
    empty = tmp_path / "empty.sb3"
    _Builder().write(empty)
    project = add_to_project(empty, result, tmp_path / "out")
    assert project.sprites == ("Sprite1", "net")
    blank = sprites_of(project_json(project.path))[0]
    assert blank["blocks"] == {} and blank["visible"] is True
    assert "net" in next(iter(blank["comments"].values()))["text"]
    # Its name gives way to an added sprite's.
    named = transpile(nn.Linear(4, 1).eval(), value, tmp_path / "Sprite1")
    assert add_to_project(empty, [named, result], tmp_path / "out").sprites == (
        "Sprite2", "Sprite1", "net",
    )


def test_added_ids_never_collide_with_the_projects(net, tmp_path):
    """A project holding a copy of the sprite, renamed, keeps its IDs; the added one gets new ones."""
    model, value, result = net
    built = build_project(result, tmp_path / "built", example_inputs=value)
    data, assets = read_archive(built.path, PROJECT_JSON)
    copied = next(target for target in data["targets"] if target["name"] == "net")
    copied["name"] = "copy"
    # Its messages renamed too: two sprites answering "net forward" would
    # both run on every call, which add_to_project refuses.
    for reference in _broadcast_inputs(copied["blocks"]):
        at = 1 if len(reference) == 3 else 0
        if reference[at].startswith("net "):
            reference[at] = "copy " + reference[at].removeprefix("net ")
    path = tmp_path / "copied.sb3"
    write_archive(path, data, assets, PROJECT_JSON)
    original = copy.deepcopy(copied)

    project = add_to_project(path, result, tmp_path / "both")
    assert project.sprites == ("Demo", "copy", "net")
    stage, demo, kept, added = project_json(project.path)["targets"]
    assert kept == original

    def ids(target):
        found = set()
        for section in ("blocks", "variables", "lists", "comments"):
            found |= set(target.get(section, {}))
        return found

    assert not ids(added) & (ids(kept) | ids(demo) | ids(stage))
    emulator = ProjectEmulator(project.path)
    emulator.broadcast("net forward")
    assert emulator.stage.variables["net status"] == interface.OK
    assert [float(item) for item in emulator.stage.lists["net output"]] == pytest.approx(
        model(value).detach().flatten().tolist(), abs=1e-5,
    )


class Adder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    @entrypoint(example_inputs=lambda: (torch.ones(1, 2),), input_names=("input",))
    def add(self, value):
        return self.linear(value)


def test_sprites_that_cant_share_a_project_are_refused(net, tmp_path):
    _, value, result = net
    path = user_project(tmp_path, value.flatten().tolist())
    small = nn.Linear(4, 2).eval()
    twin = transpile(small, value, tmp_path / "a" / "twin", name="net")
    with pytest.raises(ValueError, match="two of the sprites are named 'net'"):
        add_to_project(path, [result, twin], tmp_path / "out", replace=True)
    # Messages Scratch matches ignoring case.
    loud = transpile(small, value, tmp_path / "loud", name="NET")
    with pytest.raises(ValueError, match="'NET forward'.*same message"):
        add_to_project(path, [result, loud], tmp_path / "out", replace=True)
    # ... and added one after the other: both would answer every call.
    with_net = add_to_project(path, result, tmp_path / "with_net")
    with pytest.raises(ValueError, match="'NET forward'.*sprite 'net' of the project"):
        add_to_project(with_net, loud, tmp_path / "out")
    with pytest.raises(ValueError, match="'NET forward'.*sprite 'net' of the project"):
        add_to_project(with_net, loud, tmp_path / "out", replace=True)
    # Two sprites using one global: counter's "counter add input" is the
    # input of a model named "counter add".
    counter = transpile(Adder().eval(), None, tmp_path / "counter")
    assert counter.entrypoints[0].inputs[0].list_name == "counter add input"
    clash = transpile(nn.Linear(2, 2).eval(), torch.ones(1, 2), tmp_path / "clash", name="counter add")
    with pytest.raises(ValueError, match="'counter add input'"):
        add_to_project(path, [counter, clash], tmp_path / "out")
    assert not (tmp_path / "out.sb3").exists()


def edited(tmp_path, path, change) -> object:
    data, assets = read_archive(path, PROJECT_JSON)
    change(data)
    out = tmp_path / f"edited-{len(list(tmp_path.glob('edited-*')))}.sb3"
    write_archive(out, data, assets, PROJECT_JSON)
    return out


def test_globals_that_would_hide_a_variable_are_refused(net, tmp_path):
    _, value, result = net
    path = user_project(tmp_path, value.flatten().tolist())

    # The stage has a variable where the model needs a list.
    def variable_input(data):
        stage = data["targets"][0]
        identifier = next(sid for sid, entry in stage["lists"].items() if entry[0] == "net input")
        stage["variables"][identifier] = ["net input", 0]
        del stage["lists"][identifier]

    with pytest.raises(ValueError, match="global list 'net input'.*global variable of that name"):
        add_to_project(edited(tmp_path, path, variable_input), result, tmp_path / "out")

    # A global named like one of the sprite's own variables.
    def version(data):
        data["targets"][0]["variables"]["mine"] = ["cattorch interface version", "mine"]

    with pytest.raises(
        ValueError, match=r"global variable 'cattorch interface version'.*\(\[cattorch interface "
        r"version\] of \[net\]\)",
    ):
        add_to_project(edited(tmp_path, path, version), result, tmp_path / "out")

    # One of the user's sprites has its own list of a global's name.
    def local_output(data):
        stage, game = data["targets"][0], data["targets"][1]
        identifier = next(sid for sid, entry in stage["lists"].items() if entry[0] == "net output")
        game["lists"][identifier] = stage["lists"].pop(identifier)

    with pytest.raises(ValueError, match="sprite 'Game' of the project has its own list 'net output'"):
        add_to_project(edited(tmp_path, path, local_output), result, tmp_path / "out")
    assert not (tmp_path / "out.sb3").exists()


def test_files_of_the_wrong_kind_and_wrong_arguments(net, tmp_path):
    _, value, result = net
    path = user_project(tmp_path, value.flatten().tolist())
    with pytest.raises(ValueError, match="is a Scratch sprite .*, not a Scratch project"):
        add_to_project(result.path, result, tmp_path / "out")
    with pytest.raises(ValueError, match="is a Scratch project .*, not a Scratch sprite"):
        add_to_project(path, path, tmp_path / "out")
    with pytest.raises(ValueError, match="no such file"):
        add_to_project(tmp_path / "missing", result, tmp_path / "out")
    with pytest.raises(ValueError, match="no such file"):
        add_to_project(path, tmp_path / "missing", tmp_path / "out")
    # A path names its file as transpile and build_project name theirs.
    assert add_to_project(tmp_path / "mine", tmp_path / "net", tmp_path / "out").sprites[-1] == "net"
    not_scratch = tmp_path / "notes.sb3"
    not_scratch.write_text("hello")
    with pytest.raises(ValueError, match="is not a Scratch project"):
        add_to_project(not_scratch, result, tmp_path / "out")
    with pytest.raises(ValueError, match="sprites is empty"):
        add_to_project(path, [], tmp_path / "out")
    with pytest.raises(TypeError, match="sprites must be"):
        add_to_project(path, 3, tmp_path / "out")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="project must be a path or a ProjectResult"):
        add_to_project(result, result, tmp_path / "out")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="replace must be True or False"):
        add_to_project(path, result, tmp_path / "out", replace="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="doesn't name a file"):
        add_to_project(path, result, tmp_path / ".sb3")
