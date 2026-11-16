"""Load exported sprites and run them in the emulator, for tests."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile

import torch
from torch import nn

from cattorch.interface import TOKEN_IDS
from cattorch.transpiler import transpile
from cattorch.util.scratch.emulator import ScratchEmulator


def load_sprite(source) -> dict:
    """The sprite JSON of a ``.sprite3`` file, given its path or a transpile result."""
    with zipfile.ZipFile(getattr(source, "path", source)) as archive:
        return json.loads(archive.read("sprite.json"))


class _InterfaceView:
    """A list or variable view that takes an exported sprite's names as built."""

    def __init__(self, view, translate):
        self._view = view
        self._translate = translate

    def __getitem__(self, name):
        return self._view[self._translate(name)]

    def __setitem__(self, name, value):
        self._view[self._translate(name)] = value

    def get(self, name, default=None):
        return self._view.get(self._translate(name), default)

    def __contains__(self, name):
        return self._translate(name) in self._view

    def __getattr__(self, attribute):
        # The underlying view's storage, which tests reach into by ID.
        return getattr(self._view, attribute)


class SpriteEmulator(ScratchEmulator):
    """An exported sprite uploaded into an empty project, used as a project uses it.

    As in scratch-vm, the globals the sprite uses (``NAME input``,
    ``NAME status``, ...) are created on the stage. Tests name the interface
    as the sprite was built (``input``, ``cattorch status``,
    ``run_procedure("cattorch forward")``), and this translates by the
    documented rule (``ScratchEmulator.public``): a public list or a
    run-time variable is the global ``NAME ...``, and a public block is run
    by broadcasting ``NAME OP`` and waiting. Internal names and constants
    are as given. A sprite without the interface (no ``when I receive
    [NAME ...]`` script) is used as given.
    """

    def __init__(self, sprite: dict, **options):
        super().__init__(sprite, **options)
        prefix = f"{self.name} "
        self.exported = any(
            block.get("opcode") == "event_whenbroadcastreceived"
            and str(block["fields"]["BROADCAST_OPTION"][0]).startswith(prefix)
            for block in self.blocks.values()
        )
        if self.exported:
            self.lists = _InterfaceView(self.lists, self.public)
            self.variables = _InterfaceView(
                self.variables, lambda name: self.public(name, is_list=False),
            )

    def public(self, name: str, *, is_list: bool = True) -> str:
        return super().public(name, is_list=is_list) if self.exported else name

    def run_procedure(self, name: str) -> None:
        if self.exported and name not in self._procedures and name.startswith("cattorch "):
            self.call(name)
        elif self.exported and name not in self._procedures and self._receivers(name):
            # A message, as a result names it (``NAME forward``).
            self.broadcast(name)
        else:
            super().run_procedure(name)


def load_emulator(source) -> SpriteEmulator:
    """An emulator for a ``.sprite3`` file, given its path or a transpile result."""
    return SpriteEmulator(load_sprite(source))


def run_sprite(model: nn.Module, x: torch.Tensor) -> tuple[list[float], list[float]]:
    """Transpile a model, run through emulator, return (expected, actual)."""
    model = model.eval()
    with torch.no_grad():
        expected = model(x).flatten().tolist()

    with tempfile.TemporaryDirectory(prefix="cattorch-sprite-") as directory:
        result = transpile(model, x, os.path.join(directory, "output"))
        emulator = load_emulator(result)
    emulator.lists["input"] = [float(v) for v in x.flatten().tolist()]
    emulator.run()
    actual = [float(v) for v in emulator.lists.get("output", [])]
    return expected, actual


def assert_outputs_close(expected: list[float], actual: list[float], tolerance=1e-4):
    """Each output within ``tolerance`` of the expected value, or equal to it."""
    assert len(expected) == len(actual), (
        f"Length mismatch: expected {len(expected)}, got {len(actual)}"
    )
    for i, (e, a) in enumerate(zip(expected, actual)):
        if e == a:
            continue
        assert abs(e - a) < tolerance, (
            f"Index {i}: expected {e:.6f}, got {a:.6f}"
        )


# ── Generation sprites ───────────────────────────────────────────────────────


def logits(emulator) -> torch.Tensor:
    """The logits the last generation call wrote, in float64."""
    return torch.tensor(emulator.lists["cattorch logits"], dtype=torch.float64)


def logits_of(model, prefix):
    """The model's next-token logits after ``prefix``."""
    with torch.no_grad():
        output = model(torch.tensor([prefix]))
    if not isinstance(output, torch.Tensor):
        output = output[0] if isinstance(output, tuple) else output.logits
    return output[0, -1].flatten()


def assert_logits(emulator, model, prefix, atol=1e-4):
    """The sprite's logits are the model's after ``prefix``."""
    actual = torch.tensor(emulator.lists["cattorch logits"])
    torch.testing.assert_close(
        actual, logits_of(model, prefix), atol=atol, rtol=0,
    )


def prefill_and_decode(emulator, model, prompt, more):
    """Init, prefill ``prompt`` and decode each of ``more``, checking every call's logits."""
    emulator.run_procedure("cattorch init")
    emulator.lists[TOKEN_IDS] = list(prompt)
    emulator.run_procedure("cattorch prefill")
    assert emulator.variables["cattorch status"] == "ok"
    prefix = list(prompt)
    assert_logits(emulator, model, prefix)
    for token in more:
        prefix.append(token)
        emulator.lists[TOKEN_IDS] = [token]
        emulator.run_procedure("cattorch decode")
        assert emulator.variables["cattorch status"] == "ok"
        assert_logits(emulator, model, prefix)
    return prefix
