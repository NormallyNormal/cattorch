"""Run calls from several Scratch scripts at once, interleaved as Scratch can.

Scratch pauses a warp block that has run for 500 ms where it calls a
procedure or repeats a loop, and runs other scripts meanwhile. ``Scripts``
runs each script in its own thread and switches between them at the pause
points given, the chains that start where a procedure's body or a loop's
body does.
"""

from __future__ import annotations

import threading

LOOPS = ("control_repeat", "control_for_each", "control_repeat_until")


def spread(count, samples):
    """Up to ``samples`` of ``range(count)``, first and last included."""
    if count <= samples:
        return list(range(count))
    return sorted({round(index * (count - 1) / (samples - 1)) for index in range(samples)})


def pause_points(emulator, public, loops=True):
    """Where Scratch can pause a script: internal procedure calls, and loop repeats.

    A script calling a public block isn't paused before it runs. Without
    ``loops`` only the procedure calls count.
    """
    # A public block of an exported sprite is the internal procedure its
    # ``when I receive`` script runs (``_cattorch forward``).
    public = {*public, *(f"_{name}" for name in public)}
    points = {
        start for name, start in emulator._procedures.items()
        if name not in public and start is not None
    }
    if loops:
        points.update(
            block["inputs"]["SUBSTACK"][1] for block in emulator.blocks.values()
            if block["opcode"] in LOOPS and block["inputs"].get("SUBSTACK") is not None
        )
    return points


def record_pauses(emulator, points, hits):
    """Make the chain at each of ``points`` append its start to ``hits`` as it runs."""
    for start in points:
        emulator._chains[start] = _recording(emulator, start, hits)


def _recording(emulator, start, hits):
    # Compiled when it first runs, so that it runs the recording chains
    # nested in it.
    chain = None

    def run():
        nonlocal chain
        hits.append(start)
        if chain is None:
            chain = emulator._compile_chain(start)
        chain()
    return run


class Stopped(Exception):
    """A script stopped where it was paused."""


class Script:
    """A Scratch script making calls."""

    def __init__(self, name, calls):
        self.name = name
        self.calls = calls
        self.results: list[tuple] = []
        self.pauses = 0
        self.budget = 0
        self.stop = False
        self.done = False
        self.error: BaseException | None = None
        self.thread: threading.Thread | None = None


class Scripts:
    """Run scripts on one sprite, switching between them where they can pause.

    ``run_call(emulator, call)`` runs each call and returns what to record of
    it; the chains starting at ``pauses`` are where scripts can pause.
    """

    def __init__(self, emulator, finished, run_call, pauses):
        self.emulator = emulator
        self.finished = finished
        self.run_call = run_call
        self.condition = threading.Condition()
        self.running: Script | None = None
        self.local = threading.local()
        for start in pauses:
            emulator._chains[start] = self._pausing(start)

    def _pausing(self, start):
        # Compiled when it first runs, so that it runs the pausing chains
        # nested in it.
        chain = None

        def run():
            nonlocal chain
            script = getattr(self.local, "script", None)
            if script is not None:
                script.pauses += 1
                if script.pauses > script.budget:
                    self._pause(script)
            if chain is None:
                chain = self.emulator._compile_chain(start)
            chain()
        return run

    def _pause(self, script):
        # Each script has its own procedure arguments in Scratch.
        frames = self.emulator._arguments
        saved = frames[:]
        frames.clear()
        with self.condition:
            self.running = None
            self.condition.notify_all()
            self.condition.wait_for(lambda: self.running is script)
        frames[:] = saved
        if script.stop:
            raise Stopped

    def _body(self, script):
        self.local.script = script
        with self.condition:
            self.condition.wait_for(lambda: self.running is script)
        try:
            for call in script.calls:
                result = (script.name, *self.run_call(self.emulator, call))
                script.results.append(result)
                self.finished.append(result)
        except Stopped:
            pass
        except BaseException as error:  # noqa: BLE001 - reraised in the main thread
            script.error = error
        with self.condition:
            script.done = True
            self.running = None
            self.condition.notify_all()

    def run(self, script, pauses=None):
        """Let ``script`` run until it pauses ``pauses`` more times, or to its end."""
        if script.done:
            return
        script.budget = script.pauses + pauses if pauses is not None else float("inf")
        if script.thread is None:
            script.thread = threading.Thread(target=self._body, args=(script,), daemon=True)
            script.thread.start()
        with self.condition:
            self.running = script
            self.condition.notify_all()
            self.condition.wait_for(lambda: self.running is None)
        if script.error is not None:
            raise script.error

    def stop(self, script):
        """Stop ``script`` where it is paused."""
        script.stop = True
        self.run(script)

    def play(self, scripts, schedule):
        """Run ``scripts`` as ``schedule`` says, then each one to its end.

        ``schedule`` is a list of ``(script index, pauses)``; None pauses runs
        the script to its end, and "stop" stops it where it is paused.
        """
        for index, pauses in schedule:
            if pauses == "stop":
                self.stop(scripts[index])
            else:
                self.run(scripts[index], pauses)
        for script in scripts:
            self.run(script)
