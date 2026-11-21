"""
emulator.py
-----------
Minimal Scratch block emulator for testing transpiled sprites.

Supports the subset of Scratch emitted by cattorch:
  - Variables: set, change by
  - Lists: delete all, add, replace item, item of
  - Control: repeat, if/else
  - Operators: add, multiply, mod, gt

Usage
-----
    from cattorch.util.scratch.emulator import ScratchEmulator

    emu = ScratchEmulator(sprite_dict)
    emu.lists["input"] = [1.0, 2.0, 3.0]
    emu.run()
    result = emu.lists["output"]
"""

import decimal
import json
import math
import re
import sys
import time
from collections import Counter
from contextlib import contextmanager

from cattorch.util.scratch import jsmath
from cattorch.util.scratch.sharding import SCRATCH_LIST_LIMIT


_ARITHMETIC = frozenset({
    "operator_add", "operator_subtract", "operator_multiply", "operator_mod",
    "operator_divide",
})
_MISSING = object()
_SIMPLE_ARITHMETIC = {
    "operator_add": "+", "operator_subtract": "-", "operator_multiply": "*",
}
_ARITHMETIC_FACTORIES: dict = {}
_JS_RADIX_NUMBER = re.compile(r"0[xX][0-9a-fA-F]+|0[oO][0-7]+|0[bB][01]+")
# JavaScript's Number reads ASCII digits only (Python's \d matches any digit).
_JS_DECIMAL_NUMBER = re.compile(
    r"[+-]?(?:Infinity|(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)"
)
# What JavaScript's String.prototype.trim removes: WhiteSpace and
# LineTerminator. Python's str.strip differs (it strips \x1c-\x1f and \x85,
# and keeps \ufeff).
_JS_WHITESPACE = (
    "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007"
    "\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)
# Integers beyond this are doubles in JavaScript, not exact.
_EXACT_INTEGER = 2**53


def _js_number_string(number: float) -> str:
    """``String(number)`` in JavaScript (ECMAScript Number::toString)."""
    if number != number:
        return "NaN"
    if number in (math.inf, -math.inf):
        return "Infinity" if number > 0 else "-Infinity"
    if number == 0:
        return "0"
    sign = "-" if number < 0 else ""
    # repr gives the shortest digits that round-trip, as JavaScript uses.
    _, digit_tuple, exponent = decimal.Decimal(repr(abs(number))).as_tuple()
    digits = "".join(map(str, digit_tuple)).lstrip("0")
    stripped = digits.rstrip("0")
    exponent = int(exponent) + len(digits) - len(stripped)
    digits = stripped
    point = len(digits) + exponent
    k, n = len(digits), point
    if k <= n <= 21:
        text = digits + "0" * (n - k)
    elif 0 < n <= 21:
        text = f"{digits[:n]}.{digits[n:]}"
    elif -6 < n <= 0:
        text = "0." + "0" * -n + digits
    else:
        power = f"e{'+' if n - 1 >= 0 else '-'}{abs(n - 1)}"
        text = (digits if k == 1 else f"{digits[0]}.{digits[1:]}") + power
    return sign + text


def _js_string(value) -> str:
    """``Cast.toString``: JavaScript's ``String(value)``."""
    kind = value.__class__
    if kind is str:
        return value
    if kind is bool:
        return "true" if value else "false"
    if kind is int:
        return str(value) if -_EXACT_INTEGER <= value <= _EXACT_INTEGER else (
            _js_number_string(float(value))
        )
    if kind is float:
        if value.is_integer() and abs(value) <= _EXACT_INTEGER:
            return str(int(value))
        return _js_number_string(value)
    return str(value)


def _js_value(value):
    """A value loaded from a project as JavaScript holds it: a JSON integer
    past 2**53 is a double."""
    if value.__class__ is int and not -_EXACT_INTEGER <= value <= _EXACT_INTEGER:
        return float(value)
    return value


def _js_boolean(value) -> bool:
    """``Cast.toBoolean``: the strings ``""``, ``"0"`` and ``"false"`` are false."""
    if value is True or value is False:
        return value
    if value.__class__ is str:
        return not (value == "" or value == "0" or value.lower() == "false")
    return bool(value) and value == value


def _astral(text: str) -> bool:
    """Whether ``text`` has characters JavaScript stores as two UTF-16 units."""
    return not text.isascii() and max(text) > "\uffff"


def _utf16_length(text: str) -> int:
    """JavaScript's ``text.length``: UTF-16 code units, not characters."""
    if not _astral(text):
        return len(text)
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def _utf16_unit(text: str, index: int) -> str:
    """JavaScript's ``text.charAt(index)``: one UTF-16 code unit."""
    if not _astral(text):
        return text[index]
    units = text.encode("utf-16-le", "surrogatepass")
    return units[2 * index:2 * index + 2].decode("utf-16-le", "surrogatepass")


def _js_join(left: str, right: str) -> str:
    """Concatenate as JavaScript does: two halves of a surrogate pair make one character."""
    if (
        left and right and "\ud800" <= left[-1] <= "\udbff" and "\udc00" <= right[0] <= "\udfff"
    ):
        pair = (left[-1] + right[0]).encode("utf-16-le", "surrogatepass").decode("utf-16-le")
        return left[:-1] + pair + right[1:]
    return left + right


def _utf16_key(text: str) -> bytes:
    """A key ordering strings as JavaScript's ``<`` does, by UTF-16 code units."""
    return text.encode("utf-16-be", "surrogatepass")


def _js_lower(text: str) -> str:
    """JavaScript's ``toLowerCase``."""
    return text.lower()


def _js_is_int(value) -> bool:
    """``Cast.isInt``, which ``pick random`` reads its raw bounds with."""
    if value is True or value is False:
        return True
    if value.__class__ in (int, float):
        return value != value or float(value).is_integer()
    if value.__class__ is str:
        return "." not in value
    return False


def _arithmetic_factory(symbol: str, left_kind: str, right_kind: str):
    """Build ``factory(left, right, numeric) -> closure`` for one operand mix."""
    lines = ["def factory(left, right, numeric):", "    def run():"]
    names = []
    for name, kind in (("a", left_kind), ("b", right_kind)):
        source = "left" if name == "a" else "right"
        if kind == "const":
            names.append(source)
            continue
        lines.append(f"        {name} = {source}()")
        if kind == "raw":
            lines.append(
                f"        if {name}.__class__ is not float and {name}.__class__ is not int:"
            )
            lines.append(f"            {name} = numeric({name})")
        # Scratch's Cast.toNumber turns NaN into 0.
        lines.append(f"        if {name} != {name}:")
        lines.append(f"            {name} = 0")
        names.append(name)
    if symbol == "*":
        # JavaScript multiplies doubles: an integer product past 2**53 rounds.
        lines.append(f"        product = {names[0]} * {names[1]}")
        lines.append(
            "        if product.__class__ is int and not "
            f"-{_EXACT_INTEGER} <= product <= {_EXACT_INTEGER}:"
        )
        lines.append("            return float(product)")
        lines.append("        return product")
    else:
        lines.append(f"        return {names[0]} {symbol} {names[1]}")
    lines.append("    return run")
    namespace: dict = {}
    exec("\n".join(lines), namespace)  # noqa: S102 - generated from fixed templates
    return namespace["factory"]


def _zero():
    return 0


def _nothing():
    return None


def _constant(value):
    def constant():
        return value
    constant.constant = value
    return constant


# A read of a list sharded hundreds of ways is a sum that many reporters deep,
# and the emulator compiles and runs each reporter as a nested call.
_RECURSION_LIMIT = 100_000


@contextmanager
def _deep_reporters():
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, _RECURSION_LIMIT))
    try:
        yield
    finally:
        sys.setrecursionlimit(limit)


def _data_references(blocks: dict):
    """``(kind, id, name)`` for every variable and list reference in ``blocks``."""
    for block in blocks.values():
        if isinstance(block, list):
            if len(block) >= 3 and block[0] in (12, 13):
                yield ("variable" if block[0] == 12 else "list"), block[2], block[1]
            continue
        fields = block.get("fields", {})
        for key, kind in (("VARIABLE", "variable"), ("LIST", "list")):
            field = fields.get(key)
            if field is not None and len(field) >= 2 and field[1] is not None:
                yield kind, field[1], field[0]
        for value in block.get("inputs", {}).values():
            if not isinstance(value, list):
                continue
            for part in value[1:]:
                if isinstance(part, list) and len(part) >= 3 and part[0] in (12, 13):
                    yield ("variable" if part[0] == 12 else "list"), part[2], part[1]


class _StopScript(Exception):
    """``stop this script``: ends the procedure running it, or the script."""


class _NameView:
    """Dict-like view that maps display names to underlying ID-keyed storage.

    Allows ``emu.lists["input"] = [...]`` while internally everything is
    stored by Scratch ID so that duplicate display names (e.g. two local
    ``_index_map`` lists) are kept separate.
    """

    def __init__(self, by_id: dict, name_to_ids: dict):
        self._by_id = by_id
        self._name_to_ids = name_to_ids

    def __getitem__(self, name):
        ids = self._name_to_ids.get(name, [])
        if not ids:
            raise KeyError(name)
        return self._by_id[ids[0]]

    def __setitem__(self, name, value):
        ids = self._name_to_ids.get(name, [])
        if not ids:
            raise KeyError(name)
        self._by_id[ids[0]] = value

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default

    def __contains__(self, name):
        return name in self._name_to_ids


class ScratchEmulator:
    """Run one sprite's blocks.

    With ``stage``, the sprite also sees the stage's variables and lists, as
    in a Scratch project; :class:`ProjectEmulator` builds those for a whole
    ``.sb3``. ``on_broadcast`` is called with each broadcast message after it
    is recorded, ``on_ask`` with each ``ask and wait`` question to return the
    answer, and ``said`` records what ``say`` blocks showed.
    """

    def __init__(self, sprite: dict, *, stage: "ScratchEmulator | None" = None):
        self.name = sprite.get("name", "")
        self.blocks = sprite["blocks"]
        self._procedures = self._find_procedures()
        # The argument values of the procedure calls running, innermost last.
        self._arguments: list[dict] = []
        self.opcode_counts = Counter()
        self._costume_names = [
            costume.get("name", "") for costume in sprite.get("costumes", [])
        ]
        self._current_costume = int(sprite.get("currentCostume", 0))
        self.broadcasts = []
        self.on_broadcast = None
        self.on_ask = None
        self.said: list = []
        self._answer = stage._answer if stage is not None else [""]
        # One timer per project, as in Scratch: when it was last reset.
        self._timer_start = stage._timer_start if stage is not None else [time.perf_counter()]
        self._chains: dict = {}

        # Variables stored by ID; a sprite shares the stage's dictionaries,
        # since data IDs are unique across a project.
        self._vars = stage._vars if stage is not None else {}
        self._var_name_to_ids = {}
        for sid, entry in sprite.get("variables", {}).items():
            name, value = entry[0], _js_value(entry[1])
            self._vars[sid] = value
            self._var_name_to_ids.setdefault(name, []).append(sid)
        # The sprite's own data shadows stage data with the same name.
        for name, ids in (stage._var_name_to_ids if stage else {}).items():
            self._var_name_to_ids.setdefault(name, []).extend(ids)

        # Lists stored by ID
        self._lists = stage._lists if stage is not None else {}
        self._list_name_to_ids = {}
        for sid, entry in sprite.get("lists", {}).items():
            name, data = entry[0], [_js_value(value) for value in entry[1]]
            self._lists[sid] = data
            self._list_name_to_ids.setdefault(name, []).append(sid)
        for name, ids in (stage._list_name_to_ids if stage else {}).items():
            self._list_name_to_ids.setdefault(name, []).extend(ids)

        if stage is None and not sprite.get("isStage"):
            self._add_unresolved_globals()

        # Public name-based views for test convenience
        self.variables = _NameView(self._vars, self._var_name_to_ids)
        self.lists = _NameView(self._lists, self._list_name_to_ids)

    def _add_unresolved_globals(self) -> None:
        """Create the variables and lists the blocks use without defining, as globals.

        A sprite loaded on its own is like one uploaded into an empty project:
        scratch-vm creates on the stage each variable or list its blocks
        refer to that neither it nor the stage defines (by its ID and name, a
        variable holding 0 and a list empty), and binds other references of
        the same name and type to it.
        """
        created: dict[tuple[str, str], str] = {}
        remap: dict[str, str] = {}
        for kind, identifier, name in _data_references(self.blocks):
            if identifier in (self._vars if kind == "variable" else self._lists):
                continue
            winner = created.setdefault((kind, name), identifier)
            if winner != identifier:
                remap[identifier] = winner
                continue
            if kind == "variable":
                self._vars[identifier] = 0
                self._var_name_to_ids.setdefault(name, []).append(identifier)
            else:
                self._lists[identifier] = []
                self._list_name_to_ids.setdefault(name, []).append(identifier)
        if remap:
            from cattorch.util.scratch.remap import remap_ids

            self.blocks = remap_ids(self.blocks, remap)

    # ── The interface of an exported sprite ──────────────────────────────

    def public(self, name: str, *, is_list: bool = True) -> str:
        """The name a project uses for this exported sprite's ``name``, as built.

        A public list (``input``, ``cattorch token ids``) and a run-time
        variable (``cattorch status``) are globals, ``NAME input``,
        ``NAME status``; an internal name (``_...``), a constant
        (``cattorch max context``), and a name already global are as given.
        """
        from cattorch import interface

        if name.startswith((interface.INTERNAL_PREFIX, f"{self.name} ")):
            return name
        if is_list or name in interface.RUNTIME_VARIABLES:
            return interface.global_name(self.name, name)
        return name

    def call(self, block: str) -> None:
        """Run an exported sprite's ``block`` (``cattorch forward``) as a project does.

        It broadcasts ``NAME forward`` and waits: the ``when I receive``
        scripts run to their end.
        """
        from cattorch import interface

        message = interface.message(self.name, block)
        if not self._receivers(message):
            raise ValueError(f"nothing in sprite {self.name!r} receives {message!r}")
        self.broadcast(message)

    def _receivers(self, message: str) -> list[dict]:
        key = message.lower()
        return [
            block for block in self.blocks.values()
            if block.get("opcode") == "event_whenbroadcastreceived" and block.get("topLevel")
            and str(block["fields"]["BROADCAST_OPTION"][0]).lower() == key
        ]

    def broadcast(self, message: str) -> None:
        """Run this sprite's ``when I receive [message]`` scripts, one after another, to their end."""
        for hat in self._receivers(str(message)):
            with _deep_reporters():
                self._exec_script(hat.get("next"))

    def run(self, root_index: int = 0):
        """Run a top-level block stack.

        Parameters
        ----------
        root_index : int
            Which top-level stack to run (0-based). Default runs the first.
        """
        roots = self._find_roots()
        if root_index >= len(roots):
            raise ValueError(f"root_index {root_index} out of range, found {len(roots)} roots")
        with _deep_reporters():
            self._exec_script(roots[root_index])

    def run_procedure(self, name: str) -> None:
        """Run a named custom block without relying on top-level stack order."""
        if name not in self._procedures:
            available = ", ".join(sorted(self._procedures)) or "none"
            raise ValueError(f"procedure {name!r} not found; available: {available}")
        with _deep_reporters():
            self._exec_script(self._procedures[name])

    def _find_roots(self) -> list[str]:
        roots = [
            bid for bid, block in self.blocks.items()
            if block.get("topLevel") and block.get("parent") is None
            and block.get("opcode") != "procedures_definition"
        ]
        if not roots:
            raise ValueError("no topLevel roots found")
        return roots

    def _argument_names(self, proccode: str) -> dict[str, str]:
        """Map the argument IDs of ``proccode``'s prototype to the argument names."""
        for block in self.blocks.values():
            mutation = block.get("mutation", {})
            if block.get("opcode") == "procedures_prototype" and mutation.get("proccode") == proccode:
                return dict(zip(
                    json.loads(mutation.get("argumentids") or "[]"),
                    json.loads(mutation.get("argumentnames") or "[]"),
                ))
        return {}

    def _find_procedures(self) -> dict[str, str | None]:
        procedures = {}
        for block in self.blocks.values():
            if block.get("opcode") != "procedures_definition":
                continue
            custom_input = block.get("inputs", {}).get("custom_block")
            if not custom_input or len(custom_input) < 2:
                continue
            prototype = self.blocks.get(custom_input[1])
            if prototype is None:
                continue
            proccode = prototype.get("mutation", {}).get("proccode")
            if proccode:
                procedures[proccode] = block.get("next")
        return procedures

    # ── Execution ────────────────────────────────────────────────────────
    #
    # Blocks are compiled once into Python closures, so a loop body runs as
    # direct calls instead of re-reading the block JSON on every iteration.
    # A chain is compiled the first time it runs and cached by its first
    # block; reporters evaluated through ``_eval_reporter`` compile afresh,
    # since tests may edit those blocks between calls.

    def _exec_script(self, block_id: str | None):
        """Run a script, or a procedure as a call to it does, to its end or ``stop``."""
        try:
            self._exec_chain(block_id)
        except _StopScript:
            pass

    def _exec_chain(self, block_id: str | None):
        if block_id is None:
            return
        chain = self._chains.get(block_id)
        if chain is None:
            chain = self._chains[block_id] = self._compile_chain(block_id)
        chain()

    def _compile_chain(self, block_id: str | None):
        statements = []
        while block_id is not None:
            block = self.blocks[block_id]
            statements.append(self._compile_block(block))
            block_id = block.get("next")
        statements = tuple(statements)
        if len(statements) == 1:
            return statements[0]

        def run():
            for statement in statements:
                statement()
        return run

    def _chain_runner(self, block_id: str | None):
        """The compiled chain at ``block_id``, a loop or branch body.

        Bodies nest as a tree, so they compile with their parent block;
        only procedure calls look their chain up when they run.
        """
        if block_id is None:
            return _nothing
        chain = self._chains.get(block_id)
        if chain is None:
            chain = self._chains[block_id] = self._compile_chain(block_id)
        return chain

    def _compile_block(self, block: dict):
        opcode = block["opcode"]
        inputs = block.get("inputs", {})
        fields = block.get("fields", {})
        counts = self.opcode_counts
        variables = self._vars
        lists = self._lists
        value_of = self._compile_input

        if opcode == "data_setvariableto":
            var_id = fields["VARIABLE"][1]
            value = value_of(inputs["VALUE"])

            def run():
                counts[opcode] += 1
                variables[var_id] = value()

        elif opcode == "data_changevariableby":
            var_id = fields["VARIABLE"][1]
            value = value_of(inputs["VALUE"])

            numeric = self._scratch_numeric

            def run():
                counts[opcode] += 1
                variables[var_id] = numeric(variables[var_id]) + numeric(value())

        elif opcode == "data_deletealloflist":
            list_id = fields["LIST"][1]

            def run():
                counts[opcode] += 1
                lists[list_id] = []

        elif opcode == "data_addtolist":
            list_id = fields["LIST"][1]
            value = value_of(inputs["ITEM"])

            def run():
                counts[opcode] += 1
                item = value()
                target = lists[list_id]
                if len(target) < SCRATCH_LIST_LIMIT:
                    target.append(item)

        elif opcode == "data_replaceitemoflist":
            list_id = fields["LIST"][1]
            index_of = value_of(inputs["INDEX"])
            value = value_of(inputs["ITEM"])
            to_index = self._list_index

            def run():
                counts[opcode] += 1
                target = lists[list_id]
                index = to_index(index_of(), len(target))
                item = value()
                if index:
                    target[index - 1] = item

        elif opcode == "data_deleteoflist":
            list_id = fields["LIST"][1]
            index_of = value_of(inputs["INDEX"])
            to_index = self._list_index

            def run():
                counts[opcode] += 1
                target = lists[list_id]
                index = to_index(index_of(), len(target), accept_all=True)
                if index == "all":
                    lists[list_id] = []
                elif index:
                    target.pop(index - 1)

        elif opcode == "control_repeat":
            times_of = value_of(inputs["TIMES"])
            body = self._chain_runner(self._get_substack(inputs.get("SUBSTACK")))
            scratch_round = self._scratch_round

            def run():
                counts[opcode] += 1
                for _ in range(scratch_round(times_of())):
                    body()

        elif opcode == "control_for_each":
            times_of = value_of(inputs["VALUE"])
            var_id = fields["VARIABLE"][1]
            body = self._chain_runner(self._get_substack(inputs.get("SUBSTACK")))

            number = self._scratch_number

            def run():
                # scratch-vm compares the count with Number(VALUE), evaluated
                # again before each iteration.
                counts[opcode] += 1
                index = 0
                while True:
                    times = times_of()
                    if times.__class__ is not int and times.__class__ is not float:
                        times = number(times)
                        if times is None:
                            break
                    if not index < times:
                        break
                    index += 1
                    variables[var_id] = index
                    body()

        elif opcode == "control_repeat_until":
            condition = value_of(inputs["CONDITION"])
            body = self._chain_runner(self._get_substack(inputs.get("SUBSTACK")))

            def run():
                counts[opcode] += 1
                while not _js_boolean(condition()):
                    body()

        elif opcode == "control_if":
            condition = value_of(inputs["CONDITION"])
            body = self._chain_runner(self._get_substack(inputs.get("SUBSTACK")))

            def run():
                counts[opcode] += 1
                if _js_boolean(condition()):
                    body()

        elif opcode == "control_if_else":
            condition = value_of(inputs["CONDITION"])
            body = self._chain_runner(self._get_substack(inputs.get("SUBSTACK")))
            other = self._chain_runner(self._get_substack(inputs.get("SUBSTACK2")))

            def run():
                counts[opcode] += 1
                if _js_boolean(condition()):
                    body()
                else:
                    other()

        elif opcode == "procedures_call":
            mutation = block.get("mutation", {})
            proccode = mutation.get("proccode")
            procedures = self._procedures
            exec_chain = self._exec_chain
            argument_ids = json.loads(mutation.get("argumentids") or "[]")
            if not argument_ids:
                def run():
                    counts[opcode] += 1
                    if proccode not in procedures:
                        raise ValueError(f"procedure not found: {proccode}")
                    try:
                        exec_chain(procedures[proccode])
                    except _StopScript:
                        pass
            else:
                names = self._argument_names(proccode)
                arguments = [
                    (names.get(identifier), value_of(inputs[identifier]))
                    for identifier in argument_ids if identifier in inputs
                ]
                frames = self._arguments

                def run():
                    counts[opcode] += 1
                    if proccode not in procedures:
                        raise ValueError(f"procedure not found: {proccode}")
                    frames.append({name: value() for name, value in arguments})
                    try:
                        exec_chain(procedures[proccode])
                    except _StopScript:
                        pass
                    finally:
                        frames.pop()

        elif opcode == "control_stop":
            option = fields["STOP_OPTION"][0]
            if option != "this script":
                raise NotImplementedError(f"stop {option} is not supported")

            def run():
                counts[opcode] += 1
                raise _StopScript

        elif opcode == "looks_switchcostumeto":
            requested_of = value_of(inputs["COSTUME"])

            def run():
                counts[opcode] += 1
                self._switch_costume(requested_of())

        elif opcode in {"event_broadcast", "event_broadcastandwait"}:
            message = value_of(inputs["BROADCAST_INPUT"])
            broadcasts = self.broadcasts

            def run():
                counts[opcode] += 1
                sent = message()
                broadcasts.append(sent)
                # Receivers run to completion, as ``broadcast and wait`` does.
                if self.on_broadcast is not None:
                    self.on_broadcast(str(sent))
                else:
                    self.broadcast(str(sent))

        elif opcode == "sensing_askandwait":
            question = value_of(inputs["QUESTION"])
            answer = self._answer

            def run():
                counts[opcode] += 1
                asked = question()
                if self.on_ask is None:
                    raise RuntimeError(f"ask and wait without an answer: {asked}")
                answer[0] = self.on_ask(asked)

        elif opcode == "sensing_resettimer":
            started = self._timer_start

            def run():
                counts[opcode] += 1
                started[0] = time.perf_counter()

        elif opcode in {"looks_say", "looks_think"}:
            message = value_of(inputs["MESSAGE"])
            said = self.said

            def run():
                counts[opcode] += 1
                said.append(message())

        elif opcode in {
            "procedures_definition", "procedures_prototype", "event_whenflagclicked",
            "event_whenbroadcastreceived", "event_whenkeypressed", "event_whenthisspriteclicked",
            "event_whenstageclicked",
        }:
            # Definitions are metadata/entry points rather than executable
            # commands when encountered outside a procedure call.
            def run():
                counts[opcode] += 1

        else:
            def run():
                counts[opcode] += 1
                raise NotImplementedError(f"unknown opcode: {opcode}")
        return run

    def _switch_costume(self, requested) -> None:
        """scratch-vm's ``_setCostume``: a number is a position; text is a name
        (matched exactly), ``next costume``, ``previous costume``, or a number."""
        names = self._costume_names
        if requested.__class__ in (int, float) and requested is not True and (
            requested is not False
        ):
            self._set_costume(requested - 1)
            return
        text = _js_string(requested)
        if text in names:
            self._current_costume = names.index(text)
        elif text == "next costume":
            self._set_costume(self._current_costume + 1)
        elif text == "previous costume":
            self._set_costume(self._current_costume - 1)
        else:
            number = self._scratch_number(requested)
            whitespace = isinstance(requested, str) and requested.strip(_JS_WHITESPACE) == ""
            if number is not None and number == number and not whitespace:
                self._set_costume(number - 1)

    def _set_costume(self, index) -> None:
        """``RenderedTarget.setCostume``: rounded, wrapped into the costumes.

        ``MathUtil.wrapClamp`` wraps in doubles, so a huge index lands where
        its rounding puts it, as in Scratch, not at its exact remainder.
        """
        if not math.isfinite(index):
            index = 0
        index = float(jsmath.round(index))
        count = len(self._costume_names)
        if count:
            wrapped = index - float(math.floor(index / count)) * count
            self._current_costume = int(wrapped) if abs(wrapped) <= _EXACT_INTEGER else wrapped

    def _eval_input(self, input_spec):
        """Evaluate a Scratch input specification and return a Python value."""
        return self._compile_input(input_spec)()

    def _eval_literal(self, spec):
        """Evaluate a literal value spec like [4, "10"] or [12, "var", "id"]."""
        return self._compile_literal(spec)()

    def _eval_reporter(self, block_id: str):
        """Evaluate a reporter block (one that returns a value)."""
        return self._compile_reporter(block_id)()

    def _compile_input(self, input_spec):
        """A closure returning the value of a Scratch input specification."""
        if input_spec is None:
            return _zero
        type_code = input_spec[0]
        if type_code == 1:
            # [1, literal] or [1, [type, value]], or [1, shadow_id] for a menu
            # such as a costume menu.
            if isinstance(input_spec[1], str) and input_spec[1] in self.blocks:
                return self._compile_reporter(input_spec[1])
            return self._compile_literal(input_spec[1])
        if type_code == 2:
            # [2, block_id] — block reference (used for SUBSTACK)
            return self._compile_reporter(input_spec[1])
        if type_code == 3:
            # [3, block_id_or_var_ref, fallback]
            ref = input_spec[1]
            if isinstance(ref, str):
                return self._compile_reporter(ref)
            if isinstance(ref, list):
                # Variable or list reference: [12, name, id] or [13, name, id]
                return self._compile_literal(ref)
            return self._compile_literal(input_spec[2]) if len(input_spec) > 2 else _zero
        return _zero

    def _compile_literal(self, spec):
        if not isinstance(spec, list):
            return _constant(self._to_number(spec))
        lit_type = spec[0]
        if lit_type in (4, 5, 6, 7, 8):
            return _constant(self._to_number(spec[1]))
        if lit_type in (10, 11):
            # String literal, or a broadcast menu [11, display_name, broadcast_id]
            return _constant(spec[1])
        if lit_type == 12:
            # Variable reference: [12, display_name, var_id]
            var_id = spec[2]
            get = self._vars.get
            return lambda: get(var_id, 0)
        if lit_type == 13:
            # List reference: [13, display_name, list_id]
            list_id = spec[2]
            get = self._lists.get
            return lambda: get(list_id, [])
        return _constant(self._to_number(spec[1]) if len(spec) > 1 else 0)

    def _compile_numeric(self, input_spec):
        """Like ``_compile_input``, cast with Scratch's numeric rules."""
        value = self._compile_input(input_spec)
        constant = getattr(value, "constant", _MISSING)
        if constant is not _MISSING:
            return _constant(self._scratch_numeric(constant))
        numeric = self._scratch_numeric

        # Scratch's Cast.toNumber turns NaN into 0, even for a number.
        def run():
            result = value()
            kind = result.__class__
            if kind is float or kind is int:
                return result if result == result else 0
            return numeric(result)
        return run

    def _compile_reporter(self, block_id: str):
        block = self.blocks[block_id]
        opcode = block["opcode"]
        inputs = block.get("inputs", {})
        fields = block.get("fields", {})
        value_of = self._compile_input
        number_of = self._compile_numeric

        if opcode in _ARITHMETIC:
            if opcode in _SIMPLE_ARITHMETIC:
                return self._compile_arithmetic(
                    _SIMPLE_ARITHMETIC[opcode], inputs["NUM1"], inputs["NUM2"],
                )
            left = number_of(inputs["NUM1"])
            right = number_of(inputs["NUM2"])
            if opcode == "operator_mod":
                def mod():
                    # scratch-vm: n % m (a truncated remainder), plus m when
                    # the result's sign differs from m's. A remainder by zero,
                    # or of an infinity, is NaN.
                    a, b = left(), right()
                    if (
                        a.__class__ is int and b.__class__ is int and b != 0
                        and -_EXACT_INTEGER <= a <= _EXACT_INTEGER
                    ):
                        return a % b
                    if b == 0 or not math.isfinite(a) or b != b:
                        return math.nan
                    result = math.fmod(a, b)
                    return result + b if result / b < 0 else result
                mod.numeric = True
                return mod

            def divide():
                a, b = left(), right()
                if b != 0:
                    return a / b
                if a == 0:
                    return math.nan
                return math.copysign(math.inf, a * math.copysign(1, b))
            divide.numeric = True
            return divide

        if opcode in {"operator_equals", "operator_gt", "operator_lt"}:
            left = value_of(inputs["OPERAND1"])
            right = value_of(inputs["OPERAND2"])
            compare = self._scratch_compare
            if opcode == "operator_equals":
                return lambda: compare(left(), right()) == 0
            if opcode == "operator_gt":
                return lambda: compare(left(), right()) > 0
            return lambda: compare(left(), right()) < 0

        if opcode in {"operator_and", "operator_or"}:
            left = value_of(inputs["OPERAND1"])
            right = value_of(inputs["OPERAND2"])
            if opcode == "operator_and":
                def both():
                    a, b = left(), right()
                    return _js_boolean(a) and _js_boolean(b)
                return both

            def either():
                a, b = left(), right()
                return _js_boolean(a) or _js_boolean(b)
            return either

        if opcode == "operator_not":
            operand = value_of(inputs["OPERAND"])
            return lambda: not _js_boolean(operand())

        if opcode == "operator_mathop":
            result = self._compile_mathop(number_of(inputs["NUM"]), fields["OPERATOR"][0])
            result.numeric = True
            return result

        if opcode == "operator_round":
            value = number_of(inputs["NUM"])

            def round_half_up():
                # JavaScript's Math.round, which rounds once: floor(x + 0.5)
                # rounds 0.49999999999999994 up.
                return jsmath.round(value())
            round_half_up.numeric = True
            return round_half_up

        if opcode == "sensing_answer":
            answer = self._answer
            return lambda: answer[0]

        if opcode == "sensing_timer":
            started = self._timer_start
            return lambda: time.perf_counter() - started[0]

        if opcode == "sensing_dayssince2000":
            epoch = 946_684_800  # 2000-01-01 in Unix time
            return lambda: (time.time() - epoch) / 86_400

        if opcode == "operator_random":
            low_of = value_of(inputs["FROM"])
            high_of = value_of(inputs["TO"])
            numeric = self._scratch_numeric

            def random_value():
                import random
                low_value, high_value = low_of(), high_of()
                low, high = numeric(low_value), numeric(high_value)
                low, high = min(low, high), max(low, high)
                if low == high:
                    return low
                # Whole-number bounds, by Cast.isInt of the raw values ("1.0"
                # isn't one), pick a whole number.
                if _js_is_int(low_value) and _js_is_int(high_value):
                    return low + math.floor(random.random() * (high + 1 - low))
                return random.random() * (high - low) + low
            return random_value

        if opcode == "argument_reporter_string_number":
            name = fields["VALUE"][0]
            frames = self._arguments

            def argument():
                # Scratch reports 0 for an argument outside its procedure.
                return frames[-1].get(name, 0) if frames else 0
            return argument

        if opcode == "data_itemoflist":
            list_id = fields["LIST"][1]
            index_of = value_of(inputs["INDEX"])
            lists = self._lists
            to_index = self._list_index

            def item():
                values = lists.get(list_id, ())
                index = to_index(index_of(), len(values))
                return values[index - 1] if index else ""
            return item

        if opcode == "data_lengthoflist":
            list_id = fields["LIST"][1]
            lists = self._lists
            return lambda: len(lists.get(list_id, ()))

        if opcode == "data_itemnumoflist":
            list_id = fields["LIST"][1]
            item_of = value_of(inputs["ITEM"])
            lists = self._lists
            compare = self._scratch_compare

            def item_number():
                wanted = item_of()
                # Scratch returns a 1-based index, or 0 if not found. The VM
                # uses Cast.compare, so "1" also matches " 1" and "01".
                for index, value in enumerate(lists.get(list_id, ())):
                    if compare(value, wanted) == 0:
                        return index + 1
                return 0
            return item_number

        if opcode == "operator_length":
            text = value_of(inputs["STRING"])
            return lambda: _utf16_length(_js_string(text()))

        if opcode == "operator_letter_of":
            letter_of = self._compile_numeric(inputs["LETTER"])
            text_of = value_of(inputs["STRING"])

            def letter():
                # scratch-vm: charAt(toNumber(LETTER) - 1), of UTF-16 units.
                index = letter_of() - 1
                string = _js_string(text_of())
                if index < 0 or index >= _utf16_length(string):
                    return ""
                return _utf16_unit(string, int(index))
            return letter

        if opcode == "operator_join":
            first = value_of(inputs["STRING1"])
            second = value_of(inputs["STRING2"])
            return lambda: _js_join(_js_string(first()), _js_string(second()))

        if opcode == "looks_costumenumbername":
            return lambda: self._current_costume + 1

        if opcode == "looks_costume":
            # The costume menu's shadow reports the chosen costume's name.
            return _constant(fields["COSTUME"][0])

        def unknown():
            raise NotImplementedError(f"unknown reporter opcode: {opcode}")
        return unknown

    def _compile_arithmetic(self, symbol: str, left_spec, right_spec):
        """``left symbol right`` with each operand cast only when it must be.

        Each operand is a constant, a closure that always returns a number,
        or one whose result needs Scratch's numeric cast; the closure for
        each combination is generated once and cached.
        """
        operands = []
        for spec in (left_spec, right_spec):
            value = self._compile_input(spec)
            constant = getattr(value, "constant", _MISSING)
            if constant is not _MISSING:
                operands.append(("const", self._scratch_numeric(constant)))
            elif getattr(value, "numeric", False):
                operands.append(("number", value))
            else:
                operands.append(("raw", value))
        key = (symbol, operands[0][0], operands[1][0])
        factory = _ARITHMETIC_FACTORIES.get(key)
        if factory is None:
            factory = _ARITHMETIC_FACTORIES[key] = _arithmetic_factory(*key)
        result = factory(operands[0][1], operands[1][1], self._scratch_numeric)
        result.numeric = True
        return result

    @staticmethod
    def _compile_mathop(value, op):
        # scratch-vm's mathop, with JavaScript's Math functions as V8
        # computes them (see jsmath), so results match to the last bit.
        if op == "abs":
            return lambda: abs(value())
        if op == "floor":
            return lambda: jsmath.floor(value())
        if op == "ceiling":
            return lambda: jsmath.ceil(value())
        if op == "sqrt":
            def square_root():
                number = value()
                return math.sqrt(number) if number >= 0 else math.nan
            return square_root
        if op == "ln":
            return lambda: jsmath.log(float(value()))
        if op == "log":
            # Math.log(n) / Math.LN10, not log10.
            return lambda: jsmath.log(float(value())) / 2.302585092994046
        if op == "e ^":
            return lambda: jsmath.exp(float(value()))
        if op == "10 ^":
            return lambda: jsmath.pow(10.0, float(value()))
        if op in {"sin", "cos"}:
            function = jsmath.sin if op == "sin" else jsmath.cos

            def trigonometry():
                # parseFloat(Math.sin((Math.PI * n) / 180).toFixed(10))
                return jsmath.to_fixed(function(math.pi * float(value()) / 180), 10)
            return trigonometry
        if op == "tan":
            def tangent():
                # scratch-vm's MathUtil.tan: exact infinities at ±90 degrees,
                # otherwise rounded like sin and cos.
                number = float(value())
                if not math.isfinite(number):
                    return math.nan
                angle = math.fmod(number, 360)
                if angle in {90, -270}:
                    return math.inf
                if angle in {-90, 270}:
                    return -math.inf
                return jsmath.to_fixed(jsmath.tan(math.pi * angle / 180), 10)
            return tangent
        if op in {"asin", "acos", "atan"}:
            function = {"asin": jsmath.asin, "acos": jsmath.acos, "atan": jsmath.atan}[op]
            # (Math.asin(n) * 180) / Math.PI
            return lambda: function(float(value())) * 180 / math.pi

        def unknown():
            raise NotImplementedError(f"unknown mathop: {op}")
        return unknown

    def _get_substack(self, input_spec) -> str | None:
        """Extract a substack block ID from a SUBSTACK input."""
        if input_spec is None:
            return None
        # [2, block_id] or [3, block_id, null]
        if isinstance(input_spec[1], str):
            return input_spec[1]
        return None

    @classmethod
    def _list_index(cls, value, length: int, *, accept_all: bool = False):
        """``Cast.toListIndex``: a 1-based index into a list of ``length``, or 0
        when it is invalid; ``"all"`` when ``accept_all`` and it says so."""
        if value.__class__ is not int and value.__class__ is not float:
            if value == "all":
                return "all" if accept_all else 0
            if value == "last":
                return length
            if value in ("random", "any"):
                import random
                return random.randint(1, length) if length else 0
        number = cls._scratch_numeric(value)
        if not math.isfinite(number):
            return 0
        index = math.floor(number)
        return index if 1 <= index <= length else 0

    @classmethod
    def _to_number(cls, value):
        """A literal as the number it spells, when that loses nothing.

        scratch-vm keeps a number field's text as it is and casts it where
        it's used. Reading it as a number once is faster, and only when the
        number turns back into the same text (``"10"``, not ``"1.0"`` or
        ``"1e3"``) does it make no difference anywhere.
        """
        if not isinstance(value, str):
            return _js_value(value)
        number = cls._scratch_number(value)
        if number is None or number != number or value.strip(_JS_WHITESPACE) != value:
            return value
        if math.isfinite(number) and number == int(number):
            number = int(number)
        return number if _js_string(number) == value else value

    @staticmethod
    def _scratch_number(value):
        """Convert to a number as JavaScript's ``Number`` does.

        Returns None if the value cannot be cast. Empty and whitespace-only
        strings are 0, and ``0x``, ``0o`` and ``0b`` prefixes and ``Infinity``
        are read; Python's ``inf``, ``nan``, non-ASCII digits and digit
        underscores are not. Integers past 2**53 are doubles, as in JavaScript.
        """
        kind = value.__class__
        if kind is float:
            return value
        if kind is int or kind is bool:
            value = int(value)
            if -_EXACT_INTEGER <= value <= _EXACT_INTEGER:
                return value
            return float(value)
        if kind is not str:
            return None
        s = value.strip(_JS_WHITESPACE)
        if s == "":
            return 0
        if _JS_RADIX_NUMBER.fullmatch(s):
            whole = int(s, 0)
            return float(whole) if whole < 2**1024 else math.inf
        if _JS_DECIMAL_NUMBER.fullmatch(s):
            return float(s)
        return None

    @classmethod
    def _scratch_round(cls, value) -> int:
        """Round a loop count like ``Math.round(Cast.toNumber(value))``."""
        number = cls._scratch_numeric(value)
        if not math.isfinite(number):
            return 0 if math.isnan(number) or number < 0 else 2**31
        return int(jsmath.round(number))

    @classmethod
    def _scratch_compare(cls, left, right) -> float:
        """Compare like the Scratch VM's ``Cast.compare``.

        Empty and whitespace-only strings are not numbers here, so ``""`` is
        not equal to ``0``. Non-numeric values compare as lowercase strings.
        """
        def is_whitespace(value):
            return isinstance(value, str) and value.strip(_JS_WHITESPACE) == ""

        left_number = cls._scratch_number(left)
        right_number = cls._scratch_number(right)
        if left_number == 0 and is_whitespace(left):
            left_number = None
        elif right_number == 0 and is_whitespace(right):
            right_number = None
        if (
            left_number is None or right_number is None
            or math.isnan(left_number) or math.isnan(right_number)
        ):
            left_text = _js_lower(_js_string(left))
            right_text = _js_lower(_js_string(right))
            if _astral(left_text) or _astral(right_text):
                # JavaScript orders strings by UTF-16 code units.
                left_text, right_text = _utf16_key(left_text), _utf16_key(right_text)
            return (left_text > right_text) - (left_text < right_text)
        if left_number == right_number:
            return 0
        return left_number - right_number

    @classmethod
    def _scratch_numeric(cls, value):
        """``Cast.toNumber``: what can't be read as a number, and NaN, is 0."""
        converted = cls._scratch_number(value)
        return 0 if converted is None or converted != converted else converted


class ProjectEmulator:
    """Run a whole Scratch project: the stage and its sprites, sharing stage data.

    Scripts run to completion one after another, so ``broadcast`` and
    ``broadcast and wait`` both behave like ``broadcast and wait``. ``answers``
    are returned in order to ``ask and wait`` blocks, whose questions collect
    in ``questions``.
    """

    def __init__(self, project, *, answers=()):
        if not isinstance(project, dict):
            import json
            import zipfile

            with zipfile.ZipFile(project) as archive:
                project = json.loads(archive.read("project.json"))
        targets = project["targets"]
        self.stage = ScratchEmulator(next(target for target in targets if target.get("isStage")))
        self.sprites = {
            target["name"]: ScratchEmulator(target, stage=self.stage)
            for target in targets if not target.get("isStage")
        }
        self.answers = list(answers)
        self.questions: list = []
        for emulator in self.targets:
            emulator.on_broadcast = self.broadcast
            emulator.on_ask = self._ask

    @property
    def targets(self) -> tuple[ScratchEmulator, ...]:
        return (self.stage, *self.sprites.values())

    def _run_hats(self, opcode, matches=lambda block: True) -> None:
        for emulator in self.targets:
            hats = [
                block for block in emulator.blocks.values()
                if block.get("opcode") == opcode and block.get("topLevel") and matches(block)
            ]
            for hat in hats:
                with _deep_reporters():
                    emulator._exec_script(hat.get("next"))

    def green_flag(self) -> None:
        self._run_hats("event_whenflagclicked")

    def broadcast(self, message: str) -> None:
        self._run_hats(
            "event_whenbroadcastreceived",
            lambda block: str(block["fields"]["BROADCAST_OPTION"][0]).lower() == message.lower(),
        )

    def press(self, key: str) -> None:
        self._run_hats(
            "event_whenkeypressed",
            lambda block: block["fields"]["KEY_OPTION"][0] in {key, "any"},
        )

    def _ask(self, question):
        self.questions.append(question)
        if not self.answers:
            raise RuntimeError(f"no answer left for {question!r}")
        return self.answers.pop(0)
