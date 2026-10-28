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

import math
from collections import Counter

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
        names.append(name)
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
        self.blocks = sprite["blocks"]
        self._procedures = self._find_procedures()
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
        self._chains: dict = {}

        # Variables stored by ID; a sprite shares the stage's dictionaries,
        # since data IDs are unique across a project.
        self._vars = stage._vars if stage is not None else {}
        self._var_name_to_ids = {}
        for sid, entry in sprite.get("variables", {}).items():
            name, value = entry[0], entry[1]
            self._vars[sid] = value
            self._var_name_to_ids.setdefault(name, []).append(sid)
        # The sprite's own data shadows stage data with the same name.
        for name, ids in (stage._var_name_to_ids if stage else {}).items():
            self._var_name_to_ids.setdefault(name, []).extend(ids)

        # Lists stored by ID
        self._lists = stage._lists if stage is not None else {}
        self._list_name_to_ids = {}
        for sid, entry in sprite.get("lists", {}).items():
            name, data = entry[0], list(entry[1])
            self._lists[sid] = data
            self._list_name_to_ids.setdefault(name, []).append(sid)
        for name, ids in (stage._list_name_to_ids if stage else {}).items():
            self._list_name_to_ids.setdefault(name, []).extend(ids)

        # Public name-based views for test convenience
        self.variables = _NameView(self._vars, self._var_name_to_ids)
        self.lists = _NameView(self._lists, self._list_name_to_ids)

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
        self._exec_chain(roots[root_index])

    def run_procedure(self, name: str) -> None:
        """Run a named custom block without relying on top-level stack order."""
        if name not in self._procedures:
            available = ", ".join(sorted(self._procedures)) or "none"
            raise ValueError(f"procedure {name!r} not found; available: {available}")
        self._exec_chain(self._procedures[name])

    def _find_roots(self) -> list[str]:
        roots = [
            bid for bid, block in self.blocks.items()
            if block.get("topLevel") and block.get("parent") is None
            and block.get("opcode") != "procedures_definition"
        ]
        if not roots:
            raise ValueError("no topLevel roots found")
        return roots

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

            def run():
                counts[opcode] += 1
                variables[var_id] = float(variables[var_id]) + float(value())

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
            to_index = self._to_index

            def run():
                counts[opcode] += 1
                index = to_index(index_of())
                item = value()
                target = lists[list_id]
                if 1 <= index <= len(target):
                    target[index - 1] = item

        elif opcode == "data_deleteoflist":
            list_id = fields["LIST"][1]
            index_of = value_of(inputs["INDEX"])
            to_index = self._to_index

            def run():
                counts[opcode] += 1
                index = to_index(index_of())
                target = lists[list_id]
                if 1 <= index <= len(target):
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

            def run():
                counts[opcode] += 1
                for index in range(1, int(float(times_of())) + 1):
                    variables[var_id] = index
                    body()

        elif opcode == "control_repeat_until":
            condition = value_of(inputs["CONDITION"])
            body = self._chain_runner(self._get_substack(inputs.get("SUBSTACK")))

            def run():
                counts[opcode] += 1
                while not condition():
                    body()

        elif opcode == "control_if":
            condition = value_of(inputs["CONDITION"])
            body = self._chain_runner(self._get_substack(inputs.get("SUBSTACK")))

            def run():
                counts[opcode] += 1
                if condition():
                    body()

        elif opcode == "control_if_else":
            condition = value_of(inputs["CONDITION"])
            body = self._chain_runner(self._get_substack(inputs.get("SUBSTACK")))
            other = self._chain_runner(self._get_substack(inputs.get("SUBSTACK2")))

            def run():
                counts[opcode] += 1
                if condition():
                    body()
                else:
                    other()

        elif opcode == "procedures_call":
            proccode = block.get("mutation", {}).get("proccode")
            procedures = self._procedures
            exec_chain = self._exec_chain

            def run():
                counts[opcode] += 1
                if proccode not in procedures:
                    raise ValueError(f"procedure not found: {proccode}")
                exec_chain(procedures[proccode])

        elif opcode == "looks_switchcostumeto":
            requested_of = value_of(inputs["COSTUME"])

            def run():
                counts[opcode] += 1
                requested = requested_of()
                try:
                    self._current_costume = self._costume_names.index(str(requested))
                except ValueError:
                    # Sufficient fallback for cattorch tests; codec symbols always
                    # resolve through the exact, case-sensitive name path.
                    index = self._to_index(requested)
                    if 1 <= index <= len(self._costume_names):
                        self._current_costume = index - 1

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

        elif opcode == "sensing_askandwait":
            question = value_of(inputs["QUESTION"])
            answer = self._answer

            def run():
                counts[opcode] += 1
                asked = question()
                if self.on_ask is None:
                    raise RuntimeError(f"ask and wait without an answer: {asked}")
                answer[0] = self.on_ask(asked)

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
            # [1, literal] or [1, [type, value]]
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
        if getattr(value, "numeric", False):
            return value
        numeric = self._scratch_numeric

        def run():
            result = value()
            kind = result.__class__
            return result if kind is float or kind is int else numeric(result)
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
                    a, b = left(), right()
                    # JavaScript's remainder by zero is NaN; Scratch keeps it.
                    return a % b if b != 0 else math.nan
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
                    return bool(a) and bool(b)
                return both

            def either():
                a, b = left(), right()
                return bool(a) or bool(b)
            return either

        if opcode == "operator_not":
            operand = value_of(inputs["OPERAND"])
            return lambda: not bool(operand())

        if opcode == "operator_mathop":
            result = self._compile_mathop(number_of(inputs["NUM"]), fields["OPERATOR"][0])
            result.numeric = True
            return result

        if opcode == "operator_round":
            value = number_of(inputs["NUM"])

            def round_half_up():
                # Scratch uses JavaScript Math.round, including its handling
                # of negative half values.
                return math.floor(value() + 0.5)
            round_half_up.numeric = True
            return round_half_up

        if opcode == "sensing_answer":
            answer = self._answer
            return lambda: answer[0]

        if opcode == "operator_random":
            low_of = number_of(inputs["FROM"])
            high_of = number_of(inputs["TO"])

            def random_value():
                import random
                low, high = low_of(), high_of()
                low, high = min(low, high), max(low, high)
                if float(low).is_integer() and float(high).is_integer():
                    return random.randint(int(low), int(high))
                return random.uniform(low, high)
            return random_value

        if opcode == "data_itemoflist":
            list_id = fields["LIST"][1]
            index_of = value_of(inputs["INDEX"])
            lists = self._lists
            to_index = self._to_index

            def item():
                index = to_index(index_of())
                values = lists.get(list_id, ())
                if 1 <= index <= len(values):
                    return values[index - 1]
                return ""
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
            return lambda: len(str(text()))

        if opcode == "operator_letter_of":
            letter_of = value_of(inputs["LETTER"])
            text_of = value_of(inputs["STRING"])

            def letter():
                index = int(float(letter_of()))
                string = str(text_of())
                if 1 <= index <= len(string):
                    return string[index - 1]
                return ""
            return letter

        if opcode == "operator_join":
            first = value_of(inputs["STRING1"])
            second = value_of(inputs["STRING2"])
            return lambda: str(first()) + str(second())

        if opcode == "looks_costumenumbername":
            return lambda: self._current_costume + 1

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
        # Match JavaScript's Math functions, which return Infinity or NaN
        # where Python raises.
        if op == "e ^":
            def exp():
                try:
                    return math.exp(value())
                except OverflowError:
                    return math.inf
            return exp
        if op == "abs":
            return lambda: abs(value())
        if op in {"floor", "ceiling"}:
            rounding = math.floor if op == "floor" else math.ceil

            def round_value():
                number = value()
                return rounding(number) if math.isfinite(number) else number
            return round_value
        if op == "sqrt":
            def square_root():
                number = value()
                return math.sqrt(number) if number >= 0 else math.nan
            return square_root
        if op in {"ln", "log"}:
            logarithm = math.log if op == "ln" else math.log10

            def log_value():
                number = value()
                if math.isnan(number) or number < 0:
                    return math.nan
                if number == 0:
                    return -math.inf
                if math.isinf(number):
                    return math.inf
                return logarithm(number)
            return log_value
        if op in {"sin", "cos"}:
            function = math.sin if op == "sin" else math.cos

            def trigonometry():
                # scratch-vm: Math.round(Math.sin(Math.PI * n / 180) * 1e10) / 1e10
                number = value()
                if not math.isfinite(number):
                    return math.nan
                return math.floor(function(math.pi * number / 180) * 1e10 + 0.5) / 1e10
            return trigonometry
        if op == "tan":
            def tangent():
                # scratch-vm's MathUtil.tan: exact infinities at ±90 degrees,
                # otherwise rounded like sin and cos.
                number = value()
                if not math.isfinite(number):
                    return math.nan
                angle = math.fmod(number, 360)
                if angle in {90, -270}:
                    return math.inf
                if angle in {-90, 270}:
                    return -math.inf
                return math.floor(math.tan(math.pi * angle / 180) * 1e10 + 0.5) / 1e10
            return tangent
        if op in {"asin", "acos", "atan"}:
            function = {"asin": math.asin, "acos": math.acos, "atan": math.atan}[op]

            def inverse_trigonometry():
                number = value()
                if op != "atan" and not -1 <= number <= 1:
                    return math.nan
                return function(number) * 180 / math.pi
            return inverse_trigonometry
        if op == "10 ^":
            def power():
                try:
                    return 10.0 ** value()
                except OverflowError:
                    return math.inf
            return power

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

    @staticmethod
    def _to_index(value) -> int:
        """Convert a value to a 1-based list index, returning 0 for invalid."""
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _to_number(value):
        """Convert a value to a number if possible, like Scratch does."""
        if isinstance(value, (int, float)):
            return value
        try:
            f = float(value)
            if not math.isfinite(f):
                return f
            return int(f) if f == int(f) else f
        except (ValueError, TypeError):
            return value

    @staticmethod
    def _scratch_number(value):
        """Convert to a number using Scratch's casting rules.

        Returns None if the value cannot be cast.  Scratch treats empty
        strings and whitespace-only strings as 0.
        """
        if isinstance(value, (int, float)):
            return value
        s = str(value).strip()
        if s == "":
            return 0
        try:
            return float(s)
        except ValueError:
            return None

    @classmethod
    def _scratch_round(cls, value) -> int:
        """Round a loop count like ``Math.round(Cast.toNumber(value))``."""
        number = cls._scratch_numeric(value)
        if not math.isfinite(number):
            return 0 if math.isnan(number) or number < 0 else 2**31
        return math.floor(number + 0.5)

    @classmethod
    def _scratch_compare(cls, left, right) -> float:
        """Compare like the Scratch VM's ``Cast.compare``.

        Empty and whitespace-only strings are not numbers here, so ``""`` is
        not equal to ``0``. Non-numeric values compare as lowercase strings.
        """
        def is_whitespace(value):
            return isinstance(value, str) and value.strip() == ""

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
            left_text = str(left).lower()
            right_text = str(right).lower()
            return (left_text > right_text) - (left_text < right_text)
        if left_number == right_number:
            return 0
        return left_number - right_number

    @classmethod
    def _scratch_numeric(cls, value):
        converted = cls._scratch_number(value)
        return 0 if converted is None else converted


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
                emulator._exec_chain(hat.get("next"))

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
