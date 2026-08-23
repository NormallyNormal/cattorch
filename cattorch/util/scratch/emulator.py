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

from collections import Counter

from cattorch.util.scratch.sharding import SCRATCH_LIST_LIMIT


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
    def __init__(self, sprite: dict):
        self.blocks = sprite["blocks"]
        self._procedures = self._find_procedures()
        self.opcode_counts = Counter()
        self._costume_names = [
            costume.get("name", "") for costume in sprite.get("costumes", [])
        ]
        self._current_costume = int(sprite.get("currentCostume", 0))
        self.broadcasts = []

        # Variables stored by ID
        self._vars = {}
        self._var_name_to_ids = {}
        for sid, entry in sprite.get("variables", {}).items():
            name, value = entry[0], entry[1]
            self._vars[sid] = value
            self._var_name_to_ids.setdefault(name, []).append(sid)

        # Lists stored by ID
        self._lists = {}
        self._list_name_to_ids = {}
        for sid, entry in sprite.get("lists", {}).items():
            name, data = entry[0], list(entry[1])
            self._lists[sid] = data
            self._list_name_to_ids.setdefault(name, []).append(sid)

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
            raise ValueError(f"Procedure {name!r} not found; available: {available}")
        self._exec_chain(self._procedures[name])

    def _find_roots(self) -> list[str]:
        roots = [
            bid for bid, block in self.blocks.items()
            if block.get("topLevel") and block.get("parent") is None
            and block.get("opcode") != "procedures_definition"
        ]
        if not roots:
            raise ValueError("No topLevel roots found")
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

    def _exec_chain(self, block_id: str | None):
        while block_id is not None:
            block = self.blocks[block_id]
            self._exec_block(block_id, block)
            block_id = block.get("next")

    def _exec_block(self, block_id: str, block: dict):
        opcode = block["opcode"]
        self.opcode_counts[opcode] += 1
        inputs = block.get("inputs", {})
        fields = block.get("fields", {})

        if opcode == "data_setvariableto":
            var_id = fields["VARIABLE"][1]
            self._vars[var_id] = self._eval_input(inputs["VALUE"])

        elif opcode == "data_changevariableby":
            var_id = fields["VARIABLE"][1]
            self._vars[var_id] = float(self._vars[var_id]) + float(self._eval_input(inputs["VALUE"]))

        elif opcode == "data_deletealloflist":
            list_id = fields["LIST"][1]
            self._lists[list_id] = []

        elif opcode == "data_addtolist":
            list_id = fields["LIST"][1]
            value = self._eval_input(inputs["ITEM"])
            if len(self._lists[list_id]) < SCRATCH_LIST_LIMIT:
                self._lists[list_id].append(value)

        elif opcode == "data_replaceitemoflist":
            list_id = fields["LIST"][1]
            index = self._to_index(self._eval_input(inputs["INDEX"]))
            value = self._eval_input(inputs["ITEM"])
            lst = self._lists[list_id]
            if 1 <= index <= len(lst):
                lst[index - 1] = value

        elif opcode == "data_deleteoflist":
            list_id = fields["LIST"][1]
            index = self._to_index(self._eval_input(inputs["INDEX"]))
            lst = self._lists[list_id]
            if 1 <= index <= len(lst):
                lst.pop(index - 1)

        elif opcode == "control_repeat":
            times = int(float(self._eval_input(inputs["TIMES"])))
            substack_id = self._get_substack(inputs.get("SUBSTACK"))
            for _ in range(times):
                self._exec_chain(substack_id)

        elif opcode == "control_for_each":
            times = int(float(self._eval_input(inputs["VALUE"])))
            variable_id = fields["VARIABLE"][1]
            substack_id = self._get_substack(inputs.get("SUBSTACK"))
            for index in range(1, times + 1):
                self._vars[variable_id] = index
                self._exec_chain(substack_id)

        elif opcode == "control_repeat_until":
            substack_id = self._get_substack(inputs.get("SUBSTACK"))
            while not self._eval_input(inputs["CONDITION"]):
                self._exec_chain(substack_id)

        elif opcode == "control_if":
            condition = self._eval_input(inputs["CONDITION"])
            if condition:
                self._exec_chain(self._get_substack(inputs.get("SUBSTACK")))

        elif opcode == "control_if_else":
            condition = self._eval_input(inputs["CONDITION"])
            if condition:
                self._exec_chain(self._get_substack(inputs.get("SUBSTACK")))
            else:
                self._exec_chain(self._get_substack(inputs.get("SUBSTACK2")))

        elif opcode == "procedures_call":
            proccode = block.get("mutation", {}).get("proccode")
            if proccode not in self._procedures:
                raise ValueError(f"Procedure not found: {proccode}")
            self._exec_chain(self._procedures[proccode])

        elif opcode == "looks_switchcostumeto":
            requested = self._eval_input(inputs["COSTUME"])
            try:
                self._current_costume = self._costume_names.index(str(requested))
            except ValueError:
                # Sufficient fallback for cattorch tests; codec symbols always
                # resolve through the exact, case-sensitive name path.
                index = self._to_index(requested)
                if 1 <= index <= len(self._costume_names):
                    self._current_costume = index - 1

        elif opcode in {"event_broadcast", "event_broadcastandwait"}:
            self.broadcasts.append(self._eval_input(inputs["BROADCAST_INPUT"]))

        elif opcode in {"procedures_definition", "procedures_prototype"}:
            # Definitions are metadata/entry points rather than executable
            # commands when encountered outside a procedure call.
            return

        else:
            raise NotImplementedError(f"Unknown opcode: {opcode}")

    def _eval_input(self, input_spec):
        """Evaluate a Scratch input specification and return a Python value."""
        if input_spec is None:
            return 0

        type_code = input_spec[0]

        if type_code == 1:
            # [1, literal] or [1, [type, value]]
            return self._eval_literal(input_spec[1])

        elif type_code == 2:
            # [2, block_id] — block reference (used for SUBSTACK)
            return self._eval_reporter(input_spec[1])

        elif type_code == 3:
            # [3, block_id_or_var_ref, fallback]
            ref = input_spec[1]
            if isinstance(ref, str):
                # Block reference
                return self._eval_reporter(ref)
            elif isinstance(ref, list):
                # Variable or list reference: [12, name, id] or [13, name, id]
                return self._eval_literal(ref)
            else:
                return self._eval_literal(input_spec[2]) if len(input_spec) > 2 else 0

        return 0

    def _eval_literal(self, spec):
        """Evaluate a literal value spec like [4, "10"] or [12, "var", "id"]."""
        if not isinstance(spec, list):
            return self._to_number(spec)

        lit_type = spec[0]

        if lit_type in (4, 5, 6, 7, 8):
            # Numeric literals
            return self._to_number(spec[1])

        elif lit_type in (10,):
            # String literal
            return spec[1]

        elif lit_type == 11:
            # Broadcast menu literal: [11, display_name, broadcast_id]
            return spec[1]

        elif lit_type == 12:
            # Variable reference: [12, display_name, var_id]
            var_id = spec[2]
            return self._vars.get(var_id, 0)

        elif lit_type == 13:
            # List reference: [13, display_name, list_id]
            list_id = spec[2]
            return self._lists.get(list_id, [])

        return self._to_number(spec[1]) if len(spec) > 1 else 0

    def _eval_reporter(self, block_id: str):
        """Evaluate a reporter block (one that returns a value)."""
        block = self.blocks[block_id]
        opcode = block["opcode"]
        inputs = block.get("inputs", {})
        fields = block.get("fields", {})

        if opcode == "operator_add":
            a = self._scratch_numeric(self._eval_input(inputs["NUM1"]))
            b = self._scratch_numeric(self._eval_input(inputs["NUM2"]))
            return a + b

        elif opcode == "operator_subtract":
            a = self._scratch_numeric(self._eval_input(inputs["NUM1"]))
            b = self._scratch_numeric(self._eval_input(inputs["NUM2"]))
            return a - b

        elif opcode == "operator_multiply":
            a = self._scratch_numeric(self._eval_input(inputs["NUM1"]))
            b = self._scratch_numeric(self._eval_input(inputs["NUM2"]))
            return a * b

        elif opcode == "operator_mod":
            a = self._scratch_numeric(self._eval_input(inputs["NUM1"]))
            b = self._scratch_numeric(self._eval_input(inputs["NUM2"]))
            return a % b if b != 0 else 0

        elif opcode == "operator_equals":
            a = self._eval_input(inputs["OPERAND1"])
            b = self._eval_input(inputs["OPERAND2"])
            na = self._scratch_number(a)
            nb = self._scratch_number(b)
            if na is not None and nb is not None:
                return na == nb
            return str(a).lower() == str(b).lower()

        elif opcode == "operator_gt":
            a = self._scratch_numeric(self._eval_input(inputs["OPERAND1"]))
            b = self._scratch_numeric(self._eval_input(inputs["OPERAND2"]))
            return a > b

        elif opcode == "operator_lt":
            a = self._scratch_numeric(self._eval_input(inputs["OPERAND1"]))
            b = self._scratch_numeric(self._eval_input(inputs["OPERAND2"]))
            return a < b

        elif opcode == "operator_and":
            a = self._eval_input(inputs["OPERAND1"])
            b = self._eval_input(inputs["OPERAND2"])
            return bool(a) and bool(b)

        elif opcode == "operator_or":
            a = self._eval_input(inputs["OPERAND1"])
            b = self._eval_input(inputs["OPERAND2"])
            return bool(a) or bool(b)

        elif opcode == "operator_not":
            a = self._eval_input(inputs["OPERAND"])
            return not bool(a)

        elif opcode == "operator_subtract":
            a = float(self._eval_input(inputs["NUM1"]))
            b = float(self._eval_input(inputs["NUM2"]))
            return a - b

        elif opcode == "operator_divide":
            a = self._scratch_numeric(self._eval_input(inputs["NUM1"]))
            b = self._scratch_numeric(self._eval_input(inputs["NUM2"]))
            if b != 0:
                return a / b
            import math
            if a == 0:
                return math.nan
            return math.copysign(math.inf, a * math.copysign(1, b))

        elif opcode == "operator_mathop":
            value = self._scratch_numeric(self._eval_input(inputs["NUM"]))
            op = fields["OPERATOR"][0]
            if op == "e ^":
                import math
                return math.exp(value)
            elif op == "abs":
                return abs(value)
            elif op == "floor":
                import math
                return math.floor(value)
            elif op == "ceiling":
                import math
                return math.ceil(value)
            elif op == "sqrt":
                import math
                return math.sqrt(value)
            elif op == "ln":
                import math
                return math.log(value) if value > 0 else 0
            elif op == "log":
                import math
                return math.log10(value) if value > 0 else 0
            elif op == "10 ^":
                return 10 ** value
            else:
                raise NotImplementedError(f"Unknown mathop: {op}")

        elif opcode == "operator_round":
            import math
            value = self._scratch_numeric(self._eval_input(inputs["NUM"]))
            # Scratch uses JavaScript Math.round, including its handling of
            # negative half values.
            return math.floor(value + 0.5)

        elif opcode == "operator_random":
            import random
            low = self._scratch_numeric(self._eval_input(inputs["FROM"]))
            high = self._scratch_numeric(self._eval_input(inputs["TO"]))
            low, high = min(low, high), max(low, high)
            if float(low).is_integer() and float(high).is_integer():
                return random.randint(int(low), int(high))
            return random.uniform(low, high)

        elif opcode == "data_itemoflist":
            list_id = fields["LIST"][1]
            index = self._to_index(self._eval_input(inputs["INDEX"]))
            lst = self._lists.get(list_id, [])
            if 1 <= index <= len(lst):
                return lst[index - 1]
            return ""

        elif opcode == "data_lengthoflist":
            list_id = fields["LIST"][1]
            return len(self._lists.get(list_id, []))

        elif opcode == "data_itemnumoflist":
            list_id = fields["LIST"][1]
            item = self._eval_input(inputs["ITEM"])
            lst = self._lists.get(list_id, [])
            # Scratch returns 1-based index, or 0 if not found
            item_str = str(item).lower()
            for i, val in enumerate(lst):
                # The official VM uses Cast.compare here, including Scratch's
                # case-insensitive string comparison.
                if str(val).lower() == item_str:
                    return i + 1
            return 0

        elif opcode == "operator_length":
            string = str(self._eval_input(inputs["STRING"]))
            return len(string)

        elif opcode == "operator_letter_of":
            index = int(float(self._eval_input(inputs["LETTER"])))
            string = str(self._eval_input(inputs["STRING"]))
            if 1 <= index <= len(string):
                return string[index - 1]
            return ""

        elif opcode == "operator_join":
            a = str(self._eval_input(inputs["STRING1"]))
            b = str(self._eval_input(inputs["STRING2"]))
            return a + b

        elif opcode == "looks_costumenumbername":
            return self._current_costume + 1

        else:
            raise NotImplementedError(f"Unknown reporter opcode: {opcode}")

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
            import math
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
    def _scratch_numeric(cls, value):
        converted = cls._scratch_number(value)
        return 0 if converted is None else converted
