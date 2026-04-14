"""
emulator.py
-----------
Minimal Scratch block emulator for testing transpiled sprites.

Supports the subset of Scratch used by cattorch templates:
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

    def run(self):
        root = self._find_root()
        self._exec_chain(root)

    def _find_root(self) -> str:
        roots = [
            bid for bid, block in self.blocks.items()
            if block.get("topLevel") and block.get("parent") is None
        ]
        if len(roots) != 1:
            raise ValueError(f"Expected 1 topLevel root, found {len(roots)}: {roots}")
        return roots[0]

    def _exec_chain(self, block_id: str | None):
        while block_id is not None:
            block = self.blocks[block_id]
            self._exec_block(block_id, block)
            block_id = block.get("next")

    def _exec_block(self, block_id: str, block: dict):
        opcode = block["opcode"]
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
            self._lists[list_id].append(value)

        elif opcode == "data_replaceitemoflist":
            list_id = fields["LIST"][1]
            index = self._to_index(self._eval_input(inputs["INDEX"]))
            value = self._eval_input(inputs["ITEM"])
            lst = self._lists[list_id]
            if 1 <= index <= len(lst):
                lst[index - 1] = value

        elif opcode == "control_repeat":
            times = int(float(self._eval_input(inputs["TIMES"])))
            substack_id = self._get_substack(inputs.get("SUBSTACK"))
            for _ in range(times):
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
            return self._to_number(spec[1])

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
            a = float(self._eval_input(inputs["NUM1"]))
            b = float(self._eval_input(inputs["NUM2"]))
            return a + b

        elif opcode == "operator_subtract":
            a = float(self._eval_input(inputs["NUM1"]))
            b = float(self._eval_input(inputs["NUM2"]))
            return a - b

        elif opcode == "operator_multiply":
            a = float(self._eval_input(inputs["NUM1"]))
            b = float(self._eval_input(inputs["NUM2"]))
            return a * b

        elif opcode == "operator_mod":
            a = float(self._eval_input(inputs["NUM1"]))
            b = float(self._eval_input(inputs["NUM2"]))
            return a % b if b != 0 else 0

        elif opcode == "operator_equals":
            a = self._eval_input(inputs["OPERAND1"])
            b = self._eval_input(inputs["OPERAND2"])
            return float(a) == float(b)

        elif opcode == "operator_gt":
            a = float(self._eval_input(inputs["OPERAND1"]))
            b = float(self._eval_input(inputs["OPERAND2"]))
            return a > b

        elif opcode == "operator_lt":
            a = float(self._eval_input(inputs["OPERAND1"]))
            b = float(self._eval_input(inputs["OPERAND2"]))
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
            a = float(self._eval_input(inputs["NUM1"]))
            b = float(self._eval_input(inputs["NUM2"]))
            return a / b if b != 0 else 0

        elif opcode == "operator_mathop":
            value = float(self._eval_input(inputs["NUM"]))
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

        elif opcode == "data_itemoflist":
            list_id = fields["LIST"][1]
            index = self._to_index(self._eval_input(inputs["INDEX"]))
            lst = self._lists.get(list_id, [])
            if 1 <= index <= len(lst):
                return lst[index - 1]
            return 0

        elif opcode == "data_lengthoflist":
            list_id = fields["LIST"][1]
            return len(self._lists.get(list_id, []))

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
            return int(f) if f == int(f) else f
        except (ValueError, TypeError):
            return value
