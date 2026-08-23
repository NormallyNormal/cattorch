"""Small, typed builder for the subset of Scratch used by cattorch kernels.

The JSON exported by Scratch is deliberately treated as a lowering target,
not as source code.  Kernel implementations can be written with the objects
in this module and rendered as readable pseudocode for review.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterable, Mapping, Sequence, cast


# Expressions -----------------------------------------------------------------


class Expr:
    pass


@dataclass(frozen=True)
class Literal(Expr):
    value: int | float | str


@dataclass(frozen=True)
class Variable(Expr):
    name: str


@dataclass(frozen=True)
class ListItem(Expr):
    name: str
    index: Expr


@dataclass(frozen=True)
class ListLength(Expr):
    name: str


@dataclass(frozen=True)
class ListIndex(Expr):
    name: str
    value: Expr


@dataclass(frozen=True)
class StringItem(Expr):
    string: Expr
    index: Expr


@dataclass(frozen=True)
class StringLength(Expr):
    string: Expr


@dataclass(frozen=True)
class Join(Expr):
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Binary(Expr):
    opcode: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class MathOp(Expr):
    operator: str
    value: Expr


@dataclass(frozen=True)
class Round(Expr):
    value: Expr


@dataclass(frozen=True)
class Random(Expr):
    low: Expr
    high: Expr


@dataclass(frozen=True)
class CostumeNumber(Expr):
    """The current costume's one-based index.

    This reporter is useful for the case-sensitive costume-name lookup quirk
    in vanilla Scratch.  Ordinary Scratch string comparisons remain
    case-insensitive.
    """


def value(expr: Expr | int | float | str) -> Expr:
    return expr if isinstance(expr, Expr) else Literal(expr)


def var(name: str) -> Variable:
    return Variable(name)


def item(name: str, index: Expr | int | float) -> ListItem:
    return ListItem(name, value(index))


def length(name: str) -> ListLength:
    return ListLength(name)


def index_of(name: str, expr) -> ListIndex:
    return ListIndex(name, value(expr))


def letter(expr, index) -> StringItem:
    return StringItem(value(expr), value(index))


def string_length(expr) -> StringLength:
    return StringLength(value(expr))


def _binary(opcode: str, left, right) -> Binary:
    return Binary(opcode, value(left), value(right))


def add(left, right) -> Binary:
    return _binary("operator_add", left, right)


def sub(left, right) -> Binary:
    return _binary("operator_subtract", left, right)


def mul(left, right) -> Binary:
    return _binary("operator_multiply", left, right)


def div(left, right) -> Binary:
    return _binary("operator_divide", left, right)


def mod(left, right) -> Binary:
    return _binary("operator_mod", left, right)


def eq(left, right) -> Binary:
    return _binary("operator_equals", left, right)


def lt(left, right) -> Binary:
    return _binary("operator_lt", left, right)


def gt(left, right) -> Binary:
    return _binary("operator_gt", left, right)


# Statements ------------------------------------------------------------------


class Statement:
    pass


_UNROLL_FACTOR_LIMIT: ContextVar[int] = ContextVar(
    "cattorch_unroll_factor_limit", default=8,
)


@dataclass(frozen=True)
class SetVariable(Statement):
    name: str
    value: Expr


@dataclass(frozen=True)
class ChangeVariable(Statement):
    name: str
    value: Expr


@dataclass(frozen=True)
class ClearList(Statement):
    name: str


@dataclass(frozen=True)
class AppendList(Statement):
    name: str
    value: Expr


@dataclass(frozen=True)
class ReplaceListItem(Statement):
    name: str
    index: Expr
    value: Expr


@dataclass(frozen=True)
class DeleteListItem(Statement):
    name: str
    index: Expr


@dataclass(frozen=True)
class Repeat(Statement):
    times: Expr
    body: tuple[Statement, ...]


@dataclass(frozen=True)
class StaticUnrolledRepeat(Statement):
    """A compile-time loop whose unroll factor is selected during lowering."""

    count: int
    step: tuple[Statement, ...]
    max_factor: int = 4
    final_step: tuple[Statement, ...] | None = None
    priority: int = 1


@dataclass(frozen=True)
class RepeatUntil(Statement):
    condition: Expr
    body: tuple[Statement, ...]


@dataclass(frozen=True)
class ForEach(Statement):
    """Scratch's supported, but palette-hidden, one-based index loop."""

    variable: str
    times: Expr
    body: tuple[Statement, ...]


@dataclass(frozen=True)
class If(Statement):
    condition: Expr
    body: tuple[Statement, ...]


@dataclass(frozen=True)
class IfElse(Statement):
    condition: Expr
    then_body: tuple[Statement, ...]
    else_body: tuple[Statement, ...]


@dataclass(frozen=True)
class ProcedureCall(Statement):
    name: str


@dataclass(frozen=True)
class SwitchCostume(Statement):
    value: Expr


def set_var(name: str, expr) -> SetVariable:
    return SetVariable(name, value(expr))


def change_var(name: str, expr) -> ChangeVariable:
    return ChangeVariable(name, value(expr))


def clear(name: str) -> ClearList:
    return ClearList(name)


def append(name: str, expr) -> AppendList:
    return AppendList(name, value(expr))


def replace(name: str, index, expr) -> ReplaceListItem:
    return ReplaceListItem(name, value(index), value(expr))


def delete(name: str, index) -> DeleteListItem:
    return DeleteListItem(name, value(index))


def repeat(times, body: Iterable[Statement]) -> Repeat:
    return Repeat(value(times), tuple(body))


def static_unrolled_repeat(
    count: int,
    step: Iterable[Statement],
    *,
    max_factor: int = 4,
    final_step: Iterable[Statement] | None = None,
    priority: int = 1,
) -> StaticUnrolledRepeat:
    if count < 1:
        raise ValueError("static unrolled repeat count must be positive")
    if max_factor < 1:
        raise ValueError("static unroll factor must be positive")
    return StaticUnrolledRepeat(
        count,
        tuple(step),
        max_factor,
        None if final_step is None else tuple(final_step),
        priority,
    )


@contextmanager
def unroll_factor_limit(factor: int):
    """Temporarily cap all symbolic DSL loops compiled in this context."""
    if factor not in {1, 2, 4, 8}:
        raise ValueError("unroll factor limit must be 1, 2, 4, or 8")
    token = _UNROLL_FACTOR_LIMIT.set(factor)
    try:
        yield
    finally:
        _UNROLL_FACTOR_LIMIT.reset(token)


def repeat_until(condition, body: Iterable[Statement]) -> RepeatUntil:
    return RepeatUntil(value(condition), tuple(body))


def for_each(variable: str, times, body: Iterable[Statement]) -> ForEach:
    return ForEach(variable, value(times), tuple(body))


def if_(condition, body: Iterable[Statement]) -> If:
    return If(value(condition), tuple(body))


def if_else(condition, then_body: Iterable[Statement], else_body: Iterable[Statement]) -> IfElse:
    return IfElse(value(condition), tuple(then_body), tuple(else_body))


def call(name: str) -> ProcedureCall:
    return ProcedureCall(name)


def switch_costume(expr) -> SwitchCostume:
    return SwitchCostume(value(expr))


def costume_number() -> CostumeNumber:
    return CostumeNumber()


def mathop(operator: str, expr) -> MathOp:
    return MathOp(operator, value(expr))


def round_(expr) -> Round:
    return Round(value(expr))


def random(low, high) -> Random:
    return Random(value(low), value(high))


def join(left, right) -> Join:
    return Join(value(left), value(right))


# Program lowering -------------------------------------------------------------


_BINARY_INPUTS = {
    "operator_add": ("NUM1", "NUM2"),
    "operator_subtract": ("NUM1", "NUM2"),
    "operator_multiply": ("NUM1", "NUM2"),
    "operator_divide": ("NUM1", "NUM2"),
    "operator_mod": ("NUM1", "NUM2"),
    "operator_equals": ("OPERAND1", "OPERAND2"),
    "operator_lt": ("OPERAND1", "OPERAND2"),
    "operator_gt": ("OPERAND1", "OPERAND2"),
}


class Program:
    """A single top-level Scratch stack with declared variables and lists."""

    def __init__(
        self,
        name: str,
        *,
        variables: Iterable[str] = (),
        lists: Iterable[str] = (),
        list_values: Mapping[str, Sequence[int | float | str]] | None = None,
        variable_values: Mapping[str, int | float | str] | None = None,
        body: Iterable[Statement] = (),
        optimize_loops: bool = True,
    ):
        self.name = name
        self.variables = tuple(variables)
        self.lists = tuple(lists)
        self.list_values = {
            name: list(items) for name, items in (list_values or {}).items()
        }
        self.variable_values = dict(variable_values or {})
        unknown_variables = self.variable_values.keys() - set(self.variables)
        if unknown_variables:
            names = ", ".join(sorted(unknown_variables))
            raise ValueError(f"Initial values provided for undeclared variables: {names}")
        unknown_lists = self.list_values.keys() - set(self.lists)
        if unknown_lists:
            names = ", ".join(sorted(unknown_lists))
            raise ValueError(f"Initial values provided for undeclared lists: {names}")
        self.body = tuple(body)
        self.optimize_loops = optimize_loops

    def compile(self) -> dict:
        lowerer = _Lowerer(self)
        return lowerer.compile()

    def pseudocode(self) -> str:
        lines: list[str] = []
        _render_statements(self.body, lines, 0)
        return "\n".join(lines)

    def opcode_counts(self) -> dict[str, int]:
        """Return static opcode counts for the lowered program."""
        blocks = self.compile()["blocks"]
        return dict(Counter(block["opcode"] for block in blocks.values()))


class _Lowerer:
    def __init__(self, program: Program):
        self.program = program
        self.blocks: dict[str, dict] = {}
        self.counter = 0
        self.var_ids = {name: f"cattorch_dsl_var_{name}" for name in program.variables}
        self.list_ids = {name: f"cattorch_dsl_list_{name}" for name in program.lists}

    def _id(self) -> str:
        self.counter += 1
        return f"cattorch_dsl_block_{self.counter}"

    def compile(self) -> dict:
        if not self.program.body:
            raise ValueError("Scratch DSL programs must contain at least one statement")
        statements = self.program.body
        statements = _resolve_static_unrolling(
            statements, _UNROLL_FACTOR_LIMIT.get(),
        )
        if self.program.optimize_loops:
            statements = _optimize_index_loops(statements)
        root, _ = self._statements(statements, None)
        self.blocks[root]["topLevel"] = True
        self.blocks[root]["x"] = 0
        self.blocks[root]["y"] = 0
        return {
            "isStage": False,
            "name": self.program.name,
            "variables": {
                sid: [name, self.program.variable_values.get(name, 0)]
                for name, sid in self.var_ids.items()
            },
            "lists": {
                sid: [name, self.program.list_values.get(name, [])]
                for name, sid in self.list_ids.items()
            },
            "broadcasts": {},
            "blocks": self.blocks,
        }

    def _statements(self, statements: tuple[Statement, ...], parent: str | None):
        first = previous = None
        for statement in statements:
            block_id = self._statement(statement, parent if previous is None else previous)
            if previous is not None:
                self.blocks[previous]["next"] = block_id
            else:
                first = block_id
            previous = block_id
        return first, previous

    def _base(self, opcode: str, parent: str | None) -> dict:
        return {
            "opcode": opcode,
            "next": None,
            "parent": parent,
            "inputs": {},
            "fields": {},
            "shadow": False,
            "topLevel": False,
        }

    def _statement(self, statement: Statement, parent: str | None) -> str:
        block_id = self._id()

        if isinstance(statement, SetVariable):
            block = self._base("data_setvariableto", parent)
            block["fields"]["VARIABLE"] = [statement.name, self.var_ids[statement.name]]
            self.blocks[block_id] = block
            block["inputs"]["VALUE"] = self._expr(statement.value, block_id)
        elif isinstance(statement, ChangeVariable):
            block = self._base("data_changevariableby", parent)
            block["fields"]["VARIABLE"] = [statement.name, self.var_ids[statement.name]]
            self.blocks[block_id] = block
            block["inputs"]["VALUE"] = self._expr(statement.value, block_id)
        elif isinstance(statement, ClearList):
            block = self._base("data_deletealloflist", parent)
            block["fields"]["LIST"] = [statement.name, self.list_ids[statement.name]]
            self.blocks[block_id] = block
        elif isinstance(statement, AppendList):
            block = self._base("data_addtolist", parent)
            block["fields"]["LIST"] = [statement.name, self.list_ids[statement.name]]
            self.blocks[block_id] = block
            block["inputs"]["ITEM"] = self._expr(statement.value, block_id)
        elif isinstance(statement, ReplaceListItem):
            block = self._base("data_replaceitemoflist", parent)
            block["fields"]["LIST"] = [statement.name, self.list_ids[statement.name]]
            self.blocks[block_id] = block
            block["inputs"]["INDEX"] = self._expr(statement.index, block_id)
            block["inputs"]["ITEM"] = self._expr(statement.value, block_id)
        elif isinstance(statement, DeleteListItem):
            block = self._base("data_deleteoflist", parent)
            block["fields"]["LIST"] = [statement.name, self.list_ids[statement.name]]
            self.blocks[block_id] = block
            block["inputs"]["INDEX"] = self._expr(statement.index, block_id)
        elif isinstance(statement, Repeat):
            block = self._base("control_repeat", parent)
            self.blocks[block_id] = block
            block["inputs"]["TIMES"] = self._expr(statement.times, block_id)
            substack, _ = self._statements(statement.body, block_id)
            block["inputs"]["SUBSTACK"] = [2, substack]
        elif isinstance(statement, RepeatUntil):
            block = self._base("control_repeat_until", parent)
            self.blocks[block_id] = block
            block["inputs"]["CONDITION"] = self._expr(statement.condition, block_id)
            substack, _ = self._statements(statement.body, block_id)
            block["inputs"]["SUBSTACK"] = [2, substack]
        elif isinstance(statement, ForEach):
            block = self._base("control_for_each", parent)
            self.blocks[block_id] = block
            block["fields"]["VARIABLE"] = [
                statement.variable,
                self.var_ids[statement.variable],
            ]
            block["inputs"]["VALUE"] = self._expr(statement.times, block_id)
            substack, _ = self._statements(statement.body, block_id)
            block["inputs"]["SUBSTACK"] = [2, substack]
        elif isinstance(statement, If):
            block = self._base("control_if", parent)
            self.blocks[block_id] = block
            block["inputs"]["CONDITION"] = self._expr(statement.condition, block_id)
            substack, _ = self._statements(statement.body, block_id)
            block["inputs"]["SUBSTACK"] = [2, substack]
        elif isinstance(statement, IfElse):
            block = self._base("control_if_else", parent)
            self.blocks[block_id] = block
            block["inputs"]["CONDITION"] = self._expr(statement.condition, block_id)
            then_stack, _ = self._statements(statement.then_body, block_id)
            else_stack, _ = self._statements(statement.else_body, block_id)
            block["inputs"]["SUBSTACK"] = [2, then_stack]
            block["inputs"]["SUBSTACK2"] = [2, else_stack]
        elif isinstance(statement, ProcedureCall):
            block = self._base("procedures_call", parent)
            block["mutation"] = {
                "tagName": "mutation", "children": [],
                "proccode": statement.name, "argumentids": "[]",
            }
            self.blocks[block_id] = block
        elif isinstance(statement, SwitchCostume):
            block = self._base("looks_switchcostumeto", parent)
            self.blocks[block_id] = block
            # Scratch's costume input is a dynamic menu, not a plain string
            # or number socket. The VM will execute a generic primitive shadow
            # here, but scratch-blocks expects an actual ``looks_costume``
            # shadow while constructing the editor workspace. Without it the
            # costume menu dereferences an absent shadow block and the toolbox
            # disappears when the sprite is selected.
            shadow_id = self._id()
            default_costume = (
                str(statement.value.value)
                if isinstance(statement.value, Literal)
                else "!"
            )
            self.blocks[shadow_id] = {
                **self._base("looks_costume", block_id),
                "fields": {"COSTUME": [default_costume, None]},
                "shadow": True,
            }
            if isinstance(statement.value, Literal):
                block["inputs"]["COSTUME"] = [1, shadow_id]
            else:
                value_input = self._expr(statement.value, block_id)
                block["inputs"]["COSTUME"] = [
                    3,
                    value_input[1],
                    shadow_id,
                ]
        else:
            raise TypeError(f"Unsupported Scratch DSL statement: {type(statement).__name__}")

        return block_id

    def _expr(self, expr: Expr, parent: str) -> list:
        if isinstance(expr, Literal):
            type_code = 10 if isinstance(expr.value, str) else 4
            return [1, [type_code, expr.value]]
        if isinstance(expr, Variable):
            return [3, [12, expr.name, self.var_ids[expr.name]], [10, ""]]

        block_id = self._id()
        if isinstance(expr, ListItem):
            block = self._base("data_itemoflist", parent)
            block["fields"]["LIST"] = [expr.name, self.list_ids[expr.name]]
            self.blocks[block_id] = block
            block["inputs"]["INDEX"] = self._expr(expr.index, block_id)
        elif isinstance(expr, ListLength):
            block = self._base("data_lengthoflist", parent)
            block["fields"]["LIST"] = [expr.name, self.list_ids[expr.name]]
            self.blocks[block_id] = block
        elif isinstance(expr, ListIndex):
            block = self._base("data_itemnumoflist", parent)
            block["fields"]["LIST"] = [expr.name, self.list_ids[expr.name]]
            self.blocks[block_id] = block
            block["inputs"]["ITEM"] = self._expr(expr.value, block_id)
        elif isinstance(expr, StringItem):
            block = self._base("operator_letter_of", parent)
            self.blocks[block_id] = block
            block["inputs"]["STRING"] = self._expr(expr.string, block_id)
            block["inputs"]["LETTER"] = self._expr(expr.index, block_id)
        elif isinstance(expr, StringLength):
            block = self._base("operator_length", parent)
            self.blocks[block_id] = block
            block["inputs"]["STRING"] = self._expr(expr.string, block_id)
        elif isinstance(expr, Join):
            block = self._base("operator_join", parent)
            self.blocks[block_id] = block
            block["inputs"]["STRING1"] = self._expr(expr.left, block_id)
            block["inputs"]["STRING2"] = self._expr(expr.right, block_id)
        elif isinstance(expr, Binary):
            block = self._base(expr.opcode, parent)
            self.blocks[block_id] = block
            left_name, right_name = _BINARY_INPUTS[expr.opcode]
            block["inputs"][left_name] = self._expr(expr.left, block_id)
            block["inputs"][right_name] = self._expr(expr.right, block_id)
        elif isinstance(expr, MathOp):
            block = self._base("operator_mathop", parent)
            block["fields"]["OPERATOR"] = [expr.operator, None]
            self.blocks[block_id] = block
            block["inputs"]["NUM"] = self._expr(expr.value, block_id)
        elif isinstance(expr, Round):
            block = self._base("operator_round", parent)
            self.blocks[block_id] = block
            block["inputs"]["NUM"] = self._expr(expr.value, block_id)
        elif isinstance(expr, Random):
            block = self._base("operator_random", parent)
            self.blocks[block_id] = block
            block["inputs"]["FROM"] = self._expr(expr.low, block_id)
            block["inputs"]["TO"] = self._expr(expr.high, block_id)
        elif isinstance(expr, CostumeNumber):
            block = self._base("looks_costumenumbername", parent)
            block["fields"]["NUMBER_NAME"] = ["number", None]
            self.blocks[block_id] = block
        else:
            raise TypeError(f"Unsupported Scratch DSL expression: {type(expr).__name__}")
        # Reporter blocks never participate in command chains.
        self.blocks[block_id]["next"] = None
        return [3, block_id, [4, 0]]


def _expand_static_unrolled_repeat(
    statement: StaticUnrolledRepeat,
    factor_limit: int,
) -> tuple[Statement, ...]:
    factor = min(statement.max_factor, factor_limit, statement.count)
    # Factors are deliberately powers of two; choose the largest supported
    # value at or below both caps.
    factor = max(candidate for candidate in (1, 2, 4, 8) if candidate <= factor)
    count = statement.count
    step = statement.step
    final = statement.final_step
    result: list[Statement] = []
    if final is None:
        chunks, remainder = divmod(count, factor)
        if chunks:
            result.append(Repeat(Literal(chunks), step * factor))
        result.extend(step * remainder)
        return tuple(result)

    # The final iteration may omit index mutations that are dead after the
    # loop. Keep it outside the repeated chunks for exact prior semantics.
    advancing = count - 1
    chunks, remainder = divmod(advancing, factor)
    if chunks:
        result.append(Repeat(Literal(chunks), step * factor))
    result.extend(step * remainder)
    result.extend(final)
    return tuple(result)


def _resolve_static_unrolling(
    statements: tuple[Statement, ...],
    factor_limit: int,
) -> tuple[Statement, ...]:
    resolved: list[Statement] = []
    for statement in statements:
        if isinstance(statement, StaticUnrolledRepeat):
            expanded = _expand_static_unrolled_repeat(statement, factor_limit)
            resolved.extend(_resolve_static_unrolling(expanded, factor_limit))
        elif isinstance(statement, Repeat):
            resolved.append(Repeat(
                statement.times,
                _resolve_static_unrolling(statement.body, factor_limit),
            ))
        elif isinstance(statement, RepeatUntil):
            resolved.append(RepeatUntil(
                statement.condition,
                _resolve_static_unrolling(statement.body, factor_limit),
            ))
        elif isinstance(statement, ForEach):
            resolved.append(ForEach(
                statement.variable,
                statement.times,
                _resolve_static_unrolling(statement.body, factor_limit),
            ))
        elif isinstance(statement, If):
            resolved.append(If(
                statement.condition,
                _resolve_static_unrolling(statement.body, factor_limit),
            ))
        elif isinstance(statement, IfElse):
            resolved.append(IfElse(
                statement.condition,
                _resolve_static_unrolling(statement.then_body, factor_limit),
                _resolve_static_unrolling(statement.else_body, factor_limit),
            ))
        else:
            resolved.append(statement)
    return tuple(resolved)


def _expr_uses_variable(expr: Expr, name: str) -> bool:
    if isinstance(expr, Variable):
        return expr.name == name
    if isinstance(expr, (Literal, ListLength, CostumeNumber)):
        return False
    if isinstance(expr, ListItem):
        return _expr_uses_variable(expr.index, name)
    if isinstance(expr, ListIndex):
        return _expr_uses_variable(expr.value, name)
    if isinstance(expr, StringItem):
        return _expr_uses_variable(expr.string, name) or _expr_uses_variable(
            expr.index, name,
        )
    if isinstance(expr, StringLength):
        return _expr_uses_variable(expr.string, name)
    if isinstance(expr, Join):
        return _expr_uses_variable(expr.left, name) or _expr_uses_variable(
            expr.right, name,
        )
    if isinstance(expr, Binary):
        return _expr_uses_variable(expr.left, name) or _expr_uses_variable(
            expr.right, name,
        )
    if isinstance(expr, (MathOp, Round)):
        return _expr_uses_variable(expr.value, name)
    if isinstance(expr, Random):
        return _expr_uses_variable(expr.low, name) or _expr_uses_variable(
            expr.high, name,
        )
    raise TypeError(type(expr).__name__)


def _statement_accesses_variable(statement: Statement, name: str) -> bool:
    if isinstance(statement, SetVariable):
        return statement.name == name or _expr_uses_variable(statement.value, name)
    if isinstance(statement, ChangeVariable):
        return statement.name == name or _expr_uses_variable(statement.value, name)
    if isinstance(statement, (ClearList, ProcedureCall)):
        return False
    if isinstance(statement, SwitchCostume):
        return _expr_uses_variable(statement.value, name)
    if isinstance(statement, AppendList):
        return _expr_uses_variable(statement.value, name)
    if isinstance(statement, ReplaceListItem):
        return _expr_uses_variable(statement.index, name) or _expr_uses_variable(
            statement.value, name,
        )
    if isinstance(statement, DeleteListItem):
        return _expr_uses_variable(statement.index, name)
    if isinstance(statement, (Repeat, ForEach)):
        return (
            (isinstance(statement, ForEach) and statement.variable == name)
            or _expr_uses_variable(statement.times, name)
            or any(_statement_accesses_variable(item, name) for item in statement.body)
        )
    if isinstance(statement, RepeatUntil):
        return _expr_uses_variable(statement.condition, name) or any(
            _statement_accesses_variable(item, name) for item in statement.body
        )
    if isinstance(statement, If):
        return _expr_uses_variable(statement.condition, name) or any(
            _statement_accesses_variable(item, name) for item in statement.body
        )
    if isinstance(statement, IfElse):
        return (
            _expr_uses_variable(statement.condition, name)
            or any(_statement_accesses_variable(item, name) for item in statement.then_body)
            or any(_statement_accesses_variable(item, name) for item in statement.else_body)
        )
    raise TypeError(type(statement).__name__)


def _statement_writes_variable(statement: Statement, name: str) -> bool:
    if isinstance(statement, (SetVariable, ChangeVariable)):
        return statement.name == name
    if isinstance(statement, (Repeat, ForEach, If, RepeatUntil)):
        return (
            isinstance(statement, ForEach) and statement.variable == name
        ) or any(_statement_writes_variable(item, name) for item in statement.body)
    if isinstance(statement, IfElse):
        return any(
            _statement_writes_variable(item, name)
            for item in (*statement.then_body, *statement.else_body)
        )
    return False


def _variable_dead_after(statements: tuple[Statement, ...], start: int, name: str) -> bool:
    for statement in statements[start:]:
        if isinstance(statement, SetVariable) and statement.name == name:
            return not _expr_uses_variable(statement.value, name)
        if _statement_accesses_variable(statement, name):
            return False
    return True


def _optimize_nested(statement: Statement) -> Statement:
    if isinstance(statement, Repeat):
        return Repeat(statement.times, _optimize_index_loops(statement.body))
    if isinstance(statement, RepeatUntil):
        return RepeatUntil(
            statement.condition,
            _optimize_index_loops(statement.body),
        )
    if isinstance(statement, ForEach):
        return ForEach(
            statement.variable,
            statement.times,
            _optimize_index_loops(statement.body),
        )
    if isinstance(statement, If):
        return If(statement.condition, _optimize_index_loops(statement.body))
    if isinstance(statement, IfElse):
        return IfElse(
            statement.condition,
            _optimize_index_loops(statement.then_body),
            _optimize_index_loops(statement.else_body),
        )
    return statement


def _optimize_index_loops(
    statements: tuple[Statement, ...],
) -> tuple[Statement, ...]:
    """Use Scratch's official one-based loop when it removes index mutation."""
    statements = tuple(_optimize_nested(statement) for statement in statements)
    result: list[Statement] = []
    index = 0
    while index < len(statements):
        initial = statements[index]
        if (
            isinstance(initial, SetVariable)
            and index + 1 < len(statements)
            and isinstance(statements[index + 1], Repeat)
            and not _expr_uses_variable(
                cast(Repeat, statements[index + 1]).times, initial.name,
            )
        ):
            loop = cast(Repeat, statements[index + 1])
            body = loop.body
            preincrement = (
                isinstance(initial.value, Literal)
                and initial.value.value == 0
                and body
                and isinstance(body[0], ChangeVariable)
                and body[0].name == initial.name
                and body[0].value == Literal(1)
                and not any(
                    _statement_writes_variable(item, initial.name)
                    for item in body[1:]
                )
            )
            postincrement = (
                isinstance(initial.value, Literal)
                and initial.value.value == 1
                and body
                and isinstance(body[-1], ChangeVariable)
                and body[-1].name == initial.name
                and body[-1].value == Literal(1)
                and not any(
                    _statement_writes_variable(item, initial.name)
                    for item in body[:-1]
                )
                and _variable_dead_after(statements, index + 2, initial.name)
            )
            if preincrement or postincrement:
                optimized_body = body[1:] if preincrement else body[:-1]
                result.append(ForEach(initial.name, loop.times, optimized_body))
                index += 2
                continue
        result.append(initial)
        index += 1
    return tuple(result)


def _render_expr(expr: Expr) -> str:
    if isinstance(expr, Literal):
        return repr(expr.value)
    if isinstance(expr, Variable):
        return expr.name
    if isinstance(expr, ListItem):
        return f"{expr.name}[{_render_expr(expr.index)}]"
    if isinstance(expr, ListLength):
        return f"length({expr.name})"
    if isinstance(expr, ListIndex):
        return f"index_of({_render_expr(expr.value)}, {expr.name})"
    if isinstance(expr, StringItem):
        return f"letter({_render_expr(expr.index)}, {_render_expr(expr.string)})"
    if isinstance(expr, StringLength):
        return f"length({_render_expr(expr.string)})"
    if isinstance(expr, Join):
        return f"join({_render_expr(expr.left)}, {_render_expr(expr.right)})"
    if isinstance(expr, Binary):
        symbol = {
            "operator_add": "+", "operator_subtract": "-",
            "operator_multiply": "*", "operator_divide": "/",
            "operator_mod": "%",
            "operator_equals": "==", "operator_lt": "<", "operator_gt": ">",
        }[expr.opcode]
        return f"({_render_expr(expr.left)} {symbol} {_render_expr(expr.right)})"
    if isinstance(expr, MathOp):
        return f"{expr.operator}({_render_expr(expr.value)})"
    if isinstance(expr, Round):
        return f"round({_render_expr(expr.value)})"
    if isinstance(expr, Random):
        return f"random({_render_expr(expr.low)}, {_render_expr(expr.high)})"
    if isinstance(expr, CostumeNumber):
        return "costume_number()"
    raise TypeError(type(expr).__name__)


def _render_statements(statements: Iterable[Statement], lines: list[str], depth: int) -> None:
    indent = "  " * depth
    for statement in statements:
        if isinstance(statement, StaticUnrolledRepeat):
            lines.append(
                f"{indent}repeat {statement.count} "
                f"[unroll <= {statement.max_factor}]:"
            )
            _render_statements(statement.step, lines, depth + 1)
        elif isinstance(statement, SetVariable):
            lines.append(f"{indent}{statement.name} = {_render_expr(statement.value)}")
        elif isinstance(statement, ChangeVariable):
            lines.append(f"{indent}{statement.name} += {_render_expr(statement.value)}")
        elif isinstance(statement, ClearList):
            lines.append(f"{indent}clear {statement.name}")
        elif isinstance(statement, AppendList):
            lines.append(f"{indent}{statement.name}.append({_render_expr(statement.value)})")
        elif isinstance(statement, ReplaceListItem):
            lines.append(
                f"{indent}{statement.name}[{_render_expr(statement.index)}] = "
                f"{_render_expr(statement.value)}"
            )
        elif isinstance(statement, DeleteListItem):
            lines.append(
                f"{indent}delete {statement.name}[{_render_expr(statement.index)}]"
            )
        elif isinstance(statement, Repeat):
            lines.append(f"{indent}repeat {_render_expr(statement.times)}:")
            _render_statements(statement.body, lines, depth + 1)
        elif isinstance(statement, RepeatUntil):
            lines.append(f"{indent}repeat until {_render_expr(statement.condition)}:")
            _render_statements(statement.body, lines, depth + 1)
        elif isinstance(statement, ForEach):
            lines.append(
                f"{indent}for each {statement.variable} in {_render_expr(statement.times)}:"
            )
            _render_statements(statement.body, lines, depth + 1)
        elif isinstance(statement, If):
            lines.append(f"{indent}if {_render_expr(statement.condition)}:")
            _render_statements(statement.body, lines, depth + 1)
        elif isinstance(statement, IfElse):
            lines.append(f"{indent}if {_render_expr(statement.condition)}:")
            _render_statements(statement.then_body, lines, depth + 1)
            lines.append(f"{indent}else:")
            _render_statements(statement.else_body, lines, depth + 1)
        elif isinstance(statement, ProcedureCall):
            lines.append(f"{indent}call {statement.name}")
        elif isinstance(statement, SwitchCostume):
            lines.append(f"{indent}switch costume to {_render_expr(statement.value)}")
