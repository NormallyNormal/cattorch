"""A last check on a finished sprite: every list a block reads exists and gets values.

Every export kind ends in ``finalize_sprite``, so this runs on the blocks that
are about to be written, after every rewrite (fusions, layer sharing, storage,
programs). It catches a class of compiler bugs that would otherwise give wrong
numbers silently: a block that reads a list which no longer exists (Scratch
then reads an empty list), or a list whose items are read but which nothing
ever fills and which starts empty (a value that was never computed).

The second check covers the lists the tensor compiler makes for tensors and
weights. Such a list counts as filled when a block adds or replaces its
items, when the sprite stores it with items, or when some block reads its
length (code that reads the length handles any length, including the empty
list of a zero-element tensor).
"""

from __future__ import annotations

import re

from cattorch import interface

#: Blocks that put items in their list. Clearing doesn't: every weight list is
#: cleared before it is unpacked, and when a project is prepared for saving.
_WRITES = frozenset({"data_addtolist", "data_replaceitemoflist", "data_insertatlist"})
#: Blocks that read their list's items. Code that reads a list's length too
#: handles any length, such as a copy of a list that may be empty (a
#: zero-element tensor, an empty initial state); code that reads only items
#: expects them to be there.
_ITEM_READS = frozenset({"data_itemoflist"})
#: The lists the tensor compiler names: a tensor's list (``T3``) or a
#: weight's (``W_name``). Other lists, such as a tokenizer's tables, can be
#: empty by design and are read behind their own checks.
_COMPILED_LIST = re.compile(r"T\d+|W_.+")


def check_list_dataflow(sprite: dict, public_data_names: frozenset[str] = frozenset()) -> None:
    """Raise an internal error if a block reads a missing or never-filled list.

    It looks at each list once, so it costs one pass over the blocks.
    """
    lists = sprite.get("lists", {})
    written: set[str] = set()
    measured: set[str] = set()
    read: dict[str, str] = {}
    missing: dict[str, str] = {}
    for block in sprite.get("blocks", {}).values():
        if not isinstance(block, dict):
            if isinstance(block, list) and len(block) >= 3 and block[0] == 13:
                # A top-level list reporter.
                if block[2] not in lists:
                    missing.setdefault(block[2], block[1])
            continue
        field = block.get("fields", {}).get("LIST")
        if field is not None and len(field) >= 2:
            name, identifier = field[0], field[1]
            if identifier not in lists:
                missing.setdefault(identifier, name)
            opcode = block.get("opcode")
            if opcode in _WRITES:
                written.add(identifier)
            elif opcode in _ITEM_READS:
                read.setdefault(identifier, name)
            elif opcode == "data_lengthoflist":
                measured.add(identifier)
        for value in block.get("inputs", {}).values():
            for entry in value[1:] if isinstance(value, list) else ():
                if isinstance(entry, list) and len(entry) >= 3 and entry[0] == 13:
                    if entry[2] not in lists:
                        missing.setdefault(entry[2], entry[1])
    if missing:
        names = ", ".join(sorted(repr(name) for name in missing.values()))
        raise RuntimeError(
            f"cattorch internal error: the generated blocks read lists the sprite "
            f"doesn't have ({names}); please report this model"
        )
    never_filled = sorted(
        repr(lists[identifier][0])
        for identifier in read
        if identifier not in written
        and identifier not in measured
        and not lists[identifier][1]
        and _COMPILED_LIST.fullmatch(lists[identifier][0])
        and not interface.is_public_data(lists[identifier][0], public_data_names)
    )
    if never_filled:
        raise RuntimeError(
            f"cattorch internal error: the generated blocks read lists that nothing "
            f"fills ({', '.join(never_filled)}); please report this model"
        )
