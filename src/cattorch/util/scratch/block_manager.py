"""
block_manager.py
----------------
Manages block ID uniqueness when stringing together multiple Scratch templates.

Each call to new_context() starts a fresh namespace — block IDs from the new
template are mapped to globally unique IDs so they never collide with IDs
from previous templates.

Example
-------
    bm = BlockManager()

    bm.new_context()
    bm.get_assignment("cattorch_block_1")   # -> "cattorch_block_1_a3f9c6b0"
    bm.get_assignment("cattorch_block_2")   # -> "cattorch_block_2_a3f9c6b0"

    bm.new_context()
    bm.get_assignment("cattorch_block_1")   # -> "cattorch_block_1_b7c24f11"  (different suffix)
    bm.get_assignment("cattorch_block_2")   # -> "cattorch_block_2_b7c24f11"
"""

import json
import uuid


class BlockManager:
    def __init__(self):
        # Maps template-local block ID -> globally unique block ID
        # Reset on each new_context()
        self.assignments: dict[str, str] = {}

        # The suffix applied to all block IDs in the current context
        self._context_suffix: str = ""

    # -------------------------------------------------------------------------
    # Context management
    # -------------------------------------------------------------------------

    def new_context(self) -> None:
        """
        Start a new template context. Clears the current assignment map and
        generates a fresh suffix so all block IDs in this context are unique
        relative to every previous context.
        """
        self.assignments = {}
        self._context_suffix = uuid.uuid4().hex[:8]

    # -------------------------------------------------------------------------
    # Assignment lookup
    # -------------------------------------------------------------------------

    def get_assignment(self, block_name: str) -> str:
        """
        Return the globally unique ID for a template-local block name.
        If this is the first time we've seen this name in the current context,
        a new assignment is created and cached.

        Raises RuntimeError if called before new_context().
        """
        if not self._context_suffix:
            raise RuntimeError("Call new_context() before get_assignment().")

        if block_name not in self.assignments:
            self.assignments[block_name] = f"{block_name}_{self._context_suffix}"

        return self.assignments[block_name]

    # -------------------------------------------------------------------------
    # Bulk application — remap an entire prepared template's block IDs
    # -------------------------------------------------------------------------

    def apply_to_blocks(self, blocks: dict) -> dict:
        """
        Given a blocks dict (from a prepared template), return a new dict with
        all block IDs — both keys and all internal next/parent/inputs references —
        remapped through get_assignment().

        Must be called after new_context().
        """
        # Register all top-level IDs first so the map is complete
        for block_id in blocks:
            self.get_assignment(block_id)

        # Serialise, replace all ID strings, then rebuild with remapped keys.
        # We only replace quoted occurrences ("id") to avoid touching display names
        # or other values that might coincidentally share a substring.
        raw = json.dumps(blocks)
        for old, new in sorted(self.assignments.items(), key=lambda x: -len(x[0])):
            raw = raw.replace(f'"{old}"', f'"{new}"')

        # The keys in the parsed result are already remapped by the string
        # replacement above — no need to call get_assignment again.
        return json.loads(raw)