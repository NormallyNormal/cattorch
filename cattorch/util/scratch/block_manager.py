"""
block_manager.py
----------------
Manages block ID uniqueness when stringing together generated Scratch fragments.

Each call to new_context() starts a fresh namespace — block IDs from the new
fragment are mapped to globally unique IDs so they never collide with IDs
from previous fragments.
"""

import uuid

from cattorch.util.scratch.remap import remap_ids


class BlockManager:
    def __init__(self):
        self._suffix: str = ""

    def new_context(self) -> None:
        """Start a new fragment context with a fresh UUID suffix."""
        self._suffix = uuid.uuid4().hex[:8]

    def apply_to_blocks(self, blocks: dict) -> dict:
        """Remap all block IDs in a fragment to globally unique IDs."""
        if not self._suffix:
            raise RuntimeError("Call new_context() before apply_to_blocks().")
        mapping = {bid: f"{bid}_{self._suffix}" for bid in blocks}
        return remap_ids(blocks, mapping)
