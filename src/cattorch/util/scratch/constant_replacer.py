"""
constant_replacer.py
--------------------
Substitutes template placeholder constants (101, 102, 103, ...) with
concrete int or float values in a Scratch blocks dict.

Example
-------
    cr = ConstantReplacer({101: 60, 102: 7, 103: 6})
    new_blocks = cr.apply(blocks)
"""

import json


class ConstantReplacer:
    def __init__(self, mapping: dict[int | float, int | float]):
        """
        Parameters
        ----------
        mapping : dict
            Keys are the placeholder constants as they appear in the prepared
            template (e.g. 101, 102, 103). Values are the concrete replacements
            (int or float).

        Example
        -------
            ConstantReplacer({101: 60, 102: 7, 103: 6})
            ConstantReplacer({101: 60, 102: 7.5, 103: 6})
        """
        self.mapping = mapping

    def apply(self, blocks: dict) -> dict:
        """
        Return a new blocks dict with all placeholder constants substituted.
        Operates on the parsed JSON values directly — only replaces numeric
        literals that exactly match a mapping key.
        """
        return self._walk(blocks)

    def _replace(self, value):
        """Replace a scalar value if it matches a mapping key."""
        if isinstance(value, (int, float)) and value in self.mapping:
            return self.mapping[value]
        return value

    def _walk(self, node):
        """Recursively walk the block structure, replacing matching values."""
        if isinstance(node, dict):
            return {k: self._walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [self._walk(item) for item in node]
        return self._replace(node)