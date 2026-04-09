"""
remap.py
--------
Single-pass JSON string replacement for remapping Scratch IDs.

Used throughout cattorch to rename block, list, and variable IDs without
cascading replacements (where the output of one replacement is consumed
by a later one).
"""

import json
import re


def remap_ids(data: dict, mapping: dict[str, str]) -> dict:
    """Replace all quoted ID strings in a JSON-serializable dict.

    Uses a single-pass regex so replacements never cascade.

    Parameters
    ----------
    data : dict
        The dict to transform (e.g. a blocks dict or full sprite).
    mapping : dict
        {old_id: new_id} pairs. Identity mappings are skipped.

    Returns
    -------
    A new dict with all occurrences of "old_id" replaced by "new_id".
    """
    mapping = {k: v for k, v in mapping.items() if k != v}
    if not mapping:
        return data

    raw = json.dumps(data)
    pattern = "|".join(
        re.escape(f'"{k}"')
        for k in sorted(mapping, key=len, reverse=True)
    )
    raw = re.sub(pattern, lambda m: f'"{mapping[m.group(0)[1:-1]]}"', raw)
    return json.loads(raw)
