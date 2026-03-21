"""
block_combiner.py
-----------------
Chains two prepared+remapped Scratch sprite dicts together:
  1. Finds the last block in the first sprite's chain
  2. Sets the topmost block of the second sprite as its `next`
  3. Clears `topLevel` on that block and sets its `parent` to the last block
  4. Merges variables — same display name => single slot (first one wins)
  5. Merges lists — same display name => single slot (first one wins)
  6. Merges blocks — all blocks go into one combined dict

Usage
-----
    combined = combine(sprite_a, sprite_b)
"""


def _find_chain_end(blocks: dict, start_id: str) -> str:
    """
    Follow the `next` pointers from start_id until we reach a block
    with no `next`. Returns the ID of the last block in the chain.
    """
    current = start_id
    while True:
        nxt = blocks[current].get("next")
        if nxt is None:
            return current
        current = nxt


def _find_top_level_root(blocks: dict) -> str:
    """
    Return the ID of the single topLevel=true block that is the entry
    point of the main script chain (parent=null, topLevel=true).
    Raises if none or more than one is found.
    """
    roots = [
        bid for bid, block in blocks.items()
        if block.get("topLevel") and block.get("parent") is None
    ]
    if not roots:
        raise ValueError("No topLevel root block found.")
    if len(roots) > 1:
        raise ValueError(
            f"Multiple topLevel root blocks found: {roots}. "
            "Ensure the template has a single entry point."
        )
    return roots[0]


LOCAL_PREFIX = "_"


def _merge_slots(primary: dict, secondary: dict) -> dict:
    """
    Merge two variable or list dicts.
    - If a display name appears in both, the primary slot wins (its ID is kept).
    - Slots whose display name starts with LOCAL_PREFIX ("_") are never merged;
      each instruction keeps its own copy. If the ID collides, the secondary
      slot is given a unique ID and a remap entry is added.
    - Returns a new dict and a remapping {secondary_id: new_id} for any
      slots that were collapsed or renamed.
    """
    import uuid

    # Build display_name -> (id, entry) for the primary
    by_name: dict[str, tuple[str, list]] = {
        entry[0]: (sid, entry) for sid, entry in primary.items()
    }

    merged = dict(primary)
    remap: dict[str, str] = {}  # secondary_id -> winning/new_id

    for sid, entry in secondary.items():
        display_name = entry[0]
        is_local = display_name.startswith(LOCAL_PREFIX)

        if not is_local and display_name in by_name:
            # Shared slot: collapse secondary into primary
            winning_id = by_name[display_name][0]
            remap[sid] = winning_id
        elif sid in merged:
            # ID collision — assign a new unique ID
            new_id = f"{sid}_{uuid.uuid4().hex[:8]}"
            merged[new_id] = entry
            remap[sid] = new_id
            if not is_local:
                by_name[display_name] = (new_id, entry)
        else:
            # New slot, no collision
            merged[sid] = entry
            if not is_local:
                by_name[display_name] = (sid, entry)

    return merged, remap


def _apply_remap(blocks: dict, var_remap: dict, list_remap: dict) -> dict:
    """
    Walk every block and replace collapsed variable/list IDs with the
    winning IDs from the merge step.

    All replacements are applied in a single pass to prevent cascading
    (where the output of one replacement is consumed by a later one).
    """
    import json
    import re

    if not var_remap and not list_remap:
        return blocks

    all_remap = {**var_remap, **list_remap}

    # Drop identity remaps — they can't change anything and only add noise
    all_remap = {k: v for k, v in all_remap.items() if k != v}
    if not all_remap:
        return blocks

    raw = json.dumps(blocks)

    # Build a regex matching any quoted key, longest first
    pattern = "|".join(
        re.escape(f'"{k}"')
        for k in sorted(all_remap, key=len, reverse=True)
    )
    raw = re.sub(pattern, lambda m: f'"{all_remap[m.group(0)[1:-1]]}"', raw)

    return json.loads(raw)


def combine(sprite_a: dict, sprite_b: dict) -> dict:
    """
    Chain sprite_b's block chain onto the end of sprite_a's, merging
    variables and lists by display name.

    Parameters
    ----------
    sprite_a : dict  — the leading sprite (modified copy is returned)
    sprite_b : dict  — the trailing sprite (its chain is appended)

    Returns
    -------
    A new sprite dict with combined blocks, variables, and lists.
    """
    import copy
    a = copy.deepcopy(sprite_a)
    b = copy.deepcopy(sprite_b)

    blocks_a = a["blocks"]
    blocks_b = b["blocks"]

    # 1. Find the join points
    root_a = _find_top_level_root(blocks_a)
    last_a = _find_chain_end(blocks_a, root_a)

    root_b = _find_top_level_root(blocks_b)

    # 2. Merge variables and lists, getting ID remaps for collapsed slots
    merged_vars, var_remap = _merge_slots(
        a.get("variables", {}), b.get("variables", {})
    )
    merged_lists, list_remap = _merge_slots(
        a.get("lists", {}), b.get("lists", {})
    )

    # 3. Apply remaps to b's blocks so collapsed IDs point to the winning slots
    blocks_b = _apply_remap(blocks_b, var_remap, list_remap)

    # 4. Wire the chain: last block of a -> root of b
    blocks_a[last_a]["next"] = root_b
    blocks_b[root_b]["parent"] = last_a
    blocks_b[root_b]["topLevel"] = False
    blocks_b[root_b].pop("x", None)
    blocks_b[root_b].pop("y", None)

    # 5. Merge blocks
    combined_blocks = {**blocks_a, **blocks_b}

    return {
        **a,
        "variables": merged_vars,
        "lists": merged_lists,
        "blocks": combined_blocks,
    }