"""
tensor_replacer.py
------------------
Substitutes template list placeholders (T1, T2, T3, ...) with concrete
list names and IDs, given in order.

The sprite's own lists dict is used to resolve display name -> scratch ID,
so no external registry is needed.

Example
-------
    # T1 -> T3, T2 -> T4, T3 -> T7
    tr = TensorReplacer(sprite, ["T3", "T4", "T7"])
    new_blocks = tr.apply(sprite["blocks"])
"""


class TensorReplacer:
    def __init__(self, sprite: dict, tensor_names: list[str]):
        """
        Parameters
        ----------
        sprite : dict
            The full sprite JSON dict. Its "lists" section is used to resolve
            display name -> scratch ID for the target tensors.
        tensor_names : list of str
            The target list display names in order. The first replaces T1,
            the second replaces T2, and so on.

            Example: ["T3", "T4", "T7"] maps T1->T3, T2->T4, T3->T7.
        """
        # Build display_name -> scratch_id from the sprite's lists
        name_to_id = {
            entry[0]: sid
            for sid, entry in sprite.get("lists", {}).items()
        }

        # Build template_name -> (new_display_name, new_scratch_id)
        self._name_map: dict[str, str] = {}
        self._id_map: dict[str, str] = {}

        for i, target_name in enumerate(tensor_names):
            template_name = f"T{i + 1}"
            if target_name not in name_to_id:
                raise KeyError(
                    f"List '{target_name}' not found in sprite lists. "
                    f"Available: {list(name_to_id.keys())}"
                )
            self._name_map[template_name] = target_name
            self._id_map[template_name] = name_to_id[target_name]

    def apply(self, blocks: dict) -> dict:
        """
        Return a new blocks dict with all T1/T2/... LIST field references
        substituted with the target names and IDs.
        """
        return self._walk(blocks)

    def _walk(self, node):
        if isinstance(node, dict):
            # LIST fields look like: {"LIST": ["T1", "original_scratch_id"]}
            if "LIST" in node:
                display, sid = node["LIST"]
                if display in self._name_map:
                    return {
                        **node,
                        "LIST": [self._name_map[display], self._id_map[display]]
                    }
            return {k: self._walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [self._walk(item) for item in node]
        return node