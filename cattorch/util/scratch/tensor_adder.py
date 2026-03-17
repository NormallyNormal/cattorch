"""
tensor_adder.py
---------------
Adds named tensor lists to a Scratch sprite JSON dict.

Each tensor becomes a list slot in the sprite's "lists" section.
Tensors can optionally be preloaded with flattened weight data.

Example
-------
    import torch
    ta = TensorAdder()

    # Add empty tensors
    sprite = ta.apply(sprite, ["x", "T1", "T2"])

    # Add tensors with preloaded weights
    w1 = torch.randn(4, 8)
    w2 = torch.randn(8, 2)
    sprite = ta.apply(sprite, ["p_w1", "p_w2"], weights={"p_w1": w1, "p_w2": w2})
"""

import copy


def _make_id(name: str) -> str:
    """Derive a stable scratch ID from a tensor display name."""
    return f"cattorch_tensor_{name}"


def _flatten(tensor) -> list:
    """Flatten a tensor or nested list to a plain Python list of scalars."""
    try:
        # torch.Tensor
        return tensor.detach().flatten().tolist()
    except AttributeError:
        pass
    try:
        # numpy array
        return tensor.flatten().tolist()
    except AttributeError:
        pass
    # Plain Python list/nested list
    def _recurse(x):
        if isinstance(x, (list, tuple)):
            for item in x:
                yield from _recurse(item)
        else:
            yield x
    return list(_recurse(tensor))


class TensorAdder:
    def apply(
        self,
        sprite: dict,
        tensor_names: list[str],
        weights: dict | None = None,
    ) -> dict:
        """
        Return a new sprite dict with the given tensor names added to the
        lists section. Tensors that already exist (by display name) are
        left unchanged.

        Parameters
        ----------
        sprite : dict
            The full sprite JSON dict to add tensors to.
        tensor_names : list of str
            Display names for each tensor list to add.
            e.g. ["x", "p_w1", "p_w2", "T1", "T2"]
        weights : dict, optional
            Maps tensor display name -> tensor data (torch.Tensor, numpy
            array, or nested Python list). The data is flattened to a 1D
            list and stored as the initial list contents.
            Tensors not in this dict are added with empty contents.
        """
        sprite = copy.deepcopy(sprite)
        lists = sprite.setdefault("lists", {})
        weights = weights or {}

        # Build a set of existing display names to avoid duplicates
        existing = {entry[0] for entry in lists.values()}

        for name in tensor_names:
            if name not in existing:
                data = _flatten(weights[name]) if name in weights else []
                lists[_make_id(name)] = [name, data]
                existing.add(name)
            elif name in weights:
                # Slot already exists — update its data
                sid = next(sid for sid, entry in lists.items() if entry[0] == name)
                lists[sid][1] = _flatten(weights[name])

        return sprite

    def remove(self, sprite: dict, tensor_names: list[str]) -> dict:
        """
        Return a new sprite dict with the given tensor names removed from
        the lists section. Silently skips names that don't exist.
        """
        sprite = copy.deepcopy(sprite)
        lists = sprite.get("lists", {})

        to_delete = [
            sid for sid, entry in lists.items()
            if entry[0] in tensor_names
        ]
        for sid in to_delete:
            del lists[sid]

        return sprite