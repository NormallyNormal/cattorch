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

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

# Values per step when a StaticValues is iterated, to bound the temporary list.
_ITERATION_CHUNK = 1 << 16


class StaticValues:
    """A floating static tensor's flat values, kept as an array.

    A Python list holds every value as its own float object, about 32 bytes
    each; the tensor's own array holds them in 4. Storage encodes a weight
    from its values without ever needing them as Python objects, so a weight
    list holds one of these from the moment it is attached until storage
    replaces it with its encoded payload (or, uncompressed, with the list).

    It stands in for the list wherever static contents pass: ``len``,
    slicing (which returns a view), comparison with another StaticValues or
    a list, iteration, and ``numpy.asarray``. Anything else a list does is
    deliberately missing, so a new use fails instead of silently doing
    something else (``+`` on an array would add elementwise).
    """

    __slots__ = ("array",)

    def __init__(self, array: np.ndarray):
        if array.ndim != 1:
            raise ValueError("static values must be one-dimensional")
        self.array = array

    @classmethod
    def from_tensor(cls, tensor) -> StaticValues:
        """Wrap a floating tensor's values, sharing its memory when possible."""
        import torch

        values = tensor.detach().reshape(-1)
        if values.dtype not in (torch.float16, torch.float32, torch.float64):
            # bfloat16 and other formats numpy lacks widen exactly to float32.
            values = values.to(torch.float32)
        return cls(values.numpy(force=True))

    def __len__(self) -> int:
        return len(self.array)

    def __bool__(self) -> bool:
        return len(self.array) > 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            return StaticValues(self.array[index])
        return self.array[index].item()

    def __iter__(self) -> Iterator[float]:
        for start in range(0, len(self.array), _ITERATION_CHUNK):
            yield from self.array[start:start + _ITERATION_CHUNK].tolist()

    def __eq__(self, other) -> bool:
        # As lists of floats compare: elementwise, so NaN never equals NaN
        # (each ``tolist`` makes new NaN objects) and -0.0 equals 0.0.
        if isinstance(other, StaticValues):
            return bool(np.array_equal(self.array, other.array))
        if isinstance(other, list):
            return self.tolist() == other
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def __array__(self, dtype=None, copy=None):
        if copy:
            return np.array(self.array, dtype=dtype)
        return np.asarray(self.array, dtype=dtype)

    def tolist(self) -> list[float]:
        return self.array.tolist()

    def __repr__(self) -> str:
        return f"StaticValues({len(self.array)} x {self.array.dtype})"


def static_list_contents(values):
    """``values`` as a plain list, for the JSON a static list is written as."""
    return values.tolist() if isinstance(values, StaticValues) else values


def _make_id(name: str) -> str:
    """Derive a stable scratch ID from a tensor display name."""
    return f"cattorch_tensor_{name}"


def _flatten(tensor, sig_figs: int | None = None) -> list:
    """Flatten a tensor or nested list to a plain Python list of scalars."""
    try:
        # torch.Tensor
        values = tensor.detach().flatten().tolist()
    except AttributeError:
        try:
            # numpy array
            values = tensor.flatten().tolist()
        except AttributeError:
            # Plain Python list/nested list
            def _recurse(x):
                if isinstance(x, (list, tuple)):
                    for item in x:
                        yield from _recurse(item)
                else:
                    yield x
            values = list(_recurse(tensor))

    # Integer tensors, such as indices or token IDs, stay exact.
    is_floating = getattr(tensor, "is_floating_point", None)
    integral = callable(is_floating) and not is_floating()
    if sig_figs is not None and not integral:
        values = [float(f"%.{sig_figs}g" % v) for v in values]
    return values


def _contents(tensor, sig_figs: int | None):
    """A floating tensor's values as StaticValues; anything else as a list."""
    is_floating = getattr(tensor, "is_floating_point", None)
    if (
        sig_figs is None
        and callable(is_floating)
        and is_floating()
        and hasattr(tensor, "detach")
    ):
        return StaticValues.from_tensor(tensor)
    return _flatten(tensor, sig_figs)


class TensorAdder:
    def apply(
        self,
        sprite: dict,
        tensor_names: list[str],
        weights: dict | None = None,
        sig_figs: int | None = None,
    ) -> dict:
        """
        Return a new sprite dict with the given tensor names added to the
        lists section. Tensors that already exist (by display name) are
        left unchanged.

        Only the lists section is copied: the result shares its blocks,
        variables and other sections with ``sprite``, so a caller that keeps
        using ``sprite`` must not change them through either dict.

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
        sprite = dict(sprite)
        lists = sprite["lists"] = {
            identifier: list(entry) for identifier, entry in sprite.get("lists", {}).items()
        }
        weights = weights or {}

        # Build a set of existing display names to avoid duplicates
        existing = {entry[0] for entry in lists.values()}

        for name in tensor_names:
            if name not in existing:
                data = _contents(weights[name], sig_figs) if name in weights else []
                lists[_make_id(name)] = [name, data]
                existing.add(name)
            elif name in weights:
                # Slot already exists — update its data
                sid = next(sid for sid, entry in lists.items() if entry[0] == name)
                lists[sid][1] = _contents(weights[name], sig_figs)

        return sprite
