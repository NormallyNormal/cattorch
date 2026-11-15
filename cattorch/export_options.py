"""Common public export-option validation, independent of the input interface."""

from cattorch.adapters import ModuleAdapter
from cattorch.codegen import CodegenConfig
from cattorch.fast import FastConfig
from cattorch.quantization import QuantizationConfig
from cattorch.storage import StorageConfig


def validate_export_options(
    *,
    name=None,
    optimization="exact",
    fast_config=None,
    storage=None,
    quantization=None,
    codegen=None,
    sig_figs=None,
    adapters=(),
    example_kwargs=None,
    calibration=None,
) -> None:
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError("name must be a non-empty string")
    if optimization not in {"exact", "fast"}:
        raise ValueError(f"optimization must be 'exact' or 'fast', got {optimization!r}")
    if optimization != "fast" and fast_config is not None:
        raise ValueError("fast_config can only be used with optimization='fast'")
    for label, value, expected in (
        ("fast_config", fast_config, FastConfig),
        ("storage", storage, StorageConfig),
        ("quantization", quantization, QuantizationConfig),
        ("codegen", codegen, CodegenConfig),
    ):
        if value is not None and not isinstance(value, expected):
            raise TypeError(f"{label} must be a {expected.__name__}")
    if sig_figs is not None and (
        isinstance(sig_figs, bool) or not isinstance(sig_figs, int) or sig_figs < 1
    ):
        raise ValueError("sig_figs must be a positive integer or None")
    if isinstance(adapters, ModuleAdapter) or not isinstance(adapters, (tuple, list)):
        raise TypeError("adapters must be a tuple or list of ModuleAdapter values")
    if any(not isinstance(adapter, ModuleAdapter) for adapter in adapters):
        raise TypeError("adapters must contain ModuleAdapter values")
    if calibration is not None and (
        isinstance(calibration, (str, bytes)) or not hasattr(calibration, "__iter__")
    ):
        raise TypeError("calibration must be an iterable of calibration items, such as a list")
    if example_kwargs is not None and (
        not isinstance(example_kwargs, dict)
        or any(not isinstance(key, str) for key in example_kwargs)
    ):
        raise TypeError("example_kwargs must be a dict with string keys")


def validate_model_and_inputs(model, example_inputs, *, programs: bool = True) -> None:
    """Check ``model`` and the form of ``example_inputs``, before the model runs.

    ``example_inputs`` is a tensor, a tuple or list of tensors, ``None``
    (the model's markers) or, with ``programs``, an ``ExportProgram``,
    ``GenerationProgram`` or ``RecurrentGenerationProgram``. The tensors in
    a tuple are checked where they are used, with messages for each mistake.
    """
    import torch
    from torch import nn

    from cattorch.program import ExportProgram, GenerationProgram, RecurrentGenerationProgram

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be a torch.nn.Module, not {type(model).__name__}")
    kinds = (ExportProgram, GenerationProgram, RecurrentGenerationProgram)
    if example_inputs is None or isinstance(example_inputs, (torch.Tensor, tuple, list)):
        return
    if isinstance(example_inputs, kinds):
        if programs:
            return
        raise TypeError(
            f"example_inputs must be a tensor or a tuple of tensors here, not a "
            f"{type(example_inputs).__name__}"
        )
    if isinstance(example_inputs, dict):
        raise TypeError(
            "example_inputs can't be a dict: pass forward's positional tensors as a tensor "
            "or a tuple, and its keyword arguments with example_kwargs="
        )
    expected = "a tensor or a tuple of tensors" + (
        ", an ExportProgram, a GenerationProgram or a RecurrentGenerationProgram"
        if programs else ""
    )
    raise TypeError(f"example_inputs must be {expected}, not {type(example_inputs).__name__}")
