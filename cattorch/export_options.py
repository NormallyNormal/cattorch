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
    frontend="fx",
    adapters=(),
) -> None:
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError("name must be a non-empty string")
    if optimization not in {"exact", "fast"}:
        raise ValueError(f"optimization must be 'exact' or 'fast', got {optimization!r}")
    if frontend not in {"fx", "export"}:
        raise ValueError(f"frontend must be 'fx' or 'export', got {frontend!r}")
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
    if quantization is not None and storage is not None and storage.precision.startswith("int"):
        raise ValueError(
            "quantization cannot be combined with legacy integer StorageConfig.precision"
        )
    if sig_figs is not None and (
        isinstance(sig_figs, bool) or not isinstance(sig_figs, int) or sig_figs < 1
    ):
        raise ValueError("sig_figs must be a positive integer or None")
    if any(not isinstance(adapter, ModuleAdapter) for adapter in adapters):
        raise TypeError("adapters must contain ModuleAdapter values")
