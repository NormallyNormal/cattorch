"""
finalize_sprite.py
------------------
Takes a completed sprite dict, adds Scratch metadata, writes sprite.json,
copies in the costume SVG (renamed to a unique ID to avoid conflicts),
and zips both into a .sprite3 file.

Usage
-----
    from cattorch.util.scratch.finalize_sprite import finalize_sprite
    finalize_sprite(data, "my_model.sprite3")
"""

import copy
import hashlib
import json
import logging
import math
import shutil
import tempfile
import uuid
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

from cattorch.codegen import CodegenConfig
from cattorch.storage import BASE85_ALPHABET
from cattorch.templates.template import TEMPLATE_DIR
from cattorch.util.scratch.ids import compact_sprite_ids

log = logging.getLogger(__name__)

SCRATCH_ONLINE_JSON_LIMIT = 5 * 1024 * 1024
SCRATCH_ONLINE_JSON_WARN_SIZE = 4 * 1024 * 1024
SCRATCH_MAX_LIST_LENGTH = 200_000

SPRITE_ASSET_DIR = TEMPLATE_DIR / "sprite"


@dataclass(frozen=True)
class FinalizedSprite:
    path: Path
    archive_bytes: int
    expanded_json_bytes: int
    asset_id: str
    warnings: tuple[str, ...]


def online_json_size_warning(byte_count: int, subject: str) -> str | None:
    """Describe risk on Scratch's ordinary expanded-JSON save/upload path.

    Scratch does not have a universal 5 MiB cap on compressed ``.sb3`` or
    ``.sprite3`` archives. The commonly encountered limit applies to the
    expanded project JSON sent by the ordinary online editor. Other import and
    legacy archive-upload paths can behave differently.
    """
    size_mib = byte_count / 1024 / 1024
    if byte_count > SCRATCH_ONLINE_JSON_LIMIT:
        return (
            f"{subject} is {size_mib:.1f} MiB, above the roughly 5 MiB "
            "project.json limit used by Scratch's ordinary online save/upload "
            "path. Local import or an archive-based path may still work."
        )
    if byte_count > SCRATCH_ONLINE_JSON_WARN_SIZE:
        return (
            f"{subject} is {size_mib:.1f} MiB, approaching the roughly 5 MiB "
            "project.json limit used by Scratch's ordinary online save/upload path."
        )
    return None


def _json_safe_numbers(value):
    """Encode non-finite Scratch numbers without emitting invalid JSON.

    Scratch casts these strings back to their numeric meanings when a numeric
    reporter consumes them. Python's default ``json`` encoder instead writes
    bare Infinity/NaN tokens, which the official Scratch parser rejects.
    """
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_numbers(item) for item in value]
    return value

def _make_asset_id(sprite_name: str) -> str:
    """
    Derive a stable asset ID from the sprite name so the same model always
    produces the same ID, but different models don't conflict.
    Returns a 32-char hex string matching Scratch's md5 asset ID format.
    """
    return hashlib.md5(sprite_name.encode()).hexdigest()


_PUBLIC_DATA_NAMES = {
    "input", "output", "token_ids",
    "cattorch status", "cattorch cache length", "cattorch max context",
    "cattorch top k values", "cattorch top k ids",
    "cattorch storage compression", "cattorch storage precision",
    "cattorch storage scale precision",
}
_PUBLIC_PROCEDURES = {
    "cattorch init", "cattorch forward", "cattorch prepare for save",
    "cattorch reset cache", "cattorch prefill", "cattorch decode",
    "cattorch tokenize", "cattorch detokenize",
}


def _compact_internal_display_names(sprite: dict) -> None:
    """Shorten private UI names while preserving documented interfaces."""
    data_names: dict[str, str] = {}
    counters = {"variables": 0, "lists": 0}
    prefixes = {"variables": "v", "lists": "l"}
    for section in ("variables", "lists"):
        for identifier, entry in sprite.get(section, {}).items():
            old_name = entry[0]
            if old_name in _PUBLIC_DATA_NAMES or old_name.startswith("input_"):
                continue
            new_name = f"{prefixes[section]}{counters[section]}"
            counters[section] += 1
            entry[0] = new_name
            data_names[identifier] = new_name

    procedure_names: dict[str, str] = {}
    procedure_counter = 0
    for block in sprite.get("blocks", {}).values():
        mutation = block.get("mutation")
        if not isinstance(mutation, dict):
            continue
        proccode = mutation.get("proccode")
        if (
            isinstance(proccode, str)
            and proccode not in _PUBLIC_PROCEDURES
            and proccode not in procedure_names
        ):
            procedure_names[proccode] = f"p{procedure_counter}"
            procedure_counter += 1

    stack = [sprite.get("blocks", {})]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            mutation = value.get("mutation")
            if isinstance(mutation, dict):
                proccode = mutation.get("proccode")
                if proccode in procedure_names:
                    mutation["proccode"] = procedure_names[proccode]
            stack.extend(value.values())
        elif isinstance(value, list):
            # Scratch field/reference arrays redundantly carry a display name
            # immediately before their typed data ID.
            for index in range(1, len(value)):
                identifier = value[index]
                if (
                    isinstance(identifier, str)
                    and identifier in data_names
                    and isinstance(value[index - 1], str)
                ):
                    value[index - 1] = data_names[identifier]
            stack.extend(value)


def _trim_safe_schema_defaults(sprite: dict) -> None:
    """Drop defaults accepted and preserved by the official Scratch VM."""
    for block in sprite.get("blocks", {}).values():
        if (
            block.get("shadow") is False
            and block.get("opcode") != "procedures_prototype"
        ):
            block.pop("shadow")


def finalize_sprite(
    data: dict,
    output_path: str | Path,
    sprite_name: str = "cattorch",
    *,
    codegen: CodegenConfig | None = None,
) -> FinalizedSprite:
    """
    Finalize a sprite dict and write it as a .sprite3 zip file.

    Parameters
    ----------
    data : dict
        The completed sprite dict (blocks, variables, lists already set).
    output_path : str
        Path for the output .sprite3 file, e.g. "my_model.sprite3".
    sprite_name : str
        Name for the sprite inside Scratch. Also used to derive the asset ID.

    Returns
    -------
    Metadata for the written .sprite3 file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codegen = codegen or CodegenConfig()
    if not isinstance(codegen, CodegenConfig):
        raise TypeError("codegen must be a CodegenConfig")
    sprite = copy.deepcopy(data)
    _ensure_costume_menu_shadows(sprite)
    sprite = compact_sprite_ids(sprite, namespace=codegen.id_namespace)
    if codegen.compact_internal_names:
        _compact_internal_display_names(sprite)
    if codegen.compact_schema:
        _trim_safe_schema_defaults(sprite)

    # Derive a unique asset ID for this sprite's costume
    asset_id = _make_asset_id(sprite_name)
    md5ext = f"{asset_id}.svg"

    # Find the source SVG asset.
    source_svg_candidates = list(SPRITE_ASSET_DIR.glob("*.svg"))
    if not source_svg_candidates:
        raise FileNotFoundError(
            f"No SVG costume found in {SPRITE_ASSET_DIR}. "
            "Add a .svg file there to use as the sprite costume."
        )
    source_svg = source_svg_candidates[0]

    costume = {
        "bitmapResolution": 1,
        "dataFormat": "svg",
        "assetId": asset_id,
        "md5ext": md5ext,
        "rotationCenterX": 48,
        "rotationCenterY": 50,
    }
    uses_costume_codec = any(
        block.get("opcode") == "looks_switchcostumeto"
        for block in sprite.get("blocks", {}).values()
    )
    costume_names = BASE85_ALPHABET if uses_costume_codec else ("cat",)

    # Codec costumes share one physical SVG, adding only their JSON metadata.
    sprite.update({
        "name": sprite_name,
        "comments": {},
        "currentCostume": 0,
        "costumes": [{**costume, "name": name} for name in costume_names],
        "sounds": [],
        "volume": 100,
        "visible": True,
        "x": 0,
        "y": 0,
        "size": 100,
        "direction": 90,
        "draggable": False,
        "rotationStyle": "all around",
    })

    # Check list sizes before writing
    for sid, entry in sprite.get("lists", {}).items():
        name, contents = entry[0], entry[1]
        if len(contents) > SCRATCH_MAX_LIST_LENGTH:
            raise ValueError(
                f"List \"{name}\" has {len(contents):,} items, "
                f"which exceeds Scratch's limit of {SCRATCH_MAX_LIST_LENGTH:,}."
            )

    # Write into a temp dir then zip
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)

        # Write sprite.json
        sprite_json_path = temporary_path / "sprite.json"
        with open(sprite_json_path, "w") as f:
            json.dump(
                _json_safe_numbers(sprite),
                f,
                allow_nan=False,
                separators=(",", ":"),
            )
        raw_json_size = sprite_json_path.stat().st_size

        # Copy SVG with the new asset ID filename
        svg_dest = temporary_path / md5ext
        shutil.copy(source_svg, svg_dest)

        # Zip both into the output .sprite3
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(sprite_json_path, "sprite.json")
            zf.write(svg_dest, md5ext)

    warning_messages = []
    json_warning = online_json_size_warning(raw_json_size, "Raw sprite JSON")
    if json_warning is not None:
        warning_messages.append(json_warning)
    if (
        codegen.target_json_bytes is not None
        and raw_json_size > codegen.target_json_bytes
    ):
        target_mib = codegen.target_json_bytes / 1024 / 1024
        warning_messages.append(
            f"Raw sprite JSON exceeds the configured {target_mib:.1f} MiB "
            "code-generation target. Use unrolling='compact' to minimize "
            "generated blocks, or raise the target for archive-based use."
        )

    file_size = output_path.stat().st_size

    for message in warning_messages:
        warnings.warn(message, stacklevel=2)
    log.info("Written: %s (asset_id: %s)", output_path, asset_id)
    return FinalizedSprite(
        path=output_path,
        archive_bytes=file_size,
        expanded_json_bytes=raw_json_size,
        asset_id=asset_id,
        warnings=tuple(warning_messages),
    )


def _ensure_costume_menu_shadows(sprite: dict) -> None:
    """Give every dynamic costume input the shadow required by scratch-blocks.

    The Scratch VM accepts primitive string/number shadows for
    ``looks_switchcostumeto``, but the vanilla editor's dynamic costume menu
    assumes that the shadow is a real ``looks_costume`` block. Loading a
    sprite with the generic form otherwise crashes toolbox rendering.
    """
    blocks = sprite.get("blocks", {})
    reserved = set(blocks)
    for section in ("variables", "lists", "broadcasts", "comments"):
        reserved.update(sprite.get(section, {}))

    for block_id, block in list(blocks.items()):
        if block.get("opcode") != "looks_switchcostumeto":
            continue
        costume_input = block.get("inputs", {}).get("COSTUME")
        if not isinstance(costume_input, list) or len(costume_input) < 2:
            continue
        if (
            len(costume_input) >= 3
            and isinstance(costume_input[2], str)
            and blocks.get(costume_input[2], {}).get("opcode") == "looks_costume"
        ):
            continue
        if (
            len(costume_input) == 2
            and isinstance(costume_input[1], str)
            and blocks.get(costume_input[1], {}).get("opcode") == "looks_costume"
        ):
            continue

        default = "!"
        primary = costume_input[1]
        if (
            len(costume_input) == 2
            and isinstance(primary, list)
            and len(primary) >= 2
        ):
            default = str(primary[1])

        shadow_id = uuid.uuid4().hex
        while shadow_id in reserved:
            shadow_id = uuid.uuid4().hex
        reserved.add(shadow_id)
        blocks[shadow_id] = {
            "opcode": "looks_costume",
            "next": None,
            "parent": block_id,
            "inputs": {},
            "fields": {"COSTUME": [default, None]},
            "shadow": True,
            "topLevel": False,
        }
        block["inputs"]["COSTUME"] = (
            [1, shadow_id]
            if len(costume_input) == 2
            else [3, primary, shadow_id]
        )
