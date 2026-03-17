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
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path

from cattorch.templates.template import TEMPLATE_DIR

SPRITE_ASSET_DIR = TEMPLATE_DIR / "sprite"

def _make_asset_id(sprite_name: str) -> str:
    """
    Derive a stable asset ID from the sprite name so the same model always
    produces the same ID, but different models don't conflict.
    Returns a 32-char hex string matching Scratch's md5 asset ID format.
    """
    return hashlib.md5(sprite_name.encode()).hexdigest()


def finalize_sprite(data: dict, output_path: str, sprite_name: str = "cattorch") -> Path:
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
    Path to the written .sprite3 file.
    """
    output_path = Path(output_path)
    sprite = copy.deepcopy(data)

    # Derive a unique asset ID for this sprite's costume
    asset_id = _make_asset_id(sprite_name)
    md5ext = f"{asset_id}.svg"

    # Find the source SVG in the template dir
    source_svg_candidates = list(SPRITE_ASSET_DIR.glob("*.svg"))
    if not source_svg_candidates:
        raise FileNotFoundError(
            f"No SVG costume found in {SPRITE_ASSET_DIR}. "
            "Add a .svg file there to use as the sprite costume."
        )
    source_svg = source_svg_candidates[0]

    # Add Scratch sprite metadata
    sprite.update({
        "name": sprite_name,
        "comments": {},
        "currentCostume": 0,
        "costumes": [
            {
                "name": "cat",
                "bitmapResolution": 1,
                "dataFormat": "svg",
                "assetId": asset_id,
                "md5ext": md5ext,
                "rotationCenterX": 48,
                "rotationCenterY": 50,
            }
        ],
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

    # Write into a temp dir then zip
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write sprite.json
        sprite_json_path = tmpdir / "sprite.json"
        with open(sprite_json_path, "w") as f:
            json.dump(sprite, f)

        # Copy SVG with the new asset ID filename
        svg_dest = tmpdir / md5ext
        shutil.copy(source_svg, svg_dest)

        # Zip both into the output .sprite3
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(sprite_json_path, "sprite.json")
            zf.write(svg_dest, md5ext)

    print(f"Written: {output_path} (asset_id: {asset_id})")
    return output_path