"""
prepare_template.py
-------------------
Single-pass preparation of a Scratch sprite template JSON:
  1. Renames all opaque IDs to human-readable cattorch_ prefixed names
  2. Empties all list contents
  3. Converts numeric string literals in block inputs to int or float

Works with both full sprite JSON and blocks-only JSON.

Usage:
  python3 prepare_template.py input.json [output.json]
"""

import json
import sys
import copy


# ---------------------------------------------------------------------------
# Step 1: Rename IDs
# ---------------------------------------------------------------------------

def is_blocks_only(data: dict) -> bool:
    if "isStage" in data:
        return False
    first = next(iter(data.values()), None)
    return isinstance(first, dict) and "opcode" in first


def build_id_map(data: dict, blocks_only: bool) -> dict[str, str]:
    id_map = {}
    block_counter = 1
    var_counter = 1
    list_counter = 1
    comment_counter = 1

    if blocks_only:
        for sid in data:
            id_map[sid] = f"cattorch_block_{block_counter}"
            block_counter += 1
    else:
        for sid in data.get("variables", {}):
            id_map[sid] = f"cattorch_var_{var_counter}"
            var_counter += 1
        for sid in data.get("lists", {}):
            id_map[sid] = f"cattorch_list_{list_counter}"
            list_counter += 1
        for sid in data.get("blocks", {}):
            id_map[sid] = f"cattorch_block_{block_counter}"
            block_counter += 1
        for sid in data.get("comments", {}):
            id_map[sid] = f"cattorch_comment_{comment_counter}"
            comment_counter += 1

    return id_map


def rename_ids(data: dict) -> dict:
    blocks_only = is_blocks_only(data)
    id_map = build_id_map(data, blocks_only)
    raw = json.dumps(data)
    for old, new in sorted(id_map.items(), key=lambda x: -len(x[0])):
        raw = raw.replace(old, new)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Step 2: Empty lists
# ---------------------------------------------------------------------------

def empty_lists(data: dict) -> dict:
    for entry in data.get("lists", {}).values():
        entry[1] = []
    return data


# ---------------------------------------------------------------------------
# Step 3: Convert numeric string literals
# ---------------------------------------------------------------------------

def try_numeric(value):
    if not isinstance(value, str) or value == "":
        return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def process_input_value(val):
    if isinstance(val, list):
        result = [val[0]]  # type code — always keep as-is
        for i, item in enumerate(val[1:], 1):
            if isinstance(item, list):
                result.append(process_input_value(item))
            elif isinstance(item, str) and i == 1:
                result.append(try_numeric(item))
            else:
                result.append(item)
        return result
    return val


def process_inputs(inputs: dict) -> dict:
    result = {}
    for key, inp in inputs.items():
        if not isinstance(inp, list):
            result[key] = inp
            continue
        new_inp = [inp[0]]
        for item in inp[1:]:
            if isinstance(item, list):
                new_inp.append(process_input_value(item))
            else:
                new_inp.append(item)
        result[key] = new_inp
    return result


def convert_numerics(data: dict) -> dict:
    for block in data.get("blocks", {}).values():
        if "inputs" in block:
            block["inputs"] = process_inputs(block["inputs"])
    return data


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def prepare(data: dict) -> dict:
    data = copy.deepcopy(data)
    data = rename_ids(data)
    data = empty_lists(data)
    data = convert_numerics(data)
    return data


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "../matmul/template.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".json", "_prepared.json")

    with open(src) as f:
        data = json.load(f)

    result = prepare(data)

    with open(dst, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Prepared {src} -> {dst}")