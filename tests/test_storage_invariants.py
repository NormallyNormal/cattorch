"""What a sprite stores, what its quantization report says, and what verify
compares it with, checked against each other on a matrix of small exports.

The sprite's weight payloads are decoded here, independently of cattorch's
decoder: the Base92 groups, float16 and float32, int8, int6 and the int4
Huffman stream, and the four-row interleave, read from the blocks of the
``unpack weights`` procedures. Then, for every export:

* the decoded lists are the lists the emulator holds after ``cattorch init``;
* the report has one entry per parameter for its integer storage (and a
  ``name[float16]`` entry for a float16 copy), at the precision stored;
* the report's values per precision are the values the payloads hold;
* ``verify``'s reference, which rounds as storage stores, matches the sprite.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from cattorch import (
    EntryPoint, ExportProgram, GenerationProgram, Input, Output, ProgramCall,
    QuantizationConfig, RecurrentGenerationProgram, State, transpile, verify,
)
from sprite_helpers import SpriteEmulator, load_sprite

# ── An independent payload decoder ───────────────────────────────────────────

_ALPHABET = "".join(chr(code) for code in range(33, 127) if chr(code) not in {'"', "\\"})
_DIGIT = {character: index for index, character in enumerate(_ALPHABET)}


def _base92(text: str) -> bytes:
    padding, body = _DIGIT[text[0]], text[1:]
    assert len(body) % 16 == 0
    out = bytearray()
    for start in range(0, len(body), 16):
        high = low = 0
        for character in body[start:start + 8]:
            high = high * 92 + _DIGIT[character]
        for character in body[start + 8:start + 16]:
            low = low * 92 + _DIGIT[character]
        out += ((high << 52) | low).to_bytes(13, "big")
    return bytes(out[:-padding] if padding else out)


def _value(blocks, argument, variables):
    value = argument[1]
    if isinstance(value, list):
        if value[0] in (4, 5, 6, 7, 8, 10):
            return float(value[1]) if value[1] != "" else 0.0
        if value[0] == 12:
            return variables[value[1]]
    if isinstance(value, str) and blocks[value]["opcode"] == "data_lengthoflist":
        return ("length", blocks[value]["fields"]["LIST"][0])
    raise ValueError(argument)


def _flatten(blocks, start):
    found = []
    while start:
        block = blocks[start]
        if block["opcode"] == "control_if":
            substack = block["inputs"].get("SUBSTACK")
            if substack and substack[1]:
                found.extend(_flatten(blocks, substack[1]))
        else:
            found.append(start)
        start = block["next"]
    return found


def _huffman(table, data, start, total):
    codes, bits, count, position = [], 0, 0, start
    for _ in range(total):
        while count < 8:
            bits = (bits << 8) | (data[position] if position < len(data) else 0)
            position += 1
            count += 8
        entry = table[bits >> (count - 8)]
        codes.append(entry // 16)
        count -= entry % 16
        bits &= (1 << count) - 1
    return codes


def _definitions(blocks, text):
    return [
        block["parent"] for block in blocks.values()
        if block["opcode"] == "procedures_prototype" and text in block["mutation"]["proccode"]
    ]


def decode_payloads(sprite):
    """``(list name -> values, list name -> (precision, payload identity))``."""
    blocks = sprite["blocks"]
    variables = {entry[0]: entry[1] for entry in sprite.get("variables", {}).values()}
    operations = [
        operation for definition in _definitions(blocks, "unpack weights")
        for operation in _flatten(blocks, blocks[definition]["next"])
    ]
    state: dict = {}
    raw, decoded, scales, table = b"", [], [], None
    precision, identity = None, None
    lists, stored = {}, {}
    for operation in operations:
        block = blocks[operation]
        opcode = block["opcode"]
        if opcode == "data_setvariableto":
            name = block["fields"]["VARIABLE"][0].lstrip("_")
            value = _value(blocks, block["inputs"]["VALUE"], variables)
            state[name] = len(raw) if isinstance(value, tuple) else value
        elif opcode == "procedures_call":
            call = block["mutation"]["proccode"].lstrip("_")
            call = call[call.index("cattorch decode"):]
            if call.startswith("cattorch decode costume base92 bytes"):
                payload = state["cattorch active payload"]
                raw = _base92(payload)
            elif call in ("cattorch decode float16", "cattorch decode float32"):
                precision = call.split()[-1]
                start = int(state["decode byte start"]) - 1
                length = int(state["decode byte length"])
                width = 4 if precision == "float32" else 2
                decoded = list(np.frombuffer(
                    raw[start:start + length // width * width],
                    dtype=">f4" if precision == "float32" else ">f2",
                ).astype(np.float64))
                identity = (hash(payload), start, length)
            elif call.split()[2] in ("int8", "int6", "int4"):
                precision = call.split()[2]
                start = int(state["decode byte start"]) - 1
                total = int(state["decode total values"])
                group = int(state["decode group size"])
                scale_start = int(state["decode scale start"]) - 1
                if precision == "int8":
                    codes = [byte - 127 for byte in raw[start:start + total]]
                elif precision == "int6":
                    codes, position = [], start
                    while len(codes) < total:
                        a, b, c = (
                            raw[position + offset] if position + offset < len(raw) else 0
                            for offset in range(3)
                        )
                        codes.extend(code - 31 for code in (
                            a >> 2, ((a & 3) << 4) | (b >> 4), ((b & 15) << 2) | (c >> 6), c & 63,
                        ))
                        position += 3
                    codes = codes[:total]
                else:
                    codes = [code - 7 for code in _huffman(table, raw, start, total)]
                decoded = [
                    codes[index] * scales[scale_start + index // group] for index in range(total)
                ]
                identity = (hash(payload), start, total, scale_start, len(scales))
            else:
                raise AssertionError(f"unknown decoder {call}")
        elif opcode == "data_deletealloflist":
            name = block["fields"]["LIST"][0].lstrip("_")
            if name == "cattorch decoded scales":
                scales = []
            elif name == "cattorch huffman table":
                table = None
            elif name not in ("cattorch decoded bytes", "cattorch decoded values"):
                lists[block["fields"]["LIST"][0]] = np.array(decoded, dtype=np.float64)
                stored[block["fields"]["LIST"][0]] = (precision, identity)
        elif opcode == "control_for_each":
            target = blocks[block["inputs"]["SUBSTACK"][1]]["fields"]["LIST"][0].lstrip("_")
            if target == "cattorch decoded scales":
                scales.extend(decoded)
            elif target == "cattorch huffman table":
                table = list(raw[:256])
    for definition in _definitions(blocks, "interleave grouped weights"):
        segments: list = []
        for operation in _flatten(blocks, blocks[definition]["next"]):
            block = blocks[operation]
            opcode = block["opcode"]
            if opcode == "data_setvariableto" and (
                block["fields"]["VARIABLE"][0].lstrip("_") == "interleave group start"
            ):
                segments.append([int(_value(blocks, block["inputs"]["VALUE"], variables)) - 1])
            elif opcode == "control_repeat" and segments and len(segments[-1]) == 1:
                times = int(_value(blocks, block["inputs"]["TIMES"], variables))
                inner_block = blocks[blocks[block["inputs"]["SUBSTACK"][1]]["next"]]
                inner = int(_value(blocks, inner_block["inputs"]["TIMES"], variables))
                segments[-1] += [times * 4, inner]
            elif opcode == "data_deletealloflist":
                name = block["fields"]["LIST"][0]
                if "interleave buffer" in name:
                    continue
                values = lists[name].copy()
                for start, rows, inner in segments:
                    span = values[start:start + rows * inner]
                    values[start:start + rows * inner] = (
                        span.reshape(rows // 4, 4, inner).transpose(0, 2, 1).reshape(-1)
                    )
                lists[name] = values
                segments = []
    return lists, stored


# ── The exports ──────────────────────────────────────────────────────────────

V, D, H = 64, 32, 64


class _Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(H, D) * 0.2)
        self.w2 = nn.Parameter(torch.randn(H, D) * 0.2)
        self.w3 = nn.Parameter(torch.randn(H, D) * 0.2)

    def forward(self, value):
        hidden = F.linear(value, self.w1 * torch.tril(torch.ones(H, D)))
        hidden = hidden + F.linear(value, self.w3) + F.linear(value, self.w3.clone())
        hidden = torch.tanh(hidden + F.linear(value, self.w2))
        return torch.tanh(hidden @ self.w2)


class _Mixed(nn.Module):
    """A tied table (gathered, and read as the head through a copy), a weight
    computed with a mask built in forward, a weight read in two layouts, and
    one read directly and through a copy. No biases or buffers: every
    floating list the sprite stores is one of these weights."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.core = _Core()

    def head(self, value):
        return value @ self.emb.weight.t().contiguous()

    def encode(self, tokens):
        value = self.emb(tokens)
        # Causal attention with no weights of its own, for generation.
        value = value + F.scaled_dot_product_attention(
            value[:, None], value[:, None], value[:, None], is_causal=True,
        )[:, 0]
        return self.core(value)

    def forward(self, tokens):
        return self.head(self.encode(tokens))

    def states(self):
        return (State("hidden", torch.zeros(1, 1, D)),)

    def step(self, token, hidden):
        mixed = self.core(self.emb(token) + hidden)
        return self.head(mixed), mixed


def _program():
    return ExportProgram(entrypoints=(
        EntryPoint(
            "encode", method="encode",
            arguments=(Input("tokens", torch.zeros(1, 4, dtype=torch.long)),),
            returns=(Output("mixed"),),
        ),
        EntryPoint(
            "head", method="head", arguments=(Input("mixed", torch.zeros(1, 4, D)),),
            returns=(Output("logits"),),
        ),
    ))


def _export(kind, model, path, config):
    tokens = torch.randint(0, V, (1, 4), generator=torch.Generator().manual_seed(1))
    if kind == "tensor":
        result = transpile(model, tokens, path, quantization=config)
        return result, lambda: verify(model, tokens, result, quantization=config)
    if kind == "program":
        program = _program()
        result = transpile(model, program, path, quantization=config)
        calls = (
            ProgramCall("encode", (tokens,)),
            ProgramCall("head", (torch.randn(1, 4, D, generator=torch.Generator().manual_seed(2)),)),
        )
        return result, lambda: verify(model, program, result, quantization=config, calls=calls)
    program = (
        GenerationProgram(max_context=4) if kind == "generation"
        else RecurrentGenerationProgram(model.states(), max_context=4)
    )
    result = transpile(model, program, path, quantization=config)
    return result, lambda: verify(model, program, result, quantization=config)


_PARAMETERS = {
    "emb.weight": "emb.weight", "core.w1": "weight computed from core.w1",
    "core.w2": "core.w2", "core.w3": "core.w3",
}


@pytest.mark.parametrize("limit", [None, 1000], ids=["whole", "sharded"])
@pytest.mark.parametrize("bits", [4, 6, 8])
@pytest.mark.parametrize("kind", ["tensor", "program", "generation", "recurrent"])
def test_storage_report_and_reference_agree(tmp_path, small_list_limit, kind, bits, limit):
    if limit is not None:
        small_list_limit(limit)
    torch.manual_seed(bits)
    model = _Mixed().eval()
    config = QuantizationConfig(bits=bits, min_quantized_values=512)
    result, check = _export(kind, model, tmp_path / "sprite", config)

    # The decoded payloads are what the sprite computes with.
    sprite = load_sprite(result.path)
    lists, stored = decode_payloads(sprite)
    assert lists
    emulator = SpriteEmulator(sprite)
    emulator.run_procedure("cattorch init")
    names = {entry[0]: identifier for identifier, entry in sprite["lists"].items()}
    for name, values in lists.items():
        found = np.array([float(item) for item in emulator.lists._by_id[names[name]]])
        np.testing.assert_array_equal(found, values, err_msg=name)

    # One entry per parameter for its integer storage, at the bits stored;
    # a float16 copy has its own entry.
    report = result.quantization
    entries: dict[str, list] = {}
    for item in report.tensors:
        entries.setdefault(item.name, []).append(item)
    assert all(len(found) == 1 for found in entries.values()), report
    head_bits = 8 if kind in ("generation", "recurrent") and bits < 8 else bits
    for parameter, name in _PARAMETERS.items():
        (item,) = entries[name]
        assert (item.method, item.bits, item.values) == (
            "symmetric", head_bits if parameter == "emb.weight" else bits, H * D,
        ), item
    assert set(entries) <= {*_PARAMETERS.values(), "core.w2[float16]"}, report

    # The report's values per precision are the values the payloads hold.
    # Each stored list is counted once however many lists or payloads hold
    # it: a program's entry points each unpack the weights they read.
    contents: dict[str, set] = {}
    held: dict[str, int] = {}
    for name, (precision, _) in stored.items():
        values = lists[name].tobytes()
        if values not in contents.setdefault(precision, set()):
            contents[precision].add(values)
            held[precision] = held.get(precision, 0) + len(lists[name])
    reported: dict[str, int] = {}
    for item in report.tensors:
        precision = "float16" if item.method == "float16" else f"int{item.bits}"
        reported[precision] = reported.get(precision, 0) + item.values
    assert reported == held, (reported, held, report)

    # verify's reference holds what the sprite stores, for every use.
    outcome = check()
    assert outcome.passed, outcome
    worst = max(
        (
            value.max_abs_error
            for value in getattr(outcome, "outputs", None) or [
                output for call in getattr(outcome, "calls", ()) for output in call.outputs
            ] or [outcome]
        ),
        default=0.0,
    )
    assert math.isfinite(worst) and worst < 1e-4
