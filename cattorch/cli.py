"""The ``cattorch`` command line.

``cattorch export MODEL`` exports a model and checks it against PyTorch;
``cattorch generate`` exports a text generator, with its tokenizer and a demo
project when it can. ``MODEL`` is ``module:name`` or ``path/to/file.py:name``
(a module, a class to construct without arguments, or a function returning a
module), or a ``.pt`` file saved with ``torch.save(model)``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import torch
from torch import nn

from cattorch.errors import UnsupportedModelError


class CommandError(Exception):
    """A problem with the command's arguments, reported without a traceback."""


def load_model(spec: str) -> tuple[nn.Module, str]:
    """Load ``MODEL`` and return it with a default output name."""
    if ":" in spec and not Path(spec).exists():
        location, _, attribute = spec.rpartition(":")
        if location.endswith(".py"):
            path = Path(location)
            if not path.exists():
                raise CommandError(f"no such file: {location}")
            module_spec = importlib.util.spec_from_file_location(path.stem, path)
            assert module_spec is not None and module_spec.loader is not None
            module = importlib.util.module_from_spec(module_spec)
            sys.path.insert(0, str(path.resolve().parent))
            sys.modules[path.stem] = module
            module_spec.loader.exec_module(module)
        else:
            sys.path.insert(0, str(Path.cwd()))
            module = importlib.import_module(location)
        try:
            value = getattr(module, attribute)
        except AttributeError:
            raise CommandError(f"{location} has no attribute {attribute!r}") from None
        if callable(value) and not isinstance(value, nn.Module):
            value = value()
        name = attribute
    else:
        path = Path(spec)
        if not path.exists():
            raise CommandError(f"no such file: {spec}; use module:name or file.py:name")
        # A whole pickled module can only be loaded by unpickling it.
        value = torch.load(path, weights_only=False, map_location="cpu")
        name = path.stem
    if not isinstance(value, nn.Module):
        raise CommandError(f"{spec} is a {type(value).__name__}, not an nn.Module")
    return value.eval(), name.lower()


def _shape(text: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part) for part in text.replace("x", ",").split(",") if part)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a shape such as 1,28,28") from None
    if not shape or any(extent < 1 for extent in shape):
        raise argparse.ArgumentTypeError(f"{text!r} is not a shape such as 1,28,28")
    return shape


def _dynamic(text: str) -> tuple[int, int, int, int]:
    match = re.fullmatch(r"(\d+)(?:\.(\d+))?=(\d+):(\d+)", text)
    if match is None:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not INPUT[.DIM]=MIN:MAX, such as 0=1:32 or 0.1=1:64"
        )
    index, dim, low, high = match.groups()
    return int(index), int(dim or 0), int(low), int(high)


def _example_inputs(args, model):
    inputs = [torch.randn(shape) for shape in args.shape]
    inputs += [torch.randint(0, args.vocab, shape) for shape in args.tokens]
    if inputs:
        return tuple(inputs)
    from cattorch.markers import example_inputs, has_entrypoints

    if has_entrypoints(model) or example_inputs(model) is not None:
        return None
    raise CommandError(
        "give the model's inputs with --shape (floats) or --tokens (token IDs), "
        "or mark it with @cattorch.example"
    )


def _storage(args):
    from cattorch import QuantizationConfig, StorageConfig

    if args.precision == "float16":
        return {"storage": StorageConfig(precision="float16")}
    if args.precision in {"int8", "int4"}:
        return {"quantization": QuantizationConfig(bits=int(args.precision[3:]))}
    return {}


def _summary(result, extra: dict) -> dict:
    report = {
        "sprite": str(result.path),
        "archive_bytes": result.archive_bytes,
        "blocks": result.block_count,
        **extra,
    }
    fallbacks = getattr(result, "fallback_operations", ())
    if fallbacks:
        report["fallback_operations"] = dict(fallbacks)
    if result.warnings:
        report["warnings"] = list(result.warnings)
    return report


def _project_path(output: Path) -> Path:
    """The project beside the sprite, which keeps any dots in its name."""
    if output.suffix.lower() == ".sprite3":
        return output.with_suffix(".sb3")
    return output.with_name(f"{output.name}.sb3")


def _export(args) -> int:
    from cattorch.transpiler import transpile
    from cattorch.verify import verify
    from cattorch.project import build_project

    model, name = load_model(args.model)
    example = _example_inputs(args, model)
    dynamic: dict[int, dict[int, tuple[int, int]]] = {}
    for index, dim, low, high in args.dynamic:
        dynamic.setdefault(index, {})[dim] = (low, high)
    output = Path(args.output or name)
    result = transpile(
        model, example, output, optimization="fast" if args.fast else "exact",
        dynamic=dynamic or None, **_storage(args),
    )
    extra: dict = {}
    if not args.no_verify:
        # Lossy weight storage and approximate kernels are checked against
        # what they can reach.
        tolerance = 1e-4 if args.precision == "float32" and not args.fast else 0.05
        report = verify(
            model, example, result, atol=tolerance, rtol=1e-5 if tolerance < 0.05 else 0.05,
        )
        extra["verified"] = report.passed
        if hasattr(report, "max_abs_error"):
            extra["max_abs_error"] = report.max_abs_error
    if args.project:
        labels = args.labels.split(",") if args.labels else None
        project = build_project(
            _project_path(output), result, example_inputs=example or (), labels=labels,
        )
        extra["project"] = str(project.path)
    print(json.dumps(_summary(result, extra), indent=2))
    return 0 if extra.get("verified", True) else 1


def _generate(args) -> int:
    from cattorch import GenerationProgram, transpile, transpile_tokenizer
    from cattorch.project import build_project

    tokenizer = None
    if args.hf:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise CommandError('--hf needs transformers: pip install "cattorch[tokenizers]"') from None
        model = AutoModelForCausalLM.from_pretrained(args.hf).eval()
        name = args.hf.rstrip("/").split("/")[-1].lower()
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.hf)
    elif args.model:
        model, name = load_model(args.model)
        if args.tokenizer:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        raise CommandError("give a MODEL or --hf REPO")
    output = Path(args.output or name)
    result = transpile(
        model, GenerationProgram(max_context=args.max_context, top_k=args.top_k), output,
        optimization="fast" if args.fast else "exact", **_storage(args),
    )
    extra: dict = {"max_context": args.max_context}
    tokenizer_result = None
    if tokenizer is not None:
        try:
            tokenizer_result = transpile_tokenizer(
                tokenizer, output.with_name(f"{output.name}_tokenizer"), name="tokenizer",
            )
            extra["tokenizer"] = str(tokenizer_result.path)
        except (ValueError, TypeError) as error:
            extra["tokenizer_error"] = str(error)
    if args.project:
        if tokenizer_result is None:
            raise CommandError(
                "--project needs a tokenizer that exports to Scratch; "
                + extra.get("tokenizer_error", "pass --tokenizer")
            )
        eos = args.eos
        if eos is None and tokenizer is not None:
            eos = getattr(tokenizer, "eos_token_id", None)
        project = build_project(
            _project_path(output), result, tokenizer=tokenizer_result,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            eos_token_id=eos,
        )
        extra["project"] = str(project.path)
    print(json.dumps(_summary(result, extra), indent=2))
    return 0


def _analyze(args) -> int:
    from cattorch.benchmark import analyze_benchmark

    print(json.dumps(analyze_benchmark(args.project), indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cattorch", description="Export PyTorch models to Scratch.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def options(command):
        command.add_argument("-o", "--output", help="output path, without an extension")
        command.add_argument("--fast", action="store_true", help="use approximate kernels")
        command.add_argument(
            "--precision", choices=("float32", "float16", "int8", "int4"), default="float32",
            help="weight storage; int8 and int4 quantize matrices",
        )
        command.add_argument(
            "--project", action="store_true",
            help="also write a runnable .sb3 project around the sprite",
        )

    export = commands.add_parser("export", help="export a model as a sprite and verify it")
    export.add_argument("model", help="module:name, file.py:name, or a torch.save'd .pt file")
    export.add_argument(
        "--shape", type=_shape, action="append", default=[],
        help="a random float input of this shape, such as 1,28,28; repeat for more inputs",
    )
    export.add_argument(
        "--tokens", type=_shape, action="append", default=[],
        help="a token-ID input of this shape, after the --shape inputs",
    )
    export.add_argument("--vocab", type=int, default=2, help="token IDs range below this")
    export.add_argument(
        "--dynamic", type=_dynamic, action="append", default=[],
        help="let an input's length vary: INPUT[.DIM]=MIN:MAX, such as 0=1:32",
    )
    export.add_argument("--labels", help="comma-separated output labels for the project")
    export.add_argument("--no-verify", action="store_true", help="skip the check against PyTorch")
    options(export)
    export.set_defaults(run=_export)

    generate = commands.add_parser("generate", help="export a text generator")
    generate.add_argument("model", nargs="?", help="a forward(tokens) -> logits model")
    generate.add_argument("--hf", metavar="REPO", help="a Hugging Face model name or directory")
    generate.add_argument("--tokenizer", help="a Hugging Face tokenizer name or directory")
    generate.add_argument("--max-context", type=int, default=64)
    generate.add_argument("--top-k", type=int)
    generate.add_argument("--max-new-tokens", type=int, default=40)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--eos", type=int, help="token ID that ends a continuation")
    options(generate)
    generate.set_defaults(run=_generate)

    analyze = commands.add_parser(
        "analyze-benchmark", help="print a saved benchmark project's timings",
    )
    analyze.add_argument("project")
    analyze.set_defaults(run=_analyze)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.run(args)
    except (CommandError, UnsupportedModelError, ValueError, TypeError) as error:
        print(f"cattorch: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
