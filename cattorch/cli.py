"""The ``cattorch`` command line.

``cattorch export MODEL`` exports a model and checks it against PyTorch;
``cattorch generate`` exports a text generator (a recurrent one for a Hugging
Face Mamba, Mamba-2 or RWKV model), with its tokenizer and a demo project when
it can. ``MODEL`` is ``module:name`` or ``path/to/file.py:name``
(a module, a class to construct without arguments, or a function returning a
module), or a ``.pt`` file saved with ``torch.save(model)``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import pickle
import re
import sys
from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn

from cattorch import __version__
from cattorch.errors import CattorchError
from cattorch.util.scratch.archive import dumps


# Exit statuses, documented in docs/projects-and-cli.md.
EXIT_OK = 0
EXIT_EXPORT_FAILED = 1
EXIT_USAGE = 2
EXIT_VERIFY_FAILED = 3


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
            try:
                module = importlib.import_module(location)
            except ModuleNotFoundError as error:
                # Only the named module; a missing import inside it is the
                # module's own problem, best shown with its traceback.
                if error.name is None or not (
                    location == error.name or location.startswith(f"{error.name}.")
                ):
                    raise
                raise CommandError(
                    f"no module named {error.name!r}; run cattorch from the directory that "
                    "contains it, put it on PYTHONPATH, or use file.py:name"
                ) from None
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
        if not path.is_file():
            raise CommandError(f"{spec} is not a file")
        value = _load_pickled_module(path)
        name = path.stem
    if not isinstance(value, nn.Module):
        raise CommandError(f"{spec} is a {type(value).__name__}, not an nn.Module")
    return value.eval(), name.lower()


def _load_pickled_module(path: Path):
    """Unpickle a ``torch.save(model)`` file, explaining a missing class."""
    # Pickle stores the model's class by module and name, so the module must
    # be importable. Look beside the file and in the working directory too,
    # as ``python`` run there would; appended, so they shadow nothing.
    for directory in (str(Path.cwd()), str(path.resolve().parent)):
        if directory not in sys.path:
            sys.path.append(directory)
    try:
        # A whole pickled module can only be loaded by unpickling it.
        return torch.load(path, weights_only=False, map_location="cpu")
    except (ModuleNotFoundError, AttributeError) as error:
        if isinstance(error, AttributeError) and "Can't get attribute" not in str(error):
            raise
        defined_in_main = "__main__" in str(error)
        raise CommandError(
            f"can't load {path}: {error}. torch.save(model) stores the model's class by "
            "module and name, so that class must be importable when the file is loaded: "
            "run cattorch where its module is, or put the module on PYTHONPATH. "
            + (
                "This class was defined in the script that saved the model; move it to a "
                "module and save the model again, or "
                if defined_in_main else "Or "
            )
            + "export it with file.py:name, naming a function that builds the model"
        ) from None
    except (pickle.UnpicklingError, EOFError, RuntimeError) as error:
        raise CommandError(
            f"can't load {path} as a model saved with torch.save(model): {error}"
        ) from None


def _shape(text: str) -> tuple[int, ...]:
    message = f"{text!r} is not a shape such as 1,28,28"
    try:
        shape = tuple(int(part) for part in re.split(r"[,x]", text))
    except ValueError:
        raise argparse.ArgumentTypeError(message) from None
    if any(extent < 1 for extent in shape):
        raise argparse.ArgumentTypeError(message)
    return shape


def _positive(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive whole number")
    return value


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
    from cattorch.markers import has_entrypoints, marked_example_inputs

    if has_entrypoints(model) or marked_example_inputs(model) is not None:
        return None
    raise CommandError(
        "give the model's inputs with --shape (floats) or --tokens (token IDs), "
        "or mark it with @cattorch.example_inputs"
    )


def _calibration(path: str | None):
    """``--calibration FILE``: a ``torch.save``'d list of calibration items."""
    if path is None:
        return None
    try:
        items = torch.load(path, weights_only=True, map_location="cpu")
    except (OSError, pickle.UnpicklingError, RuntimeError, EOFError) as error:
        raise CommandError(f"can't load --calibration {path}: {error}") from None
    if isinstance(items, torch.Tensor) or not isinstance(items, (list, tuple)):
        raise CommandError(
            f"--calibration {path} must hold a list saved with torch.save, such as a list "
            "of input tensors (export) or of [1, length] token tensors (generate)"
        )
    return list(items)


def _quantization(args):
    """The ``QuantizationConfig`` the quantization flags ask for, or ``None``.

    The flags mirror ``QuantizationConfig``'s fields; per-weight settings
    belong here too.
    """
    from cattorch import QuantizationConfig

    if args.quantize is None:
        for flag, value in (("--quantize-method", args.quantize_method),
                            ("--calibration", args.calibration)):
            if value is not None:
                raise CommandError(f"{flag} needs --quantize")
        return None
    method = args.quantize_method or "symmetric"
    if method == "gptq" and args.calibration is None:
        raise CommandError("--quantize-method gptq needs --calibration")
    if method != "gptq" and args.calibration is not None:
        raise CommandError("--calibration is only used by --quantize-method gptq")
    return QuantizationConfig(bits=args.quantize, method=method)


def _export_options(args) -> dict:
    """The ``transpile`` options the common flags ask for."""
    from cattorch import StorageConfig

    options: dict = {
        "optimization": args.optimization,
        "storage": StorageConfig(precision=args.precision),
        "quantization": _quantization(args),
        "calibration": _calibration(args.calibration),
        "sig_figs": args.sig_figs,
    }
    if args.name is not None:
        options["name"] = args.name
    return options


def _summary(result) -> dict:
    """A result's fields, by their Python names, for the JSON summary."""
    fields = [
        "path", "sprite_name", "archive_bytes", "expanded_json_bytes", "block_count",
        "list_count", "sharded_lists", "warnings", "max_context", "vocabulary_size", "top_k",
        "tokenizer_type", "token_count", "fallback_operations", "approximate_operations",
    ]
    report = {}
    for field in fields:
        if hasattr(result, field):
            value = getattr(result, field)
            report[field] = str(value) if isinstance(value, Path) else value
    return report


class _Outputs(NamedTuple):
    sprite: Path
    tokenizer: Path
    project: Path


def _outputs(output: str | None, name: str) -> _Outputs:
    """Every file a command writes, from ``-o`` or the model's name.

    ``-o`` may be a bare path or end in ``.sprite3`` or ``.sb3``; other dots
    stay in the name, so ``-o net.v2`` writes ``net.v2.sprite3``.
    """
    path = Path(output or name)
    if path.suffix.lower() in {".sprite3", ".sb3"}:
        path = path.with_suffix("")
    if not path.name:
        raise CommandError(f"-o {output!r} doesn't name a file")
    return _Outputs(
        sprite=path.with_name(f"{path.name}.sprite3"),
        tokenizer=path.with_name(f"{path.name}_tokenizer.sprite3"),
        project=path.with_name(f"{path.name}.sb3"),
    )


def _export(args) -> int:
    from cattorch.transpiler import transpile
    from cattorch.verify import verify
    from cattorch.project import build_project

    options = _export_options(args)
    model, name = load_model(args.model)
    example = _example_inputs(args, model)
    dynamic: dict[int, dict[int, tuple[int, int]]] = {}
    for index, dim, low, high in args.dynamic:
        if dim in dynamic.get(index, {}):
            raise CommandError(f"--dynamic gives input {index}'s dimension {dim} twice")
        dynamic.setdefault(index, {})[dim] = (low, high)
    outputs = _outputs(args.output, name)
    result = transpile(model, example, outputs.sprite, dynamic=dynamic or None, **options)
    report = _summary(result)
    passed = True
    if not args.no_verify:
        # PyTorch runs with the values the sprite stores, so lossy storage
        # is checked tightly; approximate kernels need a looser tolerance.
        quantization = options["quantization"]
        # GPTQ can't be reproduced, so it is checked against the float model.
        gptq = quantization is not None and quantization.method == "gptq"
        stored: dict = {} if gptq else dict(
            quantization=quantization, storage=options["storage"], sig_figs=options["sig_figs"],
        )
        if args.optimization == "fast" or gptq:
            tolerance = {"atol": 0.05, "rtol": 0.05}
        elif quantization is not None or args.precision != "float32" or args.sig_figs:
            # Long sums over stored values accumulate float32 rounding.
            tolerance = {"atol": 1e-3, "rtol": 1e-4}
        else:
            tolerance = {"atol": 1e-4, "rtol": 1e-5}
        check = verify(model, example, result, **tolerance, **stored)
        passed = check.passed
        report["verify"] = {"passed": check.passed}
        if hasattr(check, "max_abs_error"):
            report["verify"]["max_abs_error"] = check.max_abs_error
    if args.project:
        labels = args.labels.split(",") if args.labels else None
        project = build_project(
            result, outputs.project, example_inputs=example or (), labels=labels,
        )
        report["project"] = {"path": str(project.path)}
    print(dumps(report, indent=2))
    return EXIT_OK if passed else EXIT_VERIFY_FAILED


def _generate(args) -> int:
    from cattorch import (
        GenerationProgram, RecurrentGenerationProgram, transpile, transpile_tokenizer,
    )
    from cattorch.huggingface_recurrent import recurrent_family
    from cattorch.project import build_project

    options = _export_options(args)
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
    outputs = _outputs(args.output, name)
    program: GenerationProgram | RecurrentGenerationProgram
    if recurrent_family(model) is not None:
        # Without --max-context a recurrent model takes any number of tokens.
        program = RecurrentGenerationProgram(max_context=args.max_context, top_k=args.top_k)
    else:
        program = GenerationProgram(
            64 if args.max_context is None else args.max_context, top_k=args.top_k,
        )
    result = transpile(model, program, outputs.sprite, **options)
    report = _summary(result)
    tokenizer_result = None
    tokenizer_error = None
    if tokenizer is not None:
        try:
            tokenizer_result = transpile_tokenizer(tokenizer, outputs.tokenizer, name="tokenizer")
            report["tokenizer"] = _summary(tokenizer_result)
        except (ValueError, TypeError) as error:
            tokenizer_error = str(error)
            report["tokenizer"] = {"error": tokenizer_error}
    if args.project:
        if tokenizer_result is None:
            raise CommandError(
                "--project needs a tokenizer that exports to Scratch; "
                + (tokenizer_error or "pass --tokenizer")
            )
        eos = args.eos_token_id
        if eos is None and tokenizer is not None:
            eos = getattr(tokenizer, "eos_token_id", None)
        project = build_project(
            result, outputs.project, tokenizer=tokenizer_result,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            eos_token_id=eos,
        )
        report["project"] = {"path": str(project.path)}
    print(dumps(report, indent=2))
    return EXIT_OK


def _analyze(args) -> int:
    import dataclasses
    import zipfile

    from cattorch.benchmark import analyze_benchmark

    if not Path(args.project).is_file():
        raise CommandError(f"no such file: {args.project}")
    try:
        analysis = analyze_benchmark(args.project)
    except (zipfile.BadZipFile, KeyError, StopIteration):
        raise CommandError(
            f"{args.project} is not a cattorch benchmark project (.sb3) saved after running"
        ) from None
    print(dumps(dataclasses.asdict(analysis), indent=2))
    return EXIT_OK


class _HelpFormatter(argparse.HelpFormatter):
    """Shows each option's default, unless its help already says it."""

    def _get_help_string(self, action):
        help_text = action.help or ""
        if (
            action.option_strings and "default" not in help_text
            and action.default not in (None, False, [], argparse.SUPPRESS)
        ):
            help_text += " (default: %(default)s)"
        return help_text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cattorch", description="Export PyTorch models to Scratch.",
        formatter_class=_HelpFormatter,
        epilog=(
            f"exit status: {EXIT_OK} success, {EXIT_EXPORT_FAILED} the export failed, "
            f"{EXIT_USAGE} invalid arguments, {EXIT_VERIFY_FAILED} the sprite didn't "
            "match PyTorch"
        ),
    )
    parser.add_argument("--version", action="version", version=f"cattorch {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    def options(command):
        command.add_argument(
            "-o", "--output",
            help="output path, with or without .sprite3 or .sb3; the sprite, tokenizer "
            "(NAME_tokenizer.sprite3) and project (NAME.sb3) are named after it "
            "(default: the model's name, lowercased)",
        )
        command.add_argument(
            "--name", help="the sprite's name in Scratch (default: the output file's name)",
        )
        command.add_argument(
            "--optimization", choices=("exact", "fast"), default="exact",
            help="fast uses approximate kernels",
        )
        command.add_argument(
            "--precision", choices=("float32", "float16"), default="float32",
            help="how stored weights are kept: StorageConfig(precision=...)",
        )
        command.add_argument(
            "--quantize", type=int, choices=(4, 6, 8), metavar="BITS",
            help="quantize matrix weights to 4, 6 or 8 bits: QuantizationConfig(bits=...) "
            "(default: no quantization)",
        )
        command.add_argument(
            "--quantize-method", choices=("symmetric", "gptq"),
            help="how --quantize quantizes; gptq needs --calibration (default: symmetric)",
        )
        command.add_argument(
            "--calibration", metavar="FILE",
            help="GPTQ calibration data: a list saved with torch.save, of input tensors "
            "(export) or [1, length] token tensors (generate) (default: none)",
        )
        command.add_argument(
            "--sig-figs", type=_positive, metavar="N",
            help="round stored values to N significant figures (default: no rounding)",
        )
        command.add_argument(
            "--project", action="store_true",
            help="also write a runnable .sb3 project around the sprite",
        )

    export = commands.add_parser(
        "export", help="export a model as a sprite and verify it", formatter_class=_HelpFormatter,
    )
    export.add_argument("model", help="module:name, file.py:name, or a torch.save'd .pt file")
    export.add_argument(
        "--shape", type=_shape, action="append", default=[],
        help="a random float input of this shape, such as 1,28,28; repeat for more inputs "
        "(default: the model's @example_inputs)",
    )
    export.add_argument(
        "--tokens", type=_shape, action="append", default=[],
        help="a token-ID input of this shape, after the --shape inputs (default: none)",
    )
    export.add_argument(
        "--vocab", type=_positive, default=2,
        help="--tokens IDs are random, below this",
    )
    export.add_argument(
        "--dynamic", type=_dynamic, action="append", default=[],
        help="let an input's length vary: INPUT[.DIM]=MIN:MAX, such as 0=1:32, like "
        "transpile(dynamic=...) (default: fixed lengths)",
    )
    export.add_argument(
        "--labels", help="comma-separated output labels for the project (default: none)",
    )
    export.add_argument(
        "--no-verify", action="store_true", help="skip the check against PyTorch",
    )
    options(export)
    export.set_defaults(run=_export)

    generate = commands.add_parser(
        "generate", help="export a text generator", formatter_class=_HelpFormatter,
    )
    generate.add_argument(
        "model", nargs="?",
        help="a forward(tokens) -> logits model, or a Hugging Face Mamba, Mamba-2 or RWKV model",
    )
    generate.add_argument(
        "--hf", metavar="REPO",
        help="a Hugging Face model name or directory, instead of MODEL (default: none)",
    )
    generate.add_argument(
        "--tokenizer",
        help="a Hugging Face tokenizer name or directory (default: --hf's tokenizer)",
    )
    generate.add_argument(
        "--max-context", type=_positive, metavar="N",
        help="the most tokens the generator reads between resets (default: 64; "
        "unlimited for a Mamba, Mamba-2 or RWKV model)",
    )
    generate.add_argument(
        "--top-k", type=_positive, metavar="K",
        help="also list the K largest logits and their token IDs, 1 to 64, in the "
        "sprite's top k lists (default: none)",
    )
    generate.add_argument(
        "--max-new-tokens", type=_positive, default=40, metavar="N",
        help="the most tokens the project's demo adds to a prompt",
    )
    generate.add_argument(
        "--temperature", type=float, default=0.8,
        help="the demo's starting sampling temperature; 0 always picks the likeliest token",
    )
    generate.add_argument(
        "--eos-token-id", type=int, metavar="ID",
        help="the token ID that ends a continuation (default: the tokenizer's end token)",
    )
    options(generate)
    generate.set_defaults(run=_generate)

    analyze = commands.add_parser(
        "analyze-benchmark", help="print a saved benchmark project's results as JSON",
        formatter_class=_HelpFormatter,
    )
    analyze.add_argument(
        "project", help="a benchmark .sb3 saved from Scratch after it ran",
    )
    analyze.set_defaults(run=_analyze)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line; return its exit status.

    ``EXIT_USAGE`` (2) is invalid arguments, from argparse or from cattorch's
    own checks of them (``ValueError`` and ``TypeError``, which cattorch
    raises for invalid arguments); ``EXIT_EXPORT_FAILED`` (1) an export that
    failed (a ``CattorchError``, or a file that can't be read or written);
    ``EXIT_VERIFY_FAILED`` (3) a sprite that didn't match PyTorch.
    """
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit:
        return int(exit.code or 0)
    try:
        return args.run(args)
    except CattorchError as error:
        print(f"cattorch: error: {error}", file=sys.stderr)
        return EXIT_EXPORT_FAILED
    except (CommandError, ValueError, TypeError) as error:
        print(f"cattorch: error: {error}", file=sys.stderr)
        return EXIT_USAGE
    # OSError covers an unreadable file, an unwritable output, and a Hugging
    # Face model or tokenizer that can't be found.
    except OSError as error:
        print(f"cattorch: error: {error}", file=sys.stderr)
        return EXIT_EXPORT_FAILED


if __name__ == "__main__":
    sys.exit(main())
