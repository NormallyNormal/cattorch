"""Run PyTorch's OpInfo samples through cattorch export and the emulator.

Each OpInfo sample becomes a small module whose tensor arguments are Scratch
inputs. A case passes when ``transpile`` exports it and ``verify`` matches
eager PyTorch. Run this file to refresh the pass list and coverage page::

    python tests/opinfo_conformance.py --write

``tests/opinfo_passing.json`` records every passing case, and
``tests/test_opinfo.py`` checks that they all still pass, so the list can only
grow. ``docs/operator-coverage.md`` is generated from the same run.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import os
import re
import signal
import sys
import tempfile
import textwrap
import warnings
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils import _pytree as pytree

ROOT = Path(__file__).resolve().parent.parent
PASSING = Path(__file__).with_name("opinfo_passing.json")
COVERAGE = ROOT / "docs" / "operator-coverage.md"

# Samples tried per operator variant; OpInfo's later samples mostly repeat the
# first ones with other broadcasting or keyword combinations.
SAMPLES_PER_OPERATOR = 4
# A guard against hangs, not a speed limit: the slowest recorded case, a 3D
# transposed convolution, emulates for about 30 s locally and over 60 s on CI
# runners.
TIMEOUT_SECONDS = 180
# Bigger samples only test the emulator's patience, not another code path.
MAX_SAMPLE_ELEMENTS = 4096
ATOL, RTOL = 1e-4, 1e-4

# Families whose meaning cattorch doesn't model: gradients, sparse or complex
# layouts, linear algebra decompositions, random sampling, and memory layout.
_SKIPPED_PREFIXES = (
    "linalg.", "fft.", "special.", "sparse.", "_refs.", "nn.functional.ctc_loss",
    "masked.", "signal.", "_native_batch_norm_legit", "_segment_reduce",
)
_SKIPPED_NAMES = frozenset({
    "bernoulli", "multinomial", "normal", "uniform", "rand_like", "randn_like",
    "randint_like", "randint", "randn", "rand", "nn.functional.dropout",
    "nn.functional.dropout2d", "nn.functional.dropout3d",
    "nn.functional.alpha_dropout", "nn.functional.feature_alpha_dropout",
    "nn.functional.rrelu", "empty", "empty_like", "empty_strided", "new_empty",
    "new_empty_strided", "empty_permuted", "item", "tolist", "equal",
    "allclose", "isclose", "is_nonzero", "storage", "resize_", "resize_as_",
    "to_sparse", "nonzero", "nonzero_static", "unique", "unique_consecutive",
    "masked_select", "argwhere", "bincount", "histc", "histogram",
    "histogramdd", "_unsafe_masked_index", "_unsafe_masked_index_put_accumulate",
    "svd_lowrank", "pca_lowrank", "lu", "lu_solve", "lu_unpack", "cholesky",
    "cholesky_inverse", "cholesky_solve", "geqrf", "ormqr", "qr", "svd",
    "pinverse", "inverse", "det", "logdet", "slogdet", "matrix_exp",
    "triangular_solve", "symeig", "eig", "eigvals", "torch._scaled_mm",
    "jiterator_unary", "jiterator_binary", "jiterator_binary_return_by_ref",
    "jiterator_4inputs_with_extra_args", "jiterator_2inputs_2outputs",
    "view_as_complex", "view_as_real", "complex", "polar", "nn.functional.embedding_bag",
    "byte", "char", "short", "int", "long", "half", "bfloat16", "chalf",
    "cdouble", "cfloat", "double", "float", "bool", "cpu", "cuda", "to",
    "data_ptr", "nn.functional.multi_head_attention_forward",
    # PyTorch's decomposition returns the attention weights where the
    # operator returns its log-sum-exp.
    "torch.ops.aten._scaled_dot_product_flash_attention_for_cpu",
})


def torch_version() -> str:
    """Major and minor version; OpInfo's samples change between releases."""
    return ".".join(torch.__version__.split("+")[0].split(".")[:2])


def recorded():
    """``(torch version, passing case keys)`` from the pass list."""
    data = json.loads(PASSING.read_text())
    return data["torch"], data["passing"]


def version_mismatch() -> str | None:
    version, _ = recorded()
    if version != torch_version():
        return (
            f"the pass list was recorded with PyTorch {version}, not "
            f"{torch_version()}; refresh it with `python tests/opinfo_conformance.py --write`"
        )
    return None


def _operator_name(op) -> str:
    return op.name + (f".{op.variant_test_name}" if op.variant_test_name else "")


def _seed(name: str) -> int:
    return zlib.crc32(name.encode())


def operator_db():
    from torch.testing._internal.common_methods_invocations import op_db
    return [
        op for op in op_db
        if torch.float32 in op.supported_dtypes("cpu")
        and op.name not in _SKIPPED_NAMES
        and not op.name.startswith(_SKIPPED_PREFIXES)
    ]


def _samples(op):
    """The first samples of ``op``, reproducibly."""
    torch.manual_seed(_seed(_operator_name(op)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples = op.sample_inputs("cpu", torch.float32, requires_grad=False)
        result = []
        for sample in samples:
            result.append(sample)
            if len(result) == SAMPLES_PER_OPERATOR:
                break
    return result


class _Sample(torch.nn.Module):
    """Call ``function`` with a sample's tensors supplied as inputs."""

    def __init__(self, function, spec, constants, positions):
        super().__init__()
        self.function = function
        self.spec = spec
        self.constants = constants
        self.positions = positions

    def forward(self, *tensors):
        leaves = list(self.constants)
        for position, tensor in zip(self.positions, tensors):
            leaves[position] = tensor
        value, args, kwargs = pytree.tree_unflatten(leaves, self.spec)
        return self.function(value, *args, **kwargs)


def build(op, sample, *, static_integers=False):
    """A module and example inputs for one sample, or a reason it's skipped.

    Every tensor in the sample is a Scratch input, unless ``static_integers``
    fixes integer and boolean tensors, such as indices, in the model.
    """
    leaves, spec = pytree.tree_flatten((sample.input, sample.args, sample.kwargs))

    def runtime(leaf):
        return isinstance(leaf, torch.Tensor) and not (
            static_integers and not leaf.is_floating_point()
        )

    if not any(isinstance(leaf, torch.Tensor) for leaf in pytree.tree_leaves(sample.input)):
        return None, "input isn't a tensor"
    positions = [index for index, leaf in enumerate(leaves) if runtime(leaf)]
    if not positions:
        return None, "no floating-point tensor"
    tensors = tuple(leaves[index].detach().clone() for index in positions)
    if any(t.is_complex() or t.layout != torch.strided for t in tensors):
        return None, "complex or sparse tensor"
    if sum(t.numel() for t in tensors) > MAX_SAMPLE_ELEMENTS:
        return None, "sample too large"
    constants = [None if runtime(leaf) else leaf for leaf in leaves]
    return _Sample(op.op, spec, constants, positions).eval(), tensors


def _outputs(value):
    return [leaf for leaf in pytree.tree_leaves(value) if isinstance(leaf, torch.Tensor)]


def _deterministic(model, inputs) -> bool:
    first = _outputs(model(*inputs))
    second = _outputs(model(*inputs))
    return len(first) == len(second) and all(
        a.shape == b.shape and torch.equal(a.nan_to_num(), b.nan_to_num())
        for a, b in zip(first, second)
    )


class _Timeout(Exception):
    pass


@contextlib.contextmanager
def _deadline(seconds):
    if not hasattr(signal, "SIGALRM"):  # Windows has no alarm signal
        yield
        return

    def expire(*_):
        raise _Timeout()

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _first_line(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}".strip().splitlines()[0]
    return text[:300]


def run_case(op, sample, directory) -> tuple[str, str]:
    """``(outcome, detail)``: pass, static, mismatch, unsupported, error or skip.

    ``static`` means the sample fails with its integer tensors as inputs but
    passes with them fixed in the model.
    """
    outcome = _run_case(op, sample, directory)
    integers = any(
        isinstance(leaf, torch.Tensor) and not leaf.is_floating_point()
        for leaf in pytree.tree_leaves((sample.input, sample.args, sample.kwargs))
    )
    if outcome[0] in {"pass", "skip"} or not integers:
        return outcome
    if _run_case(op, sample, directory, static_integers=True)[0] == "pass":
        return "static", outcome[1]
    return outcome


def _run_case(op, sample, directory, *, static_integers=False) -> tuple[str, str]:
    from cattorch import CattorchError, StorageConfig, transpile, verify

    model, inputs = build(op, sample, static_integers=static_integers)
    if model is None:
        return "skip", inputs
    try:
        # The eager reference can warn about deprecated PyTorch arguments.
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = _outputs(model(*inputs))
            if not expected:
                return "skip", "no tensor output"
            if any(t.is_complex() for t in expected):
                return "skip", "complex output"
            if not _deterministic(model, inputs):
                return "skip", "nondeterministic"
    except Exception as error:  # noqa: BLE001 - eager failures aren't ours
        return "skip", f"eager failed: {_first_line(error)}"
    path = Path(directory) / f"case-{os.getpid()}"
    try:
        with _deadline(TIMEOUT_SECONDS), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = transpile(
                model, inputs, path, storage=StorageConfig(compression=False),
            )
            report = verify(model, inputs, result, atol=ATOL, rtol=RTOL)
    except _Timeout:
        return "timeout", f"over {TIMEOUT_SECONDS} s"
    except CattorchError as error:
        return "unsupported", _first_line(error)
    except Exception as error:  # noqa: BLE001 - every crash is a finding
        return "error", _first_line(error)
    if not report.passed:
        return "mismatch", _mismatch(report)
    return "pass", ""


def _mismatch(report) -> str:
    reports = getattr(report, "outputs", None) or (report,)
    worst = max(reports, key=lambda item: item.max_abs_error)
    if worst.actual_values != worst.values_compared or worst.values_compared == 0 and (
        worst.max_abs_error == float("inf")
    ):
        return f"output has {worst.actual_values} values, expected shape {worst.expected_shape}"
    return f"max abs error {worst.max_abs_error:.3g}"


def cases(names=None):
    for op in operator_db():
        name = _operator_name(op)
        if names is not None and name not in names:
            continue
        yield op


def _run_operator(name):
    torch.set_num_threads(1)
    op = next(op for op in operator_db() if _operator_name(op) == name)
    directory = tempfile.mkdtemp(prefix="cattorch-opinfo-")
    results = {}
    try:
        samples = _samples(op)
    except Exception as error:  # noqa: BLE001
        return name, {f"{name}#0": ("skip", f"no samples: {_first_line(error)}")}
    for index, sample in enumerate(samples):
        results[f"{name}#{index}"] = run_case(op, sample, directory)
    return name, results


def run(names=None, workers=None, progress=True):
    names = [_operator_name(op) for op in cases(names)]
    results = {}
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers or os.cpu_count(), maxtasksperchild=8) as pool:
        for done, (name, outcome) in enumerate(pool.imap_unordered(_run_operator, names), 1):
            results.update(outcome)
            if progress:
                summary = Counter(value[0] for value in outcome.values())
                print(f"[{done}/{len(names)}] {name}: {dict(summary)}", flush=True)
    return dict(sorted(results.items()))


def sample(key):
    """The op and sample for a case key, for tests and debugging."""
    name, index = key.rsplit("#", 1)
    op = next(op for op in operator_db() if _operator_name(op) == name)
    return op, _samples(op)[int(index)]


def _digest(results) -> str:
    return hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()[:12]


def write_coverage(results) -> str:
    by_operator: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key, (outcome, detail) in results.items():
        by_operator[key.rsplit("#", 1)[0]].append((outcome, detail))
    rows = {"pass": [], "partial": [], "fail": []}
    for name, outcomes in sorted(by_operator.items()):
        tried = [item for item in outcomes if item[0] != "skip"]
        if not tried:
            continue
        passed = sum(outcome == "pass" for outcome, _ in tried)
        failures = [
            "passes only with constant integer arguments" if outcome == "static" else detail
            for outcome, detail in tried if outcome != "pass"
        ]
        reason = _short_reason(Counter(failures).most_common(1)[0][0]) if failures else ""
        if passed == len(tried):
            rows["pass"].append((name, passed, len(tried), reason))
        elif passed:
            rows["partial"].append((name, passed, len(tried), reason))
        else:
            rows["fail"].append((name, passed, len(tried), reason))
    totals = Counter(outcome for outcome, _ in results.values())
    tried = sum(count for outcome, count in totals.items() if outcome != "skip")
    lines = [
        "# Operator coverage",
        "",
        "[Documentation home](index.md)",
        "",
        "<!-- Generated by `python tests/opinfo_conformance.py --write`; don't edit. -->",
        "",
        "This page lists how PyTorch's own operator samples (OpInfo) fare when",
        "exported with cattorch and run in its Scratch emulator. Each operator is",
        f"tried with up to {SAMPLES_PER_OPERATOR} float32 samples, and a sample",
        "passes when its export matches eager PyTorch. A sample can fail for",
        "an argument a model rarely uses, so a partial or failing operator may",
        "still work in your model; see [supported models](supported-models.md)",
        "for what cattorch is designed to handle.",
        "",
        f"Of {tried} samples tried with PyTorch {torch_version()}, {totals['pass']} pass.",
        "Samples were skipped when their output is random, complex, or not a tensor.",
        "",
    ]
    for title, key in (
        ("Every sample passes", "pass"), ("Some samples pass", "partial"),
        ("No sample passes", "fail"),
    ):
        entries = rows[key]
        lines += [f"## {title} ({len(entries)})", ""]
        if key == "pass":
            lines += textwrap.wrap(
                ", ".join(f"`{name}`" for name, *_ in entries),
                width=79, break_long_words=False, break_on_hyphens=False,
            )
            lines.append("")
            continue
        lines += ["| Operator | Passing | Most common failure |", "|---|---|---|"]
        for name, passed, count, reason in entries:
            lines.append(f"| `{name}` | {passed}/{count} | {reason} |")
        lines.append("")
    return "\n".join(lines)


def _short_reason(detail: str) -> str:
    """A failure message without the error class and documentation links."""
    text = re.sub(r";? *(supported operations:|see) https?://\S+", "", detail)
    unsupported = re.match(
        r"UnsupportedOperationError: Unsupported operation: ([\w.]+); "
        r"exported node: [\w.]+(?:; (.*))?", text,
    )
    if unsupported:
        operation, reason = unsupported.groups()
        text = f"`{operation}`: {reason}" if reason else f"no kernel for `{operation}`"
    elif "get value out of a tracing tensor with" in text:
        operation = re.search(r"tracing tensor with ([\w.]+)", text).group(1)
        text = f"needs an input's value at export (`{operation}`)"
    else:
        text = re.sub(r"^\w+Error: ", "", text)
        text = text.replace("the FX frontend could not trace this model ", "tracing failed ")
    text = text.split("; a common cause")[0]
    return text.replace("|", "\\|").replace("\n", " ")[:160]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("operators", nargs="*", help="limit the run to these operators")
    parser.add_argument("--write", action="store_true", help="update the pass list and docs")
    parser.add_argument(
        "--check", action="store_true",
        help="fail if a case in the pass list no longer passes",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--results", type=Path, help="also save every outcome as JSON")
    args = parser.parse_args(argv)
    results = run(set(args.operators) or None, args.workers)
    if args.results:
        args.results.write_text(json.dumps(results, indent=1) + "\n")
    totals = Counter(outcome for outcome, _ in results.values())
    print(dict(totals))
    if args.check:
        if (reason := version_mismatch()) is not None:
            print(f"skipped: {reason}")
            return 0
        expected = set(recorded()[1])
        if args.operators:
            expected = {key for key in expected if key.rsplit("#", 1)[0] in args.operators}
        lost = sorted(key for key in expected if results.get(key, ("missing",))[0] != "pass")
        for key in lost:
            print(f"no longer passes: {key}: {results.get(key)}")
        gained = sum(
            outcome == "pass" and key not in expected
            for key, (outcome, _) in results.items()
        )
        if gained:
            print(f"{gained} new passing cases; run with --write to record them")
        if lost:
            return 1
    if args.write:
        if args.operators:
            parser.error("--write needs a full run")
        passing = sorted(key for key, (outcome, _) in results.items() if outcome == "pass")
        if PASSING.exists() and version_mismatch() is None:
            lost = sorted(set(recorded()[1]) - set(passing))
            if lost:
                print(f"warning: {len(lost)} previously passing cases now fail:", *lost, sep="\n  ")
        PASSING.write_text(json.dumps(
            {"torch": torch_version(), "passing": passing}, indent=1,
        ) + "\n")
        COVERAGE.write_text(write_coverage(results))
        print(f"wrote {len(passing)} passing cases ({_digest(passing)})")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
