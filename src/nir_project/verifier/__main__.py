"""Command line.

  python -m nir_project.verifier family data/families/flat_roof [more dirs...]
  python -m nir_project.verifier graph data/families/flat_roof answer.yaml --param screed=1 [--json]

Exit codes: 0 no errors, 1 verification errors, 2 bad input (missing file, broken YAML,
wrong arguments).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from nir_project.verifier.checks import Report, check_family
from nir_project.verifier.compare import verify_graph
from nir_project.verifier.instances import make_instance, reference_graph
from nir_project.verifier.model import load_family, load_graph

DEFAULT_SOURCES = Path("data/sources.yaml")
TRUE_VALUES = ("1", "true", "yes")
FALSE_VALUES = ("0", "false", "no")


class UsageError(Exception):
    """Wrong command-line input; reported with exit code 2."""


def _print_report(report: Report, title: str) -> None:
    print(f"{title}: {len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    for i in report.issues:
        where = f" [{i.instance}]" if i.instance else ""
        print(f"  {i.severity.upper():7} {i.code:28} {i.subject}: {i.message}{where}")


def _parse_params(items: list[str]) -> dict[str, bool]:
    params: dict[str, bool] = {}
    for item in items:
        name, sep, value = item.partition("=")
        value = value.strip().lower()
        if not sep or not name.strip() or value not in TRUE_VALUES + FALSE_VALUES:
            raise UsageError(f"--param expects NAME=0|1, got '{item}'")
        params[name.strip()] = value in TRUE_VALUES
    return params


def _resolve_sources(given: Path | None) -> Path | None:
    """An explicit registry must exist; a missing default only switches its checks off, loudly."""
    if given is not None:
        if not given.exists():
            raise UsageError(f"source registry {given} not found")
        return given
    if DEFAULT_SOURCES.exists():
        return DEFAULT_SOURCES
    print(
        f"warning: {DEFAULT_SOURCES} not found (run from the repository root or pass --sources);"
        " source registry checks are skipped",
        file=sys.stderr,
    )
    return None


def _run_family(dirs: list[Path], sources: Path | None) -> int:
    """Check every directory even if one fails; the worst outcome sets the exit code."""
    failed = broken = False
    for d in dirs:
        try:
            report = check_family(load_family(d, sources))
        except (ValueError, OSError) as exc:  # FormatError is a ValueError
            print(f"{d.name}: cannot load: {exc}", file=sys.stderr)
            broken = True
            continue
        _print_report(report, d.name)
        failed |= not report.ok
    return 2 if broken else 1 if failed else 0


def _run_graph(args: argparse.Namespace, sources: Path | None) -> int:
    family = load_family(args.dir, sources)
    inst = make_instance(family, _parse_params(args.param))
    pairs = [nr.pair for nr in family.non_relations if set(nr.pair) <= inst.operation_ids]
    verdict = verify_graph(reference_graph(inst), load_graph(args.graph), pairs)

    if args.json:  # stdout holds JSON only, so another program can read it
        result = {
            "instance": inst.label,
            "ok": verdict.ok,
            "issues": [asdict(i) for i in verdict.report.issues],
            "metrics": verdict.metrics.as_dict(),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_report(verdict.report, inst.label)
        m = verdict.metrics
        print(
            f"TP={m.tp} FP={m.fp} FN={m.fn} recall={m.recall:.3f} "
            f"fp_rate={m.fp_rate:.3f} (budget ok: {m.fp_budget_ok}) unknown={m.unknown_pairs}"
        )
    return 0 if verdict.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nir_project.verifier",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fam = sub.add_parser("family", help="validate the reference of one or more families")
    fam.add_argument("dirs", type=Path, nargs="+", help="data/families/<name> directories")
    fam.add_argument("--sources", type=Path, default=None, help=f"default: {DEFAULT_SOURCES}")

    gr = sub.add_parser("graph", help="verify a predicted graph against a family instance")
    gr.add_argument("dir", type=Path, help="data/families/<name>")
    gr.add_argument("graph", type=Path, help="predicted graph, YAML")
    gr.add_argument("--param", action="append", default=[], metavar="NAME=0|1")
    gr.add_argument("--sources", type=Path, default=None, help=f"default: {DEFAULT_SOURCES}")
    gr.add_argument("--json", action="store_true", help="print the report and metrics as JSON")

    args = parser.parse_args(argv)
    try:
        sources = _resolve_sources(args.sources)
        if args.command == "family":
            return _run_family(args.dirs, sources)
        return _run_graph(args, sources)
    except (UsageError, ValueError, OSError) as exc:  # FormatError is a ValueError
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
