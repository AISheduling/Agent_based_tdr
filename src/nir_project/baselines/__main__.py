"""Command line for the L1 baseline on one family instance.

  python -m nir_project.baselines data/families/flat_roof --param screed=1 --dry-run
  python -m nir_project.baselines data/families/flat_roof --param screed=1 --seed 42

`--dry-run` prints the prompt and calls nothing. Otherwise the answer is verified against the
reference of the instance; prompt, raw text and the answer in verifier format go to `--out`, and
`summary.json` records the commit, the instance, what the provider reported about the call and
the command that rescores the saved answer without a new call.

Before anything, the family must pass the verifier's `family` checks: a call against a broken
reference would be paid for and scored wrong (exit code 2).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import yaml

from nir_project.baselines.llm import LLMConfig, OpenAICompatibleClient
from nir_project.baselines.package import build_package, load_family_meta
from nir_project.baselines.prompt import PROMPT_VERSION, build_messages
from nir_project.baselines.structured import predict
from nir_project.config import load_settings
from nir_project.pipeline import current_commit
from nir_project.verifier.checks import check_family
from nir_project.verifier.compare import verify_graph
from nir_project.verifier.instances import make_instance, reference_graph
from nir_project.verifier.model import load_family

DEFAULT_SOURCES = Path("data/sources.yaml")


def _params(items: list[str]) -> dict[str, bool]:
    params: dict[str, bool] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or value.strip().lower() not in ("0", "1", "true", "false"):
            raise SystemExit(f"--param expects NAME=0|1, got '{item}'")
        params[name.strip()] = value.strip().lower() in ("1", "true")
    return params


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nir_project.baselines")
    ap.add_argument("family", type=Path, help="family folder, e.g. data/families/flat_roof")
    ap.add_argument("--param", action="append", default=[], help="NAME=0|1, repeatable")
    ap.add_argument("--config", type=Path, default=Path("configs/baseline_llm.yaml"))
    ap.add_argument("--seed", type=int, default=None, help="shuffle seed; default SEED from .env")
    ap.add_argument("--out", type=Path, default=Path("runs/baseline_llm"))
    ap.add_argument("--dry-run", action="store_true", help="print the prompt, call nothing")
    args = ap.parse_args(argv)

    settings = load_settings()
    seed = settings.seed if args.seed is None else args.seed
    try:
        family = load_family(args.family, DEFAULT_SOURCES if DEFAULT_SOURCES.exists() else None)
    except (ValueError, OSError) as exc:  # FormatError is a ValueError
        print(f"error: cannot load {args.family}: {exc}", file=sys.stderr)
        return 2
    reference_report = check_family(family)
    if not reference_report.ok:
        for issue in reference_report.errors:
            print(f"  {issue.code:28} {issue.subject}: {issue.message}", file=sys.stderr)
        print(
            f"error: the reference of {args.family.name} fails the verifier's family checks; "
            "nothing is called",
            file=sys.stderr,
        )
        return 2
    try:
        instance = make_instance(family, _params(args.param))
    except ValueError as exc:  # unknown parameter name
        print(f"error: {exc}", file=sys.stderr)
        return 2
    package = build_package(instance, load_family_meta(args.family), seed)

    if args.dry_run:
        for msg in build_messages(package):
            print(f"===== {msg['role']} =====\n{msg['content']}\n")
        return 0

    if settings.model_api_key is None:
        print("MODEL_API_KEY is not set (.env)", file=sys.stderr)
        return 2
    cfg = LLMConfig.from_yaml(args.config)
    client = OpenAICompatibleClient(
        cfg, settings.model_api_key.get_secret_value(), settings.model_base_url
    )
    result = predict(package, client)

    non_relations = [
        nr.pair for nr in family.non_relations if set(nr.pair) <= instance.operation_ids
    ]
    verdict = verify_graph(reference_graph(instance), result.answer.graph, non_relations)

    run_dir = args.out / f"{instance.label}-s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompt.json").write_text(
        json.dumps(result.messages, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (run_dir / "raw.txt").write_text(result.completion.text, encoding="utf-8")
    answer_path = run_dir / "answer.yaml"
    with answer_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(result.answer.as_graph_dict(), fh, allow_unicode=True, sort_keys=False)
    rescore = " ".join(
        [
            "python -m nir_project.verifier graph",
            args.family.as_posix(),
            answer_path.as_posix(),
            *(f"--param {k}={int(v)}" for k, v in sorted(instance.params.items())),
        ]
    )
    summary = {
        "commit": current_commit(),
        "family": args.family.as_posix(),
        "params": dict(instance.params),
        "instance": instance.label,
        "seed": seed,
        "prompt_version": PROMPT_VERSION,
        "config": asdict(cfg),
        "completion": {k: v for k, v in asdict(result.completion).items() if k != "text"},
        "parse_problems": list(result.answer.problems),
        "issues": dict(Counter(i.code for i in verdict.report.issues)),
        "metrics": verdict.metrics.as_dict(),
        "rescore": rescore,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    m = verdict.metrics
    c = result.completion
    print(f"{instance.label} seed={seed} model={c.model}")
    if c.finish_reason == "length":
        print(f"  WARNING: answer cut off at max_tokens={cfg.max_tokens} (attempts={c.attempts})")
    print(f"  edges={len(result.answer.graph.edges)} parse_problems={len(result.answer.problems)}")
    print(f"  issues: {summary['issues'] or 'none'}")
    print(
        f"  recall={m.recall:.3f} precision={m.precision:.3f} fp_rate={m.fp_rate:.3f} "
        f"(budget ok: {m.fp_budget_ok}) tp={m.tp} fp={m.fp} fn={m.fn} unknown={m.unknown_pairs}"
    )
    print(
        f"  tokens in/out={result.completion.prompt_tokens}/{result.completion.completion_tokens} "
        f"time={result.completion.seconds:.1f}s finish={c.finish_reason} -> {run_dir}"
    )
    print(f"  rescore without a call: {rescore}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
