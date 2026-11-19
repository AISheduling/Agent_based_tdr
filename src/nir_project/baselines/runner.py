"""
Run one method over every family instance x seed; verify each answer; log it all to MLflow.

  uv run python -m nir_project.baselines.runner configs/sweep_llm.yaml [--limit N] [--dry-run]

One parent MLflow run per sweep, one child run per (instance, seed). Results tables go to
`results/<sweep name>/`: `items.csv` (one row per child run: commit, config, seed, run id,
metrics) and `summary.csv` (mean per family and the macro mean over families).

Families are weighted equally in the macro mean: flat_roof has 8 instances and monolithic_floor 1,
and the 8 roof instances are variants of one template, not independent packages.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import mlflow
import yaml
from dotenv import load_dotenv

from nir_project.baselines.methods import Method, Task
from nir_project.baselines.package import build_package, load_family_meta
from nir_project.config import Settings, load_settings
from nir_project.pipeline import current_commit
from nir_project.verifier.closure import transitive_closure
from nir_project.verifier.compare import verify_graph
from nir_project.verifier.instances import Instance, enumerate_instances, reference_graph
from nir_project.verifier.model import Graph, load_family

DEFAULT_SOURCES = Path("data/sources.yaml")
SUMMARY_METRICS = (
    "recall",
    "precision",
    "fp_rate",
    "fp_budget_ok",
    "independence_kept",
    "fp_on_non_relation",
    "unknown_pairs",
    "valid",
)


@dataclass(frozen=True)
class SweepConfig:
    name: str
    method: str  # which method to build: llm-structured
    families: tuple[Path, ...]
    seeds: tuple[int, ...]
    method_config: Path | None = None
    stage: str = "baseline"
    pr: str = ""
    # Error codes that do not make an answer invalid for this method (L1 has no documents, so
    # every mandatory edge lacks a source by construction).
    allowed_codes: tuple[str, ...] = ()

    @classmethod
    def from_yaml(cls, path: Path) -> SweepConfig:
        with path.open(encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        return cls(
            name=str(raw["name"]),
            method=str(raw["method"]),
            families=tuple(Path(p) for p in raw["families"]),
            seeds=tuple(int(s) for s in raw["seeds"]),
            method_config=Path(raw["method_config"]) if raw.get("method_config") else None,
            stage=str(raw.get("stage", "baseline")),
            pr=str(raw.get("pr") or ""),
            allowed_codes=tuple(str(c) for c in raw.get("allowed_codes") or ()),
        )


@dataclass
class ItemResult:
    family: str
    instance: str
    seed: int
    run_id: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    issues: dict[str, int] = field(default_factory=dict)
    error: str = ""  # the method raised; no metrics


def independence(reference: Graph, predicted: Graph) -> tuple[int, int]:
    """(incomparable reference pairs, how many of them the answer leaves unordered).

    The reference partial orders are close to total (82-95 % of pairs ordered), so recall alone
    barely separates methods; the few parallel pairs are where technology and dispatch order part.
    """
    vocab = reference.operations
    ref = transitive_closure((e.src, e.dst) for e in reference.edges if e.type == "mandatory")
    pred = transitive_closure(
        (e.src, e.dst)
        for e in predicted.edges
        if e.type == "mandatory" and e.src in vocab and e.dst in vocab
    )
    free = [
        (a, b) for a, b in combinations(sorted(vocab), 2) if (a, b) not in ref and (b, a) not in ref
    ]
    kept = sum(1 for a, b in free if (a, b) not in pred and (b, a) not in pred)
    return len(free), kept


def iter_tasks(families: Sequence[Path], seeds: Sequence[int]) -> Iterator[Task]:
    sources = DEFAULT_SOURCES if DEFAULT_SOURCES.exists() else None
    for family_dir in families:
        family = load_family(family_dir, sources)
        meta = load_family_meta(family_dir)
        for instance in enumerate_instances(family):
            for seed in seeds:
                yield Task(instance, build_package(instance, meta, seed), meta)


def _non_relations(instance: Instance) -> list[tuple[str, str]]:
    return [
        nr.pair for nr in instance.family.non_relations if set(nr.pair) <= instance.operation_ids
    ]


def evaluate(
    task: Task, graph: Graph, allowed_codes: Sequence[str]
) -> tuple[dict[str, float], dict[str, int]]:
    """Verifier metrics plus the independence and validity measures of the runner."""
    reference = reference_graph(task.instance)
    verdict = verify_graph(reference, graph, _non_relations(task.instance))
    issues = dict(Counter(i.code for i in verdict.report.errors))
    free, kept = independence(reference, graph)
    metrics = {k: float(v) for k, v in verdict.metrics.as_dict().items()}
    metrics.update(
        {
            "edges": float(len(graph.edges)),
            "errors": float(sum(issues.values())),
            "errors_counted": float(sum(n for c, n in issues.items() if c not in allowed_codes)),
            "valid": float(all(c in allowed_codes for c in issues)),
            "incomparable_pairs": float(free),
            "independence_kept": kept / free if free else 1.0,
        }
    )
    return metrics, issues


def _flat_params(params: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flat_params(value, f"{name}."))
        else:
            flat[name] = str(value)
    return flat


def run_sweep(
    cfg: SweepConfig,
    method: Method,
    settings: Settings,
    config_path: Path,
    *,
    limit: int | None = None,
    log: Any = print,
) -> tuple[str, list[ItemResult]]:
    """Run the sweep; return the parent run id and one result per (instance, seed)."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    commit = current_commit()
    tags = {"stage": cfg.stage, "model_family": method.model_family, "pr": cfg.pr, "commit": commit}
    method_params = _flat_params(method.params())
    items: list[ItemResult] = []

    with mlflow.start_run(run_name=cfg.name) as parent:
        mlflow.set_tags(
            {**tags, "dataset": ",".join(p.name for p in cfg.families), "sweep": cfg.name}
        )
        mlflow.log_params({**method_params, "seeds": ",".join(map(str, cfg.seeds))})
        mlflow.log_artifact(str(config_path))
        if cfg.method_config is not None:
            mlflow.log_artifact(str(cfg.method_config))

        tasks = list(iter_tasks(cfg.families, cfg.seeds))[:limit]
        for n, task in enumerate(tasks, 1):
            inst = task.instance
            item = ItemResult(inst.family.name, inst.label, task.package.seed)
            with mlflow.start_run(run_name=f"{inst.label}-s{item.seed}", nested=True) as child:
                item.run_id = str(child.info.run_id)
                mlflow.set_tags({**tags, "dataset": inst.family.name, "instance": inst.label})
                mlflow.log_params({**method_params, "seed": item.seed, "instance": inst.label})
                try:
                    out = method.predict(task)
                except Exception as exc:  # one failed call must not lose the whole sweep
                    item.error = f"{type(exc).__name__}: {exc}"
                    mlflow.set_tag("error", item.error[:500])
                    log(f"[{n}/{len(tasks)}] {inst.label} s{item.seed}: FAILED {item.error}")
                    items.append(item)
                    continue
                item.metrics, item.issues = evaluate(task, out.graph, cfg.allowed_codes)
                item.metrics.update({k: float(v) for k, v in out.cost.items()})
                item.metrics["parse_problems"] = float(len(out.problems))
                mlflow.log_metrics(item.metrics)
                with tempfile.TemporaryDirectory() as tmp:
                    for name, text in out.artifacts.items():
                        (Path(tmp) / name).write_text(text, encoding="utf-8")
                    issues_text = yaml.safe_dump(
                        {"issues": item.issues, "problems": list(out.problems)}, allow_unicode=True
                    )
                    (Path(tmp) / "issues.yaml").write_text(issues_text, encoding="utf-8")
                    mlflow.log_artifacts(tmp)
                m = item.metrics
                log(
                    f"[{n}/{len(tasks)}] {inst.label} s{item.seed}: recall={m['recall']:.3f} "
                    f"precision={m['precision']:.3f} indep={m['independence_kept']:.2f} "
                    f"valid={int(m['valid'])}"
                )
            items.append(item)

        summary = summarize(items)
        for row in summary:
            prefix = str(row["family"])
            mlflow.log_metrics(
                {f"{prefix}.{k}": float(v) for k, v in row.items() if isinstance(v, float)}
            )
        return str(parent.info.run_id), items


def summarize(items: Sequence[ItemResult]) -> list[dict[str, Any]]:
    """Mean and stdev per family over instances x seeds, then the macro mean over families."""
    by_family: dict[str, list[ItemResult]] = defaultdict(list)
    for item in items:
        by_family[item.family].append(item)

    rows: list[dict[str, Any]] = []
    for family, group in sorted(by_family.items()):
        ok = [i for i in group if not i.error]
        row: dict[str, Any] = {"family": family, "runs": len(group), "failed": len(group) - len(ok)}
        for key in SUMMARY_METRICS:
            values = [i.metrics[key] for i in ok if key in i.metrics]
            row[f"{key}_mean"] = statistics.fmean(values) if values else math.nan
            row[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(row)

    if rows:
        macro: dict[str, Any] = {
            "family": "macro",
            "runs": sum(r["runs"] for r in rows),
            "failed": sum(r["failed"] for r in rows),
        }
        for key in SUMMARY_METRICS:
            means = [r[f"{key}_mean"] for r in rows if not math.isnan(r[f"{key}_mean"])]
            macro[f"{key}_mean"] = statistics.fmean(means) if means else math.nan
            macro[f"{key}_std"] = math.nan  # spread over families is not meaningful with two
        rows.append(macro)
    return rows


def write_results(
    out_dir: Path,
    items: Sequence[ItemResult],
    summary: Sequence[Mapping[str, Any]],
    *,
    commit: str,
    config_path: Path,
    parent_run_id: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = sorted({k for i in items for k in i.metrics})
    with (out_dir / "items.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["commit", "config", "family", "instance", "seed", "run_id", "error", *metric_keys]
        )
        for i in items:
            writer.writerow(
                [commit, config_path.as_posix(), i.family, i.instance, i.seed, i.run_id, i.error]
                + [i.metrics.get(k, "") for k in metric_keys]
            )
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        keys = list(summary[0]) if summary else []
        writer = csv.writer(fh)
        writer.writerow(["commit", "config", "parent_run_id", *keys])
        for row in summary:
            writer.writerow(
                [commit, config_path.as_posix(), parent_run_id]
                + [f"{v:.4f}" if isinstance(v, float) else v for v in row.values()]
            )


def build_method(cfg: SweepConfig, settings: Settings) -> Method:
    if cfg.method == "llm-structured":
        from nir_project.baselines.llm import LLMConfig, OpenAICompatibleClient
        from nir_project.baselines.structured import LLMStructured

        if settings.model_api_key is None:
            raise SystemExit("MODEL_API_KEY is not set (.env)")
        llm_cfg = LLMConfig.from_yaml(cfg.method_config or Path("configs/baseline_llm.yaml"))
        client = OpenAICompatibleClient(
            llm_cfg, settings.model_api_key.get_secret_value(), settings.model_base_url
        )
        return LLMStructured(llm_cfg, client)
    raise SystemExit(f"unknown method '{cfg.method}'")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nir_project.baselines.runner")
    ap.add_argument("config", type=Path, help="sweep config, e.g. configs/sweep_llm.yaml")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N tasks")
    ap.add_argument("--dry-run", action="store_true", help="list the tasks, call nothing")
    args = ap.parse_args(argv)

    load_dotenv()
    settings = load_settings()
    cfg = SweepConfig.from_yaml(args.config)

    if args.dry_run:
        tasks = list(iter_tasks(cfg.families, cfg.seeds))[: args.limit]
        for t in tasks:
            print(f"{t.instance.label} seed={t.package.seed} ops={len(t.package.operations)}")
        print(f"{len(tasks)} task(s)")
        return 0

    method = build_method(cfg, settings)
    parent_id, items = run_sweep(cfg, method, settings, args.config, limit=args.limit)
    summary = summarize(items)
    out_dir = settings.results_dir / cfg.name
    write_results(
        out_dir,
        items,
        summary,
        commit=current_commit(),
        config_path=args.config,
        parent_run_id=parent_id,
    )
    print(f"\nparent run {parent_id}; tables in {out_dir}")
    for row in summary:
        print(
            f"  {row['family']:18} runs={row['runs']:3} failed={row['failed']} "
            f"recall={row['recall_mean']:.3f} precision={row['precision_mean']:.3f} "
            f"budget_ok={row['fp_budget_ok_mean']:.2f} indep={row['independence_kept_mean']:.2f} "
            f"valid={row['valid_mean']:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
