"""Runner without the network: fake methods, an isolated MLflow store, tables on disk."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
import pytest

from nir_project.baselines import Completion, LLMConfig, LLMStructured, MethodOutput, Task
from nir_project.baselines.runner import (
    SweepConfig,
    independence,
    iter_tasks,
    run_sweep,
    summarize,
    write_results,
)
from nir_project.config import Settings
from nir_project.verifier import Edge, Graph, reference_graph

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = (
    ROOT / "data" / "families" / "flat_roof",
    ROOT / "data" / "families" / "monolithic_floor",
)
SWEEP = ROOT / "configs" / "sweep_llm.yaml"


class ReferenceMethod:
    """Answers with the reference itself: recall 1, nothing false."""

    name = "oracle"
    model_family = "oracle"

    def params(self) -> Mapping[str, Any]:
        return {"kind": "reference", "nested": {"a": 1}}

    def predict(self, task: Task) -> MethodOutput:
        ref = reference_graph(task.instance)
        edges = tuple(
            Edge(e.src, e.dst, "mandatory", e.basis, e.sources)
            for e in ref.edges
            if e.type == "mandatory"
        )
        return MethodOutput(
            Graph(task.package.operation_ids, edges),
            cost={"seconds": 0.0},
            artifacts={"note.txt": "ok"},
        )


class FailingMethod(ReferenceMethod):
    def predict(self, task: Task) -> MethodOutput:
        if task.instance.family.name == "monolithic_floor":
            raise RuntimeError("API down")
        return super().predict(task)


class FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, messages: Any) -> Completion:
        return Completion(text=self.text, model="fake", prompt_tokens=10, completion_tokens=5)


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.chdir(tmp_path)
    return Settings(
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}", results_dir=tmp_path / "results"
    )


def _cfg(**kw: Any) -> SweepConfig:
    base: dict[str, Any] = {
        "name": "test",
        "method": "oracle",
        "families": FAMILIES,
        "seeds": (1, 2),
    }
    base.update(kw)
    return SweepConfig(**base)


def test_sweep_config_reads_the_committed_file() -> None:
    cfg = SweepConfig.from_yaml(SWEEP)
    assert cfg.method == "llm-structured"
    assert cfg.allowed_codes == ("no_source",)
    assert all(p.exists() for p in ((ROOT / f) for f in cfg.families))


def test_tasks_cover_every_instance_and_seed() -> None:
    tasks = list(iter_tasks(FAMILIES, (1, 2, 3)))
    assert len(tasks) == (8 + 1) * 3
    assert len({(t.instance.label, t.package.seed) for t in tasks}) == len(tasks)


def test_independence_counts_parallel_pairs() -> None:
    ref = Graph(frozenset("abc"), (Edge("a", "b"), Edge("a", "c")))
    assert independence(ref, ref) == (1, 1)
    chained = Graph(frozenset("abc"), (Edge("a", "b"), Edge("b", "c")))
    assert independence(ref, chained) == (1, 0)


def test_sweep_logs_child_runs_and_scores_the_oracle(settings: Settings) -> None:
    cfg = _cfg()
    parent_id, items = run_sweep(cfg, ReferenceMethod(), settings, SWEEP, log=lambda _: None)
    assert len(items) == 9 * 2
    assert all(i.metrics["recall"] == 1.0 and i.metrics["fp"] == 0 for i in items)
    assert all(i.metrics["independence_kept"] == 1.0 and i.metrics["valid"] == 1.0 for i in items)

    runs = mlflow.search_runs(
        experiment_names=[settings.mlflow_experiment_name],
        filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
        output_format="list",
    )
    assert len(runs) == len(items)
    child = runs[0]
    assert child.data.tags["stage"] == "baseline"
    assert child.data.tags["model_family"] == "oracle"
    assert child.data.params["nested.a"] == "1"


def test_failed_item_does_not_stop_the_sweep(settings: Settings) -> None:
    _, items = run_sweep(_cfg(seeds=(1,)), FailingMethod(), settings, SWEEP, log=lambda _: None)
    failed = [i for i in items if i.error]
    assert [i.family for i in failed] == ["monolithic_floor"]
    rows = {r["family"]: r for r in summarize(items)}
    assert rows["monolithic_floor"]["failed"] == 1
    assert rows["flat_roof"]["recall_mean"] == 1.0
    assert rows["macro"]["recall_mean"] == 1.0


def test_macro_weights_families_equally(settings: Settings) -> None:
    _, items = run_sweep(_cfg(seeds=(1,)), ReferenceMethod(), settings, SWEEP, log=lambda _: None)
    for i in items:  # pretend the roof is perfect and the floor scores zero
        i.metrics["recall"] = 1.0 if i.family == "flat_roof" else 0.0
    rows = {r["family"]: r for r in summarize(items)}
    assert rows["macro"]["recall_mean"] == pytest.approx(0.5)


def test_tables_carry_commit_config_seed_and_run_id(settings: Settings, tmp_path: Path) -> None:
    parent_id, items = run_sweep(
        _cfg(seeds=(1,)), ReferenceMethod(), settings, SWEEP, log=lambda _: None
    )
    write_results(
        tmp_path / "out",
        items,
        summarize(items),
        commit="abc123",
        config_path=Path("configs/x.yaml"),
        parent_run_id=parent_id,
    )
    with (tmp_path / "out" / "items.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 9
    assert {"commit", "config", "seed", "run_id", "recall"} <= set(rows[0])
    assert rows[0]["commit"] == "abc123" and rows[0]["run_id"]
    with (tmp_path / "out" / "summary.csv").open(encoding="utf-8") as fh:
        families = [r["family"] for r in csv.DictReader(fh)]
    assert families == ["flat_roof", "monolithic_floor", "macro"]


def test_llm_method_reports_cost_and_artifacts() -> None:
    task = next(iter_tasks(FAMILIES[:1], (1,)))
    method = LLMStructured(LLMConfig(), FakeClient(json.dumps({"relations": []})))
    out = method.predict(task)
    assert out.cost["prompt_tokens"] == 10 and out.cost["completion_tokens"] == 5
    assert set(out.artifacts) == {"prompt.json", "raw.txt", "answer.yaml"}
    assert method.params()["prompt_version"].startswith("l1-")
