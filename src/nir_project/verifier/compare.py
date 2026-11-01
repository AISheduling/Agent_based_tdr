"""Compare a predicted graph with the reference by transitive closure (ADR 0001).

Only `mandatory` edges define the closure. Equivalent sets of edges (A->B->C with or without
A->C) give the same closure and therefore the same score. `unknown` is reported separately and is
never a false positive by itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from nir_project.verifier.checks import Report, check_graph
from nir_project.verifier.closure import transitive_closure
from nir_project.verifier.model import Graph, Pair

FP_BUDGET = 0.02  # main metric: recall of mandatory pairs at FP / (TP + FP) <= 2%


@dataclass(frozen=True)
class Metrics:
    tp: int
    fp: int
    fn: int
    recall: float
    precision: float
    fp_rate: float
    fp_budget_ok: bool
    fn_declared_unknown: int  # missed pairs the answer explicitly marked `unknown`
    fn_silent: int  # missed pairs the answer said nothing about
    fp_on_non_relation: int  # false pairs that contradict a confirmed independence
    fp_on_preferred: int  # false pairs that promote a reference `preferred` to mandatory
    unknown_pairs: int  # unknown pairs the answer does not also assert (never counted as FP)

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class Verdict:
    report: Report
    metrics: Metrics

    @property
    def ok(self) -> bool:
        return self.report.ok


def _hard_closure(graph: Graph, allowed: frozenset[str]) -> set[Pair]:
    """Closure of mandatory edges with both ends in `allowed` (the others are reported errors)."""
    pairs = [
        (e.src, e.dst)
        for e in graph.edges
        if e.type == "mandatory" and e.src in allowed and e.dst in allowed
    ]
    return transitive_closure(pairs)


def compute_metrics(
    reference: Graph, predicted: Graph, non_relations: Sequence[Pair] = ()
) -> Metrics:
    vocab = reference.operations
    # An answer edge counts only if both ends are operations the answer lists and the
    # vocabulary knows: `check_graph` rejects every other edge, and the score must agree.
    allowed = vocab & predicted.operations
    ref = _hard_closure(reference, vocab)
    pred = _hard_closure(predicted, allowed)
    tp, fp, fn = ref & pred, pred - ref, ref - pred

    unknown = {
        (e.src, e.dst)
        for e in predicted.edges
        if e.type == "unknown" and e.src in allowed and e.dst in allowed
    } - pred  # a pair the answer also asserts as mandatory is not an abstention
    preferred_ref = {(e.src, e.dst) for e in reference.edges if e.type == "preferred"}
    independent = {p for a, b in non_relations for p in ((a, b), (b, a))}

    return Metrics(
        tp=len(tp),
        fp=len(fp),
        fn=len(fn),
        recall=len(tp) / len(ref) if ref else 1.0,
        precision=len(tp) / len(pred) if pred else 1.0,
        fp_rate=len(fp) / len(pred) if pred else 0.0,
        fp_budget_ok=(len(fp) / len(pred) if pred else 0.0) <= FP_BUDGET,
        fn_declared_unknown=len(fn & (unknown | {(b, a) for a, b in unknown})),
        fn_silent=len(fn - (unknown | {(b, a) for a, b in unknown})),
        fp_on_non_relation=len(fp & independent),
        fp_on_preferred=len(fp & preferred_ref),
        unknown_pairs=len(unknown),
    )


def verify_graph(reference: Graph, predicted: Graph, non_relations: Sequence[Pair] = ()) -> Verdict:
    """Structural checks of the answer plus closure comparison with the reference."""
    report = check_graph(predicted, reference.operations, non_relations)
    return Verdict(report, compute_metrics(reference, predicted, non_relations))
