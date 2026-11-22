"""Compare a predicted graph with the reference by transitive closure (ADR 0001).

Only `mandatory` edges define a closure, so equivalent sets of edges (A->B->C with or without
A->C) give the same score. The main score uses the confirmed part of the reference and masks the
pairs whose obligation is not established (ADR 0001, "Неподтверждённые рёбра и их учёт в метрике"):

* `R_full` - closure of every mandatory reference edge (active conditionals arrive as mandatory);
* `R_conf` - closure of the confirmed ones: `status: source_checked`, or no status at all
  (a hand-built graph);
* `U` - `R_full - R_conf` plus the pairs of reference `unknown` edges;
* mask `M` - `U` and its reverse, minus every pair that `R_conf` orders either way round.

TP = |P & R_conf|, FN = |R_conf - P|, FP = |P - R_conf - M|; a masked pair gives nothing.
An answer whose mandatory edges form a cycle is invalid: TP = 0, FN = |R_conf|, FP as usual.
The same counts against `R_full` without a mask are reported as `*_full` (sensitivity).
`unknown` in the answer is never a false positive.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from nir_project.verifier.checks import Report, check_graph
from nir_project.verifier.closure import find_cycle, transitive_closure
from nir_project.verifier.model import Edge, Graph, Pair

FP_BUDGET = 0.02  # main metric: recall of mandatory pairs at FP / (TP + FP) <= 2%
CONFIRMED = "source_checked"


@dataclass(frozen=True)
class Metrics:
    tp: int
    fp: int
    fn: int
    recall: float
    precision: float
    fp_rate: float  # FP / (TP + FP), masked pairs excluded
    fp_budget_ok: bool
    fn_declared_unknown: int  # missed pairs the answer explicitly marked `unknown`
    fn_silent: int  # missed pairs the answer said nothing about
    fp_on_non_relation: int  # false pairs that contradict a confirmed independence
    fp_on_preferred: int  # false pairs that promote a reference `preferred` to mandatory
    unknown_pairs: int  # unknown pairs the answer does not also assert (never counted as FP)
    masked_pairs: int  # reference pairs whose obligation is not established (|U| outside R_conf)
    masked_predicted: int  # answer pairs that fall into the mask: neither TP nor FP
    cyclic: bool  # the answer's mandatory edges form a cycle: no TP is credited
    tp_full: int  # the same counts against the full reference, no mask (sensitivity)
    fp_full: int
    fn_full: int
    recall_full: float
    precision_full: float

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class Verdict:
    report: Report
    metrics: Metrics

    @property
    def ok(self) -> bool:
        return self.report.ok


def _confirmed(edge: Edge) -> bool:
    return edge.status is None or edge.status == CONFIRMED


def _pairs(graph: Graph, allowed: frozenset[str], type_: str = "mandatory") -> list[Pair]:
    """Edges of one type with both ends in `allowed` (the others are reported errors)."""
    return [
        (e.src, e.dst)
        for e in graph.edges
        if e.type == type_ and e.src in allowed and e.dst in allowed
    ]


def _reverse(pairs: Iterable[Pair]) -> set[Pair]:
    return {(b, a) for a, b in pairs}


def _ratio(num: int, den: int, empty: float) -> float:
    return num / den if den else empty


def compute_metrics(
    reference: Graph, predicted: Graph, non_relations: Sequence[Pair] = ()
) -> Metrics:
    vocab = reference.operations
    # An answer edge counts only if both ends are operations the answer lists and the
    # vocabulary knows: `check_graph` rejects every other edge, and the score must agree.
    allowed = vocab & predicted.operations

    r_full = transitive_closure(_pairs(reference, vocab))
    r_conf = transitive_closure(
        (e.src, e.dst)
        for e in reference.edges
        if e.type == "mandatory" and _confirmed(e) and e.src in vocab and e.dst in vocab
    )
    settled = r_conf | _reverse(r_conf)
    undecided = ((r_full - r_conf) | set(_pairs(reference, vocab, "unknown"))) - settled
    mask = (undecided | _reverse(undecided)) - settled

    answer_pairs = _pairs(predicted, allowed)
    cyclic = find_cycle(answer_pairs) is not None
    pred = transitive_closure(answer_pairs)

    tp: set[Pair] = set() if cyclic else pred & r_conf
    fp = pred - r_conf - mask
    fn = r_conf - tp
    tp_full: set[Pair] = set() if cyclic else pred & r_full
    fp_full = pred - r_full
    fn_full = r_full - tp_full

    # a pair the answer also asserts as mandatory is not an abstention
    unknown = set(_pairs(predicted, allowed, "unknown")) - pred
    declared = unknown | _reverse(unknown)
    preferred_ref = set(_pairs(reference, vocab, "preferred"))
    independent = {p for a, b in non_relations for p in ((a, b), (b, a))}
    scored = len(tp) + len(fp)

    return Metrics(
        tp=len(tp),
        fp=len(fp),
        fn=len(fn),
        recall=_ratio(len(tp), len(r_conf), 1.0),
        precision=_ratio(len(tp), scored, 1.0),
        fp_rate=_ratio(len(fp), scored, 0.0),
        fp_budget_ok=_ratio(len(fp), scored, 0.0) <= FP_BUDGET,
        fn_declared_unknown=len(fn & declared),
        fn_silent=len(fn - declared),
        fp_on_non_relation=len(fp & independent),
        fp_on_preferred=len(fp & preferred_ref),
        unknown_pairs=len(unknown),
        masked_pairs=len(undecided),
        masked_predicted=len(pred & mask),
        cyclic=cyclic,
        tp_full=len(tp_full),
        fp_full=len(fp_full),
        fn_full=len(fn_full),
        recall_full=_ratio(len(tp_full), len(r_full), 1.0),
        precision_full=_ratio(len(tp_full), len(tp_full) + len(fp_full), 1.0),
    )


def verify_graph(reference: Graph, predicted: Graph, non_relations: Sequence[Pair] = ()) -> Verdict:
    """Structural checks of the answer plus closure comparison with the reference."""
    report = check_graph(predicted, reference.operations, non_relations)
    return Verdict(report, compute_metrics(reference, predicted, non_relations))
