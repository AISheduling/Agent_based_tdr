"""Machine checks of a family reference and of a graph (ADR 0001, "Consequences").

Two entry points:

* `check_family` validates the reference corpus itself, instance by instance;
* `check_graph` validates one graph (a reference instance or a model answer) against a vocabulary.

Every problem is an `Issue` with a stable `code`, so tests and reports do not parse messages.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from nir_project.verifier.closure import find_cycle, transitive_closure
from nir_project.verifier.instances import Instance, enumerate_instances, reference_graph
from nir_project.verifier.model import (
    BASES,
    CAUSES,
    EXPERT_SOURCE,
    HARD_TYPES,
    STATUSES,
    TYPES,
    Edge,
    Family,
    Graph,
    Pair,
    Relation,
    SourceRef,
)

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: str
    message: str
    subject: str = ""  # relation id, operation id or pair the issue is about
    instance: str | None = None


@dataclass(frozen=True)
class Report:
    issues: tuple[Issue, ...] = ()

    @property
    def errors(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self, severity: Severity | None = None) -> set[str]:
        return {i.code for i in self.issues if severity is None or i.severity == severity}


def _error(code: str, message: str, subject: str = "") -> Issue:
    return Issue("error", code, message, subject)


def _warning(code: str, message: str, subject: str = "") -> Issue:
    return Issue("warning", code, message, subject)


# --- provenance of one relation or edge ----------------------------------------------------------


def _check_provenance(
    subject: str,
    type_: str,
    basis: Sequence[str],
    sources: Sequence[SourceRef],
    *,
    basis_may_be_empty: bool = False,
) -> list[Issue]:
    """Basis and source rules shared by reference relations and model-answer edges."""
    issues: list[Issue] = []
    hard = type_ in HARD_TYPES
    for b in basis:
        if b not in BASES:
            issues.append(_error("invalid_basis", f"unknown basis '{b}'", subject))
    if hard and not basis and not basis_may_be_empty:
        issues.append(_error("no_basis", f"{type_} relation has no basis", subject))
    if hard and basis and set(basis) <= {"statistical_repeat"}:
        issues.append(
            _error(
                "weak_basis",
                "statistical repetition alone cannot make a relation mandatory",
                subject,
            )
        )
    if hard and not sources:
        issues.append(_error("no_source", f"{type_} relation has no source", subject))
    for ref in sources:
        if ref.source != EXPERT_SOURCE and ref.locator is None:
            issues.append(_error("no_locator", f"source {ref.source} has no locator", subject))
        if ref.source == EXPERT_SOURCE and ref.locator is not None:
            issues.append(_error("expert_locator", "SRC.EXPERT must have locator null", subject))
    return issues


# --- graph checks --------------------------------------------------------------------------------


def _pairs(edges: Iterable[Edge], types: Sequence[str]) -> list[Pair]:
    return [(e.src, e.dst) for e in edges if e.type in types]


def check_graph(
    graph: Graph,
    vocabulary: frozenset[str],
    non_relations: Sequence[Pair] = (),
    *,
    provenance: bool = True,
) -> Report:
    """Checks that need only the graph and the fixed operation vocabulary.

    `provenance=False` skips basis/source rules; `check_family` already applies them to the
    reference relations and would report each problem twice.
    """
    issues: list[Issue] = []

    for op in sorted(vocabulary - graph.operations):
        issues.append(_error("uncovered_operation", f"operation {op} is not in the graph", op))
    for op in sorted(graph.operations - vocabulary):
        issues.append(_error("unknown_operation", f"operation {op} is not in the vocabulary", op))

    valid: list[Edge] = []
    for e in graph.edges:
        subject = e.id or f"{e.src}->{e.dst}"
        if e.type not in ("mandatory", "preferred", "unknown"):
            issues.append(
                _error("invalid_type", f"edge type '{e.type}' is not allowed in a graph", subject)
            )
            continue
        missing = [n for n in (e.src, e.dst) if n not in graph.operations]
        if missing:
            issues.append(
                _error("edge_unknown_operation", f"edge uses {missing}, not in operations", subject)
            )
            continue
        if provenance:
            issues.extend(_check_provenance(subject, e.type, e.basis, e.sources))
        valid.append(e)

    hard = _pairs(valid, ("mandatory",))
    cycle = find_cycle(hard)
    if cycle is not None:
        issues.append(_error("cycle", "cycle: " + " -> ".join(cycle), cycle[0]))
    closure = transitive_closure(hard)

    for a, b in non_relations:
        if (a, b) in closure or (b, a) in closure:
            issues.append(
                _error(
                    "non_relation_violated", f"{a} and {b} are independent but ordered", f"{a},{b}"
                )
            )
    for e in valid:
        if e.type == "preferred" and (e.dst, e.src) in closure:
            issues.append(
                _error(
                    "preferred_contradicts_mandatory",
                    f"preferred {e.src}->{e.dst} contradicts a mandatory order",
                    e.id or f"{e.src}->{e.dst}",
                )
            )

    touched = {n for e in valid for n in (e.src, e.dst)}
    for op in sorted(graph.operations & vocabulary - touched):
        issues.append(_warning("isolated_operation", f"operation {op} has no relations", op))
    return Report(tuple(issues))


# --- reference checks ----------------------------------------------------------------------------


def _check_static(family: Family) -> list[Issue]:
    """Checks that do not depend on parameter values."""
    issues: list[Issue] = []
    op_ids = [o.id for o in family.operations]
    for op_id, n in Counter(op_ids).items():
        if n > 1:
            issues.append(_error("duplicate_id", f"operation id {op_id} repeats", op_id))
    ops = set(op_ids)
    ext_in = {e.id for e in family.external_inputs}
    ext_out = {e.id for e in family.external_outputs}
    params = set(family.parameters)

    for op_id, param in family.optional_operations.items():
        if op_id not in ops:
            issues.append(_error("unknown_operation", f"optional operation {op_id}", op_id))
        if param not in params:
            issues.append(_error("unknown_parameter", f"{op_id}: parameter {param}", op_id))
    for op in family.operations:
        for param in op.requires_if:
            if param not in params:
                issues.append(_error("unknown_parameter", f"{op.id}: requires_if {param}", op.id))

    rel_ids = [r.id for r in (*family.relations, *family.interface_relations)]
    for rid, n in Counter(rel_ids).items():
        if n > 1:
            issues.append(_error("duplicate_id", f"relation id {rid} repeats", rid))

    for rel in family.relations:
        issues.extend(_check_relation(family, rel, ops, ops, params))
    for rel in family.interface_relations:
        issues.extend(_check_interface_relation(family, rel, ops, ext_in, ext_out))

    for nr in family.non_relations:
        for member in nr.pair:
            if member not in ops:
                issues.append(
                    _error("unknown_operation", f"non_relation uses {member}", ",".join(nr.pair))
                )
        if nr.pair[0] == nr.pair[1]:
            issues.append(
                _error("invalid_pair", "non_relation pairs an operation with itself", nr.pair[0])
            )
        if not nr.reason:
            issues.append(_error("no_reason", "non_relation has no reason", ",".join(nr.pair)))
    return issues


def _check_endpoints(rel: Relation, srcs: set[str], dsts: set[str]) -> list[Issue]:
    issues: list[Issue] = []
    if rel.src not in srcs:
        issues.append(_error("unknown_operation", f"'from' {rel.src} is unknown", rel.id))
    if rel.dst not in dsts:
        issues.append(_error("unknown_operation", f"'to' {rel.dst} is unknown", rel.id))
    return issues


def _check_status(family: Family, rel: Relation) -> list[Issue]:
    issues: list[Issue] = []
    if rel.status not in STATUSES:
        issues.append(_error("invalid_status", f"unknown status '{rel.status}'", rel.id))
    refs = [s.source for s in rel.sources]
    if family.sources is not None:
        for ref in refs:
            if ref not in family.sources:
                issues.append(
                    _error("unknown_source", f"source {ref} is not in the registry", rel.id)
                )
    if refs and all(s == EXPERT_SOURCE for s in refs) and rel.status == "source_checked":
        issues.append(
            _error("expert_unconfirmed", "expert-only relation cannot be source_checked", rel.id)
        )
    return issues


def _check_relation(
    family: Family, rel: Relation, srcs: set[str], dsts: set[str], params: set[str]
) -> list[Issue]:
    issues = _check_endpoints(rel, srcs, dsts)
    if rel.type not in TYPES:
        issues.append(_error("invalid_type", f"unknown type '{rel.type}'", rel.id))
    if rel.type == "conditional":
        if rel.condition is None:
            issues.append(
                _error("missing_condition", "conditional relation has no condition", rel.id)
            )
        elif rel.condition not in params:
            issues.append(_error("unknown_parameter", f"condition {rel.condition}", rel.id))
    elif rel.condition is not None:
        issues.append(
            _error("unexpected_condition", f"{rel.type} relation has a condition", rel.id)
        )
    issues.extend(_check_provenance(rel.id, rel.type, rel.basis, rel.sources))
    issues.extend(_check_status(family, rel))
    return issues


def _check_interface_relation(
    family: Family, rel: Relation, ops: set[str], ext_in: set[str], ext_out: set[str]
) -> list[Issue]:
    issues = _check_endpoints(rel, ext_in | ops, ext_out | ops)
    # One end must be an operation, the other an external input (before) or output (after);
    # otherwise the relation never enters an instance and would vanish without a trace.
    links_external = (rel.src in ext_in and rel.dst in ops) or (
        rel.src in ops and rel.dst in ext_out
    )
    if not issues and not links_external:
        issues.append(
            _error(
                "invalid_interface",
                f"{rel.src}->{rel.dst} must link an operation with an external input or output",
                rel.id,
            )
        )
    if rel.cause not in CAUSES:
        issues.append(_error("invalid_cause", f"unknown cause '{rel.cause}'", rel.id))
    issues.extend(
        _check_provenance(
            rel.id, "mandatory", rel.basis, rel.sources, basis_may_be_empty=rel.cause == "supply"
        )
    )
    issues.extend(_check_status(family, rel))
    return issues


def _check_states(inst: Instance) -> list[Issue]:
    """`requires` against relations, both ways (ADR 0001, "Согласованность")."""
    fam = inst.family
    ext_in = {e.id for e in fam.external_inputs}
    produces: dict[str, tuple[str, ...]] = {o.id: o.produces for o in inst.operations}
    produces.update({e.id: e.produces for e in fam.external_inputs})
    # consumers: active operations (with requires_if resolved) and external outputs
    requires: dict[str, tuple[str, ...]] = {o.id: inst.required_states(o) for o in inst.operations}
    requires.update({e.id: e.requires for e in fam.external_outputs})

    producers: dict[str, list[str]] = defaultdict(list)
    for node, states in produces.items():
        for state in states:
            producers[state].append(node)

    issues: list[Issue] = []
    for state, who in sorted(producers.items()):
        if len(who) > 1:
            issues.append(
                _error("multiple_producers", f"state {state} is produced by {who}", state)
            )

    needed: set[Pair] = set()
    for node, states in requires.items():
        for state in states:
            if state not in producers:
                issues.append(_error("no_producer", f"{node} requires {state}: no producer", node))
            needed.update((p, node) for p in producers.get(state, []))

    def is_interface(pair: Pair) -> bool:
        return pair[0] in ext_in or pair[1] not in produces

    internal = {(r.src, r.dst) for r in inst.relations if r.type in HARD_TYPES}
    interface = {(r.src, r.dst) for r in inst.interface_relations}
    needed_internal = {p for p in needed if not is_interface(p)}
    needed_interface = {p for p in needed if is_interface(p)}

    for src, dst in sorted(needed_internal - internal):
        issues.append(
            _error(
                "requires_without_relation", f"{dst} requires a result of {src}", f"{src}->{dst}"
            )
        )
    for src, dst in sorted(needed_interface - interface):
        issues.append(
            _error("requires_without_interface", f"{dst} needs input {src}", f"{src}->{dst}")
        )
    for src, dst in sorted(internal - needed_internal):
        issues.append(
            _error(
                "relation_without_requires",
                f"{src}->{dst} has no requires behind it",
                f"{src}->{dst}",
            )
        )
    for src, dst in sorted(interface - needed_interface):
        issues.append(
            _error(
                "interface_without_requires",
                f"{src}->{dst} has no requires behind it",
                f"{src}->{dst}",
            )
        )
    return issues


def _check_coverage(inst: Instance) -> list[Issue]:
    touched = {n for r in (*inst.relations, *inst.interface_relations) for n in (r.src, r.dst)}
    return [
        _error("uncovered_operation", f"operation {op.id} has no relation", op.id)
        for op in inst.operations
        if op.id not in touched
    ]


def check_instance(inst: Instance) -> Report:
    non_relations = [
        nr.pair for nr in inst.family.non_relations if set(nr.pair) <= inst.operation_ids
    ]
    graph_report = check_graph(
        reference_graph(inst), inst.operation_ids, non_relations, provenance=False
    )
    # `isolated_operation` is replaced by the stricter coverage error below.
    issues = [i for i in graph_report.issues if i.code != "isolated_operation"]
    issues.extend(_check_states(inst))
    issues.extend(_check_coverage(inst))
    return Report(tuple(issues))


def check_family(family: Family) -> Report:
    """Validate the reference corpus: static rules once, instance rules for every instance."""
    issues = _check_static(family)
    merged: dict[tuple[Severity, str, str, str], list[str]] = {}
    first: dict[tuple[Severity, str, str, str], Issue] = {}
    for inst in enumerate_instances(family):
        for issue in check_instance(inst).issues:
            key = (issue.severity, issue.code, issue.subject, issue.message)
            merged.setdefault(key, []).append(inst.label)
            first.setdefault(key, issue)
    for key, labels in merged.items():
        i = first[key]
        issues.append(Issue(i.severity, i.code, i.message, i.subject, ", ".join(labels)))
    return Report(tuple(issues))
