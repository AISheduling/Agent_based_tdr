"""Data model of a template family and of a relation graph (ADR 0001).

The loader is deliberately lenient about values (an unknown `type` or `basis` is kept as a string)
so that `checks.py` can report it as an issue instead of the loader crashing. It fails with
`FormatError` only when the shape is wrong (a missing required key, a string where a list is
expected, invalid YAML), because then there is nothing to check.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Values fixed by ADR 0001.
TYPES = ("mandatory", "conditional", "preferred", "unknown")
HARD_TYPES = ("mandatory", "conditional")
BASES = ("object_state", "result_consumption", "zone_access", "tech_card", "statistical_repeat")
STATUSES = ("source_checked", "needs_check", "expert_needed")
CAUSES = ("interface", "supply")
EXPERT_SOURCE = "SRC.EXPERT"

Pair = tuple[str, str]


class FormatError(ValueError):
    """A family or graph file lacks a required key or has the wrong shape."""


@dataclass(frozen=True)
class SourceRef:
    source: str
    locator: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class Relation:
    """One precedence relation `src -> dst` (finish-to-start), with type, basis and provenance."""

    id: str
    src: str
    dst: str
    type: str
    basis: tuple[str, ...] = ()
    sources: tuple[SourceRef, ...] = ()
    status: str | None = None
    condition: str | None = None
    cause: str | None = None  # interface relations only


@dataclass(frozen=True)
class Operation:
    id: str
    name: str = ""
    kind: str = ""
    requires: tuple[str, ...] = ()
    requires_if: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    produces: tuple[str, ...] = ()


@dataclass(frozen=True)
class External:
    """External input (produces states) or external output (requires states) of a family."""

    id: str
    produces: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    cause: str | None = None


@dataclass(frozen=True)
class NonRelation:
    pair: Pair
    reason: str = ""


@dataclass(frozen=True)
class Source:
    id: str
    role: str = ""


@dataclass(frozen=True)
class Family:
    name: str
    parameters: tuple[str, ...]
    optional_operations: Mapping[str, str]
    operations: tuple[Operation, ...]
    external_inputs: tuple[External, ...]
    external_outputs: tuple[External, ...]
    relations: tuple[Relation, ...]
    interface_relations: tuple[Relation, ...]
    non_relations: tuple[NonRelation, ...]
    sources: Mapping[str, Source] | None = None  # None: registry not supplied, skip source checks


@dataclass(frozen=True)
class Edge:
    """An edge of a graph under verification (a reference instance or a model answer)."""

    src: str
    dst: str
    type: str = "mandatory"
    basis: tuple[str, ...] = ()
    sources: tuple[SourceRef, ...] = ()
    id: str | None = None


@dataclass(frozen=True)
class Graph:
    """Operations plus edges. `operations` is the vocabulary the graph claims to cover."""

    operations: frozenset[str]
    edges: tuple[Edge, ...]


def _req(raw: Any, key: str, where: str) -> Any:
    if not isinstance(raw, Mapping):
        raise FormatError(f"{where}: expected a mapping, got {type(raw).__name__}")
    if key not in raw:
        raise FormatError(f"{where}: missing required key '{key}'")
    return raw[key]


def _list(value: Any, where: str) -> list[Any]:
    """A YAML list; a missing value is an empty list."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise FormatError(f"{where}: expected a list, got {type(value).__name__} {value!r}")
    return value


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    """A YAML mapping; a missing value is an empty mapping."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise FormatError(f"{where}: expected a mapping, got {type(value).__name__} {value!r}")
    return value


def _tuple(value: Any, where: str) -> tuple[str, ...]:
    """A list of strings. A bare string is rejected: iterating it would yield its letters."""
    return tuple(str(v) for v in _list(value, where))


def _opt_str(raw: Mapping[str, Any], key: str) -> str | None:
    return None if raw.get(key) is None else str(raw[key])


def _source_refs(value: Any, where: str) -> tuple[SourceRef, ...]:
    return tuple(
        SourceRef(
            source=str(_req(item, "source", where)),
            locator=_opt_str(item, "locator"),
            note=_opt_str(item, "note"),
        )
        for item in _list(value, where)
    )


def _relation(raw: Any, *, interface: bool) -> Relation:
    rid = str(_req(raw, "id", "relation"))
    return Relation(
        id=rid,
        src=str(_req(raw, "from", rid)),
        dst=str(_req(raw, "to", rid)),
        type="mandatory" if interface else str(_req(raw, "type", rid)),
        basis=_tuple(raw.get("basis"), f"{rid}.basis"),
        sources=_source_refs(raw.get("sources"), f"{rid}.sources"),
        status=_opt_str(raw, "status"),
        condition=_opt_str(raw, "condition"),
        cause=_opt_str(raw, "cause"),
    )


def _operation(raw: Any) -> Operation:
    oid = str(_req(raw, "id", "operation"))
    requires_if = _mapping(raw.get("requires_if"), f"{oid}.requires_if")
    return Operation(
        id=oid,
        name=str(raw.get("name", "")),
        kind=str(raw.get("kind", "")),
        requires=_tuple(raw.get("requires"), f"{oid}.requires"),
        requires_if={str(p): _tuple(s, f"{oid}.requires_if.{p}") for p, s in requires_if.items()},
        produces=_tuple(raw.get("produces"), f"{oid}.produces"),
    )


def _external(raw: Any) -> External:
    eid = str(_req(raw, "id", "external"))
    return External(
        id=eid,
        produces=_tuple(raw.get("produces"), f"{eid}.produces"),
        requires=_tuple(raw.get("requires"), f"{eid}.requires"),
        cause=_opt_str(raw, "cause"),
    )


def parse_sources(raw: Mapping[str, Any]) -> dict[str, Source]:
    """Parse the source registry (`data/sources.yaml`)."""
    registry: dict[str, Source] = {}
    for item in _list(_req(raw, "sources", "sources.yaml"), "sources.yaml: sources"):
        sid = str(_req(item, "id", "sources.yaml: sources[]"))
        registry[sid] = Source(id=sid, role=str(item.get("role", "")))
    return registry


def parse_family(
    name: str,
    ops_raw: Mapping[str, Any],
    rel_raw: Mapping[str, Any],
    sources: Mapping[str, Source] | None = None,
) -> Family:
    """Build a Family from the two parsed YAML documents."""
    non_relations: list[NonRelation] = []
    for item in _list(rel_raw.get("non_relations"), f"{name}: non_relations"):
        pair = _tuple(_req(item, "pair", "non_relations[]"), "non_relations[].pair")
        if len(pair) != 2:
            raise FormatError(f"non_relations[]: pair must have two operations, got {pair}")
        non_relations.append(NonRelation((pair[0], pair[1]), str(item.get("reason", ""))))
    optional = _mapping(ops_raw.get("optional_operations"), f"{name}: optional_operations")
    return Family(
        name=name,
        parameters=tuple(
            str(p) for p in _mapping(ops_raw.get("parameters"), f"{name}: parameters")
        ),
        optional_operations={str(o): str(p) for o, p in optional.items()},
        operations=tuple(
            _operation(o) for o in _list(_req(ops_raw, "operations", name), f"{name}: operations")
        ),
        external_inputs=tuple(
            _external(e) for e in _list(ops_raw.get("external_inputs"), f"{name}: external_inputs")
        ),
        external_outputs=tuple(
            _external(e)
            for e in _list(ops_raw.get("external_outputs"), f"{name}: external_outputs")
        ),
        relations=tuple(
            _relation(r, interface=False)
            for r in _list(rel_raw.get("relations"), f"{name}: relations")
        ),
        interface_relations=tuple(
            _relation(r, interface=True)
            for r in _list(rel_raw.get("interface_relations"), f"{name}: interface_relations")
        ),
        non_relations=tuple(non_relations),
        sources=sources,
    )


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise FormatError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise FormatError(f"{path}: top level must be a mapping")
    return data


def load_sources(path: Path) -> dict[str, Source]:
    return parse_sources(_load_yaml(path))


def load_family(family_dir: Path, sources_path: Path | None = None) -> Family:
    """Load `<dir>/<name>.yaml` and `<dir>/<name>_relations.yaml`; name = directory name."""
    name = family_dir.name
    sources = load_sources(sources_path) if sources_path is not None else None
    return parse_family(
        name,
        _load_yaml(family_dir / f"{name}.yaml"),
        _load_yaml(family_dir / f"{name}_relations.yaml"),
        sources,
    )


def parse_graph(raw: Mapping[str, Any]) -> Graph:
    """Parse a model answer: `{operations: [...], relations: [{from, to, type, ...}]}`."""
    edges: list[Edge] = []
    for i, item in enumerate(_list(_req(raw, "relations", "graph"), "graph.relations")):
        where = f"graph.relations[{i}]"
        edges.append(
            Edge(
                src=str(_req(item, "from", where)),
                dst=str(_req(item, "to", where)),
                type=str(item.get("type", "mandatory")),
                basis=_tuple(item.get("basis"), f"{where}.basis"),
                sources=_source_refs(item.get("sources"), f"{where}.sources"),
                id=_opt_str(item, "id"),
            )
        )
    operations = _tuple(_req(raw, "operations", "graph"), "graph.operations")
    return Graph(operations=frozenset(operations), edges=tuple(edges))


def load_graph(path: Path) -> Graph:
    return parse_graph(_load_yaml(path))
