"""
Parse a model's JSON text into a graph for the verifier.

The parser does not judge the answer: an edge to an unknown operation, a bad type or a cycle goes
to the verifier, which reports it. The parser only drops what cannot be an edge at all and records
why in `problems`. `operations` is the package vocabulary: the model does not list it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nir_project.baselines.package import Package
from nir_project.verifier.model import Edge, Graph, Pair

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


@dataclass(frozen=True)
class ParsedAnswer:
    graph: Graph
    reasons: Mapping[Pair, str]
    problems: tuple[str, ...]
    raw: Mapping[str, Any] | None  # the decoded JSON, None if it did not decode

    def as_graph_dict(self) -> dict[str, Any]:
        """The answer in the verifier's file format (`python -m nir_project.verifier graph`)."""
        return {
            "operations": sorted(self.graph.operations),
            "relations": [
                {
                    "id": e.id,
                    "from": e.src,
                    "to": e.dst,
                    "type": e.type,
                    "basis": list(e.basis),
                    "reason": self.reasons.get((e.src, e.dst), ""),
                }
                for e in self.graph.edges
            ],
        }


def _strings(value: Any) -> tuple[tuple[str, ...], bool]:
    """A list of strings; a lone string becomes a one-item list. Second value: was it coerced."""
    if value is None:
        return (), False
    if isinstance(value, str):
        return (value,), True
    if isinstance(value, list):
        return tuple(str(v) for v in value), False
    return (str(value),), True


def parse_answer(text: str, package: Package) -> ParsedAnswer:
    vocab = package.operation_ids
    empty = Graph(operations=vocab, edges=())
    match = _FENCE.match(text)
    body = match.group(1) if match else text
    if not body.strip():
        return ParsedAnswer(empty, {}, ("empty_answer",), None)
    try:
        raw = json.loads(body)
    except json.JSONDecodeError as exc:
        return ParsedAnswer(empty, {}, (f"invalid_json: {exc.msg}",), None)
    if not isinstance(raw, dict) or not isinstance(raw.get("relations"), list):
        return ParsedAnswer(
            empty, {}, ("no_relations_list",), raw if isinstance(raw, dict) else None
        )

    edges: list[Edge] = []
    reasons: dict[Pair, str] = {}
    problems: list[str] = []
    for i, item in enumerate(raw["relations"]):
        if not isinstance(item, dict) or not item.get("from") or not item.get("to"):
            problems.append(f"relations[{i}]: not an edge")
            continue
        basis, coerced = _strings(item.get("basis"))
        if coerced:
            problems.append(f"relations[{i}]: basis is not a list")
        src, dst = str(item["from"]).strip(), str(item["to"]).strip()
        edges.append(
            Edge(
                src=src,
                dst=dst,
                type=str(item.get("type", "mandatory")).strip(),
                basis=basis,
                id=f"M-{i + 1:03d}",
            )
        )
        reasons.setdefault((src, dst), str(item.get("reason", "")))
    return ParsedAnswer(Graph(operations=vocab, edges=tuple(edges)), reasons, tuple(problems), raw)
