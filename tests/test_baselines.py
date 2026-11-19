"""L1 baseline without the network: package, prompt, parser and the run through the verifier."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from nir_project.baselines import Completion, Package, build_package, load_family_meta, predict
from nir_project.baselines.answer import parse_answer
from nir_project.baselines.prompt import build_messages, render_user_prompt
from nir_project.verifier import load_family, make_instance, reference_graph, verify_graph
from nir_project.verifier.instances import Instance

ROOT = Path(__file__).resolve().parents[1]
FLAT_ROOF = ROOT / "data" / "families" / "flat_roof"


class FakeClient:
    """Returns a fixed text and remembers what it was asked."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[Sequence[Mapping[str, str]]] = []

    def complete(self, messages: Sequence[Mapping[str, str]]) -> Completion:
        self.calls.append(messages)
        return Completion(text=self.text, model="fake")


def _instance(**params: bool) -> Instance:
    return make_instance(load_family(FLAT_ROOF), params)


def _package(seed: int = 42, **params: bool) -> Package:
    return build_package(_instance(**params), load_family_meta(FLAT_ROOF), seed)


def _answer(edges: Sequence[tuple[str, str]], type_: str = "mandatory") -> str:
    rel = [
        {"from": a, "to": b, "type": type_, "basis": ["result_consumption"], "reason": "x"}
        for a, b in edges
    ]
    return json.dumps({"relations": rel})


def test_package_covers_instance_operations() -> None:
    inst = _instance(screed=True)
    assert _package(screed=True).operation_ids == inst.operation_ids


def test_prompt_hides_states_and_locators() -> None:
    inst = _instance(screed=True, bitumen_vapor_barrier=True, expansion_joint=True)
    text = render_user_prompt(build_package(inst, load_family_meta(FLAT_ROOF), 42))
    states = {s for o in inst.operations for s in (*o.requires, *o.produces)}
    assert states, "the family must have states for this test to mean anything"
    assert not [s for s in states if s in text]
    assert "ttk" not in text and "schedule_row" not in text


def test_shuffle_is_deterministic_and_seeded() -> None:
    file_order = [o.id for o in _instance(screed=True).operations]
    order_a = [o.id for o in _package(seed=1, screed=True).operations]
    assert order_a == [o.id for o in _package(seed=1, screed=True).operations]
    assert order_a != [o.id for o in _package(seed=2, screed=True).operations]
    assert order_a != file_order


def test_messages_ask_for_json() -> None:
    messages = build_messages(_package())
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "json" in messages[0]["content"].lower()  # DeepSeek JSON mode needs the word


@pytest.mark.parametrize(
    ("text", "problem"),
    [("", "empty_answer"), ("not json", "invalid_json"), ('{"edges": []}', "no_relations_list")],
)
def test_unparsable_answer_gives_empty_graph(text: str, problem: str) -> None:
    pkg = _package()
    parsed = parse_answer(text, pkg)
    assert parsed.graph.edges == ()
    assert parsed.graph.operations == pkg.operation_ids
    assert parsed.problems[0].startswith(problem)


def test_parser_tolerates_fences_and_string_basis() -> None:
    text = (
        '```json\n{"relations": [{"from": "ROOF.INSUL", "to": "ROOF.INSUL_ACCEPT", '
        '"basis": "object_state"}, {"from": "ROOF.INSUL"}]}\n```'
    )
    parsed = parse_answer(text, _package())
    (edge,) = parsed.graph.edges
    assert (edge.src, edge.dst, edge.type, edge.basis) == (
        "ROOF.INSUL",
        "ROOF.INSUL_ACCEPT",
        "mandatory",
        ("object_state",),
    )
    assert len(parsed.problems) == 2  # coerced basis, dropped item without `to`


def test_invented_operation_goes_to_verifier() -> None:
    inst = _instance()
    pkg = build_package(inst, load_family_meta(FLAT_ROOF), 42)
    parsed = parse_answer(_answer([("ROOF.INSUL", "ROOF.MAGIC")]), pkg)
    verdict = verify_graph(reference_graph(inst), parsed.graph)
    assert "edge_unknown_operation" in verdict.report.codes()
    assert verdict.metrics.tp == 0


def test_reference_answer_scores_full_recall() -> None:
    inst = _instance(screed=True)
    ref = reference_graph(inst)
    edges = [(e.src, e.dst) for e in ref.edges if e.type == "mandatory"]
    client = FakeClient(_answer(edges))
    result = predict(build_package(inst, load_family_meta(FLAT_ROOF), 42), client)
    metrics = verify_graph(ref, result.answer.graph).metrics
    assert len(client.calls) == 1
    assert (metrics.recall, metrics.fp) == (1.0, 0)


def test_unknown_answer_is_not_a_false_positive() -> None:
    inst = _instance()
    ref = reference_graph(inst)
    edges = [(e.src, e.dst) for e in ref.edges if e.type == "mandatory"]
    parsed = parse_answer(_answer(edges, type_="unknown"), build_package(inst, {}, 42))
    metrics = verify_graph(ref, parsed.graph).metrics
    assert (metrics.tp, metrics.fp) == (0, 0)
    # unknown is counted per declared pair, not through the closure
    assert metrics.fn_declared_unknown == len(edges)
