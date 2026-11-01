"""Verifier tests: unit tests on a toy family, negative tests on the real corpus."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Collection
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from nir_project.verifier import (
    FP_BUDGET,
    Edge,
    Family,
    FormatError,
    Graph,
    Report,
    check_family,
    check_graph,
    enumerate_instances,
    find_cycle,
    load_family,
    load_graph,
    load_sources,
    make_instance,
    parse_family,
    parse_graph,
    reference_graph,
    transitive_closure,
    verify_graph,
)
from nir_project.verifier.model import SourceRef

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ROOT / "data" / "families"
SOURCES = ROOT / "data" / "sources.yaml"
FAMILY_NAMES = ["flat_roof", "monolithic_floor"]


def src(source: str = "SRC.A", locator: str | None = "п. 1") -> dict[str, Any]:
    return {"source": source, "locator": locator}


def rel(rid: str, a: str, b: str, **kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": rid,
        "from": a,
        "to": b,
        "type": "mandatory",
        "basis": ["result_consumption"],
        "sources": [src()],
        "status": "source_checked",
    }
    return base | kw


def toy_ops() -> list[dict[str, Any]]:
    return [
        {"id": "A", "requires": [], "produces": ["a"]},
        {"id": "B", "requires": ["a"], "produces": ["b"]},
        {"id": "C", "requires": ["b"], "produces": ["c"]},
        {"id": "D", "requires": ["c"], "produces": ["d"]},
        {"id": "X", "requires": [], "produces": ["x"]},
        {"id": "Y", "requires": ["x"], "produces": ["y"]},
    ]


def toy_relations() -> list[dict[str, Any]]:
    return [
        rel("T-R1", "A", "B"),
        rel("T-R2", "B", "C"),
        rel("T-R3", "C", "D", type="conditional", condition="extra"),
        rel("T-R4", "X", "Y"),
    ]


def toy_family(**overrides: Any) -> Family:
    """A -> B -> C, optional D (needs C) when `extra`, and an independent pair A, X."""
    ops_raw: dict[str, Any] = {
        "parameters": {"extra": "add D"},
        "optional_operations": {"D": "extra"},
        "operations": toy_ops(),
        "external_inputs": [],
        "external_outputs": [],
    }
    rel_raw: dict[str, Any] = {
        "relations": toy_relations(),
        "interface_relations": [],
        "non_relations": [{"pair": ["A", "X"], "reason": "different crews"}],
    }
    ops_raw |= overrides.get("ops", {})
    rel_raw |= overrides.get("rel", {})
    return parse_family("toy", ops_raw, rel_raw)


def toy_reference(extra: bool = False) -> Graph:
    return reference_graph(make_instance(toy_family(), {"extra": extra}))


def answer(reference: Graph, *, drop: Collection[str] = (), add: tuple[Edge, ...] = ()) -> Graph:
    """The reference as a model answer, with edges (by id) removed and edges added."""
    kept = tuple(replace(e, id=None) for e in reference.edges if e.id not in drop)
    return Graph(reference.operations, kept + add)


def hard(a: str, b: str, basis: tuple[str, ...] = ("result_consumption",)) -> Edge:
    return Edge(a, b, "mandatory", basis, (SourceRef("SRC.A", "п. 1"),))


# --- closure and cycles --------------------------------------------------------------------------


def test_closure_chain() -> None:
    assert transitive_closure([("a", "b"), ("b", "c")]) == {("a", "b"), ("b", "c"), ("a", "c")}


def test_find_cycle() -> None:
    assert find_cycle([("a", "b"), ("b", "c")]) is None
    cycle = find_cycle([("a", "b"), ("b", "c"), ("c", "a")])
    assert cycle is not None and cycle[0] == cycle[-1]
    assert find_cycle([("a", "a")]) is not None


# --- the real corpus -----------------------------------------------------------------------------


@pytest.mark.parametrize("name", FAMILY_NAMES)
def test_real_family_is_clean(name: str) -> None:
    report = check_family(load_family(FAMILIES / name, SOURCES))
    assert report.ok, [(i.code, i.subject, i.message) for i in report.errors]
    assert not report.warnings


def test_instances_of_flat_roof() -> None:
    family = load_family(FAMILIES / "flat_roof", SOURCES)
    instances = enumerate_instances(family)
    assert len(instances) == 2 ** len(family.parameters)
    sizes = {i.label: len(i.operations) for i in instances}
    assert len(set(sizes.values())) > 1  # optional operations change the instance


@pytest.mark.parametrize("name", FAMILY_NAMES)
def test_reference_scores_perfectly_against_itself(name: str) -> None:
    family = load_family(FAMILIES / name, SOURCES)
    for inst in enumerate_instances(family):
        ref = reference_graph(inst)
        pairs = [nr.pair for nr in family.non_relations if set(nr.pair) <= inst.operation_ids]
        verdict = verify_graph(ref, answer(ref), pairs)
        assert verdict.ok, [(i.code, i.subject) for i in verdict.report.errors]
        assert (verdict.metrics.fp, verdict.metrics.fn) == (0, 0)
        assert verdict.metrics.recall == 1.0


# --- negative tests from the task: each defect must be caught ------------------------------------


def _real_instance() -> tuple[Family, Graph, list[tuple[str, str]]]:
    family = load_family(FAMILIES / "flat_roof", SOURCES)
    inst = make_instance(family, {"screed": True, "bitumen_vapor_barrier": True})
    pairs = [nr.pair for nr in family.non_relations if set(nr.pair) <= inst.operation_ids]
    return family, reference_graph(inst), pairs


def test_negative_cycle() -> None:
    _, ref, pairs = _real_instance()
    first = next(e for e in ref.edges if e.type == "mandatory")
    bad = answer(ref, add=(hard(first.dst, first.src),))
    verdict = verify_graph(ref, bad, pairs)
    assert not verdict.ok
    assert "cycle" in verdict.report.codes("error")


def test_negative_missing_operation() -> None:
    _, ref, pairs = _real_instance()
    victim = "ROOF.BASE_CLEAN"
    bad = Graph(
        ref.operations - {victim},
        tuple(replace(e, id=None) for e in ref.edges if victim not in (e.src, e.dst)),
    )
    verdict = verify_graph(ref, bad, pairs)
    assert not verdict.ok
    assert "uncovered_operation" in verdict.report.codes("error")
    assert verdict.metrics.fn > 0


def test_negative_removed_necessary_relation() -> None:
    _, ref, pairs = _real_instance()
    checked = 0
    for e in ref.edges:
        if e.type != "mandatory":
            continue
        bad = answer(ref, drop={e.id or ""})
        if transitive_closure([(x.src, x.dst) for x in bad.edges if x.type == "mandatory"]) == (
            transitive_closure([(x.src, x.dst) for x in ref.edges if x.type == "mandatory"])
        ):
            continue  # redundant edge: its removal is not a defect
        verdict = verify_graph(ref, bad, pairs)
        assert verdict.metrics.fn > 0 and verdict.metrics.fp == 0, e.id
        assert verdict.metrics.recall < 1.0, e.id
        checked += 1
    assert checked > 10


def test_negative_extra_historical_relation() -> None:
    family, ref, pairs = _real_instance()
    a, b = family.non_relations[0].pair
    historical = hard(a, b, basis=("statistical_repeat",))
    verdict = verify_graph(ref, answer(ref, add=(historical,)), pairs)
    assert not verdict.ok
    assert {"weak_basis", "non_relation_violated"} <= verdict.report.codes("error")
    assert verdict.metrics.fp >= 1 and verdict.metrics.fp_on_non_relation >= 1
    # One false pair among ~200 stays inside the 2% budget: the structural checks above are
    # what catch such an edge, the budget only limits how many slip through.
    assert verdict.metrics.fp_budget_ok


# --- comparison semantics ------------------------------------------------------------------------


def test_equivalent_edge_sets_score_the_same() -> None:
    ref = toy_reference()
    shortcut = answer(ref, add=(hard("A", "C"),))  # transitively redundant
    assert verify_graph(ref, shortcut).metrics == verify_graph(ref, answer(ref)).metrics
    assert verify_graph(ref, shortcut).metrics.fp == 0


def test_tp_fp_fn_counts() -> None:
    ref = toy_reference()  # closure: A<B, B<C, A<C, X<Y
    bad = answer(ref, drop={"T-R2"}, add=(hard("A", "Y"),))  # keeps A<B, X<Y; loses B<C, A<C
    m = verify_graph(ref, bad).metrics
    assert (m.tp, m.fp, m.fn) == (2, 1, 2)
    assert m.recall == 0.5 and m.fp_rate == pytest.approx(1 / 3)


def test_unknown_is_not_a_false_positive() -> None:
    ref = toy_reference()
    unknown = Edge("A", "Y", "unknown")
    m = verify_graph(ref, answer(ref, add=(unknown,))).metrics
    assert m.fp == 0 and m.unknown_pairs == 1
    abstain = Edge("B", "C", "unknown")
    m = verify_graph(ref, answer(ref, drop={"T-R2"}, add=(abstain,))).metrics
    assert m.fn == 2 and m.fn_declared_unknown == 1 and m.fn_silent == 1


def test_fp_budget_flag() -> None:
    ref = toy_reference()
    assert verify_graph(ref, answer(ref)).metrics.fp_budget_ok
    assert not verify_graph(ref, answer(ref, add=(hard("A", "Y"),))).metrics.fp_budget_ok
    assert FP_BUDGET == 0.02


def test_new_operation_is_rejected() -> None:
    ref = toy_reference()
    bad = Graph(ref.operations | {"NEW"}, answer(ref).edges)
    assert "unknown_operation" in verify_graph(ref, bad).report.codes("error")


def test_edge_provenance_is_required() -> None:
    ref = toy_reference()
    no_basis = Edge("A", "Y", "mandatory", (), (SourceRef("SRC.A", "п. 1"),))
    no_source = Edge("A", "Y", "mandatory", ("object_state",), ())
    assert "no_basis" in check_graph(answer(ref, add=(no_basis,)), ref.operations).codes()
    assert "no_source" in check_graph(answer(ref, add=(no_source,)), ref.operations).codes()
    weak_ok = Edge("A", "Y", "preferred", ("statistical_repeat",))
    assert check_graph(answer(ref, add=(weak_ok,)), ref.operations).ok


# --- reference checks on a toy family ------------------------------------------------------------


def test_toy_family_is_clean() -> None:
    assert check_family(toy_family()).ok


def test_conditional_edge_depends_on_instance() -> None:
    fam = toy_family()
    off = reference_graph(make_instance(fam, {"extra": False}))
    on = reference_graph(make_instance(fam, {"extra": True}))
    assert "D" not in off.operations and "D" in on.operations
    assert ("C", "D") in {(e.src, e.dst) for e in on.edges}


def test_family_cycle_is_reported() -> None:
    fam = toy_family(rel={"relations": [rel("T-R1", "A", "B"), rel("T-R2", "B", "A")]})
    assert "cycle" in check_family(fam).codes("error")


def test_family_relation_without_basis() -> None:
    bad = rel("T-R1", "A", "B", basis=[])
    fam = toy_family(rel={"relations": [bad, rel("T-R2", "B", "C"), rel("T-R4", "X", "Y")]})
    assert "no_basis" in check_family(fam).codes("error")


def test_family_requires_without_relation() -> None:
    fam = toy_family(rel={"relations": [rel("T-R1", "A", "B"), rel("T-R4", "X", "Y")]})
    assert "requires_without_relation" in check_family(fam).codes("error")


def test_family_relation_without_requires() -> None:
    fam = toy_family(rel={"relations": [*toy_relations(), rel("T-R5", "A", "C")]})
    assert "relation_without_requires" in check_family(fam).codes("error")


def test_family_uncovered_operation() -> None:
    lonely = {"id": "Z", "requires": [], "produces": ["z"]}
    fam = toy_family(ops={"operations": [*toy_ops(), lonely]})
    assert "uncovered_operation" in check_family(fam).codes("error")


def test_family_non_relation_violated() -> None:
    fam = toy_family(rel={"non_relations": [{"pair": ["A", "C"], "reason": "claimed independent"}]})
    assert "non_relation_violated" in check_family(fam).codes("error")


# --- loader and instance fixes --------------------------------------------------------------------


def test_string_instead_of_list_is_a_format_error() -> None:
    bad = rel("T-R1", "A", "B", basis="result_consumption")
    with pytest.raises(FormatError, match=r"T-R1\.basis: expected a list"):
        toy_family(rel={"relations": [bad]})
    with pytest.raises(FormatError, match=r"graph\.operations: expected a list"):
        parse_graph({"operations": "ABC", "relations": []})


def test_non_mapping_entry_is_a_format_error() -> None:
    with pytest.raises(FormatError, match="expected a mapping"):
        toy_family(rel={"relations": [["A", "B"]]})


def test_invalid_yaml_is_a_format_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("operations: [A\nrelations: [", encoding="utf-8")
    with pytest.raises(FormatError, match="invalid YAML"):
        load_graph(path)


def test_unknown_parameter_is_rejected() -> None:
    with pytest.raises(ValueError, match="extr"):
        make_instance(toy_family(), {"extr": True})


def test_metrics_ignore_edges_the_report_rejects() -> None:
    ref = toy_reference(extra=True)  # closure includes A<D, B<D, C<D
    bad = Graph(ref.operations - {"D"}, answer(ref).edges)  # edge C->D kept, D not listed
    verdict = verify_graph(ref, bad)
    assert "edge_unknown_operation" in verdict.report.codes("error")
    assert verdict.metrics.fn == 3  # the rejected edge earns nothing


def test_unknown_pair_also_asserted_is_not_an_abstention() -> None:
    ref = toy_reference()
    m = verify_graph(ref, answer(ref, add=(Edge("A", "B", "unknown"),))).metrics
    assert m.unknown_pairs == 0


# --- every issue code fires on a matching defect -------------------------------------------------

Mutator = Callable[[dict[str, Any], dict[str, Any]], object]
DOC = [{"source": "SRC.SP71", "locator": "п. 5.1.5"}]


def _raw(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    import yaml

    folder = FAMILIES / name
    with (folder / f"{name}.yaml").open(encoding="utf-8") as fh:
        ops = yaml.safe_load(fh)
    with (folder / f"{name}_relations.yaml").open(encoding="utf-8") as fh:
        rels = yaml.safe_load(fh)
    return ops, rels


def _rel(r: dict[str, Any], rid: str) -> dict[str, Any]:
    return next(x for x in r["relations"] if x["id"] == rid)


def _op(o: dict[str, Any], oid: str) -> dict[str, Any]:
    return next(x for x in o["operations"] if x["id"] == oid)


def _new_rel(rid: str, a: str, b: str, **kw: Any) -> dict[str, Any]:
    base = {"id": rid, "from": a, "to": b, "type": "mandatory", "basis": ["object_state"]}
    return base | {"sources": DOC, "status": "source_checked"} | kw


# Each entry breaks the real flat_roof corpus in exactly the way its issue code describes.
# F2-R01: ROOF.PREP_ACCEPT -> ROOF.BASE_CLEAN; F2-R02: conditional on bitumen_vapor_barrier.
REFERENCE_DEFECTS: dict[str, Mutator] = {
    "duplicate_id": lambda o, r: r["relations"].append(copy.deepcopy(r["relations"][0])),
    "unknown_operation": lambda o, r: _rel(r, "F2-R01").update({"from": "ROOF.NOPE"}),
    "invalid_type": lambda o, r: _rel(r, "F2-R01").update(type="mandatry"),
    "invalid_status": lambda o, r: _rel(r, "F2-R01").update(status="ok"),
    "invalid_cause": lambda o, r: r["interface_relations"][0].update(cause="weather"),
    "missing_condition": lambda o, r: _rel(r, "F2-R02").pop("condition"),
    "unexpected_condition": lambda o, r: _rel(r, "F2-R01").update(condition="screed"),
    "unknown_parameter": lambda o, r: _rel(r, "F2-R02").update(condition="scred"),
    "invalid_basis": lambda o, r: _rel(r, "F2-R01").update(basis=["object_stat"]),
    "no_basis": lambda o, r: _rel(r, "F2-R01").update(basis=[]),
    "weak_basis": lambda o, r: _rel(r, "F2-R01").update(basis=["statistical_repeat"]),
    "no_source": lambda o, r: _rel(r, "F2-R01").update(sources=[]),
    "unknown_source": lambda o, r: _rel(r, "F2-R01")["sources"][0].update(source="SRC.NOPE"),
    "no_locator": lambda o, r: _rel(r, "F2-R01")["sources"][0].update(locator=None),
    "expert_locator": lambda o, r: _rel(r, "F2-R01").update(
        sources=[{"source": "SRC.EXPERT", "locator": "п. 1"}], status="expert_needed"
    ),
    "expert_unconfirmed": lambda o, r: _rel(r, "F2-R01").update(
        sources=[{"source": "SRC.EXPERT", "locator": None}]
    ),
    "no_reason": lambda o, r: r["non_relations"][0].update(reason=""),
    "invalid_pair": lambda o, r: r["non_relations"][0].update(pair=["ROOF.LEVEL", "ROOF.LEVEL"]),
    "non_relation_violated": lambda o, r: r["non_relations"][0].update(
        pair=["ROOF.PREP_ACCEPT", "ROOF.BASE_CLEAN"]
    ),
    "cycle": lambda o, r: r["relations"].append(
        _new_rel("F2-R99", "ROOF.BASE_CLEAN", "ROOF.PREP_ACCEPT")
    ),
    "preferred_contradicts_mandatory": lambda o, r: r["relations"].append(
        _new_rel("F2-R98", "ROOF.BASE_CLEAN", "ROOF.PREP_ACCEPT", type="preferred")
    ),
    "relation_without_requires": lambda o, r: r["relations"].append(
        _new_rel("F2-R97", "ROOF.LEVEL", "ROOF.PARAPET_PLASTER")
    ),
    "requires_without_relation": lambda o, r: r["relations"].remove(_rel(r, "F2-R01")),
    "multiple_producers": lambda o, r: _op(o, "ROOF.LEVEL")["produces"].append(
        "roof_base.clean_dry"
    ),
    "no_producer": lambda o, r: _op(o, "ROOF.LEVEL")["requires"].append("magic.state"),
    "uncovered_operation": lambda o, r: o["operations"].append(
        {"id": "ROOF.LONELY", "produces": ["lonely.done"]}
    ),
    "requires_without_interface": lambda o, r: r.update(
        interface_relations=r["interface_relations"][1:]
    ),
    "interface_without_requires": lambda o, r: r["interface_relations"].append(
        _new_rel("F2-X99", "EXT.ROOF_MATERIALS_DELIVERED", "ROOF.LEVEL", cause="supply")
    ),
    "invalid_interface": lambda o, r: r["interface_relations"].append(
        _new_rel("F2-X98", "ROOF.LEVEL", "ROOF.INSUL", cause="interface")
    ),
}


def _answer_report(edit: Callable[[Graph], Graph]) -> Report:
    family = load_family(FAMILIES / "flat_roof", SOURCES)
    ref = reference_graph(make_instance(family, {}))
    return check_graph(edit(answer(ref)), ref.operations)


# Codes that only an answer can produce.
ANSWER_DEFECTS: dict[str, Callable[[Graph], Graph]] = {
    "edge_unknown_operation": lambda g: Graph(
        g.operations, (*g.edges, hard("ROOF.LEVEL", "ROOF.NOPE"))
    ),
    "isolated_operation": lambda g: Graph(
        g.operations, tuple(e for e in g.edges if "ROOF.ACCEPT" not in (e.src, e.dst))
    ),
}


@pytest.mark.parametrize("code", sorted(REFERENCE_DEFECTS))
def test_reference_defect_is_caught(code: str) -> None:
    ops, rels = _raw("flat_roof")
    REFERENCE_DEFECTS[code](ops, rels)
    report = check_family(parse_family("flat_roof", ops, rels, load_sources(SOURCES)))
    assert code in report.codes(), sorted(report.codes())


@pytest.mark.parametrize("code", sorted(ANSWER_DEFECTS))
def test_answer_defect_is_caught(code: str) -> None:
    assert code in _answer_report(ANSWER_DEFECTS[code]).codes()


def test_every_issue_code_has_a_defect_case() -> None:
    """A new check without a test here fails this test."""
    source = (ROOT / "src" / "nir_project" / "verifier" / "checks.py").read_text(encoding="utf-8")
    codes = set(re.findall(r'_(?:error|warning)\(\s*"([a-z_]+)"', source))
    assert len(codes) > 25
    missing = codes - set(REFERENCE_DEFECTS) - set(ANSWER_DEFECTS)
    assert not missing, f"no defect case for {sorted(missing)}"


# --- command line --------------------------------------------------------------------------------


def _answer_file(path: Path, ref: Graph, extra: tuple[dict[str, Any], ...] = ()) -> Path:
    import yaml

    rels: list[dict[str, Any]] = [
        {
            "from": e.src,
            "to": e.dst,
            "type": e.type,
            "basis": list(e.basis),
            "sources": [{"source": s.source, "locator": s.locator} for s in e.sources],
        }
        for e in ref.edges
    ]
    data = {"operations": sorted(ref.operations), "relations": [*rels, *extra]}
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _flat_roof_ref() -> Graph:
    return reference_graph(make_instance(load_family(FAMILIES / "flat_roof", SOURCES), {}))


def _cli(*args: str) -> int:
    from nir_project.verifier.__main__ import main

    return main([*args, "--sources", str(SOURCES)])


def test_cli_family_ok(capsys: pytest.CaptureFixture[str]) -> None:
    assert _cli("family", *[str(FAMILIES / n) for n in FAMILY_NAMES]) == 0
    assert "0 error(s)" in capsys.readouterr().out


def test_cli_family_continues_after_a_bad_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _cli("family", str(tmp_path / "nope"), str(FAMILIES / "flat_roof")) == 2
    out = capsys.readouterr()
    assert "nope: cannot load" in out.err and "flat_roof: 0 error(s)" in out.out


def test_cli_graph_detects_cycle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ref = _flat_roof_ref()
    roof = str(FAMILIES / "flat_roof")
    assert _cli("graph", roof, str(_answer_file(tmp_path / "ok.yaml", ref))) == 0
    first = next(e for e in ref.edges if e.type == "mandatory")
    back = {"from": first.dst, "to": first.src, "basis": ["object_state"], "sources": DOC}
    assert _cli("graph", roof, str(_answer_file(tmp_path / "bad.yaml", ref, (back,)))) == 1
    assert "cycle" in capsys.readouterr().out


def test_cli_json_is_pure_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _answer_file(tmp_path / "ok.yaml", _flat_roof_ref())
    assert _cli("graph", str(FAMILIES / "flat_roof"), str(path), "--json") == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True and result["metrics"]["recall"] == 1.0 and result["issues"] == []


def test_cli_rejects_bad_params(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = str(_answer_file(tmp_path / "ok.yaml", _flat_roof_ref()))
    roof = str(FAMILIES / "flat_roof")
    assert _cli("graph", roof, path, "--param", "scred=1") == 2
    assert "scred" in capsys.readouterr().err
    assert _cli("graph", roof, path, "--param", "screed=maybe") == 2


def test_cli_broken_yaml_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("operations: [A\nrelations: [", encoding="utf-8")
    assert _cli("graph", str(FAMILIES / "flat_roof"), str(path)) == 2
    assert "invalid YAML" in capsys.readouterr().err


def test_cli_missing_explicit_registry_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    from nir_project.verifier.__main__ import main

    assert main(["family", str(FAMILIES / "flat_roof"), "--sources", "nope.yaml"]) == 2
    assert "nope.yaml not found" in capsys.readouterr().err
