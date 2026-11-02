"""Family instances: parameter values select which operations and relations are in force."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product

from nir_project.verifier.model import (
    Edge,
    Family,
    Graph,
    Operation,
    Relation,
)


@dataclass(frozen=True)
class Instance:
    family: Family
    params: Mapping[str, bool]
    operations: tuple[Operation, ...]
    relations: tuple[Relation, ...]
    interface_relations: tuple[Relation, ...]

    @property
    def label(self) -> str:
        if not self.params:
            return self.family.name
        flags = ",".join(f"{k}={'1' if v else '0'}" for k, v in sorted(self.params.items()))
        return f"{self.family.name}[{flags}]"

    @property
    def operation_ids(self) -> frozenset[str]:
        return frozenset(o.id for o in self.operations)

    def required_states(self, op: Operation) -> tuple[str, ...]:
        """`requires` plus the `requires_if` states whose parameter is true in this instance."""
        states = list(op.requires)
        for param, extra in op.requires_if.items():
            if self.params.get(param, False):
                states.extend(extra)
        return tuple(states)


def make_instance(family: Family, params: Mapping[str, bool]) -> Instance:
    """Select the operations and relations in force for the given parameter values.

    A parameter left out is false. An unknown name is an error: a typo would otherwise silently
    produce the instance with that parameter switched off.
    """
    unknown = sorted(set(params) - set(family.parameters))
    if unknown:
        known = ", ".join(sorted(family.parameters)) or "none"
        raise ValueError(f"{family.name}: unknown parameter(s) {unknown}; known: {known}")
    values = {p: bool(params.get(p, False)) for p in family.parameters}

    def op_active(op_id: str) -> bool:
        param = family.optional_operations.get(op_id)
        return param is None or values.get(param, False)

    ops = tuple(o for o in family.operations if op_active(o.id))
    active = {o.id for o in ops}

    def rel_active(rel: Relation) -> bool:
        if rel.condition is not None and not values.get(rel.condition, False):
            return False
        return rel.src in active and rel.dst in active

    ext_in = {e.id for e in family.external_inputs}
    ext_out = {e.id for e in family.external_outputs}

    def interface_active(rel: Relation) -> bool:
        return (rel.src in ext_in and rel.dst in active) or (
            rel.src in active and rel.dst in ext_out
        )

    return Instance(
        family=family,
        params=values,
        operations=ops,
        relations=tuple(r for r in family.relations if rel_active(r)),
        interface_relations=tuple(r for r in family.interface_relations if interface_active(r)),
    )


def enumerate_instances(family: Family) -> list[Instance]:
    """One instance per combination of the family's boolean parameters (one if it has none)."""
    names = sorted(family.parameters)
    return [
        make_instance(family, dict(zip(names, values, strict=True)))
        for values in product((False, True), repeat=len(names))
    ]


def reference_graph(instance: Instance) -> Graph:
    """The reference of an instance as a graph.

    Active `conditional` relations are in force, so they become `mandatory` edges here;
    `preferred` and `unknown` keep their type and stay out of the closure.
    """
    edges = tuple(
        Edge(
            src=r.src,
            dst=r.dst,
            type="mandatory" if r.type == "conditional" else r.type,
            basis=r.basis,
            sources=r.sources,
            id=r.id,
        )
        for r in instance.relations
    )
    return Graph(operations=instance.operation_ids, edges=edges)
