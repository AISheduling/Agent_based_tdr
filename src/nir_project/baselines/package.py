"""Input package: everything a baseline may see about one family instance.

Left out on purpose:
- `requires`, `requires_if`, `produces`: the reference is built from them, so they are the answer;
- `ttk`, `schedule_row`, `zone`, `note`: source locators and the template's own order.

Operations are shuffled with the seed: the order in the family file is almost a topological order
(a chain in file order already scores recall 1.0 with precision 0.82-0.95).
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nir_project.verifier.instances import Instance


@dataclass(frozen=True)
class OperationView:
    id: str
    name: str
    kind: str


@dataclass(frozen=True)
class Condition:
    """A family parameter and its value in this instance."""

    name: str
    description: str
    value: bool


@dataclass(frozen=True)
class Package:
    label: str  # instance label, e.g. flat_roof[bitumen_vapor_barrier=0,...]
    family: str
    title: str
    scope: str
    conditions: tuple[Condition, ...]
    external_inputs: tuple[str, ...]  # human-readable names of what arrives from outside
    operations: tuple[OperationView, ...]
    seed: int

    @property
    def operation_ids(self) -> frozenset[str]:
        return frozenset(o.id for o in self.operations)


def load_family_meta(family_dir: Path) -> Mapping[str, Any]:
    """The raw `<family>.yaml`: descriptive fields the verifier's model does not keep."""
    path = family_dir / f"{family_dir.name}.yaml"
    with path.open(encoding="utf-8") as fh:
        raw: Mapping[str, Any] = yaml.safe_load(fh)
    return raw


def build_package(instance: Instance, meta: Mapping[str, Any], seed: int) -> Package:
    fam = instance.family
    ops = [OperationView(o.id, o.name, o.kind) for o in instance.operations]
    random.Random(f"{seed}:{instance.label}").shuffle(ops)

    descriptions: Mapping[str, Any] = meta.get("parameters") or {}
    conditions = tuple(
        Condition(p, str(descriptions.get(p, "")), bool(instance.params.get(p, False)))
        for p in fam.parameters
    )
    ext_names = {
        str(e["id"]): str(e.get("name", e["id"])) for e in meta.get("external_inputs") or ()
    }
    return Package(
        label=instance.label,
        family=fam.name,
        title=str(meta.get("name") or fam.name),
        scope=" ".join(str(meta.get("scope") or "").split()),
        conditions=conditions,
        external_inputs=tuple(ext_names.get(e.id, e.id) for e in fam.external_inputs),
        operations=tuple(ops),
        seed=seed,
    )
