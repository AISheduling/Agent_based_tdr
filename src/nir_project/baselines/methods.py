"""What every baseline looks like to the runner: a task in, a graph with cost and artifacts out.

A method sees the whole `Task`, but what it may use is its own contract: the LLM baselines read
only `package`; a classical baseline that copies the template "as written" may read `meta`
(the raw family file with `schedule_row` and `ttk`). The reference relations are never passed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from nir_project.baselines.package import Package
from nir_project.verifier.instances import Instance
from nir_project.verifier.model import Graph


@dataclass(frozen=True)
class Task:
    instance: Instance
    package: Package
    meta: Mapping[str, Any]


@dataclass(frozen=True)
class MethodOutput:
    graph: Graph
    cost: Mapping[str, float] = field(default_factory=dict)  # tokens, seconds, calls
    artifacts: Mapping[str, str] = field(default_factory=dict)  # file name -> text
    problems: tuple[str, ...] = ()  # what the method dropped while building the graph


class Method(Protocol):
    name: str
    model_family: str  # MLflow tag: llm-structured, template, ...

    def params(self) -> Mapping[str, Any]:
        """Everything that changes the output; logged as run parameters."""
        ...

    def predict(self, task: Task) -> MethodOutput: ...
