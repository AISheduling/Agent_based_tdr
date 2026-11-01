"""Minimal graph verifier (task 1, item 5): structure checks and closure comparison."""

from nir_project.verifier.checks import Issue, Report, check_family, check_graph, check_instance
from nir_project.verifier.closure import find_cycle, transitive_closure
from nir_project.verifier.compare import FP_BUDGET, Metrics, Verdict, compute_metrics, verify_graph
from nir_project.verifier.instances import (
    Instance,
    enumerate_instances,
    make_instance,
    reference_graph,
)
from nir_project.verifier.model import (
    Edge,
    Family,
    FormatError,
    Graph,
    load_family,
    load_graph,
    load_sources,
    parse_family,
    parse_graph,
)

__all__ = [
    "FP_BUDGET",
    "Edge",
    "Family",
    "FormatError",
    "Graph",
    "Instance",
    "Issue",
    "Metrics",
    "Report",
    "Verdict",
    "check_family",
    "check_graph",
    "check_instance",
    "compute_metrics",
    "enumerate_instances",
    "find_cycle",
    "load_family",
    "load_graph",
    "load_sources",
    "make_instance",
    "parse_family",
    "parse_graph",
    "reference_graph",
    "transitive_closure",
    "verify_graph",
]
