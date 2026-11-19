"""Reachability, transitive closure and cycle search on small precedence graphs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from nir_project.verifier.model import Pair


def _adjacency(pairs: Iterable[Pair]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for src, dst in pairs:
        adj[src].add(dst)
    return adj


def transitive_closure(pairs: Iterable[Pair]) -> set[Pair]:
    """All pairs (a, b), a != b, with a path a -> ... -> b.

    Works on cyclic graphs too (nodes of a cycle reach each other); callers that need a DAG
    check `find_cycle` first.
    """
    adj = _adjacency(pairs)
    closure: set[Pair] = set()
    for start in list(adj):
        seen: set[str] = set()
        stack = list(adj[start])
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adj.get(node, ()))
        closure.update((start, node) for node in seen if node != start)
    return closure


def find_cycle(pairs: Iterable[Pair]) -> list[str] | None:
    """Return one cycle as a node list `[a, b, ..., a]`, or None if the graph is acyclic."""
    adj = _adjacency(pairs)
    white, grey, black = 0, 1, 2
    color: dict[str, int] = defaultdict(int)
    for root in sorted(adj):
        if color[root] != white:
            continue
        color[root] = grey
        path = [root]
        iters = [iter(sorted(adj[root]))]
        while iters:
            nxt = next(iters[-1], None)
            if nxt is None:
                color[path.pop()] = black
                iters.pop()
            elif color[nxt] == grey:
                return [*path[path.index(nxt) :], nxt]
            elif color[nxt] == white:
                color[nxt] = grey
                path.append(nxt)
                iters.append(iter(sorted(adj.get(nxt, ()))))
    return None
