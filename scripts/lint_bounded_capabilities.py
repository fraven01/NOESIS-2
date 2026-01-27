from __future__ import annotations

from typing import Iterable, Sequence


def _extract_edges(graph: object) -> list[tuple[str, str]]:
    edges: Iterable[object] | None = None
    if hasattr(graph, "edges"):
        edges = getattr(graph, "edges")
    elif hasattr(graph, "get_edges") and callable(getattr(graph, "get_edges")):
        edges = graph.get_edges()
    elif hasattr(graph, "graph") and hasattr(graph.graph, "edges"):
        edges = getattr(graph.graph, "edges")

    if edges is None:
        return []

    result: list[tuple[str, str]] = []
    for edge in edges:
        if isinstance(edge, Sequence) and len(edge) >= 2:
            src = str(edge[0])
            dst = str(edge[1])
            result.append((src, dst))
    return result


def _is_error_node(node: str) -> bool:
    return "error" in node.lower()


def _has_forbidden_back_edge(edges: list[tuple[str, str]]) -> bool:
    edge_set = {(src, dst) for src, dst in edges if not _is_error_node(dst)}
    for src, dst in edge_set:
        if src == dst:
            return True
        if (dst, src) in edge_set:
            return True
    return False


def validate_graph_edges(graph: object) -> None:
    edges = _extract_edges(graph)
    if not edges:
        return
    if _has_forbidden_back_edge(edges):
        raise ValueError("back_edge_detected")


__all__ = ["validate_graph_edges"]
