"""Tests for EvidenceGraph."""

from ai_core.rag.evidence_graph import EvidenceGraph
from ai_core.rag.schemas import Chunk


def _chunk(
    *,
    chunk_id: str,
    document_id: str,
    section_path: tuple[str, ...],
    chunk_index: int,
    parent_ids: list[str] | None = None,
    reference_ids: list[str] | None = None,
    score: float = 0.0,
) -> Chunk:
    meta = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "section_path": list(section_path),
        "chunk_index": chunk_index,
        "parent_ids": parent_ids or [],
        "score": score,
    }
    if reference_ids:
        meta["reference_ids"] = list(reference_ids)
    return Chunk(content=f"text:{chunk_id}", meta=meta)


def test_evidence_graph_adjacency_by_section():
    chunks = [
        _chunk(
            chunk_id="c1",
            document_id="doc-1",
            section_path=("A",),
            chunk_index=0,
        ),
        _chunk(
            chunk_id="c2",
            document_id="doc-1",
            section_path=("A",),
            chunk_index=1,
        ),
        _chunk(
            chunk_id="c3",
            document_id="doc-1",
            section_path=("B",),
            chunk_index=2,
        ),
    ]
    graph = EvidenceGraph.from_chunks(chunks)
    assert graph.get_adjacent("c1") == {"c2"}
    assert graph.get_adjacent("c2") == {"c1"}
    assert graph.get_adjacent("c3") == set()


def test_evidence_graph_parent_child_edges():
    chunks = [
        _chunk(
            chunk_id="parent",
            document_id="doc-1",
            section_path=("A",),
            chunk_index=0,
        ),
        _chunk(
            chunk_id="child",
            document_id="doc-1",
            section_path=("A",),
            chunk_index=1,
            parent_ids=["parent"],
        ),
    ]
    graph = EvidenceGraph.from_chunks(chunks)
    assert graph.get_parent("child") == "parent"


def test_evidence_graph_subgraph_expansion():
    chunks = [
        _chunk(
            chunk_id="c1",
            document_id="doc-1",
            section_path=("A",),
            chunk_index=0,
        ),
        _chunk(
            chunk_id="c2",
            document_id="doc-1",
            section_path=("A",),
            chunk_index=1,
        ),
        _chunk(
            chunk_id="c3",
            document_id="doc-1",
            section_path=("A",),
            chunk_index=2,
            parent_ids=["c2"],
        ),
    ]
    graph = EvidenceGraph.from_chunks(chunks)
    subgraph = graph.get_subgraph(["c2"], max_hops=1)
    assert set(subgraph.nodes.keys()) == {"c1", "c2", "c3"}


def test_evidence_graph_reference_edges():
    chunks = [
        _chunk(
            chunk_id="a1",
            document_id="doc-a",
            section_path=("A",),
            chunk_index=0,
            reference_ids=["doc-b"],
        ),
        _chunk(
            chunk_id="b1",
            document_id="doc-b",
            section_path=("B",),
            chunk_index=0,
        ),
    ]
    graph = EvidenceGraph.from_chunks(chunks)
    assert graph.get_references("a1") == {"b1"}
    assert graph.get_backreferences("b1") == {"a1"}
