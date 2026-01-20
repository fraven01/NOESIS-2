"""Tests for passage assembly."""

from ai_core.rag.passage_assembly import assemble_passages
from ai_core.rag.schemas import Chunk


def _chunk(
    *,
    chunk_id: str,
    document_id: str,
    section_path: tuple[str, ...],
    chunk_index: int,
    score: float,
    text: str,
) -> Chunk:
    meta = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "section_path": list(section_path),
        "chunk_index": chunk_index,
        "score": score,
    }
    return Chunk(content=text, meta=meta)


def test_passage_assembly_merges_adjacent_sections():
    chunks = [
        _chunk(
            chunk_id="c1",
            document_id="doc-1",
            section_path=("A",),
            chunk_index=0,
            score=0.9,
            text="alpha",
        ),
        _chunk(
            chunk_id="c2",
            document_id="doc-1",
            section_path=("A",),
            chunk_index=1,
            score=0.8,
            text="beta",
        ),
        _chunk(
            chunk_id="c3",
            document_id="doc-1",
            section_path=("B",),
            chunk_index=2,
            score=0.7,
            text="gamma",
        ),
    ]
    passages = assemble_passages(chunks, max_tokens=200)
    assert len(passages) == 2
    assert passages[0].chunk_ids == ("c1", "c2")
    assert passages[0].section_path == ("A",)
    assert passages[1].chunk_ids == ("c3",)


def test_passage_assembly_respects_token_limit():
    long_text = "x" * 400
    chunks = [
        _chunk(
            chunk_id="c1",
            document_id="doc-1",
            section_path=("A",),
            chunk_index=0,
            score=0.9,
            text=long_text,
        ),
        _chunk(
            chunk_id="c2",
            document_id="doc-1",
            section_path=("A",),
            chunk_index=1,
            score=0.8,
            text=long_text,
        ),
    ]
    passages = assemble_passages(chunks, max_tokens=50)
    assert len(passages) == 2
    assert all(len(p.chunk_ids) == 1 for p in passages)
