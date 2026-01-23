"""Passage assembly for retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ai_core.rag.evidence_graph import EvidenceGraph
from ai_core.rag.schemas import Chunk


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class Passage:
    passage_id: str
    text: str
    chunk_ids: tuple[str, ...]
    section_path: tuple[str, ...]
    score: float


def assemble_passages(
    chunks: Sequence[Chunk],
    *,
    max_tokens: int = 450,
) -> list[Passage]:
    if not chunks:
        return []
    max_tokens = max(1, int(max_tokens))

    graph = EvidenceGraph.from_chunks(chunks)
    id_to_chunk: dict[str, Chunk] = {}
    scores: dict[str, float] = {}
    section_paths: dict[str, tuple[str, ...]] = {}

    nodes_by_rank = {node.rank: node for node in graph.nodes.values()}
    for index, chunk in enumerate(chunks):
        node = nodes_by_rank.get(index)
        if node is None:
            continue
        id_to_chunk[node.chunk_id] = chunk
        scores[node.chunk_id] = node.score
        section_paths[node.chunk_id] = node.section_path

    ordered_ids = sorted(
        id_to_chunk.keys(), key=lambda cid: (-scores.get(cid, 0.0), cid)
    )
    used: set[str] = set()
    passages: list[Passage] = []

    for anchor_id in ordered_ids:
        if anchor_id in used:
            continue
        anchor_chunk = id_to_chunk[anchor_id]
        anchor_meta = anchor_chunk.meta or {}
        anchor_section = section_paths.get(anchor_id, ())
        passage_ids: list[str] = [anchor_id]
        used.add(anchor_id)
        token_budget = max_tokens - estimate_tokens(anchor_chunk.content or "")

        neighbors = graph.get_adjacent(anchor_id, max_hops=1)
        neighbor_ids = [
            cid
            for cid in neighbors
            if cid not in used and section_paths.get(cid, ()) == anchor_section
        ]
        neighbor_ids.sort(key=lambda cid: graph.nodes[cid].rank)

        for neighbor_id in neighbor_ids:
            if token_budget <= 0:
                break
            neighbor_chunk = id_to_chunk.get(neighbor_id)
            if neighbor_chunk is None:
                continue
            chunk_text = neighbor_chunk.content or ""
            chunk_tokens = estimate_tokens(chunk_text)
            if chunk_tokens > token_budget:
                continue
            passage_ids.append(neighbor_id)
            used.add(neighbor_id)
            token_budget -= chunk_tokens

        passage_ids.sort(key=lambda cid: graph.nodes[cid].rank)
        texts = [id_to_chunk[cid].content or "" for cid in passage_ids]
        passage_text = "\n\n".join(texts).strip()
        if not passage_text:
            continue

        score = max(scores.get(cid, 0.0) for cid in passage_ids)
        passage_id = anchor_meta.get("chunk_id") or anchor_id
        passages.append(
            Passage(
                passage_id=str(passage_id),
                text=passage_text,
                chunk_ids=tuple(passage_ids),
                section_path=anchor_section,
                score=score,
            )
        )

    passages.sort(key=lambda item: (-item.score, item.passage_id))
    return passages


__all__ = ["Passage", "assemble_passages", "estimate_tokens"]
