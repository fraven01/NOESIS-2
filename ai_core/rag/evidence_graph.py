"""Evidence graph data model for retrieval-time reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ai_core.rag.schemas import Chunk


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _extract_section_path(meta: Mapping[str, Any]) -> tuple[str, ...]:
    raw = meta.get("section_path")
    if isinstance(raw, (list, tuple)):
        return tuple(str(item) for item in raw if str(item).strip())
    if isinstance(raw, str) and raw.strip():
        return (raw.strip(),)
    return ()


def _extract_parent_ids(
    meta: Mapping[str, Any], parents: Mapping[str, Any] | None
) -> list[str]:
    raw = meta.get("parent_ids")
    parent_ids: list[str] = []
    if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes, bytearray)):
        for entry in raw:
            candidate = _coerce_str(entry)
            if candidate:
                parent_ids.append(candidate)
    if parents:
        for key in parents.keys():
            candidate = _coerce_str(key)
            if candidate and candidate not in parent_ids:
                parent_ids.append(candidate)
    return parent_ids


def _extract_reference_ids(meta: Mapping[str, Any]) -> tuple[str, ...]:
    raw = meta.get("reference_ids") or meta.get("references")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    seen: set[str] = set()
    references: list[str] = []
    for entry in raw:
        candidate = _coerce_str(entry)
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        references.append(candidate)
    return tuple(references)


def _chunk_identifier(chunk: Chunk, index: int) -> str:
    meta = chunk.meta or {}
    for key in ("chunk_id", "id", "hash"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    doc_id = _coerce_str(meta.get("document_id"))
    if doc_id:
        return f"{doc_id}:{index}"
    return f"chunk-{index}"


@dataclass(frozen=True)
class EvidenceNode:
    chunk_id: str
    document_id: str | None
    section_path: tuple[str, ...]
    parent_ids: tuple[str, ...]
    reference_ids: tuple[str, ...]
    score: float
    rank: int


class EvidenceGraph:
    """In-memory evidence graph for retrieved chunks."""

    def __init__(
        self,
        *,
        nodes: Mapping[str, EvidenceNode],
        adjacency: Mapping[str, set[str]],
        parents: Mapping[str, set[str]],
        children: Mapping[str, set[str]],
        references: Mapping[str, set[str]],
        backreferences: Mapping[str, set[str]],
        doc_order: Mapping[str, list[str]],
    ) -> None:
        self._nodes = dict(nodes)
        self._adjacency = {key: set(value) for key, value in adjacency.items()}
        self._parents = {key: set(value) for key, value in parents.items()}
        self._children = {key: set(value) for key, value in children.items()}
        self._references = {key: set(value) for key, value in references.items()}
        self._backreferences = {
            key: set(value) for key, value in backreferences.items()
        }
        self._doc_order = {
            key: list(value) for key, value in doc_order.items() if value
        }

    @property
    def nodes(self) -> Mapping[str, EvidenceNode]:
        return self._nodes

    @classmethod
    def from_chunks(cls, chunks: Iterable[Chunk]) -> "EvidenceGraph":
        materialized = list(chunks)
        nodes: dict[str, EvidenceNode] = {}
        adjacency: dict[str, set[str]] = {}
        parents: dict[str, set[str]] = {}
        children: dict[str, set[str]] = {}
        references: dict[str, set[str]] = {}
        backreferences: dict[str, set[str]] = {}
        doc_index: dict[str, set[str]] = {}
        doc_order: dict[str, list[str]] = {}
        ordering: list[tuple[str, str | None, tuple[str, ...], int | None, int]] = []

        for index, chunk in enumerate(materialized):
            meta = chunk.meta or {}
            chunk_id = _chunk_identifier(chunk, index)
            document_id = _coerce_str(meta.get("document_id"))
            section_path = _extract_section_path(meta)
            parent_ids = _extract_parent_ids(meta, chunk.parents)
            reference_ids = _extract_reference_ids(meta)
            score = float(meta.get("score") or 0.0)
            node = EvidenceNode(
                chunk_id=chunk_id,
                document_id=document_id,
                section_path=section_path,
                parent_ids=tuple(parent_ids),
                reference_ids=reference_ids,
                score=score,
                rank=index,
            )
            nodes[chunk_id] = node
            chunk_index = _coerce_int(meta.get("chunk_index"))
            ordering.append((chunk_id, document_id, section_path, chunk_index, index))
            if document_id:
                doc_index.setdefault(document_id, set()).add(chunk_id)

        for chunk_id, node in nodes.items():
            if not node.parent_ids:
                continue
            for parent_id in node.parent_ids:
                if parent_id not in nodes:
                    continue
                parents.setdefault(chunk_id, set()).add(parent_id)
                children.setdefault(parent_id, set()).add(chunk_id)

        grouped: dict[
            tuple[str | None, tuple[str, ...]], list[tuple[int | None, int, str]]
        ] = {}
        for chunk_id, document_id, section_path, chunk_index, rank in ordering:
            key = (document_id, section_path)
            grouped.setdefault(key, []).append((chunk_index, rank, chunk_id))

        for entries in grouped.values():
            entries.sort(key=lambda item: (item[0] is None, item[0] or 0, item[1]))
            for idx in range(len(entries) - 1):
                current_id = entries[idx][2]
                next_id = entries[idx + 1][2]
                adjacency.setdefault(current_id, set()).add(next_id)
                adjacency.setdefault(next_id, set()).add(current_id)

        doc_grouped: dict[str, list[tuple[int | None, int, str]]] = {}
        for chunk_id, document_id, _section_path, chunk_index, rank in ordering:
            if not document_id:
                continue
            doc_grouped.setdefault(document_id, []).append(
                (chunk_index, rank, chunk_id)
            )
        for document_id, entries in doc_grouped.items():
            entries.sort(key=lambda item: (item[0] is None, item[0] or 0, item[1]))
            doc_order[document_id] = [entry[2] for entry in entries]

        for chunk_id, node in nodes.items():
            if not node.reference_ids:
                continue
            for ref_id in node.reference_ids:
                targets = doc_index.get(ref_id)
                if not targets:
                    continue
                references.setdefault(chunk_id, set()).update(targets)
                for target_id in targets:
                    backreferences.setdefault(target_id, set()).add(chunk_id)

        return cls(
            nodes=nodes,
            adjacency=adjacency,
            parents=parents,
            children=children,
            references=references,
            backreferences=backreferences,
            doc_order=doc_order,
        )

    def get_adjacent(self, chunk_id: str, *, max_hops: int = 1) -> set[str]:
        if max_hops <= 0:
            return set()
        visited: set[str] = {chunk_id}
        frontier: set[str] = {chunk_id}
        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for current in frontier:
                for neighbor in self._adjacency.get(current, set()):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
            if not next_frontier:
                break
            frontier = next_frontier
        visited.discard(chunk_id)
        return visited

    def get_parent(self, chunk_id: str) -> str | None:
        parent_ids = sorted(self._parents.get(chunk_id, set()))
        return parent_ids[0] if parent_ids else None

    def get_references(self, chunk_id: str) -> set[str]:
        return set(self._references.get(chunk_id, set()))

    def get_backreferences(self, chunk_id: str) -> set[str]:
        return set(self._backreferences.get(chunk_id, set()))

    def get_all_document_chunks(self, document_id: str) -> list[str]:
        return list(self._doc_order.get(document_id, []))

    def get_subgraph(
        self, chunk_ids: Iterable[str], *, max_hops: int = 1
    ) -> "EvidenceGraph":
        seed = {cid for cid in chunk_ids if cid in self._nodes}
        if not seed:
            return EvidenceGraph(
                nodes={},
                adjacency={},
                parents={},
                children={},
                references={},
                backreferences={},
                doc_order={},
            )

        expanded = set(seed)
        for chunk_id in list(seed):
            expanded.update(self.get_adjacent(chunk_id, max_hops=max_hops))
            parent = self.get_parent(chunk_id)
            if parent:
                expanded.add(parent)
            expanded.update(self._children.get(chunk_id, set()))
            expanded.update(self._references.get(chunk_id, set()))
            expanded.update(self._backreferences.get(chunk_id, set()))

        nodes = {cid: self._nodes[cid] for cid in expanded if cid in self._nodes}
        adjacency = {
            cid: {nbr for nbr in self._adjacency.get(cid, set()) if nbr in expanded}
            for cid in nodes
        }
        parents = {
            cid: {pid for pid in self._parents.get(cid, set()) if pid in expanded}
            for cid in nodes
        }
        children = {
            cid: {
                child for child in self._children.get(cid, set()) if child in expanded
            }
            for cid in nodes
        }
        references = {
            cid: {ref for ref in self._references.get(cid, set()) if ref in expanded}
            for cid in nodes
        }
        backreferences = {
            cid: {
                ref for ref in self._backreferences.get(cid, set()) if ref in expanded
            }
            for cid in nodes
        }
        doc_order = {}
        for doc_id, chunk_ids in self._doc_order.items():
            filtered = [cid for cid in chunk_ids if cid in expanded]
            if filtered:
                doc_order[doc_id] = filtered

        return EvidenceGraph(
            nodes=nodes,
            adjacency=adjacency,
            parents=parents,
            children=children,
            references=references,
            backreferences=backreferences,
            doc_order=doc_order,
        )


__all__ = ["EvidenceGraph", "EvidenceNode"]
