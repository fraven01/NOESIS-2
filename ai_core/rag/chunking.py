from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class SemanticTextBlock:
    text: str
    kind: str
    section_path: Tuple[str, ...] = ()


@dataclass
class SectionNode:
    path: Tuple[str, ...]
    title: str
    level: int
    heading_text: str | None = None
    body: List[str] = field(default_factory=list)
    children: List["SectionNode"] = field(default_factory=list)


@dataclass(frozen=True)
class SectionChunkPlan:
    path: Tuple[str, ...]
    level: int
    title: str
    heading_prefix: str
    parent_text: str
    chunk_bodies: List[str]


class SemanticChunker:
    """Build a section tree from parsed blocks and emit section-scoped chunks."""

    def __init__(
        self,
        *,
        sentence_splitter: Callable[[str], List[str]],
        chunkify_fn: Callable[[Sequence[str]], List[str]],
    ) -> None:
        self._sentence_splitter = sentence_splitter
        self._chunkify_fn = chunkify_fn

    def build_plans(
        self, blocks: Iterable[SemanticTextBlock]
    ) -> List[SectionChunkPlan]:
        root = SectionNode(path=(), title="", level=0)
        registry: dict[Tuple[str, ...], SectionNode] = {(): root}

        def _ensure_node(path: Tuple[str, ...]) -> SectionNode:
            if path in registry:
                return registry[path]

            parent_path = path[:-1]
            parent = _ensure_node(parent_path)
            node = SectionNode(
                path=path, title=path[-1] if path else "", level=len(path)
            )
            registry[path] = node
            parent.children.append(node)
            return node

        for block in blocks:
            path = tuple(part for part in block.section_path if part)
            node = _ensure_node(path)
            text_value = block.text.strip()
            if not text_value:
                continue

            if block.kind.lower() == "heading":
                if node.heading_text:
                    node.heading_text = f"{node.heading_text}\n\n{text_value}"
                else:
                    node.heading_text = text_value
            else:
                node.body.append(text_value)

        plans: List[SectionChunkPlan] = []
        for node in self._iter_preorder(root):
            parent_parts: List[str] = []
            if node.heading_text:
                parent_parts.append(node.heading_text.strip())
            body_text = "\n\n".join(part for part in node.body if part.strip()).strip()
            if body_text:
                parent_parts.append(body_text)

            parent_text = "\n\n".join(part for part in parent_parts if part).strip()

            chunk_bodies: List[str] = []
            if body_text:
                sentences = self._sentence_splitter(body_text)
                if not sentences:
                    sentences = [body_text]
                chunk_bodies = self._chunkify_fn(sentences)

            plans.append(
                SectionChunkPlan(
                    path=node.path,
                    level=node.level,
                    title=node.title,
                    heading_prefix=" / ".join(node.path) if node.path else "",
                    parent_text=parent_text,
                    chunk_bodies=chunk_bodies,
                )
            )

        return plans

    def _iter_preorder(self, node: SectionNode) -> Iterable[SectionNode]:
        yield node
        for child in node.children:
            yield from self._iter_preorder(child)
