from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass
class ChunkCandidate:
    body: str
    parent_ids: List[str]
    heading_prefix: str = ""


@dataclass
class ParentCaptureResult:
    chunk_candidates: List[ChunkCandidate]
    parent_nodes: Dict[str, Dict[str, Any]]


class ParentCapture:
    """Capture parent hierarchy information and chunk candidates."""

    def __init__(
        self,
        *,
        document_id: Optional[str],
        external_id: str,
        title: str,
        parent_prefix: str,
        max_depth: int,
        max_bytes: int,
    ) -> None:
        self.document_id = document_id
        self.external_id = external_id
        self.parent_prefix = parent_prefix
        self.max_depth = max_depth
        self.max_bytes = max_bytes
        self.root_id = f"{parent_prefix}#doc"
        self.parent_nodes: Dict[str, Dict[str, Any]] = {
            self.root_id: {
                "id": self.root_id,
                "type": "document",
                "title": title or None,
                "level": 0,
                "order": 0,
            }
        }
        if document_id:
            self.parent_nodes[self.root_id]["document_id"] = document_id
        self.parent_contents: Dict[str, List[str]] = {self.root_id: []}
        self.parent_content_bytes: Dict[str, int] = {self.root_id: 0}
        self.parent_stack: List[Dict[str, Any]] = []
        self.section_counter = 0
        self.order_counter = 0
        self.chunk_candidates: List[ChunkCandidate] = []

    def build_candidates(
        self,
        *,
        structured_blocks: Sequence[Mapping[str, Any]],
        fallback_segments: Sequence[str],
        full_text: str,
        target_tokens: int,
        overlap_tokens: int,
        hard_limit: int,
        mask: Callable[[str], str],
        chunk_sentences: Callable[[Sequence[str], int, int, int], List[str]],
        sentence_splitter: Callable[[str], List[str]],
        split_by_limit: Callable[[str, int], List[str]],
        token_counter: Callable[[str], int],
    ) -> ParentCaptureResult:
        if structured_blocks:
            self._from_structured_blocks(
                structured_blocks,
                mask,
                target_tokens,
                overlap_tokens,
                hard_limit,
                chunk_sentences,
                sentence_splitter,
                split_by_limit,
                token_counter,
            )
        else:
            self._from_fallback_segments(
                fallback_segments,
                mask,
                target_tokens,
                overlap_tokens,
                hard_limit,
                chunk_sentences,
                sentence_splitter,
                split_by_limit,
                token_counter,
            )

        if not self.chunk_candidates:
            self._ensure_fallback_candidate(mask(full_text))

        self._finalize_parent_nodes()

        return ParentCaptureResult(
            chunk_candidates=self.chunk_candidates,
            parent_nodes=self.parent_nodes,
        )

    def _from_structured_blocks(
        self,
        structured_blocks: Sequence[Mapping[str, Any]],
        mask: Callable[[str], str],
        target_tokens: int,
        overlap_tokens: int,
        hard_limit: int,
        chunk_sentences: Callable[[Sequence[str], int, int, int], List[str]],
        sentence_splitter: Callable[[str], List[str]],
        split_by_limit: Callable[[str, int], List[str]],
        token_counter: Callable[[str], int],
    ) -> None:
        pending_pieces: List[str] = []
        pending_parent_ids: Optional[Tuple[str, ...]] = None
        pending_heading_prefix: str = ""

        def _flush_pending() -> None:
            nonlocal pending_pieces, pending_parent_ids, pending_heading_prefix
            if not pending_pieces or pending_parent_ids is None:
                pending_pieces = []
                pending_parent_ids = None
                pending_heading_prefix = ""
                return

            combined_text = "\n\n".join(part for part in pending_pieces if part.strip())
            sentences = sentence_splitter(combined_text)
            if not sentences:
                sentences = [combined_text] if combined_text else []
            if sentences:
                bodies = chunk_sentences(
                    sentences,
                    target_tokens=target_tokens,
                    overlap_tokens=overlap_tokens,
                    hard_limit=hard_limit,
                )
                for body in bodies:
                    self.chunk_candidates.append(
                        ChunkCandidate(
                            body=body,
                            parent_ids=list(pending_parent_ids),
                            heading_prefix=pending_heading_prefix,
                        )
                    )
            pending_pieces = []
            pending_parent_ids = None
            pending_heading_prefix = ""

        section_registry: Dict[Tuple[str, ...], Dict[str, Any]] = {}

        for block in structured_blocks:
            text_value = str(block.get("text") or "")
            kind = str(block.get("kind") or "").lower()
            section_path_raw = block.get("section_path")
            path_tuple: Tuple[str, ...] = ()
            if isinstance(section_path_raw, (list, tuple)):
                path_tuple = tuple(
                    str(part).strip()
                    for part in section_path_raw
                    if isinstance(part, str) and str(part).strip()
                )
            masked_text = mask(text_value)
            if kind == "heading":
                _flush_pending()
                level = len(path_tuple) if path_tuple else 1
                while self.parent_stack and len(self.parent_stack[-1].get("path", ())) >= level:
                    self.parent_stack.pop()
                title_candidate = path_tuple[-1] if path_tuple else text_value.strip()
                section_info = self._register_section(
                    title_candidate or text_value.strip(), level, path_tuple or None
                )
                section_registry[path_tuple] = section_info
                self.parent_stack.append(section_info)
                if masked_text.strip():
                    self._append_parent_text_with_root(
                        section_info["id"], masked_text.strip(), level
                    )
                continue

            normalised_text = masked_text.strip()
            if not normalised_text:
                continue

            if path_tuple:
                desired_stack: List[Dict[str, Any]] = []
                for depth in range(1, len(path_tuple) + 1):
                    prefix_tuple = path_tuple[:depth]
                    info = section_registry.get(prefix_tuple)
                    if info is None:
                        title_candidate = prefix_tuple[-1] if prefix_tuple else ""
                        info = self._register_section(title_candidate, depth, prefix_tuple)
                        section_registry[prefix_tuple] = info
                    desired_stack.append(info)
                self.parent_stack = desired_stack
            else:
                self.parent_stack = []

            block_pieces = [normalised_text]
            if token_counter(normalised_text) > hard_limit:
                block_pieces = split_by_limit(normalised_text, hard_limit)

            for piece in block_pieces:
                piece_text = piece.strip()
                if not piece_text:
                    continue
                if self.parent_stack:
                    target_info = self.parent_stack[-1]
                    target_level = int(target_info.get("level") or 0)
                    self._append_parent_text_with_root(
                        target_info["id"], piece_text, target_level
                    )
                else:
                    self._append_parent_text(self.root_id, piece_text, 0)
                parent_ids = [self.root_id] + [info["id"] for info in self.parent_stack]
                unique_parent_ids = list(dict.fromkeys(pid for pid in parent_ids if pid))
                heading_prefix = " / ".join(path_tuple) if path_tuple else ""

                new_parent_ids = tuple(unique_parent_ids)
                if pending_parent_ids is not None and (
                    pending_parent_ids != new_parent_ids
                    or pending_heading_prefix != heading_prefix
                ):
                    _flush_pending()

                if pending_parent_ids is None:
                    pending_parent_ids = new_parent_ids
                    pending_heading_prefix = heading_prefix

                pending_pieces.append(piece_text)

        _flush_pending()

    def _from_fallback_segments(
        self,
        fallback_segments: Sequence[str],
        mask: Callable[[str], str],
        target_tokens: int,
        overlap_tokens: int,
        hard_limit: int,
        chunk_sentences: Callable[[Sequence[str], int, int, int], List[str]],
        sentence_splitter: Callable[[str], List[str]],
        split_by_limit: Callable[[str, int], List[str]],
        token_counter: Callable[[str], int],
    ) -> None:
        pending_pieces: List[str] = []
        pending_parent_ids: Optional[Tuple[str, ...]] = None

        def _flush_pending() -> None:
            nonlocal pending_pieces, pending_parent_ids
            if not pending_pieces or pending_parent_ids is None:
                pending_pieces = []
                pending_parent_ids = None
                return

            combined_text = "\n\n".join(part for part in pending_pieces if part.strip())
            sentences = sentence_splitter(combined_text)
            if not sentences:
                sentences = [combined_text] if combined_text else []
            if sentences:
                bodies = chunk_sentences(
                    sentences,
                    target_tokens=target_tokens,
                    overlap_tokens=overlap_tokens,
                    hard_limit=hard_limit,
                )
                for body in bodies:
                    self.chunk_candidates.append(
                        ChunkCandidate(body=body, parent_ids=list(pending_parent_ids))
                    )
            pending_pieces = []
            pending_parent_ids = None

        heading_pattern = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")

        for block in fallback_segments:
            stripped_block = block.strip()
            if not stripped_block:
                continue
            heading_match = heading_pattern.match(stripped_block)
            if heading_match:
                _flush_pending()
                hashes, heading_title = heading_match.groups()
                level = len(hashes)
                while self.parent_stack and int(self.parent_stack[-1].get("level") or 0) >= level:
                    self.parent_stack.pop()
                section_info = self._register_section(heading_title.strip(), level)
                self.parent_stack.append(section_info)
                self._append_parent_text_with_root(
                    section_info["id"], mask(stripped_block), level
                )
                continue

            masked_block = mask(block)
            block_pieces = [masked_block]
            if token_counter(masked_block) > hard_limit:
                block_pieces = split_by_limit(masked_block, hard_limit)
            for piece in block_pieces:
                piece_text = piece.strip()
                if not piece_text:
                    continue
                if self.parent_stack:
                    target_info = self.parent_stack[-1]
                    target_level = int(target_info.get("level") or 0)
                    self._append_parent_text_with_root(
                        target_info["id"], piece_text, target_level
                    )
                else:
                    self._append_parent_text(self.root_id, piece_text, 0)
                parent_ids = [self.root_id] + [info["id"] for info in self.parent_stack]
                unique_parent_ids = list(dict.fromkeys(pid for pid in parent_ids if pid))

                new_parent_ids = tuple(unique_parent_ids)
                if pending_parent_ids is not None and pending_parent_ids != new_parent_ids:
                    _flush_pending()

                if pending_parent_ids is None:
                    pending_parent_ids = new_parent_ids

                pending_pieces.append(piece_text)

        _flush_pending()

    def _ensure_fallback_candidate(self, fallback_text: str) -> None:
        fallback_ids = [self.root_id] + [info["id"] for info in self.parent_stack]
        unique_ids = list(dict.fromkeys(pid for pid in fallback_ids if pid))
        stripped = fallback_text.strip()
        if stripped:
            if self.parent_stack:
                target_info = self.parent_stack[-1]
                target_level = int(target_info.get("level") or 0)
                self._append_parent_text_with_root(target_info["id"], stripped, target_level)
            else:
                self._append_parent_text(self.root_id, stripped, 0)
        self.chunk_candidates.append(
            ChunkCandidate(body=fallback_text, parent_ids=unique_ids or [self.root_id])
        )

    def _append_parent_text(self, parent_id: str, text: str, level: int) -> None:
        if not text:
            return
        normalised = text.strip()
        if not normalised:
            return
        is_root_parent = parent_id == self.root_id
        if (
            not is_root_parent
            and self.max_depth > 0
            and not self._within_capture_depth(level)
        ):
            return
        if self.max_bytes > 0 and not is_root_parent:
            used = self.parent_content_bytes.get(parent_id, 0)
            remaining = self.max_bytes - used
            if remaining <= 0:
                self.parent_nodes[parent_id]["capture_limited"] = True
                return
            separator_bytes = len("\n\n".encode("utf-8")) if self.parent_contents[parent_id] else 0
            if remaining <= separator_bytes:
                self.parent_nodes[parent_id]["capture_limited"] = True
                return
            allowed = remaining - separator_bytes
            encoded = normalised.encode("utf-8")
            if len(encoded) > allowed:
                truncated = encoded[:allowed]
                preview = truncated.decode("utf-8", errors="ignore").strip()
                self.parent_nodes[parent_id]["capture_limited"] = True
                if not preview:
                    return
                self.parent_contents[parent_id].append(preview)
                appended_bytes = len(preview.encode("utf-8"))
                self.parent_content_bytes[parent_id] = used + separator_bytes + appended_bytes
                return
            self.parent_contents[parent_id].append(normalised)
            self.parent_content_bytes[parent_id] = used + separator_bytes + len(encoded)
            return
        self.parent_contents[parent_id].append(normalised)
        if self.max_bytes > 0:
            used = self.parent_content_bytes.get(parent_id, 0)
            separator_bytes = len("\n\n".encode("utf-8")) if used > 0 else 0
            self.parent_content_bytes[parent_id] = used + separator_bytes + len(
                normalised.encode("utf-8")
            )

    def _append_parent_text_with_root(self, parent_id: str, text: str, level: int) -> None:
        self._append_parent_text(parent_id, text, level)
        if parent_id != self.root_id:
            self._append_parent_text(self.root_id, text, level)

    def _register_section(
        self,
        title: str,
        level: int,
        path: Optional[Tuple[str, ...]] = None,
    ) -> Dict[str, Any]:
        self.section_counter += 1
        self.order_counter += 1
        parent_id = f"{self.parent_prefix}#sec-{self.section_counter}"
        info: Dict[str, Any] = {
            "id": parent_id,
            "type": "section",
            "title": title or None,
            "level": level,
            "order": self.order_counter,
        }
        if path is not None:
            info["path"] = path
        if self.document_id:
            info["document_id"] = self.document_id
        self.parent_nodes[parent_id] = info
        self.parent_contents[parent_id] = []
        self.parent_content_bytes[parent_id] = 0
        return info

    def _within_capture_depth(self, level: int) -> bool:
        if self.max_depth <= 0:
            return True
        return level <= self.max_depth

    def _finalize_parent_nodes(self) -> None:
        for parent_id, info in list(self.parent_nodes.items()):
            content_parts = self.parent_contents.get(parent_id) or []
            content_text = "\n\n".join(part for part in content_parts if part).strip()
            if content_text:
                enriched = dict(info)
                enriched["content"] = content_text
                self.parent_nodes[parent_id] = enriched
        if self.document_id:
            try:
                compact_root = f"{self.document_id.replace('-', '')}#doc"
                dashed_root = f"{self.document_id}#doc"
                if compact_root in self.parent_nodes and dashed_root not in self.parent_nodes:
                    cloned = dict(self.parent_nodes[compact_root])
                    cloned.setdefault("document_id", self.document_id)
                    self.parent_nodes[dashed_root] = cloned
            except Exception:
                pass
