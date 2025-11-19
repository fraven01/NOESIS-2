from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from ai_core.ingestion.blocks import StructuredBlockReader
from ai_core.ingestion.chunk_assembler import ChunkAssembler, ChunkAssemblerInput
from ai_core.ingestion.parent_capture import ChunkCandidate, ParentCapture
from ai_core.ingestion.pii import PIIMasker


def test_pii_masker_applies_digit_scrubbing(settings) -> None:
    settings.INGESTION_PII_MASK_ENABLED = True

    def fake_mask(value: str) -> str:
        return value

    def fake_config() -> Dict[str, str]:
        return {"mode": "strict", "policy": "redact"}

    masker = PIIMasker(mask_func=fake_mask, config_loader=fake_config)
    masked = masker.mask("Reference 123-456")
    assert masked == "Reference XXX-XXX"


def test_structured_block_reader_prefers_object_store_payload() -> None:
    class FakeStore:
        def __init__(self, payload: Dict[str, Any] | None) -> None:
            self.payload = payload

        def read_json(self, path: str) -> Dict[str, Any] | None:
            assert path == "blocks.json"
            return self.payload

    reader = StructuredBlockReader(
        store=FakeStore({"blocks": [{"text": "Hello"}]}),
        segmenter=lambda text: [text.upper()],
    )
    doc = reader.read({"parsed_blocks_path": "blocks.json"}, "ignored")
    assert doc.blocks == [{"text": "Hello"}]
    assert doc.fallback_segments == []

    fallback_reader = StructuredBlockReader(
        store=FakeStore(None),
        segmenter=lambda text: [part for part in text.split("\n") if part],
    )
    fallback_doc = fallback_reader.read({}, "first\nsecond")
    assert fallback_doc.blocks == []
    assert fallback_doc.fallback_segments == ["first", "second"]


def _split_sentences(text: str) -> List[str]:
    sentences = [segment.strip() for segment in text.split(".") if segment.strip()]
    return sentences or ([text] if text else [])


def _split_by_limit(text: str, limit: int) -> List[str]:
    tokens = text.split()
    return [" ".join(tokens[start : start + limit]) for start in range(0, len(tokens), limit)]


def _token_count(text: str) -> int:
    return len([token for token in text.split() if token])


def _chunk_sentences(
    sentences: Sequence[str],
    *,
    target_tokens: int,
    overlap_tokens: int,
    hard_limit: int,
) -> List[str]:
    joined = " ".join(sentences).strip()
    return [joined] if joined else []


def test_parent_capture_builds_section_candidates() -> None:
    capture = ParentCapture(
        document_id="11111111-1111-1111-1111-111111111111",
        external_id="doc-1",
        title="Doc",
        parent_prefix="11111111-1111-1111-1111-111111111111",
        max_depth=5,
        max_bytes=0,
    )
    result = capture.build_candidates(
        structured_blocks=[
            {"kind": "heading", "text": "Intro", "section_path": ["Intro"]},
            {
                "kind": "paragraph",
                "text": "Sentence one. Sentence two.",
                "section_path": ["Intro"],
            },
        ],
        fallback_segments=[],
        full_text="Sentence one. Sentence two.",
        target_tokens=50,
        overlap_tokens=5,
        hard_limit=50,
        mask=lambda value: value,
        chunk_sentences=_chunk_sentences,
        sentence_splitter=_split_sentences,
        split_by_limit=_split_by_limit,
        token_counter=_token_count,
    )
    assert result.chunk_candidates
    candidate = result.chunk_candidates[0]
    assert candidate.heading_prefix == "Intro"
    assert len(candidate.parent_ids) >= 2
    section_nodes = [info for info in result.parent_nodes.values() if info.get("type") == "section"]
    assert section_nodes
    assert section_nodes[0].get("content")


def test_chunk_assembler_enforces_hard_limit() -> None:
    assembler = ChunkAssembler(
        token_counter=_token_count,
        split_by_limit=_split_by_limit,
        normalizer=lambda text: text.strip(),
    )
    candidate = ChunkCandidate(body="one two three four five", parent_ids=["root"])
    chunk_input = ChunkAssemblerInput(
        prefix="",
        chunk_candidates=[candidate],
        hard_limit=3,
        meta={"tenant_id": "tenant", "external_id": "doc-1"},
        text_path="source.txt",
        content_hash="abc123",
        document_id="11111111-1111-1111-1111-111111111111",
    )
    chunks = assembler.assemble(chunk_input)
    assert len(chunks) == 2
    assert all(len(entry["content"].split()) <= 3 for entry in chunks)
    assert all(entry["meta"]["parent_ids"] == ["root"] for entry in chunks)
