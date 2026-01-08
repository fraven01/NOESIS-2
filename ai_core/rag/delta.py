"""Delta and deduplication helpers for crawler ingestion flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional

from documents.contracts import NormalizedDocument
from documents.normalization import document_payload_bytes, normalized_primary_text

from ai_core.rag.deduplication import (
    DedupSignatures as DeltaSignatures,
    NearDuplicateSignature as _NearDuplicateSignature,
)
from ai_core.rag.hashing import build_dedup_signatures, extract_primary_text_hash


NearDuplicateSignature = _NearDuplicateSignature


class DeltaStatus(str, Enum):
    """Possible delta outcomes for a normalized document."""

    NEW = "new"
    CHANGED = "changed"
    UNCHANGED = "unchanged"
    NEAR_DUPLICATE = "near_duplicate"


@dataclass(frozen=True)
class DeltaDecision:
    """Result of the delta evaluation using the shared decision payload."""

    decision: str
    reason: str
    attributes: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_legacy(
        cls,
        status: DeltaStatus,
        signatures: DeltaSignatures,
        version: Optional[int],
        reason: str,
        parent_document_id: Optional[str] = None,
    ) -> "DeltaDecision":
        attributes: Mapping[str, object] = MappingProxyType(
            {
                "signatures": signatures,
                "version": version,
                "parent_document_id": parent_document_id,
            }
        )
        return cls(status.value, reason, attributes)

    @property
    def status(self) -> DeltaStatus:
        return DeltaStatus(self.decision)

    @property
    def signatures(self) -> DeltaSignatures:
        return self.attributes["signatures"]  # type: ignore[index]

    @property
    def version(self) -> Optional[int]:
        return self.attributes.get("version")  # type: ignore[return-value]

    @property
    def parent_document_id(self) -> Optional[str]:
        return self.attributes.get("parent_document_id")  # type: ignore[return-value]

    def __post_init__(self) -> None:
        decision = (self.decision or "").strip()
        if not decision:
            raise ValueError("decision_required")
        reason = (self.reason or "").strip()
        if not reason:
            raise ValueError("reason_required")
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "reason", reason)

        raw_attributes = self.attributes or {}
        if isinstance(raw_attributes, Mapping):
            object.__setattr__(
                self, "attributes", MappingProxyType(dict(raw_attributes))
            )
        else:
            raise TypeError("attributes_must_be_mapping")


def evaluate_delta(
    document: NormalizedDocument,
    *,
    primary_text: Optional[str] = None,
    previous_content_hash: Optional[str] = None,
    previous_version: Optional[int] = None,
    binary_payload: Optional[bytes] = None,
    hash_algorithm: str = "sha256",
) -> DeltaDecision:
    """Evaluate content deltas and propose deduplication signatures."""

    normalized_text = normalized_primary_text(primary_text)
    stored_hash = extract_primary_text_hash(document.meta.parse_stats, hash_algorithm)

    if binary_payload is None:
        payload_bytes = document_payload_bytes(document)
    else:
        payload_bytes = binary_payload

    signatures = build_dedup_signatures(
        primary_text=primary_text,
        normalized_primary_text=normalized_text,
        stored_primary_text_hash=stored_hash,
        payload_bytes=payload_bytes,
        algorithm=hash_algorithm,
    )

    if previous_content_hash is None:
        return DeltaDecision.from_legacy(
            DeltaStatus.NEW,
            signatures,
            1,
            "no_previous_hash",
        )

    if previous_content_hash == signatures.content_hash:
        version = previous_version if previous_version is not None else 1
        return DeltaDecision.from_legacy(
            DeltaStatus.UNCHANGED,
            signatures,
            version,
            "hash_match",
        )

    version = (previous_version or 0) + 1
    return DeltaDecision.from_legacy(
        DeltaStatus.CHANGED,
        signatures,
        version,
        "hash_mismatch",
    )


__all__ = [
    "DeltaDecision",
    "DeltaSignatures",
    "DeltaStatus",
    "NearDuplicateSignature",
    "evaluate_delta",
]
