"""Delta and deduplication decisions for normalized crawler documents."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from .normalizer import NormalizedDocument


class DeltaStatus(str, Enum):
    """Possible delta outcomes for a normalized document."""

    NEW = "new"
    CHANGED = "changed"
    UNCHANGED = "unchanged"
    NEAR_DUPLICATE = "near_duplicate"


@dataclass(frozen=True)
class NearDuplicateSignature:
    """Set-based token signature used for near-duplicate checks."""

    fingerprint: str
    tokens: Tuple[str, ...]

    def __post_init__(self) -> None:
        normalized_tokens = tuple(_normalize_token_sequence(self.tokens))
        object.__setattr__(self, "tokens", normalized_tokens)
        if not self.fingerprint:
            raise ValueError("near_duplicate_fingerprint_required")


@dataclass(frozen=True)
class DeltaSignatures:
    """Stable signatures emitted for each delta evaluation."""

    content_hash: str
    near_duplicate: Optional[NearDuplicateSignature] = None

    def __post_init__(self) -> None:
        if not self.content_hash:
            raise ValueError("content_hash_required")


@dataclass(frozen=True)
class DeltaDecision:
    """Result of the delta evaluation including status and signatures."""

    status: DeltaStatus
    signatures: DeltaSignatures
    version: Optional[int]
    reason: str
    parent_document_id: Optional[str] = None


DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.92


def evaluate_delta(
    document: NormalizedDocument,
    *,
    previous_content_hash: Optional[str] = None,
    previous_version: Optional[int] = None,
    known_near_duplicates: Optional[Mapping[str, NearDuplicateSignature]] = None,
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    binary_payload: Optional[bytes] = None,
    check_near_duplicates_for_changes: bool = False,
    hash_algorithm: str = "sha256",
) -> DeltaDecision:
    """Decide whether the document content is new, changed, unchanged or a near-duplicate."""

    content_hash = _compute_content_hash(
        document, binary_payload=binary_payload, algorithm=hash_algorithm
    )
    near_signature = _compute_near_duplicate_signature(document)

    signatures = DeltaSignatures(content_hash=content_hash, near_duplicate=near_signature)

    duplicate_match = _match_near_duplicate(
        near_signature,
        known_near_duplicates,
        threshold=near_duplicate_threshold,
        exclude=document.document_id,
    )

    if previous_content_hash is None:
        status = DeltaStatus.NEW
        version = 1
        reason = "no_previous_hash"
        parent_document_id = None
        if duplicate_match is not None:
            status = DeltaStatus.NEAR_DUPLICATE
            version = None
            reason = f"near_duplicate:{duplicate_match.similarity:.3f}"
            parent_document_id = duplicate_match.document_id
        return DeltaDecision(status, signatures, version, reason, parent_document_id)

    if previous_content_hash == content_hash:
        version = previous_version if previous_version is not None else 1
        return DeltaDecision(DeltaStatus.UNCHANGED, signatures, version, "hash_match")

    if check_near_duplicates_for_changes and duplicate_match is not None:
        reason = f"near_duplicate:{duplicate_match.similarity:.3f}"
        return DeltaDecision(
            DeltaStatus.NEAR_DUPLICATE,
            signatures,
            None,
            reason,
            duplicate_match.document_id,
        )

    version = (previous_version or 0) + 1
    return DeltaDecision(DeltaStatus.CHANGED, signatures, version, "hash_mismatch")


def _compute_content_hash(
    document: NormalizedDocument,
    *,
    binary_payload: Optional[bytes] = None,
    algorithm: str = "sha256",
) -> str:
    content = document.content
    if content.primary_text:
        normalized = " ".join(content.primary_text.split())
        payload = normalized.encode("utf-8")
    else:
        if binary_payload is None:
            raise ValueError("binary_payload_required")
        payload = binary_payload

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError("unsupported_hash_algorithm") from exc
    hasher.update(payload)
    return hasher.hexdigest()


def _compute_near_duplicate_signature(document: NormalizedDocument) -> Optional[NearDuplicateSignature]:
    primary_text = document.content.primary_text
    if not primary_text:
        return None
    tokens = _tokenize(primary_text)
    if not tokens:
        return None
    normalized_tokens = tuple(sorted(set(tokens)))
    fingerprint_source = "\u001f".join(normalized_tokens)
    fingerprint = hashlib.sha1(fingerprint_source.encode("utf-8")).hexdigest()
    return NearDuplicateSignature(fingerprint=fingerprint, tokens=normalized_tokens)


def _tokenize(text: str) -> Tuple[str, ...]:
    return tuple(re.findall(r"\w+", text.lower()))


def _normalize_token_sequence(tokens: Sequence[str]) -> Tuple[str, ...]:
    normalized = tuple(sorted({token.strip().lower() for token in tokens if token.strip()}))
    if not normalized:
        raise ValueError("near_duplicate_tokens_required")
    return normalized


@dataclass(frozen=True)
class _NearDuplicateMatch:
    document_id: str
    similarity: float


def _match_near_duplicate(
    signature: Optional[NearDuplicateSignature],
    known: Optional[Mapping[str, NearDuplicateSignature]],
    *,
    threshold: float,
    exclude: Optional[str],
) -> Optional[_NearDuplicateMatch]:
    if signature is None or not known:
        return None
    best_id: Optional[str] = None
    best_similarity = 0.0
    signature_tokens = set(signature.tokens)
    for doc_id, candidate in known.items():
        if exclude is not None and doc_id == exclude:
            continue
        similarity = _jaccard(signature_tokens, set(candidate.tokens))
        if similarity > best_similarity:
            best_id = doc_id
            best_similarity = similarity
    if best_id is None or best_similarity < threshold:
        return None
    return _NearDuplicateMatch(best_id, best_similarity)


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    set_left = set(left)
    set_right = set(right)
    if not set_left and not set_right:
        return 1.0
    union = set_left | set_right
    if not union:
        return 0.0
    return len(set_left & set_right) / len(union)


__all__ = [
    "DeltaDecision",
    "DeltaSignatures",
    "DeltaStatus",
    "NearDuplicateSignature",
    "DEFAULT_NEAR_DUPLICATE_THRESHOLD",
    "evaluate_delta",
]

