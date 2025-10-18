"""Runtime policy resolution for document processing flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, MutableMapping, Optional, Tuple
from uuid import UUID


@dataclass(frozen=True)
class DocumentPolicy:
    """Resolved policy controlling document extraction behaviours."""

    caption_min_confidence: float
    pdf_ocr_enabled: bool
    pdf_mode: str
    include_pptx_notes: bool


DEFAULT_POLICY = DocumentPolicy(
    caption_min_confidence=0.0,
    pdf_ocr_enabled=True,
    pdf_mode="fast",
    include_pptx_notes=True,
)

# Tenant wide overrides keyed by the tenant identifier.
TENANT_OVERRIDES: Dict[str, Mapping[str, object]] = {}

# Collection specific overrides keyed by (tenant_id, collection_id).
COLLECTION_OVERRIDES: Dict[Tuple[str, UUID], Mapping[str, object]] = {}

# Workflow specific overrides keyed by (tenant_id, workflow_id).
WORKFLOW_OVERRIDES: Dict[Tuple[str, str], Mapping[str, object]] = {}


_POLICY_KEYS = {
    "caption_min_confidence",
    "pdf_ocr_enabled",
    "pdf_mode",
    "include_pptx_notes",
}


class PolicyError(ValueError):
    """Raised when policy overrides are malformed."""


def _normalise_bool(value: object, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    raise PolicyError(f"policy_invalid_{key}")


def _normalise_confidence(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if 0.0 <= numeric <= 1.0:
            return numeric
    raise PolicyError("policy_invalid_caption_min_confidence")


def _normalise_pdf_mode(value: object) -> str:
    if isinstance(value, str):
        candidate = value.strip()
        if candidate:
            return candidate
    raise PolicyError("policy_invalid_pdf_mode")


def _normalise_override_mapping(mapping: Mapping[str, object], scope: str) -> Dict[str, object]:
    normalised: Dict[str, object] = {}
    for key, value in mapping.items():
        if key not in _POLICY_KEYS:
            raise PolicyError(f"policy_unknown_key_{scope}")
        if key == "caption_min_confidence":
            normalised[key] = _normalise_confidence(value)
        elif key == "pdf_ocr_enabled":
            normalised[key] = _normalise_bool(value, key="pdf_ocr_enabled")
        elif key == "include_pptx_notes":
            normalised[key] = _normalise_bool(value, key="include_pptx_notes")
        elif key == "pdf_mode":
            normalised[key] = _normalise_pdf_mode(value)
    return normalised


def _apply_overrides(
    base: Dict[str, object],
    overrides: Mapping[str, object] | None,
    *,
    scope: str,
) -> None:
    if overrides is None:
        return
    if not isinstance(overrides, Mapping):
        raise PolicyError(f"policy_override_type_{scope}")
    base.update(_normalise_override_mapping(overrides, scope))


def get_policy(
    tenant_id: str,
    collection_id: Optional[UUID],
    workflow_id: Optional[str],
) -> DocumentPolicy:
    """Resolve the effective document policy for a tenant scope."""

    resolved: Dict[str, object] = {
        "caption_min_confidence": DEFAULT_POLICY.caption_min_confidence,
        "pdf_ocr_enabled": DEFAULT_POLICY.pdf_ocr_enabled,
        "pdf_mode": DEFAULT_POLICY.pdf_mode,
        "include_pptx_notes": DEFAULT_POLICY.include_pptx_notes,
    }

    tenant_override = TENANT_OVERRIDES.get(tenant_id)
    _apply_overrides(resolved, tenant_override, scope="tenant")

    if collection_id is not None:
        collection_override = COLLECTION_OVERRIDES.get((tenant_id, collection_id))
        _apply_overrides(resolved, collection_override, scope="collection")

    if workflow_id is not None:
        workflow_override = WORKFLOW_OVERRIDES.get((tenant_id, workflow_id))
        _apply_overrides(resolved, workflow_override, scope="workflow")

    return DocumentPolicy(**resolved)


PolicyProvider = Callable[[str, Optional[UUID], Optional[str]], DocumentPolicy]


__all__ = [
    "COLLECTION_OVERRIDES",
    "DEFAULT_POLICY",
    "DocumentPolicy",
    "PolicyError",
    "PolicyProvider",
    "TENANT_OVERRIDES",
    "WORKFLOW_OVERRIDES",
    "get_policy",
]

