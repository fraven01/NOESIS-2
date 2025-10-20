from uuid import uuid4

import pytest

from documents.policies import (
    COLLECTION_OVERRIDES,
    DEFAULT_POLICY,
    TENANT_OVERRIDES,
    WORKFLOW_OVERRIDES,
    PolicyError,
    get_policy,
)


@pytest.fixture(autouse=True)
def clear_policy_overrides():
    TENANT_OVERRIDES.clear()
    COLLECTION_OVERRIDES.clear()
    WORKFLOW_OVERRIDES.clear()
    yield
    TENANT_OVERRIDES.clear()
    COLLECTION_OVERRIDES.clear()
    WORKFLOW_OVERRIDES.clear()


def test_get_policy_returns_defaults_when_no_overrides():
    policy = get_policy("tenant-a", None, None)
    assert policy == DEFAULT_POLICY


def test_policy_override_precedence():
    tenant_id = "tenant-a"
    collection_id = uuid4()
    workflow_id = "workflow-1"

    TENANT_OVERRIDES[tenant_id] = {"caption_min_confidence": 0.1}
    COLLECTION_OVERRIDES[(tenant_id, collection_id)] = {
        "caption_min_confidence": 0.3,
        "pdf_mode": "accurate",
    }
    WORKFLOW_OVERRIDES[(tenant_id, workflow_id)] = {
        "caption_min_confidence": 0.8,
        "include_pptx_notes": False,
    }

    policy = get_policy(tenant_id, collection_id, workflow_id)
    assert policy.caption_min_confidence == 0.8
    assert policy.pdf_mode == "accurate"
    assert policy.include_pptx_notes is False
    assert policy.pdf_ocr_enabled is DEFAULT_POLICY.pdf_ocr_enabled


def test_policy_missing_keys_use_defaults():
    tenant_id = "tenant-a"
    TENANT_OVERRIDES[tenant_id] = {"pdf_ocr_enabled": False}

    policy = get_policy(tenant_id, None, None)
    assert policy.pdf_ocr_enabled is False
    assert policy.caption_min_confidence == DEFAULT_POLICY.caption_min_confidence
    assert policy.pdf_mode == DEFAULT_POLICY.pdf_mode
    assert policy.include_pptx_notes is DEFAULT_POLICY.include_pptx_notes


@pytest.mark.parametrize(
    "override",
    [
        {"caption_min_confidence": "high"},
        {"pdf_ocr_enabled": "yes"},
        {"include_pptx_notes": 1},
        {"pdf_mode": ""},
    ],
)
def test_policy_rejects_invalid_override_values(override):
    TENANT_OVERRIDES["tenant-a"] = override
    with pytest.raises(PolicyError):
        get_policy("tenant-a", None, None)


def test_policy_rejects_unknown_override_keys():
    TENANT_OVERRIDES["tenant-a"] = {"unknown": True}
    with pytest.raises(PolicyError):
        get_policy("tenant-a", None, None)


def test_policy_rejects_non_mapping_overrides():
    TENANT_OVERRIDES["tenant-a"] = {"pdf_ocr_enabled": False}
    WORKFLOW_OVERRIDES[("tenant-a", "wf")] = "invalid"  # type: ignore[assignment]
    with pytest.raises(PolicyError):
        get_policy("tenant-a", None, "wf")
