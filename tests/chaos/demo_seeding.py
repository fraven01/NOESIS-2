"""Chaos coverage for demo seeding profiles and invalid document scenarios."""

from __future__ import annotations

import json
from io import StringIO
import random
from pathlib import Path

import pytest
from django.core.management import call_command
from django_tenants.utils import schema_context

from ai_core.infra import object_store, rate_limit
from common.constants import (
    META_CASE_ID_KEY,
    META_TENANT_ID_KEY,
    META_TENANT_SCHEMA_KEY,
)
from customers.management.commands.create_demo_data import DemoDatasetBuilder
from documents.models import Document
from organizations.models import Organization
from organizations.utils import set_current_organization
from faker import Faker

pytestmark = pytest.mark.chaos
pytest_plugins = ["tests.chaos.fixtures"]

_DEFAULT_SEED = 1337


def _seed_profile(profile: str, *, seed: int) -> None:
    """Reset demo data and seed the requested profile deterministically."""

    call_command("create_demo_data", "--wipe", "--include-org")
    call_command(
        "create_demo_data",
        "--profile",
        profile,
        "--seed",
        str(seed),
    )


def _collect_demo_documents() -> list[Document]:
    with schema_context("demo"):
        organization = Organization.objects.get(slug="demo")
        with set_current_organization(organization):
            return list(Document.objects.filter(project__organization=organization))


def _document_slug(document: Document) -> str:
    return Path(document.file.name or "").stem


def _matches_reason(document: Document, reason: str) -> bool:
    meta = document.meta if isinstance(document.meta, dict) else {}
    if meta.get("reason") == reason:
        return True
    issues = meta.get("issues")
    if isinstance(issues, (list, tuple)) and reason in issues:
        return True
    if reason == "empty_content" and document.file and document.file.size == 0:
        return True
    return False


def _build_chaos_dataset(profile: str, seed: int):
    rng = random.Random(seed)
    faker = Faker("de_DE")
    faker.seed_instance(seed)
    return DemoDatasetBuilder(profile, faker, rng).build()


@pytest.mark.django_db
@pytest.mark.parametrize("profile", ["baseline", "demo", "chaos"])
def test_seed_profiles_emit_structured_events(profile: str, chaos_env) -> None:
    """Ensure all supported seed profiles emit structured check events."""

    chaos_env.set_seed_profile(profile)
    chaos_env.set_seed_value(_DEFAULT_SEED)

    selected_profile = chaos_env.values["DEMO_SEED_PROFILE"]
    seed_value = int(chaos_env.values["DEMO_SEED_SEED"])

    _seed_profile(selected_profile, seed=seed_value)

    stdout = StringIO()
    call_command(
        "check_demo_data",
        "--profile",
        selected_profile,
        "--seed",
        str(seed_value),
        stdout=stdout,
    )

    payload = json.loads(stdout.getvalue())

    assert payload["event"] == "check.ok"
    assert payload["profile"] == selected_profile
    counts = payload["counts"]
    assert counts["projects"] >= 1
    assert counts["documents"] >= 1
    if selected_profile == "chaos":
        assert payload.get("invalid_documents", 0) >= 1


@pytest.mark.django_db
@pytest.mark.parametrize(
    "reason",
    [
        "invalid_json",
        "empty_content",
        "title_length",
        "markdown_unclosed",
    ],
)
def test_chaos_profile_exposes_invalid_documents(reason: str, chaos_env) -> None:
    """Chaos seeds must provide representative invalid document fixtures."""

    chaos_env.set_seed_profile("chaos")
    chaos_env.set_seed_value(_DEFAULT_SEED)

    profile = chaos_env.values["DEMO_SEED_PROFILE"]
    seed_value = int(chaos_env.values["DEMO_SEED_SEED"])

    _seed_profile(profile, seed=seed_value)

    documents = _collect_demo_documents()

    assert any(
        _matches_reason(document, reason) for document in documents
    ), f"Expected demo seed to contain a document flagged with {reason}"


def test_chaos_dataset_marks_missing_type_documents(chaos_env) -> None:
    """Dataset generation still includes missing-type specs for E2E coverage."""

    chaos_env.set_seed_profile("chaos")
    chaos_env.set_seed_value(_DEFAULT_SEED)

    profile = chaos_env.values["DEMO_SEED_PROFILE"]
    seed_value = int(chaos_env.values["DEMO_SEED_SEED"])

    dataset = _build_chaos_dataset(profile, seed_value)

    assert any(
        doc.missing_type for project in dataset.projects for doc in project.documents
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    "reason",
    [
        "invalid_json",
        "empty_content",
        "markdown_unclosed",
        "title_length",
    ],
)
def test_agent_intake_handles_invalid_seed_documents(
    reason: str,
    client,
    chaos_env,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    test_tenant_schema_name: str,
) -> None:
    """Agent intake must remain robust when demo data contains faulty docs."""

    chaos_env.set_seed_profile("chaos")
    chaos_env.set_seed_value(_DEFAULT_SEED)

    profile = chaos_env.values["DEMO_SEED_PROFILE"]
    seed_value = int(chaos_env.values["DEMO_SEED_SEED"])

    _seed_profile(profile, seed=seed_value)

    monkeypatch.setattr(rate_limit, "check", lambda tenant, now=None: True)
    monkeypatch.setattr(object_store, "BASE_PATH", tmp_path)

    documents = _collect_demo_documents()

    target = next((doc for doc in documents if _matches_reason(doc, reason)), None)
    assert (
        target is not None
    ), f"Document with reason {reason} not present in chaos seed"

    meta = target.meta if isinstance(target.meta, dict) else {}
    payload_document = {
        "slug": _document_slug(target),
        "status": target.status,
        "meta": meta,
    }
    if reason == "empty_content":
        payload_document["body"] = ""
    elif reason == "invalid_json":
        with target.file.open("rb") as handle:
            payload_document["body"] = handle.read().decode("utf-8", errors="ignore")

    response = client.post(
        "/ai/intake/",
        data=json.dumps({"documents": [payload_document]}),
        content_type="application/json",
        **{
            META_TENANT_ID_KEY: test_tenant_schema_name,
            META_TENANT_SCHEMA_KEY: test_tenant_schema_name,
            META_CASE_ID_KEY: f"chaos-doc-{reason}",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body.get("received") is True
    assert body.get("tenant") == test_tenant_schema_name
