import json
import os
from io import StringIO

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.core.management.base import CommandError
from django_tenants.utils import schema_context

from customers.models import Tenant
from documents.models import Document, DocumentType
from organizations.models import Organization, OrgMembership
from organizations.utils import set_current_organization
from profiles.models import UserProfile
from projects.models import Project
from users.models import User


@pytest.mark.django_db
def test_create_demo_data_idempotent():
    first_stdout = StringIO()
    call_command("create_demo_data", stdout=first_stdout)
    first_summary = json.loads(first_stdout.getvalue())

    second_stdout = StringIO()
    call_command("create_demo_data", stdout=second_stdout)
    second_summary = json.loads(second_stdout.getvalue())

    assert first_summary["event"] == "seed.done"
    assert first_summary["profile"] == "demo"
    assert first_summary["counts"]["projects"] >= 2
    assert first_summary["counts"]["documents"] >= first_summary["counts"]["projects"]
    assert second_summary == first_summary

    assert Tenant.objects.filter(schema_name="demo").count() == 1

    with schema_context("demo"):
        user_qs = User.objects.filter(username="demo")
        assert user_qs.count() == 1
        user = user_qs.get()

        profile = UserProfile.objects.get(user=user)
        assert profile.role == UserProfile.Roles.ADMIN

        org = Organization.objects.get(slug="demo")
        with set_current_organization(org):
            assert (
                Project.objects.filter(name="Demo Project 1", organization=org).count()
                == 1
            )
            assert (
                Project.objects.filter(name="Demo Project 2", organization=org).count()
                == 1
            )
            assert (
                Document.objects.filter(
                    title="Demo Document 1", project__organization=org
                ).count()
                == 1
            )
            assert (
                Document.objects.filter(
                    title="Demo Document 2", project__organization=org
                ).count()
                == 1
            )


@pytest.mark.django_db
def test_profile_baseline_counts_stable():
    first_stdout = StringIO()
    call_command(
        "create_demo_data",
        "--profile",
        "baseline",
        "--seed",
        "1337",
        stdout=first_stdout,
    )
    first_summary = json.loads(first_stdout.getvalue())

    second_stdout = StringIO()
    call_command(
        "create_demo_data",
        "--profile",
        "baseline",
        "--seed",
        "1337",
        stdout=second_stdout,
    )
    second_summary = json.loads(second_stdout.getvalue())

    assert first_summary == second_summary
    assert second_summary["profile"] == "baseline"
    assert second_summary["counts"] == {
        "projects": 2,
        "documents": 2,
        "users": 1,
        "orgs": 1,
    }

    with schema_context("demo"):
        org = Organization.objects.get(slug="demo")
        with set_current_organization(org):
            projects = list(Project.objects.filter(organization=org).order_by("name"))
            assert len(projects) == 2
            for project in projects:
                documents = list(Document.objects.filter(project=project))
                assert len(documents) == 1
                doc = documents[0]
                assert os.path.basename(doc.file.name).startswith("doc-")


@pytest.mark.django_db
def test_create_demo_data_wipe_include_org_removes_org():
    call_command("create_demo_data")

    wipe_stdout = StringIO()
    call_command(
        "create_demo_data",
        "--wipe",
        "--include-org",
        stdout=wipe_stdout,
    )
    summary = json.loads(wipe_stdout.getvalue())

    assert summary["event"] == "seed.wipe.done"
    assert summary["counts"]["orgs"] == 1

    with schema_context("demo"):
        assert not Organization.objects.filter(slug="demo").exists()
        assert not OrgMembership.objects.filter(organization__slug="demo").exists()


@pytest.mark.django_db
def test_create_demo_data_baseline_disallows_overrides():
    with pytest.raises(CommandError):
        call_command("create_demo_data", "--profile", "baseline", "--projects", "4")


@pytest.mark.django_db
def test_create_demo_data_profile_override_bounds():
    with pytest.raises(CommandError) as excinfo:
        call_command("create_demo_data", "--profile", "demo", "--projects", "0")

    assert (
        str(excinfo.value)
        == "Invalid overrides: --projects and --docs-per-project must be between 1 and 200 for this profile."
    )

    with pytest.raises(CommandError) as docs_exc:
        call_command(
            "create_demo_data",
            "--profile",
            "chaos",
            "--docs-per-project",
            "500",
        )

    assert (
        str(docs_exc.value)
        == "Invalid overrides: --projects and --docs-per-project must be between 1 and 200 for this profile."
    )


@pytest.mark.django_db
def test_profile_demo_overrides():
    first_stdout = StringIO()
    call_command(
        "create_demo_data",
        "--profile",
        "demo",
        "--seed",
        "42",
        "--projects",
        "6",
        "--docs-per-project",
        "4",
        stdout=first_stdout,
    )
    first_summary = json.loads(first_stdout.getvalue())

    second_stdout = StringIO()
    call_command(
        "create_demo_data",
        "--profile",
        "demo",
        "--seed",
        "42",
        "--projects",
        "6",
        "--docs-per-project",
        "4",
        stdout=second_stdout,
    )
    second_summary = json.loads(second_stdout.getvalue())

    assert first_summary == second_summary
    assert second_summary["profile"] == "demo"
    assert second_summary["counts"]["projects"] == 6
    assert second_summary["counts"]["documents"] == 24

    with schema_context("demo"):
        org = Organization.objects.get(slug="demo")
        with set_current_organization(org):
            projects = list(Project.objects.filter(organization=org).order_by("name"))
            assert len(projects) == 6
            for project in projects:
                assert Document.objects.filter(project=project).count() == 4

            documents = list(Document.objects.filter(project__organization=org))
            txt_docs = [doc for doc in documents if doc.file.name.endswith(".txt")]
            md_docs = [doc for doc in documents if doc.file.name.endswith(".md")]
            json_docs = [doc for doc in documents if doc.file.name.endswith(".json")]

            assert txt_docs
            assert md_docs
            assert json_docs

            txt_word_counts = []
            for doc in txt_docs:
                doc.file.open("rb")
                txt_content = doc.file.read().decode("utf-8")
                doc.file.close()
                txt_lines = txt_content.splitlines()[1:]
                txt_words = [
                    word for segment in txt_lines for word in segment.split() if word
                ]
                if txt_words:
                    txt_word_counts.append(len(txt_words))
            assert any(20 <= count <= 60 for count in txt_word_counts)

            md_has_heading_and_list = False
            for doc in md_docs:
                doc.file.open("rb")
                md_content = doc.file.read().decode("utf-8")
                doc.file.close()
                md_lines = md_content.splitlines()
                if not md_lines:
                    continue
                if md_lines[0].startswith("# ") and all(
                    line.startswith("- ") for line in md_lines[2:]
                ):
                    md_has_heading_and_list = True
                    break
            assert md_has_heading_and_list

            for doc in json_docs:
                doc.file.open("rb")
                json_content = doc.file.read().decode("utf-8")
                doc.file.close()
                payload = json.loads(json_content)
                assert set(payload) == {"id", "project", "idx", "title"}


@pytest.mark.django_db
def test_wipe_removes_seeded_projects_and_docs_only():
    call_command("create_demo_data", "--seed", "1337")

    with schema_context("demo"):
        user = User.objects.get(username="demo")
        org = Organization.objects.get(slug="demo")
        doc_type = DocumentType.objects.get(name="Demo Type")
        with set_current_organization(org):
            manual_project = Project.objects.create(
                name="Manual Project",
                description="Manually created",
                owner=user,
                organization=org,
            )
            manual_document = Document.objects.create(
                title="Manual Document",
                project=manual_project,
                owner=user,
                type=doc_type,
            )
            manual_document.file.save(
                "manual.txt", SimpleUploadedFile("manual.txt", b"manual"), save=True
            )

            seeded_projects = [
                project.id
                for project in Project.objects.filter(organization=org)
                if project.description.startswith("[demo-seed:")
            ]
            seeded_documents = [
                document.id
                for document in Document.objects.filter(project__organization=org)
                if os.path.basename(document.file.name).startswith("doc-")
            ]

    wipe_stdout = StringIO()
    call_command("create_demo_data", "--wipe", stdout=wipe_stdout)
    summary = json.loads(wipe_stdout.getvalue())

    assert summary["event"] == "seed.wipe.done"
    assert summary["counts"]["projects"] == len(seeded_projects)
    assert summary["counts"]["documents"] == len(seeded_documents)
    assert summary["counts"]["users"] == 0
    assert summary["counts"]["orgs"] == 0

    with schema_context("demo"):
        org = Organization.objects.get(slug="demo")
        with set_current_organization(org):
            assert Project.objects.filter(id__in=seeded_projects).count() == 0
            assert Document.objects.filter(id__in=seeded_documents).count() == 0
            assert Project.objects.filter(id=manual_project.id).exists()
            assert Document.objects.filter(id=manual_document.id).exists()

        assert DocumentType.objects.filter(name="Demo Type").exists()
        assert OrgMembership.objects.filter(organization=org).exists()


@pytest.mark.django_db
def test_chaos_creates_flagged_invalid_documents():
    first_stdout = StringIO()
    call_command(
        "create_demo_data",
        "--profile",
        "chaos",
        "--seed",
        "42",
        stdout=first_stdout,
    )
    first_summary = json.loads(first_stdout.getvalue())

    second_stdout = StringIO()
    call_command(
        "create_demo_data",
        "--profile",
        "chaos",
        "--seed",
        "42",
        stdout=second_stdout,
    )
    second_summary = json.loads(second_stdout.getvalue())

    assert first_summary == second_summary
    assert second_summary["profile"] == "chaos"

    with schema_context("demo"):
        org = Organization.objects.get(slug="demo")
        with set_current_organization(org):
            documents = list(Document.objects.filter(project__organization=org))
            flagged = [
                doc
                for doc in documents
                if doc.status == Document.STATUS_PROCESSING
                or (
                    os.path.basename(doc.file.name).startswith("doc-")
                    and doc.file.size == 0
                )
            ]
            assert flagged
            for doc in flagged:
                assert doc.meta.get("invalid") is True
                assert doc.meta.get("error")

            invalid_json_docs = []
            for doc in documents:
                if not doc.file.name.endswith(".json"):
                    continue
                doc.file.open("rb")
                content = doc.file.read().decode("utf-8")
                doc.file.close()
                if (
                    doc.status == Document.STATUS_PROCESSING
                    and not content.strip().endswith("}")
                ):
                    invalid_json_docs.append(doc)

            assert 1 <= len(invalid_json_docs) <= 2
            for doc in invalid_json_docs:
                assert doc.meta.get("reason") == "invalid_json"


@pytest.mark.django_db
def test_check_demo_data_passes_for_demo():
    seed_stdout = StringIO()
    call_command("create_demo_data", stdout=seed_stdout)
    expected_summary = json.loads(seed_stdout.getvalue())

    check_stdout = StringIO()
    call_command("check_demo_data", stdout=check_stdout)
    payload = json.loads(check_stdout.getvalue())

    assert payload["event"] == "check.ok"
    assert payload["profile"] == "demo"
    assert payload["counts"]["projects"] == expected_summary["counts"]["projects"]
    assert payload["counts"]["documents"] == expected_summary["counts"]["documents"]


@pytest.mark.django_db
def test_check_demo_data_fails_on_missing_type():
    call_command("create_demo_data")

    with schema_context("demo"):
        DocumentType.objects.filter(name="Demo Type").delete()

    stdout = StringIO()
    with pytest.raises(SystemExit):
        call_command("check_demo_data", stdout=stdout)

    payload = json.loads(stdout.getvalue())
    assert payload["event"] == "check.failed"
    assert payload["reason"] == "document_type_missing"


@pytest.mark.django_db
def test_check_demo_data_fails_on_duplicate_project_slug():
    call_command("create_demo_data")

    with schema_context("demo"):
        org = Organization.objects.get(slug="demo")
        user = User.objects.get(username="demo")
        with set_current_organization(org):
            Project.objects.create(
                name="Duplicate Project",
                description="[demo-seed:proj-01] Duplicate",
                status=Project.STATUS_INITIATED,
                owner=user,
                organization=org,
            )

    stdout = StringIO()
    with pytest.raises(SystemExit):
        call_command("check_demo_data", stdout=stdout)

    payload = json.loads(stdout.getvalue())
    assert payload["event"] == "check.failed"
    assert payload["reason"] == "duplicate_slug"
    assert payload["model"] == "Project"
    assert payload["slug"] == "proj-01"


@pytest.mark.django_db
def test_check_demo_data_fails_on_duplicate_document_slug():
    call_command("create_demo_data")

    with schema_context("demo"):
        org = Organization.objects.get(slug="demo")
        user = User.objects.get(username="demo")
        doc_type = DocumentType.objects.get(name="Demo Type")
        with set_current_organization(org):
            project = Project.objects.filter(organization=org).first()
            existing = Document.objects.filter(project=project).first()
            Document.objects.create(
                title="Duplicate Document",
                file=existing.file.name,
                status=Document.STATUS_COMPLETED,
                type=doc_type,
                project=project,
                owner=user,
                meta={},
            )

    stdout = StringIO()
    with pytest.raises(SystemExit):
        call_command("check_demo_data", stdout=stdout)

    payload = json.loads(stdout.getvalue())
    assert payload["event"] == "check.failed"
    assert payload["reason"] == "duplicate_slug"
    assert payload["model"] == "Document"
    assert payload["slug"].startswith("doc-proj-")


@pytest.mark.django_db
def test_check_demo_data_chaos_requires_invalid_documents():
    call_command("create_demo_data", "--profile", "chaos")

    with schema_context("demo"):
        org = Organization.objects.get(slug="demo")
        with set_current_organization(org):
            for document in Document.objects.all():
                document.meta = {}
                document.save(update_fields=["meta"])

    stdout = StringIO()
    with pytest.raises(SystemExit):
        call_command("check_demo_data", "--profile", "chaos", stdout=stdout)

    payload = json.loads(stdout.getvalue())
    assert payload["event"] == "check.failed"
    assert payload["reason"] == "chaos_missing_invalids"
