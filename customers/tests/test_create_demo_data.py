import json
from io import StringIO

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from django_tenants.utils import schema_context

from customers.models import Tenant
from documents.models import Document
from organizations.models import Organization
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
def test_create_demo_data_wipe_removes_seeded_content():
    call_command("create_demo_data")

    wipe_stdout = StringIO()
    call_command("create_demo_data", "--wipe", stdout=wipe_stdout)
    summary = json.loads(wipe_stdout.getvalue())

    assert summary["event"] == "seed.wipe"

    with schema_context("demo"):
        org = Organization.objects.get(slug="demo")
        with set_current_organization(org):
            assert Project.objects.filter(organization=org).count() == 0
            assert Document.objects.filter(project__organization=org).count() == 0


@pytest.mark.django_db
def test_create_demo_data_baseline_profile_counts():
    stdout = StringIO()
    call_command("create_demo_data", "--profile", "baseline", stdout=stdout)
    summary = json.loads(stdout.getvalue())

    assert summary["profile"] == "baseline"
    assert summary["counts"] == {"projects": 2, "documents": 2, "users": 1, "orgs": 1}

    with schema_context("demo"):
        org = Organization.objects.get(slug="demo")
        with set_current_organization(org):
            assert Project.objects.filter(organization=org).count() == 2
            assert Document.objects.filter(project__organization=org).count() == 2


@pytest.mark.django_db
def test_create_demo_data_baseline_disallows_overrides():
    with pytest.raises(CommandError):
        call_command("create_demo_data", "--profile", "baseline", "--projects", "4")
