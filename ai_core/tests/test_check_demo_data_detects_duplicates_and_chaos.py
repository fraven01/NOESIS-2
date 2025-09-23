import json
from io import StringIO

import pytest
from django.core.management import call_command
from django_tenants.utils import schema_context

from organizations.models import Organization
from organizations.utils import set_current_organization
from projects.models import Project
from users.models import User


def _wipe_demo_seed():
    call_command("create_demo_data", "--wipe", "--include-org")


@pytest.mark.django_db
def test_check_demo_data_detects_duplicates_and_chaos():
    _wipe_demo_seed()
    call_command("create_demo_data", "--profile", "demo", "--seed", "1337")

    with schema_context("demo"):
        org = Organization.objects.get(slug="demo")
        user = User.objects.get(username="demo")
        with set_current_organization(org):
            Project.objects.create(
                name="Duplicate Demo Project",
                description="[demo-seed:proj-01] Duplicate",
                owner=user,
                organization=org,
            )

    duplicate_stdout = StringIO()
    with pytest.raises(SystemExit) as excinfo:
        call_command("check_demo_data", stdout=duplicate_stdout)

    assert excinfo.value.code == 1
    duplicate_payload = json.loads(duplicate_stdout.getvalue())
    assert duplicate_payload["reason"] == "duplicate_slug"
    assert duplicate_payload["model"] == "Project"

    _wipe_demo_seed()
    call_command("create_demo_data", "--profile", "chaos", "--seed", "1337")

    chaos_stdout = StringIO()
    call_command(
        "check_demo_data",
        "--profile",
        "chaos",
        "--seed",
        "1337",
        stdout=chaos_stdout,
    )
    chaos_payload = json.loads(chaos_stdout.getvalue())
    assert chaos_payload["event"] == "check.ok"
    assert chaos_payload.get("reason") != "chaos_missing_invalids"

    _wipe_demo_seed()
