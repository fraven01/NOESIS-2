import pytest
from django.core.management import call_command
from django_tenants.utils import schema_context

from customers.models import Tenant
from users.models import User
from profiles.models import UserProfile
from organizations.models import Organization
from projects.models import Project
from documents.models import Document
from organizations.utils import set_current_organization


@pytest.mark.django_db
def test_create_demo_data_idempotent():
    call_command("create_demo_data")
    call_command("create_demo_data")

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
