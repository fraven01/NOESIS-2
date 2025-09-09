import pytest
from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.utils.text import slugify


@pytest.mark.django_db(transaction=True)
def test_project_organization_data_migration():
    executor = MigrationExecutor(connection)
    executor.migrate([("organizations", "0001_initial"), ("projects", "0001_initial")])
    old_apps = executor.loader.project_state(
        [("organizations", "0001_initial"), ("projects", "0001_initial")]
    ).apps

    User = old_apps.get_model("users", "User")
    Project = old_apps.get_model("projects", "Project")

    user = User.objects.create(username="u")
    project = Project.objects.create(name="Legacy", description="desc", owner=user)
    project_id = project.pk

    executor = MigrationExecutor(connection)
    executor.migrate(
        [("organizations", "0001_initial"), ("projects", "0002_project_organization")]
    )
    new_apps = executor.loader.project_state(
        [("organizations", "0001_initial"), ("projects", "0002_project_organization")]
    ).apps

    Project = new_apps.get_model("projects", "Project")
    Organization = new_apps.get_model("organizations", "Organization")
    project = Project.objects.get(pk=project_id)
    assert project.organization.name == f"Legacy Org {project_id}"
    assert project.organization.slug == slugify(f"legacy-org-{project_id}")
    assert Organization.objects.count() == 1

    executor = MigrationExecutor(connection)
    executor.migrate([("organizations", "0001_initial"), ("projects", "0001_initial")])
    rev_apps = executor.loader.project_state(
        [("organizations", "0001_initial"), ("projects", "0001_initial")]
    ).apps

    Project = rev_apps.get_model("projects", "Project")
    Organization = rev_apps.get_model("organizations", "Organization")
    project = Project.objects.get(pk=project_id)
    assert not hasattr(project, "organization")
    assert Organization.objects.count() == 0

    executor = MigrationExecutor(connection)
    executor.migrate(executor.loader.graph.leaf_nodes())
