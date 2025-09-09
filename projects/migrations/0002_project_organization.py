from django.db import migrations, models
import django.db.models.deletion
from django.utils.text import slugify


def create_organizations(apps, schema_editor):
    Project = apps.get_model("projects", "Project")
    Organization = apps.get_model("organizations", "Organization")
    for project in Project.objects.all():
        slug_part = getattr(project, "slug", str(project.pk))
        org = Organization.objects.create(
            name=f"Legacy Org {slug_part}",
            slug=slugify(f"legacy-org-{slug_part}"),
        )
        project.organization = org
        project.save(update_fields=["organization"])


def reverse_create_organizations(apps, schema_editor):
    Project = apps.get_model("projects", "Project")
    for project in Project.objects.all():
        org = project.organization
        if org and org.name.startswith("Legacy Org "):
            project.organization = None
            project.save(update_fields=["organization"])
            org.delete()


class Migration(migrations.Migration):
    dependencies = [
        ("organizations", "0001_initial"),
        ("projects", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="project",
            name="organization",
            field=models.ForeignKey(
                to="organizations.organization",
                on_delete=django.db.models.deletion.CASCADE,
                null=True,
            ),
        ),
        migrations.RunPython(create_organizations, reverse_create_organizations),
        migrations.AlterField(
            model_name="project",
            name="organization",
            field=models.ForeignKey(
                to="organizations.organization",
                on_delete=django.db.models.deletion.CASCADE,
            ),
        ),
    ]
