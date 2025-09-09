from django.core.management.base import BaseCommand
from django.utils.text import slugify

from organizations.models import Organization
from projects.models import Project


class Command(BaseCommand):
    """Assign default organizations to projects without one.

    Usage::

        python manage.py assign_default_org
    """

    help = "Assign default organizations to projects without one"

    def handle(self, *args, **options):
        for project in Project.objects.filter(organization__isnull=True):
            slug_part = getattr(project, "slug", str(project.pk))
            org = Organization.objects.create(
                name=f"Legacy Org {slug_part}",
                slug=slugify(f"legacy-org-{slug_part}"),
            )
            project.organization = org
            project.save(update_fields=["organization"])
            self.stdout.write(
                self.style.SUCCESS(f"Assigned {org} to project {project.pk}")
            )
