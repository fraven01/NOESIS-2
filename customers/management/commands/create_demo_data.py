from django.core.management.base import BaseCommand
from django.core.files.base import ContentFile
from django_tenants.utils import schema_context, get_public_schema_name

from customers.models import Domain, Tenant
from users.models import User
from profiles.models import UserProfile
from organizations.models import Organization, OrgMembership
from projects.models import Project
from documents.models import Document, DocumentType
from organizations.utils import set_current_organization


class Command(BaseCommand):
    """Create demo tenant, user, and sample projects/documents."""

    help = "Ensure demo tenant, user and sample data exist"

    def handle(self, *args, **options):
        # Ensure tenant/domain from the public schema
        with schema_context(get_public_schema_name()):
            tenant, _ = Tenant.objects.get_or_create(
                schema_name="demo", defaults={"name": "Demo Tenant"}
            )
            Domain.objects.get_or_create(
                domain="demo.localhost", tenant=tenant, is_primary=True
            )
        # Ensure tenant schema exists for data seeding
        tenant.create_schema(check_if_exists=True)

        with schema_context(tenant.schema_name):
            user, created = User.objects.get_or_create(
                username="demo", defaults={"email": "demo@example.com"}
            )
            changed = False
            if created or not user.password:
                user.set_password("demo")
                changed = True
            if not user.is_staff:
                user.is_staff = True
                changed = True
            if not user.is_superuser:
                user.is_superuser = True
                changed = True
            if changed:
                user.save()

            UserProfile.objects.update_or_create(
                user=user, defaults={"role": UserProfile.Roles.ADMIN}
            )

            org, _ = Organization.objects.get_or_create(
                slug="demo", defaults={"name": "Demo Organization"}
            )
            OrgMembership.objects.get_or_create(
                organization=org,
                user=user,
                defaults={"role": OrgMembership.Role.ADMIN},
            )

            doc_type, _ = DocumentType.objects.get_or_create(
                name="Demo Type", defaults={"description": "Demo documents"}
            )

            with set_current_organization(org):
                project1, _ = Project.objects.get_or_create(
                    name="Demo Project 1",
                    defaults={
                        "description": "Erstes Demo-Projekt",
                        "owner": user,
                        "organization": org,
                    },
                )
                project2, _ = Project.objects.get_or_create(
                    name="Demo Project 2",
                    defaults={
                        "description": "Zweites Demo-Projekt",
                        "owner": user,
                        "organization": org,
                    },
                )

                if not Document.objects.filter(
                    project=project1, title="Demo Document 1"
                ).exists():
                    Document.objects.create(
                        title="Demo Document 1",
                        file=ContentFile(b"Demo content 1", name="demo1.txt"),
                        type=doc_type,
                        project=project1,
                        owner=user,
                    )
                if not Document.objects.filter(
                    project=project2, title="Demo Document 2"
                ).exists():
                    Document.objects.create(
                        title="Demo Document 2",
                        file=ContentFile(b"Demo content 2", name="demo2.txt"),
                        type=doc_type,
                        project=project2,
                        owner=user,
                    )

        self.stdout.write(self.style.SUCCESS("Demo data ensured"))
