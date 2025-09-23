import json
import random
from typing import Dict, Iterable, Tuple

from django.core.management.base import BaseCommand, CommandError
from django_tenants.utils import schema_context
from faker import Faker

from customers.management.commands.create_demo_data import (
    DemoDatasetApplier,
    DemoDatasetBuilder,
)
from documents.models import Document, DocumentType
from organizations.models import Organization
from organizations.utils import set_current_organization
from profiles.models import UserProfile
from projects.models import Project
from users.models import User


class Command(BaseCommand):
    help = "Perform smoke-checks for the demo data seeding command"

    def add_arguments(self, parser):
        parser.add_argument(
            "--profile",
            choices=["baseline", "demo", "heavy", "chaos"],
            default="demo",
            help="Dataset profile to validate (default: demo)",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=1337,
            help="Deterministic seed for faker/random (default: 1337)",
        )
        parser.add_argument(
            "--projects",
            type=int,
            help="Override project count (demo/heavy/chaos profiles only)",
        )
        parser.add_argument(
            "--docs-per-project",
            type=int,
            help="Override documents per project (demo/heavy/chaos profiles only)",
        )

    def handle(self, *args, **options):
        profile = options.get("profile", "demo")
        seed = options.get("seed", 1337)
        projects_override = options.get("projects")
        docs_override = options.get("docs_per_project")

        if profile == "baseline" and (
            projects_override is not None or docs_override is not None
        ):
            raise CommandError(
                "--projects/--docs-per-project are only allowed for demo, heavy or chaos profiles"
            )

        if projects_override is not None and projects_override <= 0:
            raise CommandError("--projects must be a positive integer")
        if docs_override is not None and docs_override <= 0:
            raise CommandError("--docs-per-project must be a positive integer")

        rng = random.Random(seed)
        faker = Faker("de_DE")
        faker.seed_instance(seed)

        dataset = DemoDatasetBuilder(
            profile,
            faker,
            rng,
            projects_override=projects_override,
            docs_override=docs_override,
        ).build()

        expected_counts = {
            "projects": dataset.project_count,
            "documents": self._expected_document_count(dataset.projects),
            "users": 1,
            "orgs": 1,
        }

        with schema_context("demo"):
            user_qs = User.objects.filter(username="demo")
            user_count = user_qs.count()
            if user_count != 1:
                self._fail(
                    "user_count_mismatch",
                    {"expected": 1, "actual": user_count},
                )
            user = user_qs.get()

            try:
                profile_obj = UserProfile.objects.get(user=user)
            except UserProfile.DoesNotExist:
                self._fail("user_profile_missing", {"username": "demo"})
            if profile_obj.role != UserProfile.Roles.ADMIN:
                self._fail(
                    "user_role_mismatch",
                    {"expected": UserProfile.Roles.ADMIN, "actual": profile_obj.role},
                )

            doc_type = DocumentType.objects.filter(name="Demo Type").first()
            if doc_type is None:
                self._fail("document_type_missing", {"name": "Demo Type"})

            org = Organization.objects.filter(slug="demo").first()
            if org is None:
                self._fail("organization_missing", {"slug": "demo"})

            counts, invalid_documents = self._validate_projects_and_documents(
                org,
                expected_counts,
                profile,
            )

        payload = {
            "event": "check.ok",
            "profile": profile,
            "counts": counts,
        }
        if profile == "chaos":
            payload["invalid_documents"] = invalid_documents
        self.stdout.write(json.dumps(payload, ensure_ascii=False))

    def _fail(self, reason: str, details: Dict[str, object]):
        payload = {
            "event": "check.failed",
            "reason": reason,
        }
        if details is not None:
            payload["details"] = details
        self.stdout.write(json.dumps(payload, ensure_ascii=False))
        raise CommandError(reason)

    def _expected_document_count(self, projects: Iterable) -> int:
        total = 0
        for project in projects:
            for doc in project.documents:
                if getattr(doc, "missing_type", False):
                    continue
                total += 1
        return total

    def _validate_projects_and_documents(
        self,
        organization: Organization,
        expected_counts: Dict[str, int],
        profile: str,
    ) -> Tuple[Dict[str, int], int]:
        with set_current_organization(organization):
            projects = list(Project.objects.filter(organization=organization))

        project_lookup: Dict[str, Project] = {}
        duplicate_project_slugs = []
        for project in projects:
            slug = DemoDatasetApplier._extract_project_slug(project.description)
            if not slug:
                continue
            if slug in project_lookup:
                duplicate_project_slugs.append(slug)
            else:
                project_lookup[slug] = project
        if duplicate_project_slugs:
            self._fail(
                "duplicate_project_slugs",
                {"slugs": sorted(set(duplicate_project_slugs))},
            )

        document_slugs = set()
        duplicate_document_slugs = []
        document_count = 0
        invalid_documents = 0

        with set_current_organization(organization):
            project_documents: Dict[str, Iterable[Document]] = {}
            for project_slug, project in project_lookup.items():
                project_documents[project_slug] = list(
                    Document.objects.filter(project=project)
                )

        for project_slug, documents in project_documents.items():
            for document in documents:
                doc_slug = DemoDatasetApplier._extract_document_slug(document.file.name)
                if not doc_slug:
                    continue
                if not doc_slug.startswith("doc-"):
                    continue
                if not doc_slug.startswith(f"doc-{project_slug}"):
                    continue
                if doc_slug in document_slugs:
                    duplicate_document_slugs.append(doc_slug)
                    continue
                document_slugs.add(doc_slug)
                document_count += 1
                if (
                    document.status == Document.STATUS_PROCESSING
                    or self._file_size(document) == 0
                ):
                    invalid_documents += 1

        if duplicate_document_slugs:
            self._fail(
                "duplicate_document_slugs",
                {"slugs": sorted(set(duplicate_document_slugs))},
            )

        actual_counts = {
            "projects": len(project_lookup),
            "documents": document_count,
            "users": 1,
            "orgs": 1,
        }

        if (
            actual_counts["projects"] != expected_counts["projects"]
            or actual_counts["documents"] != expected_counts["documents"]
        ):
            self._fail(
                "count_mismatch",
                {
                    "expected": expected_counts,
                    "actual": {
                        key: actual_counts[key]
                        for key in ["projects", "documents", "users", "orgs"]
                    },
                },
            )

        if profile == "chaos" and invalid_documents == 0:
            self._fail("invalid_documents_missing", {"expected_minimum": 1})

        return actual_counts, invalid_documents

    def _file_size(self, document: Document) -> int:
        try:
            return document.file.size
        except Exception:
            return 0
