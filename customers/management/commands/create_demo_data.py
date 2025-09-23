import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand, CommandError
from django_tenants.utils import get_public_schema_name, schema_context
from faker import Faker

from customers.models import Domain, Tenant
from documents.models import Document, DocumentType
from organizations.models import Organization, OrgMembership
from organizations.utils import set_current_organization
from profiles.models import UserProfile
from projects.models import Project
from users.models import User


@dataclass
class DocumentSpec:
    slug: str
    title: str
    filename: str
    content: Optional[bytes]
    media_type: str = "text/plain"
    broken: bool = False
    missing_type: bool = False
    legacy_titles: List[str] = field(default_factory=list)
    legacy_filenames: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ProjectSpec:
    slug: str
    name: str
    description: str
    documents: List[DocumentSpec] = field(default_factory=list)
    legacy_names: List[str] = field(default_factory=list)


class SeederDataset:
    def __init__(self, projects: Iterable[ProjectSpec]):
        self.projects = list(projects)

    @property
    def project_count(self) -> int:
        return len(self.projects)

    @property
    def document_count(self) -> int:
        return sum(len(project.documents) for project in self.projects)


class DemoDatasetBuilder:
    def __init__(
        self,
        profile: str,
        faker: Faker,
        rng: random.Random,
        projects_override: Optional[int] = None,
        docs_override: Optional[int] = None,
    ) -> None:
        self.profile = profile
        self.faker = faker
        self.rng = rng
        self.projects_override = projects_override
        self.docs_override = docs_override

    def build(self) -> SeederDataset:
        if self.profile == "baseline":
            return self._build_baseline()
        if self.profile == "demo":
            return self._build_demo()
        if self.profile == "heavy":
            return self._build_heavy()
        if self.profile == "chaos":
            return self._build_chaos()
        raise CommandError(f"Unknown profile '{self.profile}'")

    def _project_slug(self, index: int) -> str:
        return f"proj-{index:02d}"

    def _document_slug(self, project_slug: str, doc_index: int) -> str:
        return f"doc-{project_slug}-{doc_index:02d}"

    def _build_baseline(self) -> SeederDataset:
        project1 = ProjectSpec(
            slug=self._project_slug(1),
            name="Demo Project 1",
            description="Erstes Demo-Projekt",
            documents=[
                DocumentSpec(
                    slug=self._document_slug("proj-01", 1),
                    title="Demo Document 1",
                    filename="doc-proj-01-01.txt",
                    content=b"Demo content 1",
                    legacy_titles=["Demo Document 1"],
                    legacy_filenames=["demo1.txt"],
                )
            ],
        )
        project2 = ProjectSpec(
            slug=self._project_slug(2),
            name="Demo Project 2",
            description="Zweites Demo-Projekt",
            documents=[
                DocumentSpec(
                    slug=self._document_slug("proj-02", 1),
                    title="Demo Document 2",
                    filename="doc-proj-02-01.txt",
                    content=b"Demo content 2",
                    legacy_titles=["Demo Document 2"],
                    legacy_filenames=["demo2.txt"],
                )
            ],
        )
        return SeederDataset([project1, project2])

    def _build_demo(self) -> SeederDataset:
        dataset = self._build_baseline()
        project_target = self.projects_override or self.rng.randint(5, 8)
        docs_target = self.docs_override or self.rng.randint(3, 5)
        projects = list(dataset.projects)

        while len(projects) < project_target:
            index = len(projects) + 1
            projects.append(
                self._generate_project(index, docs_target, formats=("txt", "md"))
            )

        for project in projects:
            desired_docs = docs_target
            if len(project.documents) >= desired_docs:
                continue
            additional = desired_docs - len(project.documents)
            project.documents.extend(
                self._generate_documents(
                    project_slug=project.slug,
                    project_name=project.name,
                    start=len(project.documents) + 1,
                    count=additional,
                    formats=("txt", "md"),
                    allow_minor_faults=True,
                )
            )

        return SeederDataset(projects[:project_target])

    def _build_heavy(self) -> SeederDataset:
        project_target = self.projects_override or 30
        docs_target = self.docs_override or 10
        projects: List[ProjectSpec] = []
        for index in range(1, project_target + 1):
            project = self._generate_project(
                index,
                docs_target,
                formats=("txt", "md", "json"),
            )
            projects.append(project)
        return SeederDataset(projects)

    def _build_chaos(self) -> SeederDataset:
        base_dataset = self._build_demo()
        projects = list(base_dataset.projects)
        total_docs = sum(len(p.documents) for p in projects)
        broken_target = max(1, math.ceil(total_docs * 0.12))
        all_doc_refs: List[DocumentSpec] = [
            doc for project in projects for doc in project.documents
        ]
        self.rng.shuffle(all_doc_refs)
        broken_docs = all_doc_refs[:broken_target]
        toggles = [
            "empty",
            "invalid_bytes",
            "long_title",
            "missing_type",
        ]
        for index, doc in enumerate(broken_docs):
            mode = toggles[index % len(toggles)]
            if mode == "empty":
                doc.content = None
                doc.broken = True
                doc.metadata.update({"invalid": True, "reason": "empty_content"})
            elif mode == "invalid_bytes":
                doc.content = b"\x80\x81\xfe"
                doc.broken = True
                doc.metadata.update({"invalid": True, "reason": "invalid_bytes"})
            elif mode == "long_title":
                doc.title = f"{doc.title} {self.faker.pystr(min_chars=30, max_chars=30)}"
                doc.broken = True
                doc.metadata.update({"invalid": True, "reason": "title_length"})
            elif mode == "missing_type":
                doc.missing_type = True
                doc.broken = True
                doc.metadata.update({"invalid": True, "reason": "missing_type"})
        return SeederDataset(projects)

    def _generate_project(
        self,
        index: int,
        docs_target: int,
        formats: Iterable[str],
    ) -> ProjectSpec:
        slug = self._project_slug(index)
        name = f"Demo Project {index}"
        description = f"{self.faker.catch_phrase()} ({name})"
        documents = list(
            self._generate_documents(
                project_slug=slug,
                project_name=name,
                start=1,
                count=docs_target,
                formats=formats,
                allow_minor_faults=False,
            )
        )
        return ProjectSpec(slug=slug, name=name, description=description, documents=documents)

    def _generate_documents(
        self,
        project_slug: str,
        project_name: str,
        start: int,
        count: int,
        formats: Iterable[str],
        allow_minor_faults: bool,
    ) -> Iterable[DocumentSpec]:
        documents: List[DocumentSpec] = []
        formats_tuple = tuple(formats)
        for offset in range(count):
            doc_index = start + offset
            fmt = formats_tuple[offset % len(formats_tuple)]
            doc_slug = self._document_slug(project_slug, doc_index)
            filename = f"{doc_slug}.{fmt}"
            title = self._build_title(project_name, doc_index)
            content_bytes = self._build_content(fmt, title, doc_slug)
            metadata: Dict[str, object] = {}
            broken = False
            if allow_minor_faults and offset == 0 and fmt == "md":
                content_bytes = f"## {project_name} Liste [Unvollständig".encode("utf-8")
                broken = True
                metadata = {"invalid": True, "reason": "markdown_unclosed"}
            documents.append(
                DocumentSpec(
                    slug=doc_slug,
                    title=title,
                    filename=filename,
                    content=content_bytes,
                    media_type=self._media_type(fmt),
                    broken=broken,
                    metadata=metadata,
                )
            )
        return documents

    def _build_title(self, project_name: str, doc_index: int) -> str:
        base_sentence = self.faker.sentence(nb_words=6).strip()
        base_sentence = base_sentence.rstrip(".")
        title = f"{base_sentence} ({project_name} #{doc_index})"
        return title[:80]

    def _build_content(self, fmt: str, title: str, doc_slug: str) -> bytes:
        if fmt == "txt":
            text = f"{title}\n\n{self.faker.paragraph(nb_sentences=3)}"
            return text.encode("utf-8")
        if fmt == "md":
            md = f"# {title}\n\n- {self.faker.bs()}\n- {self.faker.catch_phrase()}"
            return md.encode("utf-8")
        if fmt == "json":
            payload = {
                "slug": doc_slug,
                "title": title,
                "summary": self.faker.sentence(nb_words=8),
                "owner": self.faker.name(),
            }
            return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        return title.encode("utf-8")

    def _media_type(self, fmt: str) -> str:
        if fmt == "md":
            return "text/markdown"
        if fmt == "json":
            return "application/json"
        return "text/plain"


class DemoDatasetApplier:
    PROJECT_DESC_PATTERN = re.compile(r"^\[demo-seed:(?P<slug>[^\]]+)\]\s*(?P<body>.*)$")

    def __init__(self, user: User, organization: Organization, doc_type: DocumentType):
        self.user = user
        self.organization = organization
        self.doc_type = doc_type

    def seed(self, dataset: SeederDataset) -> dict:
        project_lookup = self._ensure_projects(dataset.projects)
        document_count = self._ensure_documents(project_lookup, dataset.projects)
        return {
            "projects": len(project_lookup),
            "documents": document_count,
        }

    def wipe(self, dataset: SeederDataset) -> dict:
        deleted_documents = 0
        deleted_projects = 0
        with set_current_organization(self.organization):
            projects = list(Project.objects.filter(organization=self.organization))
            slug_index: Dict[str, Project] = {}
            name_index: Dict[str, Project] = {}
            for project in projects:
                slug = self._extract_project_slug(project.description)
                if slug:
                    slug_index.setdefault(slug, project)
                name_index.setdefault(project.name, project)
            for spec in dataset.projects:
                project = slug_index.get(spec.slug)
                if not project:
                    for legacy_name in [spec.name, *spec.legacy_names]:
                        project = name_index.get(legacy_name)
                        if project:
                            break
                if not project:
                    continue
                deleted_documents += self._wipe_documents(project, spec)
                if not Document.objects.filter(project=project).exists():
                    project.delete()
                    deleted_projects += 1
        return {"projects": deleted_projects, "documents": deleted_documents}

    def _ensure_projects(self, project_specs: Iterable[ProjectSpec]) -> Dict[str, Project]:
        project_lookup: Dict[str, Project] = {}
        with set_current_organization(self.organization):
            projects = list(Project.objects.filter(organization=self.organization))
            slug_index: Dict[str, Project] = {}
            name_index: Dict[str, Project] = {}
            for project in projects:
                slug = self._extract_project_slug(project.description)
                if slug:
                    slug_index[slug] = project
                name_index.setdefault(project.name, project)

            to_create: List[Project] = []
            pending_slug_by_name: Dict[str, str] = {}
            for spec in project_specs:
                project = slug_index.get(spec.slug)
                if not project:
                    for legacy_name in [spec.name, *spec.legacy_names]:
                        project = name_index.get(legacy_name)
                        if project:
                            break
                if project:
                    update_fields: List[str] = []
                    desired_description = self._format_project_description(spec)
                    if project.description != desired_description:
                        project.description = desired_description
                        update_fields.append("description")
                    if project.name != spec.name:
                        project.name = spec.name
                        update_fields.append("name")
                    if project.owner_id != self.user.id:
                        project.owner = self.user
                        update_fields.append("owner")
                    if update_fields:
                        project.save(update_fields=update_fields)
                    slug_index[spec.slug] = project
                    project_lookup[spec.slug] = project
                    continue

                project = Project(
                    name=spec.name,
                    description=self._format_project_description(spec),
                    owner=self.user,
                    organization=self.organization,
                )
                to_create.append(project)
                pending_slug_by_name[spec.name] = spec.slug

            if to_create:
                Project.objects.bulk_create(to_create, batch_size=25)
                created_projects = list(
                    Project.objects.filter(
                        organization=self.organization,
                        name__in=[project.name for project in to_create],
                    )
                )
                for project in created_projects:
                    slug = self._extract_project_slug(project.description)
                    if not slug:
                        slug = pending_slug_by_name.get(project.name)
                    if slug:
                        slug_index[slug] = project
                        project_lookup[slug] = project

        return project_lookup

    def _ensure_documents(
        self,
        project_lookup: Dict[str, Project],
        project_specs: Iterable[ProjectSpec],
    ) -> int:
        total_documents = 0
        with set_current_organization(self.organization):
            for spec in project_specs:
                project = project_lookup.get(spec.slug)
                if not project:
                    continue
                existing_docs = list(Document.objects.filter(project=project))
                by_slug: Dict[str, Document] = {}
                by_title: Dict[str, Document] = {}
                by_filename: Dict[str, Document] = {}
                for document in existing_docs:
                    slug = self._extract_document_slug(document.file.name)
                    if slug:
                        by_slug.setdefault(slug, document)
                    base_name = os.path.basename(document.file.name or "")
                    if base_name:
                        by_filename.setdefault(base_name, document)
                    by_title.setdefault(document.title, document)

                new_docs: List[Document] = []
                for doc_spec in spec.documents:
                    if doc_spec.missing_type:
                        continue
                    total_documents += 1
                    document = by_slug.get(doc_spec.slug)
                    if not document:
                        for legacy_title in doc_spec.legacy_titles:
                            document = by_title.get(legacy_title)
                            if document:
                                break
                    if not document:
                        for legacy_filename in [doc_spec.filename, *doc_spec.legacy_filenames]:
                            document = by_filename.get(legacy_filename)
                            if document:
                                break
                    if document:
                        self._update_document(document, doc_spec)
                        continue
                    new_docs.append(self._build_document(project, doc_spec))
                if new_docs:
                    Document.objects.bulk_create(new_docs, batch_size=25)
        return total_documents

    def _wipe_documents(self, project: Project, spec: ProjectSpec) -> int:
        doc_slugs = {doc.slug for doc in spec.documents}
        doc_filenames = {doc.filename for doc in spec.documents}
        doc_titles = {doc.title for doc in spec.documents}
        for doc in spec.documents:
            doc_titles.update(doc.legacy_titles)
            doc_filenames.update(doc.legacy_filenames)

        deleted = 0
        for document in Document.objects.filter(project=project):
            base_name = os.path.basename(document.file.name or "")
            slug = self._extract_document_slug(document.file.name)
            if (slug and slug in doc_slugs) or base_name in doc_filenames or document.title in doc_titles:
                document.delete()
                deleted += 1
        return deleted

    def _build_document(self, project: Project, doc_spec: DocumentSpec) -> Document:
        title = self._normalize_title(doc_spec.title)
        document = Document(
            title=title,
            project=project,
            owner=self.user,
            type=self.doc_type,
            status=self._target_status(doc_spec),
        )
        file_content = doc_spec.content if doc_spec.content is not None else b""
        try:
            document.file.save(
                doc_spec.filename,
                ContentFile(file_content),
                save=False,
            )
        except Exception:
            document.status = Document.STATUS_PROCESSING
            document.file.save(
                doc_spec.filename,
                ContentFile(file_content or b""),
                save=False,
            )
        return document

    def _update_document(self, document: Document, doc_spec: DocumentSpec) -> None:
        update_fields: List[str] = []
        title = self._normalize_title(doc_spec.title)
        if document.title != title:
            document.title = title
            update_fields.append("title")
        if document.owner_id != self.user.id:
            document.owner = self.user
            update_fields.append("owner")
        if document.type_id != self.doc_type.id:
            document.type = self.doc_type
            update_fields.append("type")

        target_status = self._target_status(doc_spec)
        if document.status != target_status:
            document.status = target_status
            update_fields.append("status")

        file_updated = False
        should_update_file = doc_spec.content is not None or doc_spec.broken
        if should_update_file:
            file_content = doc_spec.content if doc_spec.content is not None else b""
            try:
                if document.file and os.path.basename(document.file.name) != doc_spec.filename:
                    document.file.delete(save=False)
                document.file.save(
                    doc_spec.filename,
                    ContentFile(file_content),
                    save=False,
                )
                file_updated = True
            except Exception:
                document.status = Document.STATUS_PROCESSING
                if "status" not in update_fields:
                    update_fields.append("status")

        if file_updated:
            update_fields.append("file")

        if update_fields:
            document.save(update_fields=update_fields)

    def _format_project_description(self, spec: ProjectSpec) -> str:
        description = spec.description.strip()
        return f"[demo-seed:{spec.slug}] {description}"

    def _extract_project_slug(self, description: Optional[str]) -> Optional[str]:
        if not description:
            return None
        match = self.PROJECT_DESC_PATTERN.match(description)
        if match:
            return match.group("slug")
        return None

    def _extract_document_slug(self, file_name: Optional[str]) -> Optional[str]:
        if not file_name:
            return None
        base_name = os.path.basename(file_name)
        if base_name.startswith("doc-") and "." in base_name:
            return base_name.rsplit(".", 1)[0]
        return None

    def _normalize_title(self, title: str) -> str:
        max_length = Document._meta.get_field("title").max_length
        if len(title) > max_length:
            return title[:max_length]
        return title

    def _target_status(self, doc_spec: DocumentSpec) -> str:
        if doc_spec.metadata.get("invalid"):
            return Document.STATUS_PROCESSING
        return Document.STATUS_UPLOADED


class Command(BaseCommand):
    """Create demo tenant, user, and sample projects/documents."""

    help = "Ensure demo tenant, user and sample data exist"

    def add_arguments(self, parser):
        parser.add_argument(
            "--domain",
            help="Hostname to bind to the demo tenant (e.g. noesis-2-staging-...run.app)",
        )
        parser.add_argument(
            "--profile",
            choices=["baseline", "demo", "heavy", "chaos"],
            default="demo",
            help="Dataset profile to seed (default: demo)",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=1337,
            help="Deterministic seed for faker/random (default: 1337)",
        )
        parser.add_argument(
            "--wipe",
            action="store_true",
            help="Remove demo seed data for the selected profile instead of seeding",
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
        desired_domain = (
            options.get("domain") or os.getenv("STAGING_HOST") or "demo.localhost"
        )
        profile = options.get("profile", "demo")
        seed = options.get("seed", 1337)
        wipe_only = bool(options.get("wipe"))
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

        random.seed(seed)
        rng = random.Random(seed)
        faker = Faker("de_DE")
        faker.seed_instance(seed)

        # Ensure tenant/domain from the public schema
        with schema_context(get_public_schema_name()):
            tenant, _ = Tenant.objects.get_or_create(
                schema_name="demo", defaults={"name": "Demo Tenant"}
            )
            # Domain is globally unique (DomainMixin). Use domain-only lookup and
            # then ensure tenant/is_primary to avoid duplicate-key races across runs.
            domain_obj, created = Domain.objects.get_or_create(
                domain=desired_domain, defaults={"tenant": tenant, "is_primary": True}
            )
            if not created:
                changed = False
                if domain_obj.tenant_id != tenant.id:
                    domain_obj.tenant = tenant
                    changed = True
                if not domain_obj.is_primary:
                    domain_obj.is_primary = True
                    changed = True
                if changed:
                    domain_obj.save(update_fields=["tenant", "is_primary"])
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
                builder = DemoDatasetBuilder(
                    profile=profile,
                    faker=faker,
                    rng=rng,
                    projects_override=projects_override,
                    docs_override=docs_override,
                )
                dataset = builder.build()
                applier = DemoDatasetApplier(user=user, organization=org, doc_type=doc_type)

                if wipe_only:
                    counts = applier.wipe(dataset)
                    event = "seed.wipe"
                else:
                    counts = applier.seed(dataset)
                    event = "seed.done"

        summary = {
            "event": event,
            "profile": profile,
            "seed": seed,
            "counts": {
                "projects": counts.get("projects", 0),
                "documents": counts.get("documents", 0),
                "users": 1,
                "orgs": 1,
            },
        }
        self.stdout.write(json.dumps(summary, ensure_ascii=False))
