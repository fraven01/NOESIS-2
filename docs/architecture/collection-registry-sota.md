# SOTA: Administrierbare Collection Registry

**Status:** Future Enhancement (Post-MVP)
**Version:** 1.0
**Datum:** 2025-12-08
**Autor:** Architecture Team

---

## Executive Summary

Dieses Dokument beschreibt eine **State-of-the-Art (SOTA) Implementierung** einer administrierbaren Collection Registry für NOESIS 2. Die Registry ersetzt das aktuelle ad-hoc Collection-Management durch einen zentralen, versionierten und UI-verwaltbaren Service.

### Ziele

1. **Zentrale Definition**: Alle Collections (System & User) an einem Ort
2. **Auto-Discovery**: UI und APIs können Collections automatisch entdecken
3. **Bootstrap-Automation**: System-Collections werden automatisch bei Tenant-Setup erstellt
4. **Versioning & Migration**: Schema-Changes und Collection-Migrations trackbar
5. **Admin-UI**: Collection-Management ohne SQL
6. **Plugin-Support**: Neue Features können Collections registrieren

---

## Probleme mit aktuellem Design

### 1. **Verteilte Collection-Definitionen**
```python
# documents/collection_service.py
DEFAULT_MANUAL_COLLECTION_SLUG = "manual-search"

# ai_core/rag/collections.py
MANUAL_COLLECTION_SLUG = DEFAULT_MANUAL_COLLECTION_SLUG

# theme/views.py
collection_key = "manual-search"  # Hardcoded
```
**Problem:** Keine zentrale Source of Truth, schwer zu ändern

### 2. **Fehlende Bootstrap-Logik**
- System-Collections werden lazy erstellt (Race Conditions!)
- Kein automatischer Setup bei Tenant-Erstellung
- Manuelle Intervention nötig bei Problemen

### 3. **Keine UI für Collection-Management**
- Admin muss SQL verwenden
- Keine Visibility über existierende Collections
- Schwierig "orphaned" Dokumente zu finden

### 4. **Kein Plugin-System**
- Neue Features müssen Collection-Code manuell anpassen
- Keine Entkopplung zwischen Features und Collections

---

## SOTA Architektur

### Komponenten

```
┌─────────────────────────────────────────────────────────────┐
│                    Collection Registry                       │
│  (Zentrale Definition, Versioning, Bootstrap, Discovery)    │
└─────────────────────────────────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────┐
│  Bootstrap       │ │  Admin UI    │ │  Migration   │
│  Service         │ │  (CRUD)      │ │  System      │
└──────────────────┘ └──────────────┘ └──────────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
               ┌─────────────────────────┐
               │  DocumentCollection     │
               │  (Django Model)         │
               └─────────────────────────┘
```

### Kern-Konzepte

#### 1. Collection Definition (Deklarativ)

```python
# documents/registry/definitions.py

from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from uuid import UUID


@dataclass(frozen=True)
class CollectionDefinition:
    """Deklarative Collection-Definition."""

    # Identity
    slug: str  # Unique key (z.B. "manual-search")
    name: str  # Human-readable name
    description: str  # Beschreibung für UI

    # Type & Behavior
    type: str  # "system" | "user" | "plugin"
    auto_create: bool = True  # Bootstrap bei Tenant-Setup?
    allow_user_delete: bool = False  # Darf User löschen?

    # Metadata
    icon: Optional[str] = None  # Icon für UI (z.B. "search")
    color: Optional[str] = None  # Farbe für UI (z.B. "#3B82F6")
    metadata: Dict[str, Any] = None  # Zusätzliche Metadaten

    # Advanced
    uuid_generator: Optional[Callable[[UUID], UUID]] = None  # Custom UUID-Generierung
    scope: Optional[str] = None  # Scope (z.B. "manual", "crawler")
    embedding_profile: Optional[str] = None  # Default Embedding-Profil

    # Versioning
    schema_version: int = 1  # Schema-Version für Migrations


# Zentrale Registry
SYSTEM_COLLECTIONS = {
    "manual-search": CollectionDefinition(
        slug="manual-search",
        name="Manual Search",
        description="Documents from web search and manual ingestion",
        type="system",
        auto_create=True,
        allow_user_delete=False,
        icon="search",
        color="#3B82F6",
        scope="manual",
        metadata={
            "features": ["web_search", "manual_upload"],
            "retention_days": None,  # Infinite retention
        },
    ),
    "web-crawler": CollectionDefinition(
        slug="web-crawler",
        name="Web Crawler",
        description="Documents fetched by automated web crawler",
        type="system",
        auto_create=True,
        allow_user_delete=False,
        icon="globe",
        color="#10B981",
        scope="crawler",
        metadata={
            "features": ["automated_crawl", "link_extraction"],
            "retention_days": 90,
        },
    ),
    "email-import": CollectionDefinition(
        slug="email-import",
        name="Email Import",
        description="Documents imported from email attachments",
        type="system",
        auto_create=False,  # Only create when feature is enabled
        allow_user_delete=False,
        icon="mail",
        color="#F59E0B",
        scope="email",
        metadata={
            "features": ["email_parsing", "attachment_extraction"],
        },
    ),
}
```

#### 2. Collection Registry Service

```python
# documents/registry/service.py

from typing import Dict, List, Optional
from uuid import UUID, uuid5, NAMESPACE_URL
from django.db import transaction
from django_tenants.utils import schema_context

from customers.models import Tenant
from documents.models import DocumentCollection
from documents.registry.definitions import (
    CollectionDefinition,
    SYSTEM_COLLECTIONS,
)
from common.logging import get_logger

logger = get_logger(__name__)


class CollectionRegistry:
    """Central registry for collection definitions and management."""

    def __init__(self):
        # Start with system collections
        self._definitions: Dict[str, CollectionDefinition] = dict(SYSTEM_COLLECTIONS)

        # Plugin collections can be registered at runtime
        self._plugin_definitions: Dict[str, CollectionDefinition] = {}

    def register(self, definition: CollectionDefinition) -> None:
        """Register a new collection definition (e.g., from a plugin)."""
        if definition.slug in self._definitions:
            raise ValueError(
                f"Collection '{definition.slug}' already registered"
            )

        if definition.type == "plugin":
            self._plugin_definitions[definition.slug] = definition
        else:
            self._definitions[definition.slug] = definition

        logger.info(
            "collection_registry.registered",
            extra={
                "slug": definition.slug,
                "type": definition.type,
                "auto_create": definition.auto_create,
            },
        )

    def unregister(self, slug: str) -> None:
        """Unregister a collection definition (plugins only)."""
        if slug in self._plugin_definitions:
            del self._plugin_definitions[slug]
            logger.info(
                "collection_registry.unregistered", extra={"slug": slug}
            )
        else:
            raise ValueError(
                f"Cannot unregister system collection '{slug}'"
            )

    def get_definition(self, slug: str) -> Optional[CollectionDefinition]:
        """Get collection definition by slug."""
        return (
            self._definitions.get(slug)
            or self._plugin_definitions.get(slug)
        )

    def list_definitions(
        self,
        *,
        type_filter: Optional[str] = None,
        auto_create_only: bool = False,
    ) -> List[CollectionDefinition]:
        """List all collection definitions."""
        all_definitions = {**self._definitions, **self._plugin_definitions}

        definitions = list(all_definitions.values())

        if type_filter:
            definitions = [d for d in definitions if d.type == type_filter]

        if auto_create_only:
            definitions = [d for d in definitions if d.auto_create]

        return definitions

    @transaction.atomic
    def bootstrap_tenant(
        self, tenant: Tenant, *, dry_run: bool = False
    ) -> Dict[str, str]:
        """Bootstrap all auto-create collections for a tenant.

        Returns: Dict mapping slug -> collection_id (UUID string)
        """
        result = {}

        with schema_context(tenant.schema_name):
            auto_create = self.list_definitions(auto_create_only=True)

            logger.info(
                "collection_registry.bootstrap_start",
                extra={
                    "tenant_schema": tenant.schema_name,
                    "collections_count": len(auto_create),
                },
            )

            for definition in auto_create:
                if dry_run:
                    logger.info(
                        "collection_registry.bootstrap_dry_run",
                        extra={
                            "tenant_schema": tenant.schema_name,
                            "slug": definition.slug,
                        },
                    )
                    continue

                try:
                    collection_id = self._ensure_collection(
                        tenant, definition
                    )
                    result[definition.slug] = str(collection_id)

                    logger.info(
                        "collection_registry.bootstrap_success",
                        extra={
                            "tenant_schema": tenant.schema_name,
                            "slug": definition.slug,
                            "collection_id": str(collection_id),
                        },
                    )
                except Exception as exc:
                    logger.exception(
                        "collection_registry.bootstrap_failed",
                        extra={
                            "tenant_schema": tenant.schema_name,
                            "slug": definition.slug,
                            "error": str(exc),
                        },
                    )

        return result

    def _ensure_collection(
        self, tenant: Tenant, definition: CollectionDefinition
    ) -> UUID:
        """Ensure collection exists for tenant based on definition."""
        # Generate deterministic UUID
        if definition.uuid_generator:
            collection_uuid = definition.uuid_generator(tenant.id)
        else:
            collection_uuid = uuid5(
                NAMESPACE_URL,
                f"collection:{tenant.id}:{definition.slug}",
            )

        # Get or create
        collection, created = DocumentCollection.objects.get_or_create(
            tenant=tenant,
            key=definition.slug,
            defaults={
                "collection_id": collection_uuid,
                "name": definition.name,
                "type": definition.type,
                "visibility": "tenant",  # Default visibility
                "metadata": {
                    "description": definition.description,
                    "icon": definition.icon,
                    "color": definition.color,
                    "schema_version": definition.schema_version,
                    "auto_created": True,
                    **(definition.metadata or {}),
                },
            },
        )

        return collection.collection_id


# Global registry instance
_REGISTRY: Optional[CollectionRegistry] = None


def get_registry() -> CollectionRegistry:
    """Get the global collection registry instance."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = CollectionRegistry()
    return _REGISTRY


# Convenience functions
def register_collection(definition: CollectionDefinition) -> None:
    """Register a collection definition."""
    get_registry().register(definition)


def bootstrap_tenant_collections(
    tenant: Tenant, *, dry_run: bool = False
) -> Dict[str, str]:
    """Bootstrap collections for a tenant."""
    return get_registry().bootstrap_tenant(tenant, dry_run=dry_run)
```

#### 3. Admin UI (Django Admin Integration)

```python
# documents/admin/collection_admin.py

from django.contrib import admin
from django.utils.html import format_html
from documents.models import DocumentCollection
from documents.registry.service import get_registry


@admin.register(DocumentCollection)
class DocumentCollectionAdmin(admin.ModelAdmin):
    """Admin interface for DocumentCollection with Registry integration."""

    list_display = [
        "name",
        "key_colored",
        "type_badge",
        "tenant",
        "document_count",
        "created_at",
        "actions_links",
    ]
    list_filter = ["type", "visibility", "tenant", "created_at"]
    search_fields = ["name", "key", "collection_id"]
    readonly_fields = [
        "collection_id",
        "created_at",
        "updated_at",
        "definition_info",
    ]

    fieldsets = (
        ("Identity", {
            "fields": ("name", "key", "collection_id", "tenant"),
        }),
        ("Classification", {
            "fields": ("type", "visibility"),
        }),
        ("Metadata", {
            "fields": ("metadata", "definition_info"),
        }),
        ("Timestamps", {
            "fields": ("created_at", "updated_at"),
            "classes": ("collapse",),
        }),
    )

    def key_colored(self, obj):
        """Display key with color from metadata."""
        color = obj.metadata.get("color", "#6B7280")
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color,
            obj.key,
        )
    key_colored.short_description = "Key"

    def type_badge(self, obj):
        """Display type as badge."""
        colors = {
            "system": "#10B981",
            "user": "#3B82F6",
            "plugin": "#F59E0B",
        }
        color = colors.get(obj.type, "#6B7280")
        return format_html(
            '<span style="background: {}; color: white; padding: 2px 8px; '
            'border-radius: 4px; font-size: 11px;">{}</span>',
            color,
            obj.type or "unknown",
        )
    type_badge.short_description = "Type"

    def document_count(self, obj):
        """Display document count with link."""
        from documents.models import DocumentCollectionMembership

        count = DocumentCollectionMembership.objects.filter(
            collection=obj
        ).count()

        if count > 0:
            # Link to filtered document list
            url = f"/admin/documents/document/?collection_id={obj.id}"
            return format_html('<a href="{}">{} documents</a>', url, count)
        return "0 documents"
    document_count.short_description = "Documents"

    def definition_info(self, obj):
        """Show registry definition info if available."""
        registry = get_registry()
        definition = registry.get_definition(obj.key)

        if definition:
            return format_html(
                """
                <div style="padding: 10px; background: #F3F4F6; border-radius: 4px;">
                    <strong>Registry Definition</strong><br>
                    <strong>Description:</strong> {}<br>
                    <strong>Auto-Create:</strong> {}<br>
                    <strong>Allow User Delete:</strong> {}<br>
                    <strong>Scope:</strong> {}<br>
                    <strong>Schema Version:</strong> {}
                </div>
                """,
                definition.description,
                "✅ Yes" if definition.auto_create else "❌ No",
                "✅ Yes" if definition.allow_user_delete else "❌ No",
                definition.scope or "none",
                definition.schema_version,
            )
        return format_html(
            '<span style="color: #EF4444;">⚠️ No registry definition found</span>'
        )
    definition_info.short_description = "Registry Info"

    def actions_links(self, obj):
        """Quick action links."""
        return format_html(
            '<a class="button" href="/admin/documents/document/?collection_id={}">View Documents</a>',
            obj.id,
        )
    actions_links.short_description = "Actions"

    def has_delete_permission(self, request, obj=None):
        """Check if user can delete based on registry definition."""
        if obj is None:
            return True

        registry = get_registry()
        definition = registry.get_definition(obj.key)

        if definition:
            # Respect allow_user_delete from definition
            return definition.allow_user_delete

        # Default: allow delete for user collections
        return obj.type != "system"
```

#### 4. Management Commands

```python
# documents/management/commands/collection_registry.py

from django.core.management.base import BaseCommand
from django_tenants.utils import schema_context

from customers.models import Tenant
from documents.registry.service import get_registry


class Command(BaseCommand):
    """Manage collection registry."""

    help = "Manage collection registry"

    def add_arguments(self, parser):
        parser.add_argument(
            "action",
            choices=["list", "bootstrap", "validate", "export"],
            help="Action to perform",
        )
        parser.add_argument(
            "--tenant",
            type=str,
            help="Tenant schema name (for bootstrap)",
        )
        parser.add_argument(
            "--type",
            type=str,
            choices=["system", "user", "plugin"],
            help="Filter by type (for list)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Dry run (for bootstrap)",
        )

    def handle(self, *args, **options):
        action = options["action"]
        registry = get_registry()

        if action == "list":
            self._list_definitions(registry, options.get("type"))
        elif action == "bootstrap":
            self._bootstrap(registry, options)
        elif action == "validate":
            self._validate(registry)
        elif action == "export":
            self._export(registry)

    def _list_definitions(self, registry, type_filter):
        """List all collection definitions."""
        definitions = registry.list_definitions(type_filter=type_filter)

        self.stdout.write(
            self.style.SUCCESS(f"Collection Definitions ({len(definitions)}):")
        )

        for definition in definitions:
            self.stdout.write(f"\n  {definition.slug}")
            self.stdout.write(f"    Name: {definition.name}")
            self.stdout.write(f"    Type: {definition.type}")
            self.stdout.write(
                f"    Auto-Create: {'Yes' if definition.auto_create else 'No'}"
            )
            self.stdout.write(f"    Description: {definition.description}")

    def _bootstrap(self, registry, options):
        """Bootstrap collections for tenant(s)."""
        tenant_filter = options.get("tenant")
        dry_run = options.get("dry_run", False)

        if tenant_filter:
            tenants = Tenant.objects.filter(schema_name=tenant_filter)
        else:
            tenants = Tenant.objects.all()

        for tenant in tenants:
            self.stdout.write(f"\nBootstrapping: {tenant.schema_name}")
            result = registry.bootstrap_tenant(tenant, dry_run=dry_run)

            for slug, collection_id in result.items():
                self.stdout.write(
                    self.style.SUCCESS(f"  ✅ {slug}: {collection_id}")
                )

    def _validate(self, registry):
        """Validate collection definitions."""
        definitions = registry.list_definitions()

        self.stdout.write("Validating collection definitions...")

        for definition in definitions:
            # Check for required fields
            if not definition.slug:
                self.stdout.write(
                    self.style.ERROR(f"  ❌ Missing slug")
                )
            if not definition.name:
                self.stdout.write(
                    self.style.ERROR(
                        f"  ❌ Missing name for '{definition.slug}'"
                    )
                )

            # Check for duplicates
            # (handled by registry.register())

        self.stdout.write(
            self.style.SUCCESS(f"\n✅ All {len(definitions)} definitions valid")
        )

    def _export(self, registry):
        """Export definitions as JSON."""
        import json
        from dataclasses import asdict

        definitions = registry.list_definitions()
        export_data = [
            {
                **asdict(d),
                # Exclude non-serializable fields
                "uuid_generator": None,
            }
            for d in definitions
        ]

        self.stdout.write(json.dumps(export_data, indent=2, default=str))
```

---

## Migration Path (Pre-MVP → SOTA)

### Phase 1: Registry Infrastructure (1-2 Sprints)

**Ziel:** Registry-Service implementieren, keine Breaking Changes

1. Create `documents/registry/` package
2. Implement `CollectionDefinition` & `CollectionRegistry`
3. Register existing system collections
4. Add `get_registry()` global accessor
5. Tests für Registry-Service

**Deliverables:**
- Registry-Code funktioniert
- Backward-compatible mit existierendem Code
- Tests grün

---

### Phase 2: Bootstrap Integration (1 Sprint)

**Ziel:** Automatisches Bootstrap bei Tenant-Setup

1. Update `bootstrap_public_tenant` command
2. Add hook to Tenant creation signal
3. Add `collection_registry bootstrap` command
4. Migrate existing tenants

**Deliverables:**
- Neue Tenants haben automatisch System-Collections
- Alte Tenants können migriert werden
- Bootstrap ist idempotent

---

### Phase 3: Admin UI (1-2 Sprints)

**Ziel:** Collection-Management im Django Admin

1. Implement `DocumentCollectionAdmin`
2. Add Registry-Info-Display
3. Add bulk actions (bootstrap, validate)
4. Add collection health checks

**Deliverables:**
- Admin kann Collections verwalten
- Registry-Info sichtbar im Admin
- Health-Checks integriert

---

### Phase 4: Plugin System (2 Sprints)

**Ziel:** External Features können Collections registrieren

1. Define Plugin API
2. Implement plugin discovery
3. Add example plugin
4. Documentation für Plugin-Entwickler

**Deliverables:**
- Plugins können Collections registrieren
- Plugin-Collections werden automatisch bootstrapped
- Plugin-Doku verfügbar

---

### Phase 5: Frontend UI (2-3 Sprints)

**Ziel:** User-facing Collection-Management

1. Design Collection-Manager UI
2. Implement Collection CRUD endpoints
3. Add collection picker components
4. Add collection statistics dashboard

**Deliverables:**
- User kann Collections erstellen/bearbeiten
- Collection-Picker in Upload/Search
- Dashboard zeigt Collection-Statistiken

---

## API Design

### REST Endpoints

```python
# Collections API
GET    /api/v1/collections/                  # List collections
POST   /api/v1/collections/                  # Create collection (user type)
GET    /api/v1/collections/{id}/             # Get collection details
PUT    /api/v1/collections/{id}/             # Update collection
DELETE /api/v1/collections/{id}/             # Delete collection (if allowed)

GET    /api/v1/collections/registry/         # List registry definitions
POST   /api/v1/collections/bootstrap/        # Bootstrap collections
GET    /api/v1/collections/{id}/stats/       # Collection statistics
GET    /api/v1/collections/{id}/documents/   # Documents in collection
```

### GraphQL Schema (optional)

```graphql
type CollectionDefinition {
  slug: String!
  name: String!
  description: String!
  type: CollectionType!
  autoCreate: Boolean!
  allowUserDelete: Boolean!
  icon: String
  color: String
  schemaVersion: Int!
}

type Collection {
  id: ID!
  collectionId: UUID!
  key: String!
  name: String!
  type: CollectionType!
  tenant: Tenant!
  documentCount: Int!
  definition: CollectionDefinition
  createdAt: DateTime!
  updatedAt: DateTime!
}

enum CollectionType {
  SYSTEM
  USER
  PLUGIN
}

type Query {
  collections(
    type: CollectionType
    tenantId: ID
  ): [Collection!]!

  collectionDefinitions(
    type: CollectionType
  ): [CollectionDefinition!]!

  collection(id: ID!): Collection
}

type Mutation {
  createCollection(
    input: CreateCollectionInput!
  ): Collection!

  updateCollection(
    id: ID!
    input: UpdateCollectionInput!
  ): Collection!

  deleteCollection(
    id: ID!
  ): Boolean!

  bootstrapCollections(
    tenantId: ID!
    dryRun: Boolean = false
  ): BootstrapResult!
}
```

---

## Security Considerations

### 1. **Access Control**

```python
# documents/registry/permissions.py

from documents.models import DocumentCollection


def can_create_collection(user, tenant) -> bool:
    """Check if user can create collections."""
    # System collections can only be created by bootstrap
    # User collections can be created by tenant admins
    return user.is_tenant_admin(tenant)


def can_delete_collection(user, collection: DocumentCollection) -> bool:
    """Check if user can delete collection."""
    from documents.registry.service import get_registry

    # Check registry definition
    definition = get_registry().get_definition(collection.key)
    if definition and not definition.allow_user_delete:
        return False

    # Check user permissions
    return user.is_tenant_admin(collection.tenant)


def can_modify_collection(user, collection: DocumentCollection) -> bool:
    """Check if user can modify collection metadata."""
    # System collections can't be modified by users
    if collection.type == "system":
        return user.is_superuser

    return user.is_tenant_admin(collection.tenant)
```

### 2. **Tenant Isolation**

- Registry-Service respektiert Tenant-Context
- Bootstrap läuft im Tenant-Schema
- API-Endpoints filtern nach Tenant

### 3. **Rate Limiting**

```python
# Prevent abuse of collection creation
@ratelimit(key='user', rate='10/h', method='POST')
def create_collection(request):
    ...
```

---

## Performance Considerations

### 1. **Caching**

```python
# Cache collection definitions (rarely change)
from django.core.cache import cache

def get_definition_cached(slug: str) -> Optional[CollectionDefinition]:
    cache_key = f"collection_def:{slug}"
    definition = cache.get(cache_key)

    if definition is None:
        definition = get_registry().get_definition(slug)
        cache.set(cache_key, definition, timeout=3600)  # 1 hour

    return definition
```

### 2. **Bulk Operations**

```python
# Bootstrap all tenants in parallel
from concurrent.futures import ThreadPoolExecutor

def bootstrap_all_tenants_parallel(max_workers=4):
    tenants = Tenant.objects.all()
    registry = get_registry()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(registry.bootstrap_tenant, tenant)
            for tenant in tenants
        ]
        results = [f.result() for f in futures]

    return results
```

### 3. **Database Indexes**

```sql
-- Optimize collection lookups
CREATE INDEX idx_collection_tenant_key ON documents_documentcollection(tenant_id, key);
CREATE INDEX idx_collection_tenant_type ON documents_documentcollection(tenant_id, type);
CREATE INDEX idx_collection_collection_id ON documents_documentcollection(collection_id);
```

---

## Monitoring & Observability

### 1. **Metrics**

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram

collection_bootstrap_total = Counter(
    'collection_bootstrap_total',
    'Total collection bootstraps',
    ['tenant', 'status']
)

collection_lookup_duration = Histogram(
    'collection_lookup_duration_seconds',
    'Time to lookup collection',
    ['method']
)
```

### 2. **Logging**

```python
# Structured logging
logger.info(
    "collection_registry.bootstrap",
    extra={
        "tenant_schema": tenant.schema_name,
        "collections_created": len(result),
        "duration_ms": duration_ms,
    }
)
```

### 3. **Health Checks**

```python
# documents/health.py

def check_collection_registry_health():
    """Health check for collection registry."""
    registry = get_registry()

    # Check if all system collections are defined
    system_defs = registry.list_definitions(type_filter="system")
    if not system_defs:
        return False, "No system collections defined"

    # Check if definitions are valid
    for definition in system_defs:
        if not definition.slug or not definition.name:
            return False, f"Invalid definition: {definition.slug}"

    return True, f"{len(system_defs)} system collections registered"
```

---

## Testing Strategy

### 1. **Unit Tests**

```python
def test_registry_register():
    registry = CollectionRegistry()
    definition = CollectionDefinition(
        slug="test",
        name="Test",
        description="Test collection",
        type="plugin",
    )
    registry.register(definition)
    assert registry.get_definition("test") == definition


def test_registry_bootstrap():
    registry = CollectionRegistry()
    tenant = create_test_tenant()
    result = registry.bootstrap_tenant(tenant)
    assert "manual-search" in result
```

### 2. **Integration Tests**

```python
def test_collection_bootstrap_creates_db_records(tenant):
    registry = get_registry()
    result = registry.bootstrap_tenant(tenant)

    # Verify DB records exist
    for slug in result.keys():
        collection = DocumentCollection.objects.get(
            tenant=tenant, key=slug
        )
        assert collection.collection_id == UUID(result[slug])
```

### 3. **E2E Tests**

```python
def test_user_can_create_collection_via_ui(browser, tenant_admin):
    # Login as admin
    browser.login(tenant_admin)

    # Navigate to collections
    browser.visit("/collections/")

    # Create new collection
    browser.fill("name", "My Collection")
    browser.fill("key", "my-collection")
    browser.click("Create")

    # Verify created
    assert browser.is_text_present("My Collection")
```

---

## Documentation Requirements

### 1. **Developer Guide**

- How to define new collections
- How to register plugin collections
- How to use Registry API
- Migration guide from old code

### 2. **Admin Guide**

- How to bootstrap collections
- How to manage collections in UI
- How to troubleshoot collection issues
- How to backup/restore collections

### 3. **User Guide**

- What are collections
- How to create/manage collections
- How to organize documents with collections
- Best practices

---

## Success Metrics

### Technical Metrics

1. **Bootstrap Success Rate**: 100% (all tenants have system collections)
2. **Duplicate Rate**: 0% (no duplicate collections created)
3. **Lookup Performance**: < 10ms (average collection lookup time)
4. **API Response Time**: < 100ms (p95 for collection CRUD)

### Business Metrics

1. **Admin Efficiency**: -80% time spent on collection management
2. **Support Tickets**: -50% collection-related support requests
3. **User Satisfaction**: +30% (collection management UX)
4. **Plugin Adoption**: 3+ plugins registered in first 6 months

---

## Future Enhancements

### 1. **Collection Templates**

Pre-defined collection types for common use cases:
- "Project Documents"
- "Legal Contracts"
- "Technical Specs"

### 2. **Collection Sharing**

Share collections across tenants (with proper permissions):
```python
def share_collection(
    collection: DocumentCollection,
    target_tenant: Tenant,
    permissions: CollectionPermissions
):
    ...
```

### 3. **Collection Workflows**

Attach workflows to collections:
- Auto-tag documents on ingest
- Auto-extract entities
- Auto-notify on new documents

### 4. **Collection Analytics**

Advanced analytics dashboard:
- Document growth over time
- Most active collections
- Storage usage per collection
- Search patterns per collection

---

## References

### Internal Docs

- [AGENTS.md](../AGENTS.md) - Tool Contracts & Architecture
- [multi-tenancy.md](multi-tenancy.md) - Multi-Tenancy Architecture
- [id-guide-for-agents.md](id-guide-for-agents.md) - ID Propagation Guide

### External Resources

- Django Tenants: https://django-tenants.readthedocs.io/
- Plugin Architecture Patterns: https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html
- Collection Design Patterns: https://www.enterpriseintegrationpatterns.com/

---

**Status:** Draft (Review Required)
**Next Review:** After MVP Release
**Owner:** Architecture Team
**Stakeholders:** Product, Engineering, DevOps
