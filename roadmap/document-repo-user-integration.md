# Document Repository: User Integration Roadmap

**Status**: Planning (Pre-MVP)
**Component**: Document Repository (`documents/`)
**Dependencies**: User Management (completed in USER-MANAGEMENT branch)
**Architecture Doc**: [`docs/architecture/user-document-integration.md`](../docs/architecture/user-document-integration.md)

## Overview

This roadmap tracks the integration of user identity, authorization, and activity tracking into the document repository system. The implementation is divided into four phases with increasing complexity.

**Pre-MVP Context**: Database resets are acceptable. No backward compatibility or data migration required. This significantly simplifies implementation.

**Current State**: User context flows through the system (`ScopeContext.user_id` -> `ToolContext` -> `AuditMeta`). Documents now persist direct user attribution and membership actors; activity tracking and permissions are being layered in per phase.

**Decision (Pre-MVP)**:
- Ownership is derived from `AuditMeta.created_by_user_id` (not `initiated_by_user_id`).
- Upload worker remains an S2S hop (`service_id="upload-worker"`) and must preserve original ownership via audit meta/explicit fields.
- `audit_meta` is carried through the ingestion graph and into `upsert()` for all sources (upload, crawler, search).
- DB resets are acceptable; no backfill required.


**Target State**: Full user attribution, document-level permissions, activity tracking, and collaboration features integrated into the document lifecycle.

---

## Phase 1: Direct User Attribution (MVP-Critical)

**Goal**: Add direct user ownership to documents for efficient queries and lifecycle management.

**Effort**: 4-6 hours (simplified by pre-MVP DB reset)
**Priority**: 🔥 Critical (MVP blocker)
**Pre-MVP Advantage**: No backfill needed, clean schema migration

### Changes

#### 1.1 Database Schema

**File**: [`documents/models.py`](../documents/models.py)

Add FK fields to `Document` model:

```python
class Document(models.Model):
    # ... existing fields ...

    # User attribution (Phase 1) - Pre-MVP: Clean fields, no legacy data
    created_by = models.ForeignKey(
        'users.User',
        on_delete=models.SET_NULL,
        null=True,  # Allow NULL for system-created docs (crawler, scheduled tasks)
        blank=True,
        related_name='created_documents',
        db_index=True,
        help_text="User who created/uploaded this document (NULL for system)"
    )
    updated_by = models.ForeignKey(
        'users.User',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='updated_documents',
        help_text="User who last modified this document"
    )

    # Timestamps (already present, no change needed)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
```

**Migration** (simplified - no data migration):
```bash
# Pre-MVP: Simple schema migration, DB will be reset anyway
python manage.py makemigrations documents --name add_user_attribution
python manage.py migrate

# No backfill needed - clean start!
```

**Pre-MVP Simplification**:
- ✅ No backfill logic required
- ✅ No data migration script
- ✅ Clean schema from start
- ✅ Database reset is acceptable

#### 1.2 Domain Service Updates

**File**: [`documents/domain_service.py`](../documents/domain_service.py)

Update `ingest_document` to set `created_by`/`updated_by` from `audit_meta` (write-once for `created_by`):

```python
def ingest_document(
    self,
    *,
    tenant: Tenant,
    source: str,
    content_hash: str,
    metadata: Mapping[str, object] | None = None,
    audit_meta: Mapping[str, object] | None = None,  # NEW: Accept audit_meta
    ...
) -> PersistedDocumentIngest:
    """Persist document with user attribution."""

    # Resolve User instance
    user = None
    created_by_user_id = (audit_meta or {}).get("created_by_user_id")
    if created_by_user_id:
        try:
            user = User.objects.get(pk=int(created_by_user_id))
        except (User.DoesNotExist, ValueError):
            logger.warning(f"User {created_by_user_id} not found for attribution")

    document, created = Document.objects.get_or_create(
        tenant=tenant,
        source=source,
        hash=content_hash,
        defaults={
            "created_by": user,  # Write-once
            "updated_by": user,
            "metadata": metadata_payload,
            ...
        },
    )
    if not created:
        document.metadata = metadata_payload
        if user:
            document.updated_by = user
        document.save(update_fields=["metadata", "updated_by", "updated_at"])
```

**Tests**:
- `documents/tests/test_domain_service.py::test_ingest_document_sets_created_by`
- `documents/tests/test_domain_service.py::test_ingest_document_without_user`

#### 1.3 Upload Worker Integration

**File**: [`documents/upload_worker.py`](../documents/upload_worker.py)

Preserve original user attribution via `audit_meta` (S2S scope has no user_id):

```python
def process(..., user_id: str | None = None, workflow_id: str | None = None, ...) -> WorkerPublishResult:
    """Upload worker with preserved user attribution."""

    # Resolve workflow_id consistently (workflow_id > case_id > uuid4)
    resolved_workflow_id = workflow_id or case_id or str(uuid4())

    # Compose meta (S2S hop + explicit audit_meta)
    meta = self._compose_meta(
        tenant_id,
        case_id,
        trace_id,
        invocation_id,
        workflow_id=resolved_workflow_id,
        user_id=user_id,
    )
```

**Tests**:
- `documents/tests/test_upload_worker.py::test_upload_sets_user_attribution`
- `documents/tests/test_upload_worker.py::test_upload_without_authenticated_user`

#### 1.4 Ingestion Graph Updates

**Files**:
- [`ai_core/tasks.py`](../ai_core/tasks.py)
- [`ai_core/adapters/db_documents_repository.py`](../ai_core/adapters/db_documents_repository.py)

Ensure `audit_meta` flows through `ToolContext.metadata` and into `upsert()`:

```python
run_context = {
    "scope": scope.model_dump(mode="json"),
    "business": business.model_dump(mode="json"),
    "metadata": {
        "audit_meta": meta.get("audit_meta", {}),
        ...
    },
}

# repository.upsert(..., audit_meta=tool_context.metadata["audit_meta"])
```

**Tests**:
- `ai_core/tests/graphs/test_universal_ingestion_graph.py::test_audit_meta_propagation`

#### 1.5 API Response Updates

**File**: [`documents/serializers.py`](../documents/serializers.py) (create if not exists)

Add `created_by` to API responses:

```python
class DocumentSerializer(serializers.ModelSerializer):
    created_by = serializers.SerializerMethodField()
    updated_by = serializers.SerializerMethodField()

    class Meta:
        model = Document
        fields = ['id', 'title', 'created_by', 'updated_by', 'created_at', 'updated_at', ...]

    def get_created_by(self, obj):
        if obj.created_by:
            return {
                'id': obj.created_by.pk,
                'username': obj.created_by.username,
                'full_name': f"{obj.created_by.first_name} {obj.created_by.last_name}".strip()
            }
        return None

    def get_updated_by(self, obj):
        if obj.updated_by:
            return {
                'id': obj.updated_by.pk,
                'username': obj.updated_by.username,
            }
        return None
```

#### 1.6 Collection Membership Attribution

**File**: [`documents/models.py`](../documents/models.py)

Replace `added_by` with nullable actor fields (pre-MVP allows both NULL):

```python
class DocumentCollectionMembership(models.Model):
    # ... existing fields ...

    added_by_user = models.ForeignKey(
        'users.User',
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    added_by_service_id = models.CharField(
        max_length=100,
        null=True,
        blank=True
    )

    added_at = models.DateTimeField(auto_now_add=True)

    # Pre-MVP: no XOR constraint (both NULL allowed)
```

**Attribution Source**:
- Use `audit_meta.created_by_user_id` when available.
- Otherwise use `audit_meta.last_hop_service_id`.

### Acceptance Criteria (Pre-MVP)

- [x] `Document` model has `created_by`, `updated_by` fields (timestamps already exist)
- [x] Migration successfully adds fields (nullable for system uploads)
- [ ] ~~Backfill script~~ (NOT NEEDED - pre-MVP clean start)
- [x] New document uploads set `created_by` from authenticated user
- [x] System uploads (crawler, scheduled) set `created_by = NULL` (no error)
- [x] `DocumentCollectionMembership` has `added_by_user` / `added_by_service_id` fields (nullable)
- [x] Django Admin shows user attribution in document list/detail
- [x] API responses include `created_by` object (id, username, full_name)
- [ ] Tests pass: `npm run test:py:single -- documents/tests/test_user_attribution.py`
- [x] Documentation updated: `documents/README.md` (already done)

### Code Locations

**Modified**:
- `documents/models.py` - Add FK fields (2 new fields)
- `documents/domain_service.py` - Accept `audit_meta` parameter (~5 lines)
- `documents/upload_worker.py` - Preserve audit_meta + workflow_id (~8 lines)
- `documents/tasks.py` - Pass `user_id` + `workflow_id` into upload worker (~6 lines)
- `ai_core/tasks.py` - Carry audit_meta into ToolContext metadata (~6 lines)
- `ai_core/adapters/db_documents_repository.py` - Use audit_meta in upsert (~10 lines)
- `documents/models.py` - Add membership actor fields (~6 lines)

**Created**:
- `documents/serializers.py` - API serializers (~30 lines)
- `documents/migrations/0018_add_user_attribution_and_membership_actor.py` - Schema migration (auto-generated)
- `documents/tests/test_user_attribution.py` - User attribution tests (~100 lines)

**Updated Tests**:
- `documents/tests/test_domain_service.py` - Add audit_meta assertions (~10 lines)
- `documents/tests/test_upload_worker.py` - Test user context propagation (~15 lines)
- `ai_core/tests/graphs/test_universal_ingestion_graph.py` - Test audit_meta in context (~10 lines)

**Total LOC**: ~170 lines (excluding migration)

---

## Phase 2: Document Activity Tracking (Compliance)

**Goal**: Log all document access for compliance, security, and analytics.

**Effort**: 1 day
**Priority**: 🔥 High (Compliance requirement)

### Changes

#### 2.1 Activity Model

**File**: [`documents/models.py`](../documents/models.py)

Create `DocumentActivity` model:

```python
class DocumentActivity(models.Model):
    """Audit trail for document access and modifications."""

    document = models.ForeignKey(
        'Document',
        on_delete=models.CASCADE,
        related_name='activities',
        db_index=True
    )
    user = models.ForeignKey(
        'users.User',
        on_delete=models.SET_NULL,
        null=True,
        related_name='document_activities'
    )

    activity_type = models.CharField(
        max_length=20,
        choices=[
            ('VIEW', 'Viewed'),
            ('DOWNLOAD', 'Downloaded'),
            ('SEARCH', 'Found in Search'),
            ('SHARE', 'Shared'),
            ('UPLOAD', 'Uploaded'),
            ('DELETE', 'Deleted'),
            ('TRANSFER', 'Ownership Transferred'),
        ],
        db_index=True
    )

    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)

    # Request context
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.CharField(max_length=500, blank=True, default="")

    # Business context
    case_id = models.CharField(max_length=255, null=True, blank=True, db_index=True)
    trace_id = models.CharField(max_length=255, null=True, blank=True, db_index=True)

    # Optional metadata (JSON)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=['document', '-timestamp']),
            models.Index(fields=['user', '-timestamp']),
            models.Index(fields=['activity_type', '-timestamp']),
            models.Index(fields=['case_id', '-timestamp']),
        ]
        verbose_name = 'Document Activity'
        verbose_name_plural = 'Document Activities'

    def __str__(self):
        return f"{self.activity_type} - {self.document.id} by {self.user} at {self.timestamp}"
```

**Migration**:
```bash
python manage.py makemigrations documents --name create_document_activity
python manage.py migrate
```

#### 2.2 Activity Tracker Service

**File**: `documents/activity_service.py` (new file)

```python
"""Document activity tracking service."""

from typing import Any
from django.http import HttpRequest
from documents.models import DocumentActivity, Document
from users.models import User


class ActivityTracker:
    """Centralized activity logging for documents."""

    @staticmethod
    def log(
        *,
        document: Document,
        activity_type: str,
        user: User | None = None,
        request: HttpRequest | None = None,
        case_id: str | None = None,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> DocumentActivity:
        """
        Log document activity.

        Args:
            document: Document being accessed
            activity_type: Type of activity (VIEW, DOWNLOAD, etc.)
            user: User performing activity (optional)
            request: HTTP request (for IP, user-agent)
            case_id: Business context
            trace_id: Request tracing
            metadata: Additional context
        """
        ip_address = None
        user_agent = ""

        if request:
            ip_address = request.META.get('REMOTE_ADDR')
            user_agent = request.META.get('HTTP_USER_AGENT', '')[:500]

        return DocumentActivity.objects.create(
            document=document,
            user=user,
            activity_type=activity_type,
            ip_address=ip_address,
            user_agent=user_agent,
            case_id=case_id,
            trace_id=trace_id,
            metadata=metadata or {}
        )
```

**Tests**:
- `documents/tests/test_activity_service.py::test_log_activity_with_user`
- `documents/tests/test_activity_service.py::test_log_activity_without_user`
- `documents/tests/test_activity_service.py::test_activity_metadata`

#### 2.3 Integrate in Download View

**File**: [`documents/views.py`](../documents/views.py)

Log DOWNLOAD activity:

```python
from documents.activity_service import ActivityTracker

def document_download(request, document_id):
    """Download document with activity logging."""

    # ... authorization check ...

    document = Document.objects.get(pk=document_id)

    # Log activity BEFORE serving file
    ActivityTracker.log(
        document=document,
        activity_type='DOWNLOAD',
        user=request.user if request.user.is_authenticated else None,
        request=request,
        case_id=str(document.case_id) if document.case_id else None,
        trace_id=request.META.get('HTTP_X_TRACE_ID')
    )

    # Serve file
    return FileResponse(...)
```

**Tests**:
- `documents/tests/test_download_view.py::test_document_download_logs_activity`

#### 2.4 Integrate in Upload Worker

**File**: [`documents/upload_worker.py`](../documents/upload_worker.py)

Log UPLOAD activity:

```python
def _run(self, state: dict, meta: dict) -> dict:
    """Upload worker with activity logging."""

    # ... register document ...

    # Log upload activity
    ActivityTracker.log(
        document=document,
        activity_type='UPLOAD',
        user=user,  # From Phase 1
        case_id=state.get('case_id'),
        trace_id=meta.get('trace_id'),
        metadata={
            'source': state.get('source'),
            'workflow_id': state.get('workflow_id')
        }
    )
```

#### 2.5 Recent Documents API

**Files**:
- `documents/views.py` (new `recent_documents` view)
- `documents/urls.py` (route wiring)

```python
@require_http_methods(["GET"])
def recent_documents(request):
    """Get user's recently accessed documents."""

    if not request.user.is_authenticated:
        return JsonResponse({"detail": "Authentication required"}, status=401)

    # Get recent activity
    recent_activities = DocumentActivity.objects.filter(
        user=request.user,
        activity_type__in=["VIEW", "DOWNLOAD"],
    ).order_by("-timestamp")[:20]

    # Get unique documents (preserve order)
    document_ids = []
    seen = set()
    for activity in recent_activities:
        if activity.document_id not in seen:
            document_ids.append(activity.document_id)
            seen.add(activity.document_id)

    documents = Document.objects.filter(id__in=document_ids[:10])

    serializer = DocumentSerializer(documents, many=True)
    return JsonResponse(serializer.data, safe=False)
```

**Tests**:
- `documents/tests/test_recent_documents.py::test_recent_documents_returns_latest_unique`
- `documents/tests/test_recent_documents.py::test_recent_documents_requires_authentication`

### Acceptance Criteria

- [x] `DocumentActivity` model created with all activity types
- [x] `ActivityTracker` service logs activities with full context
- [x] Download endpoint logs DOWNLOAD activity
- [x] Upload logs UPLOAD activity (via worker)
- [x] Recent documents API returns user's last 10 accessed docs
- [x] Activity log queryable via Django Admin
- [ ] Tests pass: `npm run test:py:single -- documents/tests/test_activity_service.py`
- [ ] API documentation updated (OpenAPI schema)

### Code Locations

**Created**:
- `documents/activity_service.py` - Activity tracker
- `documents/migrations/0019_create_document_activity.py` - Schema migration
- `documents/tests/test_activity_service.py` - Activity tracking tests
- `documents/tests/test_recent_documents.py` - Recent documents API tests

**Modified**:
- `documents/models.py` - Add DocumentActivity model
- `documents/views.py` - Log download activity
- `documents/upload_worker.py` - Log upload activity
- `documents/urls.py` - Route recent documents endpoint
- `documents/admin.py` - Activity admin

---

## Phase 3: Document-Level Permissions (Strategic)

**Goal**: Fine-grained access control at document level (beyond case-based).

**Effort**: 2-3 days
**Priority**: ⚠️ Medium (Post-MVP, strategic)

### Changes

#### 3.1 Permission Model

**File**: [`documents/models.py`](../documents/models.py)

```python
class DocumentPermission(models.Model):
    """Document-level access control."""

    document = models.ForeignKey(
        'Document',
        on_delete=models.CASCADE,
        related_name='permissions'
    )
    user = models.ForeignKey(
        'users.User',
        on_delete=models.CASCADE,
        related_name='document_permissions'
    )
    # Future: group = models.ForeignKey('users.UserGroup', ...)

    permission_type = models.CharField(
        max_length=20,
        choices=[
            ('VIEW', 'View'),
            ('DOWNLOAD', 'Download'),
            ('COMMENT', 'Comment'),
            ('EDIT_META', 'Edit Metadata'),
            ('SHARE', 'Share'),
            ('DELETE', 'Delete'),
        ],
        db_index=True
    )

    granted_by = models.ForeignKey(
        'users.User',
        on_delete=models.SET_NULL,
        null=True,
        related_name='granted_document_permissions'
    )
    granted_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True, db_index=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=('document', 'user', 'permission_type'),
                name='document_permission_unique'
            )
        ]
        indexes = [
            models.Index(fields=['document', 'user']),
            models.Index(fields=['user', 'permission_type']),
        ]
```

#### 3.2 Authorization Service

**File**: `documents/authz.py` (new file)

```python
"""Document authorization service."""

from dataclasses import dataclass
from django.utils.timezone import now
from documents.models import Document, DocumentPermission
from users.models import User
from customers.models import Tenant
from cases.models import Case
from cases.authz import user_can_access_case


@dataclass
class DocumentAccessResult:
    allowed: bool
    source: str | None = None  # 'document_permission', 'case_membership', 'role_based'
    reason: str | None = None


class DocumentAuthzService:
    """Document-level authorization."""

    @staticmethod
    def user_can_access_document(
        user: User,
        document: Document,
        permission_type: str = 'VIEW',
        tenant: Tenant | None = None
    ) -> DocumentAccessResult:
        """
        Check document access permission.

        Authorization hierarchy:
        1. Document owner
        2. Explicit document permission (highest priority)
        3. Case-level membership (if document in case)
        4. Tenant role-based access (TENANT_ADMIN, LEGAL, etc.)
        """

        # 0. Owners always have access
        if document.created_by_id == user.id:
            return DocumentAccessResult(allowed=True, source='owner')

        # 1. Check explicit document permission
        permission_exists = DocumentPermission.objects.filter(
            document=document,
            user=user,
            permission_type=permission_type
        ).filter(
            models.Q(expires_at__gt=now()) | models.Q(expires_at__isnull=True)
        ).exists()

        if permission_exists:
            return DocumentAccessResult(allowed=True, source='document_permission')

        # 2. Fallback to case-level authorization
        if document.case_id:
            case = Case.objects.filter(
                tenant=tenant, external_id=document.case_id
            ).first()
            if case and user_can_access_case(user, case, tenant):
                return DocumentAccessResult(allowed=True, source='case_membership')

        # 3. Fallback to role-based (TENANT_ADMIN all-access, etc.)
        # ... (similar logic to cases/authz.py)

        return DocumentAccessResult(allowed=False, reason='no_permission')
```

**Tests**:
- `documents/tests/test_authz.py::test_explicit_permission_grants_access`
- `documents/tests/test_authz.py::test_case_membership_grants_access`
- `documents/tests/test_authz.py::test_expired_permission_denies_access`
- `documents/tests/test_share_document.py::test_share_document_owner_grants_permission`
- `ai_core/tests/test_retrieve_permissions.py::test_filter_matches_by_permissions_allows_granted_docs`

#### 3.3 Integrate in Views

**File**: [`documents/views.py`](../documents/views.py)

```python
from documents.authz import DocumentAuthzService

def document_download(request, document_id):
    """Download with permission check."""

    document = Document.objects.get(pk=document_id)

    # Check permission
    access = DocumentAuthzService.user_can_access_document(
        user=request.user,
        document=document,
        permission_type='DOWNLOAD',
        tenant=request.tenant
    )

    if not access.allowed:
        return JsonResponse({'error': 'Permission denied'}, status=403)

    # Log + serve...
```

#### 3.4 Integrate in Retrieve Node

**File**: [`ai_core/nodes/retrieve.py`](../ai_core/nodes/retrieve.py)

Filter retrieval results by document permissions:

```python
def _filter_matches_by_permissions(
    matches: list[dict[str, object]],
    *,
    context: ToolContext,
    permission_type: str = "VIEW",
) -> list[dict[str, object]]:
    """Filter matches by document-level permissions."""

    if not context.scope.user_id:
        return matches  # Anonymous: no filtering (case-level already applied)

    # Resolve user + docs, then allow only matches that pass DocumentAuthzService
    # (owner, explicit permission, case membership, or tenant-role access).
```

#### 3.5 Share Endpoint

**Files**: `documents/views.py`, `documents/urls.py`

```python
@require_http_methods(["POST"])
def share_document(request, document_id: str):
    """Share document with another user."""

    user, auth_error = _require_authenticated_user(request)
    if auth_error:
        return auth_error

    # Only owner or TENANT_ADMIN can share
    if document.created_by_id != user.id and not _user_is_tenant_admin(user):
        return error(403, "PermissionDenied", "Only owner can share document")

    permission, _ = DocumentPermission.objects.update_or_create(
        document=document,
        user=target_user,
        permission_type=permission_type,
        defaults={"granted_by": user, "expires_at": expires_at},
    )

    ActivityTracker.log(
        document=document,
        activity_type=DocumentActivity.ActivityType.SHARE,
        user=user,
        metadata={"shared_with": str(target_user.id)},
    )

    return JsonResponse({"permission_id": permission.id}, status=201)
```

### Acceptance Criteria

- [x] `DocumentPermission` model created with all permission types
- [x] `DocumentAuthzService` checks permissions hierarchically
- [x] Download view blocks unauthorized access (403)
- [x] Retrieve node filters results by user permissions
- [x] Share endpoint creates permission grants
- [x] Only document owner or TENANT_ADMIN can share
- [x] Expired permissions auto-ignored in queries
- [ ] Tests pass:
  - `npm run test:py:single -- documents/tests/test_authz.py`
  - `npm run test:py:single -- documents/tests/test_share_document.py`
  - `npm run test:py:single -- ai_core/tests/test_retrieve_permissions.py`

### Code Locations

**Created**:
- `documents/authz.py` - Authorization service
- `documents/migrations/0020_create_document_permission.py` - Schema migration
- `documents/tests/test_authz.py` - Authorization tests
- `documents/tests/test_share_document.py` - Share endpoint tests
- `ai_core/tests/test_retrieve_permissions.py` - Retrieve permission filter tests

**Modified**:
- `documents/models.py` - Add DocumentPermission model
- `documents/views.py` - Permission checks
- `documents/urls.py` - Share endpoint route
- `documents/admin.py` - Permission admin
- `ai_core/nodes/retrieve.py` - Filter by permissions

---

## Phase 4a: Preferences, Collaboration, In-App Notifications (Pre-MVP)

**Goal**: Favorites, saved searches, comments, mentions, in-app notifications.

**Effort**: 5-7 days
**Priority**: Medium (Pre-MVP scope split)

### Changes

#### 4a.1 User Preferences

**File**: [`profiles/models.py`](../profiles/models.py)

Extend `UserProfile`:

```python
class UserProfile(models.Model):
    # ... existing fields ...

    # Document preferences
    document_view_mode = models.CharField(
        max_length=10,
        choices=[('LIST', 'List'), ('GRID', 'Grid'), ('TABLE', 'Table')],
        default='LIST'
    )
    documents_per_page = models.IntegerField(default=25)

    # Notification preferences
    notify_on_document_upload = models.BooleanField(default=True)
    notify_on_mention = models.BooleanField(default=True)
    notify_on_case_document = models.BooleanField(default=False)

```

#### 4a.2 Favorites

**File**: [`documents/models.py`](../documents/models.py)

```python
class UserDocumentFavorite(models.Model):
    user = models.ForeignKey('users.User', on_delete=models.CASCADE)
    document = models.ForeignKey('Document', on_delete=models.CASCADE)
    favorited_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [('user', 'document')]
```

**API**: `POST /documents/api/favorites/`, `DELETE /documents/api/favorites/{id}/`, `GET /documents/api/favorites/`

#### 4a.3 Comments + Mentions

**File**: [`documents/models.py`](../documents/models.py)

```python
class DocumentComment(models.Model):
    document = models.ForeignKey('Document', on_delete=models.CASCADE)
    user = models.ForeignKey('users.User', on_delete=models.CASCADE)
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.CASCADE)

    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Position tracking (inline comments)
    anchor_type = models.CharField(max_length=20, null=True)
    anchor_reference = models.JSONField(null=True)

class DocumentMention(models.Model):
    comment = models.ForeignKey('DocumentComment', on_delete=models.CASCADE)
    mentioned_user = models.ForeignKey('users.User', on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
```

**Mention Parsing:**
- Primary: rich token with user_id (`<@123>`)
- Fallback: `@username` when unique (case-insensitive)

**API**: `POST /documents/api/comments/`, `GET /documents/api/comments/?document_id=...`, `PUT /documents/api/comments/{id}/`, `DELETE /documents/api/comments/{id}/`

#### 4a.4 Saved Searches + Scheduler

**File**: [`documents/models.py`](../documents/models.py)

```python
class SavedSearch(models.Model):
    user = models.ForeignKey('users.User', on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    query = models.TextField()
    filters = models.JSONField(default=dict)

    created_at = models.DateTimeField(auto_now_add=True)
    last_run_at = models.DateTimeField(null=True)
    next_run_at = models.DateTimeField(null=True)

    enable_alerts = models.BooleanField(default=True)
    alert_frequency = models.CharField(max_length=20, default='HOURLY')
```

**Celery Task**: `documents/tasks.py:run_saved_search_alerts` (hourly default, bounded + incremental)

**API**: `POST /documents/api/saved-searches/`, `GET /documents/api/saved-searches/`, `PATCH /documents/api/saved-searches/{id}/`

#### 4a.5 In-App Notifications

**Model**: `documents/models.py:DocumentNotification`

**Inbox API**: `GET/PATCH /documents/api/notifications/` and `POST /documents/api/notifications/mark_all_read/`

---

## Phase 4b: External Notifications & Advanced Collaboration (Post-MVP)

**Goal**: External notification delivery (email/push) and advanced collaboration UX.

**Notes**:
- External delivery channels, templates, and subscription rules
- Advanced mentions/threads UX, notification routing

### Changes

#### 4b.1 External Email Preferences

**File**: [`profiles/models.py`](../profiles/models.py)

- `external_email_enabled` (bool, default false)
- `external_email_frequency` (IMMEDIATE | DAILY)
- `notify_on_comment_reply` (bool, default true)

#### 4b.2 External Notification Events + Deliveries

**File**: [`documents/models.py`](../documents/models.py)

- `NotificationEvent` (event log for external notifications)
- `NotificationDelivery` (delivery queue + status tracking)

#### 4b.3 Document Subscriptions

**File**: [`documents/models.py`](../documents/models.py)

- `DocumentSubscription` (who follows a document)

#### 4b.4 Dispatch + Delivery

**Files**:
- `documents/notification_dispatcher.py` (create events + deliveries)
- `documents/tasks.py:send_pending_email_deliveries` (retry/backoff)

**Rules**:
- Recipient must have VIEW permission, otherwise no delivery
- External email requires `external_email_enabled` + relevant `notify_on_*`

### Acceptance Criteria (Phase 4a)

- [x] UserProfile extended with document preferences
- [x] Favorites API (add/remove/list) functional
- [x] Comments system (CRUD) with threading support
- [x] @Mentions parsed and stored in DocumentMention
- [x] In-app notifications created for mentions + saved searches
- [x] Saved searches with hourly scheduler (bounded + incremental)
- [x] Tests pass:
  - `npm run test:py:single -- documents/tests/test_favorites_api.py`
  - `npm run test:py:single -- documents/tests/test_comments_api.py`
  - `npm run test:py:single -- documents/tests/test_notifications_api.py`
  - `npm run test:py:single -- documents/tests/test_saved_searches.py`

### Acceptance Criteria (Phase 4b)

- [x] NotificationEvent + NotificationDelivery models
- [x] DocumentSubscription model for collaboration subscriptions
- [x] UserProfile external email preferences (`external_email_enabled`, `external_email_frequency`)
- [x] Permission gate: recipients require VIEW before delivery
- [x] Dispatcher creates email deliveries for eligible events
- [x] Celery task sends pending email deliveries with retry backoff
- [x] Mention, saved search, comment reply emit external events
- [ ] External notification delivery (email/push) beyond email MVP
- [ ] Advanced collaboration UX (threads, subscriptions UI)

### Code Locations

**Created**:
- `documents/migrations/0021_add_collaboration_phase4a.py`
- `documents/migrations/0022_create_external_notifications_phase4b.py`
- `profiles/migrations/0003_add_document_preferences.py`
- `profiles/migrations/0004_add_external_email_preferences.py`
- `documents/tests/test_favorites_api.py`
- `documents/tests/test_comments_api.py`
- `documents/tests/test_notifications_api.py`
- `documents/tests/test_saved_searches.py`
- `documents/api_views.py` - DRF viewsets
- `documents/mentions.py` - Mention parsing
- `documents/notification_service.py` - In-app notifications
- `documents/notification_dispatcher.py` - External event + delivery dispatcher
- `documents/tasks.py` - Saved search scheduler

**Modified**:
- `profiles/models.py` - User preferences + external email settings
- `documents/models.py` - Favorites, Comments, Mentions, Notifications, SavedSearch, External notifications
- `documents/urls.py` - API router wiring

---

## Rollout Strategy

### Development

1. **Feature Branch**: `feature/document-user-integration`
2. **Phase-by-phase PRs**: Merge each phase separately
3. **Testing**: Full test suite after each phase

### Staging (Pre-MVP)

1. **Deploy Phase 1**: Test user attribution on staging
2. ~~Backfill Production Data~~ (NOT NEEDED - pre-MVP)
3. **Database Reset**: Acceptable for staging environment
4. **Deploy Phase 2**: Activity tracking validation
5. **Monitor**: Check Langfuse traces, ELK logs

### Production

1. **Rolling Deployment**: Phase 1 → Phase 2 → Phase 3
2. **Feature Flags**: Use `Tenant.features` JSON field to enable per-tenant
3. **Monitoring**: Alert on permission check failures, activity log gaps

### Rollback Plan

- **Phase 1**: Revert migration (drop FK columns, keep data)
- **Phase 2**: Disable activity logging (non-breaking)
- **Phase 3**: Revert permission checks (fallback to case-level)

---

## Dependencies

### Completed (Prerequisites)

- [x] User Management MVP (USER-MANAGEMENT branch)
- [x] CaseMembership model
- [x] ScopeContext with `user_id`
- [x] AuditMeta contract

### Parallel Work (Independent)

- Document summarization (AI features)
- Named entity recognition
- Advanced search (facets, fuzzy)

### Blocked By (Future)

- Real-time collaboration (requires WebSocket infrastructure)
- Notification service (requires email/push infrastructure)
- SSO integration (for external user onboarding)

---

## Metrics & Success Criteria

### Phase 1 Success Metrics (Pre-MVP)

- 100% of new uploads have `created_by` set (when authenticated)
- System uploads have `created_by = NULL` (crawler, scheduled tasks)
- ~~Backfill rate~~ (NOT APPLICABLE - pre-MVP clean start)
- Django Admin usable for user-document queries
- API response time: <50ms for document list with user attribution

### Phase 2 Success Metrics

- 100% of downloads logged in DocumentActivity
- Activity log query performance: <100ms for last 20 activities
- Recent documents API: <200ms response time
- No PII leaks in user_agent/metadata fields

### Phase 3 Success Metrics

- Permission checks: <10ms per document
- 0 unauthorized access incidents
- Share endpoint: <300ms response time
- Retrieve node performance: <5% degradation with permission filtering

### Phase 4a Success Metrics

- Favorites feature adoption: >30% of active users
- Comments engagement: >10 comments/week (pilot)
- Saved searches: >5 active searches per power user
- In-app notification latency: <=1h (hourly scheduler)

### Phase 4b Success Metrics

- External alert delivery: <5min latency from document upload

---

## Related Documentation

- **Architecture**: [`docs/architecture/user-document-integration.md`](../docs/architecture/user-document-integration.md)
- **User Management**: [`docs/roadmap/user-management-mvp-v2.md`](./user-management-mvp-v2.md)
- **ID Propagation**: [`docs/architecture/id-propagation.md`](../docs/architecture/id-propagation.md)
- **Multi-Tenancy**: [`docs/multi-tenancy.md`](../docs/multi-tenancy.md)

---

## Backlog Items

This roadmap supersedes the following backlog items:
- Document user attribution (Phase 1)
- Activity tracking compliance (Phase 2)
- Fine-grained permissions (Phase 3)

New items to add to [`roadmap/backlog.md`](./backlog.md):
- [ ] Phase 1 implementation (user attribution)
- [ ] Phase 2 implementation (activity tracking)
- [ ] Phase 3 implementation (permissions)
- [x] Phase 4a implementation (preferences + notifications)
- [ ] Phase 4b implementation (external notifications)
