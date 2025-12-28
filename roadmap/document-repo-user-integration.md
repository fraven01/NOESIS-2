# Document Repository: User Integration Roadmap

**Status**: Planning (Pre-MVP)
**Component**: Document Repository (`documents/`)
**Dependencies**: User Management (completed in USER-MANAGEMENT branch)
**Architecture Doc**: [`docs/architecture/user-document-integration.md`](../docs/architecture/user-document-integration.md)

## Overview

This roadmap tracks the integration of user identity, authorization, and activity tracking into the document repository system. The implementation is divided into four phases with increasing complexity.

**Pre-MVP Context**: Database resets are acceptable. No backward compatibility or data migration required. This significantly simplifies implementation.

**Current State**: User context flows through the system (`ScopeContext.user_id` → `ToolContext` → `AuditMeta`), but documents lack direct user relationships and fine-grained permissions.

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

    # Timestamps (Phase 1)
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

Update `register_document_from_upload` to set `created_by`:

```python
def register_document_from_upload(
    self,
    *,
    tenant_id: str,
    workflow_id: str,
    user_id: str | None = None,  # NEW: Accept user_id
    ...
) -> DocumentRef:
    """Register document with user attribution."""

    # Resolve User instance
    user = None
    if user_id:
        try:
            user = User.objects.get(pk=int(user_id))
        except (User.DoesNotExist, ValueError):
            logger.warning(f"User {user_id} not found for document attribution")

    # Create Document with user FK
    document = Document.objects.create(
        tenant=tenant,
        created_by=user,  # NEW
        updated_by=user,  # NEW
        ...
    )
```

**Tests**:
- `documents/tests/test_domain_service.py::test_register_document_sets_created_by`
- `documents/tests/test_domain_service.py::test_register_document_without_user`

#### 1.3 Upload Worker Integration

**File**: [`documents/upload_worker.py`](../documents/upload_worker.py)

Pass `user_id` from `ScopeContext`:

```python
def _run(self, state: dict, meta: dict) -> dict:
    """Upload worker with user context."""

    # Extract user_id from meta
    scope_context = meta.get('scope_context', {})
    user_id = scope_context.get('user_id')

    # Register document with user attribution
    doc_ref = self.domain_service.register_document_from_upload(
        tenant_id=scope_context['tenant_id'],
        workflow_id=state['workflow_id'],
        user_id=user_id,  # NEW
        ...
    )
```

**Tests**:
- `documents/tests/test_upload_worker.py::test_upload_sets_user_attribution`
- `documents/tests/test_upload_worker.py::test_upload_without_authenticated_user`

#### 1.4 Ingestion Graph Updates

**File**: [`ai_core/graphs/technical/universal_ingestion_graph.py`](../ai_core/graphs/technical/universal_ingestion_graph.py)

Ensure `user_id` flows from `ToolContext` to ingestion context:

```python
def _normalize_ingestion_context(state: dict, context: ToolContext | None) -> dict:
    """Extract ingestion context with user_id."""

    return {
        'tenant_id': context.scope.tenant_id,
        'user_id': context.scope.user_id,  # Ensure this is passed
        'workflow_id': context.business.workflow_id,
        ...
    }
```

**Tests**:
- `ai_core/tests/graphs/test_universal_ingestion_graph.py::test_user_context_propagation`

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

### Acceptance Criteria (Pre-MVP)

- [ ] `Document` model has `created_by`, `updated_by`, `created_at`, `updated_at` fields
- [ ] Migration successfully adds fields (nullable for system uploads)
- [ ] ~~Backfill script~~ (NOT NEEDED - pre-MVP clean start)
- [ ] New document uploads set `created_by` from authenticated user
- [ ] System uploads (crawler, scheduled) set `created_by = NULL` (no error)
- [ ] Django Admin shows user attribution in document list/detail
- [ ] API responses include `created_by` object (id, username, full_name)
- [ ] Tests pass: `npm run test:py:single -- documents/tests/test_user_attribution.py`
- [ ] Documentation updated: `documents/README.md` ✅ (already done)

### Code Locations

**Modified**:
- `documents/models.py` - Add FK fields (4 new fields)
- `documents/domain_service.py` - Accept `user_id` parameter (~5 lines)
- `documents/upload_worker.py` - Pass `user_id` to domain service (~3 lines)
- `ai_core/graphs/technical/universal_ingestion_graph.py` - Ensure user context flows (~2 lines)

**Created**:
- `documents/serializers.py` - API serializers (~30 lines)
- `documents/migrations/000X_add_user_attribution.py` - Schema migration (auto-generated)
- `documents/tests/test_user_attribution.py` - User attribution tests (~100 lines)

**Updated Tests**:
- `documents/tests/test_domain_service.py` - Add user_id assertions (~10 lines)
- `documents/tests/test_upload_worker.py` - Test user context propagation (~15 lines)
- `ai_core/tests/graphs/test_universal_ingestion_graph.py` - Test user in context (~10 lines)

**Total LOC**: ~175 lines (excluding migration)

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
    user_agent = models.TextField(blank=True, max_length=500)

    # Business context
    case_id = models.UUIDField(null=True, blank=True, db_index=True)
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

def download_document(request, document_id):
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
- `documents/tests/test_views.py::test_download_logs_activity`

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

**File**: `documents/api.py` (new file or extend existing)

```python
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from documents.models import Document, DocumentActivity


class DocumentViewSet(viewsets.ModelViewSet):
    """Document API with user-scoped endpoints."""

    @action(detail=False, methods=['get'])
    def recent(self, request):
        """Get user's recently accessed documents."""

        if not request.user.is_authenticated:
            return Response({'error': 'Authentication required'}, status=401)

        # Get recent activity
        recent_activities = DocumentActivity.objects.filter(
            user=request.user,
            activity_type__in=['VIEW', 'DOWNLOAD']
        ).order_by('-timestamp')[:20]

        # Get unique documents (preserve order)
        document_ids = []
        seen = set()
        for activity in recent_activities:
            if activity.document_id not in seen:
                document_ids.append(activity.document_id)
                seen.add(activity.document_id)

        documents = Document.objects.filter(id__in=document_ids[:10])

        serializer = DocumentSerializer(documents, many=True)
        return Response(serializer.data)
```

**Tests**:
- `documents/tests/test_api.py::test_recent_documents`
- `documents/tests/test_api.py::test_recent_documents_unauthenticated`

### Acceptance Criteria

- [ ] `DocumentActivity` model created with all activity types
- [ ] `ActivityTracker` service logs activities with full context
- [ ] Download endpoint logs DOWNLOAD activity
- [ ] Upload logs UPLOAD activity (via worker)
- [ ] Recent documents API returns user's last 10 accessed docs
- [ ] Activity log queryable via Django Admin
- [ ] Tests pass: `npm run test:py:single -- documents/tests/test_activity_service.py`
- [ ] API documentation updated (OpenAPI schema)

### Code Locations

**Created**:
- `documents/activity_service.py` - Activity tracker
- `documents/api.py` - Recent documents endpoint
- `documents/migrations/000X_create_document_activity.py` - Schema migration
- `documents/tests/test_activity_service.py` - Activity tracking tests
- `documents/tests/test_api.py` - API tests

**Modified**:
- `documents/models.py` - Add DocumentActivity model
- `documents/views.py` - Log download activity
- `documents/upload_worker.py` - Log upload activity

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
        related_name='granted_permissions'
    )
    granted_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True, db_index=True)

    class Meta:
        unique_together = [('document', 'user', 'permission_type')]
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
        1. Explicit document permission (highest priority)
        2. Case-level membership (if document in case)
        3. Tenant role-based access (TENANT_ADMIN, LEGAL, etc.)
        """

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
            try:
                case = Case.objects.get(external_id=document.case_id)
                case_access = user_can_access_case(user, case, tenant)
                if case_access.allowed:
                    return DocumentAccessResult(allowed=True, source='case_membership')
            except Case.DoesNotExist:
                pass

        # 3. Fallback to role-based (TENANT_ADMIN all-access, etc.)
        # ... (similar logic to cases/authz.py)

        return DocumentAccessResult(allowed=False, reason='no_permission')
```

**Tests**:
- `documents/tests/test_authz.py::test_explicit_permission_grants_access`
- `documents/tests/test_authz.py::test_case_membership_grants_access`
- `documents/tests/test_authz.py::test_expired_permission_denies_access`

#### 3.3 Integrate in Views

**File**: [`documents/views.py`](../documents/views.py)

```python
from documents.authz import DocumentAuthzService

def download_document(request, document_id):
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
def _filter_by_user_permissions(matches: list, user_id: str | None) -> list:
    """Filter matches by document-level permissions."""

    if not user_id:
        return matches  # Anonymous: no filtering (case-level already applied)

    user = User.objects.get(pk=int(user_id))

    filtered = []
    for match in matches:
        document = Document.objects.get(pk=match['document_id'])
        access = DocumentAuthzService.user_can_access_document(
            user=user,
            document=document,
            permission_type='VIEW'
        )
        if access.allowed:
            filtered.append(match)

    return filtered
```

#### 3.5 Share Endpoint

**File**: `documents/api.py`

```python
@action(detail=True, methods=['post'])
def share(self, request, pk=None):
    """Share document with user."""

    document = self.get_object()

    # Only owner or TENANT_ADMIN can share
    if document.created_by != request.user and not request.user.profile.role == 'TENANT_ADMIN':
        return Response({'error': 'Only document owner can share'}, status=403)

    # Create permission
    permission = DocumentPermission.objects.create(
        document=document,
        user_id=request.data['user_id'],
        permission_type=request.data.get('permission_type', 'VIEW'),
        granted_by=request.user,
        expires_at=request.data.get('expires_at')
    )

    # Log activity
    ActivityTracker.log(
        document=document,
        activity_type='SHARE',
        user=request.user,
        metadata={'shared_with': request.data['user_id']}
    )

    return Response({'id': permission.id, 'message': 'Document shared'})
```

### Acceptance Criteria

- [ ] `DocumentPermission` model created with all permission types
- [ ] `DocumentAuthzService` checks permissions hierarchically
- [ ] Download view blocks unauthorized access (403)
- [ ] Retrieve node filters results by user permissions
- [ ] Share endpoint creates permission grants
- [ ] Only document owner or TENANT_ADMIN can share
- [ ] Expired permissions auto-ignored in queries
- [ ] Tests pass: `npm run test:py:single -- documents/tests/test_authz.py`

### Code Locations

**Created**:
- `documents/authz.py` - Authorization service
- `documents/migrations/000X_create_document_permission.py` - Schema migration
- `documents/tests/test_authz.py` - Authorization tests

**Modified**:
- `documents/models.py` - Add DocumentPermission model
- `documents/views.py` - Permission checks
- `documents/api.py` - Share endpoint
- `ai_core/nodes/retrieve.py` - Filter by permissions

---

## Phase 4: User Preferences & Collaboration (Post-MVP)

**Goal**: Favorites, saved searches, comments, mentions.

**Effort**: 5-7 days
**Priority**: 📋 Low (Nice-to-have, post-MVP)

### Changes

#### 4.1 User Preferences

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

    # Search preferences
    default_search_scope = models.CharField(
        max_length=20,
        choices=[('ALL', 'All Cases'), ('MY_CASES', 'My Cases'), ('RECENT', 'Recent')],
        default='MY_CASES'
    )
```

#### 4.2 Favorites

**File**: [`documents/models.py`](../documents/models.py)

```python
class UserDocumentFavorite(models.Model):
    user = models.ForeignKey('users.User', on_delete=models.CASCADE)
    document = models.ForeignKey('Document', on_delete=models.CASCADE)
    favorited_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [('user', 'document')]
```

**API**: `POST /documents/{id}/favorite`, `DELETE /documents/{id}/favorite`, `GET /documents/favorites`

#### 4.3 Comments

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

**API**: `POST /documents/{id}/comments`, `GET /documents/{id}/comments`, `PUT /comments/{id}`, `DELETE /comments/{id}`

#### 4.4 Saved Searches

**File**: [`documents/models.py`](../documents/models.py)

```python
class SavedSearch(models.Model):
    user = models.ForeignKey('users.User', on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    query = models.TextField()
    filters = models.JSONField(default=dict)

    created_at = models.DateTimeField(auto_now_add=True)
    last_run = models.DateTimeField(null=True)

    enable_alerts = models.BooleanField(default=False)
    alert_frequency = models.CharField(max_length=20, default='DAILY')
```

**Celery Task**: `documents/tasks.py:run_saved_search_alerts` (periodic)

### Acceptance Criteria

- [ ] UserProfile extended with document preferences
- [ ] Favorites API (add/remove/list) functional
- [ ] Comments system (CRUD) with threading support
- [ ] @Mentions parsed and stored in DocumentMention
- [ ] Saved searches with alert scheduling
- [ ] Notification service sends alerts (email/push)
- [ ] Tests pass for all Phase 4 features

### Code Locations

**Created**:
- `documents/migrations/000X_add_favorites_comments_searches.py`
- `documents/tests/test_favorites.py`
- `documents/tests/test_comments.py`
- `documents/tests/test_saved_searches.py`
- `documents/tasks.py` - Saved search alerts

**Modified**:
- `profiles/models.py` - User preferences
- `documents/models.py` - Favorites, Comments, SavedSearch
- `documents/api.py` - New endpoints

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

### Phase 4 Success Metrics

- Favorites feature adoption: >30% of active users
- Comments engagement: >10 comments/week (pilot)
- Saved searches: >5 active searches per power user
- Alert delivery: <5min latency from document upload

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
- [ ] Phase 4 implementation (collaboration)
