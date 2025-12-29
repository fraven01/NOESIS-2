# User-Document Integration Architecture

This document describes how user identity and authorization integrates with the document repository system in NOESIS 2.

**Context**: Pre-MVP implementation. Database resets are acceptable, no backward compatibility or data migration required.

## Current State (USER-MANAGEMENT branch)

### User Context Flow

**Canonical implementations:**
- User models: `users/models.py:User`, `profiles/models.py:UserProfile`
- Scope context: `ai_core/contracts/scope.py:ScopeContext` (includes `user_id`)
- Audit metadata: `ai_core/contracts/audit_meta.py:AuditMeta`
- Authorization: `cases/authz.py:user_can_access_case`, `cases/authz.py:get_accessible_cases_queryset`

### Identity Propagation

**HTTP Request → ScopeContext:**
```python
# ai_core/ids/http_scope.py:normalize_request
user = request.user
if user and user.is_authenticated:
    user_id = str(user.pk)  # String representation of User.pk
else:
    user_id = None

scope = ScopeContext(
    tenant_id=...,
    trace_id=...,
    user_id=user_id,  # Propagated through system
    ...
)
```

**ScopeContext → ToolContext:**
```python
# ai_core/tool_contracts/base.py:ToolContext
context = ToolContext(
    scope=scope,  # Contains user_id
    business=business_context,
    metadata={...}
)
```

**ToolContext → AuditMeta:**
```python
# ai_core/contracts/audit_meta.py:AuditMeta
audit = AuditMeta(
    created_by_user_id=created_by_user_id,  # Ownership signal
    initiated_by_user_id=initiated_by_user_id,  # Request initiator
    last_hop_service_id=scope.service_id,   # Service identifier (S2S hop)
    last_modified_at=now
)
```
Notes:
- For user request hops, `created_by_user_id` defaults to `scope.user_id`.
- For S2S hops, `created_by_user_id` is preserved via `audit_meta` in tool metadata.

### Current User-Document Relationships

**Document Model** (`documents/models.py:Document`):
- Direct FK to User via `created_by` / `updated_by`
- Context fields: `workflow_id`, `trace_id`, `case_id` (for traceability)
- Lifecycle: `lifecycle_state`, `lifecycle_updated_at`

**Ownership Source** (via AuditMeta):
- `AuditMeta.created_by_user_id` drives `Document.created_by` (write-once)
- `AuditMeta` is transported through graph metadata to repository upserts

**DocumentCollectionMembership** (`documents/models.py`):
- `added_by_user` (FK) / `added_by_service_id` (CharField), both nullable (pre-MVP)
- `added_at` (DateTimeField): When document was added to collection

### Authorization Model

**Case-Based Access Control:**
- Primary authorization: `CaseMembership` (M2M: `Case` ↔ `User`)
- Function: `cases/authz.py:user_can_access_case(user, case, tenant)`
- Returns: `bool`

**Document-Level Access Control:**
- Explicit grants: `documents/models.py:DocumentPermission`
- Authorization service: `documents/authz.py:DocumentAuthzService`
- Enforced in: `documents/views.py:document_download`, `ai_core/nodes/retrieve.py`

**Role-Based Rules** (tenant-type dependent):

ENTERPRISE mode (`customers/models.py:Tenant.tenant_type == 'ENTERPRISE'`):
- `TENANT_ADMIN`: All cases (read/write)
- `LEGAL`: All cases (read)
- `MANAGEMENT`: All cases (read)
- `WORKS_COUNCIL`: Depends on `Tenant.works_council_scope` (ASSIGNED | ALL)
- `STAKEHOLDER`: Membership required
- `EXTERNAL` accounts: **Always** membership-based (overrides role)

LAW_FIRM mode (`tenant_type == 'LAW_FIRM'`):
- `TENANT_ADMIN`: All cases
- Everyone else: Membership required (including LEGAL, MANAGEMENT)

**Visibility Filtering:**
- Document lifecycle visibility: `ai_core/rag/visibility.py:Visibility` (ACTIVE | ALL | DELETED)
- Applied in: `ai_core/nodes/retrieve.py` (RAG retrieval)
- Blocks deleted documents unless explicitly requested

### User Activity Tracking

**Current Implementation:**
- **CaseEvent** (`cases/models.py`): Structured event log with `payload` (JSONField)
- **Langfuse Traces**: Production observability (not persisted in DB for user queries)
- **DocumentActivity** (`documents/models.py`): Append-only activity log
  - Download events logged in `documents/views.py:document_download`
  - Upload events logged in `documents/upload_worker.py:UploadWorker`

**What's NOT tracked (yet):**
- Document views/access (non-download)
- Search queries by user
- Document modifications beyond upload/download

## Identified Gaps

### 1. No Direct User Ownership

**Problem:**
- Cannot query "documents created by user X" efficiently
- No Django Admin integration for user-document relationship
- Ownership transfer requires manual JSON manipulation

**Impact:**
- Poor UX for document management
- Difficult audit queries
- No ownership lifecycle management

**Status:** Implemented in Phase 1 (created_by/updated_by + membership attribution).

### 2. No Document-Level Permissions

**Problem:**
- Authorization is case-level only (`CaseMembership`)
- Cannot grant individual document access within a case
- No sharing mechanisms (e.g., "share document Y with user Z")

**Impact:**
- Coarse-grained access control
- Cannot implement least-privilege principle
- No external sharing capability

**Status:** Implemented in Phase 3 (DocumentPermission + authz + share endpoint).

### 3. No User Activity Audit Trail

**Problem:**
- Document downloads not logged
- No access history per document
- Compliance gap (GDPR, ISO 27001 require access logs)

**Impact:**
- Compliance risk
- Security blind spot (no detection of suspicious access)
- No usage analytics

**Status:** Implemented in Phase 2 (DocumentActivity + download/upload logging).

### 4. No User Preferences for Documents

**Problem:**
- No favorites/bookmarks
- No "recent documents" feature
- No user-specific view preferences

**Impact:**
- Poor user experience
- Reduced productivity
- No personalization

**Status:** Implemented in Phase 4a (preferences, favorites, saved searches).

### 5. No Collaboration Features

**Problem:**
- No comments/annotations system
- No @mentions or notifications
- No document-level discussions

**Impact:**
- No team collaboration
- Knowledge siloed
- Reduced engagement

**Status:** Core comments/mentions implemented in Phase 4a; external notifications pending.

## Proposed Architecture

### Phase 1: Direct User Attribution (MVP-Critical)

**New Fields on Document Model:**
```python
# documents/models.py:Document
class Document(models.Model):
    # ... existing fields ...

    # User attribution
    created_by = models.ForeignKey(
        'users.User',
        on_delete=models.SET_NULL,
        null=True,
        related_name='created_documents',
        db_index=True
    )
    updated_by = models.ForeignKey(
        'users.User',
        on_delete=models.SET_NULL,
        null=True,
        related_name='updated_documents'
    )

    # Timestamps (already present, no change needed)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
```

**Migration Strategy:**
- Add nullable FK fields
- No backfill required (DB reset acceptable)
- Future uploads: Set ownership via `AuditMeta.created_by_user_id` (not `initiated_by_user_id`)

**Integration Points:**
- `documents/domain_service.py:DocumentDomainService.ingest_document` (write-once created_by)
- `documents/upload_worker.py:UploadWorker.process` (preserve audit_meta, consistent workflow_id)
- `ai_core/tasks.py` (carry audit_meta into ToolContext metadata)
- `ai_core/adapters/db_documents_repository.py` (set created_by/updated_by from audit_meta)

**Collection Membership Attribution (Phase 1):**
- Replace `added_by` with `added_by_user` + `added_by_service_id` (nullable).
- Pre-MVP: allow both NULL; Post-MVP: add XOR constraint if desired.
- Source: `audit_meta.created_by_user_id` first, else `audit_meta.last_hop_service_id`.

### Phase 2: Document Activity Tracking (Compliance)

**New Model:**
```python
# documents/models.py:DocumentActivity
class DocumentActivity(models.Model):
    document = models.ForeignKey('Document', on_delete=models.CASCADE, db_index=True)
    user = models.ForeignKey('users.User', on_delete=models.SET_NULL, null=True)

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
    ip_address = models.GenericIPAddressField(null=True)
    user_agent = models.CharField(max_length=500, blank=True, default="")

    # Business context
    case_id = models.CharField(max_length=255, null=True, db_index=True)
    trace_id = models.CharField(max_length=255, null=True, db_index=True)

    # Optional metadata
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=['document', '-timestamp']),
            models.Index(fields=['user', '-timestamp']),
            models.Index(fields=['activity_type', '-timestamp']),
            models.Index(fields=['case_id', '-timestamp']),
        ]
```

**Integration Points:**
- `documents/views.py:document_download` - Log DOWNLOAD
- `documents/upload_worker.py:UploadWorker` - Log UPLOAD (pre-registration path)
- `documents/views.py:recent_documents` - Recent documents API

**Retention Policy:**
- Keep activity logs for configurable period (default: 90 days)
- Archive to S3 for long-term compliance
- Implement via Celery periodic task

### Phase 3: Document-Level Permissions (Strategic)

**New Model:**
```python
# documents/models.py:DocumentPermission
class DocumentPermission(models.Model):
    document = models.ForeignKey('Document', on_delete=models.CASCADE)
    user = models.ForeignKey('users.User', on_delete=models.CASCADE)
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
        ]
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

**Authorization Service:**
```python
# documents/authz.py (new file)
class DocumentAuthzService:
    """Document-level authorization service."""

    @staticmethod
    def user_can_access_document(
        user: User,
        document: Document,
        permission_type: str = 'VIEW',
        tenant: Tenant = None
    ) -> DocumentAccessResult:
        """
        Check if user can access document with given permission.

        Authorization hierarchy:
        1. Owner access
        2. Document-level permissions (explicit grant)
        3. Case-level permissions (CaseMembership)
        4. Role-based access (Tenant rules)
        """
        # Owner access
        if document.created_by_id == user.id:
            return DocumentAccessResult(allowed=True, source='owner')

        # Check explicit document permission
        # Q imported from django.db.models
        if DocumentPermission.objects.filter(
            document=document,
            user=user,
            permission_type=permission_type,
        ).filter(
            Q(expires_at__gt=now()) | Q(expires_at__isnull=True)
        ).exists():
            return DocumentAccessResult(allowed=True, source='document_permission')

        # Fallback to case-level authorization
        if document.case_id:
            case = Case.objects.filter(
                tenant=tenant, external_id=document.case_id
            ).first()
            if case and user_can_access_case(user, case, tenant):
                return DocumentAccessResult(allowed=True, source='case_membership')

        # Fallback to role-based (tenant-level all-cases access)
        # ...

        return DocumentAccessResult(allowed=False, reason='no_permission')
```

**Integration Points:**
- `documents/views.py:document_download` - Check permissions before serving
- `ai_core/nodes/retrieve.py` - Filter results by user permissions
- Share endpoint: `POST /documents/share/<uuid:document_id>/`

### Phase 4a: Preferences, Collaboration, In-App Notifications (Pre-MVP)

**User Preferences:**
```python
# profiles/models.py:UserProfile (extend existing model)
class UserProfile(models.Model):
    # ... existing fields ...

    # Document preferences
    document_view_mode = models.CharField(max_length=10, default='LIST')
    documents_per_page = models.IntegerField(default=25)

    # Notification preferences
    notify_on_document_upload = models.BooleanField(default=True)
    notify_on_mention = models.BooleanField(default=True)
    notify_on_case_document = models.BooleanField(default=False)
```

**Favorites:**
```python
# documents/models.py:UserDocumentFavorite
class UserDocumentFavorite(models.Model):
    user = models.ForeignKey('users.User', on_delete=models.CASCADE)
    document = models.ForeignKey('Document', on_delete=models.CASCADE)
    favorited_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [('user', 'document')]
```

**Comments & Collaboration:**
```python
# documents/models.py:DocumentComment
class DocumentComment(models.Model):
    document = models.ForeignKey('Document', on_delete=models.CASCADE)
    user = models.ForeignKey('users.User', on_delete=models.CASCADE)
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.CASCADE)

    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Position tracking (for inline comments)
    anchor_type = models.CharField(max_length=20, null=True)  # 'text', 'page', 'asset'
    anchor_reference = models.JSONField(null=True)  # {page: 3, bbox: [...]}

# documents/models.py:DocumentMention
class DocumentMention(models.Model):
    comment = models.ForeignKey('DocumentComment', on_delete=models.CASCADE)
    mentioned_user = models.ForeignKey('users.User', on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
```

**In-App Notifications:**
```python
# documents/models.py:DocumentNotification
class DocumentNotification(models.Model):
    user = models.ForeignKey('users.User', on_delete=models.SET_NULL, null=True)
    document = models.ForeignKey('Document', on_delete=models.SET_NULL, null=True)
    comment = models.ForeignKey('DocumentComment', on_delete=models.SET_NULL, null=True)
    event_type = models.CharField(choices=[('MENTION', 'Mention'), ...])
    payload = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    read_at = models.DateTimeField(null=True, blank=True)
```

**Mention Parsing (Phase 4a):**
- Primary: rich token with user_id (`<@123>`)
- Fallback: `@username` when the username is unique (case-insensitive)

**Saved Searches (Phase 4a):**
- `SavedSearch` model stores query + filters
- Scheduler runs hourly by default (bounded + incremental)
- Emits in-app notifications with match counts

### Phase 4b: External Notifications & Advanced Collaboration (Post-MVP)

**Scope (Phase 4b MVP):**
- External notification pipeline with persisted events + deliveries
- Document subscriptions powering reply notifications
- User-level email delivery preferences (enabled + frequency + comment reply)

**Core Models:**
- `NotificationEvent` (external event log)
- `NotificationDelivery` (email delivery queue + status)
- `DocumentSubscription` (who follows a document)

## Migration Strategy (Pre-MVP Simplified)

**Pre-MVP Context**: Database resets are acceptable. No backward compatibility or data migration required. This significantly simplifies all phases.

### Database Migrations

**Phase 1 (Simplified):**
1. Add `created_by`, `updated_by` to Document (timestamps already exist)
2. ~~Backfill script~~ - NOT NEEDED (pre-MVP clean start)
3. Create indexes on new fields (auto-generated)

**Phase 2:**
1. Create DocumentActivity model
2. Integrate logging in views/services
3. No migration of existing activities (clean start)

**Phase 3:**
1. Create DocumentPermission model
2. Implement DocumentAuthzService
3. Integrate in retrieve node and views

**Phase 4a:**
1. Extend UserProfile with document preferences
2. Create UserDocumentFavorite model
3. Create DocumentComment, DocumentMention models
4. Create DocumentNotification + SavedSearch models

**Phase 4b:**
1. Add `NotificationEvent`, `NotificationDelivery`, `DocumentSubscription`
2. Extend `UserProfile` with external email preferences
3. External notification dispatch + delivery providers
4. Collaboration enhancements beyond Phase 4a

### Backward Compatibility (Pre-MVP: NOT REQUIRED)

**AuditMeta:**
- Ownership source: `created_by_user_id` drives `Document.created_by`
- Transport: carried in `ToolContext.metadata` through ingestion to `upsert()`
- Storage: not persisted in `Document.metadata` (pre-MVP simplification)

**API Compatibility:**
- Breaking changes acceptable (pre-MVP)
- All endpoints can immediately use `created_by` (no gradual rollout)
- ~~Versioning~~ - NOT NEEDED (no production users yet)

## Testing Strategy

### Unit Tests

**Phase 1:**
- `documents/tests/test_user_attribution.py`: Test created_by/updated_by propagation
- `documents/tests/test_domain_service.py`: Update existing tests

**Phase 2:**
- `documents/tests/test_activity_service.py`: Test activity logging
- `documents/tests/test_download_view.py`: Download logging coverage

**Phase 3:**
- `documents/tests/test_authz.py`: Document permission checks
- `documents/tests/test_share_document.py`: Share endpoint
- `ai_core/tests/test_retrieve_permissions.py`: Permission-filtered retrieval

**Phase 4a:**
- `documents/tests/test_favorites_api.py`: Favorites CRUD
- `documents/tests/test_comments_api.py`: Comments system + mentions
- `documents/tests/test_notifications_api.py`: In-app notifications
- `documents/tests/test_saved_searches.py`: Saved searches + scheduler

### Integration Tests

**End-to-end flows:**
1. User uploads document → `created_by` set → Activity logged
2. User downloads document → Permission checked → Activity logged
3. User shares document → Permission granted → Notification sent
4. User searches → Results filtered by permissions → Activity logged

### Acceptance Criteria

**Phase 1 (Pre-MVP):**
- [x] Document model has `created_by`, `updated_by` FK fields
- [x] New uploads set `created_by` from `AuditMeta.created_by_user_id`
- [x] System uploads (crawler) have `created_by = NULL`
- [x] DocumentCollectionMembership has `added_by_user` / `added_by_service_id` (nullable)
- [x] Django Admin shows user attribution
- [x] API returns `created_by` in document responses
- [ ] ~~Migration backfills~~ - NOT NEEDED (pre-MVP clean start)

**Phase 2:**
- [x] DocumentActivity model exists with all activity types
- [x] Download endpoint logs DOWNLOAD activity
- [x] Upload logs UPLOAD activity
- [ ] Activity retention policy implemented
- [x] Admin interface for activity log

**Phase 3:**
- [x] DocumentPermission model supports VIEW/DOWNLOAD/COMMENT/EDIT_META
- [x] DocumentAuthzService checks permissions before access
- [x] Retrieve node filters by user permissions
- [x] Share endpoint creates permission grants
- [ ] Expiring permissions auto-cleaned (Celery task)

**Phase 4a:**
- [x] UserProfile extended with document preferences
- [x] Favorites API (add/remove/list)
- [x] Comments system (create/reply/edit/delete)
- [x] @Mentions trigger in-app notifications
- [x] Saved searches with hourly scheduler (bounded + incremental)
- [x] In-app notifications inbox endpoints
- [x] Recent documents API (delivered in Phase 2)

**Phase 4b:**
- [x] NotificationEvent + NotificationDelivery models
- [x] DocumentSubscription model for collaboration subscriptions
- [x] UserProfile external email preferences (enabled + frequency)
- [x] Permission gate: recipients require VIEW before delivery
- [x] Dispatcher creates email deliveries for eligible events
- [x] Celery task sends pending email deliveries with retry backoff
- [x] Mention, saved search, comment reply emit external events
- [ ] External notification delivery (email/push) beyond email MVP
- [ ] Collaboration enhancements (advanced mentions, threading UX)

## Code Locations

### Existing Code (to modify)

**Models:**
- `documents/models.py:Document` - Add user FK fields
- `documents/models.py:DocumentCollectionMembership` - Add `added_by_user` + `added_by_service_id` (pre-MVP allow NULL)
- `profiles/models.py:UserProfile` - Add document preferences

**Services:**
- `documents/domain_service.py:DocumentDomainService.ingest_document` - Set created_by via audit/actor fields
- `documents/upload_worker.py:UploadWorker.process` - Preserve audit_meta and workflow_id
- `documents/tasks.py:upload_document_task` - Pass audit_meta inputs through to worker
- `ai_core/tasks.py:run_ingestion_graph` - Carry audit_meta into ToolContext metadata
- `ai_core/adapters/db_documents_repository.py:DbDocumentsRepository.upsert` - Apply audit_meta to created_by/updated_by
- `documents/collection_service.py` - Update membership attribution

**Views:**
- `documents/views.py:document_download` - Add activity logging + permission check
- `documents/views.py:asset_serve` - Add activity logging

**Graphs:**
- `ai_core/graphs/technical/universal_ingestion_graph.py` - Use ToolContext metadata for persistence inputs

**Retrieval:**
- `ai_core/nodes/retrieve.py` - Filter by document permissions

### New Code (to create)

**Models:**
- `documents/models.py:DocumentActivity`
- `documents/models.py:DocumentPermission`
- `documents/models.py:UserDocumentFavorite`
- `documents/models.py:DocumentComment`
- `documents/models.py:DocumentMention`
- `documents/models.py:DocumentNotification`
- `documents/models.py:NotificationEvent`
- `documents/models.py:NotificationDelivery`
- `documents/models.py:DocumentSubscription`
- `documents/models.py:SavedSearch`

**Services:**
- `documents/authz.py:DocumentAuthzService` - Authorization logic
- `documents/activity_service.py:ActivityTracker` - Activity logging abstraction
- `documents/mentions.py` - Mention parsing helpers
- `documents/notification_service.py` - In-app notifications
- `documents/notification_dispatcher.py` - External notifications + delivery dispatch
- (Post-MVP) `documents/preferences_service.py` - User preferences management

**Views/APIs:**
- `documents/views.py:document_download` - Permission checks + download
- `documents/views.py:share_document` - Share endpoint
- `documents/urls.py` - Share route wiring
- `documents/api_views.py` - Favorites, comments, saved searches, notifications
- (Post-MVP) `documents/api.py:DocumentPermissionViewSet` - Permission management
- (Post-MVP) `documents/api.py:DocumentActivityViewSet` - Activity log (read-only)

**Tasks:**
- `documents/tasks.py:clean_expired_permissions` - Celery periodic task
- `documents/tasks.py:archive_old_activities` - Activity retention
- `documents/tasks.py:run_saved_search_alerts` - Saved search scheduler
- `documents/tasks.py:send_pending_email_deliveries` - External email delivery

**Migrations:**
- `documents/migrations/0018_add_user_attribution_and_membership_actor.py`
- `documents/migrations/0019_create_document_activity.py`
- `documents/migrations/0020_create_document_permission.py`
- `documents/migrations/0021_add_collaboration_phase4a.py`
- `profiles/migrations/0003_add_document_preferences.py`
- `documents/migrations/0022_create_external_notifications_phase4b.py`
- `profiles/migrations/0004_add_external_email_preferences.py`
- (Post-MVP) `documents/migrations/00XX_create_collaboration_models.py`

## Security Considerations

### Permission Escalation Prevention

**Constraints:**
- Only `created_by` or `TENANT_ADMIN` can grant document permissions
- External accounts are restricted from tenant-wide access in DocumentAuthzService
- Permission expiry enforced at query time (cleanup task pending)

### Audit Trail Integrity

**Immutability:**
- DocumentActivity: No updates/deletes (append-only)
- Soft-delete documents: Keep activity log intact
- Archive old activities (don't delete)

### PII Compliance

**Data Minimization:**
- User agent: Truncate to 500 chars
- IP address: Optional (configurable retention)
- Metadata: Sanitize before storage

**GDPR Right to Erasure:**
- User deletion: SET_NULL on FK fields (preserve audit trail)
- Activity log: Anonymize user_id (don't delete rows)

## Performance Considerations

### Indexing Strategy

**DocumentActivity:**
- Primary: `(document, -timestamp)` - Document history
- Secondary: `(user, -timestamp)` - User activity feed
- Tertiary: `(activity_type, -timestamp)` - Activity type analytics
- Quaternary: `(case_id, -timestamp)` - Case activity

**DocumentPermission:**
- Primary: `(document, user)` - Permission lookup
- Secondary: `(user, permission_type)` - User permission inventory
- Tertiary: `(expires_at)` - Expiry cleanup

**Query Optimization:**
- Use `select_related('created_by', 'updated_by')` to avoid N+1
- Paginate activity logs (never load all)
- Cache permission checks (Redis, 5min TTL)

### Scaling Strategy

**Activity Log Partitioning:**
- Partition by month (PostgreSQL native partitioning)
- Archive partitions older than retention period
- S3 for long-term storage

**Permission Cache:**
- Redis cache: `document:{id}:user:{id}:perms` → Set of permission types
- Invalidate on permission grant/revoke
- TTL: 5 minutes

## Observability

### Metrics (Langfuse)

**Document Activity:**
- Tag: `activity_type`, `user_id`, `document_id`, `case_id`
- Metric: Activity count by type (hourly)

**Permission Checks:**
- Tag: `permission_type`, `result` (allowed/denied)
- Metric: Authorization decision latency

**User Engagement:**
- Tag: `user_id`, `tenant_id`
- Metric: Active users (daily/weekly)

### Logging (ELK)

**Structured Logs:**
- Document access: `document_id`, `user_id`, `action`, `result`, `duration_ms`
- Permission grants: `document_id`, `grantee_user_id`, `grantor_user_id`, `permission_type`
- Failed authorization: `document_id`, `user_id`, `required_permission`, `reason`

## References

**Existing Documentation:**
- ID propagation: `docs/architecture/id-propagation.md`
- ID semantics: `docs/architecture/id-semantics.md`
- Multi-tenancy: `docs/multi-tenancy.md`
- User management MVP: `docs/roadmap/user-management-mvp-v2.md`

**Related Code:**
- User models: `users/models.py`, `profiles/models.py`
- Authorization: `cases/authz.py`, `profiles/authentication.py`
- Scope context: `ai_core/contracts/scope.py`
- Audit metadata: `ai_core/contracts/audit_meta.py`

**Standards:**
- GDPR: Right to erasure, audit trail requirements
- ISO 27001: Access control, activity logging
- OWASP: Secure permission checks, injection prevention
