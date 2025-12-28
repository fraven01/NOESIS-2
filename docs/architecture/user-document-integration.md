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
    created_by_user_id=scope.user_id,      # Document creator
    initiated_by_user_id=scope.user_id,     # Request initiator
    last_hop_service_id=scope.service_id,   # Service identifier (if S2S)
    last_modified_at=now
)
```

### Current User-Document Relationships

**Document Model** (`documents/models.py:Document`):
- **No direct FK** to User (no `created_by`/`updated_by` fields)
- Context fields: `workflow_id`, `trace_id`, `case_id` (for traceability)
- Lifecycle: `lifecycle_state`, `lifecycle_updated_at`

**Indirect Attribution** (via AuditMeta JSON):
- Stored in `NormalizedDocument.meta` during ingestion
- Not directly queryable via Django ORM
- Used for audit trail, not authorization

**DocumentCollectionMembership** (`documents/models.py`):
- `added_by` (CharField): Service or user identifier (string, not FK)
- `added_at` (DateTimeField): When document was added to collection

### Authorization Model

**Case-Based Access Control:**
- Primary authorization: `CaseMembership` (M2M: `Case` ↔ `User`)
- Function: `cases/authz.py:user_can_access_case(user, case, tenant)`
- Returns: `CaseAccessResult` with `allowed` boolean and context

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
- **No dedicated DocumentActivity model**: Downloads, views, searches not tracked

**What's NOT tracked:**
- Document downloads (no audit log)
- Document views/access
- Search queries by user
- Document modifications

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

### 2. No Document-Level Permissions

**Problem:**
- Authorization is case-level only (`CaseMembership`)
- Cannot grant individual document access within a case
- No sharing mechanisms (e.g., "share document Y with user Z")

**Impact:**
- Coarse-grained access control
- Cannot implement least-privilege principle
- No external sharing capability

### 3. No User Activity Audit Trail

**Problem:**
- Document downloads not logged
- No access history per document
- Compliance gap (GDPR, ISO 27001 require access logs)

**Impact:**
- Compliance risk
- Security blind spot (no detection of suspicious access)
- No usage analytics

### 4. No User Preferences for Documents

**Problem:**
- No favorites/bookmarks
- No "recent documents" feature
- No user-specific view preferences

**Impact:**
- Poor user experience
- Reduced productivity
- No personalization

### 5. No Collaboration Features

**Problem:**
- No comments/annotations system
- No @mentions or notifications
- No document-level discussions

**Impact:**
- No team collaboration
- Knowledge siloed
- Reduced engagement

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

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
```

**Migration Strategy:**
- Add nullable FK fields
- Backfill from `AuditMeta` JSON where possible
- Future uploads: Set from `ScopeContext.user_id`

**Integration Points:**
- `documents/domain_service.py:DocumentDomainService.register_document_from_upload`
- `documents/upload_worker.py:UploadWorker._run`
- `ai_core/graphs/technical/universal_ingestion_graph.py` (extract user from context)

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
        ],
        db_index=True
    )

    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)

    # Request context
    ip_address = models.GenericIPAddressField(null=True)
    user_agent = models.TextField(blank=True)

    # Business context
    case_id = models.UUIDField(null=True, db_index=True)
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
- `documents/views.py:download_document` - Log DOWNLOAD
- `documents/views.py:serve_asset` - Log VIEW (optional)
- `ai_core/nodes/retrieve.py` - Log SEARCH (when document returned)
- `documents/domain_service.py` - Log UPLOAD, DELETE

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
    user = models.ForeignKey('users.User', on_delete=models.CASCADE, null=True)
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
        related_name='granted_permissions'
    )
    granted_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = [('document', 'user', 'permission_type')]
        indexes = [
            models.Index(fields=['document', 'user']),
            models.Index(fields=['user', 'permission_type']),
            models.Index(fields=['expires_at']),
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
        1. Document-level permissions (explicit grant)
        2. Case-level permissions (CaseMembership)
        3. Role-based access (Tenant rules)
        """
        # Check explicit document permission
        if DocumentPermission.objects.filter(
            document=document,
            user=user,
            permission_type=permission_type,
            expires_at__gt=now() | expires_at__isnull=True
        ).exists():
            return DocumentAccessResult(allowed=True, source='document_permission')

        # Fallback to case-level authorization
        if document.case_id:
            case = Case.objects.get(external_id=document.case_id)
            case_access = user_can_access_case(user, case, tenant)
            if case_access.allowed:
                return DocumentAccessResult(allowed=True, source='case_membership')

        # Fallback to role-based (tenant-level all-cases access)
        # ...

        return DocumentAccessResult(allowed=False, reason='no_permission')
```

**Integration Points:**
- `documents/views.py:download_document` - Check permissions before serving
- `ai_core/nodes/retrieve.py` - Filter results by user permissions
- New endpoints: `POST /documents/{id}/share`, `DELETE /documents/{id}/permissions/{perm_id}`

### Phase 4: User Preferences & Collaboration (Post-MVP)

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

## Migration Strategy (Pre-MVP Simplified)

**Pre-MVP Context**: Database resets are acceptable. No backward compatibility or data migration required. This significantly simplifies all phases.

### Database Migrations

**Phase 1 (Simplified):**
1. Add `created_by`, `updated_by`, `created_at`, `updated_at` to Document (nullable for system uploads)
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

**Phase 4:**
1. Extend UserProfile with document preferences
2. Create UserDocumentFavorite model
3. Create DocumentComment, DocumentMention models

### Backward Compatibility (Pre-MVP: NOT REQUIRED)

**AuditMeta:**
- ~~Preservation~~ - Can be removed or simplified post-migration
- New system: Use FK fields as single source of truth
- ~~Dual-write~~ - NOT NEEDED (pre-MVP)

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
- `documents/tests/test_activity_tracking.py`: Test activity logging
- `documents/tests/test_views.py`: Update download tests

**Phase 3:**
- `documents/tests/test_authz.py`: Document permission checks
- `ai_core/tests/nodes/test_retrieve.py`: Permission-filtered retrieval

**Phase 4:**
- `documents/tests/test_favorites.py`: Favorites CRUD
- `documents/tests/test_comments.py`: Comments system

### Integration Tests

**End-to-end flows:**
1. User uploads document → `created_by` set → Activity logged
2. User downloads document → Permission checked → Activity logged
3. User shares document → Permission granted → Notification sent
4. User searches → Results filtered by permissions → Activity logged

### Acceptance Criteria

**Phase 1 (Pre-MVP):**
- [ ] Document model has `created_by`, `updated_by` FK fields
- [ ] New uploads set `created_by` from `ScopeContext.user_id`
- [ ] System uploads (crawler) have `created_by = NULL`
- [ ] Django Admin shows user attribution
- [ ] API returns `created_by` in document responses
- [ ] ~~Migration backfills~~ - NOT NEEDED (pre-MVP clean start)

**Phase 2:**
- [ ] DocumentActivity model exists with all activity types
- [ ] Download endpoint logs DOWNLOAD activity
- [ ] Upload logs UPLOAD activity
- [ ] Activity retention policy implemented
- [ ] Admin interface for activity log

**Phase 3:**
- [ ] DocumentPermission model supports VIEW/DOWNLOAD/COMMENT/EDIT_META
- [ ] DocumentAuthzService checks permissions before access
- [ ] Retrieve node filters by user permissions
- [ ] Share endpoint creates permission grants
- [ ] Expiring permissions auto-cleaned (Celery task)

**Phase 4:**
- [ ] UserProfile extended with document preferences
- [ ] Favorites API (add/remove/list)
- [ ] Comments system (create/reply/edit/delete)
- [ ] @Mentions trigger notifications
- [ ] Recent documents API

## Code Locations

### Existing Code (to modify)

**Models:**
- `documents/models.py:Document` - Add user FK fields
- `documents/models.py:DocumentCollectionMembership` - Change `added_by` to FK
- `profiles/models.py:UserProfile` - Add document preferences

**Services:**
- `documents/domain_service.py:DocumentDomainService.register_document_from_upload` - Set created_by
- `documents/upload_worker.py:UploadWorker._run` - Pass user_id to service
- `documents/collection_service.py` - Update membership attribution

**Views:**
- `documents/views.py:download_document` - Add activity logging + permission check
- `documents/views.py:serve_asset` - Add activity logging

**Graphs:**
- `ai_core/graphs/technical/universal_ingestion_graph.py` - Extract user from ToolContext

**Retrieval:**
- `ai_core/nodes/retrieve.py` - Filter by document permissions

### New Code (to create)

**Models:**
- `documents/models.py:DocumentActivity`
- `documents/models.py:DocumentPermission`
- `documents/models.py:UserDocumentFavorite`
- `documents/models.py:DocumentComment`
- `documents/models.py:DocumentMention`

**Services:**
- `documents/authz.py:DocumentAuthzService` - Authorization logic
- `documents/activity_service.py:ActivityTracker` - Activity logging abstraction
- `documents/preferences_service.py` - User preferences management

**Views/APIs:**
- `documents/api.py:DocumentPermissionViewSet` - Permission management
- `documents/api.py:DocumentActivityViewSet` - Activity log (read-only)
- `documents/api.py:DocumentFavoriteViewSet` - Favorites CRUD
- `documents/api.py:DocumentCommentViewSet` - Comments CRUD

**Tasks:**
- `documents/tasks.py:clean_expired_permissions` - Celery periodic task
- `documents/tasks.py:archive_old_activities` - Activity retention

**Migrations:**
- `documents/migrations/000X_add_user_attribution.py`
- `documents/migrations/000X_create_document_activity.py`
- `documents/migrations/000X_create_document_permission.py`
- `documents/migrations/000X_create_collaboration_models.py`

## Security Considerations

### Permission Escalation Prevention

**Constraints:**
- Only `created_by` or `TENANT_ADMIN` can grant document permissions
- External accounts cannot grant permissions (even if TENANT_ADMIN role)
- Permission expiry enforced at query time + periodic cleanup

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
