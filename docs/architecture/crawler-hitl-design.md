# HITL (Human-In-The-Loop) Crawler Design

## Overview

Add manual review step between fetch and ingestion.

---

## 🔄 Two-Phase Flow

### Phase 1: Fetch & Preview (HITL)

```
User submits URLs
    ↓
Celery Task: fetch_for_review
    ↓
Fetch content (no persistence yet)
    ↓
Extract preview (title, snippet, metadata)
    ↓
Store in temporary cache (Redis/DB)
    ↓
Return preview_session_id
    ↓
UI displays previews
    ↓
User reviews and selects
    ↓
User clicks "Ingest Selected"
```

### Phase 2: Ingest Selected

```
User confirms selection
    ↓
API receives selected URLs
    ↓
Celery Task: ingest_urls (from cache)
    ↓
Process selected URLs (Mode: RAG/Archive/Ephemeral)
    ↓
Return results
```

---

## 🎨 API Design

### Step 1: Create Review Session

```python
POST /api/v1/crawler/fetch-for-review
{
  "urls": ["https://example.com", "https://example.org"],
  "session_metadata": {
    "source": "intelligent_search",
    "query": "neural networks"
  }
}

Response:
{
  "session_id": "review-session-uuid",
  "task_id": "celery-task-id",
  "urls_queued": 2,
  "status": "fetching"
}
```

### Step 2: Get Preview Results

```python
GET /api/v1/crawler/review-session/{session_id}

Response:
{
  "session_id": "review-session-uuid",
  "status": "completed",  // "fetching" | "completed" | "failed"
  "previews": [
    {
      "url": "https://example.com",
      "title": "Neural Networks Explained",
      "snippet": "A neural network is a method in artificial...",
      "content_type": "text/html",
      "size_bytes": 45621,
      "fetch_timestamp": "2025-12-11T08:24:00Z",
      "metadata": {
        "language": "en",
        "author": "John Doe",
        "published_date": "2024-01-15"
      },
      "thumbnail_url": "/api/v1/crawler/thumbnail/{preview_id}",  // optional
      "status": "fetched"
    },
    {
      "url": "https://example.org",
      "title": null,
      "snippet": null,
      "status": "failed",
      "error": "Connection timeout"
    }
  ],
  "expires_at": "2025-12-11T09:24:00Z"  // 1 hour TTL
}
```

### Step 3: Ingest Selected

```python
POST /api/v1/crawler/ingest
{
  "session_id": "review-session-uuid",
  "selected_urls": ["https://example.com"],  // User selected subset
  "mode": "rag",
  "collection_id": "uuid-here",
  "embedding_profile": "standard"
}

Response:
{
  "task_id": "celery-task-id",
  "urls_queued": 1,
  "mode": "rag"
}
```

---

## 🗄️ Data Model

### ReviewSession

```python
@dataclass
class ReviewSession:
    session_id: UUID
    tenant_id: str
    case_id: str | None
    urls: list[str]
    status: Literal["fetching", "completed", "failed", "expired"]
    created_at: datetime
    expires_at: datetime
    metadata: dict[str, Any]
```

### FetchedPreview

```python
@dataclass
class FetchedPreview:
    preview_id: UUID
    session_id: UUID
    url: str
    status: Literal["fetched", "failed"]
    
    # If fetched:
    title: str | None
    snippet: str  # First 500 chars
    content_type: str
    size_bytes: int
    content_cache_key: str  # Redis key for full content
    metadata: dict[str, Any]
    thumbnail_data: bytes | None
    
    # If failed:
    error: str | None
    
    fetch_timestamp: datetime
```

---

## 🔧 Implementation

### Celery Task: fetch_for_review

```python
@shared_task
def fetch_for_review(
    session_id: str,
    urls: list[str],
    tenant_id: str,
    case_id: str | None = None,
) -> dict:
    """
    Fetch URLs and create previews for manual review.
    Does NOT persist to repository.
    """
    session = ReviewSession.objects.get(pk=session_id)
    session.status = "fetching"
    session.save()
    
    previews = []
    for url in urls:
        try:
            # Fetch content
            fetcher = HttpFetcher()
            content = fetcher.fetch(url, timeout=10)
            
            # Parse for preview
            parser = get_parser_for_content_type(content.content_type)
            parsed = parser.parse(content.body)
            
            # Extract snippet (first 500 chars)
            snippet = parsed.primary_text[:500] if parsed.primary_text else ""
            
            # Cache full content in Redis (1 hour TTL)
            cache_key = f"fetch_preview:{session_id}:{url}"
            redis_client.setex(
                cache_key,
                3600,  # 1 hour
                content.body
            )
            
            # Create preview
            preview = FetchedPreview(
                preview_id=uuid4(),
                session_id=session_id,
                url=url,
                status="fetched",
                title=parsed.title or _extract_title_from_url(url),
                snippet=snippet,
                content_type=content.content_type,
                size_bytes=len(content.body),
                content_cache_key=cache_key,
                metadata={
                    "language": parsed.content_language,
                    # ... other metadata
                },
                fetch_timestamp=datetime.now(timezone.utc),
            )
            previews.append(preview)
            
        except Exception as e:
            # Failed fetch
            preview = FetchedPreview(
                preview_id=uuid4(),
                session_id=session_id,
                url=url,
                status="failed",
                error=str(e),
                fetch_timestamp=datetime.now(timezone.utc),
            )
            previews.append(preview)
    
    # Save previews to DB
    FetchedPreview.objects.bulk_create(previews)
    
    # Update session
    session.status = "completed"
    session.save()
    
    return {
        "session_id": str(session_id),
        "previews_created": len(previews),
        "failed": sum(1 for p in previews if p.status == "failed"),
    }
```

### Celery Task: ingest_selected

```python
@shared_task
def ingest_selected(
    session_id: str,
    selected_urls: list[str],
    mode: Literal["rag", "archive", "ephemeral"],
    tenant_id: str,
    **kwargs
) -> dict:
    """
    Ingest selected URLs from review session.
    Uses cached content from fetch_for_review.
    """
    session = ReviewSession.objects.get(pk=session_id)
    previews = FetchedPreview.objects.filter(
        session_id=session_id,
        url__in=selected_urls,
        status="fetched"
    )
    
    results = {
        "completed": [],
        "failed": {},
        "documents_created": [],
    }
    
    for preview in previews:
        try:
            # Retrieve cached content
            content_body = redis_client.get(preview.content_cache_key)
            if content_body is None:
                raise ValueError("Cached content expired")
            
            # Process based on mode
            doc_id = _ingest_from_preview(
                preview=preview,
                content_body=content_body,
                mode=mode,
                tenant_id=tenant_id,
                **kwargs
            )
            
            results["completed"].append(preview.url)
            if doc_id:
                results["documents_created"].append(doc_id)
                
        except Exception as e:
            results["failed"][preview.url] = str(e)
    
    # Clean up cache
    for preview in previews:
        redis_client.delete(preview.content_cache_key)
    
    return results
```

---

## 🎨 UI/UX Flow

### 1. Fetch & Preview Screen

```
┌─────────────────────────────────────────────────┐
│ Review Fetched Content (2 of 2 fetched)        │
├─────────────────────────────────────────────────┤
│                                                 │
│ ☑ Neural Networks Explained                    │
│   https://example.com                           │
│   A neural network is a method in artificial... │
│   HTML • 45 KB • Published: 2024-01-15         │
│   [Preview Full] [Remove]                       │
│                                                 │
│ ☐ Deep Learning Tutorial (FAILED)              │
│   https://example.org                           │
│   ❌ Connection timeout                         │
│   [Retry] [Remove]                              │
│                                                 │
├─────────────────────────────────────────────────┤
│ [Select All] [Deselect All]                    │
│                                                 │
│ Mode: ● RAG  ○ Archive  ○ Ephemeral           │
│ Collection: [Neural Networks Research ▼]       │
│                                                 │
│ [Cancel]              [Ingest Selected (1)] →  │
└─────────────────────────────────────────────────┘
```

### 2. Preview Full Content (Modal)

```
┌───────────────────────────────────────────────────┐
│ Preview: Neural Networks Explained             × │
├───────────────────────────────────────────────────┤
│ URL: https://example.com                          │
│ Type: text/html • 45 KB                          │
│                                                   │
│ ┌───────────────────────────────────────────────┐ │
│ │ # Neural Networks Explained                    │ │
│ │                                                │ │
│ │ A neural network is a method in artificial     │ │
│ │ intelligence that teaches computers to process │ │
│ │ data in a way that is inspired by the human    │ │
│ │ brain...                                       │ │
│ │                                                │ │
│ │ [... full parsed content ...]                  │ │
│ └───────────────────────────────────────────────┘ │
│                                                   │
│ [Close]                  [Include in Ingestion]  │
└───────────────────────────────────────────────────┘
```

---

## ⏱️ Session Management

### TTL Strategy

```python
# Review sessions expire after 1 hour
REVIEW_SESSION_TTL = 3600  # seconds

# Cached content expires after 1 hour
CONTENT_CACHE_TTL = 3600  # seconds

# Cleanup job runs every 15 minutes
@periodic_task(run_every=timedelta(minutes=15))
def cleanup_expired_sessions():
    expired = ReviewSession.objects.filter(
        status__in=["completed", "failed"],
        created_at__lt=datetime.now() - timedelta(hours=1)
    )
    
    for session in expired:
        # Delete associated previews
        FetchedPreview.objects.filter(session_id=session.session_id).delete()
        
        # Delete session
        session.delete()
```

---

## 🔐 Security Considerations

### 1. Session Ownership

```python
# Verify user owns session before viewing
def get_review_session(session_id: UUID, user: User):
    session = ReviewSession.objects.get(pk=session_id)
    if session.tenant_id != user.tenant_id:
        raise PermissionDenied()
    return session
```

### 2. Content Sanitization

```python
# Sanitize HTML previews for XSS
from bleach import clean

def sanitize_snippet(html: str) -> str:
    return clean(
        html,
        tags=["p", "br", "strong", "em"],
        attributes={},
        strip=True
    )
```

### 3. Rate Limiting

```python
# Limit review sessions per user
@rate_limit(key="user", rate="10/hour")
def create_review_session(request):
    ...
```

---

## 📊 Database Schema

```sql
-- Review Sessions
CREATE TABLE crawler_review_session (
    session_id UUID PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    case_id VARCHAR(255),
    status VARCHAR(20) NOT NULL,  -- fetching, completed, failed, expired
    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    metadata JSONB,
    
    INDEX idx_tenant_created (tenant_id, created_at),
    INDEX idx_expires (expires_at)
);

-- Fetched Previews
CREATE TABLE crawler_fetched_preview (
    preview_id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES crawler_review_session(session_id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    status VARCHAR(20) NOT NULL,  -- fetched, failed
    
    -- If fetched:
    title TEXT,
    snippet TEXT,
    content_type VARCHAR(100),
    size_bytes INTEGER,
    content_cache_key VARCHAR(255),
    metadata JSONB,
    thumbnail_data BYTEA,
    
    -- If failed:
    error TEXT,
    
    fetch_timestamp TIMESTAMP NOT NULL,
    
    INDEX idx_session (session_id),
    INDEX idx_url (url)
);
```

---

## 🎯 Benefits of HITL

1. **Quality Control** ✅
   - Review content before ingestion
   - Filter out irrelevant results
   - Catch fetch errors early

2. **Cost Optimization** ✅
   - Skip embedding for rejected content
   - No wasted vector storage
   - User selects only relevant docs

3. **Metadata Enrichment** ✅
   - User can add tags during review
   - Correct auto-detected metadata
   - Improve search quality

4. **Transparency** ✅
   - User sees what's being ingested
   - Trust in the system
   - Better UX

---

## 🔄 Updated Flow Diagram

```
User Input: URLs
    ↓
[API] Create Review Session
    ↓
[Celery] fetch_for_review
    ├─ Fetch URL 1 → Parse → Cache → Preview ✅
    ├─ Fetch URL 2 → Parse → Cache → Preview ✅
    └─ Fetch URL 3 → Failed ❌
    ↓
[DB] Store Previews (temp, 1h TTL)
    ↓
[UI] Display Previews
    ↓
USER REVIEWS ← 👤 HUMAN IN THE LOOP
    ├─ Select URL 1 ✅
    ├─ Reject URL 2 ❌
    └─ Skip URL 3 (failed) ❌
    ↓
[API] Ingest Selected
    ↓
[Celery] ingest_selected (from cache)
    ├─ URL 1: Mode=RAG → Parse → Persist → Embed ✅
    └─ Clean up cache
    ↓
[UI] Show Results
```

---

## 💡 Future Enhancements

### Batch Actions

- "Select All"
- "Deselect All"
- "Select by criteria" (e.g., all HTML, all > 10KB)

### Metadata Editing

- Edit title before ingestion
- Add/remove tags
- Set custom metadata

### Preview Improvements

- Syntax highlighting for code
- Thumbnail generation for PDFs
- Table of contents for long documents

### Collaboration

- Share review session with team
- Collaborative selection
- Comments/annotations

---

## 🤔 Open Questions

1. **Cache Backend**: Redis or Database?
   - Recommendation: Redis (faster, built-in TTL)

2. **Thumbnail Generation**: Client-side or server-side?
   - Recommendation: Server-side (consistent quality)

3. **Diff Support**: Show changes if URL was previously ingested?
   - Recommendation: Yes, but Phase 2

4. **Bulk Operations**: Max URLs per session?
   - Recommendation: 100 URLs, auto-split larger batches

---

**Ready to implement?** 🚀

This HITL design adds ~1 day to timeline (now 6 days total) but provides massive UX improvement!
