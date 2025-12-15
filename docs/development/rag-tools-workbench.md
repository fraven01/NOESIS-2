# RAG Developer Workbench

**URL**: `/rag-tools/`
**Template**: [`theme/templates/theme/rag_tools.html`](../../theme/templates/theme/rag_tools.html)
**Access**: Internal development and debugging tool

## Overview

The RAG Developer Workbench is an HTMX-based UI for testing and debugging RAG components without requiring external API clients. It provides direct access to:

1. **Web Search** (External Knowledge + Collection Search)
2. **Crawler** (Manual URL submission)
3. **Ingestion** (Document processing trigger)

## Access & Context

The workbench automatically injects tenant context from the current session:

| Header | Source | Description |
|--------|--------|-------------|
| `X-Tenant-ID` | Session | Current tenant UUID |
| `X-Tenant-Schema` | Session | Tenant schema name |
| `X-Case-ID` | Session | Active case ID (if any) |

These headers are automatically attached to all HTMX requests via `hx-headers` attribute.

## Features

### 1. Web Search Tab

**Endpoint**: `POST /web-search/` (inferred from `hx-post` attribute)

#### Search Types

**External Knowledge**:
- Uses [`ExternalKnowledgeGraph`](../agents/overview.md#2-external-knowledge-graph)
- Searches public web sources
- Optional auto-ingestion of selected results

**Collection Search**:
- Searches within existing RAG collections
- Requires `purpose` field (e.g., "Research", "Fact Check")
- Quality modes:
  - `standard`: Default hybrid search
  - `software_docs_strict`: Optimized for technical documentation
  - `law_evergreen`: Optimized for legal/regulatory content
- Auto-ingest toggle

#### Form Fields

| Field | Required | Description |
|-------|----------|-------------|
| `query` | Yes | Search query text |
| `purpose` | Conditional | Required for Collection Search, describes search intent |
| `mode` | No | `live` (default) or `archive` |
| `workflow_id` | No | Workflow identifier (default: "web-search") |
| `collection_id` | No | Target collection (default: "default") |
| `quality_mode` | No | Only for Collection Search |
| `auto_ingest` | No | Checkbox: auto-trigger ingestion of selected result |

#### Results Panel

Displays search results in `#web-search-results` div with HTMX swap. Results are formatted by the backend view and can include:
- Result title, URL, snippet
- Relevance score
- "Ingest Selected" action button (target: `#ingestion-status-panel`)

---

### 2. Crawler Tab

**Endpoint**: `POST /crawler-submit/` (inferred)

Manually triggers crawler for one or more URLs.

#### Form Fields

| Field | Required | Description |
|-------|----------|-------------|
| `origin_url` | No | Single URL to crawl |
| `origin_urls` | No | Multiple URLs (one per line, textarea) |
| `mode` | No | `live` (default) or `archive` |
| `workflow_id` | No | Workflow identifier (default: "crawler-manual") |
| `case_id` | Hidden | Auto-populated from session context |
| `fetch` | No | Checkbox: actually fetch content |
| `dry_run` | No | Checkbox: simulate crawl without persisting |
| `shadow_mode` | No | Checkbox: crawl but don't trigger ingestion |

**Note**: Either `origin_url` OR `origin_urls` should be provided, not both (backend validation).

#### Status Panel

Results appear in `#crawler-status-area` div, showing:
- Queued URLs
- Crawl status (fetching, parsing, ingested)
- Error messages

---

### 3. Ingestion Tab

**Endpoint**: `POST /ingestion-submit/` (inferred)

Manually triggers the [`UploadIngestionGraph`](../rag/ingestion.md#upload-ingestion-graph) for existing documents.

#### Form Fields

| Field | Required | Description |
|-------|----------|-------------|
| `document_ids` | Yes | JSON array or comma-separated list of document UUIDs |
| `embedding_profile` | No | Embedding profile name (default: system default profile) |
| `case_id` | Hidden | Auto-populated from session context |

#### Input Format Examples

**JSON Array**:
```json
["550e8400-e29b-41d4-a716-446655440000", "650e8400-e29b-41d4-a716-446655440001"]
```

**Comma-Separated**:
```
550e8400-e29b-41d4-a716-446655440000, 650e8400-e29b-41d4-a716-446655440001
```

#### Response Panel

Results appear in `#ingestion-response` div, showing:
- Ingestion status per document
- Phase progress (validate, parse, embed, upsert)
- Decision outcomes (completed, skip_guardrail, skip_duplicate, error)
- Telemetry (delta decisions, guardrail results)

---

## Tab Navigation

Tabs use JavaScript-based visibility toggling:

```javascript
// Tab switching logic (DOMContentLoaded)
tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    const target = document.querySelector(tab.dataset.tabTarget);
    // Hide all other tab contents
    contents.forEach(c => c.classList.add('hidden'));
    // Show selected tab
    target.classList.remove('hidden');
    // Update tab styles (border-indigo-500 = active)
  });
});
```

**Tab Targets**:
- `#tab-search` → Web Search
- `#tab-crawler` → Crawler
- `#tab-ingestion` → Ingestion

---

## Search Type Toggle

When switching between "External Knowledge" and "Collection Search":

```javascript
function toggleSearchFields() {
  const isCollection = document.getElementById('type-collection').checked;
  if (isCollection) {
    // Show collection-specific options
    optionsPanel.classList.remove('hidden');
    // Make purpose field required
    purposeInput.required = true;
  } else {
    // Hide collection options
    optionsPanel.classList.add('hidden');
    purposeInput.required = false;
  }
}
```

**Collection-Only Fields**:
- `quality_mode` (dropdown)
- `auto_ingest` (checkbox)
- `purpose` (required text input)

---

## HTMX Integration

All forms use `hx-post` for AJAX submission with `hx-target` and `hx-swap`:

**Example**:
```html
<form hx-post="{% url 'web-search' %}"
      hx-target="#web-search-results"
      hx-swap="innerHTML">
  ...
</form>
```

**Swap Behavior**: `innerHTML` replaces content inside target div without full page reload.

---

## Backend Views (Inferred)

The following view endpoints are referenced but not defined in the template:

| View Name | URL Pattern | Method | Responsibility |
|-----------|-------------|--------|----------------|
| `web-search` | `/web-search/` | POST | Execute web search, return HTML results |
| `crawler-submit` | `/crawler-submit/` | POST | Queue crawler task, return status HTML |
| `ingestion-submit` | `/ingestion-submit/` | POST | Trigger ingestion graph, return status HTML |

**Note**: These views must:
- Accept `X-Tenant-ID`, `X-Tenant-Schema`, `X-Case-ID` headers
- Return HTMX-compatible HTML fragments (not full pages)
- Handle CSRF tokens (forms include `{% csrf_token %}`)

---

## Styling

Uses Tailwind CSS with Indigo color scheme:

- Active tab: `border-indigo-500`, `text-indigo-600`
- Inactive tab: `border-transparent`, `text-slate-500`
- Primary button: `bg-indigo-600`, `hover:bg-indigo-700`
- Form inputs: `border-slate-300`, `focus:border-indigo-500`

---

## Usage Examples

### Testing External Knowledge Search

1. Navigate to `/rag-tools/`
2. Select "External Knowledge" radio button
3. Enter query: "What are the latest RAG best practices?"
4. Set `workflow_id`: "test-search"
5. Click "Search"
6. Review results in right panel
7. (Optional) Click "Ingest Selected" to trigger crawler

### Debugging Upload Ingestion

1. Upload document via `/ai/rag/documents/upload/`
2. Copy returned `document_id`
3. Switch to "Ingestion" tab
4. Paste `document_id` into textarea
5. Select embedding profile (or use default)
6. Click "Run Ingestion"
7. Monitor transitions in response panel

### Crawler Dry-Run

1. Switch to "Crawler" tab
2. Enter URL: `https://example.com/docs`
3. Check "Dry Run" checkbox
4. Check "Fetch Content" checkbox
5. Click "Start Crawl"
6. Verify parsed content without DB persistence

---

## Security Considerations

**⚠️ WARNING**: This is a **development tool only**. It should NOT be exposed in production environments.

**Required Safeguards**:
- Behind authentication (e.g., Django staff user check)
- Rate-limited endpoints to prevent abuse
- Debug mode check (`DEBUG=True` or feature flag)
- CSRF protection enabled (already present via `{% csrf_token %}`)

**Recommended Deployment**:
```python
# In URLs config
if settings.DEBUG:
    urlpatterns += [
        path('rag-tools/', views.rag_tools_view, name='rag-tools'),
    ]
```

---

## Future Enhancements

- [ ] Real-time progress updates via WebSockets or Server-Sent Events
- [ ] Download ingestion results as JSON
- [ ] Search history panel
- [ ] Bulk document upload + ingestion in one flow
- [ ] Visual graph execution timeline (LangGraph state transitions)
- [ ] Embedding vector visualization (UMAP/t-SNE)

---

## Related Documentation

- [Upload Ingestion Graph](../rag/ingestion.md#upload-ingestion-graph)
- [External Knowledge Graph](../agents/overview.md#2-external-knowledge-graph)
- [Retrieval Contracts](../rag/retrieval-contracts.md)
- [Web Search Tool](../agents/web-search-tool.md)
