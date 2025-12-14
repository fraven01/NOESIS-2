# NOESIS-2 Architecture - 4-Layer Firm Hierarchy

## 🏢 The Firm Analogy

Think of NOESIS-2 as a company with 4 hierarchical layers:

```
┌────────────────────────────────────────────────────┐
│  LAYER 1: CUSTOMER INTERFACE (Frontend)           │
│  What the customer sees and interacts with        │
│  - UI/UX                                           │
│  - Direct user input                               │
└────────────────┬───────────────────────────────────┘
                 │
┌────────────────▼───────────────────────────────────┐
│  LAYER 2: BUSINESS MANAGERS                        │
│  Business processes, workflows, cases              │
│  - Workflows (TODO: to be implemented)            │
│  - Cases (TODO: to be implemented)                │
│  - Business logic orchestration                    │
│  - Example: /rag-tools/#framework                 │
└────────────────┬───────────────────────────────────┘
                 │
┌────────────────▼───────────────────────────────────┐
│  LAYER 3: TECHNICAL MANAGERS                       │
│  Know which workers to use for technical tasks    │
│  - LangGraph orchestrations (ai_core/graphs)      │
│  - Services (DocumentDomainService, etc.)         │
│  - Coordinators (crawler_coordinator)             │
│  Know: "I need to fetch a doc, query LLM, search" │
└────────────────┬───────────────────────────────────┘
                 │
┌────────────────▼───────────────────────────────────┐
│  LAYER 4: WORKERS (The actual work)                │
│  - Celery tasks                                    │
│  - HTTP Fetcher                                    │
│  - Parsers                                         │
│  - Repository                                      │
│  - LLM API calls                                   │
└────────────────────────────────────────────────────┘
```

**Key Insight**: HITL must flow through ALL 4 layers!

---

## 📊 Corrected Layer Mapping

### **Layer 1: Customer Interface (Frontend)**

**What**: The UI the customer interacts with

**Examples**:

- `/rag-tools/` - RAG Developer Workbench
- `/rag-tools/#search` - Web Search & Ingest
- `/rag-tools/#framework` - Framework Analysis (Business layer example!)
- Document Explorer UI

**Responsibilities**:

- User input collection
- Display results
- Trigger workflows
- **HITL**: Show previews, get user approval

**Technology**: Django templates, HTMX, JavaScript

---

### **Layer 2: Business Managers (Emerging)**

**What**: Business process orchestration using workflows and cases

**Current State**:

- ⚠️ Workflow Engine and Case Management not yet implemented
- ✅ **`FrameworkAnalysisGraph`** - First true Business Layer graph (analyzes Rahmen-BV structure)
- ✅ `case_id` and `workflow_id` context fields prepared

**Layer 2 vs Layer 3 Distinction**:

| Aspect | L2 (Business) | L3 (Technical) |
|--------|---------------|----------------|
| **Purpose** | *What* should happen for the business | *How* to technically do it |
| **Example** | FrameworkAnalysisGraph: "Analyze this BV structure" | ExternalKnowledgeGraph: "Fetch & store data" |
| **Domain Knowledge** | Business rules, compliance, policies | APIs, storage, LLM orchestration |
| **Tenant Variance** | Different workflows per tenant | Same technical implementation |

**Examples**:

- ✅ `FrameworkAnalysisGraph` - Analyzes Rahmen-Betriebsvereinbarungen (business logic)
- 🔜 Workflow Engine (future)
- 🔜 Case Management Dashboard (future)

```
Business Process: "Legal Document Review Workflow"
    ↓
Step 1: Ingest documents (calls Technical Manager)
Step 2: Analyze with LLM (calls Technical Manager)
Step 3: HITL Review (wait for human approval)
Step 4: Generate report (calls Technical Manager)
Step 5: Archive (calls Technical Manager)
```

**Responsibilities**:

- Define business workflows
- Track cases (business transactions)
- Enforce business rules
- **HITL**: Coordinate human approvals in workflows
- SLA tracking, audit trails

**Technology** (planned):

- Workflow engine (maybe Temporal.io or custom)
- Case management system
- Business rule engine

---

### **Layer 3: Technical Managers**

**What**: Know which workers to use for technical tasks

**Current State**: ✅ **Implemented!**

**Components**:

#### **A) LangGraph Orchestrations** (`ai_core/graphs/`)

```python
class CrawlerIngestionGraph:
    """
    Technical Manager for crawler ingestion.
    Knows: "I need to fetch, parse, chunk, embed"
    """
    def run(self, state):
        # Orchestrates technical steps
        fetch_result = self._fetch_worker(...)
        parsed = self._parser(...)
        chunks = self._chunker(...)
        embed_task = self._embedding_handler(...)
```

**Knows**:

- Which workers to call (fetcher, parser, chunker)
- Sequence of operations
- Error handling
- Guardrails

**Does NOT know**:

- "This is part of Case #12345 Legal Review"
- "This is Step 3 of Document Approval Workflow"
- Business SLAs or rules

#### **B) Domain Services**

```python
class DocumentDomainService:
    """
    Technical Manager for document operations.
    Knows: "I need to persist, lifecycle, collections"
    """
    def ingest_document(self, ...):
        # Orchestrates repository, lifecycle, dispatcher
        document = self._repository.create(...)
        self._lifecycle.transition(...)
        self._dispatcher.queue_embedding(...)
```

#### **C) Coordinators** (`ai_core/services/`)

```python
class CrawlerCoordinator:
    """
    Technical Manager for crawler operations.
    Knows: "I need to queue fetch tasks, trigger graphs"
    """
    def fetch_for_review(self, urls):
        task = fetch_urls_task.delay(...)
        return ReviewSession(task_id=task.id)
```

**Responsibilities**:

- Technical orchestration
- Call workers in correct order
- Handle technical errors
- **HITL**: Provide data for human review (e.g., fetch preview)
- Retry logic, timeouts

**Technology**: LangGraph, Python services, Celery coordination

---

### **Layer 4: Workers (The Actual Work)**

**What**: The actual workers doing the tasks

**Components**:

#### **A) Celery Tasks**

```python
@shared_task
def fetch_url_task(url):
    """Worker: Fetch a single URL."""
    fetcher = HttpFetcher()
    return fetcher.fetch(url)

@shared_task
def ingest_document_task(doc_data):
    """Worker: Persist a document."""
    repo = get_repository()
    return repo.upsert(doc_data)
```

#### **B) Pure Workers**

```python
class HttpFetcher:
    """Worker: HTTP fetching."""
    def fetch(self, url):
        response = requests.get(url)
        return response.content

class HtmlParser:
    """Worker: HTML parsing."""
    def parse(self, html):
        return BeautifulSoup(html).get_text()
```

#### **C) Infrastructure**

- Database (Repository)
- Vector Store (pgvector)
- Object Storage (S3/GCS)
- LLM APIs (OpenAI, etc.)

**Responsibilities**:

- Execute specific tasks
- No orchestration
- Pure functions when possible
- **HITL**: Cache data for review (e.g., Redis cache)

**Technology**: Celery, requests, BeautifulSoup, psycopg2, etc.

---

## 🔄 Crawler Flow - Corrected with All 4 Layers

### **HITL Scenario: Web Search & Ingest**

```
┌─────────────────────────────────────────────────────┐
│ LAYER 1: Frontend (Customer Interface)             │
│ /rag-tools/#search                                  │
│ User enters URLs → clicks "Fetch for Review"       │
└────────────┬────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────┐
│ LAYER 2: Business Manager (TODO - Future)          │
│ WorkflowEngine.start("Web Research Workflow")      │
│ - Case ID: 12345                                    │
│ - Workflow ID: "web_research_v1"                   │
│ - Step 1: "Fetch URLs for review"                  │
│   → Calls Technical Manager                         │
└────────────┬────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────┐
│ LAYER 3: Technical Manager                         │
│ CrawlerCoordinator.fetch_for_review()              │
│ - Knows: "I need to queue fetch tasks"             │
│ - Calls: fetch_urls_task.delay()                   │
└────────────┬────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────┐
│ LAYER 4: Workers                                    │
│ Celery Task: fetch_urls_task                       │
│ - FetchWorker.fetch_url() → HTTP request           │
│ - HtmlParser.parse() → Extract text                │
│ - Redis.cache() → Store for preview                │
└────────────┬────────────────────────────────────────┘
             │
             ↓ (Results cached)
             
👤 HUMAN IN THE LOOP ← Review happens here!
             
User reviews previews in UI
Selects URLs to ingest
Clicks "Ingest Selected"
             
             │
┌────────────▼────────────────────────────────────────┐
│ LAYER 1: Frontend                                   │
│ User selects URLs → clicks "Ingest"                │
└────────────┬────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────┐
│ LAYER 2: Business Manager (TODO)                   │
│ WorkflowEngine.continue(case_id=12345)             │
│ - Step 2: "Ingest selected documents"              │
│   → Calls Technical Manager                         │
└────────────┬────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────┐
│ LAYER 3: Technical Manager                         │
│ DocumentDomainService.ingest_document()            │
│ - Knows: "I need to persist, trigger embedding"    │
│ - Calls: repository, dispatcher                    │
│                                                     │
│ OR CrawlerIngestionGraph.run()                     │
│ - Knows: "I need to parse, chunk, embed"           │
│ - Calls: parser, chunker, embedder                 │
└────────────┬────────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────────┐
│ LAYER 4: Workers                                    │
│ - Repository.upsert() → Database                   │
│ - Parser.parse() → Extract text                    │
│ - Chunker.chunk() → Split text                     │
│ - Embedder.embed() → LLM API call                  │
└─────────────────────────────────────────────────────┘
```

**Key**: HITL flows through ALL layers:

1. Frontend shows preview
2. Business Manager waits for approval (workflow pause)
3. Technical Manager provides data for review
4. Workers cache content for preview

---

## 🎯 Corrected Crawler Architecture

### **Components per Layer:**

| Layer | Component | Location | Responsibility |
|-------|-----------|----------|----------------|
| **1. Frontend** | Search & Ingest UI | `theme/templates/theme/rag_tools.html` | User interaction |
| | Document Explorer | `theme/templates/theme/partials/tool_documents.html` | Display results |
| **2. Business** | Workflow Engine | ❌ TODO | Process orchestration |
| | Case Manager | ❌ TODO | Track business transactions |
| **3. Technical** | CrawlerCoordinator | `ai_core/services/crawler_coordinator.py` (NEW) | Queue tasks |
| | DocumentDomainService | `documents/domain_service.py` | Persist documents |
| | CrawlerIngestionGraph | `ai_core/graphs/crawler_ingestion_graph.py` | Doc processing |
| **4. Workers** | FetchWorker | `crawler/fetch_worker.py` (NEW) | HTTP fetching |
| | Celery Tasks | `crawler/tasks.py` (NEW) | Async execution |
| | Parsers | `documents/parsers.py` | Content parsing |
| | Repository | `ai_core/adapters/db_documents_repository.py` | Data persistence |

---

## 📋 Implementation Priority

### **Phase 1: Fix Layer 3 & 4 (Current - 2 weeks)**

- ✅ Technical Manager (Coordinator)
- ✅ Workers (Fetch, Ingest tasks)
- ✅ Eliminate parallel registration
- ✅ HITL infrastructure (preview cache)

### **Phase 2: Enhance Layer 1 (UI - 1 week)**

- ✅ Preview UI component
- ✅ Selection interface
- ✅ Progress indicators

### **Phase 3: Build Layer 2 (Business - Future - 4+ weeks)**

- ❌ Workflow engine
- ❌ Case management
- ❌ Business rule engine
- ❌ Audit trails

---

## ✅ Key Insights

1. **Graphs are Technical Managers**, not Business Managers
2. **Business Layer is mostly TODO** (workflows, cases)
3. **HITL must span all 4 layers**
4. **Current focus**: Fix Layers 3 & 4 (Technical + Workers)
5. **Future**: Build Layer 2 (Business processes)

---

## 🚀 Recommendation

**For Crawler Redesign**:

1. ✅ Build Layer 3 (Technical Manager) - `CrawlerCoordinator`
2. ✅ Build Layer 4 (Workers) - Fetch/Ingest tasks
3. ✅ Add HITL hooks (preview cache, approval gates)
4. ⏳ Prepare for Layer 2 (use `case_id`, `workflow_id` now)

**This aligns perfectly with the 4-layer firm hierarchy!** 🎯
