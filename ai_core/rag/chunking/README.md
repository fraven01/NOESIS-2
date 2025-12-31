# RAG Chunking Strategies

State-of-the-art chunking implementation for NOESIS 2 RAG system, combining **Late Chunking** and **Agentic Chunking** strategies.

## Architecture Overview

### Chunker Hierarchy

```
DocumentChunker (Protocol)
    ↓
HybridChunker (Main Implementation)
    ├─> LateChunker (Default)
    └─> AgenticChunker (Configurable)
```

### Chunking Modes

#### 1. Late Chunking (Default)

**Strategy**: Embed full document first, then chunk based on contextual embeddings.

**Two-Phase Implementation**:

##### Phase 1: Jaccard Similarity (Stability Compromise)

**Algorithm**:
1. Validate document size against `max_tokens` threshold
2. Split text into sentences (regex-based)
3. Detect chunk boundaries using **Jaccard similarity** (text-based)
4. Build chunks with overlap and metadata

**Jaccard Boundary Detection**:
- Compare word overlap between adjacent sentences
- Low Jaccard similarity (<threshold) = good boundary
- **Limitation**: NOT semantically aware
  - Misses paraphrases ("mandatory arbitration" vs "binding dispute resolution")
  - Misses synonyms ("large" vs "big")
  - Misses topic transitions with similar vocabulary

**Why Phase 1?**: Fallback strategy for:
- Embedding service failures (graceful degradation)
- Very large documents exceeding `max_tokens`
- Fast testing/development (no API calls)

##### Phase 2: SOTA Embedding-based Similarity (Default for Production)

**Algorithm**:
1. Validate document size against `max_tokens` threshold
2. Split text into sentences
3. **Create sliding windows** of sentences (e.g., window_size=3)
   - Example: ["A", "B", "C", "D", "E"] → ["A B C", "B C D", "C D E"]
4. **Batch embed** all windows for efficiency (batch_size=16 parallel)
5. **Compute cosine similarity** between adjacent windows
6. Detect boundaries where similarity drops below threshold
7. Build chunks with overlap and metadata

**Embedding-based Boundary Detection**:
- Uses **sliding windows** to capture local context
- Embeds overlapping windows (e.g., 3 sentences each)
- Cosine similarity between windows measures semantic continuity
- Low similarity (<threshold) = semantic topic shift = good boundary
- **Semantically aware**: Captures paraphrases, synonyms, topic transitions

**Benefits over Jaccard**:
- Detects semantic boundaries (topic transitions)
- Handles paraphrases and synonyms correctly
- Better chunk quality (self-contained semantic units)
- Improved retrieval accuracy (contextual embeddings)

**Trade-offs**:
- Higher cost (API calls for embedding)
- Slower (network latency)
- Requires embedding service availability

**Graceful Fallback**:
- On embedding failure → automatically falls back to Jaccard
- No crash, continued operation
- Logged warning for monitoring

**Model**: `embedding` label from MODEL_ROUTING.yaml
- Dev/Test: `oai-embed-small` (1536D, cheap)
- MVP Prod: `oai-embed-large` (3072D, higher quality)

**Phase 2 Configuration**:
- `use_embedding_similarity=True` - Enable Phase 2 (default: False)
- `window_size=3` - Sentences per window (default: 3)
- `batch_size=16` - Windows to embed in parallel (default: 16)
- `use_content_based_ids=True` - SHA256-based chunk IDs (default: True)

#### 2. Agentic Chunking (Configurable)

**Strategy**: Use LLM to detect semantic boundaries in complex documents.

> **⚠️ MVP Status (2025-12-30)**: AgenticChunker infrastructure is complete (rate limiting, fallback, cost tracking), but **LLM boundary detection is not yet activated**. Currently raises `NotImplementedError` and immediately falls back to LateChunker. Phase 2.5 will activate the LLM integration with structured prompts.

**Algorithm** (when LLM activated):
1. Send document to LLM with boundary detection prompt
2. Parse structured JSON response (boundary positions + reasons)
3. Validate boundaries (no overlaps, size checks)
4. Build chunks with metadata
5. Fallback to Late Chunking on failure

**Benefits** (when activated):
- Highest quality for complex documents (legal, technical specs)
- Natural semantic boundaries
- Self-contained chunks

**Model**: `agentic-chunk` label from MODEL_ROUTING.yaml
- Dev/Test: `gemini-3-flash-preview` (Pareto-optimal, latest Gemini 3)
- Production: `gemini-2.0-flash-lite-001` (stable fallback, commented in config)

**Cost Control**:
- Rate limiting: 10 calls/minute
- Token budget: 100k tokens/day
- Timeout: 30 seconds
- Automatic fallback to LateChunker on any failure

### Mode Selection (Routing Rules)

Chunker mode is selected via **RAG Routing Rules** (`config/rag_routing_rules.yaml`):

```yaml
default_chunker_mode: late

rules:
  # Late chunking by default
  - tenant: enterprise
    profile: premium
    chunker_mode: late

  # Agentic chunking for legal contracts
  - tenant: enterprise
    doc_class: legal_contract
    chunker_mode: agentic
```

**Resolution Priority**:
1. `tenant` + `doc_class` + `collection_id`
2. `tenant` + `doc_class`
3. `tenant` + `collection_id`
4. `tenant`
5. `default_chunker_mode`

## Classes

### HybridChunker

Main chunker implementing `DocumentChunker` protocol.

```python
from ai_core.rag.chunking import HybridChunker, ChunkerConfig

config = ChunkerConfig(
    mode=ChunkerMode.LATE,
    late_chunk_model="embedding",  # MODEL_ROUTING.yaml label
    agentic_chunk_model="agentic-chunk",  # MODEL_ROUTING.yaml label
    enable_quality_metrics=True,
    max_chunk_tokens=450,
    overlap_tokens=80,
)

chunker = HybridChunker(config)

chunks, stats = chunker.chunk(
    document=normalized_doc,
    parsed=parsed_result,
    context=processing_context,
    config=pipeline_config,
)
```

**Returns**:
- `chunks`: List of dicts with `chunk_id`, `text`, `parent_ref`, `metadata`
- `stats`: Dict with `chunk.count`, `chunker.mode`, `quality.scores`

### LateChunker

Late Chunking implementation with Phase 1/Phase 2 modes.

```python
from ai_core.rag.chunking import LateChunker

# Phase 1: Jaccard similarity (fast, fallback)
late_chunker_phase1 = LateChunker(
    model="embedding",  # MODEL_ROUTING.yaml label
    max_tokens=8000,
    target_tokens=450,
    overlap_tokens=80,
    use_embedding_similarity=False,  # Phase 1 (default)
)

# Phase 2: SOTA embedding-based similarity
late_chunker_phase2 = LateChunker(
    model="embedding",  # MODEL_ROUTING.yaml label
    max_tokens=8000,
    target_tokens=450,
    overlap_tokens=80,
    use_embedding_similarity=True,  # Phase 2 (SOTA)
    window_size=3,  # Sentences per window
    batch_size=16,  # Parallel embedding
    use_content_based_ids=True,  # SHA256 IDs
)

chunks = late_chunker_phase2.chunk(parsed, context)
```

### AgenticChunker

Agentic Chunking implementation with LLM boundary detection.

```python
from ai_core.rag.chunking import AgenticChunker

agentic_chunker = AgenticChunker(
    model="agentic-chunk",  # MODEL_ROUTING.yaml label
    max_retries=2,
    timeout=30,
    rate_limit=10,  # per minute
    token_budget=100000,  # per day
    fallback_chunker=late_chunker,
)

chunks = agentic_chunker.chunk(parsed, context)
```

## Integration

### Universal Ingestion Graph

Replace `SimpleDocumentChunker` in `ai_core/graphs/technical/universal_ingestion_graph.py`:

```python
# OLD
from documents.cli import SimpleDocumentChunker
chunker = SimpleDocumentChunker()

# NEW
from ai_core.rag.chunking import HybridChunker, get_default_chunker_config
chunker = HybridChunker(get_default_chunker_config())
```

### Routing Rules

Extend `config/rag_routing_rules.yaml`:

```yaml
default_profile: standard
default_chunker_mode: late  # NEW

rules:
  - tenant: enterprise
    doc_class: legal_contract
    profile: premium
    chunker_mode: agentic  # NEW
```

## Configuration

Settings in `noesis2/settings/base.py` (uses MODEL_ROUTING.yaml labels):

```python
# Chunker Mode
RAG_CHUNKER_MODE = "late"  # late | agentic | hybrid

# Late Chunking (MODEL_ROUTING.yaml label)
RAG_LATE_CHUNK_MODEL = "embedding"
RAG_LATE_CHUNK_MAX_TOKENS = 8000

# Agentic Chunking (MODEL_ROUTING.yaml label)
RAG_AGENTIC_CHUNK_MODEL = "agentic-chunk"

# Quality Metrics (MODEL_ROUTING.yaml label)
RAG_ENABLE_QUALITY_METRICS = True
RAG_QUALITY_EVAL_MODEL = "quality-eval"

# Chunk Size Configuration
RAG_CHUNK_TARGET_TOKENS = 450  # Target chunk size
RAG_CHUNK_OVERLAP_TOKENS = 80  # Overlap between chunks

# Phase 2: SOTA Embedding-based Similarity
RAG_USE_EMBEDDING_SIMILARITY = False  # Enable Phase 2 (default: False for gradual rollout)
RAG_CHUNKING_WINDOW_SIZE = 3  # Sentences per window for embedding
RAG_CHUNKING_BATCH_SIZE = 16  # Windows to embed in parallel
RAG_USE_CONTENT_BASED_IDS = True  # Use SHA256-based chunk IDs (deterministic)
```

### Environment Variables (.env)

```bash
# Enable Phase 2 SOTA embedding-based similarity
RAG_USE_EMBEDDING_SIMILARITY=true

# Adjust window size for very short/long sentences
RAG_CHUNKING_WINDOW_SIZE=3

# Adjust batch size for rate limiting
RAG_CHUNKING_BATCH_SIZE=16

# Use content-based IDs for determinism across processes
RAG_USE_CONTENT_BASED_IDS=true
```

## Content-based Chunk IDs & Collision Prevention

**P1 Fix (2025-12-30)**: Content-based chunk IDs are now **namespaced by `document_id`** to prevent collisions across documents and tenants.

### Problem (Before Fix)

Chunk IDs were generated as `sha256(text)` without document namespacing:

```python
# BUGGY (before P1 fix)
chunk_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
chunk_id = f"sha256-{chunk_hash[:32]}"
```

**Impact**:
- Identical text across different documents → **same chunk_id**
- Vector store `ON CONFLICT` upserts → **embeddings overwritten**
- Loss of per-document provenance
- Common in: templates, policies, boilerplate text

### Solution (After Fix)

Chunk IDs are now namespaced by `document_id`:

```python
# FIXED (after P1 fix)
normalized_text = chunk_text.lower().strip()
namespaced_content = f"{document_id}:{normalized_text}"
chunk_hash = hashlib.sha256(namespaced_content.encode('utf-8')).hexdigest()
chunk_id = f"sha256-{chunk_hash[:32]}"
```

**Benefits**:
- ✅ Same text in different documents → **DIFFERENT chunk IDs** (collision prevention)
- ✅ Same text in same document → **SAME chunk ID** (determinism preserved)
- ✅ Multi-tenant isolation maintained
- ✅ No more embedding overwrites in vector store

**Applied to**:
- `LateChunker` (lines 749-757 in `late_chunker.py`)
- `AgenticChunker` (lines 414-421 in `agentic_chunker.py`)

**Test Coverage**:
- `test_content_based_ids_prevent_collisions_across_documents` (collision prevention)
- `test_content_based_ids_deterministic_within_document` (determinism)

## Quality Metrics

Quality evaluation is integrated via `ai_core.rag.quality` (see [quality/README.md](../quality/README.md)):

```python
# Enable quality metrics
config = ChunkerConfig(enable_quality_metrics=True)
chunker = HybridChunker(config)

chunks, stats = chunker.chunk(...)

# Quality scores in stats
assert "quality.scores" in stats
assert "quality.mean_coherence" in stats
```

## Performance

**Target**: 2-3x current ingestion time (1-1.5s per document)

**Optimizations**:
- Batch embedding (8-16 docs in parallel)
- Caching (document embeddings, quality scores)
- Parallelization (quality evaluation per chunk)

**Monitoring**: Langfuse spans for chunking latency

## Testing

Run tests:

```bash
# Unit tests
npm run test:py -- ai_core/tests/rag/chunking/test_hybrid_chunker.py
npm run test:py -- ai_core/tests/rag/chunking/test_late_chunker.py
npm run test:py -- ai_core/tests/rag/chunking/test_agentic_chunker.py

# Integration tests
npm run test:py -- ai_core/tests/graphs/test_universal_ingestion_graph_hybrid_chunker.py

# Performance benchmarks
npm run test:py -- ai_core/tests/rag/test_chunker_performance.py
```

## Rollback

If HybridChunker causes issues:

1. Disable feature flag: `FEATURE_HYBRID_CHUNKER_ENABLED=false`
2. Revert chunker in `universal_ingestion_graph.py:961`
3. Restart workers
4. DB reset (pre-MVP): `python manage.py rag_hard_delete --confirm`

**Rollback Time**: ~5 minutes

## Migration from Old Chunkers

**SimpleDocumentChunker** (deprecated):
- 1 chunk per block, max 2048 bytes (truncates)
- Used in CLI and ingestion graph

**SemanticChunker** (deprecated):
- Section tree + token-based sliding window
- Used in legacy `ai_core/tasks.py:chunk()`

Both will be replaced by **HybridChunker** with no migration needed (pre-MVP, DB reset OK).

## References

- [AGENTS.md](../../../AGENTS.md) - Tool Contracts & Architecture
- [Implementation Plan](../../../../../../.claude/plans/functional-questing-clock.md) - SOTA Hybrid Chunker Plan
- [Reranking Contracts](../../../docs/agents/reranking-contracts.md) - RAG-aware Re-ranking
- [Quality Metrics README](../quality/README.md) - LLM-as-Judge Quality Evaluation

---

**Version**: 1.1
**Last Updated**: 2025-12-30
**Status**: Phase 2 Complete (SOTA Embedding-based Similarity)
