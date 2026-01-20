# SOTA Retrieval Architecture

**Status**: Pre-MVP
**Priority**: P0-P1
**Vision**: Metadata-first, Passage-first, Hybrid, Multi-stage

## Zielbild

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SOTA RETRIEVAL PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────────────┐ │
│  │ Query Planner│───▶│ Hybrid Candidates│───▶│ Structure-Aware Reranking  │ │
│  └──────────────┘    └──────────────────┘    └────────────────────────────┘ │
│         │                    │                           │                   │
│         ▼                    ▼                           ▼                   │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────────────┐ │
│  │ • doc_type   │    │ • Dense (embed)  │    │ • Evidence Graph traversal │ │
│  │   routing    │    │ • Lexical (BM25) │    │ • Parent/Child/Adjacent    │ │
│  │ • expansion  │    │ • Late-interact. │    │ • Confidence weighting     │ │
│  │ • constraints│    │   (ColBERT-style)│    │ • Section coherence        │ │
│  └──────────────┘    └──────────────────┘    └────────────────────────────┘ │
│                                                          │                   │
│                                                          ▼                   │
│                              ┌────────────────────────────────────────────┐  │
│                              │         Passage Assembly                   │  │
│                              │  • Merge adjacent chunks → passages       │  │
│                              │  • Respect section boundaries             │  │
│                              │  • Optimal context window                 │  │
│                              └────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Design-Prinzipien

### 1. Candidates nicht nur Dense

**Current**: Nur Dense Embeddings (pgvector cosine similarity)

**Target**: Hybrid Candidate Generation
```
Candidates = Dense ∪ Lexical ∪ Late-Interaction

Dense:          text-embedding-3-large (1536D) → pgvector
Lexical:        BM25/TF-IDF → pg_trgm oder Elasticsearch
Late-Interact:  ColBERT-style token-level matching (optional Phase 2)
```

**Rationale**: Dense allein verliert bei:
- Exact matches (Artikelnummern, Fachbegriffe)
- Rare terms (spezifische Klauseln)
- Negation ("nicht zuständig" vs "zuständig")

---

### 2. Reranking ist strukturbewusst

**Current**: LLM Reranking sieht nur `text + source`

**Target**: Feature-basiertes Reranking
```python
RerankFeatures = {
    "text_relevance": float,      # LLM-judged semantic match
    "parent_context": float,      # Parent chunk relevance
    "section_match": float,       # Query ↔ section/heading alignment
    "confidence": float,          # Chunker boundary confidence
    "adjacency_bonus": float,     # Kontinuität zu anderen Candidates
    "doc_type_match": float,      # Query intent ↔ document type
    "recency": float,             # is_latest, lifecycle_state
}

final_score = weighted_combination(RerankFeatures)
```

**Rationale**: Metadaten sind Features, nicht Deko. LLM-Reranking ist teuer - Metadaten-Features ermöglichen schnelles Pre-Filtering und Score-Adjustment.

---

### 3. Output sind Passagen, nicht Chunks

**Current**: Return einzelne Chunks (oft 200-500 tokens)

**Target**: Passage Assembly
```python
Passage = {
    "content": str,              # Merged text from adjacent chunks
    "chunks": list[ChunkRef],    # Contributing chunk IDs
    "section": str,              # Enclosing section/heading
    "span": (start_idx, end_idx),# Position in document
    "coherence_score": float,    # Internal semantic coherence
}
```

**Assembly Rules**:
1. Merge adjacent chunks (same `document_id`, consecutive `chunk_index`)
2. Respect section boundaries (don't merge across headings)
3. Target optimal context window (512-1024 tokens)
4. Preserve highest-scoring chunk as anchor

**Rationale**: LLM context is limited. Coherent passages > fragmented chunks. Users want answers, not puzzle pieces.

---

### 4. Query wird geplant

**Current**: Single query → single embedding → top-k

**Target**: Query Planning Pipeline
```python
QueryPlan = {
    "original": str,
    "doc_type_routing": list[str],     # ["policy", "contract"] → route to relevant corpus
    "expanded_queries": list[str],     # Semantic variants
    "constraints": QueryConstraints,   # Filters, date ranges, must-include
    "retrieval_strategy": str,         # "precision" | "recall" | "balanced"
}

class QueryConstraints:
    must_include: list[str]           # Required terms (lexical)
    must_exclude: list[str]           # Negative filter
    date_range: tuple[date, date]     # Temporal scope
    doc_types: list[str]              # Restrict to types
    collections: list[str]            # Restrict to collections
```

**Doc-Type Routing**:
```python
DOC_TYPE_QUERIES = {
    "policy": {
        "expansion": ["{q} Regelung", "{q} Vorschrift", "{q} Geltungsbereich"],
        "boost_sections": ["Geltungsbereich", "Anwendung", "Pflichten"],
    },
    "contract": {
        "expansion": ["{q} Klausel", "{q} Vereinbarung", "{q} Vertragspartei"],
        "boost_sections": ["Vertragsgegenstand", "Leistungen", "Vergütung"],
    },
    "technical": {
        "expansion": ["{q} Spezifikation", "{q} Anforderung", "{q} Implementierung"],
        "boost_sections": ["Architektur", "Schnittstellen", "Konfiguration"],
    },
}
```

**Rationale**: One-size-fits-all queries waste retrieval budget. Doc-type awareness improves precision dramatically.

---

### 5. Evidence Graph

**Current**: Flat list of chunks, deduplication by hash

**Target**: Graph-based Evidence Structure
```
                    ┌─────────────┐
                    │  Document   │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
      ┌─────────┐    ┌─────────┐    ┌─────────┐
      │ Section │    │ Section │    │ Section │
      │   A     │    │   B     │    │   C     │
      └────┬────┘    └────┬────┘    └────┬────┘
           │              │              │
     ┌─────┼─────┐   ┌────┼────┐   ┌────┼────┐
     ▼     ▼     ▼   ▼    ▼    ▼   ▼    ▼    ▼
   [C1]──[C2]──[C3] [C4]─[C5]─[C6] [C7]─[C8]─[C9]
         adjacent         adjacent       adjacent
```

**Graph Edges**:
```python
EvidenceEdge = Literal[
    "parent_of",      # Section → Chunk
    "child_of",       # Chunk → Section
    "adjacent_to",    # Chunk ↔ Chunk (same section)
    "references",     # Cross-document citation
    "supersedes",     # Version relationship
]
```

**Reranking over Subgraphs**:
```python
def score_subgraph(anchor_chunk: Chunk, graph: EvidenceGraph) -> float:
    """Score a chunk considering its graph neighborhood."""
    base_score = anchor_chunk.relevance_score

    # Parent context boost
    parent = graph.get_parent(anchor_chunk)
    if parent and parent.relevance_score > 0.5:
        base_score *= 1.1

    # Adjacent chunks boost (coherent passage potential)
    neighbors = graph.get_adjacent(anchor_chunk)
    relevant_neighbors = [n for n in neighbors if n.relevance_score > 0.3]
    if len(relevant_neighbors) >= 2:
        base_score *= 1.15  # Good passage candidate

    # Section coherence
    section = graph.get_section(anchor_chunk)
    if query_matches_section(query, section.heading):
        base_score *= 1.1

    return base_score
```

**Rationale**: Isolated chunks lack context. Graph structure enables:
- Better passage assembly
- Cross-chunk reasoning
- Section-level relevance signals
- Document-level coherence

---

## Implementation Phases

### Phase 1: Foundation (P0)

#### SOTA-R1.1: Evidence Graph Data Model

**Goal**: Represent chunk relationships as traversable graph.

```python
# ai_core/rag/evidence_graph.py

from dataclasses import dataclass
from typing import Literal

EdgeType = Literal["parent_of", "child_of", "adjacent_to", "references", "supersedes"]

@dataclass(frozen=True)
class EvidenceNode:
    id: str
    type: Literal["document", "section", "chunk"]
    content: str | None
    metadata: dict

@dataclass(frozen=True)
class EvidenceEdge:
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0

class EvidenceGraph:
    """In-memory graph for retrieval-time traversal."""

    def __init__(self):
        self._nodes: dict[str, EvidenceNode] = {}
        self._edges: dict[str, list[EvidenceEdge]] = {}

    def add_chunk_with_context(self, chunk: Chunk) -> None:
        """Add chunk and infer edges from metadata."""
        node = EvidenceNode(
            id=chunk.meta["chunk_id"],
            type="chunk",
            content=chunk.content,
            metadata=chunk.meta,
        )
        self._nodes[node.id] = node

        # Infer parent edge
        if parent_ids := chunk.meta.get("parent_ids"):
            for parent_id in parent_ids:
                self._add_edge(parent_id, node.id, "parent_of")
                self._add_edge(node.id, parent_id, "child_of")

        # Infer adjacency (requires chunk_index + document_id)
        # Deferred to batch processing

    def get_subgraph(self, anchor_id: str, depth: int = 1) -> "EvidenceGraph":
        """Extract local subgraph around anchor node."""
        ...

    def get_adjacent(self, chunk_id: str) -> list[EvidenceNode]:
        """Get chunks adjacent in source document."""
        ...

    def get_parent_context(self, chunk_id: str) -> EvidenceNode | None:
        """Get parent section/document node."""
        ...
```

**Acceptance**:
- [ ] EvidenceGraph model with nodes + edges
- [ ] Build graph from retrieved chunks + metadata
- [ ] Traversal methods: get_adjacent, get_parent, get_subgraph
- [ ] Unit tests for graph construction and traversal

**Pointers**:
- `ai_core/rag/evidence_graph.py` (new)
- `ai_core/rag/ingestion_contracts.py:69-95` (ChunkMeta with parent_ids, chunk_index)

---

#### SOTA-R1.2: Passage Assembly

**Goal**: Merge adjacent chunks into coherent passages.

```python
# ai_core/rag/passage_assembly.py

@dataclass
class Passage:
    content: str
    chunks: list[str]  # chunk_ids
    section: str | None
    document_id: str
    start_index: int
    end_index: int
    coherence_score: float
    anchor_score: float  # Score of highest-ranked contributing chunk

def assemble_passages(
    chunks: list[Chunk],
    graph: EvidenceGraph,
    *,
    max_tokens: int = 1024,
    min_chunks: int = 1,
    max_chunks: int = 5,
) -> list[Passage]:
    """
    Merge adjacent chunks into passages.

    Rules:
    1. Only merge chunks from same document + section
    2. Respect max_tokens limit
    3. Anchor on highest-scoring chunk, expand outward
    4. Don't cross section boundaries
    """
    # Group by document_id
    by_doc = group_by(chunks, key=lambda c: c.meta["document_id"])

    passages = []
    for doc_id, doc_chunks in by_doc.items():
        # Sort by chunk_index
        sorted_chunks = sorted(doc_chunks, key=lambda c: c.meta.get("chunk_index", 0))

        # Find contiguous runs within same section
        runs = find_contiguous_runs(sorted_chunks, graph)

        for run in runs:
            # Anchor on best chunk, expand
            passage = build_passage_from_run(run, max_tokens)
            passages.append(passage)

    return sorted(passages, key=lambda p: p.anchor_score, reverse=True)
```

**Acceptance**:
- [ ] Passage dataclass with content, chunks, section, scores
- [ ] Assembly respects section boundaries
- [ ] Token limit enforced
- [ ] Tests for various document structures

**Pointers**:
- `ai_core/rag/passage_assembly.py` (new)
- `ai_core/nodes/retrieve.py:543-600` (current diversification - replace with passage-aware)

---

#### SOTA-R1.3: Structure-Aware Rerank Features

**Goal**: Extract rerank features from metadata and graph.

```python
# ai_core/rag/rerank_features.py

@dataclass
class RerankFeatures:
    chunk_id: str
    text_relevance: float        # From initial retrieval score
    parent_relevance: float      # Parent chunk/section score (0 if no parent)
    section_match: float         # Query ↔ section heading similarity
    confidence: float            # Chunker boundary confidence
    adjacency_bonus: float       # Has relevant adjacent chunks
    doc_type_match: float        # Query intent ↔ document type
    is_latest: bool              # Version freshness
    position_in_doc: float       # Normalized position (0=start, 1=end)

def extract_features(
    chunk: Chunk,
    query: str,
    graph: EvidenceGraph,
    query_plan: QueryPlan,
) -> RerankFeatures:
    meta = chunk.meta or {}

    # Parent relevance
    parent_node = graph.get_parent_context(meta.get("chunk_id", ""))
    parent_relevance = 0.0
    if parent_node and parent_node.metadata.get("score"):
        parent_relevance = parent_node.metadata["score"]

    # Section match (simple embedding similarity or keyword overlap)
    section = meta.get("section") or meta.get("heading") or ""
    section_match = compute_section_match(query, section)

    # Adjacency bonus
    adjacent = graph.get_adjacent(meta.get("chunk_id", ""))
    relevant_adjacent = [a for a in adjacent if a.metadata.get("score", 0) > 0.3]
    adjacency_bonus = min(len(relevant_adjacent) * 0.1, 0.3)

    # Doc type match
    doc_type = meta.get("document_type", "")
    doc_type_match = 1.0 if doc_type in query_plan.doc_type_routing else 0.5

    return RerankFeatures(
        chunk_id=meta.get("chunk_id", ""),
        text_relevance=meta.get("score", 0.0),
        parent_relevance=parent_relevance,
        section_match=section_match,
        confidence=meta.get("confidence", 1.0),
        adjacency_bonus=adjacency_bonus,
        doc_type_match=doc_type_match,
        is_latest=meta.get("is_latest", True),
        position_in_doc=meta.get("chunk_index", 0) / max(meta.get("total_chunks", 1), 1),
    )

def compute_final_score(features: RerankFeatures, weights: dict[str, float]) -> float:
    """Weighted combination of features."""
    return (
        weights.get("text_relevance", 0.4) * features.text_relevance +
        weights.get("parent_relevance", 0.1) * features.parent_relevance +
        weights.get("section_match", 0.15) * features.section_match +
        weights.get("confidence", 0.1) * features.confidence +
        weights.get("adjacency_bonus", 0.1) * features.adjacency_bonus +
        weights.get("doc_type_match", 0.1) * features.doc_type_match +
        weights.get("recency", 0.05) * (1.0 if features.is_latest else 0.5)
    )
```

**Acceptance**:
- [ ] RerankFeatures dataclass with all signals
- [ ] Feature extraction from chunk metadata + graph
- [ ] Configurable weights per quality_mode
- [ ] Telemetry logs feature values for analysis

**Pointers**:
- `ai_core/rag/rerank_features.py` (new)
- `ai_core/rag/rerank.py:152-241` (integrate features)

---

### Phase 2: Hybrid Candidates (P1)

#### SOTA-R2.1: Lexical Search Integration (BM25)

**Goal**: Add BM25/lexical search alongside dense retrieval.

**Options**:
1. **pg_trgm** (PostgreSQL): Trigram similarity, already available
2. **Elasticsearch/OpenSearch**: Full BM25, requires additional infra
3. **SQLite FTS5**: Lightweight, in-process (for dev/test)
4. **Tantivy** (Rust): Fast, embeddable

**Recommended**: Start with pg_trgm (zero new infra), upgrade to Elasticsearch if needed.

```python
# ai_core/rag/lexical_search.py

def lexical_search_pg_trgm(
    query: str,
    tenant_id: str,
    *,
    top_k: int = 50,
    min_similarity: float = 0.3,
) -> list[Chunk]:
    """BM25-approximation using PostgreSQL pg_trgm."""
    sql = """
        SELECT c.id, c.content, c.metadata,
               similarity(c.content, %s) AS score
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE d.tenant_id = %s
          AND similarity(c.content, %s) > %s
        ORDER BY score DESC
        LIMIT %s
    """
    # Execute and convert to Chunk objects
    ...
```

**Acceptance**:
- [ ] Lexical search function using pg_trgm
- [ ] Returns same Chunk format as dense search
- [ ] Configurable minimum similarity threshold
- [ ] Performance acceptable (<500ms for 100k chunks)

**Pointers**:
- `ai_core/rag/lexical_search.py` (new)
- `ai_core/rag/query_builder.py` (extend for lexical)

---

#### SOTA-R2.2: Hybrid Candidate Fusion

**Goal**: Merge dense + lexical candidates with RRF (Reciprocal Rank Fusion).

```python
# ai_core/rag/hybrid_fusion.py

def reciprocal_rank_fusion(
    *result_lists: list[Chunk],
    k: int = 60,
) -> list[Chunk]:
    """
    Merge multiple ranked lists using RRF.

    RRF_score(d) = Σ 1 / (k + rank_i(d))

    where rank_i(d) is the rank of document d in list i.
    """
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list, start=1):
            chunk_id = chunk.meta.get("chunk_id", str(rank))
            scores[chunk_id] += 1.0 / (k + rank)
            chunks[chunk_id] = chunk

    # Sort by fused score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return [chunks[cid] for cid in sorted_ids]


def hybrid_retrieve(
    query: str,
    context: ToolContext,
    *,
    dense_weight: float = 0.7,
    lexical_weight: float = 0.3,
    top_k: int = 50,
) -> list[Chunk]:
    """Execute dense + lexical search and fuse results."""
    # Parallel execution
    dense_results = dense_search(query, context, top_k=top_k * 2)
    lexical_results = lexical_search(query, context, top_k=top_k * 2)

    # Fuse with RRF
    fused = reciprocal_rank_fusion(dense_results, lexical_results)

    return fused[:top_k]
```

**Acceptance**:
- [ ] RRF implementation
- [ ] Parallel dense + lexical execution
- [ ] Configurable weights
- [ ] Telemetry captures source distribution (% dense vs lexical)

**Pointers**:
- `ai_core/rag/hybrid_fusion.py` (new)
- `ai_core/nodes/retrieve.py` (integrate hybrid)

---

#### SOTA-R2.3: Query Planner

**Goal**: Analyze query to determine routing, expansion, constraints.

```python
# ai_core/rag/query_planner.py

@dataclass
class QueryPlan:
    original: str
    doc_type_routing: list[str]
    expanded_queries: list[str]
    constraints: QueryConstraints
    retrieval_strategy: Literal["precision", "recall", "balanced"]

@dataclass
class QueryConstraints:
    must_include: list[str] = field(default_factory=list)
    must_exclude: list[str] = field(default_factory=list)
    date_range: tuple[date, date] | None = None
    doc_types: list[str] = field(default_factory=list)
    collections: list[str] = field(default_factory=list)

def plan_query(
    query: str,
    corpus_stats: CorpusStats,  # doc_types present, collection info
    *,
    use_llm: bool = True,
) -> QueryPlan:
    """
    Analyze query and create retrieval plan.

    Steps:
    1. Classify query intent (factual, exploratory, comparative)
    2. Detect target document types from query terms
    3. Generate query expansions
    4. Extract implicit constraints
    """
    if use_llm:
        return _llm_query_planning(query, corpus_stats)
    else:
        return _rule_based_planning(query, corpus_stats)


def _rule_based_planning(query: str, corpus_stats: CorpusStats) -> QueryPlan:
    """Fast heuristic-based planning."""
    doc_types = []
    expansions = [query]

    # Detect policy-related queries
    if any(term in query.lower() for term in ["regelung", "vorschrift", "pflicht", "richtlinie"]):
        doc_types.append("policy")
        expansions.append(f"{query} Geltungsbereich")

    # Detect contract-related queries
    if any(term in query.lower() for term in ["vertrag", "klausel", "vereinbarung", "leistung"]):
        doc_types.append("contract")
        expansions.append(f"{query} Vertragsgegenstand")

    # ... more rules

    return QueryPlan(
        original=query,
        doc_type_routing=doc_types or corpus_stats.available_doc_types,
        expanded_queries=expansions,
        constraints=QueryConstraints(),
        retrieval_strategy="balanced",
    )
```

**Acceptance**:
- [ ] QueryPlan and QueryConstraints models
- [ ] Rule-based planner (fast, no LLM)
- [ ] Optional LLM-based planner (better quality)
- [ ] Expansion templates per doc_type

**Pointers**:
- `ai_core/rag/query_planner.py` (new)
- `ai_core/graphs/technical/rag_retrieval.py` (integrate planner)

---

### Phase 3: Advanced (P2)

#### SOTA-R3.1: Late-Interaction Retrieval (ColBERT-style)

**Goal**: Token-level matching for better precision on exact terms.

**Implementation**: Defer to Phase 3. Requires:
- ColBERT model hosting
- Token-level index
- Significant infra investment

**Alternative**: Use lexical (BM25) for exact matching, defer ColBERT.

---

#### SOTA-R3.2: Cross-Document Evidence Linking

**Goal**: Detect and use references between documents.

```python
# Evidence edges for cross-document relationships
class CrossDocEvidence:
    source_chunk: str
    target_document: str
    reference_type: Literal["cites", "supersedes", "related_to"]
    confidence: float
```

**Acceptance**:
- [ ] Detect citations/references during ingestion
- [ ] Add "references" edges to Evidence Graph
- [ ] Retrieval can follow reference edges

---

#### SOTA-R3.3: Adaptive Weight Learning

**Goal**: Learn optimal rerank weights from user feedback.

```python
# Collect implicit feedback
class RetrievalFeedback:
    query: str
    returned_passages: list[str]
    clicked_passages: list[str]  # Implicit positive
    answer_sources: list[str]    # Used in final answer
    user_rating: int | None      # Explicit feedback

# Update weights periodically
def update_weights(feedback_batch: list[RetrievalFeedback]) -> dict[str, float]:
    """Learn optimal weights from feedback."""
    ...
```

---

## Migration Path

```
Current State                          Target State
─────────────────────────────────────────────────────────────
Dense-only search            →    Hybrid (Dense + Lexical + RRF)
Flat chunk list              →    Evidence Graph
Individual chunks            →    Assembled Passages
Single query                 →    Planned + Expanded queries
Text-only reranking          →    Feature-based reranking
Metadata ignored             →    Metadata-first scoring
```

**Non-Breaking Migration**:
1. Evidence Graph is additive (built from existing metadata)
2. Passage Assembly wraps existing chunks
3. Feature reranking extends current reranking
4. Lexical search is additional candidate source
5. Query planning is optional enhancement

---

## Success Metrics

| Metric | Current | Phase 1 | Phase 2 |
|--------|---------|---------|---------|
| Retrieval Precision@10 | Baseline | +15% | +25% |
| Passage Coherence | ~30% adjacent | >70% | >85% |
| Exact Term Recall | ~60% | +10% | +20% (lexical) |
| Metadata Utilization | ~5% | >80% | >90% |
| Query→Answer Latency | Baseline | +50ms | +100ms |

---

## File Structure (Proposed)

```
ai_core/rag/
├── evidence_graph.py       # NEW: Graph data model + traversal
├── passage_assembly.py     # NEW: Chunk → Passage merging
├── query_planner.py        # NEW: Query analysis + expansion
├── rerank_features.py      # NEW: Feature extraction
├── lexical_search.py       # NEW: BM25/pg_trgm search
├── hybrid_fusion.py        # NEW: RRF fusion
├── rerank.py               # MODIFY: Integrate features
├── query_builder.py        # MODIFY: Extend for lexical
└── schemas.py              # MODIFY: Add Passage model
```

---

## Related Documents

- [AGENTS.md](../AGENTS.md) - Contracts and architecture
- [docs/rag/overview.md](../docs/rag/overview.md) - RAG architecture
- [ai_core/graphs/README.md](../ai_core/graphs/README.md) - Graph development
