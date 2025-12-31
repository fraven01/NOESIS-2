# RAG Quality Metrics

Phased quality evaluation framework for RAG chunk quality in NOESIS 2.

## Overview

This package implements **4-phase quality metrics** to progressively build evaluation infrastructure:

1. **Phase 1**: LLM-as-Judge (immediate, no ground truth needed)
2. **Phase 2**: Pseudo Queries (automated weak labels)
3. **Phase 3**: Human-in-the-Loop (golden set creation)
4. **Phase 4**: Retrieval Metrics (MRR, NDCG@k with validated queries)

## Phase 1: LLM-as-Judge

**Goal**: Measure chunk quality without ground truth labels.

**Metrics**:
- **Coherence** (0-100): Does the chunk form a coherent semantic unit?
- **Completeness** (0-100): Is the chunk self-contained?
- **Reference Resolution** (0-100): Are references resolved or contextualized?
- **Redundancy** (0-100, inverted): Does the chunk avoid unnecessary repetition?

### Usage

```python
from ai_core.rag.quality import ChunkQualityEvaluator

evaluator = ChunkQualityEvaluator(model="gpt-5-nano")

chunks = [
    {"chunk_id": "uuid-123", "text": "...", "parent_ref": "section:intro"},
    ...
]

scores = evaluator.evaluate(chunks, context)

for chunk, score in zip(chunks, scores):
    print(f"Chunk {chunk['chunk_id']}: {score.overall}/100")
    print(f"  Coherence: {score.coherence}")
    print(f"  Completeness: {score.completeness}")
```

### Storage

**Langfuse Trace Tags**:
```python
langfuse_context.update_current_trace(
    tags=[
        f"chunker_mode:{mode}",
        f"quality_coherence:{mean_coherence:.2f}",
        f"quality_overall:{mean_overall:.2f}",
    ]
)
```

**Chunk Metadata**:
```python
chunk["metadata"]["quality"] = {
    "coherence": 85,
    "completeness": 90,
    "reference_resolution": 75,
    "redundancy": 90,  # inverted
    "overall": 85,
    "evaluated_by": "llm-judge-v1",
    "evaluated_at": "2025-12-29T19:00:00Z",
}
```

## Phase 2: Pseudo Queries

**Goal**: Generate queries from chunks automatically to enable retrieval metrics.

**Strategy**:
1. For each chunk, generate 3-5 pseudo queries
2. Use chunk origin as weak label (chunk_id)
3. Store query-chunk pairs for retrieval evaluation

### Usage

```python
from ai_core.rag.quality import PseudoQueryGenerator

generator = PseudoQueryGenerator(model="gpt-5-nano")

chunk = {"chunk_id": "uuid-123", "text": "..."}

queries = generator.generate(chunk)
# ["What is the definition of X?", "How does Y work?", ...]
```

### Storage

Store in test fixtures:

```json
{
  "chunk_id": "uuid-123",
  "queries": [
    "What is the definition of X?",
    "How does Y work?"
  ],
  "document_id": "uuid-456",
  "tenant_id": "tenant-1"
}
```

### Metrics

- **MRR (Mean Reciprocal Rank)**: Average of 1/rank for first correct result
- **Recall@k**: Percentage of queries where correct chunk is in top-k

## Phase 3: Human-in-the-Loop

**Goal**: Manual review of top/bottom 5% chunks to build golden set.

### CLI Tool

```bash
python manage.py review_chunk_quality --tenant=demo --top=5 --bottom=5
```

**Output**:
```
Top 5% chunks (highest quality):
1. Chunk UUID-123 (overall: 95)
   Text: "..."
   [A] Accept  [R] Reject  [E] Edit

Bottom 5% chunks (lowest quality):
1. Chunk UUID-456 (overall: 35)
   Text: "..."
   [A] Accept  [R] Reject  [E] Edit
```

### Storage

```python
# New model: ChunkQualityReview
class ChunkQualityReview(models.Model):
    chunk_id = models.UUIDField()
    reviewer = models.ForeignKey(User)
    decision = models.CharField(choices=["accept", "reject", "edit"])
    feedback = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
```

### Golden Set

After reviewing 100 chunks (50 top + 50 bottom):
- **Validated Query-Answer Pairs**: 50-100 pairs
- **Quality**: Human-verified, production-ready
- **Usage**: Retrieval metrics (Phase 4)

## Phase 4: Retrieval Metrics

**Goal**: Track retrieval quality trends with MRR, NDCG@k.

**Activation**: After 50-100 validated query-answer pairs (Phase 3).

### Metrics

- **MRR (Mean Reciprocal Rank)**: Quality of top result
- **NDCG@k (Normalized Discounted Cumulative Gain)**: Ranking quality
- **Recall@k**: Coverage at top-k

### Usage

```python
from ai_core.rag.quality import RetrievalMetricsEvaluator

evaluator = RetrievalMetricsEvaluator()

queries = [Query(text="What is X?", true_chunk_id="uuid-123"), ...]

metrics = evaluator.evaluate(queries, ground_truth)

print(f"MRR: {metrics.mrr:.3f}")
print(f"NDCG@10: {metrics.ndcg_at_10:.3f}")
print(f"Recall@10: {metrics.recall_at_10:.3f}")
```

### Tracking

Store metrics in **Langfuse** as trends:

```python
langfuse_context.update_current_observation(
    metadata={
        "mrr": metrics.mrr,
        "ndcg_at_10": metrics.ndcg_at_10,
        "recall_at_10": metrics.recall_at_10,
    }
)
```

**Important**: Track **trends**, not absolute values. Compare against baselines (SimpleDocumentChunker).

## Integration with HybridChunker

Quality evaluation is automatically enabled when `enable_quality_metrics=True`:

```python
from ai_core.rag.chunking import HybridChunker, ChunkerConfig

config = ChunkerConfig(
    mode=ChunkerMode.LATE,
    enable_quality_metrics=True,  # Enable Phase 1
    quality_model="gpt-5-nano",
)

chunker = HybridChunker(config)

chunks, stats = chunker.chunk(...)

# Quality scores in stats
print(stats["quality.scores"])  # List[ChunkQualityScore]
print(stats["quality.mean_coherence"])  # float
print(stats["quality.mean_overall"])  # float
```

## Configuration

Settings in `config/settings.py` (will be `ai_core/settings.py` after migration):

```python
# Quality Metrics
RAG_ENABLE_QUALITY_METRICS = True
RAG_QUALITY_EVAL_MODEL = "gpt-5-nano"
RAG_QUALITY_EVAL_SAMPLE_RATE = 1.0  # 0.0-1.0 (sample 100% of chunks)

# Phase 2: Pseudo Queries
RAG_PSEUDO_QUERY_COUNT = 5  # queries per chunk
RAG_PSEUDO_QUERY_MODEL = "gpt-5-nano"

# Phase 4: Retrieval Metrics
RAG_RETRIEVAL_METRICS_ENABLED = False  # Enable after Phase 3
RAG_RETRIEVAL_METRICS_TOP_K = 10
```

## Performance

**Target**: <500ms per document for quality evaluation.

**Optimizations**:
- Parallel evaluation (ThreadPoolExecutor, 4 workers)
- Caching (chunk hash → quality score)
- Sampling (sample_rate < 1.0 for large documents)

**Monitoring**: Langfuse spans for quality evaluation latency

## Testing

Run tests:

```bash
# Unit tests
npm run test:py -- ai_core/tests/rag/quality/test_llm_judge.py
npm run test:py -- ai_core/tests/rag/quality/test_pseudo_queries.py
npm run test:py -- ai_core/tests/rag/quality/test_retrieval_metrics.py

# Integration tests
npm run test:py -- ai_core/tests/graphs/test_universal_ingestion_graph_quality.py
```

## Phased Rollout Schedule

| Phase | Week | Deliverable |
|-------|------|-------------|
| Phase 1 | 3 | LLM-as-Judge scores all chunks |
| Phase 2 | 6 | Pseudo queries generated, MRR/Recall@k computed |
| Phase 3 | 7 | Human review CLI, 50-100 validated pairs |
| Phase 4 | 8 | MRR, NDCG@k tracked in Langfuse |

## Quality Baselines

**Target Improvements** (vs. SimpleDocumentChunker):
- **LLM-as-Judge**: >= 95% of SemanticChunker baseline
- **MRR**: +10% improvement
- **Recall@10**: +15% improvement

## References

- [chunking/README.md](../chunking/README.md) - Chunking Strategies
- [docs/rag/overview.md](../../../docs/rag/overview.md) - RAG Overview
- [docs/observability/langfuse.md](../../../docs/observability/langfuse.md) - Langfuse Tracing
- [Plan](../../../../../../.claude/plans/functional-questing-clock.md) - Implementation Plan

---

**Version**: 1.0
**Last Updated**: 2025-12-29
**Status**: Phase 0 (Setup)
