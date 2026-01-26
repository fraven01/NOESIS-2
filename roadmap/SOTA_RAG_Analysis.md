# SOTA RAG Stack Analysis Report

**Version:** 1.0
**Datum:** 2026-01-22
**Scope:** NOESIS 2 RAG-Stack vs. State-of-the-Art 2026
**Status:** Baseline Assessment

---

## Executive Summary

| Komponente | Score | Hauptlücke | Priorität |
|------------|-------|------------|-----------|
| **Chunking** | 0.65 | Contextual Retrieval fehlt | P1 |
| **Retrieval** | 0.70 | pg_trgm statt BM25 (kritisch) | P1 |
| **Reranking** | 0.55 | Kein Cross-Encoder | **P0** |
| **Evaluation** | 0.00 | **BLOCKER** - keine Metriken | **P0** |
| **Gesamt** | **0.48** | | |

> **Kritisch:** Ohne implementiertes Evaluation-Framework (NDCG/MRR/Recall@k) sind alle SOTA-Aussagen in diesem Report nur qualitative Schätzungen. Evaluation ist der **Blocker** für jede weitere Optimierung.

---

## 1. Chunking Analysis

### 1.1 Aktueller Stand

```json
{
  "chunking_strategy": "Hybrid (Late + Agentic + Adaptive)",
  "implemented_modes": {
    "late_chunking": {
      "phase1": "Jaccard similarity (text-based, default fallback)",
      "phase2": "Embedding-based cosine similarity (DISABLED by default)",
      "overlap": "80 tokens (configurable)"
    },
    "agentic_chunking": "LLM-guided boundary detection (Gemini Flash)",
    "adaptive_chunking": "Structure-aware with section preservation"
  },
  "context_preservation": {
    "preserved": ["section_path", "page_index", "document_title", "list_headers", "chunk_prefix"],
    "lost": ["named_entities", "coreference_chains", "table_structure", "cross_references", "contextual_summaries"]
  }
}
```

### 1.2 Metriken (geschätzt, nicht gemessen)

| Metrik | Wert | Konfidenz |
|--------|------|-----------|
| Context Loss Score | ~0.35 | Low (keine Messung) |
| Recall@20 (geschätzt) | ~0.72 | Low |
| Abweichung zu SOTA | -18% bis -25% | Low |

### 1.3 Stärken

- ✅ Embedding-basierte Boundary Detection (Phase 2 vorhanden, aber disabled)
- ✅ LLM-guided Chunking mit Fallback-Architektur
- ✅ Content-based IDs verhindern Kollisionen
- ✅ Adaptive Chunking respektiert Dokumentstruktur
- ✅ List-aware Chunking (verhindert Listenaufteilung)

### 1.4 Lücken zu SOTA

| Feature | Status | SOTA 2026 Referenz | Gap Severity |
|---------|--------|-------------------|--------------|
| **Contextual Retrieval** | ❌ Missing | Anthropic Contextual Embeddings + Contextual BM25 | **Kritisch** |
| Embedding Similarity | ⚠️ Disabled | Standard bei allen SOTA-Systemen | Hoch |
| Entity-aware Chunks | ❌ Missing | NER + Coreference Resolution | Mittel |
| Table Parsing | ❌ Missing | Structure-preserving table chunks | Mittel |
| Hierarchical Summaries | ❌ Missing | Recursive summarization (RAPTOR) | Mittel |
| Semantic Overlap | ❌ Missing | Embedding-based statt token-based | Niedrig |

### 1.5 Contextual Retrieval Gap (Detail)

**Was fehlt:**
Anthropics "Contextual Retrieval" (2024) kombiniert:
1. **Contextual Embeddings**: Jeder Chunk wird mit einem LLM-generierten Kontext-Präfix versehen, der den Chunk im Dokumentkontext situiert
2. **Contextual BM25**: Sparse Search auf den kontextualisierten Chunks

**Aktueller Stand NOESIS 2:**
- Chunk-Prefix enthält nur strukturelle Metadaten (Title, Section Path)
- Kein LLM-generierter semantischer Kontext
- BM25 fehlt komplett (nur pg_trgm)

**Erwarteter Impact:**
- Anthropic berichtet 49% Reduktion der Retrieval-Failures
- Kombiniert mit Reranking: 67% Reduktion

### 1.6 Empfehlungen

```json
{
  "priority": 1,
  "recommendations": [
    {
      "action": "Enable RAG_USE_EMBEDDING_SIMILARITY=true",
      "effort": "Trivial (Feature-Flag)",
      "impact": "+5-8% Recall (geschätzt)",
      "blocker": "Evaluation Framework für Validierung"
    },
    {
      "action": "Implement Contextual Chunk Enrichment",
      "effort": "Medium (LLM-Integration bei Ingestion)",
      "impact": "+15-20% Recall (Anthropic-Referenz)",
      "blocker": "Token-Kosten, Evaluation Framework"
    },
    {
      "action": "Add NER + Entity Linking",
      "effort": "High",
      "impact": "Unklar ohne Evaluation",
      "blocker": "Evaluation Framework"
    }
  ]
}
```

---

## 2. Retrieval Analysis

### 2.1 Aktueller Stand

```json
{
  "retrieval_model": "pgvector (HNSW/IVFFlat) + pg_trgm (trigram)",
  "embedding_model": "oai-embed-small (1536D)",
  "hybrid_config": {
    "dense": "pgvector cosine similarity",
    "sparse": "PostgreSQL pg_trgm (trigram matching)",
    "fusion": "RRF (k=60)",
    "alpha": 0.7,
    "min_sim": 0.15
  },
  "multi_stage": {
    "oversample_factor": 4,
    "confidence_retry": true,
    "query_variants": "3-5"
  }
}
```

### 2.2 Kritische Lücke: pg_trgm vs. BM25

| Aspekt | pg_trgm (aktuell) | BM25 (SOTA) |
|--------|-------------------|-------------|
| **Algorithmus** | Character n-gram overlap | TF-IDF mit Längennormalisierung |
| **Precision** | Niedrig (fuzzy matching) | Hoch (term-basiert) |
| **Recall** | Mittel | Hoch |
| **Standardisierung** | PostgreSQL-spezifisch | Industrie-Standard |
| **Tuning** | Begrenzt (similarity threshold) | k1, b Parameter |
| **Hybrid-Kompatibilität** | Suboptimal | Referenz für RRF |

**pg_trgm ist KEIN Ersatz für BM25.** Trigram-Matching ist für Fuzzy-Suche und Typo-Toleranz konzipiert, nicht für Relevanz-Ranking. SOTA Hybrid Search basiert explizit auf Dense + BM25.

### 2.3 Metriken (geschätzt, nicht gemessen)

| Metrik | Aktuell (geschätzt) | Mit BM25 (Referenz) | Gap |
|--------|---------------------|---------------------|-----|
| Recall@20 | ~0.73 | ~0.85 | -12% |
| MRR@20 | ~0.60 | ~0.72 | -12% |
| Hybrid Lift vs. Dense | +8% | +15-20% | -7-12% |

### 2.4 Stärken

- ✅ Echte Hybrid Search Architektur (Dense + Sparse via RRF)
- ✅ Multi-stage mit Oversample → Rerank
- ✅ Query Expansion (3-5 Varianten)
- ✅ Confidence-based Retry
- ✅ Semantic Cache
- ✅ Multi-Tenant Isolation

### 2.5 Lücken zu SOTA

| Feature | Status | SOTA 2026 Referenz | Gap Severity |
|---------|--------|-------------------|--------------|
| **BM25 Sparse Search** | ❌ Missing (pg_trgm) | Elasticsearch/Lucene BM25 | **Kritisch** |
| MMR Diversity | ❌ Missing | Maximal Marginal Relevance | Hoch |
| Late Interaction | ❌ Missing | ColBERT/PLAID | Mittel |
| HyDE | ⚠️ Disabled | Standard bei Query Expansion | Mittel |
| Adaptive Retrieval | ⚠️ Basic | LLM-driven routing | Niedrig |

### 2.6 Empfehlungen

```json
{
  "priority": 1,
  "recommendations": [
    {
      "action": "Replace pg_trgm with BM25 (Elasticsearch/Meilisearch)",
      "effort": "High (neue Infrastruktur)",
      "impact": "+10-15% Hybrid Quality",
      "blocker": "Ops-Komplexität, Evaluation Framework"
    },
    {
      "action": "Enable RAG_HYDE_ENABLED=true",
      "effort": "Trivial (Feature-Flag)",
      "impact": "+3-5% Query-Document Alignment",
      "blocker": "Token-Kosten"
    },
    {
      "action": "Implement MMR Diversity Scoring",
      "effort": "Medium",
      "impact": "Verbessert Result-Diversität",
      "blocker": "Evaluation Framework"
    }
  ]
}
```

---

## 3. Reranking Analysis

### 3.1 Aktueller Stand

```json
{
  "reranker_type": "Hybrid Heuristic + Optional LLM",
  "architecture": {
    "tier1": "6-feature weighted linear combination",
    "tier2": "LLM listwise ranker (optional, disabled by default)",
    "evidence_graph": true
  },
  "features": [
    "confidence (retrieval score)",
    "parent_relevance",
    "section_match",
    "adjacency_bonus",
    "doc_type_match",
    "question_density"
  ],
  "weight_modes": ["static", "learned", "adaptive"],
  "learned_weights": {
    "feedback_types": ["used_source", "click"],
    "update_frequency": "daily batch"
  }
}
```

### 3.2 Kritische Lücke: Kein Cross-Encoder

**Cross-Encoder Reranker sind SOTA-Standard seit 2023.** Sie bieten:
- Bidirektionale Attention zwischen Query und Passage
- Signifikant höhere Precision als Bi-Encoder oder Heuristiken
- Typischer Lift: +10-18% MRR

**Aktueller Stand NOESIS 2:**
- Nur heuristische Features (keine neuronale Relevanz-Modellierung)
- LLM-Reranker ist langsam und teuer (Full LLM Call)
- Kein spezialisiertes Ranking-Modell

**Referenz-Modelle (2026):**
- `BAAI/bge-reranker-v2-m3` (multilingual, 568M params)
- `cross-encoder/ms-marco-MiniLM-L-12-v2` (English, 33M params)
- Cohere Rerank v3

### 3.3 Metriken (geschätzt, nicht gemessen)

| Metrik | Heuristic (aktuell) | Cross-Encoder (SOTA) | Gap |
|--------|---------------------|----------------------|-----|
| Recall@20 Delta | +5-8% | +12-18% | -7-10% |
| MRR@20 Delta | +10-15% | +18-25% | -8-10% |
| Latency | <10ms | 50-200ms | Trade-off |

### 3.4 Stärken

- ✅ Interpretable 6-Feature Scoring
- ✅ Evidence Graph für Struktur-Awareness
- ✅ Learned Weights via Implicit Feedback
- ✅ LLM Fallback für Edge Cases
- ✅ Cost-efficient Default (kein API-Call)

### 3.5 Lücken zu SOTA

| Feature | Status | SOTA 2026 Referenz | Gap Severity |
|---------|--------|-------------------|--------------|
| **Cross-Encoder Reranker** | ❌ Missing | BGE-reranker, Cohere Rerank | **Kritisch (P0)** |
| Pairwise Ranking Loss | ❌ Missing | LambdaMART, LambdaRank | Hoch |
| Neural Ranker | ❌ Missing | Dense passage reranker | Hoch |
| Multi-Stage Reranking | ❌ Missing | Cascade (fast → precise) | Mittel |
| A/B Metrics | ❌ Missing | NDCG, MRR tracking | **Blocker** |

### 3.6 Empfehlungen

```json
{
  "priority": 0,
  "recommendations": [
    {
      "action": "Integrate Cross-Encoder Reranker (BGE-reranker-v2-m3)",
      "effort": "Medium (Model Hosting + Integration)",
      "impact": "+10-15% MRR (Referenz-Werte)",
      "blocker": "Latency Budget, Evaluation Framework",
      "note": "P0 - höchste Priorität für Qualitätsziel"
    },
    {
      "action": "Add Cross-Encoder as Tier 2 nach Heuristic",
      "effort": "Low (wenn Modell gehostet)",
      "impact": "Cascade: schnell für einfache, präzise für schwere Queries",
      "blocker": "Query-Difficulty Classification"
    },
    {
      "action": "Implement explicit A/B testing framework",
      "effort": "Medium",
      "impact": "Ermöglicht Validierung aller Änderungen",
      "blocker": "Golden Dataset"
    }
  ]
}
```

---

## 4. Evaluation Framework

### 4.1 Aktueller Stand

```json
{
  "evaluation_status": "NOT IMPLEMENTED",
  "metrics_tracked": [],
  "golden_dataset": false,
  "a_b_testing": false,
  "automated_regression": false
}
```

### 4.2 BLOCKER-Status

> **Ohne Evaluation Framework sind alle Optimierungen im Blindflug.**

Jede SOTA-Aussage in diesem Report basiert auf:
- Qualitativen Schätzungen
- Referenz-Werten aus Papers/Benchmarks
- Keine Messung am eigenen Datenset

**Konsequenz:** Bevor irgendeine Optimierung implementiert wird, muss das Evaluation Framework stehen.

### 4.3 Erforderliche Komponenten

| Komponente | Status | Priorität |
|------------|--------|-----------|
| **Golden Answer Dataset** | ❌ Missing | P0 |
| **Recall@k Berechnung** | ❌ Missing | P0 |
| **MRR@k Berechnung** | ❌ Missing | P0 |
| **NDCG@k Berechnung** | ❌ Missing | P0 |
| **Entity Recall** | ❌ Missing | P1 |
| **A/B Test Framework** | ❌ Missing | P1 |
| **Automated Regression** | ❌ Missing | P2 |
| **Drift Detection** | ⚠️ Partial (Langfuse) | P2 |

### 4.4 Golden Dataset Requirements

```yaml
golden_dataset:
  minimum_size: 200 query-answer pairs
  coverage:
    - document_types: all supported
    - query_types: [factual, analytical, extractive, comparative]
    - difficulty: [easy, medium, hard]
  annotations:
    - relevant_passages: list[passage_id] per query
    - relevance_grade: [0, 1, 2, 3] per passage
    - expected_answer: ground truth
  refresh_cadence: quarterly
```

### 4.5 Empfehlungen

```json
{
  "priority": 0,
  "recommendations": [
    {
      "action": "Create Golden Dataset (200+ QA pairs)",
      "effort": "High (manuelles Labeling)",
      "impact": "Enabler für alle weiteren Optimierungen",
      "blocker": "Domain-Expertise für Labeling"
    },
    {
      "action": "Implement Recall/MRR/NDCG calculation",
      "effort": "Low (Standard-Formeln)",
      "impact": "Messbarkeit",
      "blocker": "Golden Dataset"
    },
    {
      "action": "Add evaluation to CI/CD",
      "effort": "Medium",
      "impact": "Regression Prevention",
      "blocker": "Evaluation Implementation"
    }
  ]
}
```

---

## 5. SOTA Comparison Summary

### 5.1 Gesamtbewertung

```json
{
  "overall_score": "0.48 / 1.0",
  "confidence": "LOW (keine Metriken)",
  "component_scores": {
    "chunking": 0.65,
    "retrieval": 0.70,
    "reranking": 0.55,
    "evaluation": 0.00
  },
  "blocker": "Evaluation Framework fehlt - alle Scores sind Schätzungen"
}
```

### 5.2 Major Gaps (priorisiert)

| # | Gap | Severity | Effort | Impact |
|---|-----|----------|--------|--------|
| 1 | **Evaluation Framework fehlt** | BLOCKER | High | Enabler |
| 2 | **Kein Cross-Encoder Reranker** | Kritisch | Medium | +10-15% MRR |
| 3 | **pg_trgm statt BM25** | Kritisch | High | +10-15% Hybrid |
| 4 | **Contextual Retrieval fehlt** | Kritisch | Medium | +15-20% Recall |
| 5 | Phase 2 Chunking disabled | Hoch | Trivial | +5-8% Recall |
| 6 | HyDE disabled | Mittel | Trivial | +3-5% |
| 7 | MMR Diversity fehlt | Mittel | Medium | Diversität |
| 8 | Entity-aware Chunks | Mittel | High | Unklar |

### 5.3 Priorisierte Roadmap

```json
{
  "phase_0_blocker": {
    "name": "Evaluation Foundation",
    "duration": "2-4 Wochen",
    "items": [
      {
        "component": "evaluation",
        "action": "Golden Dataset erstellen",
        "priority": "P0",
        "effort": "High",
        "owner": "TBD"
      },
      {
        "component": "evaluation",
        "action": "Recall/MRR/NDCG implementieren",
        "priority": "P0",
        "effort": "Low",
        "owner": "TBD"
      }
    ]
  },
  "phase_1_quick_wins": {
    "name": "Feature Flags & Low-Hanging Fruit",
    "duration": "1 Woche",
    "prerequisite": "phase_0_blocker",
    "items": [
      {
        "component": "chunking",
        "action": "RAG_USE_EMBEDDING_SIMILARITY=true",
        "priority": "P1",
        "effort": "Trivial"
      },
      {
        "component": "retrieval",
        "action": "RAG_HYDE_ENABLED=true",
        "priority": "P1",
        "effort": "Trivial"
      }
    ]
  },
  "phase_2_reranking": {
    "name": "Cross-Encoder Integration",
    "duration": "2-3 Wochen",
    "prerequisite": "phase_0_blocker",
    "items": [
      {
        "component": "reranking",
        "action": "BGE-reranker-v2-m3 hosten",
        "priority": "P0",
        "effort": "Medium"
      },
      {
        "component": "reranking",
        "action": "Als Tier 2 integrieren",
        "priority": "P0",
        "effort": "Low"
      },
      {
        "component": "reranking",
        "action": "A/B Test gegen Heuristic",
        "priority": "P0",
        "effort": "Low"
      }
    ]
  },
  "phase_3_hybrid_upgrade": {
    "name": "BM25 Integration",
    "duration": "4-6 Wochen",
    "prerequisite": "phase_0_blocker",
    "items": [
      {
        "component": "retrieval",
        "action": "Elasticsearch/Meilisearch Setup",
        "priority": "P1",
        "effort": "High"
      },
      {
        "component": "retrieval",
        "action": "BM25 Index aufbauen",
        "priority": "P1",
        "effort": "Medium"
      },
      {
        "component": "retrieval",
        "action": "pg_trgm durch BM25 ersetzen",
        "priority": "P1",
        "effort": "Medium"
      }
    ]
  },
  "phase_4_contextual": {
    "name": "Contextual Retrieval",
    "duration": "3-4 Wochen",
    "prerequisite": "phase_3_hybrid_upgrade",
    "items": [
      {
        "component": "chunking",
        "action": "Context Header Generation bei Ingestion",
        "priority": "P1",
        "effort": "Medium"
      },
      {
        "component": "chunking",
        "action": "Contextual BM25 Index",
        "priority": "P1",
        "effort": "Low"
      }
    ]
  }
}
```

---

## 6. Referenzen

### 6.1 SOTA Papers & Benchmarks

| Referenz | Relevanz | Link |
|----------|----------|------|
| Anthropic Contextual Retrieval (2024) | Chunking, Hybrid | [Blog](https://www.anthropic.com/news/contextual-retrieval) |
| MTEB Benchmark | Embedding Evaluation | [HuggingFace](https://huggingface.co/spaces/mteb/leaderboard) |
| BEIR Benchmark | Retrieval Evaluation | [GitHub](https://github.com/beir-cellar/beir) |
| MS MARCO | Reranking Benchmark | [Microsoft](https://microsoft.github.io/msmarco/) |
| ColBERT v2 | Late Interaction | [Stanford](https://github.com/stanford-futuredata/ColBERT) |
| RAPTOR | Hierarchical Retrieval | [Paper](https://arxiv.org/abs/2401.18059) |

### 6.2 Interne Referenzen

| Datei | Inhalt |
|-------|--------|
| [ai_core/rag/chunking/](../ai_core/rag/chunking/) | Chunker Implementierungen |
| [ai_core/rag/vector_client.py](../ai_core/rag/vector_client.py) | Hybrid Search |
| [ai_core/rag/rerank.py](../ai_core/rag/rerank.py) | Reranking |
| [ai_core/rag/rerank_features.py](../ai_core/rag/rerank_features.py) | Feature Engineering |
| [config/rag_routing_rules.yaml](../config/rag_routing_rules.yaml) | Routing Config |

---

## 7. Changelog

| Version | Datum | Änderung |
|---------|-------|----------|
| 1.0 | 2026-01-22 | Initial SOTA Assessment |

---

## 8. Next Steps

1. **Sofort**: Review dieses Reports mit Team
2. **Woche 1**: Golden Dataset Konzept erstellen
3. **Woche 2-4**: Evaluation Framework implementieren
4. **Danach**: Phase 1-4 nach Roadmap

> **Reminder:** Keine Optimierung ohne Messung. Evaluation ist P0.
