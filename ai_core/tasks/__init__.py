"""
Celery tasks for ingestion, graph execution, and monitoring.

For backward compatibility, all public tasks are re-exported from submodules.
"""

from __future__ import annotations

# Ingestion tasks
from .ingestion_tasks import (
    chunk,
    embed,
    extract_text,
    ingest_raw,
    ingestion_run,
    pii_mask,
    upsert,
)

# Graph tasks
from .graph_tasks import run_business_graph, run_ingestion_graph

# Monitoring tasks
from .monitoring_tasks import (
    alert_dead_letter_queue,
    cleanup_dead_letter_queue,
    embedding_drift_check,
)
from .rag_feedback_tasks import record_rag_feedback_events, update_rag_rerank_weights

# Helper utilities
from .helpers.task_utils import log_ingestion_run_end, log_ingestion_run_start

__all__ = [
    # Ingestion
    "chunk",
    "embed",
    "extract_text",
    "ingest_raw",
    "ingestion_run",
    "pii_mask",
    "upsert",
    # Graph
    "run_business_graph",
    "run_ingestion_graph",
    # Monitoring
    "alert_dead_letter_queue",
    "cleanup_dead_letter_queue",
    "embedding_drift_check",
    # RAG feedback
    "record_rag_feedback_events",
    "update_rag_rerank_weights",
    # Helpers
    "log_ingestion_run_end",
    "log_ingestion_run_start",
]
