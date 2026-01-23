"""Offline retrieval evaluation with proxy labels (used_sources)."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.graphs.technical.rag_retrieval import (
    RAG_RETRIEVAL_IO_VERSION_STRING,
    RAG_RETRIEVAL_SCHEMA_ID,
    RagRetrievalGraph,
)
from ai_core.models import RagFeedbackEvent
from ai_core.nodes import retrieve
from ai_core.rag.metrics import evaluate_ranking


@dataclass(frozen=True)
class EvalCase:
    query_text: str
    relevant_ids: set[str]
    case_id: str | None = None
    collection_id: str | None = None
    workflow_id: str | None = None
    thread_id: str | None = None
    source: str = "feedback"


@dataclass
class EvalRunResult:
    run_metadata: dict[str, Any]
    summary: dict[str, Any]
    cases: list[dict[str, Any]] = field(default_factory=list)


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _load_json_lines(path: Path) -> list[Mapping[str, Any]]:
    raw = path.read_text(encoding="utf-8").splitlines()
    rows: list[Mapping[str, Any]] = []
    for line in raw:
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, Mapping):
            rows.append(payload)
    return rows


def _load_json_payload(path: Path) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        return [payload]
    return []


def _load_query_payloads(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return _load_json_lines(path)
    return _load_json_payload(path)


def _extract_relevant_ids(payload: Mapping[str, Any]) -> set[str]:
    for key in ("relevant_ids", "source_ids", "used_sources", "relevant_sources"):
        raw = payload.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            return {str(item).strip() for item in raw if str(item).strip()}
    return set()


def _extract_query_text(payload: Mapping[str, Any]) -> str | None:
    for key in ("query", "question", "query_text", "text"):
        value = _coerce_str(payload.get(key))
        if value:
            return value
    return None


def _extract_eval_id(match: Mapping[str, Any]) -> str | None:
    meta = match.get("meta")
    if isinstance(meta, Mapping):
        chunk_id = _coerce_str(meta.get("chunk_id"))
        if chunk_id:
            return chunk_id
        doc_id = _coerce_str(meta.get("document_id"))
        if doc_id:
            return doc_id
    for key in ("id", "hash", "source"):
        value = _coerce_str(match.get(key))
        if value:
            return value
    return None


def _default_hybrid_payload(
    *, top_k: int | None = None
) -> dict[str, float | int | None]:
    payload: dict[str, float | int | None] = {
        "alpha": 0.7,
        "min_sim": 0.15,
        "top_k": 5,
        "vec_limit": 50,
        "lex_limit": 50,
        "trgm_limit": None,
        "max_candidates": 50,
        "diversify_strength": 0.3,
    }
    if top_k is not None and top_k > 0:
        payload["top_k"] = top_k
        payload["max_candidates"] = max(int(payload["max_candidates"]), top_k)
    return payload


def _coerce_sequence(value: object) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return []


def _build_eval_cases_from_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    source: str,
) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for payload in payloads:
        query_text = _extract_query_text(payload)
        if not query_text:
            continue
        relevant_ids = _extract_relevant_ids(payload)
        if not relevant_ids:
            continue
        cases.append(
            EvalCase(
                query_text=query_text,
                relevant_ids=relevant_ids,
                case_id=_coerce_str(payload.get("case_id")),
                collection_id=_coerce_str(payload.get("collection_id")),
                workflow_id=_coerce_str(payload.get("workflow_id")),
                thread_id=_coerce_str(payload.get("thread_id")),
                source=source,
            )
        )
    return cases


def _build_eval_cases_from_feedback(
    *,
    tenant_id: str,
    since,
    limit: int | None,
    case_id: str | None,
    collection_id: str | None,
    workflow_id: str | None,
    thread_id: str | None,
) -> list[EvalCase]:
    qs = RagFeedbackEvent.objects.filter(
        tenant_id=tenant_id,
        feedback_type=RagFeedbackEvent.FEEDBACK_USED_SOURCE,
        created_at__gte=since,
    )
    if case_id:
        qs = qs.filter(case_id=case_id)
    if collection_id:
        qs = qs.filter(collection_id=collection_id)
    if workflow_id:
        qs = qs.filter(workflow_id=workflow_id)
    if thread_id:
        qs = qs.filter(thread_id=thread_id)
    events = list(qs.order_by("-created_at")[: (limit or 5000)])

    grouped: dict[
        tuple[str, str | None, str | None, str | None, str | None], set[str]
    ] = {}
    for event in events:
        query_text = _coerce_str(event.query_text)
        if not query_text:
            continue
        relevant = (
            _coerce_str(event.chunk_id)
            or _coerce_str(event.source_id)
            or _coerce_str(event.source_label)
        )
        if not relevant:
            continue
        key = (
            query_text,
            _coerce_str(event.case_id),
            _coerce_str(event.collection_id),
            _coerce_str(event.workflow_id),
            _coerce_str(event.thread_id),
        )
        grouped.setdefault(key, set()).add(relevant)

    cases: list[EvalCase] = []
    for key, relevant_ids in grouped.items():
        query_text, case_value, collection_value, workflow_value, thread_value = key
        cases.append(
            EvalCase(
                query_text=query_text,
                relevant_ids=set(relevant_ids),
                case_id=case_value,
                collection_id=collection_value,
                workflow_id=workflow_value,
                thread_id=thread_value,
                source="feedback",
            )
        )
    return cases


class Command(BaseCommand):
    help = "Run offline retrieval eval using proxy labels (used_sources feedback)."

    def add_arguments(self, parser):
        parser.add_argument("--tenant-id", required=True)
        parser.add_argument("--case-id")
        parser.add_argument("--collection-id")
        parser.add_argument("--workflow-id")
        parser.add_argument("--thread-id")
        parser.add_argument("--process")
        parser.add_argument("--doc-class")
        parser.add_argument("--visibility")
        parser.add_argument("--top-k", type=int, default=5)
        parser.add_argument("--use-rerank", action="store_true")
        parser.add_argument("--limit", type=int, default=100)
        parser.add_argument("--window-days", type=int, default=30)
        parser.add_argument("--input-file")
        parser.add_argument("--output-file")

    def handle(self, *args, **options):
        tenant_id = str(options["tenant_id"]).strip()
        if not tenant_id:
            raise CommandError("--tenant-id is required")

        case_id = _coerce_str(options.get("case_id"))
        collection_id = _coerce_str(options.get("collection_id"))
        workflow_id = _coerce_str(options.get("workflow_id"))
        thread_id = _coerce_str(options.get("thread_id"))
        process = _coerce_str(options.get("process"))
        doc_class = _coerce_str(options.get("doc_class"))
        visibility = _coerce_str(options.get("visibility"))
        top_k = int(options.get("top_k") or 5)
        use_rerank = bool(options.get("use_rerank"))
        limit = int(options.get("limit") or 0) or 100
        window_days = int(options.get("window_days") or 0) or 30

        cases: list[EvalCase] = []
        input_file = _coerce_str(options.get("input_file"))
        if input_file:
            payloads = _load_query_payloads(Path(input_file))
            cases = _build_eval_cases_from_payloads(payloads, source="input")
        if not cases:
            since = timezone.now() - timedelta(days=window_days)
            cases = _build_eval_cases_from_feedback(
                tenant_id=tenant_id,
                since=since,
                limit=limit,
                case_id=case_id,
                collection_id=collection_id,
                workflow_id=workflow_id,
                thread_id=thread_id,
            )

        if not cases:
            raise CommandError("No eval cases found (check window or input file).")

        cases = cases[:limit]
        graph = RagRetrievalGraph()

        per_case: list[dict[str, Any]] = []
        recall_scores: list[float] = []
        mrr_scores: list[float] = []
        ndcg_scores: list[float] = []

        for case in cases:
            scope = ScopeContext(
                tenant_id=tenant_id,
                trace_id=uuid.uuid4().hex,
                invocation_id=uuid.uuid4().hex,
                run_id=uuid.uuid4().hex,
                service_id="rag-eval",
            )
            business = BusinessContext(
                case_id=case_id or case.case_id,
                collection_id=collection_id or case.collection_id,
                workflow_id=workflow_id or case.workflow_id,
                thread_id=thread_id or case.thread_id,
            )
            context = scope.to_tool_context(business=business)

            hybrid_payload = _default_hybrid_payload(top_k=top_k)
            retrieve_input = retrieve.RetrieveInput(
                query=case.query_text,
                process=process,
                doc_class=doc_class,
                visibility=visibility,
                top_k=top_k,
                hybrid=hybrid_payload,
            )
            graph_input = {
                "schema_id": RAG_RETRIEVAL_SCHEMA_ID,
                "schema_version": RAG_RETRIEVAL_IO_VERSION_STRING,
                "tool_context": context,
                "queries": [case.query_text],
                "retrieve": retrieve_input,
                "use_rerank": use_rerank,
            }
            result = graph.invoke(graph_input)

            snippets = result.get("snippets") or []
            ranked_ids = [
                match_id
                for match in _coerce_sequence(snippets)
                if isinstance(match, Mapping)
                for match_id in [_extract_eval_id(match)]
                if match_id
            ]
            metrics = evaluate_ranking(case.relevant_ids, ranked_ids, k=top_k)

            recall_scores.append(metrics.recall_at_k)
            mrr_scores.append(metrics.mrr_at_k)
            ndcg_scores.append(metrics.ndcg_at_k)

            per_case.append(
                {
                    "query_text": case.query_text,
                    "relevant_ids": sorted(case.relevant_ids),
                    "ranked_ids": ranked_ids,
                    "recall_at_k": metrics.recall_at_k,
                    "mrr_at_k": metrics.mrr_at_k,
                    "ndcg_at_k": metrics.ndcg_at_k,
                    "source": case.source,
                }
            )

        summary = {
            "recall_at_k": sum(recall_scores) / len(recall_scores),
            "mrr_at_k": sum(mrr_scores) / len(mrr_scores),
            "ndcg_at_k": sum(ndcg_scores) / len(ndcg_scores),
            "case_count": len(per_case),
        }
        run_metadata = {
            "tenant_id": tenant_id,
            "case_id": case_id,
            "collection_id": collection_id,
            "workflow_id": workflow_id,
            "thread_id": thread_id,
            "process": process,
            "doc_class": doc_class,
            "visibility": visibility,
            "top_k": top_k,
            "use_rerank": use_rerank,
            "rerank_label": "rerank" if use_rerank else None,
            "timestamp": timezone.now().isoformat(),
            "source": "input" if input_file else "feedback",
            "window_days": window_days if not input_file else None,
        }

        result_payload = EvalRunResult(
            run_metadata=run_metadata,
            summary=summary,
            cases=per_case,
        )
        payload = json.dumps(
            {
                "run_metadata": result_payload.run_metadata,
                "summary": result_payload.summary,
                "cases": result_payload.cases,
            },
            indent=2,
            sort_keys=True,
        )

        output_file = _coerce_str(options.get("output_file"))
        if output_file:
            Path(output_file).write_text(payload, encoding="utf-8")
            self.stdout.write(
                self.style.SUCCESS(
                    f"Wrote eval results to {output_file} (cases={len(per_case)})"
                )
            )
        else:
            self.stdout.write(payload)

        return payload
