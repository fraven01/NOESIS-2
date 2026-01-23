"""Management command for comprehensive document chunk review.

Developer-friendly one-command review that generates a full report
including chunk statistics, boundary analysis, quality scores, and recommendations.

Usage:
    # Quick review (tenant auto-resolved from chunks)
    python manage.py review_document <document_id>

    # With explicit tenant (UUID or schema name)
    python manage.py review_document <document_id> --tenant-id=<uuid-or-schema>

    # With LLM quality evaluation (slower but more detailed)
    python manage.py review_document <document_id> --with-quality

    # Export as markdown report
    python manage.py review_document <document_id> --output=report.md
"""

from __future__ import annotations

import json
import os
import re
import statistics
import uuid
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from django.core.management.base import BaseCommand, CommandError
from psycopg2 import sql

from ai_core.rag.vector_client import get_client_for_schema


class Command(BaseCommand):
    """Generate comprehensive chunk quality review for a document."""

    help = "Generate a comprehensive chunk quality review report for a document"
    requires_system_checks: list[str] = []
    requires_migrations_checks = False

    def add_arguments(self, parser):
        parser.add_argument(
            "document_id",
            type=str,
            help="Document UUID to review",
        )
        parser.add_argument(
            "--tenant-id",
            type=str,
            default=None,
            help="Tenant UUID or schema name (optional; auto-resolve when omitted)",
        )
        parser.add_argument(
            "--schema",
            type=str,
            default=None,
            help="RAG schema name (defaults to configured vector schema)",
        )
        parser.add_argument(
            "--with-quality",
            action="store_true",
            help="Run LLM-as-Judge quality evaluation (slower)",
        )
        parser.add_argument(
            "--boundary-context",
            type=int,
            default=150,
            help="Characters to show at chunk boundaries (default: 150)",
        )
        parser.add_argument(
            "--output",
            "-o",
            type=str,
            default=None,
            help="Output file path (markdown format)",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            dest="output_json",
            help="Output as JSON instead of formatted text",
        )

    def handle(self, *args, **options):
        self._load_dotenv()
        self._ensure_vector_env_defaults()
        document_id = options["document_id"]
        tenant_id = options["tenant_id"]
        schema = options["schema"]
        with_quality = options["with_quality"]
        boundary_context = options["boundary_context"]
        output_file = options["output"]
        output_json = options["output_json"]

        # Validate document_id
        try:
            doc_uuid = uuid.UUID(document_id)
        except ValueError as e:
            raise CommandError(f"Invalid document_id: {e}")

        if tenant_id is None:
            self.stderr.write(
                self.style.WARNING("No tenant provided; resolving from chunk metadata.")
            )
            schema, tenant_uuid = self._resolve_tenant_from_document(doc_uuid, schema)
        else:
            tenant_uuid = self._coerce_tenant_uuid(tenant_id, schema)

        # Fetch chunks
        schema, chunks = self._fetch_chunks_with_fallback(
            doc_uuid=doc_uuid,
            tenant_uuid=tenant_uuid,
            schema_hint=schema,
        )

        if not chunks:
            asset_result = self._fetch_chunks_from_assets(
                doc_uuid=doc_uuid,
                tenant_id=tenant_id,
            )
            if asset_result is not None:
                asset_schema, tenant_uuid, chunks = asset_result
                schema = asset_schema
            else:
                raise CommandError(
                    f"No chunks found for document {doc_uuid} in tenant {tenant_uuid} "
                    f"(schema {schema})"
                )

        # Build report
        report = self._build_report(
            doc_uuid=doc_uuid,
            tenant_uuid=tenant_uuid,
            schema=schema,
            chunks=chunks,
            with_quality=with_quality,
            boundary_context=boundary_context,
        )

        # Output
        if output_json:
            output_text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
        else:
            output_text = self._format_report(report)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output_text)
            self.stdout.write(self.style.SUCCESS(f"Report written to: {output_file}"))
        else:
            self.stdout.write(output_text)

    def _coerce_tenant_uuid(self, tenant_id: str, schema: str) -> uuid.UUID:
        """Resolve tenant IDs to the deterministic UUID used by the RAG store."""
        client = get_client_for_schema(schema)
        try:
            return client._coerce_tenant_uuid(tenant_id)
        except ValueError as exc:
            raise CommandError(f"Invalid tenant identifier: {tenant_id}") from exc

    def _resolve_tenant_from_document(
        self,
        doc_uuid: uuid.UUID,
        schema: str | None,
    ) -> tuple[str, uuid.UUID]:
        """Find a tenant_id for the document by inspecting chunk metadata."""
        schema_candidates = self._candidate_schemas(schema)
        seen: list[tuple[str, list[object]]] = []

        for schema_name in schema_candidates:
            client = get_client_for_schema(schema_name)
            self._require_vector_connection(client)
            tenant_rows = self._lookup_document_tenants(client, doc_uuid)
            if tenant_rows:
                seen.append((client._schema, tenant_rows))

        if not seen:
            discovered = self._discover_chunk_schemas()
            for schema_name in discovered:
                if schema_name in schema_candidates:
                    continue
                client = get_client_for_schema(schema_name)
                self._require_vector_connection(client)
                tenant_rows = self._lookup_document_tenants(client, doc_uuid)
                if tenant_rows:
                    seen.append((client._schema, tenant_rows))
            if not seen:
                searched = schema_candidates + [
                    s for s in discovered if s not in schema_candidates
                ]
                raise CommandError(
                    f"No chunks found for document {doc_uuid} in schemas "
                    f"{', '.join(searched)}. {self._format_vector_debug(discovered)}"
                )

        if len(seen) > 1:
            schema_list = ", ".join(
                f"{schema_name} (tenants: {', '.join(str(tid) for tid in tenants)})"
                for schema_name, tenants in seen
            )
            raise CommandError(
                "Document appears in multiple schemas; provide --schema. "
                f"Schemas found: {schema_list}"
            )

        schema_name, tenant_rows = seen[0]
        if len(tenant_rows) > 1:
            tenant_list = ", ".join(str(tid) for tid in tenant_rows)
            raise CommandError(
                "Document spans multiple tenants; provide --tenant-id. "
                f"Tenants found: {tenant_list}"
            )

        return schema_name, uuid.UUID(str(tenant_rows[0]))

    def _fetch_chunks(
        self,
        doc_uuid: uuid.UUID,
        tenant_uuid: uuid.UUID,
        schema: str,
    ) -> list[dict[str, Any]]:
        """Fetch all chunks for a document."""
        client = get_client_for_schema(schema)
        self._require_vector_connection(client)
        chunks_table = client._table("chunks")

        query = sql.SQL(
            """
            SELECT
                c.id,
                c.ord,
                c.text,
                c.metadata,
                length(c.text) as text_length
            FROM {} c
            WHERE c.document_id = %s
              AND c.tenant_id = %s
            ORDER BY c.ord
            """
        ).format(chunks_table)

        rows: list[dict[str, Any]] = []

        with client._connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(query, [doc_uuid, tenant_uuid])
                    for row in cur.fetchall():
                        rows.append(
                            {
                                "chunk_id": str(row[0]),
                                "ord": row[1],
                                "text": row[2],
                                "metadata": row[3] or {},
                                "text_length": row[4],
                            }
                        )
                except Exception as exc:
                    if getattr(exc, "pgcode", None) in {"42P01", "3F000"}:
                        return []
                    raise

        if rows:
            return rows

        doc_text = str(doc_uuid)
        metadata_query = sql.SQL(
            """
            SELECT
                c.id,
                c.ord,
                c.text,
                c.metadata,
                length(c.text) as text_length
            FROM {} c
            WHERE c.tenant_id = %s
              AND (
                c.metadata ->> 'document_version_id' = %s
                OR c.metadata ->> 'document_id' = %s
              )
            ORDER BY c.ord
            """
        ).format(chunks_table)

        with client._connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(metadata_query, [tenant_uuid, doc_text, doc_text])
                    for row in cur.fetchall():
                        rows.append(
                            {
                                "chunk_id": str(row[0]),
                                "ord": row[1],
                                "text": row[2],
                                "metadata": row[3] or {},
                                "text_length": row[4],
                            }
                        )
                except Exception as exc:
                    if getattr(exc, "pgcode", None) in {"42P01", "3F000"}:
                        return []
                    raise

        return rows

    def _fetch_chunks_with_fallback(
        self,
        *,
        doc_uuid: uuid.UUID,
        tenant_uuid: uuid.UUID,
        schema_hint: str | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        schema_candidates = self._candidate_schemas(schema_hint)
        first_schema = schema_candidates[0]
        chunks = self._fetch_chunks(doc_uuid, tenant_uuid, first_schema)
        if chunks or schema_hint:
            return first_schema, chunks

        for schema_name in schema_candidates[1:]:
            chunks = self._fetch_chunks(doc_uuid, tenant_uuid, schema_name)
            if chunks:
                return schema_name, chunks

        return first_schema, chunks

    def _lookup_document_tenants(
        self,
        client,
        doc_uuid: uuid.UUID,
    ) -> list[object]:
        chunks_table = client._table("chunks")
        query = sql.SQL(
            """
            SELECT DISTINCT c.tenant_id
            FROM {} c
            WHERE c.document_id = %s
            """
        ).format(chunks_table)

        with client._connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(query, [doc_uuid])
                    rows = [row[0] for row in cur.fetchall()]
                except Exception as exc:
                    if getattr(exc, "pgcode", None) in {"42P01", "3F000"}:
                        return []
                    raise

        if rows:
            return rows

        doc_text = str(doc_uuid)
        metadata_query = sql.SQL(
            """
            SELECT DISTINCT c.tenant_id
            FROM {} c
            WHERE c.metadata ->> 'document_version_id' = %s
               OR c.metadata ->> 'document_id' = %s
            """
        ).format(chunks_table)

        with client._connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(metadata_query, [doc_text, doc_text])
                    return [row[0] for row in cur.fetchall()]
                except Exception as exc:
                    if getattr(exc, "pgcode", None) in {"42P01", "3F000"}:
                        return []
                    raise

    def _candidate_schemas(self, schema_hint: str | None) -> list[str]:
        candidates: list[str] = []
        if schema_hint:
            candidates.append(str(schema_hint).strip())

        try:
            from django.conf import settings

            stores = getattr(settings, "RAG_VECTOR_STORES", {})
            for config in stores.values():
                schema_name = config.get("schema")
                if schema_name:
                    candidates.append(str(schema_name).strip())
        except Exception:
            pass

        if not candidates:
            candidates.append("rag")

        unique: list[str] = []
        for name in candidates:
            if name and name not in unique:
                unique.append(name)
        return unique

    def _fetch_chunks_from_assets(
        self,
        *,
        doc_uuid: uuid.UUID,
        tenant_id: str | None,
    ) -> tuple[str, uuid.UUID, list[dict[str, Any]]] | None:
        """Fallback to document assets when vector chunks are missing."""
        try:
            from django_tenants.utils import schema_context
            from documents.models import Document, DocumentAsset
        except Exception:
            return None

        tenant_schemas = self._resolve_tenant_schemas(tenant_id)
        matches: list[tuple[str, uuid.UUID, list[dict[str, Any]]]] = []

        for schema_name in tenant_schemas:
            with schema_context(schema_name):
                if not Document.objects.filter(id=doc_uuid).exists():
                    continue
                assets = list(
                    DocumentAsset.objects.filter(
                        document_id=doc_uuid,
                        metadata__asset_kind="chunk",
                    ).order_by("created_at", "asset_id")
                )
                if not assets:
                    continue
                chunks: list[dict[str, Any]] = []
                for index, asset in enumerate(assets):
                    text_value = asset.content or ""
                    if not text_value and isinstance(asset.metadata, dict):
                        text_value = (
                            asset.metadata.get("ocr_text")
                            or asset.metadata.get("text_description")
                            or ""
                        )
                    chunks.append(
                        {
                            "chunk_id": str(asset.asset_id),
                            "ord": index,
                            "text": text_value,
                            "metadata": asset.metadata or {},
                            "text_length": len(text_value),
                        }
                    )
                tenant_uuid = self._coerce_tenant_uuid(schema_name, "rag")
                matches.append((f"documents:{schema_name}", tenant_uuid, chunks))

        if not matches:
            return None
        if len(matches) > 1:
            schema_list = ", ".join(schema_name for schema_name, _, _ in matches)
            raise CommandError(
                "Document appears in multiple tenant schemas; provide --tenant-id. "
                f"Schemas found: {schema_list}"
            )
        return matches[0]

    def _resolve_tenant_schemas(self, tenant_id: str | None) -> list[str]:
        """Resolve tenant schema candidates for document asset lookup."""
        try:
            from customers.models import Tenant
            from customers.tenant_context import TenantContext
        except Exception:
            return []

        if tenant_id:
            tenant = TenantContext.resolve_identifier(tenant_id, allow_pk=True)
            if tenant is None:
                return []
            return [tenant.schema_name]

        return list(
            Tenant.objects.exclude(schema_name="public").values_list(
                "schema_name",
                flat=True,
            )
        )

    def _discover_chunk_schemas(self) -> list[str]:
        """Inspect information_schema for schemas that contain a chunks table."""
        try:
            client = get_client_for_schema("public")
            self._require_vector_connection(client)
        except Exception:
            return []

        query = """
            SELECT DISTINCT table_schema
            FROM information_schema.tables
            WHERE table_name = 'chunks'
        """
        with client._connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(query)
                    return [str(row[0]) for row in cur.fetchall()]
                except Exception:
                    return []

    def _format_vector_debug(self, discovered: list[str]) -> str:
        """Return a short hint about which vector DB we connected to."""
        dsn = os.environ.get("RAG_DATABASE_URL") or os.environ.get("DATABASE_URL") or ""
        if not dsn:
            return "Vector DSN not set."
        parsed = urlparse(dsn)
        db_name = parsed.path.lstrip("/") if parsed.path else ""
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        user = parsed.username or ""
        auth = f"{user}@" if user else ""
        target = f"{parsed.scheme}://{auth}{host}{port}/{db_name}".strip("/")
        schemas = ", ".join(discovered) if discovered else "none"
        return f"Vector DSN {target}; chunk schemas in DB: {schemas}."

    def _require_vector_connection(self, client) -> None:  # type: ignore[no-untyped-def]
        if getattr(client, "_pool", None) is None:
            raise CommandError(
                "Vector DB connection unavailable. Set RAG_URL (or the configured "
                "vector store DSN) and retry."
            )

    def _ensure_vector_env_defaults(self) -> None:
        """Ensure dev-friendly defaults for vector DB connectivity."""
        default_url = "postgresql://noesis2:noesis2@db:5432/noesis2"
        if os.environ.get("RAG_URL"):
            os.environ.setdefault("RAG_DATABASE_URL", os.environ["RAG_URL"])
            return
        if os.environ.get("RAG_DATABASE_URL"):
            os.environ.setdefault("RAG_URL", os.environ["RAG_DATABASE_URL"])
            return
        if os.environ.get("DATABASE_URL"):
            os.environ.setdefault("RAG_DATABASE_URL", os.environ["DATABASE_URL"])
            os.environ.setdefault("RAG_URL", os.environ["DATABASE_URL"])
            return
        os.environ.setdefault("RAG_DATABASE_URL", default_url)
        os.environ.setdefault("RAG_URL", default_url)
        os.environ.setdefault("DATABASE_URL", default_url)

    def _load_dotenv(self) -> None:
        """Best-effort load of .env for dev one-shot usage."""
        if os.environ.get("RAG_URL") or os.environ.get("RAG_DATABASE_URL"):
            return
        try:
            from dotenv import load_dotenv  # type: ignore
        except Exception:
            return
        load_dotenv()

    def _build_report(
        self,
        doc_uuid: uuid.UUID,
        tenant_uuid: uuid.UUID,
        schema: str,
        chunks: list[dict[str, Any]],
        with_quality: bool,
        boundary_context: int,
    ) -> dict[str, Any]:
        """Build the complete review report."""
        coverage = self._compute_coverage(doc_uuid, str(tenant_uuid), chunks)
        report: dict[str, Any] = {
            "meta": {
                "document_id": str(doc_uuid),
                "tenant_id": str(tenant_uuid),
                "schema": schema,
                "generated_at": datetime.utcnow().isoformat(),
                "chunk_count": len(chunks),
            },
            "statistics": self._compute_statistics(chunks),
            "coverage": coverage,
            "boundaries": self._analyze_boundaries(chunks, boundary_context),
            "problems": self._detect_problems(chunks),
            "recommendations": [],
            "chunks": chunks,
        }

        # Optional: LLM quality evaluation
        if with_quality:
            report["quality"] = self._evaluate_quality(chunks)
        else:
            report["quality"] = None

        # Generate recommendations based on findings
        report["recommendations"] = self._generate_recommendations(report)

        return report

    def _compute_statistics(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute chunk statistics."""
        lengths = [c["text_length"] for c in chunks]

        stats = {
            "count": len(chunks),
            "total_chars": sum(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "mean_length": statistics.mean(lengths),
            "median_length": statistics.median(lengths),
            "stdev_length": statistics.stdev(lengths) if len(lengths) > 1 else 0,
        }

        # Length distribution buckets
        buckets = {
            "tiny (<100)": 0,
            "small (100-300)": 0,
            "medium (300-800)": 0,
            "large (800-1500)": 0,
            "huge (>1500)": 0,
        }
        for length in lengths:
            if length < 100:
                buckets["tiny (<100)"] += 1
            elif length < 300:
                buckets["small (100-300)"] += 1
            elif length < 800:
                buckets["medium (300-800)"] += 1
            elif length < 1500:
                buckets["large (800-1500)"] += 1
            else:
                buckets["huge (>1500)"] += 1

        stats["distribution"] = buckets

        return stats

    def _analyze_boundaries(
        self,
        chunks: list[dict[str, Any]],
        context_chars: int,
    ) -> list[dict[str, Any]]:
        """Analyze chunk boundaries."""
        boundaries = []

        for i in range(len(chunks) - 1):
            current = chunks[i]
            next_chunk = chunks[i + 1]

            current_end = current["text"][-context_chars:].strip()
            next_start = next_chunk["text"][:context_chars].strip()

            # Detect potential issues
            issues = []

            # Check if current chunk ends mid-sentence
            if current_end and current_end[-1] not in ".!?:;\n":
                issues.append("ends_mid_sentence")

            # Check if next chunk starts with lowercase (continuation)
            if next_start and next_start[0].islower():
                issues.append("starts_lowercase")

            # Check for split references
            if next_start.startswith(("dieser", "diese", "dieses", "das ", "dies ")):
                issues.append("dangling_reference")

            # Check for numbered list continuation
            if current_end.rstrip().endswith(("1.", "2.", "3.", "a)", "b)")):
                issues.append("list_header_orphan")

            current_index = self._extract_list_index(current["text"], from_end=True)
            next_index = self._extract_list_index(next_chunk["text"], from_end=False)
            if current_index is not None and next_index == current_index + 1:
                issues.append("list_split")

            boundaries.append(
                {
                    "after_chunk": current["ord"],
                    "before_chunk": next_chunk["ord"],
                    "end_text": current_end,
                    "start_text": next_start,
                    "issues": issues,
                    "score": 100 - (len(issues) * 25),  # Simple heuristic score
                }
            )

        return boundaries

    _LIST_ITEM_RE = re.compile(r"^\s*\(?(?P<num>\d{1,4})\)?[.)]\s+")

    def _extract_list_index(self, text: str, *, from_end: bool) -> int | None:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None
        scan_lines = reversed(lines) if from_end else lines
        for line in scan_lines:
            match = self._LIST_ITEM_RE.match(line)
            if not match:
                continue
            try:
                return int(match.group("num"))
            except (TypeError, ValueError):
                return None
        return None

    def _detect_problems(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Detect potential chunking problems."""
        problems = []

        for chunk in chunks:
            chunk_problems = []

            # Too short
            if chunk["text_length"] < 50:
                chunk_problems.append(
                    {
                        "type": "too_short",
                        "severity": "warning",
                        "message": f"Chunk is very short ({chunk['text_length']} chars)",
                    }
                )

            # Too long
            if chunk["text_length"] > 2000:
                chunk_problems.append(
                    {
                        "type": "too_long",
                        "severity": "warning",
                        "message": f"Chunk is very long ({chunk['text_length']} chars)",
                    }
                )

            # Unresolved references at start
            text_start = chunk["text"][:100].lower()
            ref_patterns = [
                "dieser ",
                "diese ",
                "dieses ",
                "das oben",
                "wie erwähnt",
                "this ",
                "these ",
                "the above",
                "as mentioned",
            ]
            for pattern in ref_patterns:
                if text_start.startswith(pattern) or f" {pattern}" in text_start[:50]:
                    chunk_problems.append(
                        {
                            "type": "dangling_reference",
                            "severity": "info",
                            "message": f"Starts with potential reference: '{pattern.strip()}'",
                        }
                    )
                    break

            # Incomplete sentence (no sentence-ending punctuation)
            text_stripped = chunk["text"].strip()
            if text_stripped and text_stripped[-1] not in ".!?:\"'":
                chunk_problems.append(
                    {
                        "type": "incomplete_sentence",
                        "severity": "info",
                        "message": "Chunk doesn't end with sentence punctuation",
                    }
                )

            # Check for table/code fragments (heuristic)
            if chunk["text"].count("|") > 5:
                chunk_problems.append(
                    {
                        "type": "table_fragment",
                        "severity": "info",
                        "message": "Chunk may contain table data",
                    }
                )

            if chunk_problems:
                problems.append(
                    {
                        "chunk_ord": chunk["ord"],
                        "chunk_id": chunk["chunk_id"],
                        "problems": chunk_problems,
                    }
                )

        return problems

    def _evaluate_quality(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Run LLM-as-Judge quality evaluation."""
        try:
            from ai_core.rag.quality import (
                ChunkQualityEvaluator,
                compute_quality_statistics,
            )

            self.stderr.write(
                self.style.WARNING(
                    "Running LLM quality evaluation (this may take a moment)..."
                )
            )

            evaluator = ChunkQualityEvaluator(model="quality-eval")

            # Prepare chunks for evaluator
            eval_chunks = [
                {
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "parent_ref": c["metadata"].get("parent", "unknown"),
                }
                for c in chunks
            ]

            scores = evaluator.evaluate(eval_chunks)
            stats = compute_quality_statistics(scores)

            # Find worst chunks
            sorted_scores = sorted(scores, key=lambda s: s.overall)
            worst_5 = [
                {
                    "chunk_id": s.chunk_id,
                    "overall": s.overall,
                    "coherence": s.coherence,
                    "completeness": s.completeness,
                    "reasoning": s.reasoning,
                }
                for s in sorted_scores[:5]
            ]

            return {
                "evaluated": True,
                "statistics": stats,
                "worst_chunks": worst_5,
                "scores": [s.to_dict() for s in scores],
            }

        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Quality evaluation failed: {e}"))
            return {"evaluated": False, "error": str(e)}

    def _generate_recommendations(self, report: dict[str, Any]) -> list[str]:
        """Generate recommendations based on report findings."""
        recommendations = []
        stats = report["statistics"]
        problems = report["problems"]
        boundaries = report["boundaries"]

        # Length recommendations
        if stats["stdev_length"] > 500:
            recommendations.append(
                "High variance in chunk lengths. Consider adjusting chunker settings for more consistent sizing."
            )

        tiny_count = stats["distribution"]["tiny (<100)"]
        if tiny_count > len(problems) * 0.1:
            recommendations.append(
                f"{tiny_count} tiny chunks (<100 chars) detected. These may reduce retrieval quality."
            )

        huge_count = stats["distribution"]["huge (>1500)"]
        if huge_count > 0:
            recommendations.append(
                f"{huge_count} very large chunks (>1500 chars) detected. Consider reducing max_chunk_size."
            )

        # Boundary recommendations
        bad_boundaries = [b for b in boundaries if b["score"] < 75]
        if len(bad_boundaries) > len(boundaries) * 0.2:
            recommendations.append(
                f"{len(bad_boundaries)} potentially bad chunk boundaries detected. "
                "Consider using semantic chunking or adjusting overlap."
            )

        # Problem-based recommendations
        dangling_refs = sum(
            1
            for p in problems
            for prob in p["problems"]
            if prob["type"] == "dangling_reference"
        )
        if dangling_refs > 0:
            recommendations.append(
                f"{dangling_refs} chunks start with dangling references. "
                "Consider increasing chunk overlap or using late chunking."
            )

        # Quality-based recommendations
        if report.get("quality") and report["quality"].get("evaluated"):
            quality_stats = report["quality"]["statistics"]
            if quality_stats.get("mean_coherence", 100) < 70:
                recommendations.append(
                    f"Low average coherence ({quality_stats['mean_coherence']:.1f}). "
                    "Consider semantic chunking strategy."
                )
            if quality_stats.get("mean_completeness", 100) < 70:
                recommendations.append(
                    f"Low average completeness ({quality_stats['mean_completeness']:.1f}). "
                    "Chunks may need more context. Increase overlap or chunk size."
                )

        coverage = report.get("coverage") or {}
        if coverage.get("available") and coverage.get("status") == "low":
            ratio = coverage.get("ratio", 0.0)
            threshold = coverage.get("threshold", 0.0)
            recommendations.append(
                f"Coverage ratio {ratio:.2f} is below threshold {threshold:.2f}. "
                "Review chunk sizing or ingestion source text."
            )

        if not recommendations:
            recommendations.append(
                "No significant issues detected. Chunking quality looks good!"
            )

        return recommendations

    def _compute_coverage(
        self,
        doc_uuid: uuid.UUID,
        tenant_id: str,
        chunks: list[dict[str, Any]],
        *,
        threshold: float = 0.95,
    ) -> dict[str, Any]:
        source_text = self._load_document_text(doc_uuid, tenant_id)
        if not source_text:
            return {
                "available": False,
                "threshold": threshold,
            }

        source_text = source_text.strip()
        source_chars = len(source_text)
        if source_chars == 0:
            return {
                "available": False,
                "threshold": threshold,
            }

        chunk_chars = sum(len(chunk.get("text") or "") for chunk in chunks)
        raw_ratio = chunk_chars / source_chars if source_chars else 0.0
        ratio = min(raw_ratio, 1.0)
        status = "ok" if ratio >= threshold else "low"

        section_candidates = []
        if status == "low":
            section_candidates = self._coverage_section_hints(chunks)

        return {
            "available": True,
            "source_chars": source_chars,
            "chunk_chars": chunk_chars,
            "ratio": ratio,
            "raw_ratio": raw_ratio,
            "threshold": threshold,
            "status": status,
            "section_hints": section_candidates,
        }

    def _coverage_section_hints(
        self, chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        section_sizes: dict[str, int] = {}
        for chunk in chunks:
            section_id = self._resolve_section_hint(chunk)
            if not section_id:
                continue
            section_sizes[section_id] = section_sizes.get(section_id, 0) + len(
                chunk.get("text") or ""
            )
        ordered = sorted(section_sizes.items(), key=lambda item: item[1])
        return [
            {"section": section, "chunk_chars": size} for section, size in ordered[:8]
        ]

    def _resolve_section_hint(self, chunk: dict[str, Any]) -> str | None:
        meta = chunk.get("metadata")
        if isinstance(meta, dict):
            parent_ids = meta.get("parent_ids")
            if isinstance(parent_ids, list) and parent_ids:
                candidate = parent_ids[-1]
                if candidate:
                    return str(candidate)
            parent_ref = meta.get("parent_ref")
            if parent_ref:
                return str(parent_ref)
            section_path = meta.get("section_path")
            if isinstance(section_path, list) and section_path:
                return " > ".join(str(part) for part in section_path if part)
        section_path = chunk.get("section_path")
        if isinstance(section_path, list) and section_path:
            return " > ".join(str(part) for part in section_path if part)
        parent_ref = chunk.get("parent_ref")
        if parent_ref:
            return str(parent_ref)
        return None

    def _load_document_text(self, doc_uuid: uuid.UUID, tenant_id: str) -> str | None:
        try:
            from django_tenants.utils import schema_context
            from documents.models import DocumentVersion
        except Exception:
            return None

        tenant_schemas = self._resolve_tenant_schemas(tenant_id)
        for schema_name in tenant_schemas:
            with schema_context(schema_name):
                version = (
                    DocumentVersion.objects.filter(document_id=doc_uuid, is_latest=True)
                    .order_by("-created_at")
                    .first()
                )
                if not version:
                    continue
                normalized = version.normalized_document or {}
                if isinstance(normalized, dict):
                    text = normalized.get("content_normalized") or normalized.get(
                        "primary_text"
                    )
                    if isinstance(text, str) and text.strip():
                        return text
        return None

    def _format_report(self, report: dict[str, Any]) -> str:
        """Format report as readable text/markdown."""
        lines = []
        meta = report["meta"]
        stats = report["statistics"]
        coverage = report.get("coverage") or {}

        # Header
        lines.append("=" * 70)
        lines.append("DOCUMENT CHUNK REVIEW REPORT")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Document ID:  {meta['document_id']}")
        lines.append(f"Tenant ID:    {meta['tenant_id']}")
        lines.append(f"Schema:       {meta['schema']}")
        lines.append(f"Generated:    {meta['generated_at']}")
        lines.append(f"Total Chunks: {meta['chunk_count']}")
        lines.append("")

        # Statistics
        lines.append("-" * 70)
        lines.append("CHUNK STATISTICS")
        lines.append("-" * 70)
        lines.append(f"  Total characters: {stats['total_chars']:,}")
        lines.append(f"  Min length:       {stats['min_length']} chars")
        lines.append(f"  Max length:       {stats['max_length']} chars")
        lines.append(f"  Mean length:      {stats['mean_length']:.0f} chars")
        lines.append(f"  Median length:    {stats['median_length']:.0f} chars")
        lines.append(f"  Std deviation:    {stats['stdev_length']:.0f} chars")
        lines.append("")
        lines.append("  Size distribution:")
        for bucket, count in stats["distribution"].items():
            bar = "█" * min(count, 30)
            lines.append(f"    {bucket:18} {count:3} {bar}")
        lines.append("")

        # Coverage
        lines.append("-" * 70)
        lines.append("COVERAGE")
        lines.append("-" * 70)
        if coverage.get("available"):
            lines.append(f"  Source characters: {coverage.get('source_chars', 0):,}")
            lines.append(f"  Chunk characters:  {coverage.get('chunk_chars', 0):,}")
            lines.append(f"  Coverage ratio:    {coverage.get('ratio', 0.0):.3f}")
            lines.append(f"  Threshold:         {coverage.get('threshold', 0.0):.2f}")
            if coverage.get("status") == "low":
                lines.append("  WARNING: coverage below threshold.")
                hints = coverage.get("section_hints") or []
                if hints:
                    lines.append("  Smallest sections (by chunk chars):")
                    for hint in hints:
                        section = hint.get("section")
                        size = hint.get("chunk_chars")
                        if section:
                            lines.append(f"    - {section}: {size}")
        else:
            lines.append("  Coverage unavailable (no normalized document text found).")
        lines.append("")

        # Boundary Analysis
        lines.append("-" * 70)
        lines.append("CHUNK BOUNDARY ANALYSIS")
        lines.append("-" * 70)

        boundaries = report["boundaries"]
        bad_boundaries = [b for b in boundaries if b["issues"]]

        if bad_boundaries:
            lines.append(
                f"  Found {len(bad_boundaries)} potentially problematic boundaries:\n"
            )
            for b in bad_boundaries[:10]:  # Show max 10
                lines.append(
                    f"  --- After Chunk {b['after_chunk']} / Before Chunk {b['before_chunk']} ---"
                )
                lines.append(f"  Issues: {', '.join(b['issues'])}")
                lines.append(f"  End:   \"...{b['end_text'][-80:]}\"")
                lines.append(f"  Start: \"{b['start_text'][:80]}...\"")
                lines.append("")
            if len(bad_boundaries) > 10:
                lines.append(f"  ... and {len(bad_boundaries) - 10} more")
        else:
            lines.append("  All chunk boundaries look clean!")
        lines.append("")

        # Problems
        lines.append("-" * 70)
        lines.append("DETECTED PROBLEMS")
        lines.append("-" * 70)

        problems = report["problems"]
        if problems:
            for p in problems[:15]:  # Show max 15
                lines.append(f"  Chunk {p['chunk_ord']}:")
                for prob in p["problems"]:
                    severity_icon = "⚠️" if prob["severity"] == "warning" else "ℹ️"
                    lines.append(
                        f"    {severity_icon} [{prob['type']}] {prob['message']}"
                    )
            if len(problems) > 15:
                lines.append(
                    f"  ... and {len(problems) - 15} more chunks with problems"
                )
        else:
            lines.append("  No problems detected!")
        lines.append("")

        # Quality Scores (if evaluated)
        if report.get("quality") and report["quality"].get("evaluated"):
            lines.append("-" * 70)
            lines.append("LLM QUALITY EVALUATION")
            lines.append("-" * 70)
            quality = report["quality"]
            qstats = quality["statistics"]
            lines.append(
                f"  Mean Coherence:           {qstats.get('mean_coherence', 0):.1f}/100"
            )
            lines.append(
                f"  Mean Completeness:        {qstats.get('mean_completeness', 0):.1f}/100"
            )
            lines.append(
                f"  Mean Reference Resolution:{qstats.get('mean_reference_resolution', 0):.1f}/100"
            )
            lines.append(
                f"  Mean Redundancy:          {qstats.get('mean_redundancy', 0):.1f}/100"
            )
            lines.append(
                f"  Mean Overall:             {qstats.get('mean_overall', 0):.1f}/100"
            )
            lines.append("")

            if quality.get("worst_chunks"):
                lines.append("  Lowest scoring chunks:")
                for wc in quality["worst_chunks"][:5]:
                    lines.append(
                        f"    - Chunk {wc['chunk_id'][:8]}...: {wc['overall']:.0f}/100"
                    )
                    if wc.get("reasoning"):
                        lines.append(f"      Reason: {wc['reasoning'][:100]}")
            lines.append("")

        # Recommendations
        lines.append("-" * 70)
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 70)
        for i, rec in enumerate(report["recommendations"], 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")

        # Chunk Content
        lines.append("-" * 70)
        lines.append("CHUNK CONTENT")
        lines.append("-" * 70)
        for chunk in report.get("chunks", []):
            lines.append(
                f"Chunk {chunk['ord']} (ID: {chunk['chunk_id']}) - Length: {chunk['text_length']}"
            )
            lines.append("-" * 20)
            lines.append(chunk["text"])
            lines.append("")
            lines.append("=" * 70)
        lines.append("=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)

        return "\n".join(lines)
