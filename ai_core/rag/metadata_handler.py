"""Document metadata normalization helpers for RAG ingestion."""

from __future__ import annotations

import os
import uuid
from typing import Any, Mapping, MutableMapping, Sequence

from psycopg2 import sql
from psycopg2.extras import Json

from common.logging import get_logger

from .schemas import Chunk

logger = get_logger(__name__)


def _is_dev_environment() -> bool:
    """Return ``True`` when the current deployment resembles a dev setup."""

    env_value = (
        os.getenv("DEPLOY_ENV") or os.getenv("DEPLOYMENT_ENVIRONMENT") or ""
    ).strip()
    env_normalised = env_value.lower()
    if not env_normalised:
        return True
    return env_normalised in {
        "dev",
        "development",
        "local",
        "unknown",
    } or env_normalised.startswith("dev")


class MetadataHandler:
    """Centralized document metadata normalization and validation."""

    @staticmethod
    def coerce_map(metadata: object) -> dict[str, object]:
        """Return a shallow ``dict`` copy when *metadata* behaves like a mapping."""

        if isinstance(metadata, MutableMapping):
            return dict(metadata)
        if isinstance(metadata, Mapping):
            return dict(metadata)
        return {}

    @staticmethod
    def normalise_parent_nodes_value(value: object) -> object:
        """Return a canonical representation of ``parent_nodes`` payloads."""

        if value is None:
            return None
        if isinstance(value, Mapping):
            items = sorted(value.items(), key=lambda item: str(item[0]))
            return {
                str(key): MetadataHandler.normalise_parent_nodes_value(payload)
                for key, payload in items
            }
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [
                MetadataHandler.normalise_parent_nodes_value(item) for item in value
            ]
        if isinstance(value, uuid.UUID):
            return str(value)
        if isinstance(value, (bytes, bytearray)):
            try:
                return value.decode("utf-8")
            except Exception:
                return value.hex()
        return value

    @staticmethod
    def parent_nodes_differ(
        existing: Mapping[str, object] | None,
        desired: Mapping[str, object] | None,
    ) -> bool:
        """Return ``True`` when ``parent_nodes`` differ between both metadata maps."""

        existing_value = None
        desired_value = None
        if isinstance(existing, Mapping):
            existing_value = existing.get("parent_nodes")
        if isinstance(desired, Mapping):
            desired_value = desired.get("parent_nodes")
        return MetadataHandler.normalise_parent_nodes_value(
            existing_value
        ) != MetadataHandler.normalise_parent_nodes_value(desired_value)

    @staticmethod
    def normalise_hash_value(value: object | None) -> str | None:
        """Return a canonical string representation for hash-like values."""

        if value in {None, ""}:
            return None
        try:
            text = str(value).strip()
        except Exception:
            return None
        return text or None

    @staticmethod
    def metadata_content_hash(metadata: Mapping[str, object] | None) -> str | None:
        """Extract the ``content_hash``/``hash`` value from *metadata* if present."""

        if not isinstance(metadata, Mapping):
            return None
        for key in ("content_hash", "hash"):
            candidate = MetadataHandler.normalise_hash_value(metadata.get(key))
            if candidate:
                return candidate
        return None

    @staticmethod
    def content_hash_matches(
        existing: Mapping[str, object] | None,
        desired: Mapping[str, object] | None,
        *,
        existing_fallback: object | None = None,
        desired_fallback: object | None = None,
    ) -> bool:
        """Return ``True`` when both metadata maps reference the same content hash."""

        existing_hash = MetadataHandler.metadata_content_hash(existing)
        desired_hash = MetadataHandler.metadata_content_hash(desired)
        if existing_hash is None:
            existing_hash = MetadataHandler.normalise_hash_value(existing_fallback)
        if desired_hash is None:
            desired_hash = MetadataHandler.normalise_hash_value(desired_fallback)
        if not existing_hash or not desired_hash:
            return False
        return existing_hash == desired_hash

    @staticmethod
    def identity_needs_repair(
        metadata: Mapping[str, object] | None, canonical_id: str
    ) -> bool:
        """Return ``True`` if ``metadata`` lacks or mismatches the canonical ID."""

        if not isinstance(metadata, Mapping):
            return True

        value = metadata.get("document_id")
        if value in {None, ""}:
            return True
        try:
            value_text = str(value).strip()
        except Exception:
            return True
        if not value_text:
            return True
        return value_text != canonical_id

    @staticmethod
    def normalise_document_identity(
        doc: MutableMapping[str, object],
        resolved_id: uuid.UUID,
        metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Ensure ``doc`` and its nested structures reference ``resolved_id``."""

        # Local, fault-tolerant import to support partial runtimes and tests
        try:  # pragma: no cover - exercised indirectly via tests
            from .parents import limit_parent_payload as _limit_parent_payload  # type: ignore
        except Exception:  # pragma: no cover - defensive fallback
            _limit_parent_payload = None  # type: ignore[assignment]

        canonical_id = str(resolved_id)
        doc["id"] = resolved_id

        raw_metadata = (
            metadata if isinstance(metadata, Mapping) else doc.get("metadata")
        )
        if isinstance(raw_metadata, MutableMapping):
            metadata_dict = raw_metadata
        elif isinstance(raw_metadata, Mapping):
            metadata_dict = dict(raw_metadata)
        else:
            metadata_dict = {}
        metadata_dict["document_id"] = canonical_id
        doc["metadata"] = metadata_dict

        # Ensure a parents map is available even when upstream calls omit doc-level parents.
        parents_map = doc.get("parents")
        if not isinstance(parents_map, Mapping) or not parents_map:
            derived_parents: dict[str, object] = {}
            chunks_for_parents = doc.get("chunks")
            if isinstance(chunks_for_parents, Sequence):
                for chunk in chunks_for_parents:
                    chunk_parents = getattr(chunk, "parents", None)
                    if not isinstance(chunk_parents, Mapping):
                        # No structured parents on the chunk; try deriving
                        # a minimal root parent entry from parent_ids in meta.
                        try:
                            chunk_meta = getattr(chunk, "meta", None)
                            parent_ids = None
                            if isinstance(chunk_meta, Mapping):
                                candidate = chunk_meta.get("parent_ids")
                                if isinstance(candidate, Sequence) and not isinstance(
                                    candidate, (str, bytes, bytearray)
                                ):
                                    parent_ids = [str(pid) for pid in candidate if pid]
                            if parent_ids and any(
                                isinstance(pid, str) and pid.endswith("#doc")
                                for pid in parent_ids
                            ):
                                root_key = f"{canonical_id}#doc"
                                derived_parents[root_key] = {
                                    "document_id": canonical_id
                                }
                        except Exception:
                            pass
                        continue
                    for parent_id, payload in chunk_parents.items():
                        if isinstance(payload, Mapping):
                            derived_parents[str(parent_id)] = dict(payload)
                        else:
                            derived_parents[str(parent_id)] = payload

            if derived_parents:
                root_key = f"{canonical_id}#doc"
                alt_root_payload: object | None = None
                canonical_present = False
                for existing_key in list(derived_parents.keys()):
                    key_text = str(existing_key)
                    if not key_text.endswith("#doc"):
                        continue
                    if key_text == root_key:
                        canonical_present = True
                        continue
                    if alt_root_payload is None:
                        alt_root_payload = derived_parents[existing_key]
                    derived_parents.pop(existing_key, None)

                if not canonical_present:
                    if isinstance(alt_root_payload, Mapping):
                        root_payload = dict(alt_root_payload)
                    else:
                        root_payload = {}
                    root_payload.setdefault("document_id", canonical_id)
                    derived_parents[root_key] = root_payload
                elif root_key not in derived_parents:
                    derived_parents[root_key] = {"document_id": canonical_id}

                doc["parents"] = derived_parents

        chunks = doc.get("chunks")
        if isinstance(chunks, Sequence):
            updated_chunks: list[Chunk | MutableMapping[str, Any]] = []
            changed = False
            for chunk in chunks:
                updated_chunk = chunk
                if isinstance(chunk, Chunk):
                    if chunk.meta.get("document_id") != canonical_id:
                        updated_chunk = chunk.model_copy(
                            update={"meta": {**chunk.meta, "document_id": canonical_id}}
                        )
                        changed = True
                elif isinstance(chunk, MutableMapping):
                    if chunk.get("document_id") != canonical_id:
                        chunk["document_id"] = canonical_id
                        changed = True
                updated_chunks.append(updated_chunk)
            if changed:
                doc["chunks"] = updated_chunks

        for parent_key in ("parents", "parent_nodes"):
            parent_map = doc.get(parent_key)
            if not isinstance(parent_map, Mapping):
                continue
            normalised_parents: dict[str, object] = {}
            for parent_id, payload in parent_map.items():
                if isinstance(payload, Mapping):
                    parent_payload = dict(payload)
                    parent_payload["document_id"] = canonical_id
                else:
                    parent_payload = payload
                normalised_parents[parent_id] = parent_payload
            doc[parent_key] = normalised_parents

        # Ensure parent_nodes are present in metadata when parent mappings exist
        parents_map = doc.get("parents") or doc.get("parent_nodes")
        if not (isinstance(parents_map, Mapping) and parents_map):
            # As a last resort, derive a minimal parent_nodes map containing
            # the document root from chunk meta's parent_ids if available.
            try:
                chunks_for_parents = doc.get("chunks")
                root_detected = False
                if isinstance(chunks_for_parents, Sequence):
                    for chunk in chunks_for_parents:
                        chunk_meta = getattr(chunk, "meta", None)
                        if not isinstance(chunk_meta, Mapping):
                            continue
                        ids_value = chunk_meta.get("parent_ids")
                        if isinstance(ids_value, Sequence) and not isinstance(
                            ids_value, (str, bytes, bytearray)
                        ):
                            if any(
                                isinstance(pid, str) and pid.endswith("#doc")
                                for pid in ids_value
                            ):
                                root_detected = True
                                break
                if root_detected:
                    root_key = f"{canonical_id}#doc"
                    parents_map = {root_key: {"document_id": canonical_id}}
            except Exception:
                parents_map = parents_map
        if isinstance(parents_map, Mapping) and parents_map:
            limited_parents: Mapping[str, object] | None = None
            if _limit_parent_payload is not None:
                try:
                    limited_parents = _limit_parent_payload(parents_map)
                except Exception:
                    limited_parents = None
            if not limited_parents:
                limited_parents = {
                    parent_id: (
                        dict(payload) if isinstance(payload, Mapping) else payload
                    )
                    for parent_id, payload in parents_map.items()
                }
            metadata_dict["parent_nodes"] = dict(limited_parents)

        return metadata_dict

    @staticmethod
    def repair_persisted_metadata(
        cur,
        *,
        documents_table: sql.Identifier,
        tenant_uuid: uuid.UUID,
        document_id: uuid.UUID,
        existing_metadata: object,
        desired_metadata: Mapping[str, object] | None,
        log: object | None = None,
    ) -> None:
        canonical_id = str(document_id)
        existing_map = (
            dict(existing_metadata) if isinstance(existing_metadata, Mapping) else {}
        )

        # Decide whether identity or parent_nodes require repair
        needs_identity_fix = MetadataHandler.identity_needs_repair(
            existing_map, canonical_id
        )
        desired_map = (
            dict(desired_metadata) if isinstance(desired_metadata, Mapping) else {}
        )
        needs_parent_nodes_fix = MetadataHandler.parent_nodes_differ(
            existing_map, desired_map
        )

        if not needs_identity_fix and not needs_parent_nodes_fix:
            return

        stored_document_id = existing_map.get("document_id")
        extra = {
            "tenant_id": str(tenant_uuid),
            "document_id": canonical_id,
            "stored_document_id": (
                str(stored_document_id)
                if stored_document_id not in {None, ""}
                else None
            ),
            "parent_nodes_repair": bool(needs_parent_nodes_fix),
        }
        active_logger = log if log is not None else logger

        # In dev/test environments, proactively repair persisted metadata to reduce flakiness
        if _is_dev_environment():
            payload: dict[str, object] = dict(existing_map)
            if desired_map:
                # Merge desired fields but keep existing keys unless overwritten
                payload.update(desired_map)
            # Always enforce the canonical document identifier
            payload["document_id"] = canonical_id
            try:
                cur.execute(
                    sql.SQL(
                        """
                        UPDATE {}
                        SET metadata = %s
                        WHERE id = %s
                        """
                    ).format(documents_table),
                    (Json(payload), document_id),
                )
            except Exception:
                try:
                    active_logger.exception(  # type: ignore[call-arg]
                        "ingestion.doc.metadata_repair_failed",
                        extra=extra,
                    )
                except Exception:
                    pass
            else:
                try:
                    active_logger.info(  # type: ignore[call-arg]
                        "ingestion.doc.metadata_repaired", extra=extra
                    )
                except Exception:
                    pass
                return

        try:
            active_logger.warning(  # type: ignore[call-arg]
                "ingestion.doc.metadata_repair_required",
                extra=extra,
            )
        except Exception:
            pass
