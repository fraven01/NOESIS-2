"""Multimodal captioning interfaces and asset enrichment pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
from uuid import UUID

from common.logging import get_logger, log_context

from .contract_utils import (
    is_image_mediatype,
    normalize_optional_string,
    truncate_text,
)
from .contracts import Asset, FileBlob, InlineBlob, NormalizedDocument
from .logging_utils import (
    asset_log_fields,
    document_log_fields,
    log_call,
    log_extra_entry,
    log_extra_exit,
)
from .pipeline import DocumentPipelineConfig
from .policies import DocumentPolicy, PolicyProvider, get_policy
from .repository import DocumentsRepository
from .storage import Storage
from ai_core.infra.observability import report_generation_usage
from ai_core.infra.usage import Usage


logger = get_logger(__name__)


class MultimodalCaptioner:
    """Abstract interface for generating captions from image payloads."""

    def caption(self, image: bytes, context: Optional[str] = None) -> Dict[str, object]:
        """Return caption metadata for ``image`` given optional ``context``."""

        raise NotImplementedError


class DeterministicCaptioner(MultimodalCaptioner):
    """Deterministic stub captioner producing repeatable descriptions."""

    def __init__(self, model_name: str = "stub-vlm") -> None:
        self._model_name = model_name

    def caption(self, image: bytes, context: Optional[str] = None) -> Dict[str, object]:
        digest = hashlib.sha256(image).hexdigest()
        context_digest = hashlib.sha1((context or "").encode("utf-8")).hexdigest()
        base_text = f"stub caption {digest[:12]} ctx-{context_digest[:8]}"
        text = truncate_text(base_text, 2048) or ""
        confidence = int(digest[:2], 16) / 255
        tokens = len(image) // 4 if image else 0
        return {
            "text_description": text,
            "confidence": confidence,
            "model": self._model_name,
            "tokens": tokens,
        }


@dataclass
class AssetExtractionPipeline:
    """Pipeline orchestrating caption enrichment for document assets."""

    repository: DocumentsRepository
    storage: Storage
    captioner: MultimodalCaptioner
    config: DocumentPipelineConfig = field(default_factory=DocumentPipelineConfig)
    context_separator: str = " \u2026 "
    context_limit: int = 512
    strict_caption_validation: bool = False
    policy_provider: PolicyProvider = get_policy

    @log_call("pipeline.assets_caption")
    def process_document(self, document: NormalizedDocument) -> NormalizedDocument:
        """Caption eligible assets and persist the updated document."""

        ref = document.ref
        with log_context(
            tenant=ref.tenant_id,
            collection_id=str(ref.collection_id) if ref.collection_id else None,
        ):
            log_extra_entry(**document_log_fields(document))
            doc_copy = document.model_copy(deep=True)
            processed_assets = [self._process_asset(asset) for asset in doc_copy.assets]
            # Persist updated assets to the repository to keep the document's
            # asset view and the asset store in sync.
            for asset in processed_assets:
                try:
                    self.repository.add_asset(asset)
                except Exception:
                    # Best-effort: if persistence fails for an asset, continue
                    # to upsert the document itself; detailed errors are logged
                    # within the repository layer.
                    pass
            doc_copy.assets = processed_assets
            log_extra_exit(processed_assets=len(processed_assets))
            return self.repository.upsert(doc_copy)

    @log_call("assets.caption.process_assets")
    def process_assets(
        self,
        tenant_id: str,
        document_id: UUID,
        assets: Sequence[Asset],
    ) -> List[Asset]:
        """Caption and persist provided assets for an existing document."""

        with log_context(tenant=tenant_id):
            log_extra_entry(
                tenant_id=tenant_id,
                document_id=document_id,
                asset_count=len(assets),
            )
            processed: List[Asset] = []
            for asset in assets:
                if asset.ref.tenant_id != tenant_id:
                    raise ValueError("asset_tenant_mismatch")
                if asset.ref.document_id != document_id:
                    raise ValueError("asset_document_mismatch")
                processed_asset = self._process_asset(asset)
                stored = self.repository.add_asset(processed_asset)
                processed.append(stored)
            log_extra_exit(processed_assets=len(processed))
            return processed

    @log_call("pipeline.assets_caption.item")
    def _process_asset(self, asset: Asset) -> Asset:
        ref = asset.ref
        with log_context(tenant=ref.tenant_id, workflow_id=ref.workflow_id):
            log_extra_entry(**asset_log_fields(asset))
            description = normalize_optional_string(asset.text_description)
            if description:
                log_extra_exit(
                    caption_method=asset.caption_method, skipped="has_description"
                )
                return asset
            if not is_image_mediatype(asset.media_type):
                log_extra_exit(
                    caption_method=asset.caption_method, skipped="media_type"
                )
                return asset

            payload = self._load_payload(asset)
            context = self._build_context(asset)
            result = self._caption_asset(asset, payload, context)

            text_result = normalize_optional_string(result.get("text_description"))
            text = truncate_text(text_result, 2048)
            confidence = result.get("confidence")
            model = normalize_optional_string(result.get("model"))

            if text:
                error: Optional[ValueError] = None
                confidence_value: Optional[float] = None
                if confidence is None or not isinstance(confidence, (float, int)):
                    error = ValueError("caption_confidence_missing")
                else:
                    confidence_value = float(confidence)
                    if confidence_value < 0 or confidence_value > 1:
                        error = ValueError("caption_confidence_range")
                policy: Optional[DocumentPolicy] = None
                threshold = self.config.caption_min_confidence(ref.collection_id)
                if not error:
                    policy = self.policy_provider(
                        ref.tenant_id,
                        ref.collection_id,
                        ref.workflow_id,
                    )
                    assert policy is not None
                    threshold = max(threshold, policy.caption_min_confidence)
                    assert confidence_value is not None
                    if confidence_value < threshold:
                        error = ValueError("caption_confidence_policy")
                if not error and not model:
                    error = ValueError("caption_model_missing")

                if error:
                    if self.strict_caption_validation:
                        raise error
                    logger.warning(
                        "assets.caption.metadata_invalid",
                        error=str(error),
                        asset_id=str(ref.asset_id),
                    )
                else:
                    assert confidence_value is not None
                    log_extra_exit(
                        caption_method="vlm_caption",
                        model=model,
                        caption_confidence=confidence_value,
                    )
                    return asset.model_copy(
                        update={
                            "text_description": text,
                            "caption_method": "vlm_caption",
                            "caption_model": model,
                            "caption_confidence": confidence_value,
                            "caption_source": "vlm",
                        }
                    )

            if asset.ocr_text:
                fallback = truncate_text(asset.ocr_text, 2048)
                if fallback:
                    log_extra_exit(caption_method="ocr_only")
                    return asset.model_copy(
                        update={
                            "text_description": fallback,
                            "caption_method": "ocr_only",
                            "caption_model": None,
                            # For OCR-only fallback, do not report a model confidence
                            # to avoid conflating OCR presence with caption quality.
                            "caption_confidence": None,
                            "caption_source": "ocr",
                        }
                    )

            log_extra_exit(caption_method=asset.caption_method)
            return asset

    def _build_context(self, asset: Asset) -> Optional[str]:
        parts: List[str] = []
        if asset.context_before:
            before = truncate_text(asset.context_before, self.context_limit)
            if before:
                parts.append(before)
        if asset.context_after:
            after = truncate_text(asset.context_after, self.context_limit)
            if after:
                parts.append(after)
        if not parts:
            return None
        return self.context_separator.join(parts)

    @log_call("assets.caption.process_collection")
    def process_collection(
        self,
        tenant_id: str,
        collection_id: UUID,
        *,
        limit: int = 100,
        cursor: Optional[str] = None,
        latest_only: bool = True,
    ) -> Tuple[List[Asset], Optional[str]]:
        """Caption a collection page and return updated assets with next cursor."""

        with log_context(tenant=tenant_id, collection_id=str(collection_id)):
            log_extra_entry(
                tenant_id=tenant_id,
                collection_id=collection_id,
                limit=limit,
                cursor_present=bool(cursor),
                latest_only=latest_only,
            )
            refs, next_cursor = self.repository.list_by_collection(
                tenant_id,
                collection_id,
                limit=limit,
                cursor=cursor,
                latest_only=latest_only,
            )

            updated: List[Asset] = []
            for ref_doc in refs:
                document = self.repository.get(
                    tenant_id,
                    ref_doc.document_id,
                    ref_doc.version,
                    workflow_id=ref_doc.workflow_id,
                )
                if document is None:
                    continue

                before_assets = {a.ref.asset_id: a for a in document.assets}
                stored = self.process_document(document)
                for asset in stored.assets:
                    previous = before_assets.get(asset.ref.asset_id)
                    if previous is None and asset.text_description:
                        updated.append(asset)
                    elif (
                        previous and previous.text_description != asset.text_description
                    ):
                        updated.append(asset)

            log_extra_exit(
                processed_assets=len(updated), next_cursor_present=bool(next_cursor)
            )
            return updated, next_cursor

    @log_call("assets.caption.load_payload")
    def _load_payload(self, asset: Asset) -> bytes:
        blob = asset.blob
        with log_context(tenant=asset.ref.tenant_id):
            log_extra_entry(**asset_log_fields(asset))
            if isinstance(blob, InlineBlob):
                payload = blob.decoded_payload()
                log_extra_exit(size_bytes=len(payload))
                return payload
            if isinstance(blob, FileBlob):
                try:
                    payload = self.storage.get(blob.uri)
                except (KeyError, ValueError) as exc:
                    raise ValueError("blob_missing") from exc
                log_extra_exit(size_bytes=len(payload))
                return payload
            raise ValueError("blob_unsupported")

    @log_call("assets.caption.run")
    def _caption_asset(
        self, asset: Asset, payload: bytes, context: Optional[str]
    ) -> Dict[str, object]:
        ref = asset.ref
        with log_context(tenant=ref.tenant_id):
            log_extra_entry(**asset_log_fields(asset))
            result = self.captioner.caption(payload, context)
            model = normalize_optional_string(result.get("model"))
            confidence = result.get("confidence")
            extra: Dict[str, object] = {}
            if model:
                extra["model"] = model
            if isinstance(confidence, (int, float)):
                extra["caption_confidence"] = float(confidence)
            if extra:
                log_extra_exit(**extra)
            
            # Telemetry: Report usage
            tokens = result.get("tokens")
            if isinstance(tokens, (int, float)):
                # Heuristic mapping for stub/generic captioners
                # Real VLMs should return structured usage if possible
                # TODO: Enhance VLM usage tracking
                usage = Usage(total_tokens=int(tokens))
                report_generation_usage(usage, model=model)
                
            return result


__all__ = [
    "AssetExtractionPipeline",
    "DeterministicCaptioner",
    "MultimodalCaptioner",
]
