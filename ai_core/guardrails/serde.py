from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Iterable, Mapping

from common.guardrails import FetcherLimits
from crawler.fetcher import FetchFailure, FetchRequest, PolitenessContext

from ai_core.rag.guardrails import (
    GuardrailLimits,
    GuardrailSignals,
    QuotaLimits,
    QuotaUsage,
)


@dataclass(frozen=True)
class GuardrailState:
    config: Mapping[str, Any] | None = None
    limits: GuardrailLimits | None = None
    signals: GuardrailSignals | None = None
    error_builder: Callable[..., Any] | None = None


class GuardrailSerde:
    """Serialize guardrail and fetch payloads for crawler state."""

    @staticmethod
    def serialize_politeness(
        politeness: PolitenessContext | None,
    ) -> dict[str, Any] | None:
        if politeness is None:
            return None
        return {
            "host": politeness.host,
            "slot": politeness.slot,
            "user_agent": politeness.user_agent,
            "crawl_delay": politeness.crawl_delay,
        }

    @staticmethod
    def serialize_fetch_request(
        request: FetchRequest | None,
    ) -> dict[str, Any] | None:
        if request is None:
            return None
        return {
            "canonical_source": request.canonical_source,
            "metadata": dict(request.metadata or {}),
            "politeness": GuardrailSerde.serialize_politeness(request.politeness),
        }

    @staticmethod
    def serialize_fetch_limits(limits: FetcherLimits | None) -> dict[str, Any] | None:
        if limits is None:
            return None
        payload: dict[str, Any] = {}
        if limits.max_bytes is not None:
            payload["max_bytes"] = limits.max_bytes
        if limits.timeout is not None:
            payload["timeout_seconds"] = limits.timeout.total_seconds()
        if limits.mime_whitelist is not None:
            payload["mime_whitelist"] = list(limits.mime_whitelist)
        return payload or None

    @staticmethod
    def serialize_fetch_failure(
        failure: FetchFailure | None,
    ) -> dict[str, Any] | None:
        if failure is None:
            return None
        return {"reason": failure.reason, "temporary": failure.temporary}

    @staticmethod
    def serialize_quota_limits(limits: QuotaLimits | None) -> dict[str, Any] | None:
        if limits is None:
            return None
        payload: dict[str, Any] = {}
        if limits.max_documents is not None:
            payload["max_documents"] = limits.max_documents
        if limits.max_bytes is not None:
            payload["max_bytes"] = limits.max_bytes
        return payload or None

    @staticmethod
    def serialize_quota_usage(usage: QuotaUsage | None) -> dict[str, Any] | None:
        if usage is None:
            return None
        return {"documents": usage.documents, "bytes": usage.bytes}

    @staticmethod
    def serialize_guardrail_limits(
        limits: GuardrailLimits | None,
    ) -> dict[str, Any] | None:
        if limits is None:
            return None
        payload: dict[str, Any] = {}
        if limits.max_document_bytes is not None:
            payload["max_document_bytes"] = limits.max_document_bytes
        if limits.processing_time_limit is not None:
            payload["processing_time_limit_seconds"] = (
                limits.processing_time_limit.total_seconds()
            )
        if limits.mime_blacklist:
            payload["mime_blacklist"] = sorted(limits.mime_blacklist)
        if limits.host_blocklist:
            payload["host_blocklist"] = sorted(limits.host_blocklist)
        tenant_quota = GuardrailSerde.serialize_quota_limits(limits.tenant_quota)
        if tenant_quota is not None:
            payload["tenant_quota"] = tenant_quota
        host_quota = GuardrailSerde.serialize_quota_limits(limits.host_quota)
        if host_quota is not None:
            payload["host_quota"] = host_quota
        return payload or None

    @staticmethod
    def serialize_guardrail_signals(
        signals: GuardrailSignals | None,
    ) -> dict[str, Any] | None:
        if signals is None:
            return None
        payload: dict[str, Any] = {}
        for attr in (
            "tenant_id",
            "provider",
            "canonical_source",
            "host",
            "document_bytes",
            "mime_type",
        ):
            value = getattr(signals, attr, None)
            if value is not None:
                payload[attr] = value
        processing_time = getattr(signals, "processing_time", None)
        if processing_time is not None:
            payload["processing_time_seconds"] = processing_time.total_seconds()
        tenant_usage = GuardrailSerde.serialize_quota_usage(
            getattr(signals, "tenant_usage", None)
        )
        if tenant_usage is not None:
            payload["tenant_usage"] = tenant_usage
        host_usage = GuardrailSerde.serialize_quota_usage(
            getattr(signals, "host_usage", None)
        )
        if host_usage is not None:
            payload["host_usage"] = host_usage
        return payload or None

    @staticmethod
    def to_payload(
        *,
        limits: GuardrailLimits | None,
        signals: GuardrailSignals | None,
        config: Mapping[str, Any] | None = None,
        error_builder: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        serialized_limits = GuardrailSerde.serialize_guardrail_limits(limits)
        payload["limits"] = serialized_limits
        serialized_signals = GuardrailSerde.serialize_guardrail_signals(signals)
        payload["signals"] = serialized_signals
        if config is not None:
            payload["config"] = dict(config)
        if error_builder is not None:
            payload["error_builder"] = error_builder
        return payload

    @staticmethod
    def _coerce_timedelta(value: Any) -> timedelta | None:
        if value is None:
            return None
        if isinstance(value, timedelta):
            return value
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric <= 0:
            return None
        return timedelta(seconds=numeric)

    @staticmethod
    def _deserialize_quota_limits(value: Any) -> QuotaLimits | None:
        if isinstance(value, QuotaLimits):
            return value
        if not isinstance(value, Mapping):
            return None
        max_documents = value.get("max_documents")
        max_bytes = value.get("max_bytes")
        if max_documents is None and max_bytes is None:
            return None
        return QuotaLimits(
            max_documents=int(max_documents) if max_documents is not None else None,
            max_bytes=int(max_bytes) if max_bytes is not None else None,
        )

    @staticmethod
    def _deserialize_quota_usage(value: Any) -> QuotaUsage | None:
        if isinstance(value, QuotaUsage):
            return value
        if not isinstance(value, Mapping):
            return None
        documents = value.get("documents")
        bytes_used = value.get("bytes")
        if documents is None and bytes_used is None:
            return None
        return QuotaUsage(
            documents=int(documents or 0),
            bytes=int(bytes_used or 0),
        )

    @staticmethod
    def _deserialize_guardrail_limits(value: Any) -> GuardrailLimits | None:
        if isinstance(value, GuardrailLimits):
            return value
        if not isinstance(value, Mapping):
            return None
        kwargs: dict[str, Any] = {}
        if value.get("max_document_bytes") is not None:
            kwargs["max_document_bytes"] = int(value["max_document_bytes"])
        limit_seconds = value.get("processing_time_limit_seconds")
        if limit_seconds is None:
            limit_seconds = value.get("processing_time_limit")
        processing_time = GuardrailSerde._coerce_timedelta(limit_seconds)
        if processing_time is not None:
            kwargs["processing_time_limit"] = processing_time
        mime_blacklist = value.get("mime_blacklist")
        if isinstance(mime_blacklist, Iterable):
            kwargs["mime_blacklist"] = frozenset(
                str(entry).strip().lower()
                for entry in mime_blacklist
                if entry not in (None, "")
            )
        host_blocklist = value.get("host_blocklist")
        if isinstance(host_blocklist, Iterable):
            kwargs["host_blocklist"] = frozenset(
                str(entry).strip().lower()
                for entry in host_blocklist
                if entry not in (None, "")
            )
        tenant_quota = GuardrailSerde._deserialize_quota_limits(
            value.get("tenant_quota")
        )
        if tenant_quota is not None:
            kwargs["tenant_quota"] = tenant_quota
        host_quota = GuardrailSerde._deserialize_quota_limits(value.get("host_quota"))
        if host_quota is not None:
            kwargs["host_quota"] = host_quota
        if not kwargs:
            return None
        return GuardrailLimits(**kwargs)

    @staticmethod
    def _deserialize_guardrail_signals(value: Any) -> GuardrailSignals | None:
        if isinstance(value, GuardrailSignals):
            return value
        if not isinstance(value, Mapping):
            return None
        kwargs: dict[str, Any] = {}
        for key in (
            "tenant_id",
            "provider",
            "canonical_source",
            "host",
            "document_bytes",
            "mime_type",
        ):
            if key in value and value[key] is not None:
                kwargs[key] = value[key]
        processing_time = GuardrailSerde._coerce_timedelta(
            value.get("processing_time_seconds") or value.get("processing_time")
        )
        if processing_time is not None:
            kwargs["processing_time"] = processing_time
        tenant_usage = GuardrailSerde._deserialize_quota_usage(
            value.get("tenant_usage")
        )
        if tenant_usage is not None:
            kwargs["tenant_usage"] = tenant_usage
        host_usage = GuardrailSerde._deserialize_quota_usage(value.get("host_usage"))
        if host_usage is not None:
            kwargs["host_usage"] = host_usage
        if not kwargs:
            return None
        return GuardrailSignals(**kwargs)

    @staticmethod
    def from_payload(payload: object) -> GuardrailState:
        if payload is None:
            return GuardrailState()
        if isinstance(payload, GuardrailLimits):
            return GuardrailState(limits=payload)
        if isinstance(payload, GuardrailSignals):
            return GuardrailState(signals=payload)
        if not isinstance(payload, Mapping):
            return GuardrailState()

        limits = GuardrailSerde._deserialize_guardrail_limits(payload.get("limits"))
        signals = GuardrailSerde._deserialize_guardrail_signals(payload.get("signals"))
        error_builder = payload.get("error_builder")
        if not callable(error_builder):
            error_builder = None
        config_candidate = payload.get("config")
        if isinstance(config_candidate, Mapping):
            config = config_candidate
        elif limits is None and signals is None:
            config = payload
        else:
            config = None
        return GuardrailState(
            config=config,
            limits=limits,
            signals=signals,
            error_builder=error_builder,
        )


__all__ = ["GuardrailSerde", "GuardrailState"]
