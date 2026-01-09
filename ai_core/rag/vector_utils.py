"""Vector normalization and formatting helpers for RAG."""

from __future__ import annotations

import math
import time
import re
import struct
from array import array
from typing import Callable, Dict, Sequence

from common.logging import get_log_context, get_logger

from .embeddings import EmbeddingClientError, get_embedding_client
from .normalization import normalise_text

__all__ = [
    "_coerce_vector_values",
    "_normalise_vector",
    "embed_query",
    "format_vector",
    "format_vector_lenient",
]

logger = get_logger(__name__)

_ZERO_EPSILON = 1e-12


def _normalise_vector(values: Sequence[float] | None) -> list[float] | None:
    """Scale ``values`` to unit length if possible."""

    if not values:
        return None
    try:
        floats = [float(value) for value in values]
    except (TypeError, ValueError):
        return None

    norm_sq = math.fsum(value * value for value in floats)
    if norm_sq <= _ZERO_EPSILON:
        return None

    norm = math.sqrt(norm_sq)
    if not math.isfinite(norm) or norm <= _ZERO_EPSILON:
        return None

    scale = 1.0 / norm
    return [value * scale for value in floats]


def _coerce_vector_values(value: object) -> list[float] | None:
    """Attempt to coerce ``value`` into a list of floats."""

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1].strip()
        if not stripped:
            return []
        parts = [component for component in re.split(r"[\s,]+", stripped) if component]
        if not parts:
            return []
        try:
            return [float(component) for component in parts]
        except (TypeError, ValueError):
            return None
    if isinstance(value, memoryview):
        view = value
        if view.ndim == 1 and view.format in {"f", "d"}:
            try:
                return [float(component) for component in view]
            except (TypeError, ValueError):
                return None
        return _coerce_vector_values(view.tobytes())
    if isinstance(value, (bytes, bytearray)):
        data = bytes(value)
        if len(data) >= 2:
            dimension = struct.unpack("!H", data[:2])[0]
            payload = data[2:]
            if dimension == 0 and payload:
                return None
            for format_char in ("f", "d"):
                component_size = struct.calcsize(f"!{format_char}")
                expected_length = dimension * component_size
                if dimension == 0 and not payload:
                    return []
                if len(payload) != expected_length:
                    continue
                try:
                    unpacked = struct.unpack(f"!{dimension}{format_char}", payload)
                except struct.error:
                    continue
                return [float(component) for component in unpacked]
        for typecode in ("f", "d"):
            try:
                arr = array(typecode)
                arr.frombytes(data)
            except (ValueError, OverflowError):
                continue
            return [float(component) for component in arr]
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        try:
            return [float(component) for component in value]
        except (TypeError, ValueError):
            return None
    tolist = getattr(value, "tolist", None)
    if callable(tolist):  # pragma: no branch - defensive
        try:
            converted = tolist()
        except Exception:  # pragma: no cover - defensive
            return None
        if isinstance(converted, Sequence) and not isinstance(
            converted, (str, bytes, bytearray)
        ):
            try:
                return [float(component) for component in converted]
            except (TypeError, ValueError):
                return None
    values_attr = getattr(value, "values", None)
    if isinstance(values_attr, Sequence) and not isinstance(
        values_attr, (str, bytes, bytearray)
    ):
        try:
            return [float(component) for component in values_attr]
        except (TypeError, ValueError):
            return None
    return None


def format_vector(values: Sequence[float], *, expected_dim: int) -> str:
    floats = [float(v) for v in values]
    if len(floats) != expected_dim:
        raise ValueError("Embedding dimension mismatch")
    return "[" + ",".join(f"{value:.6f}" for value in floats) + "]"


def format_vector_lenient(values: Sequence[float]) -> str:
    """Format a vector without enforcing provider dimension."""

    floats = [float(v) for v in values]
    return "[" + ",".join(f"{value:.6f}" for value in floats) + "]"


def embed_query(
    query: str, *, log=None, get_client: Callable[[], object] | None = None
) -> list[float]:
    client = get_client() if get_client is not None else get_embedding_client()
    normalised = normalise_text(query)
    text = normalised or ""
    started = time.perf_counter()
    result = client.embed([text])
    duration_ms = (time.perf_counter() - started) * 1000

    if not result.vectors:
        raise EmbeddingClientError("Embedding provider returned no vectors")
    vector = result.vectors[0]
    if not isinstance(vector, list):
        vector = list(vector)
    try:
        vector = [float(value) for value in vector]
    except (TypeError, ValueError) as exc:
        raise EmbeddingClientError(
            "Embedding vector contains non-numeric values"
        ) from exc
    try:
        expected_dim = client.dim()
    except EmbeddingClientError:
        expected_dim = len(vector)
    if len(vector) != expected_dim:
        raise EmbeddingClientError(
            "Embedding dimension mismatch between query and provider"
        )

    normalised_vector = _normalise_vector(vector)
    if normalised_vector is None:
        vector = [0.0 for _ in vector]
    else:
        vector = normalised_vector

    context = get_log_context()
    tenant_id = context.get("tenant")
    extra: Dict[str, object] = {
        "tenant_id": tenant_id or "-",
        "len_text": len(text),
        "model_name": result.model,
        "model_used": result.model_used,
        "duration_ms": duration_ms,
        "attempts": result.attempts,
    }
    timeout_s = result.timeout_s
    if timeout_s is not None:
        extra["timeout_s"] = timeout_s
    key_alias = context.get("key_alias")
    if key_alias:
        extra["key_alias"] = key_alias
    active_logger = log if log is not None else logger
    active_logger.info("rag.query.embed", extra=extra)
    return vector
