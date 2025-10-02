"""LiteLLM-backed embedding utilities."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from django.conf import settings

from ai_core.infra.config import get_config
from common.logging import get_logger

try:  # pragma: no cover - optional dependency for tests
    from litellm import embedding as litellm_embedding  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    litellm_embedding = None


logger = get_logger(__name__)


class EmbeddingClientError(RuntimeError):
    """Base class for embedding client errors."""


class EmbeddingProviderUnavailable(EmbeddingClientError):
    """Raised when LiteLLM dependency is unavailable."""


@dataclass(slots=True)
class EmbeddingBatchResult:
    """Return value for embedding batches."""

    vectors: List[List[float]]
    model: str
    model_used: str
    attempts: int
    timeout_s: float | None


class EmbeddingClient:
    """Client that proxies embedding calls to LiteLLM."""

    def __init__(
        self,
        *,
        provider: str,
        primary_model: str,
        fallback_model: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self._provider = provider
        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self._batch_size = max(1, int(batch_size))
        self._dim: int | None = None
        self._dim_model: str | None = None

    @classmethod
    def from_settings(cls) -> "EmbeddingClient":
        provider = getattr(settings, "EMBEDDINGS_PROVIDER", "litellm")
        primary = getattr(settings, "EMBEDDINGS_MODEL_PRIMARY", "")
        fallback = getattr(settings, "EMBEDDINGS_MODEL_FALLBACK", None)
        batch_size = getattr(settings, "EMBEDDINGS_BATCH_SIZE", 64)
        if not primary:
            raise EmbeddingClientError("Primary embedding model not configured")
        return cls(
            provider=provider,
            primary_model=primary,
            fallback_model=fallback,
            batch_size=batch_size,
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def dim(self) -> int:
        if self._dim is not None:
            return self._dim
        result = self.embed([" "])
        if not result.vectors:
            raise EmbeddingClientError("Empty embedding response")
        # ``embed`` call above updates ``self._dim`` via ``_record_dimension``.
        if self._dim is None:
            # Defensive fallback when the provider returned vectors but the
            # recording logic did not run (e.g. future refactors).
            self._record_dimension(result.model, len(result.vectors[0]))
        return self._dim

    def embed(self, texts: Sequence[str]) -> EmbeddingBatchResult:
        if not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence")
        inputs: List[str] = [
            text if isinstance(text, str) else str(text) for text in texts
        ]
        if not inputs:
            return EmbeddingBatchResult(
                vectors=[],
                model=self._primary_model,
                model_used="primary",
                attempts=1,
                timeout_s=None,
            )

        last_error: Exception | None = None
        model_candidates = self._candidate_models()
        cfg = get_config()
        timeout = cfg.timeouts.get("embeddings")
        timeout_s = self._normalise_timeout(timeout)

        for attempt, model in enumerate(model_candidates, start=1):
            try:
                vectors = self._invoke_provider(model, inputs, cfg, timeout)
            except Exception as exc:  # pragma: no cover - requires network failure
                last_error = exc
                should_retry = attempt < len(
                    model_candidates
                ) and self._should_use_fallback(exc)
                status_code = self._extract_status_code(exc)
                logger.warning(
                    "embeddings.batch_failed",
                    model=model,
                    exc_type=exc.__class__.__name__,
                    exc_message=str(exc),
                    status_code=status_code,
                    attempt=attempt,
                    retry=should_retry,
                )
                if should_retry:
                    continue
                break

            if vectors:
                self._record_dimension(model, len(vectors[0]))
            model_used = "fallback" if attempt > 1 else "primary"
            return EmbeddingBatchResult(
                vectors=vectors,
                model=model,
                model_used=model_used,
                attempts=attempt,
                timeout_s=timeout_s,
            )

        message = "Embedding request failed"
        if last_error is not None:
            message = f"{message}: {last_error}"
        raise EmbeddingClientError(message)

    def _candidate_models(self) -> Tuple[str, ...]:
        if self._fallback_model and self._fallback_model != self._primary_model:
            return (self._primary_model, self._fallback_model)
        return (self._primary_model,)

    def _record_dimension(self, model: str, current_dim: int) -> None:
        """Persist the latest embedding dimensionality for ``model``."""

        previous_dim = self._dim
        previous_model = self._dim_model
        if previous_dim is None or previous_dim != current_dim:
            if previous_dim is not None and previous_dim != current_dim:
                logger.info(
                    "embeddings.dimension_changed",
                    model=model,
                    previous_dim=previous_dim,
                    current_dim=current_dim,
                    previous_model=previous_model,
                )
            self._dim = current_dim
        self._dim_model = model

    def _invoke_provider(
        self,
        model: str,
        inputs: Sequence[str],
        cfg,
        timeout,
    ) -> List[List[float]]:
        if litellm_embedding is None:
            raise EmbeddingProviderUnavailable("litellm package is not installed")

        api_base = cfg.litellm_base_url.rstrip("/") + "/v1"
        kwargs = {
            "model": model,
            "input": list(inputs),
            "api_base": api_base,
            "api_key": cfg.litellm_api_key,
        }
        if timeout:
            kwargs["timeout"] = timeout

        response = litellm_embedding(**kwargs)
        data = getattr(response, "data", None)
        if not isinstance(data, Sequence):
            raise EmbeddingClientError("Unexpected embedding response structure")

        vectors: List[List[float]] = []
        for item in data:
            embedding_values = getattr(item, "embedding", None)
            if embedding_values is None and isinstance(item, dict):
                embedding_values = item.get("embedding")
            if embedding_values is None:
                raise EmbeddingClientError("Embedding item missing values")
            try:
                vector = [float(value) for value in embedding_values]
            except (TypeError, ValueError) as exc:
                raise EmbeddingClientError("Invalid embedding value") from exc
            if not vector:
                raise EmbeddingClientError("Embedding vector is empty")
            vectors.append(vector)

        if len(vectors) != len(inputs):
            raise EmbeddingClientError("Embedding count mismatch")
        if vectors:
            expected_len = len(vectors[0])
            for vector in vectors[1:]:
                if len(vector) != expected_len:
                    raise EmbeddingClientError("Embedding dimensions mismatch in batch")
        return vectors

    @staticmethod
    def _normalise_timeout(timeout: object | None) -> float | None:
        if timeout is None:
            return None
        if isinstance(timeout, (int, float)):
            return float(timeout)
        text = str(timeout)
        try:
            return float(text)
        except (TypeError, ValueError):
            if hasattr(timeout, "total"):  # httpx.Timeout like
                total = getattr(timeout, "total")
                if isinstance(total, (int, float)):
                    return float(total)
            return None

    @staticmethod
    def _extract_status_code(exc: Exception) -> int | None:
        for attr in ("status_code", "status", "code"):
            value = getattr(exc, attr, None)
            if isinstance(value, int):
                return value
            try:
                if value is not None and str(value).isdigit():
                    return int(value)
            except Exception:
                continue
        response = getattr(exc, "response", None)
        if response is not None:
            for attr in ("status_code", "status"):
                value = getattr(response, attr, None)
                if isinstance(value, int):
                    return value
        return None

    def _should_use_fallback(self, exc: Exception) -> bool:
        status_code = self._extract_status_code(exc)
        if status_code is not None:
            if status_code == 429 or 500 <= status_code < 600:
                return True
        exc_name = exc.__class__.__name__.lower()
        if "timeout" in exc_name:
            return True
        if isinstance(exc, TimeoutError):
            return True
        message = str(exc)
        if message and "timeout" in message.lower():
            return True
        return False


_default_client: EmbeddingClient | None = None


def get_embedding_client() -> EmbeddingClient:
    global _default_client
    if _default_client is None:
        _default_client = EmbeddingClient.from_settings()
    return _default_client


def reset_embedding_client() -> None:
    global _default_client
    _default_client = None
