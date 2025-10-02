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
        self._dim = len(result.vectors[0])
        return self._dim

    def embed(self, texts: Sequence[str]) -> EmbeddingBatchResult:
        if not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence")
        inputs: List[str] = [text if isinstance(text, str) else str(text) for text in texts]
        if not inputs:
            return EmbeddingBatchResult(vectors=[], model=self._primary_model)

        last_error: Exception | None = None
        if self._fallback_model and self._fallback_model != self._primary_model:
            model_candidates: Tuple[str, ...] = (
                self._primary_model,
                self._fallback_model,
            )
        else:
            model_candidates = (self._primary_model,)

        for model in model_candidates:
            try:
                vectors = self._invoke_provider(model, inputs)
            except Exception as exc:  # pragma: no cover - requires network failure
                last_error = exc
                logger.warning(
                    "embeddings.batch_failed",
                    model=model,
                    exc_type=exc.__class__.__name__,
                    exc_message=str(exc),
                )
                continue

            if vectors and self._dim is None:
                self._dim = len(vectors[0])
            return EmbeddingBatchResult(vectors=vectors, model=model)

        message = "Embedding request failed"
        if last_error is not None:
            message = f"{message}: {last_error}"
        raise EmbeddingClientError(message)

    def _invoke_provider(self, model: str, inputs: Sequence[str]) -> List[List[float]]:
        if litellm_embedding is None:
            raise EmbeddingProviderUnavailable("litellm package is not installed")

        cfg = get_config()
        api_base = cfg.litellm_base_url.rstrip("/") + "/v1"
        timeout = cfg.timeouts.get("embeddings")
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


_default_client: EmbeddingClient | None = None


def get_embedding_client() -> EmbeddingClient:
    global _default_client
    if _default_client is None:
        _default_client = EmbeddingClient.from_settings()
    return _default_client


def reset_embedding_client() -> None:
    global _default_client
    _default_client = None
