"""LiteLLM-backed embedding utilities."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, TypeVar
import os

from django.conf import settings

from ai_core.infra.config import get_config
from ai_core.infra.circuit_breaker import get_litellm_circuit_breaker
from common.logging import get_log_context, get_logger
from ai_core.infra.observability import observe_span, report_generation_usage
from ai_core.infra.usage import Usage

try:  # pragma: no cover - optional dependency for OpenAI SDK
    from openai import OpenAI
except Exception:  # pragma: no cover - handled at runtime
    OpenAI = None


logger = get_logger(__name__)
T = TypeVar("T")


class EmbeddingClientError(RuntimeError):
    """Base class for embedding client errors."""


class EmbeddingProviderUnavailable(EmbeddingClientError):
    """Raised when LiteLLM dependency is unavailable."""


class EmbeddingTimeoutError(EmbeddingClientError, TimeoutError):
    """Raised when an embedding call exceeds the configured timeout."""


@dataclass(slots=True)
class EmbeddingBatchResult:
    """Return value for embedding batches."""

    vectors: List[List[float]]
    model: str
    model_used: str
    attempts: int
    timeout_s: float | None
    retry_delays: Tuple[float, ...] | None = None
    usage: Usage | None = None


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

    @observe_span("embeddings.generate")
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
                usage=Usage(),
            )

        last_error: Exception | None = None
        model_candidates = self._candidate_models()
        cfg = get_config()
        timeout_setting = cfg.timeouts.get("embeddings")
        if timeout_setting is None:
            timeout_setting = getattr(settings, "EMBEDDINGS_TIMEOUT_SECONDS", None)
        timeout_s = self._normalise_timeout(timeout_setting)
        log_context = get_log_context()
        key_alias = log_context.get("key_alias")

        for attempt, model in enumerate(model_candidates, start=1):
            try:
                vectors, usage = self._invoke_provider(model, inputs, cfg, timeout_s)

                # Instrument: Report usage to current span
                report_generation_usage(usage, model=model)

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
                    key_alias=key_alias,
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
                usage=usage,
            )

        message = "Embedding request failed"
        if last_error is not None:
            message = f"{message}: {last_error}"
        error_context = {
            "attempts": len(model_candidates),
            "key_alias": key_alias,
            "models": model_candidates,
        }
        if last_error is not None:
            error_context["exc_type"] = last_error.__class__.__name__
            error_context["exc_message"] = str(last_error)
        logger.error("embeddings.batch_failed_final", **error_context)
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
        timeout_s: float | None,
    ) -> Tuple[List[List[float]], Usage]:
        breaker = get_litellm_circuit_breaker()
        if not breaker.allow_request():
            raise EmbeddingClientError("LiteLLM circuit breaker open")
        # Use OpenAI SDK to communicate with LiteLLM proxy
        # This is the recommended approach for LiteLLM proxy usage
        if OpenAI is None:
            raise EmbeddingProviderUnavailable("openai package is not installed")

        api_base = cfg.litellm_base_url.rstrip("/")
        api_key = os.environ.get("LITELLM_MASTER_KEY", "sk-1234")

        # Create OpenAI client pointing to LiteLLM proxy
        client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=timeout_s if timeout_s and timeout_s > 0 else None,
        )

        def _call():
            response = client.embeddings.create(
                input=list(inputs),
                model=model,
            )
            return response

        try:
            response = self._execute_with_timeout(_call, timeout_s)
        except Exception:
            breaker.record_failure(reason="embedding_error")
            raise
        else:
            breaker.record_success()

        if not response.data:
            return [], Usage()

        vectors: List[List[float]] = []
        for item in response.data:
            embedding_values = item.embedding
            if embedding_values is None:
                return [], Usage()
            try:
                vector = [float(value) for value in embedding_values]
            except (TypeError, ValueError) as exc:
                raise EmbeddingClientError("Invalid embedding value") from exc
            if not vector:
                return [], Usage()
            vectors.append(vector)

        if not vectors:
            return [], Usage()

        if len(vectors) != len(inputs):
            raise EmbeddingClientError("Embedding count mismatch")
        if vectors:
            expected_len = len(vectors[0])
            for vector in vectors[1:]:
                if len(vector) != expected_len:
                    raise EmbeddingClientError("Embedding dimensions mismatch in batch")

        try:
            usage = Usage.from_provider_response(response)
        except Exception:
            usage = Usage()

        return vectors, usage

    def _execute_with_timeout(self, fn: Callable[[], T], timeout_s: float | None) -> T:
        if timeout_s is None or timeout_s <= 0:
            return fn()
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fn)
        try:
            result = future.result(timeout=timeout_s)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise EmbeddingTimeoutError(
                f"Embedding call exceeded {timeout_s} seconds"
            ) from exc
        except Exception as exc:
            if self._is_timeout_exception(exc):
                raise EmbeddingTimeoutError("Embedding call timed out") from exc
            raise
        finally:
            executor.shutdown(wait=future.done(), cancel_futures=True)
        return result

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
        if self._is_timeout_exception(exc):
            return True
        return False

    @staticmethod
    def _is_timeout_exception(exc: Exception) -> bool:
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
