from __future__ import annotations

from typing import Any, Dict, Tuple

from ai_core.infra.mask_prompt import mask_prompt, mask_response
from ai_core.infra.pii_flags import get_pii_config
from ai_core.infra.prompts import load
from ai_core.infra.tracing import trace
from ai_core.llm import client
from ai_core.tool_contracts import (
    RateLimitedError as ToolRateLimitedError,
    TimeoutError as ToolTimeoutError,
    UpstreamServiceError,
)
from ai_core.llm.client import LlmClientError, RateLimitError


def run(
    state: Dict[str, Any], meta: Dict[str, str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compose an answer from retrieved snippets using the LLM."""
    prompt = load("retriever/answer")
    meta["prompt_version"] = prompt["version"]
    meta_with_version = dict(meta)
    return _run(prompt, state, meta=meta_with_version)


@trace("compose")
def _run(
    prompt: Dict[str, str], state: Dict[str, Any], *, meta: Dict[str, str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    snippets_text = "\n".join(s.get("text", "") for s in state.get("snippets", []))
    question = state.get("question", "")
    full_prompt = f"{prompt['text']}\n\nQuestion: {question}\nContext:\n{snippets_text}"
    pii_config = get_pii_config()
    masked = mask_prompt(full_prompt, config=pii_config)
    try:
        result = client.call("synthesize", masked, meta)
    except RateLimitError as exc:
        raise ToolRateLimitedError(str(getattr(exc, "detail", "rate limited"))) from exc
    except LlmClientError as exc:
        status = getattr(exc, "status", None)
        # Map LLM client errors to tool-level typed errors for the graph layer
        try:
            code = int(status) if isinstance(status, int) else int(str(status))
        except Exception:
            code = None
        message = str(getattr(exc, "detail", None) or exc) or "LLM error"
        if code in {408, 504}:
            raise ToolTimeoutError(message) from exc
        # For remaining 4xx/5xx, surface as upstream dependency error
        raise UpstreamServiceError(message) from exc
    answer = mask_response(result["text"], config=pii_config)
    new_state = dict(state)
    new_state["answer"] = answer
    return new_state, {"answer": answer, "prompt_version": prompt["version"]}
