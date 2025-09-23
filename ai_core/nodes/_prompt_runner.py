"""Utilities for running simple prompt-based nodes."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from ai_core.infra.mask_prompt import mask_prompt, mask_response
from ai_core.infra.pii_flags import get_pii_config
from ai_core.infra.prompts import load
from ai_core.infra.tracing import trace
from ai_core.llm import client


ResultShaper = Callable[[Dict[str, Any]], Tuple[Any, Dict[str, Any]]]


def run_prompt_node(
    *,
    trace_name: str,
    prompt_alias: str,
    llm_label: str,
    state_key: str,
    state: Dict[str, Any],
    meta: Dict[str, str],
    result_shaper: Optional[ResultShaper] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute a prompt-driven node with shared boilerplate logic."""

    prompt = load(prompt_alias)
    meta["prompt_version"] = prompt["version"]
    meta_with_version = dict(meta)

    @trace(trace_name)
    def _execute(
        current_state: Dict[str, Any], *, meta: Dict[str, str]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        base = current_state.get("text", "")
        full_prompt = f"{prompt['text']}\n\n{base}"
        pii_config = get_pii_config()
        masked = mask_prompt(full_prompt, config=pii_config)
        result = client.call(llm_label, masked, meta)

        if result_shaper is not None:
            value, metadata_extra = result_shaper(result)
        else:
            value, metadata_extra = result["text"], {}

        value = mask_response(value, config=pii_config)

        new_state = dict(current_state)
        new_state[state_key] = value

        node_meta = {
            state_key: value,
            **metadata_extra,
            "prompt_version": prompt["version"],
        }
        return new_state, node_meta

    return _execute(state, meta=meta_with_version)
