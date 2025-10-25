"""Utilities for running simple prompt-based nodes."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Optional, Tuple

from django.conf import settings

from ai_core.infra.mask_prompt import mask_prompt, mask_response
from ai_core.infra.pii_flags import get_pii_config
from ai_core.infra.prompts import load
from ai_core.infra.observability import emit_event, observe_span, update_observation
from ai_core.llm import client


ResultShaper = Callable[[Dict[str, Any]], Tuple[Any, Dict[str, Any]]]

_GUARDRAIL_SAMPLE_RATE = 0.2


def _normalise_guardrail_payload(
    meta: Dict[str, Any], metadata_extra: Dict[str, Any]
) -> Dict[str, Any] | None:
    """Extract guardrail metadata from result/meta dictionaries."""

    guard: Dict[str, Any] = {}

    def _apply(source: Dict[str, Any]) -> None:
        if not isinstance(source, dict):
            return
        candidate = source.get("guardrail")
        if isinstance(candidate, dict):
            for key in (
                "rule_id",
                "outcome",
                "redactions",
                "tool_blocked",
                "reason_code",
            ):
                if key in candidate and key not in guard and candidate[key] is not None:
                    guard[key] = candidate[key]
        for key in ("rule_id", "outcome", "redactions", "tool_blocked", "reason_code"):
            prefixed = f"guardrail_{key}"
            if prefixed in source and key not in guard and source[prefixed] is not None:
                guard[key] = source[prefixed]

    _apply(metadata_extra)
    _apply(meta)

    if not guard:
        return None

    if "rule_id" in guard:
        guard["rule_id"] = str(guard["rule_id"])

    if "outcome" in guard and guard["outcome"] is not None:
        guard["outcome"] = str(guard["outcome"])

    if "tool_blocked" in guard:
        guard["tool_blocked"] = bool(guard["tool_blocked"])

    if "redactions" in guard:
        redactions = guard["redactions"]
        if isinstance(redactions, (list, tuple)):
            guard["redactions"] = [
                str(value) for value in redactions if value is not None
            ]
        elif redactions is None:
            guard.pop("redactions", None)
        else:
            guard["redactions"] = [str(redactions)]

    return guard


def _determine_branch(guardrail: Dict[str, Any]) -> str:
    outcome = guardrail.get("outcome")
    if isinstance(outcome, str) and outcome:
        return outcome
    if guardrail.get("tool_blocked"):
        return "blocked"
    return "allowed"


def _should_emit_guardrail_event(guardrail: Dict[str, Any]) -> bool:
    rule_id = guardrail.get("rule_id")
    if not isinstance(rule_id, str) or not rule_id:
        return False

    allowlist = getattr(settings, "AI_GUARDRAIL_SAMPLE_ALLOWLIST", ()) or ()
    try:
        allowed = {str(item) for item in allowlist}
    except Exception:
        allowed = {str(allowlist)} if allowlist else set()

    if rule_id not in allowed:
        return False

    try:
        return random.random() < _GUARDRAIL_SAMPLE_RATE
    except Exception:
        return False


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

    @observe_span(name=trace_name)
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

        metadata_extra = dict(metadata_extra or {})

        value = mask_response(value, config=pii_config)

        guardrail_payload = _normalise_guardrail_payload(meta, metadata_extra)

        if guardrail_payload:
            branch = _determine_branch(guardrail_payload)
            try:
                update_observation(metadata={"node.branch_taken": branch})
            except Exception:
                pass

            if _should_emit_guardrail_event(guardrail_payload):
                event_payload = {"event": "guardrail.result"}
                for key in (
                    "rule_id",
                    "outcome",
                    "tool_blocked",
                    "reason_code",
                    "redactions",
                ):
                    if key in guardrail_payload:
                        event_payload[key] = guardrail_payload[key]
                try:
                    emit_event(event_payload)
                except Exception:
                    pass

        new_state = dict(current_state)
        new_state[state_key] = value

        node_meta = {
            state_key: value,
            **metadata_extra,
            "prompt_version": prompt["version"],
        }
        return new_state, node_meta

    return _execute(state, meta=meta_with_version)
