from __future__ import annotations

import re
from typing import Optional

from ai_core.infra.pii import mask_text
from ai_core.infra.pii_flags import get_pii_config

_SUMMARY_PATTERN = re.compile(
    r"(?:\n\n)?\[REDACTED: tags=<(?P<tags>[^>]*)> policy=<(?P<policy>[^>]*)>\]\s*$"
)
_PLACEHOLDER_PATTERN = re.compile(
    r"\[REDACTED_([A-Z0-9_]+)\]|<([A-Z0-9_]+)_[0-9a-fA-F]{8}>"
)


def _strip_summary_block(text: str) -> str:
    match = _SUMMARY_PATTERN.search(text)
    if not match:
        return text
    return text[: match.start()]


def _collect_tags(text: str) -> list[str]:
    tags = {
        group1 or group2
        for group1, group2 in _PLACEHOLDER_PATTERN.findall(text)
        if (group1 or group2)
    }
    return sorted(tags)


def _apply_mask(
    text: str,
    config: dict,
    *,
    placeholder_only: bool = False,
    include_summary: bool = True,
) -> str:
    if text is None:
        return text

    if placeholder_only:
        return "XXXX"

    hmac_key: Optional[bytes] = config["hmac_secret"] if config["deterministic"] else None
    masked = mask_text(
        text,
        config["policy"],
        config["deterministic"],
        hmac_key,
        mode=config.get("mode", "industrial"),
        name_detection=config.get("name_detection", False),
        session_scope=config.get("session_scope"),
    )
    base = _strip_summary_block(masked)
    if include_summary:
        tags = _collect_tags(base)
        if tags:
            summary = f"[REDACTED: tags=<{','.join(tags)}> policy=<{config['policy']}>]"
            return f"{base}\n\n{summary}"
    return base


def mask_prompt(
    prompt: str,
    *,
    placeholder_only: bool = False,
    config: Optional[dict] = None,
) -> str:
    """Mask prompt text according to the active PII configuration."""

    pii_config = config or get_pii_config()
    mode = str(pii_config.get("mode", "")).lower()
    policy = str(pii_config.get("policy", "")).lower()
    if mode == "off" or policy == "off":
        return prompt
    return _apply_mask(
        prompt,
        pii_config,
        placeholder_only=placeholder_only,
        include_summary=True,
    )


def mask_response(
    text: str,
    *,
    config: Optional[dict] = None,
    include_summary: bool = False,
) -> str:
    """Mask LLM responses when post-response redaction is enabled."""

    pii_config = config or get_pii_config()
    mode = str(pii_config.get("mode", "")).lower()
    policy = str(pii_config.get("policy", "")).lower()
    if mode == "off" or policy == "off":
        return text
    if not pii_config.get("post_response"):
        return text
    return _apply_mask(
        text,
        pii_config,
        include_summary=include_summary,
    )
