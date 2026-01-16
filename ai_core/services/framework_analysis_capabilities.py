"""Capabilities for framework analysis graph."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from ai_core.infra.prompts import load as load_prompt
from ai_core.llm import client as llm_client
from ai_core.tools.framework_contracts import ComponentLocation


def normalize_gremium_identifier(suggestion: str, raw_name: str) -> str:
    """Normalize gremium identifier for storage."""
    normalized = suggestion.upper()
    # Replace umlauts
    normalized = normalized.replace("\u00dc", "UE")
    normalized = normalized.replace("\u00c4", "AE")
    normalized = normalized.replace("\u00d6", "OE")
    normalized = normalized.replace("\u00fc", "ue")
    normalized = normalized.replace("\u00e4", "ae")
    normalized = normalized.replace("\u00f6", "oe")
    # Replace special chars with underscore
    normalized = re.sub(r"[^A-Z0-9_]", "_", normalized)
    # Remove consecutive underscores
    normalized = re.sub(r"_+", "_", normalized)
    # Remove leading/trailing underscores
    normalized = normalized.strip("_")
    return normalized


def extract_toc_from_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract table of contents from parent node metadata."""
    parent_map: dict[str, dict[str, Any]] = {}

    for chunk in chunks:
        parents = chunk.get("meta", {}).get("parents", [])
        for parent in parents:
            parent_id = parent.get("id")
            if parent_id and parent_id not in parent_map:
                parent_type = parent.get("type", "")
                # Only include structural elements
                if parent_type in {"heading", "section", "article", "document"}:
                    parent_map[parent_id] = {
                        "id": parent_id,
                        "title": parent.get("title", ""),
                        "type": parent_type,
                        "level": parent.get("level", 0),
                        "order": parent.get("order", 0),
                    }

    # Sort by level and order
    toc_entries = sorted(
        parent_map.values(), key=lambda p: (p.get("level", 0), p.get("order", 0))
    )

    return toc_entries


def parse_llm_json_response(text: str) -> dict[str, Any]:
    """Parse JSON payloads from LLM responses."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
    raise ValueError("LLM response is not valid JSON")


def call_llm_json_prompt(
    *,
    prompt_key: str,
    prompt_input: str,
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Load prompt, call LLM, and parse JSON output."""
    prompt = load_prompt(prompt_key)
    full_prompt = f"{prompt['text']}\n\n{prompt_input}"

    meta_payload = dict(meta)
    meta_payload["prompt_version"] = prompt["version"]

    result = llm_client.call("analyze", full_prompt, meta_payload)
    text = str(result.get("text") or "")
    return parse_llm_json_response(text)


def validate_component_locations(
    located_components: Mapping[str, Any],
    *,
    high_confidence_threshold: float = 0.8,
) -> dict[str, Any]:
    """Validate component locations and return per-component summaries."""
    validations: dict[str, Any] = {}
    for component, location in located_components.items():
        resolved = ComponentLocation.from_partial(location)
        validations[component] = resolved.validation_summary(
            high_confidence_threshold=high_confidence_threshold
        )
    return validations
