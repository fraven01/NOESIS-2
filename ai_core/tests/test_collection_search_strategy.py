from __future__ import annotations

from ai_core.services.collection_search.strategy import (
    SearchStrategyRequest,
    extract_strategy_payload,
    fallback_strategy,
    llm_strategy_generator,
)


def test_extract_strategy_payload_accepts_code_block_json() -> None:
    payload = """```json
    {
      "queries": ["alpha guide", "alpha requirements", "alpha official docs"],
      "policies_applied": ["tenant-default"],
      "preferred_sources": [],
      "disallowed_sources": [],
      "notes": "Focus on official docs."
    }
    ```"""

    data = extract_strategy_payload(payload)

    assert data["queries"][0] == "alpha guide"
    assert data["policies_applied"] == ["tenant-default"]


def test_fallback_strategy_generates_richer_queries() -> None:
    request = SearchStrategyRequest(
        tenant_id="tenant-1",
        query="Acme telemetry",
        quality_mode="software_docs_strict",
        purpose="docs-gap-analysis",
    )

    strategy = fallback_strategy(request)

    assert strategy.queries[0] == "Acme telemetry"
    assert len(strategy.queries) >= 3
    assert any("docs gap analysis" in query for query in strategy.queries)
    assert any("official documentation" in query for query in strategy.queries)


def test_llm_strategy_generator_requests_json_mode(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_call(label, prompt, metadata, response_format=None, extra_params=None):
        captured["label"] = label
        captured["response_format"] = response_format
        return {
            "text": (
                '{"queries":["alpha guide","alpha requirements","alpha official docs"],'
                '"policies_applied":[],"preferred_sources":[],"disallowed_sources":[]}'
            )
        }

    monkeypatch.setattr(
        "ai_core.services.collection_search.strategy.llm_client.call", fake_call
    )

    request = SearchStrategyRequest(
        tenant_id="tenant-1",
        query="alpha",
        quality_mode="standard",
        purpose="collection_search",
    )

    strategy = llm_strategy_generator(request)

    assert captured["response_format"] == {"type": "json_object"}
    assert strategy.queries[0] == "alpha"
