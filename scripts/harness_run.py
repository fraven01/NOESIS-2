from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_core.agent.harness.golden import load_golden_queries
from ai_core.agent.harness.validate import validate_artifact_v0
from ai_core.agent.runtime import AgentRuntime
from ai_core.agent.runtime_config import RuntimeConfig
from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts.base import ToolContext


_FIXED_TRACE_ID = "trace-harness-001"
_FIXED_INVOCATION_ID = "invocation-harness-001"
_FIXED_SERVICE_ID = "harness-runner"
_FIXED_TENANT_ID = "tenant-dev"


def _build_tool_context() -> ToolContext:
    scope = ScopeContext(
        tenant_id=_FIXED_TENANT_ID,
        trace_id=_FIXED_TRACE_ID,
        invocation_id=_FIXED_INVOCATION_ID,
        service_id=_FIXED_SERVICE_ID,
        run_id="harness-run",
        timestamp=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    )
    business = BusinessContext()
    return ToolContext(scope=scope, business=business, metadata={})


def _build_artifact(
    record: dict[str, Any], flow_input: dict[str, Any]
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "run_id": record["run_id"],
        "inputs_hash": "",
        "decision_log": record.get("decision_log", []),
        "stop_decision": record["stop_decision"],
    }
    output = record.get("output", {})
    retrieval_matches = output.get("retrieval_matches", [])
    artifact["retrieval"] = {
        "matches_count": len(retrieval_matches),
        "matches_sample": retrieval_matches[:3],
    }
    artifact["answer"] = {"text": output.get("answer", "")}
    artifact["citations"] = output.get("citations", [])
    artifact["claim_to_citation"] = output.get("claim_to_citation", {})
    return artifact


def run_harness(output_path: Path) -> dict[str, Any]:
    runtime = AgentRuntime()
    tool_context = _build_tool_context()
    runtime_config = RuntimeConfig(execution_scope="TENANT")

    queries = load_golden_queries()
    artifacts: list[dict[str, Any]] = []
    passed = 0
    failed = 0

    for query in queries:
        record = runtime.start(
            tool_context=tool_context,
            runtime_config=runtime_config,
            flow_name=query.flow_name,
            flow_input=query.input.model_dump(),
        )
        artifact = _build_artifact(record, query.input.model_dump())
        validate_artifact_v0(artifact)
        artifacts.append({"id": query.id, "artifact": artifact})
        passed += 1

    report = {
        "run_id": "harness-report",
        "queries": artifacts,
        "summary": {
            "passed_count": passed,
            "failed_count": failed,
        },
    }

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run harness golden queries.")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    output_path = Path(args.output)
    run_harness(output_path)


if __name__ == "__main__":
    main()
