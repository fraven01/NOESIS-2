from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

from ai_core.agent.runtime import AgentRuntime
from ai_core.agent.runtime_config import RuntimeConfig
from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts.base import ToolContext


_FIXED_RUN_ID = "dev-run-001"
_FIXED_TRACE_ID = "trace-dev-001"
_FIXED_INVOCATION_ID = "invocation-dev-001"
_FIXED_SERVICE_ID = "dev-runner"
_FIXED_TENANT_ID = "tenant-dev"
_FIXED_TS = "2026-01-01T00:00:00Z"


def _build_tool_context() -> ToolContext:
    scope = ScopeContext(
        tenant_id=_FIXED_TENANT_ID,
        trace_id=_FIXED_TRACE_ID,
        invocation_id=_FIXED_INVOCATION_ID,
        service_id=_FIXED_SERVICE_ID,
        run_id=_FIXED_RUN_ID,
        timestamp=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    )
    business = BusinessContext()
    return ToolContext(scope=scope, business=business, metadata={})


def _hash_inputs(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _flow_input(flow_name: str) -> dict[str, object]:
    if flow_name == "rag_query":
        return {"question": "hello", "top_k": 2}
    return {"query": "hello"}


def run_agent(flow_name: str, output_path: Path) -> dict[str, object]:
    runtime = AgentRuntime()
    tool_context = _build_tool_context()
    runtime_config = RuntimeConfig(execution_scope="TENANT")

    flow_input = _flow_input(flow_name)
    record = runtime.start(
        tool_context=tool_context,
        runtime_config=runtime_config,
        flow_name=flow_name,
        flow_input=flow_input,
    )

    artifact: dict[str, object] = {
        "run_id": record["run_id"],
        "inputs_hash": _hash_inputs(flow_input),
        "decision_log": record.get("decision_log", []),
        "stop_decision": record["stop_decision"],
    }

    if flow_name == "rag_query":
        output = record.get("output", {})
        retrieval_matches = output.get("retrieval_matches", [])
        artifact["retrieval"] = {
            "matches_count": len(retrieval_matches),
            "matches_sample": retrieval_matches[:3],
        }
        artifact["answer"] = {"text": output.get("answer", "")}
        artifact["citations"] = output.get("citations", [])
        artifact["claim_to_citation"] = output.get("claim_to_citation", {})

    output_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run agent runtime flow and emit harness artifact."
    )
    parser.add_argument(
        "--flow-name", default="dummy_flow", help="Flow name to execute"
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    output_path = Path(args.output)
    run_agent(args.flow_name, output_path)


if __name__ == "__main__":
    main()
