from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from ai_core.agent.flows.registry import get_flow_spec
from ai_core.agent.runtime_config import RuntimeConfig
from ai_core.rag.vector_store import NullVectorStore, VectorStoreRouter
from ai_core.tool_contracts.base import ToolContext

_FORBIDDEN_ID_FIELDS = {"tenant_id", "user_id", "case_id", "workflow_id"}
_FIXED_TS = "2026-01-01T00:00:00Z"
_RUNTIME_DEPS: ContextVar[dict[str, Any]] = ContextVar("runtime_deps", default={})


def get_runtime_dependencies() -> dict[str, Any]:
    return _RUNTIME_DEPS.get()


class AgentRuntime:
    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}
        self._deps: dict[str, Any] = {
            "vector_router": VectorStoreRouter(
                {"global": NullVectorStore("global")},
                default_scope="global",
            )
        }

    def start(
        self,
        *,
        tool_context: ToolContext,
        runtime_config: RuntimeConfig,
        flow_name: str,
        flow_input: BaseModel | Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(tool_context, ToolContext):
            raise TypeError("tool_context must be a ToolContext instance")
        if not isinstance(runtime_config, RuntimeConfig):
            raise TypeError("runtime_config must be a RuntimeConfig instance")

        contract, execute = get_flow_spec(flow_name)
        validated_input = self._validate_flow_input(contract, flow_input)

        run_id = str(uuid4())
        token = _RUNTIME_DEPS.set(self._deps)
        try:
            decision_log = [
                self._decision_event(
                    run_id=run_id,
                    kind="start",
                    status="started",
                    reason="flow started",
                    evidence_refs=[],
                    stop_decision=None,
                )
            ]
            result = execute(
                tool_context,
                runtime_config,
                validated_input,
                run_id=run_id,
            )
            if isinstance(result, tuple) and len(result) == 2:
                output, capability_events = result
            else:
                output, capability_events = result, []
            stop_decision = {
                "status": "succeeded",
                "reason": "flow completed",
                "evidence_refs": [],
            }
            decision_log.extend(capability_events)
            decision_log.append(
                self._decision_event(
                    run_id=run_id,
                    kind="stop",
                    status=stop_decision["status"],
                    reason=stop_decision["reason"],
                    evidence_refs=stop_decision["evidence_refs"],
                    stop_decision=stop_decision,
                )
            )
            record = {
                "run_id": run_id,
                "flow_name": flow_name,
                "input": validated_input,
                "output": output,
                "decision_log": decision_log,
                "stop_decision": stop_decision,
                "status": "completed",
            }
            self._runs[run_id] = record
            return record
        finally:
            _RUNTIME_DEPS.reset(token)

    def resume(
        self,
        *,
        tool_context: ToolContext,
        runtime_config: RuntimeConfig,
        flow_name: str,
        flow_input: BaseModel | Mapping[str, Any],
        resume_input: BaseModel | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(tool_context, ToolContext):
            raise TypeError("tool_context must be a ToolContext instance")
        if not isinstance(runtime_config, RuntimeConfig):
            raise TypeError("runtime_config must be a RuntimeConfig instance")

        contract, execute = get_flow_spec(flow_name)
        validated_input = self._validate_flow_input(contract, flow_input)
        if resume_input is not None:
            self._assert_id_free(resume_input)

        run_id = str(uuid4())
        token = _RUNTIME_DEPS.set(self._deps)
        try:
            decision_log = [
                self._decision_event(
                    run_id=run_id,
                    kind="start",
                    status="resumed",
                    reason="flow resumed",
                    evidence_refs=[],
                    stop_decision=None,
                )
            ]
            result = execute(
                tool_context,
                runtime_config,
                validated_input,
                run_id=run_id,
            )
            if isinstance(result, tuple) and len(result) == 2:
                output, capability_events = result
            else:
                output, capability_events = result, []
            stop_decision = {
                "status": "succeeded",
                "reason": "flow completed",
                "evidence_refs": [],
            }
            decision_log.extend(capability_events)
            decision_log.append(
                self._decision_event(
                    run_id=run_id,
                    kind="stop",
                    status=stop_decision["status"],
                    reason=stop_decision["reason"],
                    evidence_refs=stop_decision["evidence_refs"],
                    stop_decision=stop_decision,
                )
            )
            record = {
                "run_id": run_id,
                "flow_name": flow_name,
                "input": validated_input,
                "resume_input": resume_input,
                "output": output,
                "decision_log": decision_log,
                "stop_decision": stop_decision,
                "status": "completed",
            }
            self._runs[run_id] = record
            return record
        finally:
            _RUNTIME_DEPS.reset(token)

    def status(self, run_id: str) -> dict[str, Any] | None:
        return self._runs.get(run_id)

    def cancel(self, run_id: str) -> bool:
        record = self._runs.get(run_id)
        if record is None:
            return False
        record["status"] = "cancelled"
        return True

    def _validate_flow_input(
        self, contract: object, flow_input: BaseModel | Mapping[str, Any]
    ) -> BaseModel:
        input_model = getattr(contract, "InputModel")
        self._assert_id_free(flow_input)
        if isinstance(flow_input, BaseModel):
            payload = flow_input.model_dump()
            return input_model.model_validate(payload)
        if isinstance(flow_input, Mapping):
            return input_model.model_validate(flow_input)
        raise TypeError("flow_input must be a pydantic model or mapping")

    def _assert_id_free(self, value: BaseModel | Mapping[str, Any]) -> None:
        if isinstance(value, BaseModel):
            keys = set(value.model_fields.keys())
        elif isinstance(value, Mapping):
            keys = set(value.keys())
        else:
            raise TypeError("flow_input must be a pydantic model or mapping")
        forbidden = sorted(keys & _FORBIDDEN_ID_FIELDS)
        if forbidden:
            raise ValueError(
                "flow_input must not include ID fields: " + ", ".join(forbidden)
            )

    def _decision_event(
        self,
        *,
        run_id: str,
        kind: str,
        status: str,
        reason: str,
        evidence_refs: list[str],
        stop_decision: dict[str, Any] | None,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "run_id": run_id,
            "ts": _FIXED_TS,
            "kind": kind,
            "status": status,
            "reason": reason,
            "evidence_refs": evidence_refs,
        }
        if stop_decision is not None:
            event["stop_decision"] = stop_decision
        return event


__all__ = ["AgentRuntime", "get_runtime_dependencies"]
