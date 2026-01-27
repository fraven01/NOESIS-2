from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from pydantic import BaseModel

from ai_core.agent.capabilities.base import CapabilitySpec
from ai_core.agent.runtime_config import RuntimeConfig
from ai_core.tool_contracts.base import ToolContext


_REGISTRY: Dict[str, CapabilitySpec] = {}


def _load_default_capabilities() -> None:
    # Register known capabilities via import side effects.
    try:
        import ai_core.agent.capabilities.rag_retrieve  # noqa: F401
    except ImportError:
        pass
    try:
        import ai_core.agent.capabilities.rag_compose  # noqa: F401
    except ImportError:
        pass
    try:
        import ai_core.agent.capabilities.rag_evidence  # noqa: F401
    except ImportError:
        pass


def _wrap_entrypoint(
    entrypoint: Callable[[BaseModel], BaseModel],
) -> Callable[[ToolContext, RuntimeConfig, BaseModel], BaseModel]:
    def _execute(
        _tool_context: ToolContext,
        _runtime_config: RuntimeConfig,
        input_model: BaseModel,
    ) -> BaseModel:
        return entrypoint(input_model)

    return _execute


def _normalize_spec(spec: CapabilitySpec) -> CapabilitySpec:
    input_model = spec.input_model or spec.input_schema
    output_model = spec.output_model or spec.output_schema
    execute = spec.execute
    if execute is None and spec.entrypoint is not None:
        execute = _wrap_entrypoint(spec.entrypoint)
    return CapabilitySpec(
        name=spec.name,
        version=spec.version,
        io_spec_version=spec.io_spec_version,
        input_model=input_model,
        output_model=output_model,
        execute=execute,
        io_spec=spec.io_spec,
        entrypoint=spec.entrypoint,
        input_schema=spec.input_schema,
        output_schema=spec.output_schema,
    )


def register_capability(spec: CapabilitySpec) -> None:
    if not spec.name:
        raise ValueError("capability name must be provided")
    if spec.io_spec is None:
        raise ValueError(f"capability '{spec.name}' missing io_spec")
    normalized = _normalize_spec(spec)
    if normalized.input_model is None or normalized.output_model is None:
        raise ValueError(f"capability '{spec.name}' missing input/output schema")
    if normalized.execute is None:
        raise ValueError(f"capability '{spec.name}' missing execute callable")
    _REGISTRY[spec.name] = normalized


def get_capability(name: str) -> CapabilitySpec:
    _load_default_capabilities()
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"capability '{name}' is not registered") from exc


def list_capabilities() -> list[CapabilitySpec]:
    return list(_REGISTRY.values())


def execute(
    name: str,
    tool_context: ToolContext,
    runtime_config: RuntimeConfig,
    input_payload: BaseModel | Mapping[str, Any],
) -> BaseModel:
    spec = get_capability(name)
    if spec.input_model is None or spec.output_model is None or spec.execute is None:
        raise ValueError(f"capability '{name}' missing required schema or execute")
    if isinstance(input_payload, BaseModel):
        payload = input_payload.model_dump()
    else:
        payload = dict(input_payload)
    validated_input = spec.input_model.model_validate(payload)
    raw_output = spec.execute(tool_context, runtime_config, validated_input)
    if isinstance(raw_output, BaseModel):
        output_payload = raw_output.model_dump()
    else:
        output_payload = raw_output
    return spec.output_model.model_validate(output_payload)


__all__ = [
    "CapabilitySpec",
    "register_capability",
    "get_capability",
    "list_capabilities",
    "execute",
]
