from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Callable, Tuple


_FLOW_MODULES: dict[str, str] = {
    "dummy_flow": "ai_core.agent.flows.dummy_flow.contract",
    "rag_query": "ai_core.agent.flows.rag_query.contract",
}

_FLOW_EXECUTORS: dict[str, str] = {
    "dummy_flow": "ai_core.agent.flows.dummy_flow.execute",
    "rag_query": "ai_core.agent.flows.rag_query.execute",
}


def get_flow_contract(flow_name: str) -> ModuleType:
    module_path = _FLOW_MODULES.get(flow_name)
    if module_path is None:
        raise KeyError(f"unknown flow: {flow_name}")
    return import_module(module_path)


def get_flow_execute(flow_name: str) -> Callable[..., dict]:
    module_path = _FLOW_EXECUTORS.get(flow_name)
    if module_path is None:
        raise KeyError(f"unknown flow: {flow_name}")
    module = import_module(module_path)
    execute = getattr(module, "execute", None)
    if not callable(execute):
        raise AttributeError(f"flow '{flow_name}' has no execute() callable")
    return execute


def get_flow_spec(flow_name: str) -> Tuple[ModuleType, Callable[..., dict]]:
    return get_flow_contract(flow_name), get_flow_execute(flow_name)


__all__ = ["get_flow_contract", "get_flow_execute", "get_flow_spec"]
