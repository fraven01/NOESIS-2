from __future__ import annotations

import inspect
import typing

from pydantic import BaseModel

from ai_core import nodes
from ai_core.tool_contracts import ToolContext


NODE_MODULES = (
    nodes.assess,
    nodes.classify,
    nodes.compose,
    nodes.draft_blocks,
    nodes.extract,
    nodes.needs,
    nodes.retrieve,
)


def test_node_run_signatures_use_tool_context_and_params() -> None:
    for module in NODE_MODULES:
        run = getattr(module, "run", None)
        assert callable(run), f"{module.__name__}.run is missing"

        unwrapped = inspect.unwrap(run)
        signature = inspect.signature(unwrapped)
        params = list(signature.parameters.values())
        hints = typing.get_type_hints(unwrapped, globalns=unwrapped.__globals__)
        assert len(params) == 2, f"{module.__name__}.run should take 2 args"

        context_param = params[0]
        input_param = params[1]
        assert (
            context_param.name == "context"
        ), f"{module.__name__}.run first arg must be 'context'"
        context_hint = hints.get(context_param.name, context_param.annotation)
        assert (
            context_hint is ToolContext
        ), f"{module.__name__}.run first arg must be ToolContext"
        assert input_param.name in {
            "params",
            "input",
        }, f"{module.__name__}.run second arg must be params/input"
        input_hint = hints.get(input_param.name, input_param.annotation)
        assert inspect.isclass(input_hint) and issubclass(
            input_hint, BaseModel
        ), f"{module.__name__}.run second arg must be a Pydantic model"
