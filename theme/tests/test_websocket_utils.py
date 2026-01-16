from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from ai_core.tool_contracts import ContextError
from theme.websocket_utils import build_websocket_context


def _user(pk: object, *, authenticated: bool = True) -> SimpleNamespace:
    return SimpleNamespace(pk=pk, is_authenticated=authenticated)


def test_build_websocket_context_authenticated_user() -> None:
    user_pk = uuid4()
    scope, business = build_websocket_context(
        request={"user": _user(user_pk)},
        tenant_id="tenant",
        case_id="case-1",
        thread_id="thread-1",
        workflow_id="rag-chat-manual",
    )

    assert scope.user_id == str(user_pk)
    assert scope.run_id
    assert scope.trace_id
    assert scope.invocation_id
    UUID(scope.trace_id)
    UUID(scope.invocation_id)
    UUID(scope.run_id)
    assert business.case_id == "case-1"
    assert business.thread_id == "thread-1"


def test_build_websocket_context_unauthenticated_user() -> None:
    scope, _ = build_websocket_context(
        request={"user": _user(uuid4(), authenticated=False)},
        tenant_id="tenant",
    )

    assert scope.user_id is None


def test_build_websocket_context_rejects_invalid_user_id() -> None:
    with pytest.raises(ContextError):
        build_websocket_context(
            request={"user": _user("not-a-uuid")},
            tenant_id="tenant",
        )
