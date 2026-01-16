import pytest
from unittest.mock import MagicMock, patch
from ai_core.graphs.technical.retrieval_augmented_generation import (
    RetrievalAugmentedGenerationGraph,
    _load_history,
    _append_history,
    _trim_history,
)


@pytest.fixture
def mock_checkpointer():
    with patch(
        "ai_core.graphs.technical.retrieval_augmented_generation.ThreadAwareCheckpointer"
    ) as mock:
        instance = mock.return_value
        instance.load.return_value = {}
        instance.save.return_value = None
        yield instance


@pytest.fixture
def mock_graph_invoke():
    with patch(
        "ai_core.graphs.technical.retrieval_augmented_generation._build_compiled_graph"
    ) as mock_build:
        mock_compiled = MagicMock()
        mock_build.return_value = mock_compiled
        yield mock_compiled.invoke


def test_rag_run_loads_and_saves_history(mock_checkpointer, mock_graph_invoke):
    # Setup
    graph = RetrievalAugmentedGenerationGraph()
    state = {"question": "What is AI?", "query": "What is AI?"}
    meta = {
        "scope_context": {
            "tenant_id": "test-tenant",
            "trace_id": "test-trace",
            "invocation_id": "inv1",
            "run_id": "run1",
        },
        "business_context": {"case_id": "c1", "thread_id": "th1"},
    }

    # Mock history in store
    mock_checkpointer.load.return_value = {
        "chat_history": [
            {"role": "user", "content": "Prev Q"},
            {"role": "assistant", "content": "Prev A"},
        ]
    }

    # Mock graph execution result
    mock_graph_invoke.return_value = {
        "state": {
            # Graph should have received history in input state
            "chat_history": [
                {"role": "user", "content": "Prev Q"},
                {"role": "assistant", "content": "Prev A"},
            ],
            "question": "What is AI?",
        },
        "result": {
            "answer": "AI is magic.",
            "retrieval": {},
            "snippets": [],
            "prompt_version": "v1",
        },
    }

    # Execute
    final_state, result = graph.run(state, meta)
    assert result["answer"] == "AI is magic."

    # Verify Load
    mock_checkpointer.load.assert_called_once()

    # Verify graph input state contained history
    call_args = mock_graph_invoke.call_args[0][0]
    assert call_args["state"]["chat_history"][0]["content"] == "Prev Q"

    # Verify Save
    mock_checkpointer.save.assert_called_once()
    saved_state = mock_checkpointer.save.call_args[0][1]  # (ctx, state)
    saved_history = saved_state["chat_history"]

    # Should contain Prev Q, Prev A, New Q, New A
    assert len(saved_history) == 4
    assert saved_history[-2]["content"] == "What is AI?"  # user
    assert saved_history[-1]["content"] == "AI is magic."  # assistant


def test_rag_history_limit(mock_checkpointer, mock_graph_invoke):
    # Test strict limit logic via env var helper
    with patch.dict("os.environ", {"RAG_CHAT_HISTORY_MAX_MESSAGES": "2"}):
        graph = RetrievalAugmentedGenerationGraph()
        # Mock Checkpointer returns 4 items
        history = [{"role": "user", "content": f"msg{i}"} for i in range(4)]
        mock_checkpointer.load.return_value = {"chat_history": history}

        mock_graph_invoke.return_value = {
            "state": {"question": "New Q"},
            "result": {"answer": "New Answer"},
        }

        # Run
        meta = {
            "scope_context": {
                "tenant_id": "t",
                "trace_id": "tr",
                "invocation_id": "i",
                "run_id": "r",
            },
            "business_context": {"thread_id": "th"},
        }
        graph.run({"question": "New Q"}, meta)

        # Check saved
        assert mock_checkpointer.save.called
        saved_state = mock_checkpointer.save.call_args[0][1]
        saved_history = saved_state["chat_history"]

        # Limit is 2. So we expect 2 items.
        # Logic: append (New Q, New A) -> total 4+2=6 items. Trim to 2.
        # Should remain: New Q, New A.
        assert len(saved_history) == 2
        assert saved_history[0]["content"] == "New Q"
        assert saved_history[1]["content"] == "New Answer"


def test_helpers():
    # Unit test helpers
    h = []
    _append_history(h, role="u", content="c")
    assert h == [{"role": "u", "content": "c"}]

    source = [{"role": "u", "content": "c"}, {"bad": "entry"}]
    clean = _load_history({"chat_history": source})
    assert len(clean) == 1
    assert clean[0]["content"] == "c"

    trimmed = _trim_history(h * 10, limit=2)
    assert len(trimmed) == 2
