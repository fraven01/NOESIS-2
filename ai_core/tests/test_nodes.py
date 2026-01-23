import pytest
from types import SimpleNamespace

from ai_core.infra.mask_prompt import mask_prompt
from ai_core.infra.prompts import load
from ai_core.nodes import (
    retrieve,
    compose,
    extract,
    classify,
    assess,
    draft_blocks,
    needs,
)
from ai_core.nodes._prompt_runner import run_prompt_node
from ai_core.rag.schemas import Chunk
from ai_core.rag.vector_client import HybridSearchResult
from ai_core.contracts.business import BusinessContext
from ai_core.contracts.scope import ScopeContext
from ai_core.tool_contracts import (
    ContextError,
    InconsistentMetadataError,
    InputError,
    NotFoundError,
    ToolContext,
)


def _tool_context(
    *,
    tenant_id: str,
    case_id: str | None,
    run_id: str,
    tenant_schema: str | None = None,
) -> ToolContext:
    scope = ScopeContext(
        tenant_id=tenant_id,
        trace_id="trace-1",
        invocation_id="inv-1",
        run_id=run_id,
        tenant_schema=tenant_schema,
        service_id="test-worker",
    )
    business = BusinessContext(case_id=case_id)
    return scope.to_tool_context(business=business)


class _DummyProfile:
    def __init__(self, vector_space: str) -> None:
        self.vector_space = vector_space


class _DummyConfig:
    def __init__(self, profile: str, vector_space: str) -> None:
        self.embedding_profiles = {profile: _DummyProfile(vector_space)}


def _patch_routing(
    monkeypatch, profile: str = "standard", space: str = "rag/standard@v1"
):
    monkeypatch.setattr(
        "ai_core.nodes.retrieve.resolve_embedding_profile",
        lambda *, tenant_id, process=None, doc_class=None, collection_id=None, workflow_id=None: profile,
    )
    monkeypatch.setattr(
        "ai_core.nodes.retrieve.resolve_vector_space_full",
        lambda _profile_id: SimpleNamespace(
            vector_space=SimpleNamespace(id=space, schema="rag")
        ),
    )


def test_retrieve_hybrid_search(monkeypatch):
    _patch_routing(monkeypatch)

    chunk = Chunk(
        content="Hybrid Result",
        meta={
            "id": "doc-1",
            "source": "src",
            "hash": "h1",
            "score": 0.83,
            "tenant_id": "tenant-1",
            "case_id": "case-1",
            "page_number": 3,
            "line_start": 12,
            "line_end": 14,
            "chunk_id": "chunk-1234567890",
        },
    )
    hybrid_result = HybridSearchResult(
        chunks=[chunk],
        vector_candidates=37,
        lexical_candidates=41,
        fused_candidates=42,
        duration_ms=1.1,
        alpha=0.6,
        min_sim=0.2,
        vec_limit=40,
        lex_limit=30,
    )
    hybrid_result.deleted_matches_blocked = 3

    class _Router:
        def __init__(self):
            self.calls = []

        def hybrid_search(self, query, **kwargs):
            self.calls.append({"query": query, **kwargs})
            return hybrid_result

    router = _Router()
    monkeypatch.setattr("ai_core.nodes.retrieve._get_router", lambda: router)

    state = {
        "query": "Hello",
        "filters": {"category": "demo"},
        "process": "review",
        "doc_class": "invoice",
        "hybrid": {
            "alpha": 0.6,
            "min_sim": 0.2,
            "top_k": 25,
            "vec_limit": 40,
            "lex_limit": 30,
        },
    }

    params = retrieve.RetrieveInput.from_state(state)
    context = _tool_context(tenant_id="tenant-1", case_id="case-1", run_id="run-1")

    result = retrieve.run(context, params)

    assert router.calls[0]["top_k"] == 10  # clamped to TOPK_MAX
    assert router.calls[0]["max_candidates"] == 40
    assert router.calls[0]["process"] == "review"
    assert router.calls[0]["doc_class"] == "invoice"
    assert router.calls[0]["visibility"] is None
    assert router.calls[0]["visibility_override_allowed"] is False

    match = result.matches[0]
    assert match["id"] == "doc-1"
    assert match["score"] == pytest.approx(0.83)
    assert match["source"] == "src"
    assert match["citation"] == "src · S.3 · Z.12-14"
    assert match["meta"]["tenant_id"] == "tenant-1"
    assert match["meta"]["case_id"] == "case-1"
    assert isinstance(result.meta.took_ms, int)
    assert result.meta.took_ms >= 0

    meta_payload = result.meta
    assert meta_payload.alpha == pytest.approx(0.6)
    assert meta_payload.min_sim == pytest.approx(0.2)
    assert meta_payload.top_k_effective == router.calls[0]["top_k"]
    assert meta_payload.matches_returned == len(result.matches)
    assert meta_payload.max_candidates_effective >= meta_payload.top_k_effective
    assert meta_payload.vector_candidates == 37
    assert meta_payload.lexical_candidates == 41
    assert meta_payload.deleted_matches_blocked == 3
    assert meta_payload.routing.profile == "standard"
    assert meta_payload.routing.vector_space_id == "rag/standard@v1"
    assert meta_payload.visibility_effective == "active"


def test_retrieve_scoped_router(monkeypatch):
    _patch_routing(monkeypatch)

    hybrid_result = HybridSearchResult(
        chunks=[],
        vector_candidates=0,
        lexical_candidates=0,
        fused_candidates=0,
        duration_ms=0.0,
        alpha=0.7,
        min_sim=0.15,
        vec_limit=50,
        lex_limit=50,
    )

    class _TenantClient:
        def __init__(self) -> None:
            self.calls = []

        def hybrid_search(self, query, **kwargs):
            self.calls.append({"query": query, **kwargs})
            return hybrid_result

    class _Router:
        def __init__(self) -> None:
            self.calls = []
            self.client = _TenantClient()

        def for_tenant(self, tenant_id, tenant_schema=None):
            self.calls.append((tenant_id, tenant_schema))
            return self.client

    router = _Router()
    monkeypatch.setattr("ai_core.nodes.retrieve._get_router", lambda: router)

    state = {"query": "hi", "hybrid": {}}
    params = retrieve.RetrieveInput.from_state(state)
    context = _tool_context(
        tenant_id="tenant-42",
        case_id="case-42",
        run_id="run-1",
        tenant_schema="schema-42",
    )

    retrieve.run(context, params)

    assert router.calls == [("tenant-42", "schema-42")]
    assert router.client.calls[0]["top_k"] == 5


def test_retrieve_requires_tenant_id():
    params = retrieve.RetrieveInput(query="Hello", hybrid={})
    context = _tool_context(tenant_id="", case_id="c1", run_id="run-1")
    with pytest.raises(ContextError, match="tenant_id required"):
        retrieve.run(context, params)


def test_retrieve_unknown_hybrid_key(monkeypatch):
    _patch_routing(monkeypatch)
    params = retrieve.RetrieveInput(query="Hello", hybrid={"alpha": 0.5, "unknown": 1})
    context = _tool_context(tenant_id="tenant-1", case_id="c1", run_id="run-1")
    with pytest.raises(InputError, match=r"Unknown hybrid parameter\(s\): unknown"):
        retrieve.run(context, params)


def test_retrieve_deduplicates_matches(monkeypatch):
    _patch_routing(monkeypatch)

    chunks = [
        Chunk(
            content="First",
            meta={
                "id": "doc-1",
                "chunk_id": "chunk-1",
                "score": 0.4,
                "source": "a",
                "tenant_id": "tenant-1",
                "case_id": "c1",
            },
        ),
        Chunk(
            content="Second",
            meta={
                "id": "doc-1",
                "chunk_id": "chunk-1",
                "score": 0.9,
                "source": "b",
                "extra": "x",
                "tenant_id": "tenant-1",
                "case_id": "c1",
            },
        ),
        Chunk(
            content="Third",
            meta={
                "id": "doc-1",
                "chunk_id": "chunk-2",
                "score": 0.6,
                "source": "d",
                "tenant_id": "tenant-1",
                "case_id": "c1",
            },
        ),
        Chunk(
            content="Fourth",
            meta={
                "id": "doc-2",
                "chunk_id": "chunk-3",
                "score": 0.5,
                "source": "c",
                "tenant_id": "tenant-1",
                "case_id": "c1",
            },
        ),
    ]
    hybrid_result = HybridSearchResult(
        chunks=chunks,
        vector_candidates=3,
        lexical_candidates=3,
        fused_candidates=3,
        duration_ms=0.0,
        alpha=0.7,
        min_sim=0.15,
        vec_limit=50,
        lex_limit=50,
    )

    class _Router:
        def hybrid_search(self, *_args, **_kwargs):
            return hybrid_result

    monkeypatch.setattr("ai_core.nodes.retrieve._get_router", lambda: _Router())

    state = {"query": "hello", "hybrid": {"top_k": 5}}
    params = retrieve.RetrieveInput.from_state(state)
    context = _tool_context(tenant_id="tenant-1", case_id="c1", run_id="run-1")

    result = retrieve.run(context, params)

    assert len(result.matches) == 3
    assert [match["meta"]["chunk_id"] for match in result.matches] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]
    top_match = result.matches[0]
    assert top_match["id"] == "doc-1"
    assert top_match["source"] == "b"
    assert top_match["score"] == pytest.approx(0.9)
    doc_1_chunks = [match for match in result.matches if match["id"] == "doc-1"]
    assert len(doc_1_chunks) == 2
    assert {match["meta"]["chunk_id"] for match in doc_1_chunks} == {
        "chunk-1",
        "chunk-2",
    }


def test_retrieve_diversification_allows_same_document_chunks():
    matches = [
        {
            "id": "doc-1",
            "text": "alpha beta",
            "score": 0.9,
            "source": "a",
            "meta": {"chunk_id": "chunk-1", "document_id": "doc-1"},
        },
        {
            "id": "doc-1",
            "text": "alpha beta gamma",
            "score": 0.85,
            "source": "a",
            "meta": {"chunk_id": "chunk-2", "document_id": "doc-1"},
        },
        {
            "id": "doc-2",
            "text": "alpha beta",
            "score": 0.84,
            "source": "b",
            "meta": {"chunk_id": "chunk-3", "document_id": "doc-2"},
        },
    ]

    diversified = retrieve._apply_diversification(
        list(matches),
        top_k=2,
        strength=1.0,
    )

    assert [match["meta"]["chunk_id"] for match in diversified[:2]] == [
        "chunk-1",
        "chunk-2",
    ]


def test_retrieve_raises_on_chunks_without_ids(monkeypatch):
    _patch_routing(monkeypatch)

    chunk = Chunk(content="Invalid", meta={"tenant_id": "tenant-1"})
    hybrid_result = HybridSearchResult(
        chunks=[chunk],
        vector_candidates=1,
        lexical_candidates=0,
        fused_candidates=1,
        duration_ms=0.1,
        alpha=0.7,
        min_sim=0.15,
        vec_limit=5,
        lex_limit=5,
    )

    class _Router:
        def hybrid_search(self, *_args, **_kwargs):
            return hybrid_result

    monkeypatch.setattr("ai_core.nodes.retrieve._get_router", lambda: _Router())

    params = retrieve.RetrieveInput.from_state({"query": "hi", "hybrid": {}})
    context = _tool_context(tenant_id="tenant-1", case_id="case-1", run_id="run-1")

    with pytest.raises(InconsistentMetadataError, match="reindex required"):
        retrieve.run(context, params)


def test_retrieve_raises_not_found_when_no_candidates(monkeypatch):
    _patch_routing(monkeypatch)

    hybrid_result = HybridSearchResult(
        chunks=[],
        vector_candidates=0,
        lexical_candidates=0,
        fused_candidates=0,
        duration_ms=0.0,
        alpha=0.5,
        min_sim=0.1,
        vec_limit=10,
        lex_limit=10,
    )

    class _Router:
        def hybrid_search(self, *_args, **_kwargs):
            return hybrid_result

    monkeypatch.setattr("ai_core.nodes.retrieve._get_router", lambda: _Router())

    params = retrieve.RetrieveInput.from_state({"query": "missing", "hybrid": {}})
    context = _tool_context(tenant_id="tenant-1", case_id="case-1", run_id="run-1")

    with pytest.raises(NotFoundError, match="No matching documents"):
        retrieve.run(context, params)


def _mock_call(called):
    def inner(label, prompt, metadata, **kwargs):
        called["label"] = label
        called["prompt"] = prompt
        called["meta"] = metadata
        return {"text": "resp", "usage": {}, "model": "m"}

    return inner


def test_compose_masks_and_sets_version(monkeypatch):
    called = {}

    def _mock_compose_call(label, prompt, metadata, **kwargs):
        called["label"] = label
        called["prompt"] = prompt
        called["meta"] = metadata
        called["response_format"] = kwargs.get("response_format")
        return {
            "text": (
                '{"reasoning":{"analysis":"A","gaps":[]},"answer_markdown":"resp",'
                '"used_sources":[],"suggested_followups":[]}'
            ),
            "usage": {},
            "model": "m",
        }

    monkeypatch.setattr("ai_core.llm.client.call", _mock_compose_call)
    state = {
        "question": "Number 1234?",
        "snippets": [{"text": "Answer", "source": "s"}],
    }
    prompt = load("retriever/answer")
    snippets_text = compose._format_snippet_context(state["snippets"])
    expected_prompt = mask_prompt(
        f"{prompt['text']}\n\nQuestion: {state['question']}\nContext:\n{snippets_text}"
    )
    context = _tool_context(tenant_id="t1", case_id="c1", run_id="run-1")
    params = compose.ComposeInput(
        question=state["question"],
        snippets=state["snippets"],
    )
    result = compose.run(context, params)
    assert called["label"] == "synthesize"
    assert called["prompt"] == expected_prompt
    assert called["meta"]["prompt_version"] == "v2"
    assert called["response_format"] is None
    assert result.answer == "resp"


def test_extract_classify_assess(monkeypatch):
    called = {}
    monkeypatch.setattr("ai_core.llm.client.call", _mock_call(called))
    context = _tool_context(tenant_id="t1", case_id="c1", run_id="run-1")
    params = extract.ExtractInput(text="Fact 42")
    extract_prompt = mask_prompt(f"{load('extract/items')['text']}\n\n{params.text}")
    extract_result = extract.run(context, params)
    assert called["label"] == "extract"
    assert called["prompt"] == extract_prompt
    assert extract_result.items == "resp"

    classify_prompt = mask_prompt(
        f"{load('classify/mitbestimmung')['text']}\n\n{params.text}"
    )
    classify_result = classify.run(
        context,
        classify.ClassifyInput(text=params.text),
    )
    assert called["label"] == "classify"
    assert called["prompt"] == classify_prompt
    assert classify_result.classification == "resp"

    assess_prompt = mask_prompt(f"{load('assess/risk')['text']}\n\n{params.text}")
    assess_result = assess.run(
        context,
        assess.AssessInput(text=params.text),
    )
    assert called["label"] == "analyze"
    assert called["prompt"] == assess_prompt
    assert assess_result.risk == "resp"


def test_prompt_runner_default(monkeypatch):
    called = {}

    def fake_load(alias):
        called["alias"] = alias
        return {"text": "Prompt", "version": "v42"}

    def fake_mask(value, **kwargs):
        called["masked"] = value
        return value

    def fake_call(label, prompt, metadata):
        called["label"] = label
        called["prompt"] = prompt
        called["meta"] = metadata
        return {"text": "resp"}

    monkeypatch.setattr("ai_core.nodes._prompt_runner.load", fake_load)
    monkeypatch.setattr("ai_core.nodes._prompt_runner.mask_prompt", fake_mask)
    monkeypatch.setattr("ai_core.nodes._prompt_runner.client.call", fake_call)

    context = _tool_context(tenant_id="t1", case_id="c1", run_id="run-1")
    result = run_prompt_node(
        trace_name="unit",
        prompt_alias="scope/test",
        llm_label="label",
        context=context,
        text="Sensitive",
    )

    assert called["alias"] == "scope/test"
    assert called["label"] == "label"
    assert called["prompt"].endswith("\n\nSensitive")
    assert called["meta"]["prompt_version"] == "v42"
    assert result.value == "resp"
    assert result.prompt_version == "v42"
    assert result.metadata == {}


def test_prompt_runner_with_result_shaper(monkeypatch):
    monkeypatch.setattr(
        "ai_core.nodes._prompt_runner.load",
        lambda alias: {"text": "Prompt", "version": "v1"},
    )

    monkeypatch.setattr(
        "ai_core.nodes._prompt_runner.mask_prompt", lambda value, **kwargs: value
    )

    def fake_call(label, prompt, metadata):
        return {"text": "resp", "usage": {}}

    monkeypatch.setattr("ai_core.nodes._prompt_runner.client.call", fake_call)

    context = _tool_context(tenant_id="t1", case_id="c1", run_id="run-1")
    result = run_prompt_node(
        trace_name="unit",
        prompt_alias="alias",
        llm_label="label",
        context=context,
        text="",
        result_shaper=lambda result: (result["text"].upper(), {"raw": result["text"]}),
    )

    assert result.value == "RESP"
    assert result.metadata["raw"] == "resp"
    assert result.prompt_version == "v1"


def test_prompt_runner_guardrail_emits_event(monkeypatch, settings):
    settings.AI_GUARDRAIL_SAMPLE_ALLOWLIST = ["rule-allow"]

    monkeypatch.setattr(
        "ai_core.nodes._prompt_runner.load",
        lambda alias: {"text": "Prompt", "version": "v1"},
    )
    monkeypatch.setattr(
        "ai_core.nodes._prompt_runner.mask_prompt", lambda value, **kwargs: value
    )
    monkeypatch.setattr(
        "ai_core.nodes._prompt_runner.mask_response", lambda value, **kwargs: value
    )
    monkeypatch.setattr(
        "ai_core.nodes._prompt_runner.client.call",
        lambda label, prompt, metadata: {"text": "resp"},
    )

    events = []
    observations = []

    monkeypatch.setattr("ai_core.nodes._prompt_runner.emit_event", events.append)
    monkeypatch.setattr(
        "ai_core.nodes._prompt_runner.update_observation",
        lambda **fields: observations.append(fields),
    )
    monkeypatch.setattr("ai_core.nodes._prompt_runner.random.random", lambda: 0.1)

    context = _tool_context(tenant_id="t1", case_id="c1", run_id="run-1")
    _ = run_prompt_node(
        trace_name="unit",
        prompt_alias="alias",
        llm_label="label",
        context=context,
        text="",
        result_shaper=lambda result: (
            result["text"],
            {
                "guardrail": {
                    "rule_id": "rule-allow",
                    "outcome": "blocked",
                    "tool_blocked": True,
                    "reason_code": "PII",
                    "redactions": ["[REDACTED: email]"],
                }
            },
        ),
    )

    assert observations == [{"metadata": {"node.branch_taken": "blocked"}}]
    assert events == [
        {
            "event": "guardrail.result",
            "rule_id": "rule-allow",
            "outcome": "blocked",
            "tool_blocked": True,
            "reason_code": "PII",
            "redactions": ["[REDACTED: email]"],
        }
    ]


def test_prompt_runner_guardrail_sampling_skips_event(monkeypatch, settings):
    settings.AI_GUARDRAIL_SAMPLE_ALLOWLIST = ["rule-allow"]

    monkeypatch.setattr(
        "ai_core.nodes._prompt_runner.load",
        lambda alias: {"text": "Prompt", "version": "v1"},
    )
    monkeypatch.setattr(
        "ai_core.nodes._prompt_runner.mask_prompt", lambda value, **kwargs: value
    )
    monkeypatch.setattr(
        "ai_core.nodes._prompt_runner.mask_response", lambda value, **kwargs: value
    )
    monkeypatch.setattr(
        "ai_core.nodes._prompt_runner.client.call",
        lambda label, prompt, metadata: {"text": "resp"},
    )

    events = []
    observations = []

    monkeypatch.setattr("ai_core.nodes._prompt_runner.emit_event", events.append)
    monkeypatch.setattr(
        "ai_core.nodes._prompt_runner.update_observation",
        lambda **fields: observations.append(fields),
    )
    monkeypatch.setattr("ai_core.nodes._prompt_runner.random.random", lambda: 0.9)

    metadata = {
        "guardrail": {
            "rule_id": "rule-allow",
            "outcome": "allow",
            "tool_blocked": False,
        }
    }

    context = _tool_context(tenant_id="t1", case_id="c1", run_id="run-1")
    _ = run_prompt_node(
        trace_name="unit",
        prompt_alias="alias",
        llm_label="label",
        context=context,
        text="",
        metadata=metadata,
    )

    assert observations == [{"metadata": {"node.branch_taken": "allow"}}]
    assert events == []


def test_draft_blocks(monkeypatch):
    called = {}
    monkeypatch.setattr("ai_core.llm.client.call", _mock_call(called))
    context = _tool_context(tenant_id="t1", case_id="c1", run_id="run-1")
    result = draft_blocks.run(context, draft_blocks.DraftBlocksInput())
    assert called["label"] == "draft"
    # all three prompt segments should be present
    assert "Systembeschreibung" in called["prompt"]
    assert "Funktionsliste" in called["prompt"]
    assert "Standard-Klauselvorschläge" in called["prompt"]
    assert result.draft == "resp"


def test_needs_mapping():
    context = _tool_context(tenant_id="t1", case_id="c1", run_id="run-1")
    params = needs.NeedsInput(info_state={"purpose": "Acme", "extra": "foo"})
    result = needs.run(context, params)
    assert result.filled == ["purpose"]
    assert result.missing == ["deployment_model", "main_components"]
    assert result.ignored == ["extra"]


def test_tracing_called(monkeypatch):
    spans: list[str] = []

    class FakeContext:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self) -> None:
            spans.append(f"enter:{self.name}")
            return None

        def __exit__(self, exc_type, exc, tb) -> bool:
            spans.append(f"exit:{self.name}")
            return False

    class FakeTracer:
        def start_as_current_span(
            self, name: str, attributes=None, **kwargs
        ):  # noqa: D401
            spans.append(name)
            return FakeContext(name)

    monkeypatch.setattr("ai_core.infra.observability.tracing_enabled", lambda: True)
    monkeypatch.setattr("ai_core.infra.observability._get_tracer", lambda: FakeTracer())
    monkeypatch.setattr(
        "ai_core.infra.observability.update_observation", lambda **_: None
    )
    called = {}

    def _mock_compose_call(label, prompt, metadata, **kwargs):
        called["label"] = label
        called["prompt"] = prompt
        called["meta"] = metadata
        return {
            "text": (
                '{"reasoning":{"analysis":"A","gaps":[]},"answer_markdown":"resp",'
                '"used_sources":[],"suggested_followups":[]}'
            ),
            "usage": {},
            "model": "m",
        }

    monkeypatch.setattr("ai_core.llm.client.call", _mock_compose_call)
    context = _tool_context(tenant_id="t1", case_id="c1", run_id="run-1")
    params = compose.ComposeInput(question="Q?", snippets=[])
    compose.run(context, params)
    assert spans[0] == "compose"
    assert spans[1] == "enter:compose"
    assert spans[-1] == "exit:compose"
