from ai_core.graphs.technical import info_intake, retrieval_augmented_generation

META = {
    "scope_context": {
        "tenant_id": "t1",
        "case_id": "c1",
        "trace_id": "tr",
        "invocation_id": "inv-1",
        "run_id": "run-1",
    }
}


def test_info_intake_adds_meta():
    state, result = info_intake.run({}, META)
    assert state["meta"] == META
    assert result["tenant_id"] == META["scope_context"]["tenant_id"]


def test_retrieval_graph_build_exposes_runner():
    graph = retrieval_augmented_generation.build_graph()
    assert hasattr(graph, "run"), "retrieval graph must provide a run method"
