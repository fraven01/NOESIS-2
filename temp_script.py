import json

import django


def check(label, value):
    try:
        json.dumps(value)
    except TypeError as exc:
        print(f"{label} not json serializable: {exc}")


def main():
    django.setup()

    from ai_core.graphs.crawler_ingestion_graph import CrawlerIngestionGraph
    from ai_core.tests.graphs.test_crawler_ingestion_graph import _build_state

    graph = CrawlerIngestionGraph()
    state = _build_state()
    state = graph.start_crawl(state)
    updated_state, result = graph.run(state, {"tenant_id": "tenant", "case_id": "case"})

    check("result", result)
    check("transitions", updated_state.get("transitions", {}))
    check("summary", updated_state.get("summary"))
    check("artifacts", updated_state.get("artifacts"))


if __name__ == "__main__":
    main()
