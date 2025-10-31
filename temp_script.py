import django
django.setup()
from ai_core.graphs.crawler_ingestion_graph import CrawlerIngestionGraph
from ai_core.tests.graphs.test_crawler_ingestion_graph import _build_state
import json

graph = CrawlerIngestionGraph()
state = _build_state()
state = graph.start_crawl(state)
updated_state, result = graph.run(state, {'tenant_id': 'tenant', 'case_id': 'case'})

def check(label, value):
    try:
        json.dumps(value)
    except TypeError as exc:
        print(f"{label} not json serializable: {exc}")

check('result', result)
check('transitions', updated_state.get('transitions', {}))
check('summary', updated_state.get('summary'))
check('artifacts', updated_state.get('artifacts'))
