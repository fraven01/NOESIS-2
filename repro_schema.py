from ai_core.schemas import CrawlerRunRequest
from pydantic import ValidationError

print(
    f"CrawlerRunRequest.collection_id type: {CrawlerRunRequest.model_fields['collection_id'].annotation}"
)

data = {
    "workflow_id": "crawler-demo",
    "mode": "live",
    "origins": [{"url": "https://example.com"}],
    "collection_id": "00000000-0000-0000-0000-000000000000",
}

try:
    req = CrawlerRunRequest.model_validate(data)
    print("Validation SUCCESS")
    print(req.collection_id)
except ValidationError as e:
    print("Validation FAILED")
    print(e.json())
