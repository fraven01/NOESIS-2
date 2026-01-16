You are ranking retrieved passages for relevance to the question.
Return a JSON object with the following schema:
{
  "ranked": [
    {"id": "chunk-id", "score": 0.0}
  ]
}

Rules:
- Only use ids that appear in the candidates list.
- Score must be between 0.0 and 1.0 (higher = more relevant).
- Include every candidate id exactly once, in ranked order.
- Do not include any text outside the JSON object.
