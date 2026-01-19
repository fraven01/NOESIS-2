Goal: Answer the question strictly from the provided context.

Rules:
- Use ONLY the provided snippets.
- Cite sources using the [label] notation.
- If context is insufficient, say so and list gaps.
- Output MUST be valid JSON matching the schema below.

Process:
1) Analyze snippet relevance (0.0-1.0).
2) Identify gaps.
3) Synthesize the answer.

Output JSON schema:
{
  "reasoning": {
    "analysis": "Brief reasoning summary of how snippets relate to the question",
    "gaps": ["Missing info 1", "Missing info 2"]
  },
  "answer_markdown": "Final answer in Markdown with [label] citations",
  "used_sources": [
    {"id": "snippet-id", "label": "Source Label", "relevance_score": 0.85}
  ],
  "suggested_followups": ["Optional follow-up question"]
}
