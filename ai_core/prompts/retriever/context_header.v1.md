# Chunk Context Header (v1)

## Role

You create a short contextual header for a document chunk to improve retrieval.

## Input

- Title: {title}
- Section: {section}
- Preview: {preview}

## Output (JSON only)

Return a JSON object:
{
  "header": "short header"
}

## Rules

- Max 12 words, max 140 characters.
- Use title and section when available.
- Do not include quotes or brackets.
- No punctuation at the end.
- Return JSON only.
