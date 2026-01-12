# RAG Citations Backlog

## Goal

Improve citation quality and UX beyond basic source labels by adding page/coords
and deep links into the document viewer.

## Work Items

- [ ] **Snippet metadata enrichment**: ensure retrieval snippets carry
  `document_id`, `page`, and `coords` (pointers: `ai_core/nodes/retrieve.py`,
  `ai_core/rag/vector_client.py`; acceptance: snippet payloads include these
  fields when available from chunk metadata).
- [ ] **Viewer deep links**: build URLs that open the document viewer at
  `document_id` + `page/coords` (pointer: `documents/views.py`, UI templates;
  acceptance: citation link opens viewer on the referenced page).
- [ ] **Citation UI component**: replace the simple list with a compact citation
  block (source label + page + highlight) and show up to N citations per answer
  (pointers: `theme/templates/theme/partials/chat_message.html`,
  `theme/templates/theme/workbench.html`; acceptance: citations render with page
  numbers and jump links).
- [ ] **Fallback behavior**: if page/coords are missing, keep download link only
  (acceptance: no UI breakage, clear fallback label).
