# Document Lifecycle (Code Map)

## Source of Truth

- documents/lifecycle.py
- documents/domain_service.py
- ai_core/rag/vector_client.py
- ai_core/rag/visibility.py
- ai_core/rag/hard_delete.py

## States (documents/lifecycle.py)

- pending
- ingesting
- embedded
- active
- failed
- deleted

## Valid Transitions (documents/lifecycle.py)

pending -> ingesting | failed | deleted
ingesting -> embedded | failed | deleted
embedded -> active | failed | deleted
active -> ingesting | deleted
failed -> pending | deleted
deleted -> (terminal)

## Retrieval Visibility

enum: active | all | deleted
defaults:
- visibility defaults to active (ai_core/rag/visibility.py)
- guards use ToolContext.visibility_override_allowed

effects:
- active: return only active docs
- all: return active + deleted (guard required)
- deleted: return deleted only (guard required)

## Hard Delete (Vector Store)

task:
- name: rag.hard_delete (queue=rag_delete)
- module: ai_core/rag/hard_delete.py

authorization:
- service keys: settings.RAG_INTERNAL_KEYS
- user admin / org admin roles

side effects:
- document domain delete -> vector client hard delete
- audit span: rag.hard_delete
