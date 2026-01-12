# RAG Thread Registry (Backlog / Future)

Status: Draft
Date: 2026-01-09
Owner: ai_core

## Goal

Provide a DB-backed registry for chat threads so the UI can list/switch threads
and reuse the same backend for single-thread and multi-thread experiences.

## Scope

- Implement a thread registry service in `ai_core` (API-first, UI-agnostic).
- Add a DB-backed store for thread metadata (minimal CRUD + list).
- Keep graph/tool contracts unchanged; use existing `BusinessContext.thread_id`.
- Dev workbench keeps manual thread input until registry is ready.

## Non-Goals

- No new headers or meta keys beyond existing `thread_id`.
- No automatic migration of legacy chat history (manual threads only for now).
- No changes to checkpointer semantics; registry is only metadata + discovery.

## Proposed Service Interface (Draft)

- `create_thread(context, title=None, scope=None) -> ThreadRecord`
- `list_threads(context, scope=None, limit=50) -> list[ThreadRecord]`
- `get_thread(context, thread_id) -> ThreadRecord | None`
- `update_thread(context, thread_id, title=None, archived=None) -> ThreadRecord`

`scope` should support existing business dimensions:
- global (no case/collection)
- case-only
- collection-only
- case + collection

## Data Model Sketch (Draft)

- `thread_id` (UUID, primary key)
- `tenant_id` (UUID, required)
- `case_id` (optional)
- `collection_id` (optional)
- `created_by_user_id` (optional)
- `title` (optional)
- `created_at`, `updated_at`, `last_message_at`
- `archived` (bool)
- `metadata` (JSON, optional)

## Acceptance Criteria

- Threaded UI can create and list threads from the registry; selecting a thread
  uses its `thread_id` for RAG requests and restores its checkpointer history.
- Single-thread UI uses a fixed `thread_id` and does not require registry calls.
- Registry filters are tenant-scoped and optionally case/collection-scoped.
- No new IDs/headers/meta keys are introduced without explicit approval.

## Open Questions

- Default title strategy (user-supplied vs auto-generated).
- Retention/archival policy and ownership/visibility rules.
- Whether workflow_id should be included in scope (future).

## Dependencies

- BusinessContext already includes `thread_id` (Phase 3 decision).
- Auth rules for listing threads (case/collection membership) are TBD.
