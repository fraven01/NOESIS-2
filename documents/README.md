# Documents App

This application manages the lifecycle, storage, and metadata of documents within the system.

## Overview

The documents app provides:
- **Document Storage**: Multi-tenant document repository with blob storage abstraction
- **Lifecycle Management**: State tracking (pending → ingesting → embedded → active)
- **User Attribution**: Document ownership and activity tracking (see [User Integration](#user-integration))
- **Format Support**: PDF, DOCX, XLSX, PPTX, HTML, Markdown, Images (with OCR)
- **RAG Integration**: Semantic chunking, embedding, and hybrid search
- **Multi-Tenancy**: Schema-per-tenant isolation with case-based access control

## User Integration

**Status**: Roadmap (USER-MANAGEMENT branch completed)

The document repository integrates with the user management system to provide:
- User attribution (`created_by`, `updated_by` on documents, sourced from `AuditMeta.created_by_user_id`)
- Document activity tracking (downloads/uploads; views/searches pending)
- Document-level permissions (fine-grained access control)
- User preferences (favorites, recent documents, saved searches)
- Collaboration features (comments, annotations, @mentions)
- In-app notifications for mentions and saved search alerts (Phase 4a)

Ownership uses `AuditMeta.created_by_user_id` (not `initiated_by_user_id`) and is preserved
through S2S hops via audit meta.

**Documentation**:
- **Architecture**: [`docs/architecture/user-document-integration.md`](../docs/architecture/user-document-integration.md)
- **Roadmap**: [`roadmap/document-repo-user-integration.md`](../roadmap/document-repo-user-integration.md)

**Implementation Phases**:
1. **Phase 1** (MVP-Critical): Direct user attribution on documents
2. **Phase 2** (Compliance): Document activity tracking
3. **Phase 3** (Strategic): Document-level permissions
4. **Phase 4** (Post-MVP): User preferences & collaboration features

See the roadmap for detailed implementation plan, code locations, and acceptance criteria.

## Media Type Handling

Correct MediaType (MIME type) handling is critical for proper file processing and display. The system follows a strict propagation flow to ensure the correct type is preserved from upload to display.

### Propagation Flow

1. **Upload**: The browser sends the `Content-Type` header (e.g., `application/pdf`).
2. **Upload Worker**:
    * Captures `content_type` from the upload.
    * Stores it in `FileBlob.media_type`.
    * **Crucially**, explicitly sets `pipeline_config["media_type"]` in the document metadata.
3. **Ingestion Graph**: Preserves the `pipeline_config`.
4. **Document Space Service**: Infers the display MediaType based on a priority list.

### Inference Priority

When determining the MediaType for a document (e.g., for UI display), the `DocumentSpaceService` checks sources in the following order:

1. **Pipeline Config** (`doc.meta.pipeline_config["media_type"]`) - **Highest Priority**. This is the authoritative type set during ingestion.
2. **External Reference** (if available).
3. **Metadata** (`doc.meta.metadata["content_type"]`).
4. **Origin URI** (Extension detection from URL).
5. **Title** (Extension detection from filename).

### Fallback Strategy

If no MediaType can be inferred from the above sources:

* **Default**: `application/octet-stream` (Neutral binary type).
* **Legacy Behavior**: Previously fell back to `text/html`, which caused display issues for PDFs. This has been corrected.

### Debugging

If a document displays with the wrong MediaType (e.g., `text/html` instead of `application/pdf`):

1. Check `documents/upload_worker.py`: Ensure `FileBlob` and `pipeline_config` are being populated with the `content_type`.
2. Check the database: Inspect `document.meta['pipeline_config']`.
3. Check `DocumentSpaceService`: Verify `_infer_media_type_from_doc` logic.

### Supported Types & Parsers

The system includes dedicated parsers for the following media types:

* **PDF**: `application/pdf`
* **Word**: `application/vnd.openxmlformats-officedocument.wordprocessingml.document`
* **Excel**:
  * `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` (XLSX)
  * `application/vnd.ms-excel` (Legacy XLS)
* **PowerPoint**: `application/vnd.openxmlformats-officedocument.presentationml.presentation`
* **Markdown**: `text/markdown`, `text/x-markdown`
* **HTML**: `text/html`, `application/xhtml+xml`
* **Text**: `text/plain`
* **Images**:
  * `image/jpeg`, `image/png`, `image/webp`
  * `image/gif`, `image/tiff`, `image/bmp`
