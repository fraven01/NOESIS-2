# Dokumenten-Contracts Referenz

Diese Referenz bündelt alle Felder und Fehlermeldungen der Dokument-Verträge. Sie ergänzt die [Architekturübersicht](../architektur/documents-subsystem.md) um eine feldgenaue Sicht und verweist aufeinander über interne Anker.

## Schnellnavigation
- [DocumentRef](#documentref)
- [DocumentMeta](#documentmeta)
- [BlobLocator](#bloblocator)
  - [FileBlob](#fileblob)
  - [InlineBlob](#inlineblob)
  - [ExternalBlob](#externalblob)
- [NormalizedDocument](#normalizeddocument)
- [AssetRef](#assetref)
- [Asset](#asset)
- [Hinweise](#hinweise)

## DocumentRef
Zweck: Identifiziert ein Dokument innerhalb eines Tenants und optional einer Collection. Wird von [NormalizedDocument](#normalizeddocument) und dem Repository verwendet.

| Feld | Typ | Pflicht | Constraints | Fehlercodes |
| --- | --- | --- | --- | --- |
| tenant_id | str | Ja | NFKC, Trim, Invisibles entfernen, Länge ≤ 128 | `tenant_empty`, `tenant_too_long` |
| workflow_id | str | Ja | NFKC, Trim, Invisibles entfernen, Regex `[A-Za-z0-9._-]+`, Länge ≤ 128 | `workflow_empty`, `workflow_invalid_char`, `workflow_too_long` |
| document_id | UUID | Ja | Akzeptiert UUID-Objekt oder String (Trim); muss gültig sein | `uuid_empty`, `uuid_invalid`, `uuid_type` |
| collection_id | UUID | Nein | Optional, gleiche Normalisierung wie `document_id` | `uuid_empty`, `uuid_invalid`, `uuid_type` |
| version | str | Nein | Regex `[A-Za-z0-9._-]+`, Länge ≤ 64, Leere Strings ⇒ `None` | `version_too_long`, `version_invalid` |

### Beispiele
**Gültig**
```json
{
  "tenant_id": "acme",
  "workflow_id": "ingest-2024",
  "document_id": "5c6a9f0e-6d45-4f58-9a51-5c9045e40f6d",
  "collection_id": "00000000-0000-0000-0000-000000000123",
  "version": "v2.1"
}
```

**Ungültig (Version)**
```json
{
  "tenant_id": "acme",
  "workflow_id": "ingest-2024",
  "document_id": "5c6a9f0e-6d45-4f58-9a51-5c9045e40f6d",
  "version": "release 1"
}
```
→ Fehlercode `version_invalid`.

## DocumentMeta
Zweck: Beschreibt Titel, Sprache, Tags und externe Referenzen. Wird von [NormalizedDocument](#normalizeddocument) eingebettet.

| Feld | Typ | Pflicht | Constraints | Fehlercodes |
| --- | --- | --- | --- | --- |
| tenant_id | str | Ja | Wie [DocumentRef](#documentref) | `tenant_empty`, `tenant_too_long` |
| workflow_id | str | Ja | Wie [DocumentRef](#documentref) | `workflow_empty`, `workflow_invalid_char`, `workflow_too_long` |
| title | str | Nein | Nach Normalisierung Länge ≤ 256, Leerwerte ⇒ `None` | `title_too_long` |
| language | str | Nein | BCP-47-ähnlich, Segmente 1–8 alphanumerisch, keine doppelten/trailenden `-` | `language_invalid` |
| tags | list[str] | Nein | Normalisierung, Deduplizierung, stabile Sortierung, Regex `[A-Za-z0-9._-]+`, Länge ≤ 64 | `tags_type`, `tag_invalid`, `tag_too_long` |
| origin_uri | str | Nein | Normalisierung, Leerwerte ⇒ `None` | — |
| crawl_timestamp | datetime | Nein | Muss tz-aware sein, wird nach UTC konvertiert | `crawl_timestamp_naive` |
| external_ref | dict[str,str] | Nein | ≤ 16 Einträge, Schlüssel ≤ 128, Werte ≤ 512, beide normalisiert | `external_ref_too_many`, `external_ref_key_empty`, `external_ref_key_too_long`, `external_ref_value_empty`, `external_ref_value_too_long` |

### Beispiele
**Gültig**
```json
{
  "tenant_id": "acme",
  "workflow_id": "ingest-2024",
  "title": "Monthly Revenue Report",
  "language": "en-US",
  "tags": ["finance", "q1"],
  "origin_uri": "https://source.example/reports/2024-03",
  "crawl_timestamp": "2024-03-01T12:00:00+00:00",
  "external_ref": {
    "provider": "confluence",
    "id": "PAGE-12345"
  }
}
```

**Ungültig (Sprache & External Ref)**
```json
{
  "tenant_id": "acme",
  "workflow_id": "ingest-2024",
  "language": "--de",
  "external_ref": {
    "": "value"
  }
}
```
→ Fehlercodes `language_invalid` und `external_ref_key_empty`.

## BlobLocator
Union aus drei Varianten, die den Payload eines Dokuments oder Assets beschreiben. Wird von [NormalizedDocument](#normalizeddocument) und [Asset](#asset) verwendet.

| Variante | Typ-Feld | Zweck | Besondere Fehlercodes |
| --- | --- | --- | --- |
| [FileBlob](#fileblob) | `"file"` | Verweist auf persistierte Payload in Storage | `uri_empty`, `sha256_invalid`, `size_negative` |
| [InlineBlob](#inlineblob) | `"inline"` | Transportiert Base64-kodierte Bytes für spätere Persistenz | `sha256_invalid`, `size_negative`, `inline_size_mismatch`, `inline_checksum_mismatch` |
| [ExternalBlob](#externalblob) | `"external"` | Referenziert externe Speicherpfade | Literal-Validierung (Pydantic), `sha256_invalid` (falls gesetzt) |

### FileBlob
| Feld | Typ | Pflicht | Constraints | Fehlercodes |
| --- | --- | --- | --- | --- |
| type | Literal["file"] | Ja | Discriminator | — |
| uri | str | Ja | Nicht leer, Normalisierung | `uri_empty` |
| sha256 | str | Ja | 64-stelliger Hex-String | `sha256_invalid` |
| size | int | Ja | ≥ 0 | `size_negative` |

**Gültig**
```json
{
  "type": "file",
  "uri": "memory://blob-001",
  "sha256": "4d186321c1a7f0f354b297e8914ab24083dc2c4c795c305a89602a3e0e4e0fef",
  "size": 14
}
```

**Ungültig (Checksumme)**
```json
{
  "type": "file",
  "uri": "memory://blob-001",
  "sha256": "abcd",
  "size": 14
}
```
→ Fehlercode `sha256_invalid`.

### InlineBlob
| Feld | Typ | Pflicht | Constraints | Fehlercodes |
| --- | --- | --- | --- | --- |
| type | Literal["inline"] | Ja | Discriminator | — |
| media_type | str | Ja | Lowercase `type/subtype`, keine Parameter | `media_type_empty`, `media_type_invalid` |
| base64 | str | Ja | Gültiges Base64, Trim | `base64_invalid` |
| sha256 | str | Ja | 64-stelliger Hex-String | `sha256_invalid` |
| size | int | Ja | ≥ 0, muss zum dekodierten Payload passen | `size_negative`, `inline_size_mismatch`, `inline_checksum_mismatch` |

**Gültig**
```json
{
  "type": "inline",
  "media_type": "text/plain",
  "base64": "SGVsbG8=",
  "sha256": "64ec88ca00b268e5ba1a35678a1b5316d212f4f366b247724e2e1cda5fb0b3af",
  "size": 5
}
```

**Ungültig (Base64 & Größe)**
```json
{
  "type": "inline",
  "media_type": "text/plain",
  "base64": "SGVsbG8=",
  "sha256": "64ec88ca00b268e5ba1b5316d212f4f366b247724e2e1cda5fb0b3af",
  "size": 6
}
```
→ Fehlercode `inline_size_mismatch`.

### ExternalBlob
| Feld | Typ | Pflicht | Constraints | Fehlercodes |
| --- | --- | --- | --- | --- |
| type | Literal["external"] | Ja | Discriminator | — |
| kind | Literal["http", "https", "s3", "gcs"] | Ja | Erlaubte Zielsysteme | `literal_error` (Pydantic) |
| uri | str | Ja | Nicht leer, Normalisierung | `uri_empty` |
| sha256 | str | Nein | Falls gesetzt: 64-stelliger Hex-String | `sha256_invalid` |

**Gültig**
```json
{
  "type": "external",
  "kind": "https",
  "uri": "https://cdn.example/document.pdf",
  "sha256": "d4735e3a265e16eee03f59718b9b5d03019c07d8b6c5d2dc5c2f932a6c2f1e3f"
}
```

**Ungültig (Kind)**
```json
{
  "type": "external",
  "kind": "ftp",
  "uri": "ftp://legacy",
  "sha256": "d4735e3a265e16eee03f59718b9b5d03019c07d8b6c5d2dc5c2f932a6c2f1e3f"
}
```
→ Fehlercode `literal_error`.

## NormalizedDocument
Zweck: Vollständiger Dokument-Contract inkl. Blob und Assets. Verweist auf [DocumentRef](#documentref), [DocumentMeta](#documentmeta), [BlobLocator](#bloblocator) und [Asset](#asset).

| Feld | Typ | Pflicht | Constraints | Fehlercodes |
| --- | --- | --- | --- | --- |
| ref | DocumentRef | Ja | Muss identischen Tenant/Workflow liefern wie `meta` und Assets | `meta_tenant_mismatch`, `meta_workflow_mismatch`, `asset_tenant_mismatch`, `asset_document_mismatch`, `asset_collection_mismatch`, `asset_workflow_mismatch` |
| meta | DocumentMeta | Ja | Siehe [DocumentMeta](#documentmeta) | `meta_workflow_mismatch` |
| blob | BlobLocator | Ja | Siehe [BlobLocator](#bloblocator) | — |
| checksum | str | Ja | 64-stelliger Hex-String; im Strict-Mode == Blob-SHA | `checksum_invalid`, `document_checksum_missing`, `document_checksum_mismatch` |
| created_at | datetime | Ja | UTC, tz-aware | `created_at_naive` |
| source | Literal[...] | Nein | `upload`, `crawler`, `integration`, `other` | — |
| assets | list[Asset] | Nein | Default `[]`, alle Assets müssen Tenant/Document matchen | Siehe [Asset](#asset) |

### Beispiele
**Gültig (Upload)**
```json
{
  "ref": {
    "tenant_id": "acme",
    "workflow_id": "ingest-2024",
    "document_id": "c7f8b4f4-1b7b-4ad2-9da6-0f8df1d96c90"
  },
  "meta": {
    "tenant_id": "acme",
    "workflow_id": "ingest-2024",
    "title": "Product Brochure",
    "language": "en",
    "tags": ["marketing"]
  },
  "blob": {
    "type": "file",
    "uri": "memory://blob-42",
    "sha256": "0f343b0931126a20f133d67c2b018a3b6d91ef1ed97e1a6f760fd9e3c8f2b518",
    "size": 2048
  },
  "checksum": "0f343b0931126a20f133d67c2b018a3b6d91ef1ed97e1a6f760fd9e3c8f2b518",
  "created_at": "2024-05-02T10:15:00+00:00",
  "source": "upload",
  "assets": []
}
```

**Ungültig (Asset-Mismatch)**
```json
{
  "ref": {
    "tenant_id": "acme",
    "document_id": "c7f8b4f4-1b7b-4ad2-9da6-0f8df1d96c90"
  },
  "meta": {
    "tenant_id": "acme"
  },
  "blob": {
    "type": "file",
    "uri": "memory://blob-42",
    "sha256": "0f343b0931126a20f133d67c2b018a3b6d91ef1ed97e1a6f760fd9e3c8f2b518",
    "size": 2048
  },
  "checksum": "0f343b0931126a20f133d67c2b018a3b6d91ef1ed97e1a6f760fd9e3c8f2b518",
  "created_at": "2024-05-02T10:15:00+00:00",
  "assets": [
    {
      "ref": {
        "tenant_id": "other",
        "workflow_id": "ingest-2024",
        "asset_id": "54cc8d65-a74a-4ef0-bca9-0fdb45eb3a0f",
        "document_id": "c7f8b4f4-1b7b-4ad2-9da6-0f8df1d96c90"
      },
      "media_type": "image/png",
      "blob": {
        "type": "file",
        "uri": "memory://asset-1",
        "sha256": "0f343b0931126a20f133d67c2b018a3b6d91ef1ed97e1a6f760fd9e3c8f2b518",
        "size": 512
      },
      "caption_method": "none",
      "created_at": "2024-05-02T10:15:00+00:00",
      "checksum": "0f343b0931126a20f133d67c2b018a3b6d91ef1ed97e1a6f760fd9e3c8f2b518"
    }
  ]
}
```
→ Fehlercode `asset_tenant_mismatch`.

## AssetRef
Zweck: Referenz auf ein einzelnes Asset und das dazugehörige Dokument.

| Feld | Typ | Pflicht | Constraints | Fehlercodes |
| --- | --- | --- | --- | --- |
| tenant_id | str | Ja | Wie [DocumentRef](#documentref) | `tenant_empty`, `tenant_too_long` |
| workflow_id | str | Ja | Wie [DocumentRef](#documentref) | `workflow_empty`, `workflow_invalid_char`, `workflow_too_long` |
| asset_id | UUID | Ja | Akzeptiert String/UUID | `uuid_empty`, `uuid_invalid`, `uuid_type` |
| document_id | UUID | Ja | Muss zur Dokument-ID passen | `uuid_empty`, `uuid_invalid`, `uuid_type` |
| collection_id | UUID | Nein | Optional, übernimmt Wert aus Dokument | `uuid_empty`, `uuid_invalid`, `uuid_type` |

## Asset
Zweck: Beschreibt ein extrahiertes Asset (z. B. Bild) inklusive Kontext, Blob und Captioning.

| Feld | Typ | Pflicht | Constraints | Fehlercodes |
| --- | --- | --- | --- | --- |
| ref | AssetRef | Ja | Muss denselben Tenant/Document/Workflow wie [NormalizedDocument](#normalizeddocument) haben | `asset_tenant_mismatch`, `asset_document_mismatch`, `asset_collection_mismatch`, `asset_workflow_mismatch` |
| media_type | str | Ja | Lowercase `type/subtype`, keine Parameter | `media_type_empty`, `media_type_invalid`, `media_type_mismatch`, `media_type_guard` |
| blob | BlobLocator | Ja | Siehe [BlobLocator](#bloblocator); Inline-Blobs müssen Media-Type matchen | `media_type_mismatch`, `asset_checksum_missing`, `asset_checksum_mismatch` |
| origin_uri | str | Nein | Normalisierung, Leerwerte ⇒ `None` | — |
| page_index | int | Nein | ≥ 0 | `page_index_negative` |
| bbox | list[float] | Nein | Länge 4, Werte 0–1, x1>x0, y1>y0 | `bbox_invalid` |
| context_before | str | Nein | ≤ 2048 Bytes (UTF-8-sicher) | — |
| context_after | str | Nein | ≤ 2048 Bytes | — |
| ocr_text | str | Nein | ≤ 8192 Bytes | — |
| text_description | str | Nein | ≤ 2048 Bytes | — |
| caption_method | Literal[...] | Ja | `vlm_caption`, `ocr_only`, `manual`, `none` | `caption_model_required`, `caption_confidence_required` (bei `vlm_caption`) |
| caption_model | str | Nein | Normalisiert, Leerwerte ⇒ `None` | — |
| caption_confidence | float | Nein | 0.0–1.0 | `caption_confidence_range` |
| created_at | datetime | Ja | UTC, tz-aware | `created_at_naive` |
| checksum | str | Ja | 64-stelliger Hex-String | `checksum_invalid`, `asset_checksum_missing`, `asset_checksum_mismatch` |

### Beispiele
**Gültig (Captioned Image)**
```json
{
  "ref": {
    "tenant_id": "acme",
    "workflow_id": "ingest-2024",
    "asset_id": "54cc8d65-a74a-4ef0-bca9-0fdb45eb3a0f",
    "document_id": "c7f8b4f4-1b7b-4ad2-9da6-0f8df1d96c90"
  },
  "media_type": "image/png",
  "blob": {
    "type": "file",
    "uri": "memory://asset-1",
    "sha256": "b1946ac92492d2347c6235b4d2611184f23a0a0d6a688b8b0f2e64d5a458d7f8",
    "size": 512
  },
  "bbox": [0.05, 0.10, 0.45, 0.60],
  "context_before": "Total revenue by region",
  "context_after": "Values in million EUR",
  "caption_method": "vlm_caption",
  "caption_model": "stub-v1",
  "caption_confidence": 0.82,
  "created_at": "2024-05-02T10:15:00+00:00",
  "checksum": "b1946ac92492d2347c6235b4d2611184f23a0a0d6a688b8b0f2e64d5a458d7f8"
}
```

**Ungültig (Bounding Box)**
```json
{
  "ref": {
    "tenant_id": "acme",
    "asset_id": "54cc8d65-a74a-4ef0-bca9-0fdb45eb3a0f",
    "document_id": "c7f8b4f4-1b7b-4ad2-9da6-0f8df1d96c90"
  },
  "media_type": "image/png",
  "blob": {
    "type": "file",
    "uri": "memory://asset-1",
    "sha256": "b1946ac92492d2347c6235b4d2611184f23a0a0d6a688b8b0f2e64d5a458d7f8",
    "size": 512
  },
  "bbox": [0.4, 0.4, 0.2, 0.5],
  "caption_method": "none",
  "created_at": "2024-05-02T10:15:00+00:00",
  "checksum": "b1946ac92492d2347c6235b4d2611184f23a0a0d6a688b8b0f2e64d5a458d7f8"
}
```
→ Fehlercode `bbox_invalid`.

## Hinweise
- Alle Zeitstempel sind UTC und tz-aware; naive Werte führen zu `created_at_naive` bzw. `crawl_timestamp_naive`.
- Checksummen verwenden immer hex-kodierte SHA-256-Werte. Strikter Modus verlangt Gleichheit zwischen `checksum` und Blob-SHA (`document_checksum_mismatch`, `asset_checksum_mismatch`).
- Bounding Boxes sind normiert (0–1) und folgen der Reihenfolge `[x0, y0, x1, y1]`.
- Media-Types werden immer kleingeschrieben und akzeptieren keine Parameter (`media_type_invalid` bei Eingaben wie `text/html; charset=utf-8`).
- Fehlercodes sind stabil und können im CLI sowie in API-Antworten gespiegelt werden.
