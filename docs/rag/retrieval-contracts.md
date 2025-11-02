# Retrieval Contracts

Die Retrieval-Tools des AI Core kapseln Eingaben und Ausgaben über vier Pydantic-Modelle: `RetrieveInput`, `RetrieveRouting`, `RetrieveMeta` und `RetrieveOutput`. Dieser Leitfaden dokumentiert Pflicht- und Optionalfelder, Standardwerte, Validierungsregeln und zeigt Beispielpayloads.

Die Implementierung befindet sich unter [`ai_core/nodes/retrieve.py`](../../ai_core/nodes/retrieve.py). Für Hybrid-Parameter greift zusätzlich [`ai_core/nodes/_hybrid_params.py`](../../ai_core/nodes/_hybrid_params.py) sowie die Standardwerte in [`ai_core/settings.py`](../../ai_core/settings.py).

## RetrieveInput

`RetrieveInput` beschreibt die erlaubten Felder für Retrieval-Aufrufe. Optionalitäten und Guards werden bereits beim Bau des Models oder vor dem Router-Aufruf enforced.

| Feld | Typ | Pflicht? | Default | Validierung & Hinweise |
| --- | --- | --- | --- | --- |
| `query` | `str` | Nein | `""` | Leerzeichen werden nicht getrimmt; Leerstring löst dennoch einen Call aus. |
| `filters` | `Mapping[str, Any] \| None` | Nein | `None` | Muss eine Mapping-Instanz sein, sonst `InputError("filters must be a mapping when provided")`. Wird unverändert an `hybrid_search` durchgereicht. |
| `process` | `str \| None` | Nein | `None` | Wird normalized (`lower()` + Trim) und für Routing (`RetrieveRouting.process`) genutzt. |
| `doc_class` | `str \| None` | Nein | `None` | Wie `process`; bestimmt Routing-Regeln. |
| `collection_id` | `str \| None` | Nein | `None` | Trim auf nicht-leeren String. Geht sowohl in Routing als auch an den Vector-Store. |
| `workflow_id` | `str \| None` | Nein | `None` | Trim + Normalisierung für Routing. |
| `visibility` | `str \| None` | Nein | `None` | Unterstützt `active`, `all`, `deleted`. Fehlt der Wert oder ist er leer, wird `active` angenommen. Ungültige Werte lösen einen Fehler im Vector-Client aus. |
| `visibility_override_allowed` | `bool \| None` | Nein | `None` | Wird via `coerce_bool_flag` normalisiert. `None` übernimmt den Flag aus `ToolContext.visibility_override_allowed`. |
| `hybrid` | `Mapping[str, Any] \| None` | **Ja** | – | Pflichtfeld. Muss Mapping sein, sonst `InputError`. Unbekannte Keys führen zu `ValueError("Unknown hybrid parameter(s)")`. |
| `top_k` | `int \| None` | Nein | `None` | Optionaler Override für `hybrid.top_k`. Werte <1 werden auf `1`, Werte > `10` auf `10` begrenzt. |

### Hybrid-Parameter

Die `hybrid`-Sektion akzeptiert ausschließlich folgende Keys:

| Key | Typ | Default | Clamp/Normalisierung |
| --- | --- | --- | --- |
| `alpha` | `float` | `0.7` | Zwischen `0` und `1`. |
| `min_sim` | `float` | `0.15` | Zwischen `0` und `1`. |
| `top_k` | `int` | `5` | Mindestens `1`, höchstens `10`. |
| `vec_limit` | `int` | `50` | Mindestens `1`; beeinflusst vector candidate pool. |
| `lex_limit` | `int` | `50` | Mindestens `1`; beeinflusst lexical candidate pool. |
| `trgm_limit` | `float \| None` | `None` | Optionaler Trigram-Grenzwert, auf `0..1` begrenzt. |
| `max_candidates` | `int \| None` | `max(vec_limit, lex_limit)` | Mindestens `top_k`, `vec_limit`, `lex_limit`. |
| `diversify_strength` | `float` | `0.3` | Zwischen `0` und `1`; wird später in `1-strength` für MMR umgerechnet. |

Jede Normalisierung wird zurück in den `state` geschrieben (`HybridParameters.as_dict()`), sodass nachgelagerte Komponenten (Graph-Nodes, Telemetrie) denselben Wertestand sehen.

### Sichtbarkeitsregeln

- **Standard:** `visibility` bleibt `None`, wodurch der Vector-Store automatisch `active` verwendet.
- **Erweiterte Modi:** `all` und `deleted` sind nur sichtbar, wenn `visibility_override_allowed=True`. Der Flag kann entweder explizit im Request gesetzt oder (falls `None`) aus dem `ToolContext` geerbt werden.
- **Ablehnung:** Wird ein erweiterter Modus ohne Override-Flag angefordert, erzwingt der Client `active` und protokolliert `rag.visibility.override_denied`.

## RetrieveRouting

`RetrieveRouting` beschreibt das effektive Routing pro Aufruf:

| Feld | Beschreibung |
| --- | --- |
| `profile` | Aufgelöstes Embedding-Profil via `resolve_embedding_profile`. |
| `vector_space_id` | Zugehöriger Vector-Space (z. B. `rag/global`). |
| `process`, `doc_class`, `collection_id`, `workflow_id` | Normalisierte Selektoren, `None` wenn nicht gesetzt. |

## RetrieveMeta

`RetrieveMeta` liefert Kennzahlen zur Suche:

| Feld | Bedeutung |
| --- | --- |
| `routing` | `RetrieveRouting`-Payload (siehe oben). |
| `took_ms` | Dauer in Millisekunden (`int`, >=0). |
| `alpha` | Effektiver Alpha-Wert nach Normalisierung. |
| `min_sim` | Effektiver Similarity-Cutoff. |
| `top_k_effective` | Tatsächlich verwendetes `top_k` nach Clamping. |
| `matches_returned` | Anzahl der zurückgegebenen Chunks nach Diversifizierung. |
| `max_candidates_effective` | Finaler Kandidatenpool (>= `top_k`). |
| `vector_candidates` | Anzahl Vector-Kandidaten vor Fusion. |
| `lexical_candidates` | Anzahl Lexikal-Kandidaten vor Fusion. |
| `deleted_matches_blocked` | Count der Treffer, die wegen Sichtbarkeitsregeln ausgeschlossen wurden. |
| `visibility_effective` | Tatsächlich angewendete Sichtbarkeit (`active`, `all`, `deleted`). |
| `diversify_strength` | Eingestellter Diversifizierungsgrad (0..1). |

## RetrieveOutput

`RetrieveOutput` kapselt die finale Antwort:

- `matches`: Liste normalisierter Chunk-Metadaten. Jedes Element enthält mindestens `id`, `text`, `score`, `source`, `hash`. Weitere Metadaten bleiben unter `match["meta"]` erhalten. Parent-Kontexte werden bei Bedarf unter `meta.parents` ergänzt.
- `meta`: Instanz von `RetrieveMeta` (s.o.).

## Beispiel-Request

```json
{
  "query": "Wie lautet die Kündigungsfrist?",
  "filters": {
    "project": "compliance",
    "case_id": "case-7"
  },
  "process": "review",
  "doc_class": "policy",
  "collection_id": "hr",
  "visibility": "all",
  "visibility_override_allowed": true,
  "hybrid": {
    "alpha": 0.55,
    "min_sim": 0.35,
    "top_k": 3,
    "vec_limit": 10,
    "lex_limit": 8,
    "diversify_strength": 0.4
  }
}
```

## Beispiel-Antwort

```json
{
  "matches": [
    {
      "id": "doc-1",
      "text": "…",
      "score": 0.90,
      "source": "vector",
      "hash": "0b73…",
      "citation": "Mitarbeiterhandbuch · S.12 · Abschnitt Kündigungsfristen",
      "meta": {
        "document_id": "1111…",
        "chunk_id": "8c9d…",
        "parent_ids": ["section-1"],
        "parents": [
          {"id": "section-1", "heading": "Kündigungsfristen", "content": "…"}
        ]
      }
    }
  ],
  "meta": {
    "routing": {
      "profile": "standard",
      "vector_space_id": "rag/global",
      "process": "review",
      "doc_class": "policy",
      "collection_id": "hr",
      "workflow_id": null
    },
    "took_ms": 42,
    "alpha": 0.55,
    "min_sim": 0.35,
    "top_k_effective": 3,
    "matches_returned": 3,
    "max_candidates_effective": 10,
    "vector_candidates": 6,
    "lexical_candidates": 4,
    "deleted_matches_blocked": 1,
    "visibility_effective": "all",
    "diversify_strength": 0.4
  }
}
```

> **Hinweis:** `took_ms`, Kandidatenzahlen und Scores hängen von den realen Daten ab und dienen hier nur als Beispiel. Bei leeren Ergebnissen wird ein `NotFoundError` geworfen, sofern der Router nicht tenant-gescoped ist.

## Nutzung in LangGraph & API

- LangGraph-Nodes übergeben `RetrieveInput`-Instanzen oder `RetrieveInput.from_state(state)` an `retrieve.run(context, params)`.
- HTTP-Clients rufen `/v1/ai/rag/query/` mit der oben beschriebenen Payload auf. Die API-Referenz verweist direkt auf dieses Kapitel.

