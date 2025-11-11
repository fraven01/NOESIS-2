# Reranking Contracts

Die folgenden Verträge beschreiben das RAG-aware Re-Ranking in NOESIS 2 und
ergänzen die bestehenden Tool-Kontexte.

## ScoringContext Transport

`ScoringContext` wird vom Web-Service oder dem aufrufenden Agenten als
JSON-kodiertes Objekt über `ToolContext.meta["scoring_context"]` an den Worker
übergeben. Dadurch bleibt `ToolContext` selbst unverändert, während die
Reranking-spezifischen Signale typisiert vorliegen.

```json
{
  "meta": {
    "scoring_context": {
      "question": "Which policies govern employee data residency?",
      "purpose": "Compile legal summary",
      "jurisdiction": "DE",
      "output_target": "briefing",
      "preferred_sources": ["intranet://legal"],
      "disallowed_sources": ["wiki://drafts"],
      "collection_scope": "employee-compliance",
      "version_target": null,
      "freshness_mode": "standard",
      "min_diversity_buckets": 3
    }
  }
}
```

`freshness_mode` steuert die Altersgrenze der Kandidaten: `standard` wendet ein
12-Monats-Limit an, `software_docs_strict` erlaubt Inhalte bis zu 36 Monate und
`law_evergreen` deaktiviert den Cutoff vollständig für normative Quellen.
`min_diversity_buckets` definiert, wie viele unterschiedliche Hosts/Domain-Typen
mindestens in der finalen Auswahl vertreten sein müssen; fehlt der Wert, greift
der Worker-Standard (derzeit 3).

## SearchCandidate Datumsfelder

`SearchCandidate.detected_date` muss einen Zeitzonen-Offset enthalten. Naive
Zeitstempel führen zu einem Validierungsfehler. Aware Werte werden intern nach
UTC normalisiert, wodurch `.model_dump_json()` ISO-8601-Werte mit Offset
(`+00:00` bzw. `Z`) erzeugt.

## HybridResult Beispiel

Das Reranking liefert `HybridResult`. `ranked` enthält die vollständige Sortierung,
`top_k` die abgeschnittene Menge für HITL/Frontend, `coverage_delta` beschreibt
die Facet-Änderung in natürlicher Sprache und `recommended_ingest` sammelt
Quellenlücken.

```json
{
  "ranked": [
    {
      "candidate_id": "doc-417",
      "score": 92.5,
      "reason": "Addresses German residency controls",
      "gap_tags": ["LEGAL"],
      "risk_flags": [],
      "facet_coverage": {
        "LEGAL": 0.9,
        "PROCEDURAL": 0.6
      },
      "custom_facets": {
        "CUSTOM_REPORTING": 0.4
      }
    }
  ],
  "top_k": [
    {
      "candidate_id": "doc-417",
      "score": 92.5,
      "reason": "Addresses German residency controls",
      "gap_tags": ["LEGAL"],
      "risk_flags": [],
      "facet_coverage": {
        "LEGAL": 0.9,
        "PROCEDURAL": 0.6
      },
      "custom_facets": {
        "CUSTOM_REPORTING": 0.4
      }
    }
  ],
  "coverage_delta": "Adds LEGAL coverage for residency obligations",
  "recommended_ingest": [
    {
      "candidate_id": "doc-812",
      "reason": "Fills monitoring obligations for the DE market"
    }
  ]
}
```

Die Flags enthalten zusätzlich eine Debug-Sektion:

```json
"flags": {
  "rag_unavailable": false,
  "llm_timeout": false,
  "debug": {
    "pre_filter": {
      "dropped": [
        {"id": "cand-1", "reason": "stale"}
      ],
      "mmr": {"selected": [{"id": "cand-4", "max_overlap": 0.1}]}
    },
    "llm": {"fallback": null, "cache_hit": false, "llm_items": 5},
    "fusion": {
      "fused_scores": {"doc-417": 0.031},
      "rrf_components": {
        "doc-417": {
          "rrf_term": 0.01,
          "rag_bonus": 0.016,
          "domain_bonus": 0.009,
          "policy_bonus": 0.003
        }
      },
      "rrf_k": 90,
      "diversity_buckets": ["government", "commercial"]
    }
  }
}
```

Dies ermöglicht HITL und Telemetrie die Nachvollziehbarkeit der Fusion sowie der
vorherigen Filter- und LLM-Schritte. Zusätzlich protokolliert der Worker
Cache-Treffer (`hybrid.rag_cache_hit`, `hybrid.llm_cache_hit`) sowie das verwendete
RRF-K (`hybrid.rrf_k_used`) und die finale Diversitätsanzahl. Die Daten sind in
Langfuse zur Auswertung verfügbar.

`LLMScoredItem` besitzt neben `facet_coverage` ein Feld `custom_facets`. Keys
werden als `CUSTOM_*` serialisiert und erlauben tenantspezifische Metriken ohne
den Enum anzupassen.

## Typische Validierungsfehler

| Fehler | Ursache | Behebung |
| --- | --- | --- |
| `detected_date` ohne Zeitzone | Zeitstempel ist naiv (`2024-06-01T09:00:00`) | Offset oder `Z` ergänzen, z. B. `2024-06-01T09:00:00+02:00` |
| `min_diversity_buckets must be at least 1` | Konfigurationswert < 1 | Wert auf mindestens 1 setzen oder Feld weglassen |
| `invalid coverage dimension` | Facet-Key nicht im Enum (`"LEGAL"`, …) | Enum-Strings verwenden; Custom-Facets in `custom_facets` verschieben |
| `custom facet scores must be numeric` | Custom-Facet-Wert ist String oder negativ | Float-Wert im Bereich `0.0`–`1.0` liefern |

## HITL Auto-Approval & Coverage Monitoring

Bleibt eine HITL-Entscheidung länger als 24 Stunden aus, markiert der Business-Graph
den Schritt als `auto_approved` und übernimmt die aktuellen `top_k`. Das Ergebnis
enthält `hitl.auto_approved = true` sowie die automatisch erzeugte Entscheidung.

Der Coverage-Verifier ergänzt eine `summary` mit `ingested_count`, `failed_count`,
`pending_count` und `success_ratio`, sodass Teil-Erfolge transparent im Frontend
angezeigt werden können.

## LLM Control Parameter

Der Worker ruft das Score-Task mit `temperature = 0.3` und einem expliziten
`max_tokens`-Budget auf (Standard `2000`). Über `control.max_tokens` kann dieser
Wert pro Request angepasst werden.

## Migration Hinweis

`ScoreResultsData.results` akzeptiert nun explizit `SearchCandidate`. Bestehende
Clients, die bereits `ScoreResultInput` liefern, bleiben kompatibel, sollten aber
auf die erweiterten Felder (inkl. `detected_date`) migrieren, um Facet- und
Aktualitäts-Signale vollständig zu nutzen.
