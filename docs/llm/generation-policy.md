# Generation Policy (Feature Request)

Status: proposal

Owner: AI Core

## Problem

Heute fehlt eine einheitliche Policy, die steuert, wie groß/teuer/„lang“ LLM‑Antworten sein dürfen. Wir haben aktuell nur globale ENV‑Defaults (z. B. `LITELLM_MAX_TOKENS`) oder feste Werte im Code. Dadurch entstehen Zielkonflikte:

- Interaktive Pfade sollen schnell reagieren (kleine Antworten, kurze Timeouts).
- Longform/Reports (z. B. Gutachten) benötigen deutlich größere Outputs und längere Timeouts.
- Betrieb braucht harte Leitplanken (Kosten/Timeouts), ohne die Business‑Logik überall anzupassen.

## Ziele

- Große Outputs ermöglichen, aber explizit und kontrolliert.
- Interaktive Standardpfade performant halten.
- Steuerung zentral definieren (Policy), nicht „ad‑hoc“ in jedem Aufruf.
- Rückwärtskompatibel und schrittweise einführbar.

## Nicht‑Ziele

- Kein unmittelbarer Umbau der Orchestrierung/Business‑Logik.
- Kein Provider‑Lock‑in in die Policy (Provider‑unabhängige Parameter).

## Vorschlag (Kurzfassung)

Führe eine „Generation Policy“ ein, die pro Request/Schritt berechnet, welche Generationsparameter an den LLM‑Client gehen. Präzedenz der Quellen:

1) Request/Graph: `meta.generation.{max_tokens, temperature, timeout}` (Orchestrator/Knoten)
2) Label‑Defaults: pro Routing‑Label (z. B. `synthesize`, `draft`, `reasoning`)
3) ENV‑Fallback: `LITELLM_MAX_TOKENS`, `LITELLM_TEMPERATURE`, `LITELLM_TIMEOUTS`
4) Hard‑Caps: Systemweite Obergrenzen (z. B. min/max Tokens, Timeout‑Ceilings)

Die Policy wird im LLM‑Client zusammengeführt (Merge), bevor der Request an LiteLLM geht.

## Parameter

- `max_tokens` (int): obere Grenze für die Antwortlänge
- `temperature` (float): Kreativität/Streuung
- `timeout` (int, Sekunden): Client‑Timeout je Label
- (optional später) `top_p`, `presence_penalty`, `frequency_penalty`, `stop`

## Betriebsmodi (Richtwerte)

- interactive (Default):
  - `max_tokens`: 256–512
  - `timeout`: 20–30 s
  - `temperature`: 0.2–0.5

- extended:
  - `max_tokens`: 1024–2048
  - `timeout`: 60–90 s

- report/longform:
  - `max_tokens`: 4096–8192
  - `timeout`: 120–300 s
  - bevorzugt asynchron (siehe unten)

Hinweis: Konkrete Werte variieren je Provider/Preis und werden in Label‑Defaults/ENV justiert.

## Datenvertrag (Request → Policy)

Orchestrator oder ein Knoten kann optional pro Schritt steuern:

```json
{
  "meta": {
    "generation": {
      "max_tokens": 512,
      "temperature": 0.2,
      "timeout": 30
    }
  }
}
```

Die Felder sind optional; fehlende Werte fallen auf Label‑Defaults → ENV → Caps zurück.

## Label‑Defaults (optional)

Ein YAML‑Artefakt kann später eingeführt werden (z. B. `config/generation_policy.yaml`):

```yaml
labels:
  synthesize:
    max_tokens: 512
    temperature: 0.2
    timeout: 30
  report:
    max_tokens: 4096
    temperature: 0.3
    timeout: 180
```

Die Policy löst pro Label (Routing) diese Defaults auf und merged sie mit `meta.generation` und ENV.

## ENV‑Fallback & Caps

- `LITELLM_MAX_TOKENS`
- `LITELLM_TEMPERATURE`
- `LITELLM_TIMEOUTS` (JSON pro Label, existiert bereits)

Systemweite Caps (z. B. 16 ≤ max_tokens ≤ 8192) verhindern Fehlkonfigurationen.

## Asynchron für Longform

Für sehr große Outputs (z. B. Gutachten) empfiehlt sich ein asynchroner Pfad:

1) HTTP startet Longform‑Task (202/`run_id`).
2) Celery generiert iterativ (Kapitelweise oder streamed) und persistiert Zwischenergebnisse.
3) Status/Ergebnis via `/status`/`/result` APIs abholen.

Vorteile: Keine Web‑Timeouts, bessere Nutzererfahrung, klare Kostenkontrolle.

## Observability & Kosten

- Tokens/Costs pro Schritt loggen (bestehende Ledger‑Events nutzen) und in Langfuse/ELK sichtbar machen.
- Warnungen/Alerts bei Überschreitung (z. B. >80% Budget).

## Rollout (inkrementell)

1) Client‑Merge (klein, kurzfristig):
   - Lese `meta.generation.{max_tokens, temperature, timeout}` im LLM‑Client zusätzlich zu ENV.
2) (Optional) Label‑Defaults einführen (YAML + Resolver).
3) Orchestrator/Knoten setzen für ausgewählte Schritte `meta.generation`.
4) Longform‑Pfad asynchronisieren (separates Ticket).

## Offene Punkte

- Provider‑spezifische Limits (Kontextfenster) dynamisch berücksichtigen?
- Kosten‑Budget als Policy‑Parameter (`budget_tokens`/`budget_cost`)?
- Streaming vs. Non‑Streaming pro Label definieren?

## Akzeptanzkriterien

- Interaktive Flows bleiben < 30 s bei Standardprompts.
- Longform kann kontrolliert > 2k Tokens generieren (asynchron bevorzugt).
- Policy lässt sich pro Label/Schritt überschreiben, ohne Codeänderungen am Client.

