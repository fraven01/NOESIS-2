# Plausibilitätsprüfung für lokalisierte Bausteine

## Ziel
Prüfe, ob ein identifizierter Baustein tatsächlich an der gefundenen Stelle im Dokument steht und inhaltlich plausibel ist.

## Kontext
Du erhältst:
1. **Baustein-Name**: z.B. "systembeschreibung", "auswertungen"
2. **Identifizierte Chunks**: Die Chunks, die als Baustein kartiert wurden
3. **Kontext**: 2 Chunks davor + 2 Chunks danach
4. **Parent-Überschrift**: Die zugehörige Überschrift aus dem ToC

## Aufgabe

**1. Inhaltliche Validierung**
Prüfe, ob die Chunks tatsächlich den erwarteten Baustein beschreiben:

**Systembeschreibung** sollte enthalten:
- Technische Komponenten, Architektur
- System-Module, Schnittstellen
- NICHT: Rechtliche Rahmenbedingungen, Verfahren

**Funktionsbeschreibung** sollte enthalten:
- Use Cases, Features, Workflows
- Was das System tut (Prozesse)
- NICHT: Wie es technisch umgesetzt ist

**Auswertungen** sollte enthalten:
- Konkrete Berichte, Dashboards
- Datenfelder, Auswertungskriterien
- NICHT: Allgemeine Überwachungsbegriffe

**Zugriffsrechte** sollte enthalten:
- Rollen, Berechtigungen, Rechte-Matrix
- Wer darf was einsehen/ändern
- NICHT: Technische Zugriffsmechanismen

**2. Kontext-Analyse**
Prüfe die Nachbar-Chunks:
- Macht die Position Sinn? (z.B. Systembeschreibung nach Präambel, vor Funktionen)
- Gibt es Brüche? (z.B. plötzlich anderes Thema)

**3. Warnsignale**
- Sehr kurz (< 200 Wörter)
- Nur Verweis auf externe Doku ("siehe Handbuch X")
- Überschrift passt nicht zum Inhalt

## Ausgabe (JSON)

```json
{
  "component": "systembeschreibung",
  "plausible": true,
  "confidence": 0.91,
  "reason": "Chunks enthalten technische Spezifikationen, Systemarchitektur und Schnittstellen. Typische Elemente einer Systembeschreibung vorhanden.",
  "why_not": null,
  "context_analysis": {
    "before": "Präambel und Geltungsbereich (typische Reihenfolge)",
    "after": "Funktionsbeschreibung (logische Abfolge)"
  },
  "warnings": []
}
```

**Wenn nicht plausibel:**
```json
{
  "component": "zugriffsrechte",
  "plausible": false,
  "confidence": 0.35,
  "reason": "Chunks beschreiben primär rechtliche Rahmenbedingungen nach BetrVG, nicht konkrete Rollen/Berechtigungen.",
  "why_not": "Erwartete Inhalte (Rollen-Matrix, Berechtigungskonzept, Rechte-Vergabe) fehlen. Nur allgemeine Mitbestimmungsrechte erwähnt.",
  "context_analysis": {
    "before": "Geltungsbereich",
    "after": "Verfahrensregeln"
  },
  "warnings": [
    "Sehr kurzer Abschnitt (nur 150 Wörter)",
    "Keine konkrete Rollen-Auflistung vorhanden"
  ],
  "alternative_interpretation": "Möglicherweise nur Verweis - echte Zugriffsrechte in separatem Dokument geregelt"
}
```

## Confidence-Skala

- **0.9-1.0**: Sehr plausibel, alle erwarteten Elemente vorhanden
- **0.7-0.8**: Plausibel, aber unvollständig oder vage
- **0.5-0.6**: Unsicher, nur teilweise passend
- **<0.5**: Nicht plausibel, falsche Zuordnung

## Wichtig
- Sei kritisch, aber fair
- Fokus auf **Inhalt**, nicht Formulierungen
- Wenn Zweifel: `confidence < 0.7` + klare Begründung in `why_not`
- `warnings` sind optional, aber hilfreich für HITL-Review
