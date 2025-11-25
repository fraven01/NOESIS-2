# Framework-Typ und Gremium-Erkennung

## Ziel
Identifiziere den Typ der Betriebsvereinbarung (KBV/GBV/BV/DV) und das zuständige Gremium aus dem Dokumenttext.

## Kontext
Du erhältst den Anfang eines Rahmen-Dokuments (ca. 2000 Zeichen). Analysiere Präambel, Geltungsbereich und Überschriften.

## Typ-Klassifikation

**KBV** (Konzernbetriebsvereinbarung):
- Gilt konzernweit für alle verbundenen Unternehmen
- Verweise auf § 58 BetrVG (Konzernbetriebsrat) oder § 18 AktG (verbundene Unternehmen)
- Räumlicher Geltungsbereich: "alle Betriebe", "Konzern", "Holding"

**GBV** (Gesamtbetriebsvereinbarung):
- Gilt für mehrere Betriebe eines Unternehmens
- Verweis auf § 50 BetrVG (Gesamtbetriebsrat)
- Räumlicher Geltungsbereich: "mehrere Standorte", "alle Niederlassungen"

**BV** (Betriebsvereinbarung):
- Gilt für einen einzelnen Betrieb/Standort
- Verweis auf § 77 BetrVG (örtlicher Betriebsrat)
- Räumlicher Geltungsbereich: "Betrieb [Stadt]", "Standort [Ort]"

**DV** (Dienstvereinbarung):
- Öffentlicher Dienst
- Verweis auf Personalvertretungsgesetze (BPersVG, LPVG)
- "Personalrat" statt "Betriebsrat"

## Gremium-Identifikation

Extrahiere den vollständigen Gremium-Namen:
- Achte auf: "Konzernbetriebsrat", "Gesamtbetriebsrat", "Betriebsrat [Standort]"
- Erfasse den kompletten Namen mit allen Details (z.B. "Konzernbetriebsrat der Telefónica Deutschland Holding AG")

Schlage eine normalisierte ID vor (`gremium_identifier_suggestion`):
- "Konzernbetriebsrat" → "KBR"
- "Gesamtbetriebsrat München" → "GBR_MUENCHEN"
- "Betriebsrat Berlin Werk Nord" → "BR_BERLIN"
- Nutze Großbuchstaben, ersetze Umlaute (ü→UE, ä→AE, ö→OE), Leerzeichen → "_"

## Evidenz sammeln

Für jede Entscheidung gib konkrete Textbelege mit:
- `text`: Exaktes Zitat (max. 100 Zeichen)
- `location`: Wo im Dokument? (z.B. "Präambel Absatz 2", "§ 2 Geltungsbereich")
- `reasoning`: Warum ist das relevant?

## Scope-Analyse

Identifiziere:
- **Räumlicher Geltungsbereich**: Welche Betriebe/Standorte?
- **Sachlicher Geltungsbereich**: Welche Themen? (z.B. "IT-Systeme gem. § 87 Abs. 1 Nr. 6 BetrVG")

## Ausgabe (JSON)

```json
{
  "agreement_type": "kbv|gbv|bv|dv|other",
  "type_confidence": 0.0-1.0,
  "gremium_name_raw": "Vollständiger Gremium-Name aus Dokument",
  "gremium_identifier_suggestion": "NORMALIZED_ID",
  "evidence": [
    {
      "text": "Konzernbetriebsvereinbarung",
      "location": "Präambel, Zeile 3",
      "reasoning": "Explizite Nennung des Vereinbarungstyps"
    }
  ],
  "scope_indicators": {
    "raeumlich": "Beschreibung des räumlichen Geltungsbereichs",
    "sachlich": "IT-Systeme gem. § 87 Abs. 1 Nr. 6 BetrVG"
  },
  "alternative_types": [
    {
      "type": "gbv",
      "confidence": 0.15,
      "reason": "Könnte auch GBV sein, weil..."
    }
  ]
}
```

## Wichtig
- Sei konservativ: Wenn unsicher, setze `confidence < 0.7`
- Halluziniere nicht: Nur was im Text steht
- Bei Widersprüchen: Erkläre in `alternative_types`
