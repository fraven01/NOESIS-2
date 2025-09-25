# AGENTS Leitfaden – Components

## Regeln
- Jede Komponente muss die folgenden Dateien enthalten:
  - `Component.tsx` – Implementierung
  - `Component.test.tsx` – Tests mit Vitest und jest-axe
  - `Component.stories.tsx` – Storybook Story
- Verwende den `cn`-Helper und ausschließlich freigegebene UI-Basiskomponenten.
- Tests werden mit Vitest/jest-axe geschrieben; Stories leben in Storybook.

## Beispielhafte Ordnerstruktur

```text
components/
└─ ui/
   └─ Button/
      ├─ Button.tsx
      ├─ Button.test.tsx
      └─ Button.stories.tsx
```

```text
components/
└─ forms/
   └─ Input/
      ├─ Input.tsx
      ├─ Input.test.tsx
      └─ Input.stories.tsx
```
