# Frontend-Überblick

## Verzeichnisstruktur
- `theme/static_src/`: Enthält `input.css` sowie Design-Tokens unter `styles/`. Dieses Verzeichnis wird als Eingang für die CSS-Pipeline genutzt.
- `theme/components/`: React/TypeScript-Komponenten. Halte dich an die Vorgaben in `theme/AGENTS.md` und dem Frontend Master Prompt.
- `.storybook/`: Konfiguration für Storybook, um Komponenten isoliert zu entwickeln und zu dokumentieren.

## Build-Prozess
- PostCSS mit Tailwind CSS v4 transformiert `theme/static_src/input.css` zu `theme/static/css/output.css`.
- Design-Tokens in `theme/static_src/styles/tokens.css` werden in den Build eingebunden und können projektweit genutzt werden.
- Der Build wird über `npm run build:css` oder automatisch innerhalb von `npm run dev` ausgeführt.

## Tests und Storybook
- Frontend-Tests laufen mit Vitest: `npm test`.
- Storybook-Entwicklungsumgebung starten: `npm run storybook`.

## Weitere Leitfäden
- Gesamtprojektregeln: [AGENTS Leitfaden](../AGENTS.md)
- Frontend-spezifische Regeln: [AGENTS Leitfaden – Frontend](../theme/AGENTS.md)
- Vollständige Komponenten-Guidelines: [Frontend Master Prompt](frontend-master-prompt.md)
