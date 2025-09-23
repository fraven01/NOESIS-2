# Lokales Setup ohne Docker (Fallback)

> 📌 **Empfehlung:** Für den täglichen Entwicklungsbetrieb wird das Docker-Compose-Setup
> aus der [README](../../README.md#entwicklungsworkflow-mit-docker) empfohlen. Die
> folgenden Schritte dienen als Fallback für Systeme ohne Docker-Unterstützung.

## Voraussetzungen

- Python 3.12+
- Node.js und npm
- PostgreSQL-Server mit Zugriffsdaten
- Redis-Server

## Repository vorbereiten

```bash
git clone https://github.com/fraven01/NOESIS-2.git
cd NOESIS-2
```

## Python-Umgebung einrichten

```bash
python -m venv .venv

# Aktivieren
## Linux/macOS
source .venv/bin/activate
## Windows PowerShell
.\\.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Frontend-Abhängigkeiten installieren

```bash
npm install
```

## Anwendung konfigurieren

1. `.env.example` nach `.env` kopieren.
2. Datenbank- und Redis-DSNs auf lokale Dienste anpassen.

## Datenbank initialisieren

```bash
python manage.py migrate
python manage.py createsuperuser
```

## Entwicklung starten

```bash
npm run dev
```

Die `dev`-NPM-Task startet den Django-Entwicklungsserver und den Tailwind-Watcher
parallel. Stelle sicher, dass deine lokale PostgreSQL- und Redis-Instanz erreichbar
ist. Für Tests oder RAG-spezifische Tasks müssen die Compose-Skripte aus der Docker-
Dokumentation manuell nachgebildet werden (z. B. `docs/rag/schema.sql` gegen die DB
ausführen).
