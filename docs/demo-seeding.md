# Demo-Seeding Leitfaden

## Zweck
- Schnell reproduzierbare Demo-Daten für das `demo`-Tenant-Schema.
- Vergleichbare Seeds für Schulung, Demos und Smoke-Checks.

## Profile
| Profil | Projekte | Dokumente je Projekt | Besonderheiten |
| --- | --- | --- | --- |
| baseline | 2 | 1 | kleinste Variante für Smoke-Checks |
| demo | 5–8 | 3–5 | Mischung aus Text/Markdown, kleinere Unsauberkeiten |
| heavy | 30 | 10 | Bulk-Creation für Performance-Tests |
| chaos | wie demo | wie demo | 10–15 % absichtlich markierte Fehler |

## Beispiele
```bash
python manage.py create_demo_data --profile baseline --seed 1337
python manage.py create_demo_data --profile demo --projects 6 --docs-per-project 4 --seed 1337
python manage.py create_demo_data --profile heavy --seed 42
python manage.py create_demo_data --profile chaos --seed 99
python manage.py create_demo_data --wipe --include-org
python manage.py check_demo_data --profile demo --seed 1337
```

## Idempotenz
- Wiederholte Aufrufe aktualisieren bestehende Objekte anhand stabiler Slugs.
- `--seed` und Faker (`de_DE`) sorgen für deterministische Inhalte.

## Chaos-Fälle
- Dokumente mit `meta.invalid=true`, leeren Bodies oder bewusst kaputtem JSON.
- Smoke-Checks (`check_demo_data`) melden Abweichungen als strukturierte Events.

## Was nicht tut
- Keine echten PII- oder Kundendaten.
- Kein Überschreiben von OpenTelemetry-Providern.
- Keine Operationen auf fremden Tenants oder Migrationen.

## Troubleshooting
- Fehlendes `demo`-Schema → `python manage.py migrate_schemas` und Tenant-Rechte prüfen.
- Berechtigungsfehler → sicherstellen, dass die lokale DB-Nutzerrolle Schreibrechte im Tenant-Schema besitzt.
- Domain-Konflikt → vorhandene Einträge in `django_tenants_domain` prüfen oder löschen.
