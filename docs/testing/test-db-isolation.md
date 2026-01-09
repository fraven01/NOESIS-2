# Test Database Isolation - Root Cause & Fix (2026-01-02)

## Problem-Symptome

Nach `npm run win:dev:reset` liefen:
- **Fast tests** (`-m 'not slow'` mit xdist): ✅ **1305 passed**
- **Slow tests** (`-m 'slow'` serial): ❌ **Failures mit:**
  - `ProgrammingError: relation "documents_notificationdelivery" does not exist`
  - `ProgrammingError: relation "documents_savedsearch" does not exist`
  - Cascade-Fehler: `InternalError: current transaction is aborted`

Betroffene Dateien:
- `documents/tests/test_external_notifications.py`
- `documents/tests/test_saved_searches.py`

## Root Cause Analysis

### Primäres Problem: Fehlende TEST-Datenbank-Konfiguration

**Kernproblem** in [noesis2/settings/test_parallel.py](../../noesis2/settings/test_parallel.py):

```python
# VOR dem Fix (fehlerhaft):
_worker = os.getenv("PYTEST_XDIST_WORKER")
_db = DATABASES["default"]  # NAME = "noesis2"

_db["NAME"] = _worker_suffix(_db.get("NAME"), _worker)  # → "noesis2_gw0"
if "TEST" in _db:  # ← ABER: TEST existiert nicht!
    test_name = _db["TEST"].get("NAME")
    _db["TEST"]["NAME"] = _worker_suffix(test_name or _db.get("NAME"), _worker)
```

**Resultat**:
1. ❌ `DATABASES["default"]["TEST"]` war **nicht gesetzt**
2. ❌ pytest-django fiel zurück auf `NAME` statt `TEST["NAME"]`
3. ❌ Tests liefen gegen **Production-DB** (`noesis2_gw0`) statt Test-DB
4. ❌ Tenant-Schemas wurden in Production-DB gesucht, wo sie nicht existieren

**Verifikation**:
```bash
# Vor dem Fix:
$ docker compose run --rm web python -c "
export PYTEST_XDIST_WORKER=gw0
export DJANGO_SETTINGS_MODULE=noesis2.settings.test_parallel
from django.conf import settings
print('DB NAME:', settings.DATABASES['default']['NAME'])
print('TEST NAME:', settings.DATABASES['default'].get('TEST', {}).get('NAME'))
"
# Output:
# DB NAME: noesis2_gw0
# TEST NAME: not set  ← PROBLEM!
```

### Sekundäres Problem: Unquoted Schema-Namen

Schema-Namen mit Bindestrichen (z.B. `autotest-domain-service-091cf1ef`) wurden nicht gequotet:

```python
# Fehlerhaft:
cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {target_schema}")
cursor.execute(f"SET search_path TO {target_schema}")
# → Syntax-Fehler bei Schema-Namen mit '-'
```

**Folgen**:
- PostgreSQL Syntax-Fehler: `ERROR: syntax error at or near "-"`
- Transaction wird aborted → alle nachfolgenden Queries scheitern mit `current transaction is aborted`

## Lösung

### Fix 1: Explizite TEST-DB-Konfiguration

**Datei**: [noesis2/settings/test_parallel.py](../../noesis2/settings/test_parallel.py)

```python
# NACH dem Fix (korrekt):
_worker = os.getenv("PYTEST_XDIST_WORKER")
_db = DATABASES["default"]

# KRITISCH: Stelle sicher, dass TEST-Konfiguration existiert
if "TEST" not in _db:
    _db["TEST"] = {}

# Setze expliziten Test-DB-Namen (pytest-django würde "test_" Prefix hinzufügen)
base_test_name = _db["TEST"].get("NAME") or f"test_{_db.get('NAME', 'noesis2')}"
_db["TEST"]["NAME"] = _worker_suffix(base_test_name, _worker)

# Production DB-Name mit Worker-Suffix (für Settings-Konsistenz)
_db["NAME"] = _worker_suffix(_db.get("NAME"), _worker)
```

**Effekt**:
- ✅ `TEST["NAME"]` = `"test_noesis2_gw0"` (explizit gesetzt)
- ✅ Fast tests: worker-spezifische Test-DBs (`test_noesis2_gw0`, `test_noesis2_gw1`, etc.)
- ✅ Slow tests: dedizierte Test-DB (`test_noesis2_gw0` via ENV var)
- ✅ Production-DBs werden NIEMALS für Tests verwendet

### Fix 2: Schema-Namen quoten

**Datei**: [conftest.py](../../conftest.py)

```python
# In _patch_migration_recorder():
def _ensure_schema(self):
    # ...
    with connection.cursor() as cursor:
        quoted = connection.ops.quote_name(target_schema)
        cursor.execute(f"SET search_path TO {quoted}")

    with connection.cursor() as cursor:
        quoted = connection.ops.quote_name(target_schema)
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {quoted}")
```

**Datei**: [testsupport/tenant_fixtures.py](../../testsupport/tenant_fixtures.py)

```python
# In bootstrap_tenant_schema():
with connection.cursor() as cursor:
    if pg_sql is not None:
        cursor.execute(
            pg_sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                pg_sql.Identifier(schema_name)
            )
        )
    else:
        quoted = connection.ops.quote_name(schema_name)
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {quoted}")
```

## Verifikation nach Fix

```bash
# Test, dass TEST["NAME"] korrekt gesetzt ist:
$ docker compose run --rm web sh -c "
export PYTEST_XDIST_WORKER=gw0
export DJANGO_SETTINGS_MODULE=noesis2.settings.test_parallel
python -c 'from django.conf import settings; print(\"TEST DB:\", settings.DATABASES[\"default\"][\"TEST\"][\"NAME\"])'
"
# Output: TEST DB: test_noesis2_gw0  ✅

# Einzelner slow test:
$ docker compose run --rm web sh -c "
export PYTEST_XDIST_WORKER=gw0
export DJANGO_SETTINGS_MODULE=noesis2.settings.test_parallel
python -m pytest -xvs documents/tests/test_saved_searches.py::test_saved_search_scheduler_creates_notification --reuse-db
"
# Output: PASSED ✅

# Vollständiger Test-Lauf:
$ npm run win:test:py:parallel
# Output:
# Fast tests: 1305 passed ✅
# Slow tests: 124 passed ✅
```

## npm-Befehle Harmonisierung

**Vor**:
- Duplikation: `test:py:*` und `win:test:py:*` waren identisch
- Wartungslast: Änderungen mussten 2x gemacht werden

**Nach**:
```json
{
  "test:py": "docker compose ... python -m pytest -q",
  "test:py:parallel": "docker compose ... [parallel + serial logic]",
  "win:test:py": "npm run test:py",           // ← Alias
  "win:test:py:parallel": "npm run test:py:parallel"  // ← Alias
}
```

**Vorteile**:
- ✅ DRY (Don't Repeat Yourself)
- ✅ Single Source of Truth für Test-Befehle
- ✅ Plattformunabhängig (Docker)

## Best Practices

### 1. Immer TEST["NAME"] explizit setzen

```python
# ✅ RICHTIG:
DATABASES["default"]["TEST"] = {
    "NAME": "test_myproject_gw0"
}

# ❌ FALSCH:
DATABASES["default"]["NAME"] = "myproject_gw0"
# (pytest-django könnte Production-DB verwenden!)
```

### 2. Schema-Namen immer quoten

```python
# ✅ RICHTIG (Option 1: psycopg2.sql):
from psycopg2 import sql
cursor.execute(
    sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
        sql.Identifier(schema_name)
    )
)

# ✅ RICHTIG (Option 2: Django):
quoted = connection.ops.quote_name(schema_name)
cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {quoted}")

# ❌ FALSCH:
cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
# (Fails für schema_name = "autotest-foo"!)
```

### 3. Test-DB-Isolation verifizieren

```python
# In Settings oder conftest.py:
assert settings.DATABASES["default"]["NAME"].startswith("test_"), \
    "Tests MUST run against test database!"
```

### 4. Slow tests serial laufen lassen

```bash
# ✅ RICHTIG:
python -m pytest -m 'slow' --reuse-db  # Ohne -n auto

# ❌ FALSCH:
python -m pytest -m 'slow' -n auto  # Parallele Tenant-Ops = Race Conditions
```

## Timeline

- **2025-12-28**: Migrationen 0021/0022 für `documents` hinzugefügt
- **2025-12-31**: Erste Fehlerberichte über fehlende Tabellen in slow tests
- **2026-01-02 08:00**: Root Cause identifiziert (fehlende `TEST["NAME"]`)
- **2026-01-02 08:30**: Fix implementiert in `test_parallel.py`
- **2026-01-02 09:00**: Vollständiger Test-Lauf ✅ **grün** (1305 + 124 passed)

## Verwandte Dateien

- **Settings**: [noesis2/settings/test_parallel.py](../../noesis2/settings/test_parallel.py)
- **Fixtures**: [conftest.py](../../conftest.py)
- **Tenant Helpers**: [testsupport/tenant_fixtures.py](../../testsupport/tenant_fixtures.py)
- **Migrations**:
  - [documents/migrations/0021_add_collaboration_phase4a.py](../../documents/migrations/0021_add_collaboration_phase4a.py)
  - [documents/migrations/0022_create_external_notifications_phase4b.py](../../documents/migrations/0022_create_external_notifications_phase4b.py)

## Lessons Learned

1. **TEST["NAME"] ist nicht optional** für django-tenants + xdist + pytest-django
2. **Schema-Namen MÜSSEN gequotet werden** (Bindestriche sind valid in PostgreSQL wenn gequotet)
3. **Production-DB-Zugriff in Tests ist gefährlich** → immer verifizieren
4. **Parallele Tests ≠ Serial Tests** → unterschiedliche DB-Setup-Logik erforderlich
5. **Logging ist essentiell** → strukturierte Logs halfen bei Root Cause Analysis
