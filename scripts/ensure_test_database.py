#!/usr/bin/env python3
"""Ensure the dedicated test database exists for local pytest runs."""
from __future__ import annotations

import os
import sys
import time
from urllib.parse import urlparse, urlunparse

import psycopg2
from psycopg2 import OperationalError, errors, sql

DEFAULT_TEST_URL = "postgresql://noesis2:noesis2@db:5432/noesis2_test"
DEFAULT_ADMIN_DB = "postgres"
MAX_RETRIES = int(os.environ.get("ENSURE_TEST_DB_RETRIES", "10"))
SLEEP_SECONDS = float(os.environ.get("ENSURE_TEST_DB_SLEEP", "1"))


def _admin_dsn(test_url: str, fallback_db: str = DEFAULT_ADMIN_DB) -> tuple[str, str]:
    parsed = urlparse(test_url)
    database = parsed.path.lstrip("/") or fallback_db
    admin_path = f"/{fallback_db}"
    admin_url = urlunparse(parsed._replace(path=admin_path))
    return admin_url, database


def ensure_database(admin_url: str, database: str) -> bool:
    """Create ``database`` via ``admin_url`` if it does not yet exist."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with psycopg2.connect(admin_url) as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM pg_database WHERE datname = %s", (database,)
                    )
                    if cur.fetchone():
                        print(
                            f"ensure_test_database: '{database}' already present",
                            flush=True,
                        )
                        return False
                    cur.execute(
                        sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database))
                    )

        except errors.DuplicateDatabase:
            print(f"ensure_test_database: '{database}' already present", flush=True)
            return False
        except OperationalError as exc:  # pragma: no cover - connection retries only
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"ensure_test_database: could not reach Postgres after {MAX_RETRIES} attempts"
                ) from exc
            time.sleep(SLEEP_SECONDS)
        else:
            print(f"ensure_test_database: created '{database}'", flush=True)
            return True
    return False


def main() -> int:
    test_url = os.environ.get("AI_CORE_TEST_DATABASE_URL", DEFAULT_TEST_URL)
    admin_url, database = _admin_dsn(test_url)
    try:
        created = ensure_database(admin_url, database)
    except Exception as exc:  # pragma: no cover - surfaced in shell
        print(str(exc), file=sys.stderr)
        return 1
    status = "created" if created else "present"
    print(f"ensure_test_database: status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
