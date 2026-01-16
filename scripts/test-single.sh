#!/bin/bash
# Script to run a single pytest test file or test case
# Usage: ./scripts/test-single.sh path/to/test.py
# Usage: ./scripts/test-single.sh path/to/test.py::test_function

set -e

# Match the Compose project metadata that other scripts rely on.
PROJECT_NAME=${COMPOSE_PROJECT_NAME:-noesis-2}
if command -v cygpath >/dev/null 2>&1; then
    PROJECT_DIR=$(cygpath -w "$PWD")
else
    PROJECT_DIR="$PWD"
fi

export COMPOSE_PROJECT_NAME="$PROJECT_NAME"
export COMPOSE_PROJECT_DIRECTORY="$PROJECT_DIR"

# Keep the dev services running so single tests reuse the same infra.
docker compose -f docker-compose.dev.yml up -d db redis toxiproxy litellm

if [ -z "$1" ]; then
    echo "Error: No test path provided"
    echo "Usage: npm run test:py:single -- path/to/test.py"
    echo "Usage: npm run test:py:single -- path/to/test.py::test_function"
    exit 1
fi

docker compose -f docker-compose.dev.yml run --rm web sh -c \
    "pip install -q -r requirements.txt -r requirements-dev.txt && DJANGO_SETTINGS_MODULE=noesis2.settings.test_parallel python -m pytest -q -v --reuse-db $1"
