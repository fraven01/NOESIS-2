#!/bin/bash
# Script to run a single pytest test file or test case
# Usage: ./scripts/test-single.sh path/to/test.py
# Usage: ./scripts/test-single.sh path/to/test.py::test_function

set -e

if [ -z "$1" ]; then
    echo "Error: No test path provided"
    echo "Usage: npm run test:py:single -- path/to/test.py"
    echo "Usage: npm run test:py:single -- path/to/test.py::test_function"
    exit 1
fi

docker compose -f docker-compose.dev.yml run --rm web sh -c \
    "pip install -q -r requirements.txt -r requirements-dev.txt && python -m pytest -q -v $1"
