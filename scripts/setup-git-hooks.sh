#!/usr/bin/env bash
set -euo pipefail

git config core.hooksPath .githooks
chmod +x .githooks/pre-push || true
echo "Git hooks installed (core.hooksPath=.githooks)"

