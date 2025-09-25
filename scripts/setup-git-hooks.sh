#!/usr/bin/env bash
set -euo
{ set -o pipefail; } 2>/dev/null || true

git config core.hooksPath .githooks
chmod +x .githooks/pre-push || true
echo "Git hooks installed (core.hooksPath=.githooks)"
