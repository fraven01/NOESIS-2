#!/bin/bash

set -euo pipefail

# Define the output file and the directory to scan
OUTPUT_FILE="doku.txt"
DOCS_DIR="docs"

# Common directory prunes (aligned with .gitignore and typical noise)
PRUNE_DIRS=(
  './.git'
  './node_modules'
  './.venv'
  './venv'
  './env.bak'
  './venv.bak'
  './.pytest_cache'
  './.ruff_cache'
  './.mypy_cache'
  '*/.pytest_cache'
  '*/.ruff_cache'
  '*/.mypy_cache'
  './build'
  './dist'
  './downloads'
  './eggs'
  './.eggs'
  './lib'
  './lib64'
  './htmlcov'
  './.tox'
  './.nox'
  './.cache'
  './.hypothesis'
  './cover'
  './instance'
  './.scrapy'
  './docs/_build'
  './.pybuilder'
  './target'
  './.ipynb_checkpoints'
  './profile_default'
  './.pdm-build'
  './.pixi'
  './__pypackages__'
  './.pyre'
  './.pytype'
  './cython_debug'
  './site'
  './logs'
  './staticfiles'
  './__pycache__'
  '*/__pycache__'
  '*/build'
  '*/dist'
  '*/downloads'
  '*/eggs'
  '*/.eggs'
  '*/lib'
  '*/lib64'
)

# Build a reusable prune expression for find
build_prune_expr() {
  PRUNE_EXPR=()
  for d in "${PRUNE_DIRS[@]}"; do
    PRUNE_EXPR+=( -path "$d" -prune -o )
  done
}

# Remove the output file if it already exists to start fresh
rm -f "$OUTPUT_FILE"

# 0) Project directory tree (filtered)
echo "--- PROJECT DIRECTORY TREE (filtered) ---" >> "$OUTPUT_FILE"
{
  build_prune_expr
  find . "${PRUNE_EXPR[@]}" -type d -print | sort | while read -r dir; do
    # Normalize and compute depth
    rel="${dir#./}"
    if [[ "$dir" == "." ]]; then
      echo "./"
      continue
    fi
    depth=$(awk -F'/' '{print NF-1}' <<< "$dir")
    name=$(basename "$dir")
    indent=""
    # depth 1 => top-level: no indent; depth 2+ => indent
    for ((i=1; i<depth; i++)); do indent+="  "; done
    echo "${indent}${name}/"
  done
} >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Helper to append a file with a header and trailing newline
append_with_header() {
  local file="$1"
  echo "--- $file ---" >> "$OUTPUT_FILE"
  cat "$file" >> "$OUTPUT_FILE"
  echo "" >> "$OUTPUT_FILE"
}

# 1) Append all files from docs/ (recursive)
if [[ -d "$DOCS_DIR" ]]; then
  # Sorted for determinism
  while IFS= read -r file; do
    append_with_header "$file"
  done < <(find "$DOCS_DIR" -type f | sort)
fi

# 2) Append all README.md files from across the repo (including root)
#    - Case-insensitive match for readme.md
#    - Exclude docs/ (already included above) and pruned directories
while IFS= read -r readme; do
  append_with_header "$readme"
done < <(
  build_prune_expr; find . "${PRUNE_EXPR[@]}" -path ./docs -prune -o -type f -iname 'readme.md' -print | sort
)

# 3) Append all AGENTS.md files across the repo
while IFS= read -r agents; do
  append_with_header "$agents"
done < <(
  build_prune_expr; find . "${PRUNE_EXPR[@]}" -type f -iname 'agents.md' -print | sort
)

echo "Documentation consolidated into $OUTPUT_FILE"
