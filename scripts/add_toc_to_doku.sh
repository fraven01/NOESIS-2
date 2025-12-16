#!/bin/bash

# Adds a Table of Contents to existing doku.txt
INPUT_FILE="doku.txt"
OUTPUT_FILE="doku-with-toc.txt"

# Extract all section headers
echo "# NOESIS 2 - Complete Documentation" > "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "**Generated**: $(date '+%Y-%m-%d %H:%M')" >> "$OUTPUT_FILE"
echo "**Size**: $(wc -l < "$INPUT_FILE") lines, $(wc -c < "$INPUT_FILE" | awk '{printf "%.1f MB", $1/1024/1024}')" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "## Table of Contents" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Generate TOC
grep "^--- " "$INPUT_FILE" | nl -w3 -s'. ' >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Append original content
cat "$INPUT_FILE" >> "$OUTPUT_FILE"

echo "Enhanced documentation with TOC: $OUTPUT_FILE"
