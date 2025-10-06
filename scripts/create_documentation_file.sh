#!/bin/bash

# Define the output file and the directory to scan
OUTPUT_FILE="doku.txt"
DOCS_DIR="docs"

# Remove the output file if it already exists to start fresh
rm -f $OUTPUT_FILE

# Find all files in the specified directory and its subdirectories
find "$DOCS_DIR" -type f | while read -r file
do
  # Write the file path as a header into the output file
  echo "--- $file ---" >> "$OUTPUT_FILE"
  # Append the content of the file
  cat "$file" >> "$OUTPUT_FILE"
  # Add a newline for separation
  echo "" >> "$OUTPUT_FILE"
done

echo "Documentation consolidated into $OUTPUT_FILE"
