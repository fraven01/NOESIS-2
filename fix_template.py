file_path = r"theme\templates\theme\partials\tool_documents.html"

try:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # The problematic pattern
    bad_pattern = """<option value="{{ collection.selector }}" {% if
                            collection.selector==selected_collection_identifier %}selected{% endif %}>"""

    # The fixed pattern
    fixed_pattern = """<option value="{{ collection.selector }}" {% if collection.selector == selected_collection_identifier %}selected{% endif %}>"""

    if bad_pattern in content:
        print("Found bad pattern. Fixing...")
        new_content = content.replace(bad_pattern, fixed_pattern)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print("File updated successfully.")
    else:
        # Fallback: Try looser match just in case whitespace is slightly different
        print("Exact match not found. Trying line-by-line processing...")
        lines = content.splitlines()
        new_lines = []
        skip_next = False
        fixed = False

        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue

            if '<option value="{{ collection.selector }}" {% if' in line:
                # Check if next line completes it
                if (
                    i + 1 < len(lines)
                    and "collection.selector==selected_collection_identifier %}selected{% endif %}>"
                    in lines[i + 1].strip()
                ):
                    combined = line.strip() + " " + lines[i + 1].strip()
                    # Clean up the join
                    combined = combined.replace("{% if collection", "{% if collection")
                    new_lines.append(fixed_pattern)  # Just use our known good string
                    skip_next = True
                    fixed = True
                    print(f"Fixed split lines {i} and {i+1}")
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        if fixed:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines))
            print("File updated via line processing.")
        else:
            print("Could not identify the broken code block.")

except Exception as e:
    print(f"Error: {e}")
