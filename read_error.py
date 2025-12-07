try:
    with open("error_dump.html", "r", encoding="utf-8") as f:
        # Scan for exception_value class
        lines = f.readlines()
        found = False
        for i, line in enumerate(lines):
            if "Exception Type" in line or "Exception Value" in line:
                print(f"MATCH LINE {i}: {line.strip()}")
                # Print next few lines which usually contain the value
                for j in range(i + 1, min(len(lines), i + 5)):
                    print(lines[j].strip())

        if not found:
            print("Exception not found in dump. Top of file:")
            for line in lines[:20]:
                print(line.strip())

except Exception as e:
    print(f"Error reading dump: {e}")
