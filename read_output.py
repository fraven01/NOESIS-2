try:
    with open("test_output.txt", "r", encoding="utf-16") as f:
        content = f.read()
except Exception:
    with open("test_output.txt", "r", encoding="utf-8") as f:
        content = f.read()

lines = content.splitlines()
found = False
for i, line in enumerate(lines):
    if "FAILED" in line or "Error" in line or "DEBUG" in line:
        print(f"Match at line {i}: {line}")
        # Print context
        start = max(0, i - 10)
        end = min(len(lines), i + 50)
        for j in range(start, end):
            print(lines[j])
        print("-" * 20)
        found = True

if not found:
    print("No failure lines found. Printing last 100 lines:")
    for line in lines[-100:]:
        print(line)
