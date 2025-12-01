try:
    with open("test_output_debug.log", "r", encoding="utf-8") as f:
        content = f.read()
except UnicodeDecodeError:
    with open("test_output_debug.log", "r", encoding="utf-16") as f:
        content = f.read()

lines = content.splitlines()
printing = False
for line in lines:
    if "DEBUG: unsafe_identifier" in line:
        print(line)
    if "FAILED" in line or "Error" in line or "E   " in line:
        printing = True
    if printing:
        print(line)
        if "___" in line and "FAILED" not in line:  # End of failure block usually
            printing = False
