import os
import sys


def read_last_lines(filename, n=2000):
    try:
        with open(filename, "rb") as f:
            f.seek(0, os.SEEK_END)
            filesize = f.tell()
            blocksize = 4096
            dat_file = b""
            lines_found = 0
            pos = filesize
            while pos > 0 and lines_found < n:
                pos -= blocksize
                if pos < 0:
                    pos = 0
                f.seek(pos)
                block = f.read(blocksize)
                if pos > 0:
                    pass
                lines_found += block.count(b"\n")
                dat_file = block + dat_file

            lines = dat_file.splitlines()[-n:]
            for line in lines:
                try:
                    print(line.decode("utf-8"))
                except UnicodeDecodeError:
                    print(line.decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)


if __name__ == "__main__":
    read_last_lines(r"f:\NOESIS-2\NOESIS-2\logs\app\noesis-app.log", 2000)
