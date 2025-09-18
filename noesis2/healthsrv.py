from __future__ import annotations

import os
import socket
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - required by BaseHTTPRequestHandler
        # Minimal health endpoint responding 200 OK for any path
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args):  # noqa: A003 - match superclass
        # Silence default logging to keep container logs clean
        return


def main() -> None:
    port_str = os.getenv("PORT", "8080")
    try:
        port = int(port_str)
    except ValueError:
        print(f"Invalid PORT value: {port_str}", file=sys.stderr)
        port = 8080
    server = HTTPServer(("0.0.0.0", port), _Handler)
    # Ensure socket reuse to avoid restart bind errors
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
