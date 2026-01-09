from __future__ import annotations

import logging
import os
import socket
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


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO)


def main() -> None:
    port_str = os.getenv("PORT", "8080")
    try:
        port = int(port_str)
    except ValueError:
        LOGGER.error("healthsrv.invalid_port", extra={"port": port_str})
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
