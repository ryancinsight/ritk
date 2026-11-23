"""Serve the packaged study app over IPv4 loopback without host-name discovery."""
from functools import partial
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingTCPServer
import sys


def serve(bundle: Path):
    handler = partial(SimpleHTTPRequestHandler, directory=bundle.resolve(strict=True))
    # HTTPServer.server_bind reverse-resolves the host even for a numeric address.
    # Local Safari qualification needs neither DNS nor access to the runner's LAN.
    with ThreadingTCPServer(("127.0.0.1", 17823), handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    serve(Path(sys.argv[1]))
