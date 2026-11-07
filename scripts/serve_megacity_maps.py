#!/usr/bin/env python3
"""Serve the megacity Google Maps viewer with API key injection.

Usage:
  export GOOGLE_MAPS_API_KEY=your_key
  PYTHONPATH=src .venv/bin/python scripts/serve_megacity_maps.py
  # open http://127.0.0.1:8765/
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "megacity_maps"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, api_key: str = "", **kwargs):
        self._api_key = api_key
        super().__init__(*args, directory=str(WEB), **kwargs)

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            html = (WEB / "index.html").read_text(encoding="utf-8")
            html = html.replace("__GOOGLE_MAPS_API_KEY__", self._api_key)
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        return super().do_GET()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if not key:
        print(
            "WARNING: GOOGLE_MAPS_API_KEY is not set. The map tiles will not load.\n"
            "  export GOOGLE_MAPS_API_KEY=…",
            file=sys.stderr,
        )

    if not (WEB / "data" / "cities.json").exists():
        print(
            "WARNING: web/megacity_maps/data/cities.json missing. "
            "Run scripts/run_megacity_placement.py first.",
            file=sys.stderr,
        )

    handler = partial(Handler, api_key=key)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {WEB} at http://{args.host}:{args.port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
