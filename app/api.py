"""Tiny JSON API backing the frontend's live-data refresh and free-text ticker lookup.

The frontend (frontend/) ports the regression, diagnostics, and rolling math
from src/factor_lab/ directly into JS and runs it client-side against bundled
monthly snapshots. This process exists only for what JS in a browser cannot do
itself: refreshing the Ken French factor/industry files live, and fetching an
arbitrary Yahoo Finance ticker (which has no bundled snapshot, same as today).

Started on demand and reaped when idle by the same launcher.py machinery the
other demos use (see demos.json's "api" kind). Deliberately stdlib only.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

from factor_lab.data import load_factors, load_industry_portfolios, load_yfinance_asset

DEFAULT_PORT = 8502


def frame_payload(frame: pd.DataFrame) -> dict:
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in frame.index],
        "series": {
            column: [None if pd.isna(v) else round(float(v), 6) for v in frame[column]]
            for column in frame.columns
        },
    }


def series_payload(series: pd.Series) -> dict:
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in series.index],
        "values": [None if pd.isna(v) else round(float(v), 6) for v in series],
    }


class Handler(BaseHTTPRequestHandler):
    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        query = parse_qs(parsed.query)
        refresh = query.get("refresh", ["false"])[0].lower() == "true"

        try:
            if path == "/factors":
                self._json(200, frame_payload(load_factors(force_refresh=refresh)))
            elif path == "/industries":
                self._json(200, frame_payload(load_industry_portfolios(force_refresh=refresh)))
            elif path == "/ticker":
                symbol = query.get("symbol", [""])[0].strip().upper()
                if not symbol:
                    self._json(400, {"error": "symbol is required"})
                    return
                self._json(200, series_payload(load_yfinance_asset(symbol)))
            else:
                self._json(404, {"error": f"unknown route {parsed.path!r}"})
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            self._json(502, {"error": "internal error"})

    def log_message(self, format: str, *args) -> None:
        pass


def main() -> None:
    host = "127.0.0.1"
    port = DEFAULT_PORT
    for arg in sys.argv[1:]:
        if arg.startswith("--port="):
            port = int(arg.split("=", 1)[1])
        elif arg.startswith("--host="):
            host = arg.split("=", 1)[1]
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
