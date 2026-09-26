"""A stand-in llama-server for spawn-mode tests (standard library only).

It accepts the real llama-server command line (``-m``, ``--alias``, ``--host``, ``--port`` and
ignores the rest), answers ``/health`` with 503 for ``FAKE_LLAMA_LOAD_S`` seconds and then 200,
serves ``/props`` (``FAKE_LLAMA_NO_TOOLS=1`` reports a template without tool calls) and
``POST /slots/{id}?action=save|restore`` (restore fails for files never saved). With
``FAKE_LLAMA_EXIT=<code>`` it exits right away, like a failed model load. It stops on SIGTERM
or CTRL_BREAK (the default handlers).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="")
    parser.add_argument("-a", "--alias", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args, _ = parser.parse_known_args()
    exit_code = os.environ.get("FAKE_LLAMA_EXIT")
    if exit_code:
        return int(exit_code)
    load_s = float(os.environ.get("FAKE_LLAMA_LOAD_S", "0"))
    tools = os.environ.get("FAKE_LLAMA_NO_TOOLS") != "1"
    ready_at = time.monotonic() + load_s
    saved: set[str] = set()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *a: Any) -> None:
            pass

        def _json(self, code: int, body: Any) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                if time.monotonic() < ready_at:
                    self._json(503, {"error": {"code": 503, "message": "Loading model"}})
                else:
                    self._json(200, {"status": "ok"})
            elif path == "/props":
                self._json(
                    200,
                    {
                        "model_alias": args.alias,
                        "model_path": args.model,
                        "chat_template_caps": {"supports_tool_calls": tools},
                        "total_slots": 3,
                    },
                )
            else:
                self._json(404, {"error": {"code": 404}})

        def do_POST(self) -> None:
            url = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            action = parse_qs(url.query).get("action", [""])[0]
            filename = str(body.get("filename", ""))
            if not url.path.startswith("/slots/"):
                self._json(404, {"error": {"code": 404}})
            elif action == "save":
                saved.add(filename)
                self._json(200, {"filename": filename, "n_saved": 1})
            elif action == "restore" and filename in saved:
                self._json(200, {"filename": filename, "n_restored": 1})
            else:
                self._json(400, {"error": {"code": 400, "message": "failed to restore"}})

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever(poll_interval=0.05)
    return 0


if __name__ == "__main__":
    sys.exit(main())
