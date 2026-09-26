"""A stand-in llama-server for the launcher tests (standard library only).

Accepts the real command line (``-m``, ``--alias``, ``--host``, ``--port``; the rest is
ignored). Knobs (environment):
  DUMMY_LLAMA_ARGV_LOG  append the argv (JSON) to this file at start
  DUMMY_LLAMA_EXIT      exit with this code right away (a failed model load)
  DUMMY_LLAMA_FAIL_UNTIL_NCMOE  exit 1 unless --n-cpu-moe >= this value (an OOM at low N)
  DUMMY_LLAMA_LOAD_S    answer /health with 503 for this long, then 200
  DUMMY_LLAMA_NO_TOOLS  report a template without tool calls
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
    parser.add_argument("--n-cpu-moe", dest="ncmoe", default=None)
    parser.add_argument("--cpu-moe", action="store_true")
    args, _ = parser.parse_known_args()
    log = os.environ.get("DUMMY_LLAMA_ARGV_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(sys.argv[1:]) + "\n")
    code = os.environ.get("DUMMY_LLAMA_EXIT")
    if code:
        return int(code)
    need = os.environ.get("DUMMY_LLAMA_FAIL_UNTIL_NCMOE")
    if need and not args.cpu_moe and (args.ncmoe is None or int(args.ncmoe) < int(need)):
        print("cudaMalloc failed: out of memory", file=sys.stderr)
        return 1
    load_s = float(os.environ.get("DUMMY_LLAMA_LOAD_S", "0"))
    tools = os.environ.get("DUMMY_LLAMA_NO_TOOLS") != "1"
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
            if time.monotonic() < ready_at:
                self._json(503, {"error": {"code": 503, "message": "Loading model"}})
            elif path == "/health":
                self._json(200, {"status": "ok"})
            elif path == "/props":
                self._json(200, {
                    "model_alias": args.alias,
                    "model_path": args.model,
                    "chat_template_caps": {"supports_tool_calls": tools},
                    "total_slots": 3,
                })
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

    ThreadingHTTPServer.daemon_threads = True
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever(poll_interval=0.05)
    return 0


if __name__ == "__main__":
    sys.exit(main())
