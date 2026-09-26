"""A stand-in child process for the launcher tests (standard library only).

Options:
  --exit CODE      exit with CODE after --after seconds (default: run until stopped)
  --after S        delay before --exit (default 0)
  --starts FILE    append "<pid>\n" to FILE at start (counts restarts)
  --ignore-term    ignore SIGTERM / CTRL_BREAK (tests the kill after the graceful timeout)
  --term-code CODE exit with CODE on SIGTERM / CTRL_BREAK (a graceful shutdown)
  --http PORT      serve GET (any path, e.g. /healthz) on 127.0.0.1:PORT (0 = any; see
                   --port-file)
  --wedge-after S  after S seconds stop answering HTTP (accept, never reply)
  --dump-file FILE answer AIVTUBE_DUMP_REQUEST by writing the stacks to FILE
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exit", type=int, default=None)
    p.add_argument("--after", type=float, default=0.0)
    p.add_argument("--starts", default="")
    p.add_argument("--ignore-term", action="store_true")
    p.add_argument("--term-code", type=int, default=None)
    p.add_argument("--http", type=int, default=None)
    p.add_argument("--port-file", default="")
    p.add_argument("--wedge-after", type=float, default=None)
    p.add_argument("--dump-file", default="")
    args = p.parse_args()

    if args.starts:
        with open(args.starts, "a", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()}\n")
    if args.ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    if args.term_code is not None:
        code = args.term_code

        def graceful(signum: int, frame: Any) -> None:
            os._exit(code)

        signal.signal(signal.SIGTERM, graceful)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, graceful)
    if args.dump_file:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
        from aivtube.launcher.childside import start_dump_request_watcher

        out = open(args.dump_file, "a", encoding="utf-8")  # noqa: SIM115 - lives for the process
        start_dump_request_watcher(file=out, interval_s=0.05)

    wedged = threading.Event()
    if args.http is not None:
        started = time.perf_counter()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *a: Any) -> None:
                pass

            def do_GET(self) -> None:
                if args.wedge_after is not None and time.perf_counter() - started > args.wedge_after:
                    wedged.set()
                    time.sleep(3600)
                body = b'{"ok": true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        ThreadingHTTPServer.daemon_threads = True
        server = ThreadingHTTPServer(("127.0.0.1", args.http), Handler)
        if args.port_file:
            Path(args.port_file).write_text(str(server.server_address[1]), encoding="utf-8")
        threading.Thread(target=server.serve_forever, daemon=True).start()

    if args.exit is not None:
        time.sleep(args.after)
        return int(args.exit)
    while True:
        time.sleep(0.05)


if __name__ == "__main__":
    sys.exit(main())
