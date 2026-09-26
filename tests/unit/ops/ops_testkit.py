"""Helpers for the ops tests: a local file server with HTTP Range support and fault knobs."""

from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class FileServer:
    """Serves ``files`` (url path → bytes) on 127.0.0.1.

    ``cut_after``: close the connection after this many body bytes (once per path, the first
    ``cuts`` requests). ``ignore_range``: always answer 200 with the whole file.
    """

    def __init__(self, files: dict[str, bytes], *, cut_after: int | None = None, cuts: int = 1,
                 ignore_range: bool = False) -> None:
        self.files = files
        self.cut_after = cut_after
        self.cuts_left = {p: cuts for p in files}
        self.ignore_range = ignore_range
        self.requests: list[tuple[str, str | None, int]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *a: Any) -> None:
                pass

            def do_GET(self) -> None:
                data = outer.files.get(self.path)
                rng = self.headers.get("Range")
                if data is None:
                    outer.requests.append((self.path, rng, 404))
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                start = 0
                if rng and not outer.ignore_range:
                    start = int(rng.removeprefix("bytes=").split("-", 1)[0])
                if start >= len(data) and start:
                    outer.requests.append((self.path, rng, 416))
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{len(data)}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = data[start:]
                code = 206 if start else 200
                outer.requests.append((self.path, rng, code))
                self.send_response(code)
                if start:
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                if outer.cut_after is not None and outer.cuts_left.get(self.path, 0) > 0:
                    outer.cuts_left[self.path] -= 1
                    self.wfile.write(body[: outer.cut_after])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_manifest(root: Path, entries: dict[str, dict[str, Any]]) -> Path:
    lines = ["schema_version = 1", ""]
    for name, fields in entries.items():
        lines.append(f"[models.{name}]")
        for key, value in fields.items():
            if isinstance(value, bool):
                lines.append(f"{key} = {'true' if value else 'false'}")
            elif isinstance(value, int):
                lines.append(f"{key} = {value}")
            elif isinstance(value, list):
                lines.append(f"{key} = [{', '.join(repr(v).replace(chr(39), chr(34)) for v in value)}]")
            else:
                lines.append(f'{key} = "{value}"')
        lines.append("")
    path = root / "models" / "manifest.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
