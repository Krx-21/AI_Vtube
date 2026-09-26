"""The emergency endpoint: auth, CORS, hard kill / rearm semantics and the llm routes (§2.11)."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from launcher_testkit import child_argv, lines, wait_until

from aivtube.launcher.emergency import EmergencyServer, notify_freeze
from aivtube.launcher.jobobject import JobObject
from aivtube.launcher.supervisor import ProcessSpec, Supervisor

TOKEN = "t0ken-for-tests-1234567890"
PANEL_PORT = 8770
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(
    url: str,
    method: str = "POST",
    *,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout: float = 10.0,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw or b"{}"), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        hdrs = dict(exc.headers)
        exc.close()
        return exc.code, json.loads(raw or b"{}"), hdrs


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


class FakeLlama:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.calls: list[tuple[str, float | None]] = []

    def ensure(self, timeout_s: float) -> bool:
        self.calls.append(("ensure", timeout_s))
        return self.ok

    def stop(self) -> None:
        self.calls.append(("stop", None))

    def restart(self) -> bool:
        self.calls.append(("restart", None))
        return True

    def status(self) -> dict[str, Any]:
        return {"phase": "ready" if self.ok else "failed"}


@pytest.fixture
def setup(tmp_path: Path) -> Iterator[tuple[EmergencyServer, Supervisor, Path, list[str], FakeLlama]]:
    starts = tmp_path / "voice-starts"
    sup = Supervisor(
        [
            ProcessSpec("core", child_argv(), {}, tmp_path, backoff=(0.05, 0.2)),
            ProcessSpec("voice", child_argv("--starts", str(starts)), {}, tmp_path,
                        backoff=(0.05, 0.2)),
        ],
        JobObject(),
        poll_s=0.02,
    )
    frozen: list[str] = []
    llama = FakeLlama()

    def freeze(reason: str) -> bool:
        frozen.append(reason)
        return True

    srv = EmergencyServer("127.0.0.1", 0, TOKEN, sup, {"local4b": llama}, panel_port=PANEL_PORT,
                          freeze=freeze, gpu=lambda: {"free_mib": 4096.0})
    sup.start()
    srv.start()
    assert wait_until(lambda: len(lines(starts)) == 1)
    try:
        yield srv, sup, starts, frozen, llama
    finally:
        srv.stop()
        sup.stop(1.0)


def test_binds_loopback_only_and_needs_a_token(tmp_path: Path) -> None:
    sup = Supervisor([], None)
    with pytest.raises(ValueError):
        EmergencyServer("0.0.0.0", 0, TOKEN, sup)
    with pytest.raises(ValueError):
        EmergencyServer("127.0.0.1", 0, "", sup)


@pytest.mark.parametrize(
    "headers, query",
    [({}, ""), ({"Authorization": "Bearer wrong"}, ""), ({}, "?token=wrong"),
     ({"X-Aivtube-Token": "nope"}, "")],
)
def test_requests_without_the_token_are_refused(
    setup: Any, headers: dict[str, str], query: str
) -> None:
    srv, sup, _, _, _ = setup
    code, body, _ = call(f"{srv.url}/hardkill{query}", headers=headers)
    assert code == 403 and body["ok"] is False
    code, _, _ = call(f"{srv.url}/status{query}", "GET", headers=headers)
    assert code == 403
    assert sup.state("voice") == "running"


@pytest.mark.parametrize(
    "headers, query",
    [(auth(), ""), ({"X-Aivtube-Token": TOKEN}, ""), ({}, f"?token={TOKEN}")],
)
def test_token_forms(setup: Any, headers: dict[str, str], query: str) -> None:
    srv, *_ = setup
    code, body, _ = call(f"{srv.url}/status{query}", "GET", headers=headers)
    assert code == 200
    assert body["voice"] == "up" and body["vram"] == {"free_mib": 4096.0}
    assert set(body["children"]) == {"core", "voice"}
    assert body["llm"]["local4b"]["phase"] == "ready"


def test_hardkill_holds_voice_until_rearm(setup: Any) -> None:
    srv, sup, starts, frozen, _ = setup
    pid = sup.status()["voice"]["pid"]
    code, body, _ = call(f"{srv.url}/hardkill", headers=auth(), body=b'{"why": "test"}')
    assert code == 200 and body == {"ok": True, "voice": "down", "elapsed_ms": body["elapsed_ms"]}
    st = sup.status()["voice"]
    assert st["state"] == "held" and st["pid"] is None
    assert wait_until(lambda: frozen == ["hardkill"], timeout=5)
    time.sleep(0.4)
    assert len(lines(starts)) == 1, "a held voice worker must not restart"
    code, status, _ = call(f"{srv.url}/status", "GET", headers=auth())
    assert status["voice"] == "down" and status["held"] is True and status["hardkills"] == 1
    assert sup.status()["core"]["state"] == "running"  # the core is not touched
    # restart is refused while held; rearm releases it
    code, _, _ = call(f"{srv.url}/restart/voice", headers=auth())
    assert code == 409
    code, body, _ = call(f"{srv.url}/rearm", headers=auth())
    assert code == 200 and body["voice"] == "up"
    assert wait_until(lambda: len(lines(starts)) == 2)
    assert sup.status()["voice"]["pid"] not in (None, pid)
    code, body, _ = call(f"{srv.url}/rearm", headers=auth())  # nothing to rearm
    assert code == 409 and body["ok"] is False


@pytest.mark.timing
def test_hardkill_latency_is_under_300ms(setup: Any) -> None:
    """Acceptance (§2.11): POST /hardkill → the voice worker is gone within 300 ms."""
    srv, sup, *_ = setup
    t0 = time.perf_counter()
    code, body, _ = call(f"{srv.url}/hardkill", headers=auth())
    took = time.perf_counter() - t0
    assert code == 200 and sup.state("voice") == "held"
    assert took < 0.3, f"hard kill took {took * 1000:.0f} ms"
    assert body["elapsed_ms"] < 300


def test_restart_routes(setup: Any) -> None:
    srv, sup, _, _, llama = setup
    pid = sup.status()["core"]["pid"]
    code, body, _ = call(f"{srv.url}/restart/core", headers=auth())
    assert code == 200 and body["ok"] is True
    assert sup.status()["core"]["pid"] != pid
    code, _, _ = call(f"{srv.url}/restart/local4b", headers=auth())
    assert code == 200 and ("restart", None) in llama.calls
    code, _, _ = call(f"{srv.url}/restart/nope", headers=auth())
    assert code == 404


def test_llm_routes(setup: Any) -> None:
    srv, _, _, _, llama = setup
    code, body, _ = call(f"{srv.url}/llm/ensure/local4b?timeout_s=7.5", headers=auth())
    assert code == 200 and body["ok"] is True and body["server"] == "local4b"
    assert llama.calls[-1] == ("ensure", 7.5)
    llama.ok = False
    code, body, _ = call(f"{srv.url}/llm/ensure/local4b?timeout_s=bad", headers=auth())
    assert code == 503 and body["ok"] is False
    assert llama.calls[-1] == ("ensure", 60.0)
    code, _, _ = call(f"{srv.url}/llm/stop/local4b", headers=auth())
    assert code == 200 and llama.calls[-1] == ("stop", None)
    for path in ("/llm/ensure/local99", "/llm/stop/local99", "/nope"):
        code, _, _ = call(srv.url + path, headers=auth())
        assert code == 404
    code, _, _ = call(f"{srv.url}/hardkill", "GET", headers=auth())
    assert code == 404


def test_cors_panel_origin_only(setup: Any) -> None:
    srv, sup, *_ = setup
    good = f"http://127.0.0.1:{PANEL_PORT}"
    code, _, hdrs = call(f"{srv.url}/status", "GET", headers={**auth(), "Origin": good})
    assert code == 200 and hdrs.get("Access-Control-Allow-Origin") == good
    code, _, hdrs = call(f"{srv.url}/status", "GET",
                         headers={**auth(), "Origin": f"http://localhost:{PANEL_PORT}"})
    assert code == 200
    code, _, hdrs = call(f"{srv.url}/hardkill", headers={**auth(), "Origin": "https://evil.example"})
    assert code == 403 and "Access-Control-Allow-Origin" not in hdrs
    assert sup.state("voice") == "running"
    # preflight
    req_headers = {
        "Origin": good,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization",
        "Access-Control-Request-Private-Network": "true",
    }
    req = urllib.request.Request(f"{srv.url}/hardkill", method="OPTIONS", headers=req_headers)
    with _OPENER.open(req, timeout=5) as resp:
        assert resp.status == 204
        assert resp.headers["Access-Control-Allow-Origin"] == good
        assert "Authorization" in resp.headers["Access-Control-Allow-Headers"]
        assert resp.headers["Access-Control-Allow-Private-Network"] == "true"
    code, _, _ = call(f"{srv.url}/hardkill", "OPTIONS", headers={"Origin": "http://evil"})
    assert code == 403


def test_panel_spa_no_cors_hardkill(setup: Any) -> None:
    """The panel's fallback is a "simple" no-cors POST with ``?token=`` from the panel origin
    (``src/aivtube/panel/static/index.html``): no preflight, no custom headers."""
    srv, sup, _, frozen, _ = setup
    headers = {"Origin": f"http://127.0.0.1:{PANEL_PORT}", "Content-Type": "text/plain"}
    code, body, _ = call(f"{srv.url}/hardkill?token={TOKEN}", headers=headers, body=b"")
    assert code == 200 and body["voice"] == "down"
    assert sup.state("voice") == "held"
    assert wait_until(lambda: frozen == ["hardkill"], timeout=5)


def test_oversized_body_is_refused(setup: Any) -> None:
    srv, sup, *_ = setup
    code, body, _ = call(f"{srv.url}/hardkill", headers=auth(), body=b"x" * (65 * 1024))
    assert code == 413 and body["ok"] is False
    assert sup.state("voice") == "running"


def test_non_loopback_host_header_is_refused(setup: Any) -> None:
    srv, sup, *_ = setup
    code, _, _ = call(f"{srv.url}/hardkill", headers={**auth(), "Host": "attacker.example:8779"})
    assert code == 403
    assert sup.state("voice") == "running"


class _PanelStub:
    def __init__(self) -> None:
        self.seen: list[tuple[str, dict[str, Any], dict[str, str]]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *a: Any) -> None:
                pass

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                outer.seen.append((self.path, json.loads(self.rfile.read(n)), dict(self.headers)))
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def test_notify_freeze_posts_to_the_panel() -> None:
    panel = _PanelStub()
    try:
        assert notify_freeze(panel.url, "ptok", reason="hardkill") is True
        path, body, headers = panel.seen[0]
        assert path == "/api/cmd" and body["kind"] == "freeze"
        assert headers["Authorization"] == "Bearer ptok"
    finally:
        panel.close()
    assert notify_freeze("http://127.0.0.1:9", "x", reason="r", timeout_s=0.5) is False
