"""``aivtube.llm.llamacpp`` and ``llama_args``: admin client, caps assertion, command builder."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx2
import pytest

from aivtube.config import load_config
from aivtube.config.schema import LlamaServerConfig
from aivtube.llm.llama_args import build_llama_argv, root_url, server_root_url, slot_save_path
from aivtube.llm.llamacpp import (
    LauncherServerManager,
    LlamaAdminError,
    LlamaCppAdmin,
    LlamaServerManager,
    LlamaServerSpec,
    TemplateCapsError,
    check_tool_caps,
    props_match,
    slot_filename,
)
from aivtube.testing.fakes import DEFAULT_PROPS, FakeClock

ROOT = Path(__file__).resolve().parents[3]


class Server:
    """A MockTransport llama-server: scripted status codes per path, requests recorded."""

    def __init__(self, props: dict[str, Any] | None = None) -> None:
        self.props = props if props is not None else dict(DEFAULT_PROPS)
        self.health = [200]
        self.slot_status = 200
        self.requests: list[httpx2.Request] = []
        self.down = False

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.down:
            raise httpx2.ConnectError("refused", request=request)
        path = request.url.path
        if path == "/health":
            code = self.health.pop(0) if len(self.health) > 1 else self.health[0]
            body = {"status": "ok"} if code == 200 else {"error": {"code": code}}
            return httpx2.Response(code, json=body)
        if path == "/props":
            return httpx2.Response(200, json=self.props)
        if path == "/slots":
            return httpx2.Response(200, json=[{"id": 0, "is_processing": False}])
        if path.startswith("/slots/"):
            if self.slot_status != 200:
                return httpx2.Response(self.slot_status, json={"error": {"message": "no file"}})
            return httpx2.Response(200, json={"id_slot": int(path.rsplit("/", 1)[1])})
        return httpx2.Response(404)

    def client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler))


def admin(server: Server, clock: Any = None) -> LlamaCppAdmin:
    return LlamaCppAdmin("http://127.0.0.1:8080/v1", server.client(), clock=clock)


# --- admin ------------------------------------------------------------------------------------


async def test_health_words() -> None:
    srv = Server()
    srv.health = [200, 503, 500]
    a = admin(srv)
    assert [await a.health() for _ in range(3)] == ["ok", "loading", "down"]
    srv.down = True
    assert await a.health() == "down"
    assert srv.requests[0].url.path == "/health"  # the root endpoint, not /v1/health


async def test_health_has_a_deadline(fake_clock: FakeClock) -> None:
    async def hang(request: httpx2.Request) -> httpx2.Response:
        await fake_clock.sleep(60)
        return httpx2.Response(200)

    a = LlamaCppAdmin(
        "http://127.0.0.1:8080",
        httpx2.AsyncClient(transport=httpx2.MockTransport(hang)),
        clock=fake_clock,
    )
    task = asyncio.create_task(a.health())
    await fake_clock.run_for(2.1)
    assert await task == "down"


async def test_assert_tool_caps(fixtures_dir: Path) -> None:
    sse = fixtures_dir / "sse"
    good = json.loads((sse / "llamacpp_props.json").read_text(encoding="utf-8"))
    bad = json.loads((sse / "llamacpp_props_no_template.json").read_text(encoding="utf-8"))
    await admin(Server(good)).assert_tool_caps()
    with pytest.raises(TemplateCapsError, match="chat-template-file"):
        await admin(Server(bad)).assert_tool_caps()
    with pytest.raises(TemplateCapsError):
        check_tool_caps({})
    with pytest.raises(TemplateCapsError):
        check_tool_caps({"chat_template_caps": {"supports_tool_calls": "yes"}})


async def test_props_errors_are_admin_errors() -> None:
    srv = Server()
    srv.down = True
    with pytest.raises(LlamaAdminError):
        await admin(srv).props()


async def test_save_and_restore_slot_post_the_filename() -> None:
    srv = Server()
    a = admin(srv)
    assert await a.save_slot(0, "pailin-abc123.bin") is True
    assert await a.restore_slot(1, "pailin-abc123.bin") is True
    assert await a.erase_slot(2) is True
    save, restore, erase = srv.requests
    assert (save.method, save.url.path, save.url.params["action"]) == ("POST", "/slots/0", "save")
    assert json.loads(save.content) == {"filename": "pailin-abc123.bin"}
    assert (restore.url.path, restore.url.params["action"]) == ("/slots/1", "restore")
    assert (erase.url.path, erase.url.params["action"]) == ("/slots/2", "erase")
    srv.slot_status = 400  # e.g. the file does not exist
    assert await a.restore_slot(0, "pailin-missing.bin") is False
    srv.down = True
    assert await a.save_slot(0, "pailin-abc123.bin") is False


@pytest.mark.parametrize("bad", ["../etc/passwd", "a/b.bin", "a\\b.bin", "", ".hidden", "x..bin"])
async def test_slot_files_must_be_bare_names(bad: str) -> None:
    srv = Server()
    assert await admin(srv).save_slot(0, bad) is False
    assert not srv.requests


def test_slot_filename() -> None:
    assert slot_filename("pailin", "3fa9c0") == "pailin-3fa9c0.bin"
    with pytest.raises(ValueError):
        slot_filename("../x", "y")


def test_props_match() -> None:
    props = {"model_alias": "pailin-30b", "model_path": r"C:\ai\models\typhoon.Q4_K_M.gguf"}
    assert props_match(props, alias="pailin-30b", model="models/llm/typhoon.Q4_K_M.gguf")
    assert props_match(props, alias="pailin-30b", model="models/llm/TYPHOON.Q4_K_M.gguf")
    assert not props_match(props, alias="pailin-4b")
    assert not props_match(props, alias="pailin-30b", model="models/other.gguf")
    assert props_match({"model_alias": "a,pailin-30b"}, alias="pailin-30b", model="x.gguf")
    assert not props_match({}, alias="pailin-30b")


# --- command builder ----------------------------------------------------------------------------


def _server(**kw: Any) -> LlamaServerConfig:
    base: dict[str, Any] = {
        "model": "models/llm/m.gguf",
        "alias": "pailin-30b",
        "port": 8080,
        "extra_args": [],
    }
    return LlamaServerConfig(**{**base, **kw})


def _opt(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_fit_placement(tmp_path: Path) -> None:
    argv = build_llama_argv(
        _server(ctx=24576, parallel=3, threads=8, fit_target_mib=3584), root=tmp_path
    )
    assert argv[0] == str(tmp_path / "vendor/llama.cpp/llama-server.exe")
    assert _opt(argv, "-m") == str(tmp_path / "models/llm/m.gguf")
    assert _opt(argv, "--alias") == "pailin-30b"
    assert _opt(argv, "--host") == "127.0.0.1" and _opt(argv, "--port") == "8080"
    assert _opt(argv, "-c") == "24576" and _opt(argv, "-np") == "3" and _opt(argv, "-t") == "8"
    assert "--jinja" in argv
    assert _opt(argv, "--slot-save-path") == str(tmp_path / "data/kv")
    assert _opt(argv, "--fit") == "on" and _opt(argv, "--fit-target") == "3584"
    assert "--n-cpu-moe" not in argv and "-ngl" not in argv


def test_pinned_placement_and_override(tmp_path: Path) -> None:
    server = _server(placement="pinned", pinned_n_cpu_moe=34)
    argv = build_llama_argv(server, root=tmp_path)
    assert _opt(argv, "--fit") == "off" and _opt(argv, "-ngl") == "all"
    assert _opt(argv, "--n-cpu-moe") == "34" and "--fit-target" not in argv
    assert _opt(build_llama_argv(server, root=tmp_path, n_cpu_moe=36), "--n-cpu-moe") == "36"
    all_cpu = build_llama_argv(server, root=tmp_path, n_cpu_moe="all")
    assert "--cpu-moe" in all_cpu and "--n-cpu-moe" not in all_cpu


def test_all_gpu_placement_does_not_duplicate_ngl(tmp_path: Path) -> None:
    argv = build_llama_argv(_server(placement="all_gpu", extra_args=["-ngl", "all"]), root=tmp_path)
    assert argv.count("-ngl") == 1 and _opt(argv, "--fit") == "off"
    argv = build_llama_argv(_server(placement="all_gpu"), root=tmp_path)
    assert _opt(argv, "-ngl") == "all"


def test_extra_args_go_last_and_absolute_paths_stay(tmp_path: Path) -> None:
    model = tmp_path / "abs.gguf"
    exe = tmp_path / "opt" / "llama-server"  # absolute on Windows too (it has a drive)
    argv = build_llama_argv(
        _server(model=str(model), extra_args=["-kvu", "--metrics"]),
        root=tmp_path / "somewhere" / "else",
        exe=exe,
    )
    assert argv[0] == str(exe) and _opt(argv, "-m") == str(model)
    assert argv[-2:] == ["-kvu", "--metrics"]


def test_raw_toml_mapping_is_accepted(tmp_path: Path) -> None:
    raw = {"model": "m.gguf", "alias": "pailin-4b", "port": 8081, "placement": "all_gpu"}
    argv = build_llama_argv(raw, root=tmp_path)
    assert _opt(argv, "--port") == "8081" and _opt(argv, "-c") == "16384"
    assert server_root_url(raw) == "http://127.0.0.1:8081"
    with pytest.raises(ValueError, match="placement"):
        build_llama_argv({**raw, "placement": "moon"}, root=tmp_path)
    with pytest.raises(KeyError):
        build_llama_argv({"alias": "x"}, root=tmp_path)


def test_slot_save_path(tmp_path: Path) -> None:
    assert slot_save_path(_server(), root=tmp_path) == tmp_path / "data" / "kv"
    absolute = tmp_path / "kv"
    assert slot_save_path({"slot_save_dir": str(absolute)}, root=Path("/x")) == absolute
    argv = build_llama_argv(_server(), root=tmp_path)
    assert _opt(argv, "--slot-save-path") == str(slot_save_path(_server(), root=tmp_path))


def test_root_url() -> None:
    assert root_url("http://127.0.0.1:8080/v1") == "http://127.0.0.1:8080"
    assert root_url("http://127.0.0.1:8080/v1/") == "http://127.0.0.1:8080"
    assert root_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"


def test_defaults_config_builds_both_servers() -> None:
    cfg = load_config(ROOT, env={})
    manager = LlamaServerManager.from_config(cfg, mode="adopt", log_dir=ROOT / "logs")
    specs = manager._specs
    assert set(specs) == {"local30b", "local4b"}
    big = specs["local30b"]
    assert big.base_url == "http://127.0.0.1:8080" and big.alias == "pailin-30b"
    assert big.model.endswith("typhoon2.5-qwen3-30b-a3b.Q4_K_M.gguf")
    assert _opt(list(big.argv), "--fit") == "on" and "-kvu" in big.argv
    small = specs["local4b"]
    assert _opt(list(small.argv), "-ngl") == "all" and list(small.argv).count("-ngl") == 1
    assert small.log_path == ROOT / "logs" / "llama-local4b.log"


def test_llama_args_is_stdlib_only() -> None:
    code = (
        "import sys; import aivtube.llm.llama_args; "
        "bad = [m for m in ('openai', 'httpx2', 'pydantic', 'numpy') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


# --- managers (MockTransport) -------------------------------------------------------------------


def _spec(**kw: Any) -> LlamaServerSpec:
    base: dict[str, Any] = {
        "name": "local30b",
        "base_url": "http://127.0.0.1:8080",
        "alias": "pailin-30b",
        "model": "models/llm/typhoon2.5-qwen3-30b-a3b.Q4_K_M.gguf",
    }
    return LlamaServerSpec(**{**base, **kw})


async def test_adopt_mode_accepts_a_matching_server() -> None:
    srv = Server()
    m = LlamaServerManager([_spec()], mode="adopt", http=srv.client())
    assert await m.ensure_running("local30b", 1.0) is True
    assert m.adopted("local30b") and m.process("local30b") is None
    assert await m.ensure_running("nope", 1.0) is False
    await m.stop("local30b")  # never stops an adopted server
    assert (await m.props("local30b"))["model_alias"] == "pailin-30b"
    assert await m.save_slot("local30b", 0, "pailin-x.bin") is True
    assert await m.save_slot("nope", 0, "pailin-x.bin") is False
    with pytest.raises(LlamaAdminError):
        await m.props("nope")


async def test_adopt_mode_rejects_a_mismatched_server() -> None:
    srv = Server({**DEFAULT_PROPS, "model_alias": "someone-else"})
    m = LlamaServerManager([_spec()], mode="adopt", http=srv.client())
    assert await m.ensure_running("local30b", 1.0) is False


async def test_adopt_mode_raises_on_a_template_without_tools() -> None:
    caps = {"supports_tool_calls": False}
    srv = Server({**DEFAULT_PROPS, "chat_template_caps": caps})
    m = LlamaServerManager([_spec()], mode="adopt", http=srv.client())
    with pytest.raises(TemplateCapsError):
        await m.ensure_running("local30b", 1.0)
    unchecked = LlamaServerManager([_spec()], mode="adopt", http=srv.client(), check_caps=False)
    assert await unchecked.ensure_running("local30b", 1.0) is True


async def test_adopt_mode_waits_for_loading(fake_clock: FakeClock) -> None:
    srv = Server()
    srv.health = [503, 503, 503, 200]
    m = LlamaServerManager([_spec()], mode="adopt", http=srv.client(), clock=fake_clock)
    task = asyncio.create_task(m.ensure_running("local30b", 5.0))
    await fake_clock.run_for(1.0)
    assert await task is True
    srv.health = [503]
    task = asyncio.create_task(m.ensure_running("local30b", 2.0))
    await fake_clock.run_for(2.5)
    assert await task is False


async def test_spawn_mode_without_a_command_fails() -> None:
    srv = Server()
    srv.down = True
    m = LlamaServerManager([_spec()], mode="spawn", http=srv.client())
    assert await m.ensure_running("local30b", 1.0) is False


async def test_launcher_manager_fast_path_and_unknown() -> None:
    srv = Server()
    launcher_calls: list[httpx2.Request] = []

    def launcher(request: httpx2.Request) -> httpx2.Response:
        launcher_calls.append(request)
        return httpx2.Response(200, json={"ok": True})

    m = LauncherServerManager(
        "http://127.0.0.1:8779",
        "tok",
        {"local30b": admin(srv)},
        http=httpx2.AsyncClient(transport=httpx2.MockTransport(launcher)),
    )
    assert await m.ensure_running("local30b", 1.0) is True
    assert not launcher_calls  # already healthy: the launcher is not bothered
    assert await m.ensure_running("nope", 1.0) is False
    await m.stop("local30b")
    (stop,) = launcher_calls
    assert stop.url.path == "/llm/stop/local30b"
    assert (
        stop.headers["authorization"] == "Bearer tok" and stop.headers["x-aivtube-token"] == "tok"
    )


def test_launcher_manager_from_config() -> None:
    cfg = load_config(ROOT, env={})
    m = LauncherServerManager.from_config(cfg, token="t")
    assert m.emergency_url == f"http://127.0.0.1:{cfg.ports.emergency}"
    admin30 = m.admin("local30b")
    assert admin30 is not None and admin30.base_url == "http://127.0.0.1:8080"


class _Proc:
    """A stand-in ``asyncio.subprocess.Process`` for the stop logic."""

    def __init__(self, *, signal_error: bool = False) -> None:
        self.returncode: int | None = None
        self.pid = 4242
        self.signal_error = signal_error
        self.signals = 0
        self.killed = False
        self.exited = asyncio.Event()

    def terminate(self) -> None:
        self.send_signal(15)

    def send_signal(self, sig: int) -> None:
        self.signals += 1
        if self.signal_error:
            raise OSError("no console to deliver CTRL_BREAK to")

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.exited.set()

    async def wait(self) -> int:
        await self.exited.wait()
        assert self.returncode is not None
        return self.returncode


async def test_stop_kills_when_the_stop_signal_cannot_be_sent() -> None:
    m = LlamaServerManager([_spec(argv=("llama-server",))], http=Server().client())
    proc = _Proc(signal_error=True)
    m._procs["local30b"] = proc  # type: ignore[assignment]
    await m.stop("local30b")
    assert proc.signals == 1 and proc.killed and m.process("local30b") is None


async def test_cancelled_stop_still_kills_the_process(fake_clock: FakeClock) -> None:
    m = LlamaServerManager(
        [_spec(argv=("llama-server",))], http=Server().client(), clock=fake_clock
    )
    proc = _Proc()
    m._procs["local30b"] = proc  # type: ignore[assignment]
    task = asyncio.create_task(m.stop("local30b"))
    await fake_clock.run_for(1.0)  # waiting for a graceful exit that never comes
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert proc.killed  # no orphan holding the port and VRAM
