"""The launcher's view of the configuration, read from the raw TOML layers (stdlib only).

The launcher cannot import pydantic, so it reads ``aivtube.config.layers.collect_layers`` (the
same layering as ``load_config``) and picks the few values it needs, with the defaults of
``config/defaults.toml``. Full validation runs in a subprocess during preflight
(``aivtube.ops.configcheck``).

Tokens live in ``data/state/tokens.json`` (panel and emergency: stable across runs, so OBS
docks and Stream Deck URLs keep working); the IPC bus token is new for every run.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aivtube.config.errors import ConfigError
from aivtube.config.layers import collect_layers
from aivtube.launcher.childside import BUS_TOKEN_ENV, EMERGENCY_TOKEN_ENV, PANEL_TOKEN_ENV

__all__ = ["LauncherSettings", "Tokens", "load_settings", "load_tokens"]

log = logging.getLogger("aivtube.launcher.settings")

LOOPBACK = "127.0.0.1"


def _table(data: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    node: Any = data
    for key in keys:
        node = node.get(key, {}) if isinstance(node, Mapping) else {}
    return node if isinstance(node, Mapping) else {}


def _num(data: Mapping[str, Any], key: str, default: float, *, path: str) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(
            f"{path}.{key}",
            f"{path}.{key} must be a number, got {value!r}.",
            f"{path}.{key} ต้องเป็นตัวเลข แต่ได้ {value!r}",
            "Compare with config/defaults.toml.",
            hint_th="เทียบกับค่าใน config/defaults.toml",
        )
    return float(value)


def _int(data: Mapping[str, Any], key: str, default: int, *, path: str) -> int:
    return int(_num(data, key, default, path=path))


def _bool(data: Mapping[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    return value if isinstance(value, bool) else default


def _pair(data: Mapping[str, Any], key: str, default: tuple[float, float], *, path: str
          ) -> tuple[float, float]:
    value = data.get(key, list(default))
    if (
        isinstance(value, list | tuple)
        and len(value) == 2
        and all(isinstance(v, int | float) and not isinstance(v, bool) for v in value)
        and 0 < float(value[0]) <= float(value[1])
    ):
        return float(value[0]), float(value[1])
    raise ConfigError(
        f"{path}.{key}",
        f"{path}.{key} must be [min, max] seconds, got {value!r}.",
        f"{path}.{key} ต้องเป็น [ต่ำสุด, สูงสุด] วินาที แต่ได้ {value!r}",
        "For example: [0.5, 30.0].",
        hint_th="ตัวอย่าง: [0.5, 30.0]",
    )


@dataclass(frozen=True)
class LauncherSettings:
    """Everything the launcher reads from the config (see the module docstring)."""

    root: Path
    profile: str
    raw: Mapping[str, Any]
    ports: Mapping[str, int]
    data_dir: Path
    log_dir: Path
    models_dir: Path
    voice_worker: bool
    fakes: bool
    games_enabled: bool
    servers: Mapping[str, Mapping[str, Any]]
    chain: tuple[str, ...]
    providers: Mapping[str, Mapping[str, Any]]
    keep_llm: bool = True
    core_timeout_s: float = 5.0
    graceful_timeout_s: float = 5.0
    restart_backoff_s: tuple[float, float] = (0.5, 30.0)
    crash_loop_restarts: int = 5
    crash_loop_window_s: float = 120.0
    llm_health_interval_s: float = 2.0
    llm_health_failures: int = 3
    llm_load_timeout_s: float = 300.0
    llm_max_load_failures: int = 3
    vram_alarm_mib: int = 500
    gpu_poll_s: float = 5.0
    keep_awake: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def state_dir(self) -> Path:
        return self.data_dir / "state"

    @property
    def emergency_url(self) -> str:
        return f"http://{LOOPBACK}:{self.ports['emergency']}"

    @property
    def panel_url(self) -> str:
        return f"http://{LOOPBACK}:{self.ports['panel']}"

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def llm_servers_in_chain(self) -> list[str]:
        """Names of the ``[llm.servers]`` that the effective chain's llama.cpp providers use."""
        out: list[str] = []
        for name in self.chain:
            prov = self.providers.get(name, {})
            server = prov.get("server") if isinstance(prov, Mapping) else None
            if (
                isinstance(server, str)
                and server in self.servers
                and prov.get("enabled", True) is not False
                and server not in out
            ):
                out.append(server)
        return out

    def autostart_servers(self) -> list[str]:
        """Servers to start (or adopt) at launch: ``autostart = "always"``, not faked."""
        if self.fakes:
            return []
        return [n for n, s in self.servers.items() if s.get("autostart", "on_demand") == "always"]


def load_settings(
    root: Path,
    *,
    profile: str | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> LauncherSettings:
    """Read the raw layers of ``root``'s config. Raises ``ConfigError``."""
    root = Path(root).resolve()
    layers = collect_layers(root, profile=profile, cli_overrides=cli_overrides, env=env)
    raw = layers.merged
    ports_raw = _table(raw, "ports")
    ports = {
        "panel": _int(ports_raw, "panel", 8770, path="ports"),
        "bus": _int(ports_raw, "bus", 8771, path="ports"),
        "emergency": _int(ports_raw, "emergency", 8779, path="ports"),
        "neuro_sdk": _int(ports_raw, "neuro_sdk", 8000, path="ports"),
    }
    app = _table(raw, "app")
    lcfg = _table(raw, "launcher")
    llm = _table(raw, "llm")
    servers = {
        str(k): dict(v) for k, v in _table(llm, "servers").items() if isinstance(v, Mapping)
    }
    for name, server in servers.items():
        _int(server, "port", 0, path=f"llm.servers.{name}")
    providers = {
        str(k): dict(v) for k, v in _table(llm, "providers").items() if isinstance(v, Mapping)
    }
    chain_raw = llm.get("chain", [])
    chain = tuple(str(c) for c in chain_raw) if isinstance(chain_raw, list) else ()
    consent = _bool(_table(raw, "privacy"), "cloud_llm_consent", False)
    if not consent:
        chain = tuple(c for c in chain if not providers.get(c, {}).get("cloud", False))
    logging_cfg = _table(raw, "logging")
    data_dir = str(app.get("data_dir", "data"))
    models_dir = str(app.get("models_dir", "models"))
    log_dir = str(logging_cfg.get("dir", "logs"))

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    return LauncherSettings(
        root=root,
        profile=layers.profile,
        raw=raw,
        ports=ports,
        data_dir=resolve(data_dir),
        log_dir=resolve(log_dir),
        models_dir=resolve(models_dir),
        voice_worker=_bool(app, "voice_worker", True),
        fakes=_bool(app, "fakes", False),
        games_enabled=_bool(_table(raw, "games"), "enabled", False),
        servers=servers,
        chain=chain,
        providers=providers,
        keep_llm=_bool(lcfg, "keep_llm", True),
        core_timeout_s=_num(lcfg, "core_timeout_s", 5.0, path="launcher"),
        graceful_timeout_s=_num(lcfg, "graceful_timeout_s", 5.0, path="launcher"),
        restart_backoff_s=_pair(lcfg, "restart_backoff_s", (0.5, 30.0), path="launcher"),
        crash_loop_restarts=_int(lcfg, "crash_loop_restarts", 5, path="launcher"),
        crash_loop_window_s=_num(lcfg, "crash_loop_window_s", 120.0, path="launcher"),
        llm_health_interval_s=_num(lcfg, "llm_health_interval_s", 2.0, path="launcher"),
        llm_health_failures=_int(lcfg, "llm_health_failures", 3, path="launcher"),
        llm_load_timeout_s=_num(lcfg, "llm_load_timeout_s", 300.0, path="launcher"),
        llm_max_load_failures=_int(lcfg, "llm_max_load_failures", 3, path="launcher"),
        vram_alarm_mib=_int(lcfg, "vram_alarm_mib", 500, path="launcher"),
        gpu_poll_s=_num(lcfg, "gpu_poll_s", 5.0, path="launcher"),
        keep_awake=_bool(lcfg, "keep_awake", True),
    )


@dataclass(frozen=True)
class Tokens:
    panel: str
    emergency: str
    bus: str

    def values(self) -> list[str]:
        return [self.panel, self.emergency, self.bus]


def _new_token() -> str:
    return secrets.token_urlsafe(24)


def load_tokens(state_dir: Path, env: Mapping[str, str] | None = None) -> Tokens:
    """Panel/emergency tokens from ``state_dir/tokens.json`` (created on first use), unless
    ``AIVTUBE_PANEL_TOKEN``/``AIVTUBE_EMERGENCY_TOKEN`` are set; a fresh bus token per run
    (``AIVTUBE_BUS_TOKEN`` overrides it)."""
    source = os.environ if env is None else env
    path = state_dir / "tokens.json"
    stored: dict[str, Any] = {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            stored = loaded
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        log.warning("%s is unreadable; new tokens will be written", path)
    changed = False
    for key in ("panel", "emergency"):
        value = stored.get(key)
        if not isinstance(value, str) or len(value) < 16:
            stored[key] = _new_token()
            changed = True
    if changed:
        try:
            state_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(stored, indent=2), encoding="utf-8")
            with contextlib.suppress(OSError):
                os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except OSError:
            log.warning("cannot save %s; tokens change at every start", path, exc_info=True)
    return Tokens(
        panel=source.get(PANEL_TOKEN_ENV) or str(stored["panel"]),
        emergency=source.get(EMERGENCY_TOKEN_ENV) or str(stored["emergency"]),
        bus=source.get(BUS_TOKEN_ENV) or _new_token(),
    )
