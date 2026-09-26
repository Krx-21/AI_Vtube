"""llama-server command builder (ARCHITECTURE.md §4.11). Standard library only.

The launcher (stdlib only) and the standalone ``LlamaServerManager`` both build the command
line here, from a ``[llm.servers.<name>]`` table: either the validated ``LlamaServerConfig`` or
the raw mapping the launcher reads from the TOML layers.

Placement:
- ``fit`` (default): ``--fit on --fit-target <MiB>`` with an explicit ``-c``, so ``--fit`` only
  places layers and never shrinks the context;
- ``pinned`` (written by ``bench llm --tune``): ``--fit off -ngl all --n-cpu-moe N``;
- ``all_gpu`` (the 4B): ``--fit off -ngl all``.

``--fit`` and ``--n-cpu-moe`` must not be combined: ``--n-cpu-moe`` is implemented as tensor
overrides, which ``--fit`` refuses to place around.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

__all__ = ["LOOPBACK", "build_llama_argv", "root_url", "server_root_url", "slot_save_path"]

LOOPBACK = "127.0.0.1"

_DEFAULTS: Mapping[str, Any] = {
    "exe": "vendor/llama.cpp/llama-server.exe",
    "ctx": 16384,
    "parallel": 3,
    "placement": "fit",
    "fit_target_mib": 3584,
    "pinned_n_cpu_moe": 0,
    "threads": 8,
    "slot_save_dir": "data/kv",
    "extra_args": (),
}


def _field(server: Any, name: str) -> Any:
    """Read ``name`` from a pydantic model / object or from a raw TOML mapping."""
    if isinstance(server, Mapping):
        if name in server:
            return server[name]
    elif hasattr(server, name):
        return getattr(server, name)
    if name in _DEFAULTS:
        return _DEFAULTS[name]
    raise KeyError(f"llama server config has no {name!r}")


def _abs(value: str | Path, root: Path) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else root / path)


def _has_flag(args: Sequence[str], *flags: str) -> bool:
    return any(a.split("=", 1)[0] in flags for a in args)


def build_llama_argv(
    server: Any,
    *,
    root: Path,
    exe: str | Path | None = None,
    host: str = LOOPBACK,
    n_cpu_moe: int | Literal["all"] | None = None,
) -> list[str]:
    """The full llama-server command line (executable first) for one server table.

    ``root`` resolves relative ``exe``/``model``/``slot_save_dir`` paths (the AI_Vtube folder).
    ``exe`` overrides the configured executable. ``n_cpu_moe`` overrides ``pinned_n_cpu_moe``
    for ``placement = "pinned"`` (the launcher raises it after an out-of-memory load);
    ``"all"`` becomes ``--cpu-moe``.
    """
    placement = str(_field(server, "placement"))
    extra = [str(a) for a in _field(server, "extra_args")]
    argv = [
        _abs(exe if exe is not None else _field(server, "exe"), root),
        "-m",
        _abs(_field(server, "model"), root),
        "--alias",
        str(_field(server, "alias")),
        "--host",
        host,
        "--port",
        str(int(_field(server, "port"))),
        "-c",
        str(int(_field(server, "ctx"))),
        "-np",
        str(int(_field(server, "parallel"))),
        "-t",
        str(int(_field(server, "threads"))),
        "--jinja",
        "--slot-save-path",
        _abs(_field(server, "slot_save_dir"), root),
    ]
    if placement == "fit":
        argv += ["--fit", "on", "--fit-target", str(int(_field(server, "fit_target_mib")))]
    elif placement == "pinned":
        n = n_cpu_moe if n_cpu_moe is not None else int(_field(server, "pinned_n_cpu_moe"))
        argv += ["--fit", "off", "-ngl", "all"]
        argv += ["--cpu-moe"] if n == "all" else ["--n-cpu-moe", str(int(n))]
    elif placement == "all_gpu":
        argv += ["--fit", "off"]
        if not _has_flag(extra, "-ngl", "--gpu-layers", "--n-gpu-layers"):
            argv += ["-ngl", "all"]
    else:
        raise ValueError(f"unknown llama placement {placement!r} (fit | pinned | all_gpu)")
    return argv + extra


def slot_save_path(server: Any, *, root: Path) -> Path:
    """The absolute ``--slot-save-path`` directory. llama-server refuses to start ("not a
    directory") when it does not exist, so whoever spawns the server creates it first."""
    return Path(_abs(_field(server, "slot_save_dir"), root))


def root_url(base_url: str) -> str:
    """The server root for an OpenAI base URL (``…/v1`` stripped): llama-server's ``/health``,
    ``/props`` and ``/slots`` live there, not under ``/v1``."""
    url = base_url.rstrip("/")
    return url[: -len("/v1")] if url.endswith("/v1") else url


def server_root_url(server: Any, host: str = LOOPBACK) -> str:
    """``http://127.0.0.1:<port>``: the server root (``/health``, ``/props``, ``/slots``)."""
    return f"http://{host}:{int(_field(server, 'port'))}"
