"""``models/manifest.toml``: pinned models and binaries; resumable download and sha256 verify (§9).

Standard library only (``urllib``, ``hashlib``, ``zipfile``), so the stdlib-only launcher can use
it for its preflight.

- **Pull.** Downloads go to ``<dest>.part`` and resume with an HTTP ``Range`` request (206);
  a server that ignores the range (200) restarts the file. Network errors are retried with
  backoff, resuming each time. The file must then match ``size`` and ``sha256`` before it is
  renamed into place. ``unpack = "zip"`` archives are downloaded to ``models/.downloads/`` and
  extracted into ``dest`` (a folder).
- **Verify.** A full sha256 of every present file. Results are cached in
  ``models/.verified.json`` keyed by path, size and mtime, so the launcher preflight can
  check big files cheaply (:meth:`ModelManifest.quick_check`).
- An entry with an empty ``sha256`` (or ``size = 0``) is *unpinned*: it is reported as such, not
  as corrupt, and the computed digest is printed so it can be pinned.
- One writer per file: a pull holds an OS lock on ``models/.downloads/<name>.lock`` (released
  by the OS if the process dies), so a second ``setup`` or ``models pull`` never appends to the
  same ``.part`` file; it fails with "already being downloaded" instead.

``python -m aivtube.ops.models {pull,verify} [names…]`` runs it directly (``aivtube models`` in
the CLI calls the same functions).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import http.client
import json
import logging
import os
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import IO, Any, Literal

__all__ = [
    "MANIFEST",
    "STAMP_FILE",
    "CheckResult",
    "ModelBusy",
    "ModelEntry",
    "ModelError",
    "ModelManifest",
    "cuda_variant",
    "main",
    "sha256_file",
]

log = logging.getLogger("aivtube.ops.models")

MANIFEST = Path("models") / "manifest.toml"
STAMP_FILE = ".verified.json"
DOWNLOADS = ".downloads"
CHUNK = 1 << 20
CHEAP_SHA_BYTES = 64 * 1024 * 1024
"""Files up to this size are hashed even in a quick check."""

Status = Literal["ok", "missing", "size", "sha", "unpinned", "partial"]
Progress = Callable[[str, int, int], None]


class ModelError(RuntimeError):
    """A download or verification failed."""


class ModelBusy(ModelError):
    """Another process is downloading this entry right now."""


class _FileLock:
    """A non-blocking exclusive OS lock on ``path`` (``flock`` / ``msvcrt.locking``).

    The file itself is never deleted: unlinking a lock file races with a process that is
    about to lock it.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: IO[bytes] | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = self.path.open("a+b")
        try:
            if sys.platform == "win32":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined,unused-ignore]
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined,unused-ignore]
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        with contextlib.suppress(OSError):
            if sys.platform == "win32":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined,unused-ignore]
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined,unused-ignore]
        fh.close()


@dataclass(frozen=True)
class ModelEntry:
    """One ``[models.<name>]`` table."""

    name: str
    url: str
    sha256: str
    size: int
    licence: str
    required_by: tuple[str, ...]
    dest: str
    variant: str = ""
    unpack: str = ""
    optional: bool = False
    notes: str = ""

    @property
    def pinned(self) -> bool:
        return bool(self.sha256) and self.size > 0

    @property
    def archive(self) -> bool:
        return self.unpack == "zip"

    def file_name(self) -> str:
        return PurePosixPath(self.url.split("?", 1)[0]).name or self.name


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    path: Path
    detail: str = ""
    sha256: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "unpinned")


def cuda_variant(driver_major: int | None) -> str:
    """The llama.cpp build for an NVIDIA driver: CUDA 13.4 needs R580+, else 12.4 (§9)."""
    return "cuda-13.4" if driver_major is not None and driver_major >= 580 else "cuda-12.4"


def sha256_file(path: Path, progress: Callable[[int], None] | None = None) -> str:
    h = hashlib.sha256()
    done = 0
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
            done += len(chunk)
            if progress is not None:
                progress(done)
    return h.hexdigest()


def _entry(name: str, raw: Mapping[str, Any]) -> ModelEntry:
    def text(key: str) -> str:
        value = raw.get(key, "")
        if not isinstance(value, str):
            raise ModelError(f"manifest: models.{name}.{key} must be a string")
        return value

    size = raw.get("size", 0)
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ModelError(f"manifest: models.{name}.size must be a non-negative integer")
    required = raw.get("required_by", [])
    if not isinstance(required, list):
        raise ModelError(f"manifest: models.{name}.required_by must be a list")
    entry = ModelEntry(
        name=name,
        url=text("url"),
        sha256=text("sha256").lower(),
        size=size,
        licence=text("licence"),
        required_by=tuple(str(r) for r in required),
        dest=text("dest"),
        variant=text("variant"),
        unpack=text("unpack"),
        optional=bool(raw.get("optional", False)),
        notes=text("notes"),
    )
    if not entry.url or not entry.dest:
        raise ModelError(f"manifest: models.{name} needs url and dest")
    if entry.sha256 and (len(entry.sha256) != 64 or not all(c in "0123456789abcdef" for c in entry.sha256)):
        raise ModelError(f"manifest: models.{name}.sha256 is not a hex sha256")
    if entry.unpack not in ("", "zip"):
        raise ModelError(f"manifest: models.{name}.unpack must be \"zip\" or empty")
    if PurePosixPath(entry.dest).is_absolute() or ".." in PurePosixPath(entry.dest).parts:
        raise ModelError(f"manifest: models.{name}.dest must stay inside the project folder")
    return entry


@dataclass
class ModelManifest:
    """The parsed manifest plus the project root its ``dest`` paths are relative to."""

    entries: dict[str, ModelEntry]
    root: Path
    path: Path
    _stamp_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def load(cls, path: Path, *, root: Path | None = None) -> ModelManifest:
        """Read ``path`` (``models/manifest.toml``); ``root`` defaults to its grandparent."""
        path = Path(path)
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ModelError(f"cannot read {path}: {exc}") from exc
        models = data.get("models", {})
        if not isinstance(models, dict):
            raise ModelError(f"{path}: [models] must be a table")
        entries = {
            str(n): _entry(str(n), raw) for n, raw in models.items() if isinstance(raw, dict)
        }
        return cls(entries, Path(root) if root is not None else path.resolve().parent.parent, path)

    # -- selection -----------------------------------------------------------------------------

    def names(self) -> list[str]:
        return list(self.entries)

    def entry(self, name: str) -> ModelEntry:
        try:
            return self.entries[name]
        except KeyError:
            raise ModelError(f"unknown model {name!r}; known: {', '.join(self.entries)}") from None

    def for_variant(self, variant: str) -> list[str]:
        """Every non-optional entry, keeping only ``variant`` among the variant entries."""
        return [
            n for n, e in self.entries.items()
            if not e.optional and (not e.variant or e.variant == variant)
        ]

    def required_by(self, key: str, *, prefix: bool = False) -> list[ModelEntry]:
        """Entries needed by a config key (``llm.servers.local30b``, ``vad.model`` …); with
        ``prefix`` also those needed by its sub-keys (``llm.servers.local30b.exe``)."""
        return [
            e for e in self.entries.values()
            if any(r == key or (prefix and r.startswith(key + ".")) for r in e.required_by)
        ]

    def dest_path(self, entry: ModelEntry) -> Path:
        return self.root / PurePosixPath(entry.dest)

    def archive_path(self, entry: ModelEntry) -> Path:
        return self.root / "models" / DOWNLOADS / entry.file_name()

    def file_path(self, entry: ModelEntry) -> Path:
        """The file whose size/sha256 the manifest pins (the archive for ``unpack``)."""
        return self.archive_path(entry) if entry.archive else self.dest_path(entry)

    def lock_path(self, entry: ModelEntry) -> Path:
        return self.root / "models" / DOWNLOADS / f"{entry.name}.lock"

    def downloading(self, name: str) -> bool:
        """Whether another process is pulling ``name`` right now (it holds the lock)."""
        lock = _FileLock(self.lock_path(self.entry(name)))
        try:
            if not lock.acquire():
                return True
        except OSError:
            return False
        lock.release()
        return False

    # -- stamps ------------------------------------------------------------------------------

    @property
    def stamp_path(self) -> Path:
        return self.root / "models" / STAMP_FILE

    def _read_stamps(self) -> dict[str, Any]:
        try:
            data = json.loads(self.stamp_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _key(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def stamped_sha(self, path: Path) -> str | None:
        """The sha256 recorded for ``path`` if its size and mtime are unchanged."""
        try:
            st = path.stat()
        except OSError:
            return None
        rec = self._read_stamps().get(self._key(path))
        if (
            isinstance(rec, dict)
            and rec.get("size") == st.st_size
            and rec.get("mtime_ns") == st.st_mtime_ns
            and isinstance(rec.get("sha256"), str)
        ):
            return str(rec["sha256"])
        return None

    def stamp(self, path: Path, sha: str) -> None:
        try:
            st = path.stat()
        except OSError:
            return
        with self._stamp_lock:
            data = self._read_stamps()
            data[self._key(path)] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha}
            try:
                self.stamp_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.stamp_path.with_name(STAMP_FILE + ".tmp")
                tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
                os.replace(tmp, self.stamp_path)
            except OSError:
                log.warning("cannot write %s", self.stamp_path, exc_info=True)

    # -- checks ------------------------------------------------------------------------------

    def check(
        self,
        name: str,
        *,
        full: bool = True,
        progress: Callable[[int], None] | None = None,
    ) -> CheckResult:
        """Check one entry. ``full=False`` hashes only small files and trusts stamps."""
        entry = self.entry(name)
        path = self.file_path(entry)
        if entry.archive and not path.is_file():
            marker = self._extracted_marker(entry)
            if marker.is_file() and marker.read_text(encoding="utf-8").strip() == entry.sha256:
                return CheckResult(name, "ok", self.dest_path(entry), "extracted", entry.sha256)
        if not path.is_file():
            part = path.with_name(path.name + ".part")
            if part.is_file():
                return CheckResult(name, "partial", path, f"{part.stat().st_size} bytes so far")
            return CheckResult(name, "missing", path)
        size = path.stat().st_size
        if entry.size and size != entry.size:
            return CheckResult(
                name, "size", path, f"size {size} bytes, expected {entry.size}"
            )
        sha = self.stamped_sha(path)
        if sha is None and (full or size <= CHEAP_SHA_BYTES):
            sha = sha256_file(path, progress)
            self.stamp(path, sha)
        if sha is None:
            status: Status = "ok" if entry.pinned else "unpinned"
            return CheckResult(name, status, path, "size ok; sha256 not checked yet")
        if not entry.sha256:
            return CheckResult(name, "unpinned", path, f"sha256 {sha} (not pinned)", sha)
        if sha != entry.sha256:
            return CheckResult(name, "sha", path, f"sha256 {sha}, expected {entry.sha256}", sha)
        return CheckResult(name, "ok", path, "verified", sha)

    def verify(self, names: Iterable[str] | None = None, *, full: bool = True) -> dict[str, bool]:
        """``{name: ok}`` for ``names`` (default: every entry that is present or partial)."""
        out: dict[str, bool] = {}
        for name in names if names is not None else self.entries:
            out[name] = self.check(name, full=full).ok
        return out

    # -- pull --------------------------------------------------------------------------------

    def pull(
        self,
        names: Sequence[str],
        *,
        background: bool = False,
        progress: Progress | None = None,
        opener: urllib.request.OpenerDirector | None = None,
        retries: int = 5,
        timeout_s: float = 30.0,
    ) -> threading.Thread | None:
        """Download and verify ``names`` in order (skipping verified ones). With
        ``background`` the work runs on a daemon thread, which is returned."""
        for name in names:
            self.entry(name)  # unknown names fail now, not in the thread

        def work() -> None:
            for name in names:
                self.pull_one(name, progress=progress, opener=opener, retries=retries,
                              timeout_s=timeout_s)

        if not background:
            work()
            return None

        def guarded() -> None:
            try:
                work()
            except Exception:
                log.exception("background model download failed")

        thread = threading.Thread(target=guarded, name="models-pull", daemon=True)
        thread.start()
        return thread

    def pull_one(
        self,
        name: str,
        *,
        progress: Progress | None = None,
        opener: urllib.request.OpenerDirector | None = None,
        retries: int = 5,
        timeout_s: float = 30.0,
    ) -> CheckResult:
        entry = self.entry(name)
        current = self.check(name, full=True)
        if current.ok and (not entry.archive or self._extracted(entry)):
            return current
        lock = _FileLock(self.lock_path(entry))
        try:
            locked = lock.acquire()
        except OSError as exc:
            raise ModelError(f"{name}: cannot create the download lock: {exc}") from exc
        if not locked:
            raise ModelBusy(f"{name} is already being downloaded by another process")
        try:
            return self._pull_locked(entry, progress=progress, opener=opener, retries=retries,
                                     timeout_s=timeout_s)
        finally:
            lock.release()

    def _pull_locked(
        self,
        entry: ModelEntry,
        *,
        progress: Progress | None,
        opener: urllib.request.OpenerDirector | None,
        retries: int,
        timeout_s: float,
    ) -> CheckResult:
        name = entry.name
        current = self.check(name, full=True)  # again: another process may have finished it
        if current.ok and (not entry.archive or self._extracted(entry)):
            return current
        path = self.file_path(entry)
        if current.status in ("size", "sha"):
            log.warning("%s: %s; downloading again", name, current.detail)
            with contextlib.suppress(OSError):
                path.unlink()
        if not path.is_file():
            self._download(entry, path, progress=progress, opener=opener, retries=retries,
                           timeout_s=timeout_s)
        result = self.check(name, full=True)
        if not result.ok:
            with contextlib.suppress(OSError):
                path.unlink()
            raise ModelError(f"{name}: {result.status}: {result.detail}")
        if entry.archive:
            self._extract(entry, path, result.sha256)
        if result.status == "unpinned":
            log.warning("%s is not pinned in the manifest; its sha256 is %s", name, result.sha256)
        return result

    def _download(
        self,
        entry: ModelEntry,
        path: Path,
        *,
        progress: Progress | None,
        opener: urllib.request.OpenerDirector | None,
        retries: int,
        timeout_s: float,
    ) -> None:
        if not entry.url.startswith(("https://", "http://")):
            raise ModelError(f"{entry.name}: unsupported URL {entry.url!r}")
        part = path.with_name(path.name + ".part")
        part.parent.mkdir(parents=True, exist_ok=True)
        opener = opener or urllib.request.build_opener()
        attempt = 0
        while True:
            try:
                if self._fetch(entry, part, opener, progress, timeout_s):
                    break
                problem = "the server closed the download early"
            except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
                if isinstance(exc, urllib.error.HTTPError):
                    exc.close()
                    if exc.code in (401, 403, 404, 410):
                        raise ModelError(f"{entry.name}: HTTP {exc.code} for {entry.url}") from exc
                if attempt >= retries:
                    raise ModelError(f"{entry.name}: download failed: {exc}") from exc
                problem = str(exc)
            attempt += 1
            if attempt > retries:
                raise ModelError(f"{entry.name}: download incomplete after {retries} retries")
            delay = min(30.0, 0.5 * 2 ** (attempt - 1))
            log.warning("%s: %s; resuming in %.1f s (%d/%d)", entry.name, problem, delay,
                        attempt, retries)
            time.sleep(delay)
        os.replace(part, path)

    def _fetch(
        self,
        entry: ModelEntry,
        part: Path,
        opener: urllib.request.OpenerDirector,
        progress: Progress | None,
        timeout_s: float,
    ) -> bool:
        """One request. Returns ``True`` when ``part`` is complete."""
        have = part.stat().st_size if part.is_file() else 0
        if entry.size and have == entry.size:
            return True
        if entry.size and have > entry.size:
            part.unlink()
            have = 0
        req = urllib.request.Request(entry.url, headers={"User-Agent": "aivtube-models/1"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            resp = opener.open(req, timeout=timeout_s)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and have:
                exc.close()
                return True  # nothing left to send: the size check decides
            raise
        with resp:
            status = resp.status
            if have and status == 206:
                mode = "ab"
            else:
                mode, have = "wb", 0  # the server ignored the range: start again
            total = entry.size or have + int(resp.headers.get("Content-Length") or 0)
            with part.open(mode) as out:
                done = have
                while True:
                    try:
                        chunk = resp.read(CHUNK)
                    except http.client.IncompleteRead as exc:
                        out.write(exc.partial)  # keep what arrived; the retry resumes after it
                        raise
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(entry.name, done, total)
        size = part.stat().st_size
        return not entry.size or size >= entry.size

    # -- archives ----------------------------------------------------------------------------

    def _extracted_marker(self, entry: ModelEntry) -> Path:
        return self.dest_path(entry) / f".{entry.name}.extracted"

    def _extracted(self, entry: ModelEntry) -> bool:
        marker = self._extracted_marker(entry)
        try:
            content = marker.read_text(encoding="utf-8").strip()
        except OSError:
            return False
        return content == entry.sha256 if entry.sha256 else bool(content)

    def _extract(self, entry: ModelEntry, archive: Path, sha: str) -> None:
        dest = self.dest_path(entry)
        dest.mkdir(parents=True, exist_ok=True)
        root = dest.resolve()
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                target = (dest / member.filename).resolve()
                if target != root and root not in target.parents:
                    raise ModelError(f"{entry.name}: unsafe path in archive: {member.filename}")
            zf.extractall(dest)
        self._extracted_marker(entry).write_text(sha, encoding="utf-8")
        log.info("%s extracted into %s", entry.name, dest)


# --- command line ---------------------------------------------------------------------------


class ProgressLine:
    """A single updating progress line (``\\r``), at most every ``every_s`` seconds."""

    def __init__(self, stream: IO[str] | None = None, every_s: float = 0.5) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.every_s = every_s
        self._last = 0.0
        self._t0: dict[str, tuple[float, int]] = {}

    def __call__(self, name: str, done: int, total: int) -> None:
        now = time.perf_counter()
        t0, start = self._t0.setdefault(name, (now, done))
        if now - self._last < self.every_s and done < total:
            return
        self._last = now
        rate = (done - start) / max(1e-6, now - t0) / 2**20
        pct = f"{100 * done / total:5.1f}%" if total else "  ?  "
        line = f"\r{name:24s} {pct} {done / 2**30:7.2f}/{total / 2**30:.2f} GiB {rate:7.1f} MiB/s"
        with contextlib.suppress(OSError, ValueError):
            self.stream.write(line + ("\n" if total and done >= total else ""))
            self.stream.flush()


def main(argv: list[str] | None = None) -> int:
    """``python -m aivtube.ops.models {pull,verify} [names…] [--root DIR]``."""
    parser = argparse.ArgumentParser(prog="aivtube models")
    parser.add_argument("action", choices=["pull", "verify", "list"])
    parser.add_argument("names", nargs="*")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--variant", default="", help="cuda-13.4 | cuda-12.4 (default: detect)")
    parser.add_argument("--then-profile", default="",
                        help="after a successful pull, switch active_profile from light to this")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = args.root or _find_root()
    manifest = ModelManifest.load(root / MANIFEST, root=root)
    variant = args.variant or _detect_variant()
    names = args.names or manifest.for_variant(variant)
    if args.action == "list":
        for n in names:
            e = manifest.entry(n)
            print(f"{n:24s} {e.size / 2**30:8.2f} GiB  {e.licence:12s} {e.dest}")
        return 0
    if args.action == "verify":
        bad = 0
        for n in names:
            r = manifest.check(n, full=True)
            mark = "✔" if r.ok else "✖"
            print(f"{mark} {n:24s} {r.status:9s} {r.detail}")
            bad += 0 if r.ok or (r.status == "missing" and manifest.entry(n).optional) else 1
        return 1 if bad else 0
    failed = 0
    bar = ProgressLine()
    for n in names:
        try:
            r = manifest.pull_one(n, progress=bar)
        except ModelError as exc:
            failed += 1
            print(f"✖ {exc}", file=sys.stderr)
            continue
        print(f"✔ {n:24s} {r.status:9s} {r.detail}")
    if failed:
        return 1
    if args.then_profile:
        _promote_profile(root, args.then_profile)
    return 0


def _find_root() -> Path:
    from aivtube.config.layers import find_root

    return find_root()


def _detect_variant() -> str:
    from aivtube.launcher.gpu import driver_major, query_gpus

    gpus = query_gpus() or []
    return cuda_variant(driver_major(str(gpus[0].get("driver", ""))) if gpus else None)


def _promote_profile(root: Path, profile: str) -> None:
    """Switch ``active_profile`` from ``light`` to ``profile`` once the big model verified."""
    from aivtube.config.layers import USER_FILE, read_toml

    user = read_toml(root / USER_FILE, required=False)
    if user.get("active_profile") == "light":
        from aivtube.config import write_user_overrides

        write_user_overrides(root, {"active_profile": profile})
        print(f"active_profile: light → {profile}")


if __name__ == "__main__":
    sys.exit(main())
