"""On-disk phrase cache: ``<root>/<identity>/<sha>.pcm`` plus a ``.json`` sidecar (§2.6, §4.10).

Holds pre-synthesised phrases per voice identity: ``"Filtered."``, the fillers (``อืม…``), the
brain-freeze line and anything else the character lists in ``cached_phrases``. Keys come from
``phrase_key(voice, text)``: ``(identity, voice, rate, pitch, volume, text)``, so a prosody
change never replays stale audio. The sidecar records the sample rate, the sample count and the
word marks (for heard-text truncation). Writes are atomic (temp file + ``os.replace``), and a
small in-memory LRU keeps hot phrases off the disk.
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import json
import logging
import os
import re
import threading
import unicodedata
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from aivtube.contracts.types import VoiceSpec
from aivtube.contracts.voice import AudioChunk, WordMark

__all__ = ["DiskPhraseCache", "normalize_phrase", "phrase_key"]

log = logging.getLogger("aivtube.voice.tts.cache")

_SPACES = re.compile(r"\s+")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_FORMAT = 1


def normalize_phrase(text: str) -> str:
    """Cache form of a phrase: NFC, trimmed, single spaces."""
    return _SPACES.sub(" ", unicodedata.normalize("NFC", text)).strip()


def phrase_key(voice: VoiceSpec, text: str) -> tuple[str, ...]:
    """``(identity, voice, rate, pitch, volume, normalised text)``."""
    return (
        voice.identity,
        voice.voice,
        voice.rate,
        voice.pitch,
        voice.volume,
        normalize_phrase(text),
    )


class DiskPhraseCache:
    """``PhraseCache`` on disk. The first key element names the directory (the identity)."""

    def __init__(self, root: Path | str, *, memory_items: int = 64) -> None:
        self.root = Path(root)
        self._memory: collections.OrderedDict[
            tuple[str, ...], tuple[AudioChunk, tuple[WordMark, ...]]
        ]
        self._memory = collections.OrderedDict()
        self._memory_items = memory_items
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    # --- PhraseCache protocol -------------------------------------------------------------
    def get(self, key: tuple[str, ...]) -> AudioChunk | None:
        entry = self._load(tuple(key))
        return entry[0] if entry is not None else None

    def put(self, key: tuple[str, ...], audio: AudioChunk, marks: Iterable[WordMark] = ()) -> None:
        key = tuple(key)
        pcm = np.ascontiguousarray(np.asarray(audio.pcm, dtype=np.int16).reshape(-1))
        marks_t = tuple(marks)
        folder, pcm_path, meta_path = self._paths(key)
        meta = {
            "format": _FORMAT,
            "key": list(key),
            "sample_rate": int(audio.sample_rate),
            "samples": int(pcm.size),
            "marks": [[m.text, m.offset_s, m.duration_s] for m in marks_t],
        }
        try:
            folder.mkdir(parents=True, exist_ok=True)
            _atomic_write(pcm_path, pcm.astype("<i2").tobytes())
            _atomic_write(meta_path, json.dumps(meta, ensure_ascii=False).encode("utf-8"))
        except OSError as exc:
            log.warning("phrase cache write failed for %r: %s", key[-1:], exc)
        self._remember(key, AudioChunk(pcm.copy(), int(audio.sample_rate)), marks_t)

    # --- extras ---------------------------------------------------------------------------
    def get_marks(self, key: tuple[str, ...]) -> tuple[WordMark, ...]:
        entry = self._load(tuple(key))
        return entry[1] if entry is not None else ()

    def get_with_marks(
        self, key: tuple[str, ...]
    ) -> tuple[AudioChunk, tuple[WordMark, ...]] | None:
        return self._load(tuple(key))

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, tuple):
            return False
        with self._lock:
            if key in self._memory:
                return True
        return self._paths(key)[2].is_file()

    def delete(self, key: tuple[str, ...]) -> None:
        key = tuple(key)
        with self._lock:
            self._memory.pop(key, None)
        _folder, pcm_path, meta_path = self._paths(key)
        for p in (meta_path, pcm_path):
            with contextlib.suppress(OSError):
                p.unlink(missing_ok=True)

    # --- internals ------------------------------------------------------------------------
    def _paths(self, key: tuple[str, ...]) -> tuple[Path, Path, Path]:
        name = _UNSAFE.sub("_", key[0]).strip("._") if key else ""
        folder = self.root / (name or "_")
        digest = hashlib.sha256("\x1f".join(key).encode("utf-8")).hexdigest()[:40]
        return folder, folder / f"{digest}.pcm", folder / f"{digest}.json"

    def _remember(
        self, key: tuple[str, ...], audio: AudioChunk, marks: tuple[WordMark, ...]
    ) -> None:
        with self._lock:
            self._memory[key] = (audio, marks)
            self._memory.move_to_end(key)
            while len(self._memory) > self._memory_items:
                self._memory.popitem(last=False)

    def _load(self, key: tuple[str, ...]) -> tuple[AudioChunk, tuple[WordMark, ...]] | None:
        with self._lock:
            hit = self._memory.get(key)
            if hit is not None:
                self._memory.move_to_end(key)
                self.hits += 1
                return hit
        _folder, pcm_path, meta_path = self._paths(key)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            raw = pcm_path.read_bytes()
        except (OSError, ValueError):
            self.misses += 1
            return None
        try:
            if meta.get("key") != list(key) or meta.get("format") != _FORMAT:
                raise ValueError("key or format mismatch")
            samples = int(meta["samples"])
            if len(raw) != 2 * samples:
                raise ValueError(f"size {len(raw)} != {2 * samples}")
            pcm = np.frombuffer(raw, dtype="<i2").astype(np.int16)
            marks = tuple(WordMark(str(t), float(o), float(d)) for t, o, d in meta.get("marks", []))
            audio = AudioChunk(pcm, int(meta["sample_rate"]))
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("phrase cache entry %s is corrupt (%s); dropping it", meta_path.name, exc)
            self.delete(key)
            self.misses += 1
            return None
        self.hits += 1
        self._remember(key, audio, marks)
        return audio, marks


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()
