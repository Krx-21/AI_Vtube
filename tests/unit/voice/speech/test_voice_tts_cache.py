"""``DiskPhraseCache``: contract suite, layout, persistence, marks and corruption (voice.tts)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from aivtube.contracts.types import VoiceSpec
from aivtube.contracts.voice import AudioChunk, PhraseCache, WordMark
from aivtube.testing.contracts import case_id, phrase_cache_suite
from aivtube.voice.tts import DiskPhraseCache, normalize_phrase, phrase_key

VOICE = VoiceSpec("premwadee", "th-TH-PremwadeeNeural", rate="+8%", pitch="+20Hz")


def _audio(n: int = 2400, sr: int = 24000) -> AudioChunk:
    return AudioChunk((np.arange(n) % 200 - 100).astype(np.int16), sr)


def _suite(tmp_path_factory: pytest.TempPathFactory) -> list[Any]:
    return phrase_cache_suite(lambda: DiskPhraseCache(tmp_path_factory.mktemp("phrases")))


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "suite_case" in metafunc.fixturenames:
        cases = phrase_cache_suite(lambda: None)  # names only; rebuilt with a temp dir below
        metafunc.parametrize("suite_case", [c.__name__ for c in cases])


def test_disk_cache_passes_the_phrase_cache_suite(
    suite_case: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    cases = {case_id(c): c for c in _suite(tmp_path_factory)}
    cases[suite_case]()


def test_layout_is_identity_dir_with_pcm_and_json(tmp_path: Path) -> None:
    cache = DiskPhraseCache(tmp_path)
    assert isinstance(cache, PhraseCache)
    key = phrase_key(VOICE, "Filtered.")
    assert key == ("premwadee", "th-TH-PremwadeeNeural", "+8%", "+20Hz", "+0%", "Filtered.")
    cache.put(key, _audio(), [WordMark("Filtered.", 0.05, 0.6)])
    folder = tmp_path / "premwadee"
    pcms = list(folder.glob("*.pcm"))
    metas = list(folder.glob("*.json"))
    assert len(pcms) == 1 and len(metas) == 1 and pcms[0].stem == metas[0].stem
    meta = json.loads(metas[0].read_text(encoding="utf-8"))
    assert meta["sample_rate"] == 24000 and meta["samples"] == 2400 and meta["key"] == list(key)
    assert pcms[0].stat().st_size == 4800
    assert not list(folder.glob("*.tmp"))


def test_persists_across_instances_with_marks(tmp_path: Path) -> None:
    key = phrase_key(VOICE, "  อืม…  ")
    assert key[-1] == "อืม…"
    DiskPhraseCache(tmp_path).put(key, _audio(), [WordMark("อืม…", 0.0, 0.4)])
    fresh = DiskPhraseCache(tmp_path)
    got = fresh.get_with_marks(key)
    assert got is not None
    audio, marks = got
    assert np.array_equal(audio.pcm, _audio().pcm) and audio.sample_rate == 24000
    assert marks == (WordMark("อืม…", 0.0, 0.4),)
    assert fresh.get_marks(key) == marks
    assert key in fresh and ("x",) not in fresh and "x" not in fresh
    fresh.delete(key)
    assert fresh.get(key) is None and key not in fresh


def test_prosody_is_part_of_the_key(tmp_path: Path) -> None:
    cache = DiskPhraseCache(tmp_path)
    cache.put(phrase_key(VOICE, "ขอบคุณค่ะ"), _audio())
    faster = VoiceSpec(VOICE.identity, VOICE.voice, rate="+18%", pitch=VOICE.pitch)
    assert cache.get(phrase_key(faster, "ขอบคุณค่ะ")) is None
    assert cache.get(phrase_key(VOICE, "ขอบคุณค่ะ")) is not None


def test_corrupt_entries_are_dropped(tmp_path: Path) -> None:
    key = phrase_key(VOICE, "เอ่อ…")
    DiskPhraseCache(tmp_path).put(key, _audio())
    pcm = next((tmp_path / "premwadee").glob("*.pcm"))
    pcm.write_bytes(b"\x00\x01\x02")  # truncated
    fresh = DiskPhraseCache(tmp_path)
    assert fresh.get(key) is None
    assert not pcm.exists()
    assert fresh.misses == 1


def test_memory_lru_is_bounded_and_unsafe_identity_names_are_sanitised(tmp_path: Path) -> None:
    cache = DiskPhraseCache(tmp_path, memory_items=2)
    keys = [("../../evil id", "v", f"t{i}") for i in range(3)]
    for k in keys:
        cache.put(k, _audio(10))
    assert len(cache._memory) == 2
    assert all(p.is_relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert (tmp_path / "evil_id").is_dir()
    assert cache.get(keys[0]) is not None  # evicted from memory, read back from disk


def test_normalize_phrase() -> None:
    assert normalize_phrase("  สวัสดี \n ค่ะ ") == "สวัสดี ค่ะ"
