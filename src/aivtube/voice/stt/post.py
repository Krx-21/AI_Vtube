"""Transcript post-processing: name aliases, short-audio drop and echo drop (§4.7, STT brief D).

Typhoon RT often mishears the character's name (``ไพลิน`` → ``ไทลิน``/``ไทยลิน``/``ไทลิล``); the
character's ``[stt_aliases]`` map fixes that by substring replacement (Thai has no spaces
between words). Transcripts of audio shorter than ``min_audio_s`` are dropped (hallucination
guard), and so are transcripts that match her own recent TTS text (``difflib`` ratio above
``echo_ratio``), which happens with speakers and a leaky room.
"""

from __future__ import annotations

import dataclasses
import difflib
import re
import unicodedata
from collections.abc import Mapping, Sequence

from aivtube.contracts.types import Transcript

__all__ = ["NamePostProcessor", "echo_similarity"]

_SPACES = re.compile(r"\s+")
_NOISE = re.compile(r"[\s​-‍﻿.,!?…:;\"'()\[\]{}\-–—~*_/]+")


def _squash(text: str) -> str:
    """Comparison form: NFC, casefolded, without spaces, zero-widths and punctuation."""
    return _NOISE.sub("", unicodedata.normalize("NFC", text).casefold())


def echo_similarity(text: str, reference: str) -> float:
    """Best ``difflib`` ratio of ``text`` against ``reference`` or any same-length window of it.

    A short transcript of her own voice matches a small part of the last seconds of TTS text,
    so the whole-string ratio alone would miss it.
    """
    a, b = _squash(text), _squash(reference)
    if not a or not b:
        return 0.0
    best = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    n = len(a)
    if n >= len(b):
        return best
    step = max(1, n // 4)
    matcher = difflib.SequenceMatcher(None, autojunk=False)
    matcher.set_seq2(a)
    for start in range(0, len(b) - n + step, step):
        window = b[start : start + n]
        matcher.set_seq1(window)
        if matcher.real_quick_ratio() <= best or matcher.quick_ratio() <= best:
            continue
        best = max(best, matcher.ratio())
        if best >= 0.999:
            break
    return best


class NamePostProcessor:
    """``TranscriptPostProcessor``: alias map, short-audio drop and echo drop.

    ``aliases`` maps each canonical spelling to its mishearings, e.g.
    ``{"ไพลิน": ["ไทลิน", "ไทยลิน", "ไทลิล"]}``. Latin aliases match case-insensitively.
    Returns ``None`` to drop the transcript (empty, too short, or an echo).
    """

    def __init__(
        self,
        aliases: Mapping[str, Sequence[str]],
        *,
        min_audio_s: float = 0.3,
        echo_ratio: float = 0.6,
        echo_min_chars: int = 4,
    ) -> None:
        self.min_audio_s = min_audio_s
        self.echo_ratio = echo_ratio
        self.echo_min_chars = echo_min_chars
        self._map: dict[str, str] = {}
        for canonical, wrongs in aliases.items():
            canon = unicodedata.normalize("NFC", canonical.strip())
            if not canon:
                continue
            self._map.setdefault(canon.casefold(), canon)  # the canonical form maps to itself
            for wrong in wrongs:
                w = unicodedata.normalize("NFC", str(wrong).strip())
                if w and w.casefold() not in self._map:
                    self._map[w.casefold()] = canon
        # longest first, so a canonical spelling that contains an alias is kept intact
        keys = sorted(self._map, key=len, reverse=True)
        self._pattern = (
            re.compile("|".join(re.escape(k) for k in keys), re.IGNORECASE) if keys else None
        )

    def fix_names(self, text: str) -> str:
        """Replace every alias with its canonical spelling."""
        if self._pattern is None:
            return text
        return self._pattern.sub(lambda m: self._map.get(m.group(0).casefold(), m.group(0)), text)

    def __call__(self, t: Transcript, recent_tts_text: str) -> Transcript | None:
        if t.audio_s < self.min_audio_s:
            return None
        text = _SPACES.sub(" ", unicodedata.normalize("NFC", t.text)).strip()
        if not text:
            return None
        text = self.fix_names(text)
        if (
            recent_tts_text
            and len(_squash(text)) >= self.echo_min_chars
            and echo_similarity(text, recent_tts_text) > self.echo_ratio
        ):
            return None
        if text == t.text:
            return t
        return dataclasses.replace(t, text=text)
