"""Small, dependency-free chat text rules: normalisation, near-duplicate keys, name and
question detection (ARCHITECTURE.md §4.3; research chat.md "Thai text" pitfalls).

Thai has no spaces between words, so names are matched as substrings of normalised text.
Viewers insert zero-width characters (U+200B is common in Thai), stretch letters
(``ไพลินนนน``) and spam laughter (``55555``); every rule here normalises that away. These
helpers never touch the text that is displayed or spoken.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

__all__ = [
    "AliasMatcher",
    "collapse_runs",
    "dedupe_key",
    "is_question",
    "normalize",
    "strip_zero_width",
]

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­᠎‎‏"), None)
_THAI_TONES = dict.fromkeys(range(0x0E48, 0x0E4C), None)  # mai ek, tho, tri, chattawa
_WS = re.compile(r"\s+")
_RUNS = re.compile(r"(.)\1+", re.DOTALL)

# Sentence-final Thai politeness/softening particles, stripped before looking for a question word.
_PARTICLES = (
    "ครับผม",
    "คร้าบ",
    "ครับ",
    "ค้าบ",
    "คับ",
    "ค่ะ",
    "คะ",
    "จ้า",
    "จ้ะ",
    "จ๊ะ",
    "น้า",
    "นะ",
    "ฮะ",
    "อ่ะ",
    "อะ",
    "เนี่ย",
    "วะ",
)
# Question words/particles that end a Thai question ("กินข้าวยัง", "ชอบอะไร", "ไปไหม").
_QUESTION_ENDINGS = (
    "ไหม",
    "มั้ย",
    "มั๊ย",
    "ไม๊",
    "หรอ",
    "เหรอ",
    "หรือ",
    "รึ",
    "ยังไง",
    "อย่างไร",
    "อะไร",
    "ทำไม",
    "ที่ไหน",
    "ไหน",
    "เมื่อไหร่",
    "เมื่อไร",
    "เท่าไหร่",
    "เท่าไร",
    "ป่าว",
    "เปล่า",
    "ป่ะ",
    "ปะ",
    "มะ",
    "ยัง",
    "บ้าง",
    "ใคร",
)
# Unambiguous question words that make a question wherever they appear ("ทำไมไม่ไลฟ์เมื่อวาน").
_QUESTION_ANYWHERE = (
    "ทำไม",
    "ที่ไหน",
    "เมื่อไหร่",
    "เมื่อไร",
    "เท่าไหร่",
    "กี่โมง",
    "อะไรอยู่",
    "ไหนอยู่",
    "ใช่ไหม",
    "ได้ไหม",
    "ได้มั้ย",
)
_EN_QUESTION_START = re.compile(
    r"^(what|why|how|who|whom|where|when|which|can you|could you|do you|did you|are you|"
    r"is it|will you|would you|have you)\b"
)


def _trim_tail(text: str) -> str:
    """Drop trailing punctuation, emoji, spaces and ``555`` laughter.

    Uses Unicode categories, not ``\\W``: Python's ``\\W`` matches Thai vowel and tone marks.
    """
    end = len(text)
    while end and (text[end - 1] == "5" or unicodedata.category(text[end - 1])[0] not in "LMN"):
        end -= 1
    return text[:end]


def strip_zero_width(text: str) -> str:
    """Remove zero-width and soft-hyphen characters."""
    return text.translate(_ZERO_WIDTH)


def collapse_runs(text: str) -> str:
    """Collapse every run of one repeated character to a single character."""
    return _RUNS.sub(r"\1", text)


def normalize(text: str) -> str:
    """NFKC, zero-width strip, casefold and single spaces (display text is never changed)."""
    t = strip_zero_width(unicodedata.normalize("NFKC", text))
    return _WS.sub(" ", t).strip().casefold()


# NFKC decomposes SARA AM (ำ → ํา), so the word lists are compared in normalised form too.
_N_PARTICLES = tuple(normalize(p) for p in _PARTICLES)
_N_ENDINGS = tuple(normalize(q) for q in _QUESTION_ENDINGS)
_N_ANYWHERE = tuple(normalize(q) for q in _QUESTION_ANYWHERE)


def dedupe_key(text: str) -> str:
    """Key under which near-identical messages ("555555", "5555 !!", "ไพลินนน") collide.

    Keeps only letters, marks and digits of the normalised text and collapses repeated
    characters. Text made only of symbols or emoji keys on its normalised form instead.
    """
    norm = normalize(text)
    core = "".join(ch for ch in norm if unicodedata.category(ch)[0] in "LMN")
    core = collapse_runs(core)
    return core if core else collapse_runs(norm.replace(" ", ""))


def _match_form(text: str) -> str:
    """Form used for alias matching: normalised, no spaces, no Thai tone marks, no runs."""
    t = normalize(text).translate(_THAI_TONES)
    return collapse_runs(_WS.sub("", t))


class AliasMatcher:
    """``name_matcher`` for :class:`~aivtube.chat.window.ScoredChatWindow`.

    True when the text contains one of the character's aliases after NFKC, zero-width
    stripping, casefolding, space removal, tone-mark removal and run collapsing, so
    ``"ไพ ลินนน"``, ``"ไพ่ลิน"``, ``"@PaiLin"`` and ``"ไพ​ลิน"`` all match ``ไพลิน``.
    """

    __slots__ = ("_forms", "aliases")

    def __init__(self, aliases: Iterable[str]) -> None:
        names = (aliases,) if isinstance(aliases, str) else tuple(aliases)
        self.aliases: tuple[str, ...] = names
        forms = {_match_form(a) for a in names}
        self._forms: tuple[str, ...] = tuple(sorted(f for f in forms if len(f) >= 2))

    def __call__(self, text: str) -> bool:
        if not self._forms or not text:
            return False
        form = _match_form(text)
        return any(alias in form for alias in self._forms)

    def __repr__(self) -> str:
        return f"AliasMatcher({list(self.aliases)!r})"


def is_question(text: str) -> bool:
    """Heuristic question shape: a ``?`` or a sentence-final Thai/English question form."""
    if "?" in text or "？" in text:
        return True
    t = normalize(text)
    if _EN_QUESTION_START.match(t) or any(q in t for q in _N_ANYWHERE):
        return True
    t = _trim_tail(t)
    changed = True
    while changed and t:
        changed = False
        for p in _N_PARTICLES:
            if t.endswith(p) and len(t) > len(p):
                t = _trim_tail(t[: -len(p)])
                changed = True
                break
    return t.endswith(_N_ENDINGS)
