"""Thai text helpers: character classes, safe cut points, particles, the lazily warmed newmm
word tokenizer, and the small matching utilities used by intake and safety (§4.6, §7).

Pure Python at import time. pythainlp is imported only when the tokenizer is warmed or first
used (import ≈ 70 ms, first newmm call ≈ 350 ms while the dictionary loads, then ≈ 0.06 ms per
sentence). Call ``warm_up_pythainlp()`` once at startup, in a worker thread
(``await asyncio.to_thread(warm_up_pythainlp)``), so no event loop ever pays that cost.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import threading
import time
import unicodedata
from collections.abc import Callable, Sequence
from typing import Final

__all__ = [
    "AMBIGUOUS_PARTICLES",
    "ATTACH_ALWAYS",
    "ATTACH_LEFT_PARTICLES",
    "FOLLOWING_VOWELS",
    "LEADING_VOWELS",
    "NO_CUT_AFTER",
    "NO_CUT_BEFORE",
    "SENTENCE_FINAL_PARTICLES",
    "THAI_COMBINING",
    "NameMatcher",
    "ThaiWordTokenizer",
    "attach_left",
    "collapse_repeats",
    "ends_with_final_particle",
    "estimate_tokens",
    "is_combining",
    "is_question",
    "is_safe_cut",
    "is_thai",
    "is_thai_letter",
    "newmm",
    "nfkc_casefold",
    "strip_zero_width",
    "thai_digits_to_arabic",
    "warm_up_pythainlp",
]

log = logging.getLogger("aivtube.text.thai")

# --- character classes ------------------------------------------------------------------

#: Thai combining marks U+0E31, U+0E34–U+0E3A, U+0E47–U+0E4E (all category Mn).
THAI_COMBINING: Final = frozenset(
    [chr(0x0E31)]
    + [chr(c) for c in range(0x0E34, 0x0E3B)]
    + [chr(c) for c in range(0x0E47, 0x0E4F)]
)
#: Following vowels (SARA A, SARA AA, SARA AM, LAKKHANGYAO) close a syllable: never cut before.
FOLLOWING_VOWELS: Final = frozenset(chr(c) for c in (0x0E30, 0x0E32, 0x0E33, 0x0E45))
#: Leading vowels start a syllable before its consonant: never cut right after one.
LEADING_VOWELS: Final = frozenset("เแโใไ")
#: A cut never goes right before these: marks, following vowels, ๆ, ฯ, ZWNJ/ZWJ, VS15/VS16, keycap.
NO_CUT_BEFORE: Final = (
    THAI_COMBINING
    | FOLLOWING_VOWELS
    | frozenset("ๆฯ")
    | frozenset(chr(c) for c in (0x200C, 0x200D, 0xFE0E, 0xFE0F, 0x20E3))
)
#: A cut never goes right after these (leading vowels, ZWJ).
NO_CUT_AFTER: Final = LEADING_VOWELS | frozenset([chr(0x200D)])

_THAI_LETTERS: Final = frozenset(
    [chr(c) for c in range(0x0E01, 0x0E2F)]  # consonants (incl. RU/LU)
    + [chr(c) for c in (0x0E30, 0x0E32, 0x0E33, 0x0E45)]  # following vowel letters
    + [chr(c) for c in range(0x0E40, 0x0E45)]  # leading vowels
)


def is_thai(ch: str) -> bool:
    """True for any character of the Thai block (U+0E00–U+0E7F)."""
    return "\U00000e00" <= ch <= "\U00000e7f"


def is_thai_letter(ch: str) -> bool:
    """True for Thai consonants and vowel letters (not marks, digits, ๆ or ฯ)."""
    return ch in _THAI_LETTERS


def is_combining(ch: str) -> bool:
    """True for Thai combining marks and any other Unicode mark (Mn/Mc/Me)."""
    return ch in THAI_COMBINING or unicodedata.category(ch)[0] == "M"


def is_safe_cut(text: str, k: int) -> bool:
    """Whether cutting ``text`` at index ``k`` keeps every grapheme and Thai syllable start whole.

    False before a combining mark, following vowel, ๆ, ฯ or emoji joiner/modifier, and after a
    leading vowel. The ends (``k <= 0`` or ``k >= len(text)``) are always safe.
    """
    if k <= 0 or k >= len(text):
        return True
    right = text[k]
    if right in NO_CUT_BEFORE or is_combining(right):
        return False
    if "\U0001f3fb" <= right <= "\U0001f3ff":  # emoji skin-tone modifiers
        return False
    return text[k - 1] not in NO_CUT_AFTER


# --- particles ----------------------------------------------------------------------------

#: A space after one of these ends a clause: a strong chunk boundary (TTS brief §4).
SENTENCE_FINAL_PARTICLES: Final[tuple[str, ...]] = (
    "นะคะ", "นะครับ", "นะคับ", "ค่ะ", "คะ", "ค่า", "ครับ", "คับ", "ค้าบ", "คร้าบ",
    "จ้า", "จ้ะ", "จ๊ะ", "นะ", "น้า", "เลย", "ล่ะ", "หรอ", "เหรอ", "มั้ย", "ไหม", "สิ",
    "ซิ", "ฮะ", "เนอะ", "แหละ", "ด้วย", "กัน", "เยย", "ชิมิ", "อะ", "อ่ะ", "วะ", "ว่ะ",
    "เว้ย", "แล้ว", "ป่ะ", "เนี่ย", "ฮ่า", "ๆ",
)  # fmt: skip

#: Particles that belong to the clause before them: a chunk never starts with one.
#: Entries that also start common words (คะแนน, อะไร, ค่าใช้จ่าย, กันยายน, สิบ …) count only
#: when they stand alone: followed by a non-letter or by another particle (``กันนะคะ``).
ATTACH_LEFT_PARTICLES: Final[tuple[str, ...]] = (
    "ค่ะ", "คะ", "ค่า", "ครับ", "คับ", "ค้าบ", "คร้าบ", "นะ", "น้า", "จ้า", "จ้ะ", "จ๊ะ",
    "ฮะ", "ล่ะ", "แหละ", "เนอะ", "สิ", "ซิ", "หรอ", "เหรอ", "มั้ย", "ไหม", "ป่ะ", "อะ",
    "อ่ะ", "กัน",
)  # fmt: skip
#: Always attach to the left, whatever follows (repetition marks, laughter, เลย, ด้วย).
ATTACH_ALWAYS: Final[tuple[str, ...]] = ("ๆ", "ฯ", "ฮ่า", "555", "เลย", "ด้วย")
AMBIGUOUS_PARTICLES: Final = frozenset({"คะ", "ค่า", "นะ", "สิ", "ซิ", "ฮะ", "อะ", "กัน", "ไหม"})

_ATTACH_SORTED: Final = tuple(sorted(ATTACH_LEFT_PARTICLES, key=len, reverse=True))
_FINAL_PARTICLES: Final = tuple(SENTENCE_FINAL_PARTICLES)


def ends_with_final_particle(text: str) -> bool:
    """True when ``text`` ends with a sentence-final particle (elongations such as จ้าาา count)."""
    tail = collapse_repeats(text.rstrip()[-12:], 1)
    return tail.endswith(_FINAL_PARTICLES)


def attach_left(rest: str, complete: bool) -> bool | None:
    """Whether ``rest`` starts with a particle that must stay with the text before it.

    ``complete`` means nothing more will follow ``rest``. Returns ``None`` when more text is
    needed to decide: ``rest`` may still grow into a particle, or a lone ambiguous particle
    has no right context yet.
    """
    if rest.startswith(ATTACH_ALWAYS):
        return True
    pos = count = 0
    last = ""
    while True:
        match = next((p for p in _ATTACH_SORTED if rest.startswith(p, pos)), None)
        if match is None:
            break
        pos += len(match)
        count += 1
        last = match
    if count >= 2 or (count == 1 and last not in AMBIGUOUS_PARTICLES):
        return True
    tail = rest[pos:]
    if not complete:
        growable = _ATTACH_SORTED + ATTACH_ALWAYS if count == 0 else _ATTACH_SORTED
        if any(len(p) > len(tail) and p.startswith(tail) for p in growable):
            return None
    if count == 0:
        return False
    # A lone ambiguous particle is a particle only when no Thai word continues it.
    return not tail or not (is_thai_letter(tail[0]) or is_combining(tail[0]))


# --- matching normalisation (§7 steps 1, 2, 4, 5, 6) ---------------------------------------

_ZERO_WIDTH: Final = dict.fromkeys(
    (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD, 0x200E, 0x200F, 0x180E,
     *range(0x202A, 0x202F), *range(0x2066, 0x206A)),
    None,
)  # fmt: skip
_THAI_DIGITS: Final = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_SARA_AM_DECOMPOSED: Final = chr(0x0E4D) + chr(0x0E32)


def strip_zero_width(text: str) -> str:
    """Remove zero-width and bidi control characters (U+200B/C/D, U+2060, U+FEFF, U+00AD …)."""
    return text.translate(_ZERO_WIDTH)


def nfkc_casefold(text: str) -> str:
    """NFKC + casefold, with SARA AM recomposed.

    NFKC decomposes ``ำ`` (U+0E33) into NIKHAHIT + SARA AA; this puts it back so Thai
    words keep their usual spelling and the function is idempotent.
    """
    return unicodedata.normalize("NFKC", text).casefold().replace(_SARA_AM_DECOMPOSED, "ำ")


def collapse_repeats(text: str, max_run: int = 2) -> str:
    """Shorten runs of one repeated character to ``max_run`` (``ควายยยย`` → ``ควายย``)."""
    if max_run < 1:
        raise ValueError("max_run must be >= 1")
    out: list[str] = []
    prev = ""
    run = 0
    for ch in text:
        run = run + 1 if ch == prev else 1
        prev = ch
        if run <= max_run:
            out.append(ch)
    return "".join(out)


def thai_digits_to_arabic(text: str) -> str:
    """Map Thai digits ๐–๙ to 0–9."""
    return text.translate(_THAI_DIGITS)


def estimate_tokens(text: str) -> int:
    """Rough LLM token count: Thai characters / 2 plus other non-space characters / 4."""
    thai = other = 0
    for ch in text:
        if is_thai(ch):
            thai += 1
        elif not ch.isspace():
            other += 1
    return math.ceil(thai / 2 + other / 4)


# --- question detection ---------------------------------------------------------------------

_Q_TRAILING: Final = " \t\r\n\"'”’)]}»」』~….!"
_Q_POLITE: Final = tuple(
    sorted(
        ("ครับ", "คับ", "ค่ะ", "ค่า", "จ้ะ", "จ้า", "นะ", "น้า", "อะ", "อ่ะ", "วะ", "ว่ะ",
         "เนี่ย", "เว้ย", "ฮะ"),
        key=len,
        reverse=True,
    )
)  # fmt: skip
_Q_WORDS: Final = (
    "หรือเปล่า", "รึเปล่า", "หรือยัง", "รึยัง", "หรือไม่", "ใช่ไหม", "ใช่มั้ย", "เมื่อไหร่",
    "เมื่อไร", "เท่าไหร่", "เท่าไร", "อย่างไร", "ยังไง", "ทำไม", "อะไร", "ที่ไหน", "ไหน",
    "ใคร", "มั้ย", "มั๊ย", "ไม๊", "ไหม", "เหรอ", "หรอ", "ป่าว", "กี่", "ชิมิ",
)  # fmt: skip
# Not after ไม่มี/ไม่ว่า ("nobody", "no matter"), not before a tone mark (ไหม้ = burn),
# not before ก็ (indefinite: อะไรก็ได้), and หรอ not in หรอก.
_QUESTION_RE: Final = re.compile(
    r"(?<!ไม่มี)(?<!ไม่ว่า)(?:"
    + "|".join("หรอ(?!ก)" if w == "หรอ" else w for w in _Q_WORDS)
    + r")(?![\U00000e47-\U00000e4b])(?!\s*ก็)"
)
_Q_END_ONLY: Final = ("บ้าง", "ปะ", "ป่ะ", "ไง", "ยัง")
_EN_QUESTION: Final = re.compile(
    r"^(?:what|why|how|who|whom|whose|where|when|which|do|does|did|is|are|was|were|can|could|"
    r"would|will|should|shall|may|might|have|has|am|isn't|aren't|don't|doesn't|didn't|won't|"
    r"can't|wanna)\b",
    re.IGNORECASE,
)


def is_question(text: str) -> bool:
    """Heuristic: does ``text`` ask something? (``?``, Thai question words or particles, English
    interrogatives). Indefinite uses such as ``อะไรก็ได้`` or ``ไม่มีใคร`` do not count."""
    t = strip_zero_width(text).strip()
    if not t:
        return False
    if "?" in t or "？" in t:
        return True
    core = t.rstrip(_Q_TRAILING)
    if core.endswith("คะ") and not core.endswith("นะคะ"):
        return True  # คะ (high tone) is the polite question particle; ค่ะ is the statement one
    changed = True
    while changed:
        changed = False
        for p in _Q_POLITE:
            if core.endswith(p) and len(core) > len(p):
                core = core[: -len(p)].rstrip(_Q_TRAILING)
                changed = True
                break
    if core.endswith(_Q_END_ONLY) or _QUESTION_RE.search(t):
        return True
    return bool(_EN_QUESTION.match(t.lstrip("\"'([“‘")))


# --- name matching ----------------------------------------------------------------------------


def _match_form(text: str) -> str:
    return nfkc_casefold(strip_zero_width(text))


class NameMatcher:
    """Is the character addressed? NFKC + zero-width strip + casefold, then substring matching.

    Thai has no spaces between words, so Thai aliases match anywhere. Latin-only aliases
    (``pailin``) must not touch other Latin letters or digits (``ai`` does not match ``said``).
    Spaced-out spellings (``ไพ ลิน``, ``p a i l i n``) are matched on a despaced copy.
    """

    def __init__(self, aliases: Sequence[str]) -> None:
        thai: set[str] = set()
        latin: set[str] = set()
        for alias in aliases:
            form = _match_form(alias).strip()
            if not form:
                continue
            despaced = "".join(form.split())
            (latin if despaced.isascii() else thai).update({form, despaced})
        self._thai = tuple(sorted(thai))
        self._latin: re.Pattern[str] | None = None
        if latin:
            alt = "|".join(re.escape(a) for a in sorted(latin, key=len, reverse=True))
            self._latin = re.compile(rf"(?<![a-z0-9])(?:{alt})(?![a-z0-9])")

    def __call__(self, text: str) -> bool:
        form = _match_form(text)
        for candidate in (form, "".join(form.split())):
            if any(a in candidate for a in self._thai):
                return True
            if self._latin is not None and self._latin.search(candidate):
                return True
        return False


# --- the newmm tokenizer ----------------------------------------------------------------------


class ThaiWordTokenizer:
    """pythainlp word tokenizer (``keep_whitespace=True``, so tokens join back to the input).

    Loaded lazily: ``warm()`` imports pythainlp and loads the dictionary (≈ 0.35 s, blocking).
    Calling the tokenizer before ``warm()`` warms it on the spot and logs a warning when that
    happens on an event-loop thread. If pythainlp cannot be imported the tokenizer returns the
    whole text as one token, so callers simply find no word boundaries.
    """

    def __init__(self, engine: str = "newmm") -> None:
        self.engine = engine
        self.warm_s: float | None = None
        self._fn: Callable[..., list[str]] | None = None
        self._failed = False
        self._lock = threading.Lock()

    @property
    def warmed(self) -> bool:
        return self._fn is not None or self._failed

    @property
    def available(self) -> bool:
        """False after pythainlp failed to import."""
        return not self._failed

    def warm(self) -> float:
        """Load pythainlp and the dictionary. Returns the seconds spent (0.0 when already warm)."""
        if self.warmed:
            return 0.0
        with self._lock:
            if self.warmed:
                return 0.0
            t0 = time.perf_counter()
            try:
                from pythainlp.tokenize import word_tokenize

                word_tokenize("ไพลินทดสอบภาษาไทย", engine=self.engine, keep_whitespace=True)
            except Exception:  # ImportError, or a broken dictionary install
                log.warning(
                    "pythainlp tokenizer unavailable; Thai word cuts disabled", exc_info=True
                )
                self._failed = True
            else:
                self._fn = word_tokenize
            self.warm_s = time.perf_counter() - t0
            return self.warm_s

    def __call__(self, text: str) -> list[str]:
        if not text:
            return []
        if not self.warmed:
            if _on_event_loop():
                log.warning(
                    "pythainlp warmed lazily on the event loop (~0.35 s stall); "
                    "call warm_up_pythainlp() in a thread at startup"
                )
            self.warm()
        fn = self._fn
        if fn is None:
            return [text]
        return list(fn(text, engine=self.engine, keep_whitespace=True))


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


#: The shared newmm tokenizer (thread-safe to warm; read-only afterwards).
newmm: Final = ThaiWordTokenizer("newmm")


def warm_up_pythainlp() -> float:
    """Warm everything the text package uses from pythainlp (tokenizer + number words).

    Blocking (≈ 0.4 s the first time): run it in a thread at startup. Returns seconds spent.
    """
    t0 = time.perf_counter()
    newmm.warm()
    try:
        from pythainlp.util import num_to_thaiword

        num_to_thaiword(21)
    except Exception:  # pragma: no cover - pythainlp is a core dependency
        log.warning("pythainlp.util unavailable; local number reading degrades", exc_info=True)
    return time.perf_counter() - t0
