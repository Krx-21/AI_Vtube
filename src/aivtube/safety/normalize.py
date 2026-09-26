"""Matching forms for the tier-0 filter (ARCHITECTURE.md §7, "Normalisation").

These forms are used for matching only; callers always keep the original text for display.

``norm``:
    1. NFKC (with SARA AM recomposed), 2. zero-width/bidi characters stripped,
    3. ``pythainlp.util.normalize`` (vowel/tone-mark order, duplicate marks, stray spaces),
    4. casefold, 5. runs of one character longer than 2 shortened to 2,
    6. Thai digits mapped to Arabic. Whitespace runs become one space.
``despaced``:
    ``.-_*`` between two letters removed (``ค.ว.ย``, ``fu-ck``) and runs of single letters
    separated by spaces joined (``f u c k``, ``ค ว ย``). Catches letter-by-letter spelling (T5).
``leet``:
    the Latin leet map (0→o, 1→i, 3→e, 4→a, @→a, $→s) over the despaced form.

pythainlp is imported lazily (the first call loads it, ~70 ms); call :func:`warm_up` in a
worker thread at startup. Without pythainlp step 3 is skipped and tokens fall back to the
whole text.
"""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from aivtube.text.thai import (
    collapse_repeats,
    is_combining,
    newmm,
    nfkc_casefold,
    strip_zero_width,
    thai_digits_to_arabic,
)

__all__ = [
    "LEET_MAP",
    "MatchForms",
    "clean",
    "despace",
    "has_thai",
    "leet",
    "match_forms",
    "norm",
    "thai_normalize",
    "warm_up",
]

log = logging.getLogger("aivtube.safety.normalize")

#: The §7 Latin leet map.
LEET_MAP: Final = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "@": "a", "$": "s"})
_LEET_CHARS: Final = frozenset("0134@$")
_SARA_AM_DECOMPOSED: Final = chr(0x0E4D) + chr(0x0E32)
_THAI_RE: Final = re.compile(r"[\u0e01-\u0e5b]")
_WS_RE: Final = re.compile(r"\s+")
# A "word character" for despacing: letters and digits of any script plus Thai combining marks.
_W: Final = r"(?:[^\W_]|[\u0e31\u0e34-\u0e3a\u0e47-\u0e4e])"
_INNER_SEP_RE: Final = re.compile(rf"(?<={_W})[.\-_*]+(?={_W})")

_normalize_fn: Callable[[str], str] | None = None
_normalize_failed = False
_normalize_lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class MatchForms:
    """Every form the tier-0 filter matches against (``tokens`` are newmm tokens of ``norm``)."""

    norm: str
    despaced: str
    leet: str
    tokens: tuple[str, ...]


def _load_normalize() -> Callable[[str], str] | None:
    global _normalize_fn, _normalize_failed
    if _normalize_fn is not None or _normalize_failed:
        return _normalize_fn
    with _normalize_lock:
        if _normalize_fn is None and not _normalize_failed:
            try:
                from pythainlp.util import normalize as fn
            except Exception:  # ImportError or a broken install: degrade, never crash the gate
                log.warning("pythainlp.util.normalize unavailable; Thai matching degrades")
                _normalize_failed = True
            else:
                _normalize_fn = fn
    return _normalize_fn


def warm_up() -> float:
    """Load pythainlp (normaliser and newmm dictionary). Blocking: run it in a thread."""
    _load_normalize()
    return newmm.warm()


def has_thai(text: str) -> bool:
    return _THAI_RE.search(text) is not None


def thai_normalize(text: str) -> str:
    """``pythainlp.util.normalize`` (identity when pythainlp is missing)."""
    fn = _load_normalize()
    return fn(text) if fn is not None else text


def clean(text: str) -> str:
    """NFKC + zero-width strip, keeping case: the display-safe base for masking and stripping."""
    return unicodedata.normalize("NFKC", strip_zero_width(text)).replace(_SARA_AM_DECOMPOSED, "ำ")


def norm(text: str) -> str:
    """The normalised matching form (§7 steps 1–6)."""
    s = nfkc_casefold(strip_zero_width(text))
    s = _WS_RE.sub(" ", s).strip()
    if s and has_thai(s):
        s = thai_normalize(s)
    return thai_digits_to_arabic(collapse_repeats(s, 2))


def _is_short(token: str) -> bool:
    """At most one letter/digit (Thai combining marks and punctuation do not count)."""
    count = 0
    for ch in token:
        if ch.isalnum() and not is_combining(ch):
            count += 1
            if count > 1:
                return False
    return True


def despace(normed: str) -> str:
    """Join letter-by-letter spellings in an already normalised string."""
    s = _INNER_SEP_RE.sub("", normed)
    if " " not in s:
        return s
    parts = s.split(" ")
    out: list[str] = []
    run: list[str] = []
    for part in parts:
        if part and _is_short(part):
            run.append(part)
            continue
        if run:
            out.append("".join(run) if len(run) >= 2 else run[0])
            run = []
        out.append(part)
    if run:
        out.append("".join(run) if len(run) >= 2 else run[0])
    return " ".join(out)


def leet(text: str) -> str:
    """Apply the Latin leet map (returns ``text`` itself when nothing maps)."""
    if _LEET_CHARS.isdisjoint(text):
        return text
    return text.translate(LEET_MAP)


def match_forms(text: str) -> MatchForms:
    """All matching forms of ``text``. Tokenising costs ~0.1 ms per 100 Thai characters."""
    n = norm(text)
    d = despace(n)
    tokens = tuple(t for t in newmm(n) if not t.isspace())
    return MatchForms(norm=n, despaced=d, leet=leet(d), tokens=tokens)
