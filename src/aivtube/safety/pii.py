"""PII and link detection for the tier-0 filter (ARCHITECTURE.md §7, "Regex").

Built-in patterns: URLs (scheme, ``www.`` and bare domains on common TLDs), e-mail addresses,
``@handles``, Thai mobile numbers (``0[689]`` + 8 digits, or ``+66``), and 13-digit Thai
national IDs (checksum-validated to avoid masking random long numbers). Digits may be Thai
(``๐``–``๙``) or separated by spaces, dots or dashes. Patterns run on :func:`~.normalize.clean`
text (NFKC, zero-width stripped, case kept), so the spans can be masked for display.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from aivtube.safety.normalize import clean

__all__ = [
    "BUILTIN_PATTERNS",
    "PLACEHOLDERS",
    "PiiSpan",
    "find_pii",
    "mask_pii",
    "mask_spans",
    "thai_id_valid",
]

_TLDS: Final = (
    "com|net|org|info|biz|io|gg|co|me|ly|tv|app|dev|link|bet|vip|club|site|online|shop|store"
    "|xyz|live|win|top|fun|click|work|life|one|asia|cc|to|th|ru|cn|uk|us|tk|ml|ga|cf|gq|ws|in"
)
_URL_CHARS: Final = r"[^\s<>\"'`]"

#: ``(kind, pattern)`` in priority order: an earlier kind wins when spans overlap. Every
#: pattern is linear in the text length (bounded repeats, lookbehinds at run starts).
BUILTIN_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "email",
        re.compile(r"(?<![\w.+-])[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63}){1,8}", re.IGNORECASE),
    ),
    ("url", re.compile(rf"(?:https?://|www\.){_URL_CHARS}+", re.IGNORECASE)),
    (
        "url",
        re.compile(
            rf"(?<![a-z0-9@.-])(?:[a-z0-9](?:[a-z0-9-]{{0,61}}[a-z0-9])?\.)+(?:{_TLDS})"
            rf"(?![a-z0-9-])(?:[/?#]{_URL_CHARS}*)?",
            re.IGNORECASE,
        ),
    ),
    ("national_id", re.compile(r"(?<![\d.])\d(?:[ \-]?\d){12}(?![\d])")),
    ("phone", re.compile(r"(?<![\d.])(?:\+66[ \-]?|[0๐])[689๖๘๙](?:[ .\-]?\d){8}(?!\d)")),
    ("handle", re.compile(r"(?<![\w@.])@[a-z0-9_][a-z0-9_.]{1,29}", re.IGNORECASE)),
)

#: Display placeholder per kind; ``url`` and ``link`` use the configured ``mask_text``.
PLACEHOLDERS: Final[Mapping[str, str]] = {
    "email": "[อีเมล]",
    "phone": "[เบอร์โทร]",
    "national_id": "[เลขบัตร]",
    "handle": "[บัญชี]",
}

_DIGITS: Final = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


@dataclass(frozen=True, slots=True)
class PiiSpan:
    start: int
    end: int
    kind: str  # email | url | national_id | phone | handle | link (list rules)
    rule: str


def thai_id_valid(raw: str) -> bool:
    """Checksum of a 13-digit Thai national ID (separators and Thai digits allowed)."""
    digits = [int(c) for c in raw.translate(_DIGITS) if c.isdigit()]
    if len(digits) != 13:
        return False
    total = sum(d * (13 - i) for i, d in enumerate(digits[:12]))
    return (11 - total % 11) % 10 == digits[12]


def _contained(start: int, end: int, ranges: Sequence[tuple[int, int]]) -> bool:
    return any(a <= start and end <= b for a, b in ranges)


def find_pii(
    text: str,
    extra: Iterable[tuple[str, re.Pattern[str]]] = (),
    *,
    allow_ranges: Sequence[tuple[int, int]] = (),
    handle_aliases: Sequence[str] = (),
) -> list[PiiSpan]:
    """Non-overlapping PII spans in ``text`` (already :func:`clean`), ordered by position.

    ``extra`` holds list rules (``kind`` ``link``) and runs after the built-ins. Spans inside
    ``allow_ranges`` are skipped, as are ``@handles`` containing one of ``handle_aliases``
    (the character's own account).
    """
    found: list[PiiSpan] = []
    for kind, pattern in (*BUILTIN_PATTERNS, *extra):
        for m in pattern.finditer(text):
            start, end = m.span()
            if start == end:
                continue
            value = m.group(0)
            if kind == "national_id" and not thai_id_valid(value):
                continue
            if kind == "handle" and any(a in value.casefold() for a in handle_aliases):
                continue
            if allow_ranges and _contained(start, end, allow_ranges):
                continue
            rule = kind if kind != "link" else pattern.pattern
            found.append(PiiSpan(start, end, kind, rule))
    if len(found) < 2:
        return found
    # Earlier patterns win: keep a span unless it overlaps one already kept (kept spans are
    # disjoint, so a bisect on their starts finds the only two candidates for overlap).
    starts: list[int] = []
    kept: list[PiiSpan] = []
    for span in found:
        i = bisect_right(starts, span.start)
        if i > 0 and kept[i - 1].end > span.start:
            continue
        if i < len(kept) and kept[i].start < span.end:
            continue
        starts.insert(i, span.start)
        kept.insert(i, span)
    return kept


def mask_spans(text: str, spans: Sequence[PiiSpan], *, mask_text: str = "[ลิงก์]") -> str:
    """Replace each span (non-overlapping, any order) with its placeholder."""
    out: list[str] = []
    pos = 0
    for span in sorted(spans, key=lambda s: s.start):
        out.append(text[pos : span.start])
        out.append(PLACEHOLDERS.get(span.kind, mask_text))
        pos = span.end
    out.append(text[pos:])
    return "".join(out)


def mask_pii(
    text: str,
    *,
    mask_text: str = "[ลิงก์]",
    extra: Iterable[tuple[str, re.Pattern[str]]] = (),
) -> str:
    """Clean ``text`` and mask every PII span (for audit rows and logs)."""
    base = clean(text)
    spans = find_pii(base, extra)
    return mask_spans(base, spans, mask_text=mask_text) if spans else base
