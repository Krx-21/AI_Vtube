"""Incremental Thai speech chunker: streamed LLM text → TTS-sized chunks (§4.6; TTS brief §4).

Feed LLM deltas with ``feed()``; it returns zero or more chunks. Call ``flush()`` at the end
of the stream. While the LLM stalls, ``poll(now)`` (or ``stall_flush()``) emits what is
already safe to speak, cut at the last boundary, never mid-word.

**Sizes.** The first chunk is short for latency (``first_min_chars``–``first_max_chars``,
8–60); later chunks are longer (``min_chars``–``max_chars``, 40–160; 60–160 on Azure F0) so
there are fewer TTS requests and better prosody. ``ChunkerConfig.from_constraints`` builds the
config from the voice worker's ``TTSConstraints`` at the start of each utterance.

**Where it cuts, best first:** punctuation or a newline (``!?…。`` / ``.`` / ``\\n``), a space
after a sentence-final particle (ค่ะ ครับ นะ …), any other Thai clause space, then, when a run
has no such boundary, a newmm word boundary (at ``word_split_chars`` or at the hard cap), then
a syllable-safe cut at ``max_chars``.

**Never:**

- before a Thai combining mark (U+0E31, U+0E34–U+0E3A, U+0E47–U+0E4E), a following vowel, ๆ or
  ฯ, nor right after a leading vowel (เ แ โ ใ ไ);
- inside or next to a number, time or amount (``100 บาท``, ``เวลา 02:30``, ``3.5``,
  ``$29.99``, ``1,250.50``), or inside a Latin word or between two Latin words
  (``Minecraft Java Edition``);
- before a particle that belongs to the previous clause (``… กันนะคะ``, ``ๆ``, ``555``).

**Chunks are exact slices of the input.** A chunk keeps the whitespace that followed its cut,
so ``"".join(chunks)`` equals the stream with only leading whitespace and the final trailing
whitespace removed (and the excess of a whitespace run longer than ``max_chars``, which no
chunk can hold); captions and heard text keep their spacing. Strip or normalise a chunk
before synthesis (``normalize_cloud`` / ``normalize_local`` do). Each chunk, whitespace
included, is at most ``max_chars`` long (``first_max_chars`` for the first).

**Deterministic.** Every decision reads a fixed window of text (``LOOKAHEAD`` characters of
right context), so the output does not depend on how the stream was split into deltas (the
stall flush aside, which depends on timing by design).

Pure Python and allocation-light: well under 1 ms per delta. Word boundaries come from the
shared pythainlp newmm tokenizer; warm it at startup (``warm_up_pythainlp`` in a thread), or it
warms on first use (≈ 0.35 s, with a warning when that happens on an event loop).
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

from aivtube.text import thai as th
from aivtube.text.thai import warm_up_pythainlp

if TYPE_CHECKING:
    from aivtube.contracts.infra import Clock
    from aivtube.contracts.speech import TTSConstraints

__all__ = [
    "LOOKAHEAD",
    "ChunkerConfig",
    "ChunkerSettings",
    "ThaiSpeechChunker",
    "WordTokenizer",
    "no_word_split",
    "warm_up_pythainlp",
]

log = logging.getLogger("aivtube.text.chunker")

#: A tokenizer whose tokens join back to its input (pythainlp ``keep_whitespace=True`` style).
WordTokenizer = Callable[[str], Sequence[str]]

#: Characters of right context every boundary decision may read.
LOOKAHEAD: Final = 12

_WAIT, _NONE, _SOFT, _STRONG, _HARD = -1, 0, 1, 2, 3

_HARD_PUNCT: Final = frozenset("!?！？。…")
_SOFT_PUNCT: Final = frozenset(",;:，、；：")
_CLOSERS: Final = frozenset(")]}»”’」』\"'")
_STRICT_OPENERS: Final = frozenset("([{«“‘「『")
_STRICT_CLOSERS: Final = frozenset(")]}»”’」』")
_CURRENCY: Final = frozenset("$฿€£¥₩")
_LAUGH_TOKEN: Final = re.compile(r"5{3,}\+*")
_THAI_ABBREV: Final = re.compile(r"(?:^|[\s(])(?:[ก-ฮ]{1,2}\.)+$")
_LATIN_ABBREV: Final = re.compile(r"(?:[A-Za-z]\.)+")
_TITLES: Final = frozenset(
    {"mr.", "mrs.", "ms.", "dr.", "st.", "prof.", "sr.", "jr.", "vs.", "no.",
     "ดร.", "นพ.", "พญ.", "ผศ.", "รศ.", "ศ.", "อ.", "น.ส.", "ด.ช.", "ด.ญ."}
)  # fmt: skip


def no_word_split(text: str) -> list[str]:
    """A ``WordTokenizer`` that finds no word boundaries (disables newmm cuts)."""
    return [text] if text else []


# --- config -------------------------------------------------------------------------------


class ChunkerSettings(Protocol):
    """The ``[tts.chunker]`` config section (``AppConfig.tts.chunker``)."""

    @property
    def first_min_chars(self) -> int: ...
    @property
    def first_max_chars(self) -> int: ...
    @property
    def min_chars(self) -> int: ...
    @property
    def max_chars(self) -> int: ...
    @property
    def strong_min_chars(self) -> int: ...
    @property
    def stall_flush_ms(self) -> int: ...


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    """Chunk size limits (characters) and the stall-flush delay.

    - ``first_min_chars`` / ``first_max_chars``: the first chunk of a stream.
    - ``min_chars``: a plain clause space may end a later chunk from this length.
    - ``strong_min_chars``: punctuation or a particle boundary may end a later chunk from here.
    - ``max_chars``: hard cap for every chunk.
    - ``stall_flush_s``: ``poll`` flushes after this long without a delta.
    """

    first_min_chars: int = 8
    first_max_chars: int = 60
    min_chars: int = 40
    max_chars: int = 160
    strong_min_chars: int = 20
    stall_flush_s: float = 0.5

    def __post_init__(self) -> None:
        sizes = (
            self.first_min_chars,
            self.first_max_chars,
            self.min_chars,
            self.max_chars,
            self.strong_min_chars,
        )
        if min(sizes) < 1:
            raise ValueError(f"chunker sizes must be >= 1: {self}")
        if not self.first_min_chars <= self.first_max_chars <= self.max_chars:
            raise ValueError(f"need first_min_chars <= first_max_chars <= max_chars: {self}")
        if self.min_chars > self.max_chars:
            raise ValueError(f"need min_chars <= max_chars: {self}")
        if not self.stall_flush_s > 0:
            raise ValueError(f"stall_flush_s must be > 0: {self}")

    @property
    def word_split_chars(self) -> int:
        """Where an unspaced later run is cut at a word boundary:
        ``min(max_chars, max(2 × min_chars, first_max_chars))`` (80 by default)."""
        return min(self.max_chars, max(2 * self.min_chars, self.first_max_chars))

    @classmethod
    def from_constraints(
        cls, constraints: TTSConstraints, *, base: ChunkerConfig | None = None
    ) -> ChunkerConfig:
        """Sizes from the voice worker's ``TTSConstraints``; ``first_max_chars``,
        ``strong_min_chars`` and the stall delay from ``base`` (clamped to fit).

        When the backend raises ``min_chars`` above ``base`` (Azure F0: 60 for its request
        quota), ``strong_min_chars`` rises by the same amount (20 → 40), so punctuation does
        not keep producing short chunks that would spend the quota anyway.
        """
        b = base or cls()
        mx = max(1, constraints.max_chars)
        first_min = min(max(1, constraints.first_min_chars), mx)
        mn = min(max(1, constraints.min_chars), mx)
        strong = b.strong_min_chars + max(0, mn - b.min_chars)
        return cls(
            first_min_chars=first_min,
            first_max_chars=min(max(b.first_max_chars, first_min), mx),
            min_chars=mn,
            max_chars=mx,
            strong_min_chars=max(1, min(strong, mn)),
            stall_flush_s=b.stall_flush_s,
        )

    @classmethod
    def from_settings(cls, settings: ChunkerSettings) -> ChunkerConfig:
        """From the ``[tts.chunker]`` config section."""
        return cls(
            first_min_chars=settings.first_min_chars,
            first_max_chars=settings.first_max_chars,
            min_chars=settings.min_chars,
            max_chars=settings.max_chars,
            strong_min_chars=settings.strong_min_chars,
            stall_flush_s=settings.stall_flush_ms / 1000.0,
        )


@dataclass(frozen=True, slots=True)
class _Limits:
    min_n: int  # a plain space qualifies from this content length
    strong_n: int  # punctuation / particle boundaries qualify from here
    word_at: int | None  # voluntary word-boundary cut point (later chunks only)
    max_n: int


@dataclass(frozen=True, slots=True)
class _Site:
    kind: int
    cut: int = 0  # the chunk would end here (exclusive), trailing whitespace included
    n: int = 0  # content characters before the cut
    latin: bool = False  # a space inside a Latin phrase: last-resort cut before a mid-word one


_WAIT_SITE: Final = _Site(_WAIT)
_PENDING: Final = -1  # checkpoint needs more text


def _is_ascii_word(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def _is_ascii_print(ch: str) -> bool:
    return "!" <= ch <= "~"


def _is_symbol(ch: str) -> bool:
    return ord(ch) >= 0x1F000 or unicodedata.category(ch) == "So"


def _dot_kind(text: str) -> str:
    """What the ``.`` ending ``text`` is: ``stop``, ``title`` (Mr./ดร.), ``number`` (``3.``)
    or ``abbr`` (ค.ศ., น., e.g.)."""
    parts = text.split()
    tok = parts[-1] if parts else text
    if tok.casefold() in _TITLES:
        return "title"
    if tok.endswith("..."):
        return "stop"
    if tok[:-1].isdigit():
        return "number"
    if _LATIN_ABBREV.fullmatch(tok) or _THAI_ABBREV.search(text):
        return "abbr"
    return "stop"


def _cut_ok(text: str, k: int) -> bool:
    """A cut inside a run of text (word or forced cut): whole graphemes, not inside a Latin
    token or a number, not before punctuation, a closer, an emoji or an attach-left particle."""
    if k <= 0 or k >= len(text):
        return False
    a, c = text[k - 1], text[k]
    if a.isspace() or c.isspace() or not th.is_safe_cut(text, k):
        return False
    if _is_ascii_print(a) and _is_ascii_print(c):
        return False
    if a.isdigit() or c.isdigit() or a in _CURRENCY or c in _CURRENCY or c == "%":
        return False
    if a in _STRICT_OPENERS or c in _STRICT_CLOSERS or c in _CLOSERS:
        return False
    if c in _HARD_PUNCT or c in _SOFT_PUNCT or c == "." or _is_symbol(c):
        return False
    return not th.attach_left(text[k:], True)


# --- the chunker --------------------------------------------------------------------------


class ThaiSpeechChunker:
    """Incremental Thai(+English) speech chunker. See the module docstring for the rules.

    ``word_tokenize`` defaults to the shared pythainlp newmm tokenizer; pass ``no_word_split``
    to disable word-boundary cuts. ``clock`` (a ``Clock``) times the stall flush; without one,
    ``time.perf_counter`` is used. One instance handles one stream at a time; ``flush()`` and
    ``reset()`` start a new stream (the next chunk is a "first" chunk again).
    """

    def __init__(
        self,
        cfg: ChunkerConfig | None = None,
        *,
        word_tokenize: WordTokenizer | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.cfg = cfg or ChunkerConfig()
        self._tokenize: WordTokenizer = word_tokenize if word_tokenize is not None else th.newmm
        self._clock = clock
        self._tokenize_failed = False
        self._buf = ""
        self._emitted = 0
        self._at_start = True
        self._last_feed: float | None = None
        self._stalled = False
        self._sites: dict[int, _Site] = {}
        self._decisions: dict[tuple[str, int], int | None] = {}

    # --- public API -----------------------------------------------------------------------
    @property
    def pending(self) -> str:
        """Text received but not emitted yet."""
        return self._buf

    @property
    def emitted(self) -> int:
        """Chunks emitted in the current stream."""
        return self._emitted

    def feed(self, delta: str) -> list[str]:
        """Add streamed text; return the chunks that are now final."""
        if not delta:
            return []
        self._last_feed = self._now()
        self._stalled = False
        if self._at_start:
            delta = delta.lstrip()
            if not delta:
                return []
            self._at_start = False
        self._buf += delta
        out: list[str] = []
        while (cut := self._find(final=False)) is not None:
            out.extend(self._emit(cut, final=False))
        return out

    def flush(self) -> list[str]:
        """End of stream: emit everything left (still respecting ``max_chars``), then reset."""
        out: list[str] = []
        while self._buf.strip():
            cut = self._find(final=True)
            out.extend(self._emit(len(self._buf) if cut is None else cut, final=True))
        self.reset()
        return out

    def reset(self) -> None:
        """Drop buffered text and start a new stream (after a barge-in or a new utterance)."""
        self._buf = ""
        self._emitted = 0
        self._at_start = True
        self._last_feed = None
        self._stalled = False
        self._clear_caches()

    def stall_deadline(self) -> float | None:
        """When ``poll`` would stall-flush if no delta arrives (``None``: nothing to flush)."""
        if self._stalled or self._last_feed is None or not self._buf.strip():
            return None
        return self._last_feed + self.cfg.stall_flush_s

    def poll(self, now: float | None = None) -> list[str]:
        """Stall flush once ``stall_flush_s`` has passed since the last delta (500 ms default).

        ``now`` comes from the same clock as the chunker's (``clock.now()``). Fires at most once
        per stall; the next ``feed`` re-arms it.
        """
        deadline = self.stall_deadline()
        if deadline is None:
            return []
        if (self._now() if now is None else now) < deadline:
            return []
        self._stalled = True
        return self.stall_flush()

    def stall_flush(self) -> list[str]:
        """Emit the buffer up to its last safe boundary (normally the last space), keeping the
        partial word after it. Emits nothing if that piece would be shorter than
        ``first_min_chars`` or there is no safe boundary. Never cuts mid-word."""
        b = self._buf
        if not b.strip():
            return []
        mx = self._limits().max_n
        best: int | None = None
        for i in self._site_positions(b, mx):
            site = self._classify(b, i, mx, final=False)
            if site.kind == _WAIT:
                site = self._end_site(b, i, mx)
            if site.kind >= _SOFT and site.n >= self.cfg.first_min_chars:
                best = site.cut
        return [] if best is None else self._emit(best, final=False)

    # --- scanning -------------------------------------------------------------------------
    def _now(self) -> float:
        return self._clock.now() if self._clock is not None else time.perf_counter()

    def _limits(self) -> _Limits:
        c = self.cfg
        if self._emitted == 0:
            return _Limits(c.first_min_chars, c.first_min_chars, None, c.first_max_chars)
        word_at = c.word_split_chars
        return _Limits(
            c.min_chars, c.strong_min_chars, word_at if word_at < c.max_chars else None, c.max_chars
        )

    @staticmethod
    def _site_positions(b: str, limit: int) -> Iterator[int]:
        """Candidate boundaries whose content fits in ``limit`` characters: the start of each
        whitespace run (up to ``limit``), and hard punctuation / ``.`` / closers not followed
        by whitespace (below ``limit``)."""
        n = len(b)
        for i in range(min(n, limit + 1)):
            ch = b[i]
            if ch.isspace():
                if i > 0 and not b[i - 1].isspace():
                    yield i
            elif (
                i < limit
                and (ch in _HARD_PUNCT or ch == "." or ch in _CLOSERS)
                and (i + 1 >= n or not b[i + 1].isspace())
            ):
                yield i

    def _find(self, final: bool) -> int | None:
        """The next cut, or ``None`` while more text is needed."""
        b = self._buf
        if not b.strip():
            return None
        lim = self._limits()
        mx = lim.max_n
        word_done = lim.word_at is None
        fallback: dict[int, int] = {}  # kind -> last below-threshold cut (n >= strong_n)
        weak: int | None = None
        for i in self._site_positions(b, mx):
            if not word_done and lim.word_at is not None and i >= lim.word_at:
                word_done = True
                cut = self._word_checkpoint(b, lim, final)
                if cut is not None:
                    return None if cut == _PENDING else cut
            site = self._classify(b, i, mx, final)
            if site.kind == _WAIT:
                return None
            if site.kind == _NONE:
                if site.latin:
                    weak = site.cut
                continue
            if site.n >= (lim.min_n if site.kind == _SOFT else lim.strong_n):
                return site.cut
            if site.n >= lim.strong_n:
                fallback[site.kind] = site.cut
        content = len(b.rstrip()) if final else len(b)
        if not word_done and lim.word_at is not None and content > lim.word_at:
            cut = self._word_checkpoint(b, lim, final)
            if cut is not None:
                return None if cut == _PENDING else cut
        if final and len(b.rstrip()) <= mx:
            return len(b)
        if not final and len(b) < mx + LOOKAHEAD:
            return None
        return self._forced(b, lim, final, fallback, weak)

    def _classify(self, b: str, i: int, mx: int, final: bool) -> _Site:
        """Classify the boundary site at ``i`` using at most ``LOOKAHEAD`` characters after it."""
        cached = self._sites.get(i)
        if cached is not None:
            return cached
        end = i + 1 + LOOKAHEAD
        complete = final or len(b) >= end
        v = b[:end]
        if v[i].isspace():
            j = i + 1
            while j < len(v) and v[j].isspace():
                j += 1
            n = len(v[:i].strip())
            if n == 0:
                site = _Site(_NONE)
            elif j >= len(v):
                if not complete:
                    return _WAIT_SITE
                run = v[i:j]
                kind = _HARD if ("\n" in run or "\r" in run) else self._left_kind(v[:i])
                site = _Site(kind, min(j, mx), n)
            else:
                kind, latin = self._space_kind(v, i, j, complete)
                if kind == _WAIT:
                    return _WAIT_SITE
                site = _Site(kind, min(j, mx), n, latin)
        else:
            n = len(v[: i + 1].strip())
            if i + 1 >= len(v):
                if not complete:
                    return _WAIT_SITE
                site = _Site(self._left_kind(v[: i + 1]), i + 1, n)
            else:
                site = _Site(self._attached_kind(v, i), i + 1, n)
        if len(b) >= end:  # decided on a full window: the same for every longer buffer
            self._sites[i] = site
        return site

    @staticmethod
    def _left_kind(left: str) -> int:
        """A boundary right after ``left`` with no right context (end of text, or a stall)."""
        t = left.rstrip()
        if not t:
            return _NONE
        last = t[-1]
        k = len(t) - 1
        while k > 0 and t[k] in _CLOSERS:
            k -= 1
        if t[k] in _HARD_PUNCT:
            return _HARD
        if t[k] == ".":
            return _HARD if _dot_kind(t[: k + 1]) == "stop" else _NONE
        if last in _SOFT_PUNCT:
            return _SOFT
        parts = t.split()
        if _LAUGH_TOKEN.fullmatch(parts[-1]) or _is_symbol(last) or th.ends_with_final_particle(t):
            return _STRONG
        if last in th.NO_CUT_AFTER:
            return _NONE
        if th.is_thai_letter(last) or (th.is_combining(last) and th.is_thai(last)):
            return _SOFT
        return _NONE

    @staticmethod
    def _space_kind(v: str, i: int, j: int, complete: bool) -> tuple[int, bool]:
        """Classify the whitespace run ``v[i:j]``; ``v[j]`` is its first right character."""
        run = v[i:j]
        if "\n" in run or "\r" in run:
            return _HARD, False
        left = v[:i]
        lc, r = v[i - 1], v[j]
        k = i - 1
        while k > 0 and v[k] in _CLOSERS:
            k -= 1
        blocked_right = r in th.NO_CUT_BEFORE or th.is_combining(r)
        if v[k] in _HARD_PUNCT:
            return (_NONE if r in _HARD_PUNCT or blocked_right else _HARD), False
        if v[k] == ".":
            dot = _dot_kind(v[: k + 1])
            if dot == "stop":
                return (_NONE if r in _HARD_PUNCT or blocked_right else _HARD), False
            if dot != "abbr":  # "title" (ดร. สมชาย) or "number" (3. …)
                return _NONE, False
        if lc in _SOFT_PUNCT:
            return (_NONE if r.isdigit() or r in _CURRENCY or blocked_right else _SOFT), False
        if lc in _STRICT_OPENERS or r in _STRICT_CLOSERS:
            return _NONE, False
        if _LAUGH_TOKEN.fullmatch(left.split()[-1]):
            return _STRONG, False  # "… 5555 ต่อไป": laughter closes the clause
        attach = th.attach_left(v[j:], complete)
        if attach is None:
            return _WAIT, False
        if attach:
            return _NONE, False
        if lc.isdigit() or r.isdigit() or lc in _CURRENCY or r in _CURRENCY or r == "%":
            return _NONE, False  # "100 บาท", "เวลา 02:30"
        if _is_ascii_word(lc) and _is_ascii_word(r):
            return _NONE, True  # "Minecraft Java Edition"
        if blocked_right or lc in th.NO_CUT_AFTER or _is_symbol(r):
            return _NONE, False  # emoji stay with the clause before them
        if _is_symbol(lc) or th.ends_with_final_particle(left):
            return _STRONG, False
        if th.is_thai(lc) or th.is_thai(r):
            return _SOFT, False
        return (_SOFT if len(left.strip()) > 20 else _NONE), False

    @staticmethod
    def _attached_kind(v: str, i: int) -> int:
        """Punctuation at ``i`` directly followed by a non-space ``v[i + 1]``."""
        ch, r = v[i], v[i + 1]
        blocked_right = r in th.NO_CUT_BEFORE or th.is_combining(r)
        if ch in _HARD_PUNCT:
            if r in _HARD_PUNCT or r in _CLOSERS or r == "." or blocked_right:
                return _NONE
            if i > 0 and _is_ascii_print(v[i - 1]) and _is_ascii_print(r):
                return _NONE  # inside a Latin token: "a?b=1", "Yahoo!Japan"
            return _HARD
        if ch == ".":  # "3.14", "e.g." never; an ellipsis glued to Thai is a pause
            return _HARD if v[max(0, i - 2) : i + 1] == "..." and th.is_thai_letter(r) else _NONE
        k = i  # closers after hard punctuation: '!"ต่อไป'
        while k > 0 and v[k] in _CLOSERS:
            k -= 1
        if v[k] in _HARD_PUNCT and r not in _CLOSERS and r not in _HARD_PUNCT and not blocked_right:
            return _HARD
        return _NONE

    def _end_site(self, b: str, i: int, mx: int) -> _Site:
        """Stall only: a site touching the buffer end counts when its left side alone is a
        strong boundary (punctuation, a final particle, laughter, an emoji)."""
        if b[i].isspace():
            if b[i:].strip():
                return _WAIT_SITE
            kind, cut, left = self._left_kind(b[:i]), min(len(b), mx), b[:i]
        else:
            if i + 1 != len(b):
                return _WAIT_SITE
            kind, cut, left = self._left_kind(b[: i + 1]), i + 1, b[: i + 1]
        if kind < _STRONG:
            return _WAIT_SITE
        return _Site(kind, cut, len(left.strip()))

    # --- word and forced cuts -------------------------------------------------------------
    def _word_checkpoint(self, b: str, lim: _Limits, final: bool) -> int | None:
        """At ``word_split_chars``, cut an unspaced run at a word boundary in
        ``[min_chars, word_split_chars]``. Declines (``None``) when whitespace or punctuation is
        near (a natural boundary is coming) or no word boundary fits; ``_PENDING`` while the
        window is incomplete."""
        at = lim.word_at
        assert at is not None
        key = ("word", at)
        if key in self._decisions:
            return self._decisions[key]
        end = at + LOOKAHEAD
        if not final and len(b) < end:
            return _PENDING
        window = b[:end]
        complete = final and len(b) <= end
        probe = window[at:].rstrip() if complete else window[at:]
        result: int | None = None
        if not any(c.isspace() or c in _HARD_PUNCT for c in probe):
            content = len(b.rstrip())
            # At the end of the stream a remainder that fits one chunk is split evenly.
            target = content / 2 if final and content <= lim.max_n else None
            result = self._word_cut(window, complete, lim.min_n, at, target)
        if len(b) >= end:
            self._decisions[key] = result
        return result

    def _forced(
        self, b: str, lim: _Limits, final: bool, fallback: dict[int, int], weak: int | None
    ) -> int:
        """The text reached ``max_chars`` without a qualifying boundary: the best natural
        boundary seen, else a word boundary, else a space inside a Latin phrase, else a
        syllable-safe cut."""
        for kind in (_HARD, _STRONG, _SOFT):
            if kind in fallback:
                return fallback[kind]
        mx = lim.max_n
        key = ("forced", mx)
        cached = self._decisions.get(key)
        if cached is not None:
            return cached
        end = mx + LOOKAHEAD
        window = b[:end]
        complete = final and len(b) <= end
        cut = self._word_cut(window, complete, lim.strong_n, mx, None)
        if cut is None:
            cut = weak if weak is not None else _safe_cut(window, lim.strong_n, mx)
        if len(b) >= end:
            self._decisions[key] = cut
        return cut

    def _word_cut(
        self, window: str, complete: bool, lo: int, hi: int, target: float | None
    ) -> int | None:
        valid = [k for k in self._word_bounds(window, complete) if lo <= k <= hi]
        valid = [k for k in valid if _cut_ok(window, k)]
        if not valid:
            return None
        if target is None:
            return valid[-1]
        return min(valid, key=lambda k: (abs(k - target), -k))

    def _word_bounds(self, window: str, complete: bool) -> list[int]:
        """Boundaries between tokens of ``window`` (the window end excluded; without the last
        token's start when more text follows, since that token may be a partial word)."""
        if self._tokenize_failed:
            return []
        try:
            tokens = list(self._tokenize(window))
        except Exception:
            log.warning("word tokenizer failed; word-boundary cuts disabled", exc_info=True)
            self._tokenize_failed = True
            return []
        if "".join(tokens) != window:
            return []
        bounds: list[int] = []
        pos = 0
        for tok in tokens[:-1]:
            pos += len(tok)
            bounds.append(pos)
        if not complete and bounds:
            bounds.pop()
        return bounds

    # --- emission -------------------------------------------------------------------------
    def _emit(self, cut: int, *, final: bool) -> list[str]:
        b = self._buf
        piece, rest = b[:cut], b[cut:]
        if final and not rest.strip():
            piece, rest = piece.rstrip(), ""
        self._buf = rest
        self._clear_caches()
        if not piece.strip():
            return []
        self._emitted += 1
        return [piece]

    def _clear_caches(self) -> None:
        self._sites.clear()
        self._decisions.clear()


def _safe_cut(window: str, lo: int, hi: int) -> int:
    """Largest syllable-safe cut in ``[lo, hi]``: prefer the start of a syllable (a leading
    vowel) near ``hi``, then any cut ``_cut_ok`` accepts, then any grapheme-safe cut."""
    floor = max(1, lo)
    for k in range(hi, max(floor, hi - 24) - 1, -1):
        if window[k] in th.LEADING_VOWELS and _cut_ok(window, k):
            return k
    for k in range(hi, floor - 1, -1):
        if _cut_ok(window, k):
            return k
    for k in range(hi, 0, -1):
        if th.is_safe_cut(window, k):
            return k
    return hi
