"""Tier-0 ``TextFilter``: Thai-aware keyword, regex and PII rules (ARCHITECTURE.md §7).

Synchronous; well under 1 ms for a chat line or a speech chunk on a warm filter (the cost
grows by about 0.25 ms per 100 characters, so callers keep texts short). How a text is
checked:

1. **Match rules** (see :mod:`aivtube.safety.lists`) run over every matching form
   (:mod:`aivtube.safety.normalize`) through one Aho-Corasick automaton. ``token`` rules only
   count when the hit starts and ends on newmm token boundaries, so ``หี`` never fires inside
   ``หีบ``; a trailing stretch (``สัสส``) still counts. Only the whitespace-delimited
   segment around a ``token`` candidate is tokenised, lazily and memoised. ``substring``
   rules count anywhere, also in the leet form.
   ``allow`` phrases are neutralised first, except for fail-closed categories, whose Thai
   ``token`` keys also match as plain substrings (over-blocking is preferred there).
2. **PII rules** (built-in regexes plus ``pii`` list rules) run on the display text.
3. **Injection rules** strip role tokens from input and names; **replace rules** patch speech.

Output checks look at ``ctx.prev_tail + text`` so a phrase split across chunks is caught, and
only hits that the tail alone does not already contain count. Verdicts follow the §7 table:

=================================== ============= ===================================
category                            in / name     out / tool / memory / game
=================================== ============= ===================================
slur, sexual, doxx, violent_extreme DROP          BLOCK
self_harm (the gate adds an alert)  DROP          BLOCK
monarchy_112 (``fail_closed``)      DROP          BLOCK
gambling_scam                       DROP          BLOCK
politics (``politics`` setting)     REVIEW        REVIEW (block → DROP/BLOCK, allow → PASS)
pii / url                           MASK          BLOCK
injection (role tokens)             REPLACE       –
=================================== ============= ===================================

Layers: base < private < ``overlays`` < per-platform < per-character. The platform and
character layers are picked from ``ctx.platform`` / ``ctx.character``; every combination is
compiled up front. ``reload()`` (``FILTER_RELOAD``) re-reads every file and swaps the compiled
state atomically; it blocks for tens of milliseconds, so call it through
``asyncio.to_thread``. A failed reload keeps the previous lists.
"""

from __future__ import annotations

import logging
import os
import re
import time
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import accumulate
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias

import ahocorasick

from aivtube.contracts.safety import FilterContext, FilterResult, Verdict
from aivtube.safety.lists import (
    MATCH_CATEGORIES,
    SPAN_CATEGORIES,
    FilterListError,
    ReplaceRule,
    RuleSet,
    list_files,
    load_rule_dir,
    load_rule_file,
)
from aivtube.safety.normalize import clean, despace, leet, norm, warm_up
from aivtube.safety.pii import PiiSpan, find_pii, mask_spans
from aivtube.text.thai import newmm

__all__ = ["TIER", "KeywordRegexFilter", "Politics", "category_verdict"]

log = logging.getLogger("aivtube.safety.keyword")

TIER: Final = "tier0"
Politics: TypeAlias = Literal["review", "block", "allow"]

_INPUT_DIRS: Final = frozenset({"in", "name"})
_STOP: Final = frozenset({Verdict.DROP, Verdict.BLOCK})
_ALLOW_MARK: Final = "\ufffc"  # neither a letter nor a despace separator
_PRIORITY: Final = {c: i for i, c in enumerate(MATCH_CATEGORIES)}
_Key: TypeAlias = tuple[str | None, str | None]


def category_verdict(category: str, direction: str, politics: Politics = "review") -> Verdict:
    """The §7 table: what a hit in ``category`` means for ``direction``."""
    stop = Verdict.DROP if direction in _INPUT_DIRS else Verdict.BLOCK
    if category == "politics":
        return {"review": Verdict.REVIEW, "block": stop, "allow": Verdict.PASS}[politics]
    if category == "pii":
        return Verdict.MASK if direction in _INPUT_DIRS else Verdict.BLOCK
    if category == "injection":
        return Verdict.REPLACE if direction in _INPUT_DIRS else Verdict.PASS
    return stop  # every other category (and anything unknown) stops the text


@dataclass(eq=False, slots=True)
class _Entry:
    """One compiled key of a match rule (hashable by identity)."""

    key: str
    rule_id: str
    category: str
    token: bool  # verify newmm token boundaries
    latin: bool  # ASCII key: also matched on the leet forms
    fail_closed: bool
    priority: int


class _Bounds:
    """Token boundaries of one segment, addressed with absolute offsets into the form."""

    __slots__ = ("_base", "_ordered", "_set")

    def __init__(self, base: int, ordered: tuple[int, ...], as_set: frozenset[int]) -> None:
        self._base = base
        self._ordered = ordered
        self._set = as_set

    def __contains__(self, offset: object) -> bool:
        return isinstance(offset, int) and offset - self._base in self._set

    def before(self, offset: int) -> int:
        """The last boundary strictly before ``offset``."""
        i = bisect_left(self._ordered, offset - self._base)
        return self._base + self._ordered[max(0, i - 1)]

    def after(self, offset: int) -> int:
        """The first boundary strictly after ``offset``."""
        i = bisect_right(self._ordered, offset - self._base)
        return self._base + self._ordered[min(i, len(self._ordered) - 1)]


def _local_bounds(form: str, start: int, end: int) -> _Bounds | None:
    """newmm token boundaries of the whitespace-delimited segment around ``form[start:end]``;
    ``None`` means "no tokenizer: accept every hit".

    newmm never joins words across whitespace, so tokenising only that segment gives the
    same cuts as tokenising the whole text, and repeated segments hit the cache.
    """
    if not newmm.available:
        return None
    a = form.rfind(" ", 0, start) + 1
    b = form.find(" ", end)
    if b < 0:
        b = len(form)
    cached = _segment_bounds(form[a:b])
    if cached is None:
        return None
    return _Bounds(a, *cached)


@lru_cache(maxsize=4096)
def _segment_bounds(segment: str) -> tuple[tuple[int, ...], frozenset[int]] | None:
    offsets = (0, *accumulate(len(t) for t in newmm(segment)))
    if offsets[-1] != len(segment):  # tokens do not join back to the input: be conservative
        return None
    return offsets, frozenset(offsets)


def _on_boundaries(form: str, start: int, end: int, bounds: _Bounds) -> bool:
    if start not in bounds:
        return False
    return _end_ok(form, end, bounds)


def _end_ok(form: str, end: int, bounds: _Bounds) -> bool:
    """``end`` is a boundary, or only a stretch of the hit's last letter follows (``สัสส``)."""
    if end in bounds:
        return True
    last = form[end - 1]
    j = end
    while j < len(form) and form[j] == last:
        j += 1
    return j != end and j in bounds


_words: frozenset[str] | None = None
_RELAXED_MIN: Final = 3  # two-letter keys (หี) only ever match on exact token boundaries
_SWALLOW_MAX: Final = 12  # longest extension tried when looking for a longer word
_LOOKAHEAD: Final = 24  # characters tokenised after that longer word


def _dictionary() -> frozenset[str]:
    """pythainlp's Thai word list (already loaded by newmm); empty when unavailable."""
    global _words
    if _words is None:
        try:
            from pythainlp.corpus import thai_words

            _words = frozenset(thai_words())
        except Exception:  # pragma: no cover - pythainlp is a core dependency
            _words = frozenset()
    return _words


def _is_thai(ch: str) -> bool:
    return "\u0e00" <= ch <= "\u0e7f"


def _is_latin_alnum(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def _is_word(text: str, words: frozenset[str]) -> bool:
    """A real dictionary word of 2+ letters (a trailing ๆ is ignored), or non-Thai text."""
    if not text or not _is_thai(text[0]):
        return True
    word = text.rstrip("ๆ")
    return len(word) >= 2 and word in words


def _swallowed(form: str, start: int, end: int) -> bool:
    """The hit is the start of a longer, different dictionary word whose extra letters are
    not a word themselves (``เหี้ย|ม`` in ``เหี้ยมมาก``, ``หี|บ``, ``ทักษิณ|า``). A compound
    whose extra part is a word (``การเมือง|การปกครอง``) is still a hit."""
    if end >= len(form) or not _is_thai(form[end]):
        return False
    # Only the text right after the hit matters: memoise on that segment (spam repeats).
    return _swallowed_segment(form[start : end + _SWALLOW_MAX + _LOOKAHEAD], end - start)


@lru_cache(maxsize=4096)
def _swallowed_segment(segment: str, key_len: int) -> bool:
    words = _dictionary()
    for j in range(key_len + 1, min(len(segment), key_len + _SWALLOW_MAX) + 1):
        if segment[:j] not in words or _is_word(segment[key_len:j], words):
            continue
        rest = segment[j:]
        if not rest or not _is_thai(rest[0]) or rest[0] == "ๆ":
            return True
        following = newmm(rest[:_LOOKAHEAD])
        if following and _is_word(following[0], words):
            return True
    return False


#: Words newmm often glues onto a neighbour (prefixes before a hit, particles after one).
_LEFT_AFFIXES: Final = frozenset((
    "การ", "ความ", "นัก", "ผู้", "ชาว", "พวก", "คน", "ไอ้", "อี", "ขี้", "โคตร",
    "ใน", "ที่", "ของ", "กับ", "และ", "แต่", "ก็", "จะ", "ไม่", "ว่า",
))  # fmt: skip
_RIGHT_AFFIXES: Final = frozenset((
    "แล้ว", "นี้", "นั้น", "นี่", "นั่น", "ปี", "ไป", "มา", "อยู่", "กัน", "เลย", "ด้วย",
    "นะ", "จ้ะ", "จ้า", "ครับ", "ค่ะ", "คะ", "จัง", "มาก", "จริง", "อีก", "หน่อย", "ไหม",
    "มั้ย", "เหรอ", "ซะ", "สิ", "เถอะ", "เว้ย", "วะ", "ว่ะ", "โว้ย", "ที", "บ้าง", "เอง",
    "ได้", "แน่", "ครั้ง", "ละ", "ล่ะ",
))  # fmt: skip


def _relaxed_ok(form: str, start: int, end: int, bounds: _Bounds) -> bool:
    """newmm glued a prefix or particle onto the hit (``การเลือกตั้ง``, ``เลือก|ตั้งปี``,
    ``อยู่แล้ว``). Accept when the part of each straddling token outside the hit is such an
    affix; a content word there (``คลิป|หลุดโลก``, ``ขาย|ตัวละคร``) means a different
    compound. Never inside a Latin word."""
    if start not in bounds:
        if _is_latin_alnum(form[start - 1]) and _is_latin_alnum(form[start]):
            return False
        if form[bounds.before(start) : start] not in _LEFT_AFFIXES:
            return False
    if not _end_ok(form, end, bounds):
        if _is_latin_alnum(form[end - 1]) and _is_latin_alnum(form[end]):
            return False
        if form[end : bounds.after(end)].rstrip("ๆ") not in _RIGHT_AFFIXES:
            return False
    return True


def _token_hit(form: str, start: int, end: int, bounds: _Bounds | None) -> bool:
    """Does a ``token`` rule matching ``form[start:end]`` count as a word hit?"""
    if bounds is None:  # no tokenizer: over-block rather than miss
        return True
    if _dictionary() and _swallowed(form, start, end):
        return False
    if _on_boundaries(form, start, end, bounds):
        return True
    return end - start >= _RELAXED_MIN and _relaxed_ok(form, start, end, bounds)


class _Matcher:
    """Compiled rules of one layer combination (immutable after construction)."""

    def __init__(
        self,
        rules: RuleSet,
        *,
        fail_closed: frozenset[str],
        key_cache: dict[str, str],
    ) -> None:
        def key(text: str) -> str:
            k = key_cache.get(text)
            if k is None:
                k = key_cache[text] = norm(text)
            return k

        allow_keys: set[str] = set()
        for text in rules.allow:
            k = key(text)
            if k:
                allow_keys.update({k, k.replace(" ", "")})
        self._allow: Any = None
        if allow_keys:
            self._allow = ahocorasick.Automaton()
            for k in allow_keys:
                self._allow.add_word(k, len(k))
            self._allow.make_automaton()

        by_key: dict[str, list[_Entry]] = {}
        self._regex: list[tuple[re.Pattern[str], _Entry]] = []
        self._pii: list[tuple[str, re.Pattern[str]]] = []
        self._strip: list[tuple[re.Pattern[str], str]] = []
        for rule in rules.deny:
            fc = rule.category in fail_closed
            prio = _PRIORITY.get(rule.category, len(_PRIORITY))
            if rule.category in SPAN_CATEGORIES:
                pat = rule.text if rule.match == "regex" else re.escape(rule.text)
                compiled = re.compile(pat, re.IGNORECASE)
                if rule.category == "pii":
                    self._pii.append(("link", compiled))
                else:
                    self._strip.append((compiled, rule.rule_id))
                continue
            if rule.match == "regex":
                entry = _Entry(rule.text, rule.rule_id, rule.category, False, False, fc, prio)
                self._regex.append((re.compile(rule.text, re.IGNORECASE), entry))
                continue
            k = key(rule.text)
            for variant in {k, k.replace(" ", "")}:
                if not variant or (not fc and variant in allow_keys):
                    continue
                latin = variant.isascii()
                entry = _Entry(
                    variant,
                    rule.rule_id,
                    rule.category,
                    # fail-closed Thai keys match as substrings: over-block rather than miss
                    rule.match == "token" and (latin or not fc),
                    latin,
                    fc,
                    prio,
                )
                by_key.setdefault(variant, []).append(entry)
        self._ac: Any = None
        if by_key:
            self._ac = ahocorasick.Automaton()
            for k, entries in by_key.items():
                self._ac.add_word(k, tuple(entries))
            self._ac.make_automaton()
        self._replace = [self._compile_replace(r) for r in rules.replace]
        self.entries = sum(len(v) for v in by_key.values()) + len(self._regex)

    @staticmethod
    def _compile_replace(rule: ReplaceRule) -> tuple[re.Pattern[str], str, frozenset[str]]:
        if rule.match == "regex":
            return re.compile(rule.pattern), rule.replacement, rule.directions
        return re.compile(re.escape(rule.pattern), re.IGNORECASE), rule.replacement, rule.directions

    # -- matching ------------------------------------------------------------------------

    def scan(self, text: str) -> dict[_Entry, int]:
        """Hit counts per entry (the maximum over all matching forms)."""
        base = norm(text)
        if not base:
            return {}
        forms: list[tuple[str, bool]] = [(base, False)]
        desp = despace(base)
        if desp != base:
            forms.append((desp, False))
        for form, _ in list(forms):
            lf = leet(form)
            if lf != form and all(lf != f for f, _ in forms):
                forms.append((lf, True))
        counts: dict[_Entry, int] = {}
        for form, latin_only in forms:
            for entry, n in self._scan_form(form, latin_only).items():
                if n > counts.get(entry, 0):
                    counts[entry] = n
        return counts

    def _neutralize(self, form: str) -> str:
        if self._allow is None:
            return form
        spans = [(end - n + 1, end + 1) for end, n in self._allow.iter(form)]
        if not spans:
            return form
        chars = list(form)
        for a, b in spans:
            chars[a:b] = _ALLOW_MARK * (b - a)
        return "".join(chars)

    def _scan_form(self, form: str, latin_only: bool) -> dict[_Entry, int]:
        found: dict[_Entry, int] = {}
        neutral = self._neutralize(form)
        if neutral is form:
            self._pass(form, latin_only, found, None)
        else:  # allow phrases present: they never loosen fail-closed categories
            self._pass(neutral, latin_only, found, False)
            self._pass(form, latin_only, found, True)
        return found

    def _pass(
        self, form: str, latin_only: bool, found: dict[_Entry, int], fail_closed: bool | None
    ) -> None:
        """One AC + regex pass; ``fail_closed`` restricts to (True) or excludes (False) them."""
        if self._ac is not None:
            for end, entries in self._ac.iter(form):
                for e in entries:
                    if latin_only and not e.latin:
                        continue
                    if fail_closed is not None and e.fail_closed is not fail_closed:
                        continue
                    if e.token:
                        start = end - len(e.key) + 1
                        bounds = _local_bounds(form, start, end + 1)
                        if not _token_hit(form, start, end + 1, bounds):
                            continue
                    found[e] = found.get(e, 0) + 1
        if latin_only:
            return
        for pattern, e in self._regex:
            if fail_closed is not None and e.fail_closed is not fail_closed:
                continue
            n = sum(1 for _ in pattern.finditer(form))
            if n:
                found[e] = found.get(e, 0) + n

    def pii(
        self, text: str, aliases: Sequence[str], allow_ranges: Sequence[tuple[int, int]]
    ) -> list[PiiSpan]:
        return find_pii(text, self._pii, allow_ranges=allow_ranges, handle_aliases=aliases)

    def allow_ranges(self, text: str) -> list[tuple[int, int]]:
        """Allow-phrase occurrences in display text (for exempting PII such as own links)."""
        if self._allow is None:
            return []
        low = text.lower()
        if len(low) != len(text):
            return []
        return [(end - n + 1, end + 1) for end, n in self._allow.iter(low)]

    def strip(self, text: str) -> tuple[str, str | None]:
        """Remove role tokens and injection phrases; returns (text, first rule id or None)."""
        first: str | None = None
        for pattern, rule_id in self._strip:
            new = pattern.sub(" ", text)
            if new != text:
                first = first or rule_id
                text = new
        if first is not None:
            text = re.sub(r"[ \t]{2,}", " ", text).strip()
        return text, first

    def replace(self, text: str, direction: str) -> str:
        for pattern, repl, directions in self._replace:
            if direction in directions:
                text = pattern.sub(repl, text)
        return text


@dataclass(frozen=True, slots=True)
class _State:
    matchers: Mapping[_Key, _Matcher]
    platforms: frozenset[str]
    characters: frozenset[str]
    files: tuple[Path, ...]
    snapshot: tuple[tuple[str, int, int], ...]
    counts: Mapping[str, int]
    loaded_at: float

    def matcher(self, platform: str | None, character: str | None) -> _Matcher:
        p = platform if platform in self.platforms else None
        c = character if character in self.characters else None
        return self.matchers[(p, c)]


def _stat(path: Path) -> tuple[str, int, int]:
    try:
        st = os.stat(path)
    except OSError:
        return (str(path), -1, -1)
    return (str(path), st.st_mtime_ns, st.st_size)


class KeywordRegexFilter:
    """Tier-0 filter over committed base lists, private lists and overlays (see module doc).

    Construction reads and compiles every list (blocking, tens of ms) and, with ``warm``,
    loads the pythainlp dictionary (~0.5 s): build it in a thread at startup.
    """

    def __init__(
        self,
        base_dir: Path,
        private_dir: Path | None = None,
        overlays: Sequence[Path] = (),
        *,
        platform_overlays: Mapping[str, Path] | None = None,
        character_overlays: Mapping[str, Path] | None = None,
        politics: Politics = "review",
        fail_closed: Iterable[str] = ("monarchy_112",),
        mask_text: str = "[ลิงก์]",
        handle_aliases: Mapping[str, Sequence[str]] | None = None,
        name: str = TIER,
        warm: bool = True,
    ) -> None:
        self.name = name
        self.base_dir = Path(base_dir)
        self.private_dir = Path(private_dir) if private_dir is not None else None
        self.overlays = tuple(Path(p) for p in overlays)
        self.platform_overlays = {k: Path(v) for k, v in (platform_overlays or {}).items()}
        self.character_overlays = {k: Path(v) for k, v in (character_overlays or {}).items()}
        self.politics: Politics = politics
        self.fail_closed = frozenset(fail_closed) | {"monarchy_112"}
        self.mask_text = mask_text
        self._aliases: dict[str, tuple[str, ...]] = {
            char: tuple(
                sorted({a.casefold() for a in names if a.isascii() and len(a.strip()) >= 3})
            )
            for char, names in (handle_aliases or {}).items()
        }
        self.reloads = 0
        self._state = self._build()
        if warm:
            self.warm()

    # -- lifecycle -----------------------------------------------------------------------

    def warm(self) -> float:
        """Load pythainlp and run one check (blocking; call in a thread at startup)."""
        t0 = time.perf_counter()
        warm_up()
        _dictionary()
        ctx = FilterContext("out", "", prev_tail="ทดสอบ")
        self.check("ส่งหีบห่อมาให้หน่อยนะ fuck", ctx)
        return time.perf_counter() - t0

    def reload(self) -> None:
        """Re-read every list file and swap the compiled rules atomically (blocking).

        Raises :class:`FilterListError` on a bad file and keeps the previous lists.
        """
        try:
            state = self._build()
        except FilterListError:
            log.exception("filter reload failed; keeping the previous lists")
            raise
        self._state = state
        self.reloads += 1

    def reload_if_changed(self) -> bool:
        """Reload when a list file was added, removed or modified. Returns True on reload."""
        if self._snapshot() == self._state.snapshot:
            return False
        self.reload()
        return True

    @property
    def files(self) -> tuple[Path, ...]:
        return self._state.files

    def stats(self) -> dict[str, Any]:
        st = self._state
        return {
            "files": [str(p) for p in st.files],
            "counts": dict(st.counts),
            "layers": len(st.matchers),
            "reloads": self.reloads,
        }

    def _watched(self) -> list[Path]:
        paths: list[Path] = [self.base_dir]
        if self.private_dir is not None:
            paths.append(self.private_dir)
        paths.extend(self.overlays)
        paths.extend(self.platform_overlays.values())
        paths.extend(self.character_overlays.values())
        return paths

    def _snapshot(self) -> tuple[tuple[str, int, int], ...]:
        out: list[tuple[str, int, int]] = []
        for path in self._watched():
            if path.is_dir():
                out.extend(_stat(p) for p in list_files(path))
            else:
                out.append(_stat(path))
        return tuple(out)

    @staticmethod
    def _optional(path: Path) -> RuleSet:
        if not path.is_file():
            log.info("filter overlay %s not found; skipped", path)
            return RuleSet()
        return load_rule_file(path)

    def _build(self) -> _State:
        snapshot = self._snapshot()
        common = load_rule_dir(self.base_dir, required=True)
        if not common.deny:
            raise FilterListError(
                self.base_dir, "the base lists are empty", "รายการคำกรองหลักว่างเปล่า"
            )
        if self.private_dir is not None:
            common = common + load_rule_dir(self.private_dir, required=False)
        for path in self.overlays:
            common = common + self._optional(path)
        platform = {k: self._optional(p) for k, p in self.platform_overlays.items()}
        character = {k: self._optional(p) for k, p in self.character_overlays.items()}
        cache: dict[str, str] = {}
        matchers: dict[_Key, _Matcher] = {}
        for p in (None, *platform):
            for c in (None, *character):
                rules = common
                if p is not None:
                    rules = rules + platform[p]
                if c is not None:
                    rules = rules + character[c]
                matchers[(p, c)] = _Matcher(rules, fail_closed=self.fail_closed, key_cache=cache)
        files = common.files + tuple(
            f for r in (*platform.values(), *character.values()) for f in r.files
        )
        state = _State(
            matchers=matchers,
            platforms=frozenset(platform),
            characters=frozenset(character),
            files=files,
            snapshot=snapshot,
            counts=common.counts(),
            loaded_at=time.perf_counter(),
        )
        log.info(
            "tier-0 filter loaded: %d files, %d layers, %s",
            len(files),
            len(matchers),
            ", ".join(f"{k}={v}" for k, v in state.counts.items() if v),
        )
        return state

    # -- TextFilter ----------------------------------------------------------------------

    @staticmethod
    def _pii(m: _Matcher, text: str, aliases: Sequence[str], direction: str) -> list[PiiSpan]:
        spans = m.pii(text, aliases, m.allow_ranges(text))
        if spans and direction == "name":  # a YouTube author name *is* an @handle
            spans = [s for s in spans if s.kind != "handle"]
        return spans

    def check(self, text: str, ctx: FilterContext) -> FilterResult:
        m = self._state.matcher(ctx.platform, ctx.character)
        direction = ctx.direction
        tail = ctx.prev_tail
        full = tail + text if tail else text

        counts = m.scan(full) if full else {}
        if counts and tail:
            old = m.scan(tail)
            if old:
                counts = {e: n for e, n in counts.items() if n > old.get(e, 0)}
        fail_closed = any(e.fail_closed for e in counts)
        review: _Entry | None = None
        for e in sorted(counts, key=lambda x: x.priority):
            verdict = category_verdict(e.category, direction, self.politics)
            if verdict in _STOP:
                return FilterResult(
                    verdict, "", TIER, rule=e.rule_id, category=e.category, fail_closed=fail_closed
                )
            if verdict is Verdict.REVIEW and review is None:
                review = e

        aliases = self._aliases.get(ctx.character, ())
        spans = self._pii(m, clean(full), aliases, direction)
        if spans and tail:  # only spans reaching into the new chunk are new
            boundary = len(clean(tail))
            spans = [s for s in spans if s.end > boundary]
        if spans and direction not in _INPUT_DIRS:
            return FilterResult(Verdict.BLOCK, "", TIER, rule=spans[0].rule, category="pii")

        verdict = Verdict.PASS
        rule: str | None = None
        category: str | None = None
        out = text
        if direction in _INPUT_DIRS:
            work = clean(text)
            if spans:
                own = spans if not tail else self._pii(m, work, aliases, direction)
                if own:
                    work = mask_spans(work, own, mask_text=self.mask_text)
                    verdict, rule, category = Verdict.MASK, own[0].rule, "pii"
            stripped, strip_rule = m.strip(work)
            if strip_rule is not None:
                if not stripped:
                    return FilterResult(
                        Verdict.DROP, "", TIER, rule=strip_rule, category="injection"
                    )
                work = stripped
                if verdict is Verdict.PASS:
                    verdict, rule, category = Verdict.REPLACE, strip_rule, "injection"
            patched = m.replace(work, direction)
            if patched != work and verdict is Verdict.PASS:
                verdict, rule = Verdict.REPLACE, "replace"
            if verdict is not Verdict.PASS:
                out = patched
        else:
            patched = m.replace(text, direction)
            if patched != text:
                verdict, rule, out = Verdict.REPLACE, "replace", patched
        if review is not None:
            verdict, rule, category = Verdict.REVIEW, review.rule_id, review.category
        return FilterResult(verdict, out, TIER, rule=rule, category=category)
