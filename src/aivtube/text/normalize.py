"""TTS text normalisers and the speakability check (§4.6; TTS brief §5).

Applied per chunk, after chunking and after the safety gate:

- ``normalize_cloud`` for edge-tts / Azure. Those voices read digits, times, currencies and
  English natively, so it only fixes what they get wrong: emoji (Edge speaks their names),
  markdown and stage directions, URLs, ``555`` laughter (read as a number), elongations
  (``ว้าวววว``), ``!!!`` runs and Thai digits. The literal ``Filtered.`` is never touched.
- ``normalize_local`` for Piper/VITS voices whose vocabulary is Thai script only: everything
  above, then the character lexicon (``characters/<id>/lexicon.toml``), Thai number reading
  (pythainlp ``num_to_thaiword`` / ``bahttext``), numeric ranges (``10-20`` → ``10 ถึง 20``),
  ๆ expansion, acronyms spelled in Thai, and finally any remaining Latin letters removed (Piper
  drops them anyway) together with the punctuation they leave behind. ``Filtered.`` is read
  through the lexicon (built-in fallback ``ฟิลเทอร์ด``), since these voices cannot say Latin.

Both return plain single-line text with collapsed whitespace. The re-exports at the bottom make
this module the one import for the whole ``text.normalize`` API of modules.json.
"""

from __future__ import annotations

import functools
import logging
import re
import sys
import unicodedata
from collections.abc import Callable, Mapping
from typing import Final

from aivtube.text.tags import EmotionTagExtractor
from aivtube.text.thai import (
    NameMatcher,
    _on_event_loop,
    collapse_repeats,
    estimate_tokens,
    is_question,
    is_thai_letter,
    newmm,
    nfkc_casefold,
    strip_zero_width,
    thai_digits_to_arabic,
)

__all__ = [
    "FILTERED",
    "EmotionTagExtractor",
    "NameMatcher",
    "apply_lexicon",
    "collapse_repeats",
    "estimate_tokens",
    "is_question",
    "is_speakable",
    "nfkc_casefold",
    "normalize_cloud",
    "normalize_local",
    "strip_zero_width",
    "thai_digits_to_arabic",
]

log = logging.getLogger("aivtube.text.normalize")

#: The canned "blocked output" phrase (§4.10). Cloud normalisation keeps it byte-exact.
FILTERED: Final = "Filtered."
#: Local voices cannot read Latin; the character lexicon may override this reading.
_BUILTIN_LOCAL_LEXICON: Final[Mapping[str, str]] = {FILTERED: "ฟิลเทอร์ด"}
_PLACEHOLDER: Final = chr(0xE000)  # private use: marks a protected "Filtered."

# --- cloud patterns ---------------------------------------------------------------------

_PRIVATE_USE: Final = re.compile("[\U0000e000-\U0000f8ff]")
_MD_IMAGE_OR_LINK: Final = re.compile(r"!?\[([^\]\n]{0,200})\]\([^)\s]{0,500}\)")
_BOLD: Final = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
# *หัวเราะ* stage directions are not spoken (never across a protected "Filtered.").
_ACTION: Final = re.compile("\\*[^*\n\U0000e000]{1,60}\\*")
_CODE: Final = re.compile(r"`+([^`]*)`+")
_URL: Final = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|(?<![A-Za-z0-9@.])[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\."
    r"(?:com|net|org|gg|io|tv|th|me|ly|app|dev|xyz|co|link|live)(?:/\S*)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
# List bullets / numbers at a line start ("- ", "• ", "3. ", "1) "), repeated ones included.
_LINE_MARKER: Final = re.compile(r"(?m)^\s*(?:(?:[-+•●▪]|\d{1,2}[.)])\s+)+")
_EMOJI: Final = re.compile(
    "["
    "\U0001f000-\U0001faff"  # pictographs, emoticons, transport, flags, skin tones
    "\U00002600-\U000027bf"  # misc symbols, dingbats
    "\U00002b00-\U00002bff"  # misc symbols and arrows
    "\U00002300-\U000023ff"  # misc technical (watch, hourglass, media keys)
    "\U00002190-\U000021ff"  # arrows
    "\U000025a0-\U000025ff"  # geometric shapes
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0000200d\U000020e3"  # ZWJ, keycap
    "\U000e0020-\U000e007f"  # tag characters (subdivision flags)
    "\U000000a9\U000000ae\U00002122\U00003030\U0000303d\U00003297\U00003299\U00002049\U0000203c"
    "]+"
)
_MD_CHARS: Final = re.compile(r"[*_`#>~|^\[\]{}<>\\]+")
_LAUGH_UNITS: Final = (
    "บาท|คน|ครั้ง|ปี|วัน|เดือน|ชั่วโมง|ชม|นาที|วินาที|ตัว|อัน|ชิ้น|เมตร|กิโล|กรัม|ลิตร|"
    "ดอลลาร์|เหรียญ|บิท|คะแนน|แต้ม|วิว|ซับ|ล้าน|พัน|หมื่น|แสน|เปอร์เซ็นต์|%|km|kg|views|subs|bits"
)
_LAUGH: Final = re.compile(
    r"(?<![\d.,$฿€£¥])5{3,}\+*(?![\d.,]*\d)(?!\s*(?:" + _LAUGH_UNITS + r"))",
    re.IGNORECASE,
)
# Lowercase Latin only: "Wooow" → "Wow", but "XIII" and "AAA" stay.
_ELONGATION: Final = re.compile(
    "([\U00000e01-\U00000e2e\U00000e30-\U00000e3a\U00000e40-\U00000e4ea-z])\\1{2,}"
)
_PUNCT_RUN: Final = re.compile(r"[!?！？]{2,}")
_DOTS: Final = re.compile(r"\.{4,}")
_ELLIPSES: Final = re.compile(r"…{2,}")
_MAIYAMOK_SPACE: Final = re.compile(r"\s+ๆ")
_WS: Final = re.compile(r"\s+")


def _punct_run(m: re.Match[str]) -> str:
    run = m.group(0)
    return "?" if "?" in run or "？" in run else "!"


def normalize_cloud(text: str, lexicon: Mapping[str, str] | None = None) -> str:
    """Normalise one chunk for a cloud voice (edge-tts / Azure).

    ``lexicon`` optionally applies pronunciation fixes a cloud voice needs (whole-word,
    case-insensitive); ``Filtered.`` is protected from it and from every other rule.
    """
    t = _PRIVATE_USE.sub("", strip_zero_width(text))
    t = unicodedata.normalize("NFC", t).replace(FILTERED, _PLACEHOLDER)
    t = thai_digits_to_arabic(t)
    t = _MD_IMAGE_OR_LINK.sub(r"\1", t)
    t = _BOLD.sub(r"\2", t)
    t = _ACTION.sub(" ", t)
    t = _CODE.sub(r"\1", t)
    t = _URL.sub(" ลิงก์ ", t)
    t = _EMOJI.sub(" ", t)
    t = _MD_CHARS.sub(" ", t)
    t = _LINE_MARKER.sub("", t)  # after emoji/markdown, which may have hidden a marker
    t = _LAUGH.sub(" ฮ่าฮ่าฮ่า ", t)
    t = _MAIYAMOK_SPACE.sub("ๆ", t)
    t = _ELONGATION.sub(r"\1", t)
    t = _PUNCT_RUN.sub(_punct_run, t)
    t = _DOTS.sub("...", t)
    t = _ELLIPSES.sub("…", t)
    if lexicon:
        t = apply_lexicon(t, lexicon)
    t = t.replace(_PLACEHOLDER, FILTERED)
    return _LINE_MARKER.sub("", _WS.sub(" ", t).strip())  # "-555" → "- ฮ่าฮ่าฮ่า" → "ฮ่าฮ่าฮ่า"


# --- lexicon ----------------------------------------------------------------------------


def _lexicon_key(text: str) -> str:
    return " ".join(text.casefold().split())


@functools.lru_cache(maxsize=32)
def _compile_lexicon(
    items: tuple[tuple[str, str], ...],
) -> tuple[re.Pattern[str], dict[str, str]] | None:
    table: dict[str, str] = {}
    parts: list[str] = []
    for key, value in sorted(items, key=lambda kv: len(kv[0]), reverse=True):
        norm = _lexicon_key(key)
        if not norm or norm in table:
            continue
        table[norm] = value
        body = r"\s+".join(re.escape(w) for w in key.split())
        if key[0].isascii() and key[0].isalnum():
            body = r"(?<![A-Za-z0-9])" + body
        if key[-1].isascii() and key[-1].isalnum():
            body += r"(?![A-Za-z0-9])"
        parts.append(body)
    if not parts:
        return None
    return re.compile("|".join(parts), re.IGNORECASE), table


def apply_lexicon(text: str, lexicon: Mapping[str, str]) -> str:
    """Replace lexicon words (case-insensitive; Latin keys only as whole words; longest first).

    One pass, so a replacement is never re-replaced. Thai keys match anywhere (no spaces
    between Thai words). Multi-word keys match across any whitespace.
    """
    if not lexicon:
        return text
    compiled = _compile_lexicon(tuple(sorted((str(k), str(v)) for k, v in lexicon.items())))
    if compiled is None:
        return text
    pattern, table = compiled
    return pattern.sub(lambda m: table.get(_lexicon_key(m.group(0)), m.group(0)), text)


# --- local (Thai-script-only voices) ------------------------------------------------------

_DIGIT_WORDS: Final = ("ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า")
_LATIN_LETTER_TH: Final = {
    "A": "เอ", "B": "บี", "C": "ซี", "D": "ดี", "E": "อี", "F": "เอฟ", "G": "จี", "H": "เอช",
    "I": "ไอ", "J": "เจ", "K": "เค", "L": "แอล", "M": "เอ็ม", "N": "เอ็น", "O": "โอ", "P": "พี",
    "Q": "คิว", "R": "อาร์", "S": "เอส", "T": "ที", "U": "ยู", "V": "วี", "W": "ดับเบิลยู",
    "X": "เอ็กซ์", "Y": "วาย", "Z": "แซด",
}  # fmt: skip
_CURRENCY_TH: Final = {"$": "ดอลลาร์", "€": "ยูโร", "£": "ปอนด์", "¥": "เยน", "₩": "วอน"}

_PHONE: Final = re.compile(r"(?<![\d.,])(0\d{1,2})-?(\d{3})-?(\d{3,4})(?![\d.,]*\d)")
_TIME_COLON: Final = re.compile(
    r"(?<![\d.:])([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?(?![\d:])(?:\s*(?:น\.|นาฬิกา))?"
)
_TIME_DOT: Final = re.compile(r"(?<![\d.:])([01]?\d|2[0-3])\.([0-5]\d)\s*(?:น\.|นาฬิกา)")
_CURRENCY_PREFIX: Final = re.compile(r"([$฿€£¥₩])\s?(\d[\d,]*(?:\.\d+)?)")
_BAHT_DECIMAL: Final = re.compile(r"(\d[\d,]*\.\d{1,2})\s*บาท")
_PERCENT: Final = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*%")
_DEGREES: Final = re.compile(r"°\s*([CF])?")
_NEGATIVE: Final = re.compile(r"(?<![^\s(])-(?=\d)")  # "-5" at a word start, not "10-20"
# "10-20 คน" → "10 ถึง 20 คน"; not dates or codes with more dashes ("2026-09-26").
_RANGE: Final = re.compile(r"(?<![\d.,\-–])(\d[\d,.]*)\s?[-–]\s?(\d[\d,.]*)(?![\d.,]*[-–]\d)")
_NUMBER: Final = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{3})+|\d+)((?:\.\d+)+)?(?!\d)")
_ACRONYM: Final = re.compile(r"(?<![A-Za-z])[A-Z]{1,5}(?![A-Za-z])")
_NON_THAI_LETTERS: Final = re.compile("[^\\W\\d_\U00000e00-\U00000e7f]+")
# Punctuation left alone once Latin words are gone: " ." joins the word before; leading goes.
_ORPHAN_PUNCT: Final = re.compile("\\s+([^\\w\\s\U00000e00-\U00000e7f]+)(?=\\s|$)")
_LEADING_PUNCT: Final = re.compile("^[^\\w\U00000e00-\U00000e7f]+")


@functools.cache
def _pythainlp_util() -> tuple[Callable[[int], str], Callable[[float], str]] | None:
    if "pythainlp.util" not in sys.modules and _on_event_loop():
        log.warning(
            "pythainlp.util imported lazily on the event loop; "
            "call warm_up_pythainlp() in a thread at startup"
        )
    try:
        from pythainlp.util import bahttext, num_to_thaiword
    except Exception:  # pragma: no cover - pythainlp is a core dependency
        log.warning("pythainlp.util unavailable; numbers are read digit by digit", exc_info=True)
        return None
    return num_to_thaiword, bahttext


def _digits(s: str) -> str:
    return "".join(_DIGIT_WORDS[int(d)] for d in s if d.isdigit())


def _int_words(s: str) -> str:
    s = s.replace(",", "")
    util = _pythainlp_util()
    if util is None or len(s) > 15 or (len(s) > 1 and s.startswith("0")):
        return _digits(s)
    return util[0](int(s))


def _number_words(int_part: str, frac: str | None) -> str:
    words = _int_words(int_part)
    if not frac:
        return words
    parts = frac.lstrip(".").split(".")
    if len(parts) == 1:  # decimal: digits read one by one (3.14 → สามจุดหนึ่งสี่)
        return f"{words}จุด{_digits(parts[0])}"
    return words + "".join(f"จุด{_int_words(p)}" for p in parts)  # version 1.21.4


def _amount_words(amount: str) -> str:
    int_part, _, frac = amount.partition(".")
    return _number_words(int_part, frac or None)


def _baht(amount: str) -> str:
    clean = amount.replace(",", "")
    util = _pythainlp_util()
    if "." in clean and util is not None:
        try:
            return util[1](float(clean))
        except (ValueError, TypeError):  # pragma: no cover - defensive
            pass
    return _amount_words(clean) + "บาท"


def _time_words(m: re.Match[str]) -> str:
    hours, minutes = int(m.group(1)), int(m.group(2))
    seconds = m.group(3) if m.lastindex and m.lastindex >= 3 else None
    out = f"{_int_words(str(hours))}นาฬิกา"
    if minutes:
        out += f"{_int_words(str(minutes))}นาที"
    if seconds and int(seconds):
        out += f"{_int_words(str(int(seconds)))}วินาที"
    return f" {out} "


def _currency_words(m: re.Match[str]) -> str:
    symbol, amount = m.group(1), m.group(2)
    if symbol == "฿":
        return f" {_baht(amount)} "
    return f" {_amount_words(amount.replace(',', ''))}{_CURRENCY_TH[symbol]} "


def _expand_maiyamok(text: str) -> str:
    """``ดีๆ`` → ``ดีดี``: repeat the word before ๆ (newmm tokens). Without the tokenizer the
    mark is dropped: repeating a whole unspaced run would be worse than saying the word once."""
    if "ๆ" not in text:
        return text
    tokens = newmm(text)
    if not newmm.available:
        return text.replace("ๆ", " ")
    out: list[str] = []
    prev = ""
    for tok in tokens:
        while "ๆ" in tok:  # newmm yields "ดี", "ๆ" or a single "ดี ๆ" token
            before, _, tok = tok.partition("ๆ")
            word = before.strip()
            if word:
                out.append(word + word)
                prev = word
            else:
                out.append(prev)
        out.append(tok)
        if any(is_thai_letter(c) or c.isalnum() for c in tok):
            prev = tok.strip()
    return "".join(out)


def normalize_local(text: str, lexicon: Mapping[str, str] | None = None) -> str:
    """Normalise one chunk for a Thai-script-only local voice (Piper/VITS).

    Runs ``normalize_cloud``, applies ``lexicon`` (the character's EN→Thai readings, falling
    back to a built-in reading of ``Filtered.``), reads numbers, times, phone numbers,
    currencies and percentages in Thai words, expands ๆ, spells acronyms, and removes any
    Latin letters left. Uses pythainlp (warm it at startup; see ``warm_up_pythainlp``).
    """
    t = normalize_cloud(text)
    merged = dict(_BUILTIN_LOCAL_LEXICON)
    if lexicon:
        merged.update(lexicon)
    t = apply_lexicon(t, merged)
    t = _PHONE.sub(lambda m: " " + " ".join(_digits(g) for g in m.groups()) + " ", t)
    t = _TIME_DOT.sub(_time_words, t)
    t = _TIME_COLON.sub(_time_words, t)
    t = _CURRENCY_PREFIX.sub(_currency_words, t)
    t = _BAHT_DECIMAL.sub(lambda m: f" {_baht(m.group(1))} ", t)
    t = _PERCENT.sub(lambda m: f"{_amount_words(m.group(1).replace(',', ''))}เปอร์เซ็นต์", t)
    t = _DEGREES.sub(
        lambda m: "องศา" + {"C": "เซลเซียส", "F": "ฟาเรนไฮต์"}.get(m.group(1) or "", ""), t
    )
    t = _RANGE.sub(r"\1 ถึง \2", t)
    t = _NEGATIVE.sub("ลบ", t)
    t = _NUMBER.sub(lambda m: _number_words(m.group(1), m.group(2)), t)
    t = _expand_maiyamok(t)
    t = _ACRONYM.sub(lambda m: "".join(_LATIN_LETTER_TH[c] for c in m.group(0)), t)
    t = _NON_THAI_LETTERS.sub(" ", t)
    t = _ORPHAN_PUNCT.sub(r"\1", _WS.sub(" ", t).strip())
    return _LEADING_PUNCT.sub("", t)


# --- speakability -------------------------------------------------------------------------


_LIST_MARKER_ONLY: Final = re.compile(r"\(?\d{1,2}[.)]")


def is_speakable(text: str) -> bool:
    """True when a cloud voice would say something.

    After ``normalize_cloud`` the text must still have a letter (any script) or a number.
    Whitespace, punctuation, emoji, markdown, stage directions and a bare list marker such as
    ``"3."`` are not speakable (the reply pipeline merges them into the next chunk); ``555``
    laughter and a number such as ``"42"`` are.
    """
    t = normalize_cloud(text)
    has_digit = False
    for ch in t:
        if is_thai_letter(ch) or (ch.isalpha() and not ("\U00000e00" <= ch <= "\U00000e7f")):
            return True
        has_digit = has_digit or ch.isdigit()
    return has_digit and _LIST_MARKER_ONLY.fullmatch(t) is None
