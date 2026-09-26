"""Streaming ``[emotion]`` tag extraction for LLM replies (§4.6, §4.8 persona rules).

The persona asks for one ``[emotion]`` tag, only when the emotion changes. The model does not
always comply, and a tag can arrive split across deltas (``"[hap"`` + ``"py] …"``), so the
extractor is a character-level state machine: its output does not depend on how the stream
was split.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

__all__ = ["EmotionTagExtractor", "TagEvent"]

#: ``(text, emotion)``: ``emotion`` is the new emotion taking effect at the start of ``text``,
#: or ``None`` when it does not change there.
TagEvent = tuple[str, str | None]

_INLINE_WS: Final = frozenset(" \t")


class EmotionTagExtractor:
    """Removes ``[tag]`` markers from streamed text and reports emotion changes.

    - A known tag (case-insensitive, surrounding spaces ignored) is reported only when it
      differs from ``last_emotion``; a repeated tag is removed silently.
    - Unknown bracketed tags (``[หัวเราะ]``, ``[แชท]``) are removed: they are stage directions
      or echoed prompt markup, never speech.
    - ``[`` followed by a newline, another ``[`` or more than ``MAX_TAG_CHARS`` characters
      without ``]`` is ordinary text.
    - When a removed tag sits between whitespace (or at the start of the stream), the space
      after it is dropped, so ``"สวัสดี [happy] ทุกคน"`` becomes ``"สวัสดี ทุกคน"``.

    ``feed`` returns ``(text, emotion)`` events in stream order. An emotion change always
    starts a new event, possibly with empty text (a tag at the end of a delta).
    """

    MAX_TAG_CHARS: Final = 24

    def __init__(self, known: Iterable[str], *, current: str | None = None) -> None:
        self._known: dict[str, str] = {}
        for name in known:
            key = name.strip().casefold()
            if key:
                self._known.setdefault(key, name.strip())
        self.last_emotion: str | None = current
        self._pending: str | None = None  # "[..." collected so far
        self._eat_ws = False
        self._after_ws = True  # last text character was whitespace (or nothing yet)

    @property
    def known(self) -> frozenset[str]:
        return frozenset(self._known.values())

    def feed(self, delta: str) -> list[TagEvent]:
        out: list[TagEvent] = []
        text: list[str] = []
        emotion: str | None = None

        def close() -> None:
            nonlocal emotion
            if text or emotion is not None:
                out.append(("".join(text), emotion))
            text.clear()
            emotion = None

        for ch in delta:
            pending = self._pending
            if pending is not None:
                if ch == "]":
                    self._pending = None
                    new = self._resolve(pending[1:])
                    if new is not None and new != self.last_emotion:
                        close()
                        emotion = new
                        self.last_emotion = new
                    self._eat_ws = self._after_ws
                    continue
                if ch not in "[\n\r" and len(pending) <= self.MAX_TAG_CHARS:
                    self._pending = pending + ch
                    continue
                self._pending = None  # not a tag after all: the bracket was plain text
                for c in pending:
                    self._text_char(c, text)
            if ch == "[":
                self._pending = "["
            else:
                self._text_char(ch, text)
        close()
        return out

    def flush(self) -> list[TagEvent]:
        """End of stream: a dangling ``[…`` is dropped when it can still become a known tag
        (or is empty), otherwise emitted as text. Resets the stream state; ``last_emotion``
        is kept."""
        text: list[str] = []
        pending = self._pending
        if pending is not None:
            partial = pending[1:].strip().casefold()
            if partial and not any(k.startswith(partial) for k in self._known):
                for c in pending:
                    self._text_char(c, text)
        self.reset()
        return [("".join(text), None)] if text else []

    def reset(self) -> None:
        """Forget any partial tag and whitespace state (new stream); keeps ``last_emotion``."""
        self._pending = None
        self._eat_ws = False
        self._after_ws = True

    # --- internals ------------------------------------------------------------------------
    def _resolve(self, body: str) -> str | None:
        return self._known.get(body.strip().casefold())

    def _text_char(self, ch: str, text: list[str]) -> None:
        if self._eat_ws:
            if ch in _INLINE_WS:
                return
            self._eat_ws = False
        text.append(ch)
        self._after_ws = ch.isspace()
