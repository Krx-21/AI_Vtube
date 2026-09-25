"""``ConfigError``: a configuration problem explained in English and Thai, with a fix hint.

Standard library only, so the stdlib-only launcher can import it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["ConfigError"]


class ConfigError(Exception):
    """A config problem at ``path`` (a dotted key or a file path).

    ``source`` names the layer that supplied the bad value (``config/user.toml``,
    ``environment``, …) when it is known. ``issues`` lists every problem found; the
    exception's own fields describe the first one.
    """

    path: str
    message_en: str
    message_th: str
    hint: str
    hint_th: str
    source: str | None
    issues: tuple[ConfigError, ...]

    def __init__(
        self,
        path: str,
        message_en: str,
        message_th: str,
        hint: str = "",
        *,
        hint_th: str = "",
        source: str | None = None,
        issues: Sequence[ConfigError] = (),
    ) -> None:
        super().__init__(message_en)
        self.path = path
        self.message_en = message_en
        self.message_th = message_th
        self.hint = hint
        self.hint_th = hint_th
        self.source = source
        self.issues = tuple(issues) or (self,)

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            _rebuild,
            (self.path, self.message_en, self.message_th, self.hint, self.hint_th, self.source),
        )

    def _block(self) -> list[str]:
        where = self.path or "config"
        if self.source:
            where += f" ({self.source})"
        lines = [f"{where}: {self.message_en}", f"    {self.message_th}"]
        if self.hint:
            lines.append(f"    hint: {self.hint}")
        if self.hint_th:
            lines.append(f"    วิธีแก้: {self.hint_th}")
        return lines

    def __str__(self) -> str:
        lines = self._block()
        extra = len(self.issues) - 1
        if extra > 0:
            lines.append(f"    (+{extra} more / และอีก {extra} ปัญหา)")
        return "\n".join(lines)

    def format_all(self) -> str:
        """Every issue, one block each (for ``aivtube config show`` and ``doctor``)."""
        return "\n".join(line for issue in self.issues for line in issue._block())


def _rebuild(
    path: str, en: str, th: str, hint: str, hint_th: str, source: str | None
) -> ConfigError:
    return ConfigError(path, en, th, hint, hint_th=hint_th, source=source)
