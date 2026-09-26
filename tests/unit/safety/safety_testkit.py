"""Helpers shared by the safety tests (unique module name: tests have no __init__.py)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from aivtube.contracts.safety import Direction, FilterContext
from aivtube.contracts.types import ChatMessage, ChatUser, Platform

REPO: Final = Path(__file__).resolve().parents[3]
BASE_DIR: Final = REPO / "config" / "filters" / "base"
REDTEAM_DIR: Final = REPO / "tests" / "fixtures" / "redteam"

#: Readable stand-ins for invisible characters in the red-team files.
ESCAPES: Final = {
    "{ZW}": chr(0x200B),
    "{ZWNJ}": chr(0x200C),
    "{ZWJ}": chr(0x200D),
    "{WJ}": chr(0x2060),
    "{BOM}": chr(0xFEFF),
    "{SHY}": chr(0x00AD),
}
SPLIT: Final = "‖"


def unescape(text: str) -> str:
    for token, ch in ESCAPES.items():
        text = text.replace(token, ch)
    return text


def ctx(
    direction: str = "out",
    *,
    prev_tail: str = "",
    character: str = "pailin",
    platform: str | None = None,
) -> FilterContext:
    return FilterContext(
        cast(Direction, direction), character, platform=platform, prev_tail=prev_tail
    )


def chat(
    text: str, name: str = "viewer", *, platform: Platform = Platform.TWITCH, user_id: str = "u-1"
) -> ChatMessage:
    user = ChatUser(platform, user_id, name)
    return ChatMessage(platform, "m-1", user, text, 1.0, 1.0)


@dataclass(frozen=True, slots=True)
class BlockCase:
    category: str
    text: str  # the whole phrase (split marker removed)
    head: str | None = None  # set for a phrase split across two output chunks
    tail: str | None = None
    line: int = 0

    @property
    def id(self) -> str:
        return f"L{self.line}-{self.category}"


def _lines(path: Path) -> list[tuple[int, str]]:
    out = []
    for no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append((no, line))
    return out


def load_block_cases() -> list[BlockCase]:
    cases = []
    for no, line in _lines(REDTEAM_DIR / "block.txt"):
        category, _, text = line.partition(": ")
        text = unescape(text)
        if SPLIT in text:
            head, tail = text.split(SPLIT, 1)
            cases.append(BlockCase(category, head + tail, head, tail, no))
        else:
            cases.append(BlockCase(category, text, line=no))
    return cases


def load_pass_lines() -> list[str]:
    return [unescape(line) for _, line in _lines(REDTEAM_DIR / "pass.txt")]
