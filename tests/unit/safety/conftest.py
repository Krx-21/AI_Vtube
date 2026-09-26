"""Fixtures for the safety tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from safety_testkit import BASE_DIR

from aivtube.safety import KeywordRegexFilter


@pytest.fixture(scope="session")
def base_dir() -> Path:
    return BASE_DIR


@pytest.fixture(scope="session")
def tier0() -> KeywordRegexFilter:
    """The committed base lists, warm (shared: tests must not reload it)."""
    return KeywordRegexFilter(BASE_DIR, handle_aliases={"pailin": ["pailin", "ไพลิน"]})


@pytest.fixture(scope="session")
def tier0_block_politics() -> KeywordRegexFilter:
    return KeywordRegexFilter(BASE_DIR, politics="block", warm=False)


@pytest.fixture
def lists(tmp_path: Path) -> Path:
    """A private copy of the base lists that a test may edit and reload."""
    dst = tmp_path / "base"
    dst.mkdir()
    for src in BASE_DIR.glob("*.toml"):
        (dst / src.name).write_bytes(src.read_bytes())
    return dst
