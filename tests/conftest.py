"""Shared pytest configuration for aivtube (ARCHITECTURE.md §10).

Markers (declared in pyproject.toml):
- ``nightly``: needs real models/binaries; skipped unless ``AIVTUBE_TEST_ASSETS`` points at an
  existing directory (see the ``test_assets`` fixture).
- ``hardware``: needs the streaming PC; skipped unless ``AIVTUBE_HARDWARE=1``.
- ``windows``: Windows-only behaviour; skipped on other platforms.
- ``network``: talks to the public internet; skipped unless ``AIVTUBE_NETWORK=1`` (default runs
  must pass offline).
- ``timing``: wall-clock assertions; skipped when ``AIVTUBE_SKIP_TIMING=1``.

Fixtures that need ``aivtube.infra`` import it lazily and fall back to the fakes, so these tests
run before (or without) the infra package.
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from aivtube.contracts.infra import EventBus
from aivtube.testing.fakes import FakeClock, FakeEventBus, RealClock

FIXTURES = Path(__file__).parent / "fixtures"


def _assets_dir() -> Path | None:
    raw = os.environ.get("AIVTUBE_TEST_ASSETS", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    assets = _assets_dir()
    skips = {
        "nightly": (
            None
            if assets is not None
            else pytest.mark.skip(reason="nightly: set AIVTUBE_TEST_ASSETS to a model directory")
        ),
        "hardware": (
            None
            if _flag("AIVTUBE_HARDWARE")
            else pytest.mark.skip(reason="hardware: set AIVTUBE_HARDWARE=1 on the streaming PC")
        ),
        "windows": (
            None if sys.platform == "win32" else pytest.mark.skip(reason="windows-only behaviour")
        ),
        "network": (
            None
            if _flag("AIVTUBE_NETWORK")
            else pytest.mark.skip(reason="network: set AIVTUBE_NETWORK=1 to allow internet access")
        ),
        "timing": (
            pytest.mark.skip(reason="timing: AIVTUBE_SKIP_TIMING=1")
            if _flag("AIVTUBE_SKIP_TIMING")
            else None
        ),
    }
    for item in items:
        for marker, skip in skips.items():
            if skip is not None and marker in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """``tests/fixtures`` (committed, small, licence-clean files)."""
    return FIXTURES


@pytest.fixture(scope="session")
def silero_onnx(fixtures_dir: Path) -> Path:
    """The committed Silero VAD v6.2.3 model (MIT, 2.3 MB)."""
    return fixtures_dir / "silero_vad.onnx"


@pytest.fixture(scope="session")
def test_assets() -> Path:
    """``$AIVTUBE_TEST_ASSETS`` (real local models); skips the test when unset."""
    path = _assets_dir()
    if path is None:
        pytest.skip("AIVTUBE_TEST_ASSETS is not set (or not a directory)")
    return path


@pytest.fixture
def fake_clock() -> FakeClock:
    """A deterministic ``Clock``: drive it with ``advance()``/``await run_for()``."""
    return FakeClock()


@pytest.fixture
def real_clock() -> Any:
    """``infra.SystemClock`` when available, otherwise the perf_counter ``RealClock`` fake."""
    try:
        from aivtube.infra import SystemClock
    except ImportError:
        return RealClock()
    return SystemClock()


@pytest.fixture
def fake_bus(fake_clock: FakeClock) -> Iterator[FakeEventBus]:
    """An in-process bus with ``history`` for assertions, stamped by ``fake_clock``."""
    bus = FakeEventBus(fake_clock)
    yield bus
    for sub in bus.subscriptions:
        sub.close()


@pytest.fixture
async def bus(real_clock: Any) -> AsyncIterator[EventBus]:
    """The real ``infra.AsyncEventBus`` when available, otherwise ``FakeEventBus``.

    Tests that need ``history`` or other fake-only helpers should use ``fake_bus``.
    """
    try:
        from aivtube.infra import AsyncEventBus
    except ImportError:
        yield FakeEventBus(real_clock)
        return
    real = AsyncEventBus(real_clock)
    yield real
    close = getattr(real, "close", None)
    if callable(close):
        close()
