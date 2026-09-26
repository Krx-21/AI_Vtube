"""LoopLagMonitor: measures loop lag and dumps stacks while the loop is still stuck."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from aivtube.infra import LoopLagMonitor, MetricsRegistry


def test_thresholds_are_validated() -> None:
    with pytest.raises(ValueError):
        LoopLagMonitor(interval_s=0.5, warn_s=0.25, dump_s=2.0)


@pytest.mark.timing
async def test_lag_is_measured_and_warned(caplog: pytest.LogCaptureFixture) -> None:
    metrics = MetricsRegistry()
    seen: list[float] = []
    mon = LoopLagMonitor(
        interval_s=0.02, warn_s=0.1, dump_s=5.0, metrics=metrics, on_lag=seen.append, watchdog=False
    )
    task = asyncio.create_task(mon.run())
    await asyncio.sleep(0.05)
    time.sleep(0.3)  # block the loop
    await asyncio.sleep(0.1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert mon.max_lag_s >= 0.2
    assert mon.warnings >= 1 and seen and seen[0] >= 0.2
    assert metrics.get("loop_lag_max_ms") >= 200
    assert any("lagged" in r.getMessage() for r in caplog.records)


@pytest.mark.timing
async def test_watchdog_dumps_stacks_during_a_stall(tmp_path: Path) -> None:
    out = tmp_path / "core.fault"
    with out.open("w", encoding="utf-8") as fh:
        mon = LoopLagMonitor(interval_s=0.02, warn_s=0.1, dump_s=0.3, dump_file=fh)
        task = asyncio.create_task(mon.run())
        await asyncio.sleep(0.05)
        time.sleep(0.8)  # a wedged loop
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    text = out.read_text(encoding="utf-8")
    assert mon.dumps == 1
    assert "event loop stalled" in text
    assert "test_watchdog_dumps_stacks_during_a_stall" in text  # the blocking frame is visible
