"""Building the filter and the gate from the validated config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from safety_testkit import REPO, chat, ctx

from aivtube.config import load_characters, load_config
from aivtube.config.schema import AppConfig
from aivtube.contracts.safety import Verdict
from aivtube.contracts.types import Platform
from aivtube.infra import SystemClock
from aivtube.safety import build_keyword_filter, build_safety_gate, ops_sink
from aivtube.testing.fakes import FakeEventBus, FakeTaskSupervisor


@pytest.fixture(scope="module")
def cfg() -> AppConfig:
    return load_config(REPO, env={})


def with_twitch_overlay(cfg: AppConfig, path: Path) -> AppConfig:
    safety = cfg.safety.model_copy(update={"platform_overlays": {Platform.TWITCH: str(path)}})
    return cfg.model_copy(update={"safety": safety})


def test_filter_from_config(cfg: AppConfig, tmp_path: Path) -> None:
    overlay = tmp_path / "twitch.toml"
    overlay.write_text(
        '[[deny]]\ncategory = "slur"\nmatch = "substring"\ntext = "zzspamzz"\n', "utf-8"
    )
    chars = load_characters(cfg).values()
    filt = build_keyword_filter(with_twitch_overlay(cfg, overlay), chars, warm=False)
    names = {p.name for p in filt.files}
    assert {"slur.toml", "monarchy_112.toml", "filters.toml", "twitch.toml"} <= names
    assert filt.politics == cfg.safety.politics
    assert filt.check("zzspamzz", ctx("in", platform="twitch")).verdict is Verdict.DROP
    assert filt.check("zzspamzz", ctx("in", platform="youtube")).verdict is Verdict.PASS
    # the character's aliases exempt its own @handle
    assert filt.check("@pailin_th สวัสดี", ctx("in", character="pailin")).verdict is Verdict.PASS


async def test_gate_from_config(cfg: AppConfig) -> None:
    rows: list[dict[str, Any]] = []

    class OpsDb:
        async def log_moderation(self, **row: Any) -> None:
            rows.append(row)

    clock = SystemClock()
    audit_sink = ops_sink(OpsDb())
    gate = build_safety_gate(
        cfg,
        load_characters(cfg).values(),
        bus=FakeEventBus(clock),
        clock=clock,
        tasks=FakeTaskSupervisor(),
        audit_sink=audit_sink,
        warm=False,
    )
    res, name = gate.check_input(chat("บาคาร่าออนไลน์", "มะลิ"), character="pailin")
    assert (res.verdict, name) == (Verdict.DROP, "มะลิ")
    assert gate.audit is not None
    await gate.audit.flush()
    assert rows and rows[0]["category"] == "gambling_scam"
    assert gate.cfg == cfg.safety
