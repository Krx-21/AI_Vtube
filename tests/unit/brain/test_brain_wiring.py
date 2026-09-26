"""Brain wiring: config flattening and a light import (no native stacks in the core)."""

from __future__ import annotations

import subprocess
import sys

from aivtube.brain.arbiter import ArbiterConfig
from aivtube.brain.intake import IntakeConfig
from aivtube.brain.loop import BrainSettings, brain_cfg
from aivtube.config.schema import AppConfig


def test_brain_cfg_feeds_every_brain_part() -> None:
    app = AppConfig.model_validate(
        {
            "brain": {"decision_watchdog_s": 12.0, "chat_min_interval_s": 5.0},
            "mic": {"addressing": "name_or_question", "mode": "ptt"},
            "llm": {"temperature": 0.7, "max_reply_tokens": 200},
            "tts": {"filler_after_s": 1.5},
            "app": {"auto_live": True},
        }
    )
    cfg = brain_cfg(app)
    settings = BrainSettings.from_mapping(cfg)
    assert settings.decision_watchdog_s == 12.0 and settings.temperature == 0.7
    assert settings.max_reply_tokens == 200 and settings.filler_after_s == 1.5
    assert settings.auto_live and settings.critical_cut_wait_ms == 150
    arbiter = ArbiterConfig.from_mapping(cfg)
    assert arbiter.chat_min_interval_s == 5.0 and arbiter.chat_gather_s == 1.0
    intake = IntakeConfig.from_mapping(cfg)
    assert intake.addressing == "name_or_question" and intake.mic_mode == "ptt"
    assert intake.max_msg_chars == 300 and intake.read_aloud_ratio == 0.75
    assert IntakeConfig.from_app(app) == intake


def test_importing_the_brain_loads_no_native_stack() -> None:
    code = (
        "import sys\n"
        "import aivtube.brain.loop, aivtube.brain.background, aivtube.brain.intake\n"
        "heavy = ('numpy', 'sounddevice', 'sherpa_onnx', 'onnxruntime', 'av', 'edge_tts',\n"
        "         'azure', 'livekit', 'soxr', 'pythainlp', 'openai')\n"
        "print(','.join(m for m in heavy if m in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True
    )
    assert out.stdout.strip() == ""
